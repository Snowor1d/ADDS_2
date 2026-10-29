"""Counterfactual action-value probe for the current critic (read-only).

Ground truth for one robot-decision. From a state reached by the learned
stochastic policy, robot 0 takes one candidate action for one decision while
its teammates take fixed policy actions; afterwards every robot follows the
stochastic policy to the end of the episode. Timing, reward and discount
follow learn/rollout.run_episode and learn/sac.robot_targets:

  * decisions on the frame clock, held up to decision_max_steps or, with
    decision events, until the first frame after an arrival or a wall block;
  * robot 0's learning reward per step = team terms (task + team shaping)
    + robot 0's own terms (collision, approach shaping), with the terminal
    potential correction when the task ends the episode;
  * discount gamma_step = cfg.gamma() ** (1 / ACTION_SCALE) per step.

Candidates share the random numbers within a repeat (common random numbers);
repeats reseed `random`, `np.random` and torch. The critic's Q for the same
candidates is read at the branch state with the moves the environment
accepted, as the replay stores them.

Usage: true_adv_v2.py SCEN T0_STEPS CKPT OUT [--reps 3] [--rep-start 0] [--max-steps N]
                      [--overrides JSON]

SCEN is a validation scenario index, or SITE:SIZE:K for a probe level on any
corpus crop (`probe_level`), e.g. kreuzberg:200:0 or hongdae:100:1.

Repeat r always uses the same seeds, so --rep-start adds repeats to an
earlier run of the same state (merge_reps.py joins them).
"""
import argparse
import copy
import json
import os
import random
import sys
import time

ROOT = "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

torch.set_num_threads(1)

from configs import decision_max_steps, decision_on_events, frame_steps, resolve_config
from learn.replay import joint_observation
from learn.sac import OBS_KEYS, SACAgent, to_torch
from learn.zero_shot import build_eval_model, validation_levels, validation_seed
from sim import robot_action as ra
from sim.observation import ObservationHistory, build_static_layers
from sim.rewards import PER_ROBOT_VERSIONS, ROBOT_TERMS, TaskReward


def probe_level(cfg, spec):
    """(name, level, crowd seed) for a validation index or SITE:SIZE:K.

    A SITE:SIZE:K level is made by the training sampler from a seed fixed by
    its name, as validation levels are, in a range of its own (960000-964999;
    validation uses 955000-959999 and the final zero-shot 970000 and up)."""
    import hashlib
    spec = str(spec)
    if ":" not in spec:
        name, lv = validation_levels(cfg)[int(spec)]
        return name, lv, validation_seed(name)
    from learn.training_maps import OsmTrainingMaps
    site, size, k = spec.split(":")
    if site == str(cfg.FINAL_ZERO_SHOT_SITE):
        raise ValueError("the final zero-shot site is not probed")
    name = f"probe_{site}_{int(size)}m_{int(k)}"
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 5000 + 960_000
    maps = OsmTrainingMaps(cfg, (site,), (int(size),))
    lv = maps.make_level((site, int(size)), random.Random(seed))
    if lv is None:
        raise RuntimeError(f"no hazard placed on {spec}")
    return name, lv, validation_seed(name)


class Episode:
    """run_episode's loop, split so it can stop at a due decision and be
    deep-copied there."""

    def __init__(self, model, cfg, static, seed):
        self.m, self.cfg = model, cfg
        self.F, self.K = frame_steps(cfg), decision_max_steps(cfg)
        self.on_events = decision_on_events(cfg)
        self.R = int(cfg.MAX_ROBOTS)
        self.max_steps = int(cfg.MAX_STEPS)
        self.per_robot = cfg.REWARD_VERSION in PER_ROBOT_VERSIONS
        self.hist = ObservationHistory(cfg, static, len(model.robots), seed=seed)
        self.rew = TaskReward(model, cfg)
        self.step, self.hold, self.event, self.done = 0, None, False, False
        self.last_frame = -1
        self.acc = []            # robot 0's learning reward per step

    def next_decision(self) -> bool:
        """Advance to the next due decision (frame recorded, not yet acted
        on). False when the episode ended first."""
        while not self.done and self.step < self.max_steps:
            if self.step % self.F == 0 and self.last_frame != self.step:
                self.last_frame = self.step
                self.hist.record(self.m)
                if (self.hold is None or self.hold >= self.K
                        or (self.on_events and self.event)):
                    return True
            self._sim_step()
        return False

    def decide(self, actions: np.ndarray) -> np.ndarray:
        applied = np.zeros((self.R, ra.ACTION_DIM), np.float32)
        for i, rb in enumerate(self.m.robots[:self.R]):
            move = ra.apply_to(rb, actions[i])
            applied[i] = actions[i]
            applied[i, 0], applied[i, 1] = float(move[0]), float(move[1])
        self.hold, self.event = 0, False
        return applied

    def _sim_step(self):
        self.m.step()
        comps = self.rew.step(self.m)
        r = sum(comps.values())
        if self.per_robot:
            r0 = (r - sum(comps[t] for t in ROBOT_TERMS)
                  + self.rew.last_robot_penalties[0])
        else:
            r0 = r
        self.hold += 1
        self.step += 1
        if self.on_events and any(getattr(rb, "decision_event", None)
                                  for rb in self.m.robots[:self.R]):
            self.event = True
        if self.m.should_finish():
            team, own = self.rew.terminal_correction()
            r0 += team + ((own[0] if own else 0.0) if self.per_robot
                          else sum(own))
            self.done = True
        self.acc.append(float(r0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", help="validation index or SITE:SIZE:K")
    ap.add_argument("t0", type=int, help="branch at the first decision at or after this step")
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap the rollout after the branch (smoke tests)")
    ap.add_argument("--overrides", default="",
                    help="JSON settings to run under, e.g. the checkpoint's training values")
    args = ap.parse_args()

    over = json.loads(args.overrides) if args.overrides else {}
    cfg = resolve_config(over or None, check_data=False)
    print(f"gamma {cfg.gamma()} decision max steps {cfg.decision_max_steps()} "
          f"overrides {over}", flush=True)
    agent = SACAgent(cfg, device="cpu")
    ck = agent.load(args.ckpt)
    agent.policy.eval()
    if ck.get("config_fingerprint") not in (None, cfg.fingerprint):
        print(f"WARNING config {cfg.fingerprint} != checkpoint "
              f"{ck.get('config_fingerprint')}", flush=True)
    g_step = cfg.gamma() ** (1.0 / max(1, int(cfg.ACTION_SCALE)))

    name, lv, seed = probe_level(cfg, args.scen)
    m = build_eval_model(lv, seed)
    static = build_static_layers(m, cfg)
    ep = Episode(m, cfg, static, seed)
    torch.manual_seed(seed)

    def policy(e, det=False):
        obs = e.hist.team_observations()
        with torch.no_grad():
            t = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
            if det:
                a = agent.policy.deterministic_action(*(t[k] for k in OBS_KEYS))
            else:
                a, _ = agent.policy.sample_action(*(t[k] for k in OBS_KEYS))
        return a.numpy()

    # Prefix under the stochastic policy, up to the branch decision.
    while ep.next_decision():
        if ep.step >= args.t0:
            break
        ep.decide(policy(ep))
    assert not ep.done, "episode ended before the branch"
    branch_step = ep.step

    base = policy(ep)                         # teammates' actions, fixed
    rb0 = ep.m.robots[0]
    x, y = float(rb0.xy[0]), float(rb0.xy[1])
    zone = ep.m.danger_zone
    lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)

    def toward(tx, ty, speed, mode):
        dx, dy = tx - x, ty - y
        s = max(1.0, abs(dx) / lim, abs(dy) / lim)
        return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)

    from validation.shuttle_baseline import ShuttlePolicy, _centre
    sp = ShuttlePolicy(ep.m, len(ep.m.robots), cfg=cfg)
    ex = sp._exit_mesh(ep.m.find_mesh((x, y)))
    exc = _centre(ex) if ex is not None else (x, y)
    samples = [policy(ep)[0] for _ in range(2)]
    cands = {
        "policy_det": policy(ep, det=True)[0],
        "policy_s1": samples[0],
        "policy_s2": samples[1],
        "in_off": toward(zone.cx, zone.cy, 1.0, "off"),
        "in_guide": toward(zone.cx, zone.cy, 0.6, "guide"),
        "out_guide": toward(exc[0], exc[1], 0.6, "guide"),
        "out_off": toward(exc[0], exc[1], 1.0, "off"),
        "still_guide": ra.encode((0.0, 0.0), "guide", speed=0.0),
    }

    memo_shared = {id(cfg): cfg, id(static): static}
    horizons = {"200s": 400, "400s": 800}
    out = {"scenario": name, "scen": args.scen, "level_spec": str(args.scen),
           "overrides": over, "map_size": int(lv.width), "t0_request": args.t0,
           "branch_step": branch_step, "gamma": cfg.gamma(), "g_step": g_step,
           "inside_at_branch": int(ep.m.agents_in_danger()),
           "n_ref": float(ep.rew.n_ref), "robot0_xy": [x, y],
           "robot0_signed_distance": float(zone.signed_distance(x, y)),
           "reps": args.reps, "rep_ids": list(range(args.rep_start, args.rep_start + args.reps)), "cands": {k: v.tolist() for k, v in cands.items()},
           "mc": {k: [] for k in cands}, "mc_h": {h: {k: [] for k in cands} for h in horizons},
           "first_hold": {k: [] for k in cands}, "steps_run": {k: [] for k in cands},
           "applied": {}}
    t_start = time.time()
    for rep in range(args.rep_start, args.rep_start + args.reps):
        for nm, a0 in cands.items():
            e = copy.deepcopy(ep, dict(memo_shared))
            e.acc = []
            random.seed(10_000 * rep + 7 + seed)
            np.random.seed((10_000 * rep + 7 + seed) % 2 ** 32)
            torch.manual_seed(1000 + rep)
            acts = base.copy()
            acts[0] = a0
            applied = e.decide(acts)
            if rep == 0:
                out["applied"][nm] = applied.tolist()
            first = None
            cap = (branch_step + args.max_steps) if args.max_steps else None
            while e.next_decision():
                if first is None:
                    first = e.step - branch_step
                if cap is not None and e.step >= cap:
                    break
                e.decide(policy(e))
            if first is None:
                first = e.step - branch_step
            r = np.asarray(e.acc, np.float64)
            disc = g_step ** np.arange(len(r))
            out["mc"][nm].append(float((r * disc).sum()))
            for h, n in horizons.items():
                out["mc_h"][h][nm].append(float((r[:n] * disc[:n]).sum()))
            out["first_hold"][nm].append(int(first))
            out["steps_run"][nm].append(int(len(r)))
        print(f"[{name} t0={args.t0}] rep {rep} done {time.time() - t_start:.0f}s", flush=True)

    # Critic at the branch state, robot 0's own output.
    joint = joint_observation(cfg, [static], [ep.hist.window()])
    tobs = to_torch({k: joint[k] for k in OBS_KEYS + ("priv", "priv_scalars")}, "cpu")
    mask = torch.as_tensor(joint["mask"])
    out["q1"], out["q2"] = {}, {}
    with torch.no_grad():
        for nm, applied in out["applied"].items():
            at = torch.as_tensor(np.asarray(applied, np.float32))[None]
            out["q1"][nm] = float(agent.q1(tobs, at, mask)[0, 0])
            out["q2"][nm] = float(agent.q2(tobs, at, mask)[0, 0])
    out["elapsed_s"] = time.time() - t_start
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
