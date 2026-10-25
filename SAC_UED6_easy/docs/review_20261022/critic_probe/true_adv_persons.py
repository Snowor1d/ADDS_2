"""Counterfactual probe with robot 0's return split by person.

Same branch, candidates, seeds and rollouts as true_adv_v2.py, so a state's
totals reproduce that probe exactly. Robot 0's learning reward is a sum over
persons plus robot 0's own terms:

  person p, per step  person_time and remaining_path while inside, reentry on
                      entering, progress shaping g*phi_p(s') - phi_p(s) with
                      phi_p = -[inside] min(1, escape_m / D_ref) / N_ref, and
                      the terminal correction -g*phi_p(s_T)
  robot               last_robot_penalties[0] (collision, approach shaping)

The split is checked against the engine's reward every step. Persons present
at the branch are kept one by one; persons who arrive later are one
"inflow" component. Whether a person interacted with robot 0 after the branch
(followed it, or was exposed to its signal within the inform radius) is
recorded per rollout; it never reads rewards.

With --continuation det every robot follows the deterministic policy after
the branch decision, which removes the robots' own sampling noise from the
returns and leaves the crowd's.

Usage: true_adv_persons.py SCEN T0 CKPT OUT.npz --rep-start 0 --reps 5 [--continuation det]
"""
import argparse
import copy
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import true_adv_v2 as P  # noqa: E402  (chdirs to the project root)
import exo_rng  # noqa: E402

import numpy as np
import torch

from configs import resolve_config
from learn.sac import OBS_KEYS, SACAgent, to_torch
from learn.zero_shot import build_eval_model, validation_levels, validation_seed
from sim import robot_action as ra
from sim.observation import build_static_layers


class PersonEpisode(P.Episode):
    """Episode that also books robot 0's reward to persons."""

    def start_tracking(self, ids_at_branch):
        rew = self.rew
        self.idx = {pid: i for i, pid in enumerate(ids_at_branch)}
        self.r0_id = self.m.robots[0].unique_id
        self.Rp = np.zeros(len(ids_at_branch))
        self.R_inflow = 0.0
        self.R_robot = 0.0
        self.disc = 1.0
        self.follow = np.zeros(len(ids_at_branch), bool)
        people = self._by_id()
        self.expo_start = np.zeros(len(ids_at_branch))
        for pid, i in self.idx.items():
            if pid in people:
                self.expo_start[i] = self._expo(people[pid])
        self.phi = {}
        for a in rew._inside(self.m):
            self.phi[a.unique_id] = -min(1.0, float(self.m.escape_distance(a.xy)) / rew.d_ref) / rew.n_ref
        assert abs(sum(self.phi.values()) - rew._phi_prog) < 1e-9, "phi mismatch at branch"
        self.max_err = 0.0

    def _by_id(self):
        return {a.unique_id: a for a in self.m.crowds}

    def _expo(self, a):
        return float(getattr(a, "_inform_exposure", {}).get(self.r0_id, 0))

    def _book(self, pid, v):
        i = self.idx.get(pid)
        if i is None:
            self.R_inflow += self.disc * v
        else:
            self.Rp[i] += self.disc * v

    def _sim_step(self):
        rew, w = self.rew, self.rew.weights
        exo_rng.EXO.t = self.step               # read only while installed
        clear_before = set(rew._clear)
        n_before = len(self.acc)
        super()._sim_step()                      # appends robot 0's reward
        if not hasattr(self, "idx"):
            return
        r0 = self.acc[n_before]
        inside = rew._inside(self.m)
        total = 0.0
        phi_new = {}
        for a in inside:
            pid = a.unique_id
            esc = min(1.0, float(self.m.escape_distance(a.xy)) / rew.d_ref)
            v = (-w["person_time"] * rew.dt / rew.n_ref
                 - w["remaining_path"] * esc * rew.dt / rew.n_ref)
            if pid in clear_before:
                v += -w["reentry"] / rew.n_ref
            phi_new[pid] = -esc / rew.n_ref
            self._book(pid, v)
            total += v
        if w["shape_progress"]:
            for pid in set(phi_new) | set(self.phi):
                f = w["shape_progress"] * (rew.g_step * phi_new.get(pid, 0.0)
                                           - self.phi.get(pid, 0.0))
                if self.done:          # terminal state: its potential is zero
                    f += -w["shape_progress"] * rew.g_step * phi_new.get(pid, 0.0)
                self._book(pid, f)
                total += f
        self.phi = phi_new
        robot = self.rew.last_robot_penalties[0] if self.per_robot else 0.0
        if self.done:
            _, own = self.rew.terminal_correction()
            robot += (own[0] if own else 0.0) if self.per_robot else sum(own)
        self.R_robot += self.disc * robot
        total += robot
        self.max_err = max(self.max_err, abs(total - r0))
        for a in self.m.crowds:
            i = self.idx.get(a.unique_id)
            if i is not None and getattr(a, "following_robot_id", None) == self.r0_id:
                self.follow[i] = True
        self.disc *= rew.g_step

    def exposure(self):
        out = np.zeros(len(self.idx))
        people = self._by_id()
        for pid, i in self.idx.items():
            if pid in people:
                out[i] = self._expo(people[pid]) - self.expo_start[i]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", type=int)
    ap.add_argument("t0", type=int)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--continuation", default="stoch", choices=["stoch", "det"],
                    help="every robot's policy after the branch decision")
    ap.add_argument("--hold", type=int, default=1,
                    help="robot 0 keeps the candidate's absolute target, mode and speed for this many decisions")
    ap.add_argument("--exo", action="store_true",
                    help="person-level common random numbers (exo_rng.py) after the branch")
    args = ap.parse_args()

    cfg = resolve_config(check_data=False)
    agent = SACAgent(cfg, device="cpu")
    agent.load(args.ckpt, policy_only=True)
    agent.policy.eval()
    name, lv = validation_levels(cfg)[args.scen]
    seed = validation_seed(name)
    m = build_eval_model(lv, seed)
    static = build_static_layers(m, cfg)
    ep = PersonEpisode(m, cfg, static, seed)
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

    while ep.next_decision():
        if ep.step >= args.t0:
            break
        ep.decide(policy(ep))
    branch_step = ep.step
    base = policy(ep)
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
        "policy_det": policy(ep, det=True)[0], "policy_s1": samples[0],
        "policy_s2": samples[1],
        "in_off": toward(zone.cx, zone.cy, 1.0, "off"),
        "in_guide": toward(zone.cx, zone.cy, 0.6, "guide"),
        "out_guide": toward(exc[0], exc[1], 0.6, "guide"),
        "out_off": toward(exc[0], exc[1], 1.0, "off"),
        "still_guide": ra.encode((0.0, 0.0), "guide", speed=0.0),
    }
    names = list(cands)
    # The absolute target each candidate asks for (offset = move / 2 * range,
    # from where the robot stands), held for --hold decisions.
    held = {}
    for nm, v in cands.items():
        v = np.asarray(v, np.float32)
        mode = cfg.ROBOT_MODES[int(np.argmax(v[ra.MODE]))]
        held[nm] = ((x + float(v[0]) * lim / 2.0, y + float(v[1]) * lim / 2.0),
                    mode, ra.speed_fraction(v))

    def reencode(e, nm):
        (tx, ty), mode, speed = held[nm]
        r = e.m.robots[0]
        dx, dy = tx - float(r.xy[0]), ty - float(r.xy[1])
        s_ = max(1.0, abs(dx) / lim, abs(dy) / lim)
        return ra.encode((2 * dx / (lim * s_), 2 * dy / (lim * s_)), mode, speed=speed)
    ids = [a.unique_id for a in ep.rew._alive(ep.m)]
    xy = np.asarray([[float(a.xy[0]), float(a.xy[1])] for a in ep.rew._alive(ep.m)])
    dist0 = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
    inside0 = np.asarray([bool(zone.contains(px, py)) for px, py in xy])
    C, M, Pn = len(names), args.reps, len(ids)
    Rp = np.zeros((C, M, Pn), np.float32)
    R_inflow = np.zeros((C, M))
    R_robot = np.zeros((C, M))
    total = np.zeros((C, M))
    follow = np.zeros((C, M, Pn), bool)
    expo = np.zeros((C, M, Pn), np.float32)
    max_err = 0.0
    memo_shared = {id(cfg): cfg, id(static): static}
    if args.exo:
        exo_rng.install()
    t_start = time.time()
    for k, rep in enumerate(range(args.rep_start, args.rep_start + args.reps)):
        for c, nm in enumerate(names):
            e = copy.deepcopy(ep, dict(memo_shared))
            e.acc = []
            e.start_tracking(ids)
            random.seed(10_000 * rep + 7 + seed)
            np.random.seed((10_000 * rep + 7 + seed) % 2 ** 32)
            torch.manual_seed(1000 + rep)
            exo_rng.EXO.reset(10_000 * rep + 7 + seed)
            acts = base.copy()
            acts[0] = cands[nm]
            e.decide(acts)
            n_dec = 1
            while e.next_decision():
                if args.max_steps and e.step >= branch_step + args.max_steps:
                    break
                a_next = policy(e, det=args.continuation == "det")
                if n_dec < args.hold:
                    a_next[0] = reencode(e, nm)
                e.decide(a_next)
                n_dec += 1
            r = np.asarray(e.acc, np.float64)
            total[c, k] = float((r * np.power(e.rew.g_step, np.arange(len(r)))).sum())
            Rp[c, k] = e.Rp
            R_inflow[c, k] = e.R_inflow
            R_robot[c, k] = e.R_robot
            follow[c, k] = e.follow
            expo[c, k] = e.exposure()
            max_err = max(max_err, e.max_err)
            gap = abs(total[c, k] - (e.Rp.sum() + e.R_inflow + e.R_robot))
            assert gap < 1e-6 * max(1.0, abs(total[c, k])), (nm, rep, gap)
        print(f"[{name} t0={args.t0}] rep {rep} done {time.time() - t_start:.0f}s "
              f"max step err {max_err:.1e}", flush=True)
    assert max_err < 1e-9, max_err
    np.savez_compressed(args.out, names=np.asarray(names), rep_ids=np.arange(args.rep_start, args.rep_start + M),
                        Rp=Rp, R_inflow=R_inflow, R_robot=R_robot, total=total,
                        follow=follow, expo=expo, ids=np.asarray(ids), dist0=dist0, inside0=inside0,
                        branch_step=branch_step, robot0_xy=np.asarray([x, y]),
                        continuation=args.continuation, exo=bool(args.exo), hold=int(args.hold))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
