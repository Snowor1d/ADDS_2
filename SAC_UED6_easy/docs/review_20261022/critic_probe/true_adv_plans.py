"""Counterfactual probe of two-phase plans as one decision of robot 0.

A plan is a list of phases (target, mode, speed). A phase's target is fixed
when the phase starts:

  center  the hazard centre
  mid     halfway from where the robot is to the hazard centre
  exit    the centre of the nearest safe mesh from where the robot is (the
          shuttle baseline's exit rule, used here only to place a target)
  stay    where the robot is

and the phase ends when robot 0 is within ARRIVE_M of its waypoint or after
PHASE_MAX_S (STAY_S for "stay"); the next phase starts at the next decision
frame, and after the last phase robot 0 follows the policy again. Phase
changes happen at team decision frames, as in true_adv_v2's engine, and
robot 0's own arrival is a decision event. Everything else is the
`true_adv_persons.py --continuation det --exo` condition: same branch,
seeds and person-level common random numbers, so "policy_det" reproduces
that run's candidate of the same name and the plans pair with its
single-decision candidates repeat by repeat.

Usage: true_adv_plans.py SCEN T0 CKPT OUT.npz --rep-start 0 --reps 5
"""
import argparse
import copy
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import true_adv_persons as TP  # noqa: E402  (chdirs to the project root)
import exo_rng  # noqa: E402

import numpy as np
import torch

from configs import resolve_config
from learn.sac import OBS_KEYS, SACAgent, to_torch
from learn.zero_shot import build_eval_model, validation_levels, validation_seed
from sim import robot_action as ra
from sim.observation import build_static_layers

ARRIVE_M = 3.0
PHASE_MAX_S = 90.0
STAY_S = 60.0

PLANS = {
    "policy_det": None,
    "S_in_off": [("center", "off", 1.0)],
    "S_out_guide": [("exit", "guide", 0.6)],
    "S_still_guide": [("stay", "guide", 0.0)],
    "P_in_off__out_guide": [("center", "off", 1.0), ("exit", "guide", 0.6)],
    "P_mid_off__out_guide": [("mid", "off", 1.0), ("exit", "guide", 0.6)],
    "P_in_guide__out_guide": [("center", "guide", 0.6), ("exit", "guide", 0.6)],
    "P_in_off__still_guide": [("center", "off", 1.0), ("stay", "guide", 0.0)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scen", type=int)
    ap.add_argument("t0", type=int)
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rep-start", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke tests only")
    ap.add_argument("--overrides", default="",
                    help='JSON settings to run under, e.g. the checkpoint\'s training values')
    args = ap.parse_args()

    import json
    over = json.loads(args.overrides) if args.overrides else {}
    cfg = resolve_config(over or None, check_data=False)
    print(f"gamma {cfg.gamma()} decision max steps {cfg.decision_max_steps()} "
          f"overrides {over}", flush=True)
    agent = SACAgent(cfg, device="cpu")
    agent.load(args.ckpt, policy_only=True)
    agent.policy.eval()
    name, lv = validation_levels(cfg)[args.scen]
    seed = validation_seed(name)
    m = build_eval_model(lv, seed)
    static = build_static_layers(m, cfg)
    ep = TP.PersonEpisode(m, cfg, static, seed)
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

    # The same prefix, branch and random draws as true_adv_persons.py.
    while ep.next_decision():
        if ep.step >= args.t0:
            break
        ep.decide(policy(ep))
    branch_step = ep.step
    base = policy(ep)
    _samples = [policy(ep)[0] for _ in range(2)]      # keep torch's stream aligned
    det0 = policy(ep, det=True)[0]
    zone = ep.m.danger_zone
    lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    dt = float(cfg.AGENT_TIME_STEP)
    from validation.shuttle_baseline import ShuttlePolicy, _centre
    sp = ShuttlePolicy(ep.m, len(ep.m.robots), cfg=cfg)

    def target(kind, e):
        r = e.m.robots[0]
        x, y = float(r.xy[0]), float(r.xy[1])
        if kind == "center":
            return float(zone.cx), float(zone.cy)
        if kind == "mid":
            return (x + float(zone.cx)) / 2.0, (y + float(zone.cy)) / 2.0
        if kind == "exit":
            ex = sp._exit_mesh(e.m.find_mesh((x, y)))
            return _centre(ex) if ex is not None else (x, y)
        return x, y

    def command(e, tx, ty, mode, speed):
        r = e.m.robots[0]
        dx, dy = tx - float(r.xy[0]), ty - float(r.xy[1])
        s = max(1.0, abs(dx) / lim, abs(dy) / lim)
        return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)

    def arrived(e, tx, ty):
        r = e.m.robots[0]
        wp = getattr(r, "_waypoint", None)
        gx, gy = wp[0] if wp is not None else (tx, ty)
        return math.hypot(float(r.xy[0]) - gx, float(r.xy[1]) - gy) < ARRIVE_M

    names = list(PLANS)
    C, M = len(names), args.reps
    total = np.zeros((C, M))
    plan_steps = np.full((C, M), -1, np.int32)
    phase1_steps = np.full((C, M), -1, np.int32)
    phase1_arrived = np.zeros((C, M), bool)
    memo_shared = {id(cfg): cfg, id(static): static}
    exo_rng.install()
    t_start = time.time()
    for k, rep in enumerate(range(args.rep_start, args.rep_start + args.reps)):
        for c, nm in enumerate(names):
            plan = PLANS[nm]
            e = copy.deepcopy(ep, dict(memo_shared))
            e.acc = []
            random.seed(10_000 * rep + 7 + seed)
            np.random.seed((10_000 * rep + 7 + seed) % 2 ** 32)
            torch.manual_seed(1000 + rep)
            exo_rng.EXO.reset(10_000 * rep + 7 + seed)
            acts = base.copy()
            phase, start = 0, branch_step
            if plan is None:
                acts[0] = det0
            else:
                kind, mode, speed = plan[0]
                tgt = target(kind, e)
                acts[0] = command(e, *tgt, mode, speed)
            e.decide(acts)
            while e.next_decision():
                if args.max_steps and e.step >= branch_step + args.max_steps:
                    break
                a_next = policy(e, det=True)
                if plan is not None and phase < len(plan):
                    kind, mode, speed = plan[phase]
                    limit = (STAY_S if kind == "stay" else PHASE_MAX_S) / dt
                    done = (e.step - start >= limit
                            or (kind != "stay" and arrived(e, *tgt)))
                    if done:
                        if phase == 0:
                            phase1_steps[c, k] = e.step - branch_step
                            phase1_arrived[c, k] = e.step - start < limit
                        phase += 1
                        start = e.step
                        if phase < len(plan):
                            kind, mode, speed = plan[phase]
                            tgt = target(kind, e)
                        else:
                            plan_steps[c, k] = e.step - branch_step
                    if phase < len(plan):
                        a_next[0] = command(e, *tgt, mode, speed)
                e.decide(a_next)
            r = np.asarray(e.acc, np.float64)
            total[c, k] = float((r * np.power(e.rew.g_step, np.arange(len(r)))).sum())
        print(f"[{name} t0={args.t0}] rep {rep} done {time.time() - t_start:.0f}s", flush=True)
    np.savez_compressed(args.out, names=np.asarray(names), total=total,
                        gamma=cfg.gamma(), decision_max_steps=cfg.decision_max_steps(),
                        rep_ids=np.arange(args.rep_start, args.rep_start + M),
                        plan_steps=plan_steps, phase1_steps=phase1_steps,
                        phase1_arrived=phase1_arrived, branch_step=branch_step)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
