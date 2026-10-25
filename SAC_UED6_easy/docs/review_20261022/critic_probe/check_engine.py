"""The probe's Episode engine must reproduce learn/rollout.run_episode.

Same level, seeds and policy: robot 0's per-step learning reward (team
step_rewards + its robot_penalties) and the decision boundaries must match.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import true_adv_v2 as P  # noqa: E402  (chdirs to the project root)

import numpy as np
import torch

from configs import resolve_config
from learn.rollout import run_episode
from learn.sac import OBS_KEYS, SACAgent, to_torch
from learn.zero_shot import build_eval_model, validation_levels, validation_seed
from sim.observation import build_static_layers

ckpt, scen, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = resolve_config(check_data=False)
agent = SACAgent(cfg, device="cpu")
agent.load(ckpt, policy_only=True)
agent.policy.eval()
name, lv = validation_levels(cfg)[scen]
seed = validation_seed(name)


def act(obs, rec=None):
    with torch.no_grad():
        t = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
        a, _ = agent.policy.sample_action(*(t[k] for k in OBS_KEYS))
    return a.numpy()


# Reference: run_episode.
m = build_eval_model(lv, seed)
torch.manual_seed(seed)
ref_r0, ref_dec = [], []


def emit(tr):
    if tr.action is None or tr.frame_only:
        return
    ref_dec.append(tr.step)
    for k, r in enumerate(tr.step_rewards):
        ref_r0.append(r + tr.robot_penalties[k][0])


run_episode(m, cfg, act, gamma=cfg.gamma(), max_steps=steps, seed=seed,
            static=build_static_layers(m, cfg), emit=emit)

# Probe engine.
m = build_eval_model(lv, seed)
static = build_static_layers(m, cfg)
ep = P.Episode(m, cfg, static, seed)
ep.max_steps = steps
torch.manual_seed(seed)
dec = []
while ep.next_decision():
    dec.append(ep.last_frame // ep.F)
    ep.decide(act(ep.hist.team_observations()))

a, b = np.asarray(ref_r0), np.asarray(ep.acc)
print("steps", len(a), len(b), "decisions", len(ref_dec), len(dec))
print("decision frames equal:", ref_dec == dec)
print("max |r0 diff|:", float(np.abs(a - b).max()) if len(a) == len(b) else "length differs")
print("sum r0", a.sum(), b.sum())
