"""Rebuild each probe's branch state and save the critic's inputs there.

Replays true_adv_v2's prefix with the same seeds and checks that it lands on
the same step and robot position, and that the checkpoint critic gives the
same Q as the probe recorded. Writes results/<name>_state.npz next to each
results/<name>.json: the joint observation, mask, and one applied action
matrix per candidate.

Probes run on any level (validation index or SITE:SIZE:K, true_adv_v2
`probe_level`) and under their recorded overrides.

Usage: dump_states.py CKPT [RESULTS_DIR]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
import true_adv_v2 as P  # noqa: E402  (chdirs to the project root)

import numpy as np
import torch

from configs import resolve_config
from learn.replay import joint_observation
from learn.sac import OBS_KEYS, SACAgent, to_torch
from learn.zero_shot import build_eval_model, validation_levels, validation_seed
from sim.observation import build_static_layers

KEYS = OBS_KEYS + ("priv", "priv_scalars")


def main():
    ckpt = sys.argv[1]
    rdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")
    cfg = resolve_config(check_data=False)
    agent = SACAgent(cfg, device="cpu")
    agent.load(ckpt)
    agent.policy.eval()
    for path in sorted(glob.glob(os.path.join(rdir, "s*_t*.json"))):
        out = path[:-5] + "_state.npz"
        if os.path.exists(out) or path.endswith(("_r3.json", "_more.json")):
            continue
        d = json.load(open(path))
        cfg = resolve_config(d.get("overrides") or None, check_data=False)
        name, lv, seed = P.probe_level(cfg, d.get("level_spec", d["scen"]))
        m = build_eval_model(lv, seed)
        static = build_static_layers(m, cfg)
        ep = P.Episode(m, cfg, static, seed)
        torch.manual_seed(seed)
        while ep.next_decision():
            if ep.step >= d["t0_request"]:
                break
            obs = ep.hist.team_observations()
            with torch.no_grad():
                t = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
                a, _ = agent.policy.sample_action(*(t[k] for k in OBS_KEYS))
            ep.decide(a.numpy())
        xy = [float(ep.m.robots[0].xy[0]), float(ep.m.robots[0].xy[1])]
        assert ep.step == d["branch_step"], (path, ep.step, d["branch_step"])
        assert np.allclose(xy, d["robot0_xy"]), (path, xy, d["robot0_xy"])
        joint = joint_observation(cfg, [static], [ep.hist.window()])
        cands = list(d["cands"])
        applied = np.asarray([d["applied"][c] for c in cands], np.float32)
        tobs = to_torch({k: joint[k] for k in KEYS}, "cpu")
        mask = torch.as_tensor(joint["mask"])
        with torch.no_grad():
            q1 = [float(agent.q1(tobs, torch.as_tensor(a)[None], mask)[0, 0])
                  for a in applied]
        err = max(abs(a - d["q1"][c]) for a, c in zip(q1, cands))
        assert err < 1e-3, (path, err)
        np.savez_compressed(out, **{k: joint[k] for k in KEYS + ("mask",)},
                            applied=applied, cands=np.asarray(cands))
        print("saved", out, "q1 check", f"{err:.1e}", flush=True)


if __name__ == "__main__":
    main()
