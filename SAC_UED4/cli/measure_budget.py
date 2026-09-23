"""Measure what one run costs before choosing BUFFER_SIZE and the 400 m density.

docs/outdoor_madrl_redesign.md §6 and §9 stage 5. For each crop size:

  * simulation cost per step at the target density, with three robots;
  * observation cost per decision (measurement + building the team's input);
  * replay bytes per decision instant, batch rebuild time, one GPU update,
    peak GPU memory and the process's resident memory.

    python3 -m cli.measure_budget [--sizes 100 200 400] [--steps 40]
                                  [--batch 128] [--out docs/replay_memory_budget.json]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import tempfile
import time

import numpy as np


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def measure(size: int, steps: int, batch: int, density=None, level=None,
            label=None) -> dict:
    import torch

    import sim.model as M
    from configs import resolve_config
    from learn.replay import ReplayBuffer, StaticStore
    from learn.rollout import run_episode
    from learn.sac import SACAgent
    from sim.observation import build_static_layers
    from ued.level import generate_city_level

    cfg = resolve_config(check_data=False)
    rng = random.Random(size)
    if level is None:
        level = generate_city_level(rng, difficulty=5, crowd_size=None,
                                    width=size, height=size, seed=size,
                                    robot_num=3)
    level.robot_num = 3
    level.augmentation = "identity"
    random.seed(0)
    np.random.seed(0)
    t0 = time.perf_counter()
    model = M.FightingModel(int(level.crowd_size), level.width, level.height,
                            robot="Q", level=level)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    static = build_static_layers(model, cfg)
    static_s = time.perf_counter() - t0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAgent(cfg, device=device)
    with tempfile.TemporaryDirectory() as tmp:
        store = StaticStore(tmp)
        store.put(static)
        buf = ReplayBuffer(cfg, 4 * steps, store)

        def emit(tr):
            buf.push(0, tr.step, tr.record, static.key, tr.action,
                     tr.step_rewards, tr.terminal)

        t0 = time.perf_counter()
        res = run_episode(model, cfg,
                          lambda o, r: agent.act(o, epsilon=0.2),
                          gamma=0.99, max_steps=steps, static=static,
                          emit=emit)
        episode_s = time.perf_counter() - t0
        n_valid = sum(buf.valid_transition(i) for i in range(buf.size))
        b = min(batch, max(1, n_valid))
        t0 = time.perf_counter()
        sample = buf.sample(b)
        sample_s = time.perf_counter() - t0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        agent.update(sample)            # warm-up (kernels, allocator)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent.update(sample)
        if device == "cuda":
            torch.cuda.synchronize()
        update_s = time.perf_counter() - t0
        gpu_peak = (torch.cuda.max_memory_allocated() / 2 ** 20
                    if device == "cuda" else None)
        per_instant = buf.nbytes() / buf.capacity
        static_mb = sum(a.nbytes for a in static.arrays().values()) / 2 ** 20

    decisions = max(1, res.decisions)
    return {
        "label": label or f"generated_{size}m",
        "size_m": size, "crowd": int(level.crowd_size),
        "density": float(getattr(level, "crowd_density", 0) or 0),
        "robots": 3, "steps": res.steps,
        "model_build_s": build_s, "static_layers_s": static_s,
        "static_layers_mb": static_mb,
        "sim_ms_per_step": res.sim_ms_per_step,
        "episode_wall_s": episode_s,
        "obs_and_act_ms_per_decision":
            1000.0 * (episode_s - res.sim_ms_per_step * res.steps / 1000.0)
            / decisions,
        "projected_episode_min_at_max_steps": (
            episode_s / max(1, res.steps)) * 2000 / 60.0,
        "replay_bytes_per_instant": per_instant,
        "replay_gb_at_100k": per_instant * 100_000 / 1e9,
        "batch": b, "batch_rebuild_s": sample_s,
        "gpu_update_s": update_s, "gpu_peak_mb": gpu_peak,
        "rss_mb": _rss_mb(),
        "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--final", action="store_true",
                    help="also measure the final zero-shot crop")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for size in args.sizes:
        row = measure(size, args.steps, args.batch)
        rows.append(row)
        print(json.dumps(row, indent=1))
    if args.final:
        from configs import resolve_config
        from learn.zero_shot import draw_final_hazard, final_base_level
        cfg = resolve_config()
        base, _ = final_base_level(cfg)
        lv, _ = draw_final_hazard(base, cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS[0], cfg)
        row = measure(int(lv.width), args.steps, args.batch, level=lv,
                      label=f"final_{cfg.FINAL_ZERO_SHOT_SITE}")
        rows.append(row)
        print(json.dumps(row, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
