"""Replay of one fixed policy, for the offline critic testbed.

Worker processes roll out the checkpoint's stochastic policy on the training
map source, as the trainer's workers do but with no curriculum level queue,
no exploration and no learning; the main process pushes every frame into a
ReplayBuffer (format 2) and saves it with its static layers. Run it under the
checkpoint's own decision settings (--overrides), so the replay's decisions,
rewards and discount are the ones the probe's ground truth uses.

The machine is shared with training runs, so the worker count follows free
memory: it starts --workers, adds one (up to --max-workers) while more than
--add-free-gb is available, and retires the newest when less than
--min-free-gb is (it exits after the episode it is running).

Usage: collect_replay.py CKPT OUT_DIR --episodes 800 --workers 4 --max-workers 10 \
           --overrides '{"GAMMA_WAYPOINT": 0.995, ...}'
Writes OUT_DIR/replay_buffer.npz and OUT_DIR/replay_static/, every
--save-every finished episodes and at the end.
"""
import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time

ROOT = "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch


def worker(wid, overrides, ckpt, seed, out_q, stop, retire):
    from configs import resolve_config
    import sim.model as model_module
    from learn.ADDS_AS_reinforcement import episode_uid, make_level_source
    from learn.networks import PolicyNetwork
    from learn.rollout import run_episode
    from sim.observation import build_static_layers

    torch.set_num_threads(1)
    cfg = resolve_config(overrides or None, check_data=False)
    rng = random.Random(seed * 7919 + wid)
    np.random.seed((seed + wid) % (2 ** 32))
    random.seed(seed + wid)
    torch.manual_seed(seed * 1000 + wid)
    policy = PolicyNetwork(cfg).eval()
    policy.load_state_dict(torch.load(ckpt, map_location="cpu",
                                      weights_only=False)["policy"])
    fresh_level = make_level_source(cfg)

    def act(obs, rec):
        with torch.no_grad():
            t = {k: torch.as_tensor(obs[k]) for k in ("ego", "mid", "glob", "state")}
            a, _ = policy.sample_action(t["ego"], t["mid"], t["glob"], t["state"])
        return a.numpy()

    idx = 0
    while not (stop.is_set() or retire.is_set()):
        level = fresh_level(rng)
        model = model_module.FightingModel(int(level.crowd_size), int(level.width),
                                           int(level.height), robot="Q", level=level)
        static = build_static_layers(model, cfg)
        uid = episode_uid(wid, idx, 0)
        out_q.put(("start", uid, static.meta(), static.arrays()))
        res = run_episode(model, cfg, act, gamma=cfg.gamma(),
                          seed=rng.randrange(1 << 30), static=static,
                          emit=lambda tr: out_q.put(("tr", uid, static.key, tr)))
        out_q.put(("done", uid, res.steps, res.decisions, float(res.total_reward)))
        idx += 1


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out_dir")
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-workers", type=int, default=10)
    ap.add_argument("--add-free-gb", type=float, default=5.0)
    ap.add_argument("--min-free-gb", type=float, default=2.0)
    ap.add_argument("--capacity", type=int, default=320_000)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overrides", default="")
    args = ap.parse_args()

    from configs import resolve_config
    from learn.replay import ReplayBuffer, StaticStore
    from sim.observation import StaticLayers

    over = json.loads(args.overrides) if args.overrides else {}
    cfg = resolve_config(over or None, check_data=False)
    print(f"gamma {cfg.gamma()} decision max steps {cfg.decision_max_steps()} "
          f"scope {cfg.ROBOT_DECISION_SCOPE} overrides {over}", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    store = StaticStore(os.path.join(args.out_dir, "replay_static"))
    buf = ReplayBuffer(cfg, args.capacity, store)
    path = os.path.join(args.out_dir, "replay_buffer.npz")

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=20_000)
    stop = ctx.Event()
    procs, next_wid = [], 0            # (process, retire event)

    def start_worker():
        nonlocal next_wid
        retire = ctx.Event()
        p = ctx.Process(target=worker, args=(next_wid, over, args.ckpt, args.seed, q,
                                             stop, retire), daemon=True)
        p.start()
        procs.append((p, retire))
        next_wid += 1

    for _ in range(args.workers):
        start_worker()
    done, t0, rets = 0, time.time(), []
    last_check = time.time()
    try:
        while done < args.episodes:
            if time.time() - last_check > 60:
                last_check = time.time()
                free = mem_available_gb()
                live = [(p, r) for p, r in procs if p.is_alive() and not r.is_set()]
                if free < args.min_free_gb and len(live) > 1:
                    live[-1][1].set()
                    print(f"free {free:.1f} GB: retiring a worker, {len(live) - 1} left", flush=True)
                elif free > args.add_free_gb and len(live) < args.max_workers:
                    start_worker()
                    print(f"free {free:.1f} GB: started a worker, {len(live) + 1} now", flush=True)
            try:
                msg = q.get(timeout=5.0)
            except queue.Empty:
                if not any(p.is_alive() for p, _ in procs):
                    raise RuntimeError("every worker died")
                continue
            if msg[0] == "start":
                store.put(StaticLayers.from_parts(msg[2], msg[3]))
            elif msg[0] == "tr":
                _, uid, key, tr = msg
                buf.push_transition(uid, key, tr)
                if tr.final:
                    buf.end_episode(uid)
            else:
                done += 1
                rets.append(msg[4])
                if done % 10 == 0:
                    print(f"{done} episodes, {len(buf)} frames, {buf.n_valid()} robot "
                          f"decisions, return {np.mean(rets[-10:]):.0f}, "
                          f"{time.time() - t0:.0f}s", flush=True)
                if done % args.save_every == 0:
                    buf.save(path, cfg.schema_versions())
    finally:
        stop.set()
        for p, _ in procs:
            p.terminate()
    buf.save(path, cfg.schema_versions())
    print(f"saved {path}: {len(buf)} frames, {buf.n_valid()} robot decisions, "
          f"{done} episodes, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
