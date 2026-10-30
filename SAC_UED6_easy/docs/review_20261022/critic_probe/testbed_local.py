"""Critics trained on the return of the people near the deciding robot.

critic_diagnosis.md section 10 found that the people near a robot carry most
of its decision's effect and little of the noise. This testbed checks whether
a critic trained on that return ranks the robot's actions better, judged by
the whole team's return (the probe ground truth).

Each variant is the current critic (CentralizedCritic: encoders from the
checkpoint, the action pathway fresh) regressed by Monte Carlo on one target
of person_book.py, per robot decision of a collect_replay.py --person-returns
replay:

  local20, local40  people within 20 / 40 m of the robot at its decision, 200 s
  global            the whole team, 200 s (the control for local*)
  global_full       the whole team to the end of the episode

All variants train on the same batches. Every --eval-every updates (JSONL, the
format of testbed_spatial.py, so analyze_spatial.py reads it): held-out error
and the values of robot 0's candidates on every probe state.

Usage: testbed_local.py --ckpt CK --replay-dir DIR --out OUT.jsonl --overrides JSON \
           [--variants local20:0,local40:0,global:0,global_full:0] [--save-dir DIR]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed_spatial as T  # noqa: E402  (chdirs to the project root)

import numpy as np
import torch
import torch.nn.functional as F

from configs import resolve_config
from learn.networks import CentralizedCritic
from learn.replay import BatchPrefetcher, ReplayBuffer, StaticStore
from learn.sac import SACAgent, permute_robots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="local20:0,local40:0,global:0,global_full:0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--probe-dirs", default=",".join([os.path.join(T.HERE, "results"),
                                                      os.path.join(T.HERE, "results", "gen")]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--updates", type=int, default=12000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--gpu-frac", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda" and args.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_frac))
    cfg = resolve_config(json.loads(args.overrides) if args.overrides else None, check_data=False)
    B = int(args.batch or cfg.BATCH_SIZE)
    agent = SACAgent(cfg, device="cpu")
    ck = agent.load(args.ckpt)

    store = StaticStore(os.path.join(args.replay_dir, "replay_static"))
    replay = ReplayBuffer(cfg, 400_000, store)
    replay.load(os.path.join(args.replay_dir, "replay_buffer.npz"), cfg.schema_versions())
    z = np.load(os.path.join(args.replay_dir, "local_targets.npz"))
    names = [str(n) for n in z["names"]]
    tg, ok = z["targets"], z["valid"]
    print(f"replay {len(replay)} frames, {replay.n_valid()} robot decisions, "
          f"{int(ok.sum())} with targets, booking error {float(z['max_err']):.1e}", flush=True)
    sel = tg[ok]
    for k, n in enumerate(names):
        print(f"   {n:12s} mean {sel[:, k].mean():8.1f} sd {sel[:, k].std():7.1f}", flush=True)

    def batch_of(b):
        i, j = b["indices"], b["agent_index"]
        out = T.to_batch(b, dev)
        out["targets"] = torch.as_tensor(tg[i, j], device=dev)            # (B, K)
        out["has"] = torch.as_tensor(ok[i, j], device=dev, dtype=torch.float32)
        return out

    V = []
    for spec in args.variants.split(","):
        target, init = (spec.split(":") + ["0"])[:2]
        k = names.index(target)
        name = target if int(init) == 0 else f"{target}_s{init}"
        torch.manual_seed(1000 + int(init))
        q = T.make_critic(cfg, "vector", ck["q1"], float(sel[:, k].mean()), dev)
        opt = torch.optim.AdamW(q.parameters(), lr=float(cfg.LR), weight_decay=float(cfg.WD_Q))
        V.append(dict(name=name, k=k, q=q, opt=opt))
    probes = T.load_probes(args.probe_dirs.split(","))
    print(f"{len(probes)} probe states", flush=True)
    held = [batch_of(replay.sample(B, np.random.default_rng(10_000 + s))) for s in range(4)]
    log = open(args.out, "a")

    ref = CentralizedCritic(cfg, T.ra.ACTION_DIM).to(dev)
    ref.load_state_dict(ck["q1"])
    ref.eval()
    log.write(json.dumps({"update": -1, "variant": "checkpoint",
                          "probe": T.probe_values(ref, ref, probes, dev)}) + "\n")
    del ref

    def evaluate(u, elapsed):
        for v in V:
            v["q"].eval()
            err = []
            with torch.no_grad():
                for b in held:
                    rows = torch.arange(b["mask"].shape[0], device=dev)
                    q = v["q"](b["obs"], b["action"], b["mask"])[rows, b["agent_index"]]
                    e = (q - b["targets"][:, v["k"]]) ** 2 * b["has"]
                    err.append(float(e.sum() / b["has"].sum().clamp(min=1)))
            log.write(json.dumps({"update": u, "variant": v["name"], "mse": float(np.mean(err)),
                                  "elapsed_s": elapsed,
                                  "probe": T.probe_values(v["q"], v["q"], probes, dev)}) + "\n")
            log.flush()
            print(f"u {u} {v['name']}: mse {np.mean(err):.1f} ({elapsed:.0f}s)", flush=True)
            v["q"].train()
            if args.save_dir and u > 0:
                os.makedirs(args.save_dir, exist_ok=True)
                sd = v["q"].state_dict()
                torch.save({"kind": "vector", "name": v["name"], "update": u, "q1": sd, "q2": sd},
                           os.path.join(args.save_dir, f"{v['name']}_u{u}.pt"))

    pf = BatchPrefetcher(replay, B, seed=args.seed)
    t_start = time.time()
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            evaluate(u, time.time() - t_start)
        if u == args.updates:
            break
        raw = pf.get()
        b = batch_of(raw)
        b = {**permute_robots(b), "targets": b["targets"], "has": b["has"]}
        rows = torch.arange(b["mask"].shape[0], device=dev)
        j = b["agent_index"]
        for v in V:
            v["opt"].zero_grad()
            q = v["q"](b["obs"], b["action"], b["mask"])[rows, j]
            loss = ((q - b["targets"][:, v["k"]]) ** 2 * b["has"]).sum() / b["has"].sum().clamp(min=1)
            loss.backward()
            v["opt"].step()
    pf.stop()
    log.close()


if __name__ == "__main__":
    main()
