"""Every critic variant in one process, trained on the same batches.

One replay load serves all variants, and each update draws one batch that
every variant trains on (its own n-step target where it asks for one), so the
variants differ only in architecture, loss and target. See testbed.py for
the variants and metrics.

Scalar heads start with their output bias at the mean target, so the fresh
heads do not spend their first updates learning the offset; categorical heads
start near the middle of their support.

Usage: testbed_multi.py --variants mse:1,hlgauss:1,dueling:1,hlgauss_dueling:1,mse:3 ...
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import testbed as T  # noqa: E402  (chdirs to the project root)

import numpy as np
import torch

from configs import resolve_config
from learn.replay import ReplayBuffer, StaticStore
from learn.sac import SACAgent
from sim import robot_action as ra


def multi_batch(replay, B, ns, g_step, rng):
    """One set of B start decisions; for each n in `ns`, the n-decision
    return, bootstrap factor and bootstrap state along next_dec."""
    nmax = max(ns)
    with replay.lock:
        dec = np.flatnonzero(replay.has_action[:replay.size])
        R = replay.robot_penalties.shape[2]
        idx = []
        per = {n: dict(nxt=[], rt=[], ro=[], boot=[]) for n in ns}
        while len(idx) < B:
            i = int(dec[rng.integers(dec.size)])
            if not replay.valid_transition(i):
                continue
            cur, team, own, disc, term = i, 0.0, np.zeros(R), 1.0, 0.0
            snaps = []
            for j in range(nmax):
                h = int(replay.hold[cur])
                pw = g_step ** np.arange(h)
                team += disc * float((replay.step_rewards[cur, :h] * pw).sum())
                own = own + disc * (replay.robot_penalties[cur, :h, :] * pw[:, None]).sum(0)
                disc *= g_step ** h
                nx = int(replay.next_dec[cur])
                if replay.terminal[cur]:
                    term = 1.0
                snaps.append((nx, team, own.copy(), disc * (1.0 - term)))
                if term or not (j < nmax - 1 and replay.has_action[nx]
                                and replay.valid_transition(nx)):
                    break
                cur = nx
            idx.append(i)
            for n in ns:
                nx, t, o, b = snaps[min(n, len(snaps)) - 1]
                per[n]["nxt"].append(nx)
                per[n]["rt"].append(t)
                per[n]["ro"].append(o)
                per[n]["boot"].append(b)
        s = replay._joint(idx)
        out = {"obs": {k: s[k] for k in T.KEYS}, "mask": s["mask"],
               "action": replay.action[idx].copy(), "n": {}}
        for n in ns:
            s2 = replay._joint(per[n]["nxt"])
            out["n"][n] = {"next_obs": {k: s2[k] for k in T.KEYS},
                           "next_mask": s2["mask"],
                           "r_team": np.asarray(per[n]["rt"], np.float32),
                           "r_own": np.asarray(per[n]["ro"], np.float32),
                           "boot": np.asarray(per[n]["boot"], np.float32)}
    return out


def to_dev(b, dev):
    t = lambda v: (T.to_torch(v, dev) if isinstance(v, dict)
                   else torch.as_tensor(v, dtype=torch.float32, device=dev))
    return {"obs": t(b["obs"]), "mask": t(b["mask"]), "action": t(b["action"]),
            "n": {n: {k: t(v) for k, v in d.items()} for n, d in b["n"].items()}}


def permute(b):
    mask = b["mask"]
    B, N = mask.shape
    perms = torch.argsort(torch.rand(B, N, device=mask.device), dim=1)

    def p(x):
        idx = perms
        while idx.ndim < x.ndim:
            idx = idx.unsqueeze(-1)
        return x.gather(1, idx.expand(*x.shape[:2], *x.shape[2:]))

    po = lambda o: {k: (p(v) if k in T.OBS_KEYS else v) for k, v in o.items()}
    return {"obs": po(b["obs"]), "mask": p(b["mask"]), "action": p(b["action"]),
            "n": {n: {"next_obs": po(d["next_obs"]), "next_mask": p(d["next_mask"]),
                      "r_team": d["r_team"], "r_own": p(d["r_own"]), "boot": d["boot"]}
                  for n, d in b["n"].items()}}


def target(agent, v, d):
    agent.q1_target, agent.q2_target = v["q1t"], v["q2t"]
    with torch.no_grad():
        v2 = agent._expected_next_value(d["next_obs"], d["next_mask"], per_robot=True)
        return d["r_team"][:, None] + d["r_own"] + d["boot"][:, None] * v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="mse:1,hlgauss:1,dueling:1,hlgauss_dueling:1,mse:3")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--static-dir", required=True)
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--bins", type=int, default=401)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--gpu-frac", type=float, default=0.6,
                    help="cap on this process's share of GPU memory (shared GPU)")
    ap.add_argument("--fill-test", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda" and args.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_frac))
    cfg = resolve_config(check_data=False)
    B = int(args.batch or cfg.BATCH_SIZE)
    variants = [(s.split(":")[0], int(s.split(":")[1])) for s in args.variants.split(",")]
    ns = sorted({n for _, n in variants})

    agent = SACAgent(cfg, device=args.device)
    ck = agent.load(args.ckpt)
    agent.policy.eval()
    for p in agent.policy.parameters():
        p.requires_grad_(False)
    g_step = agent.gamma ** (1.0 / max(1, int(cfg.ACTION_SCALE)))

    store = StaticStore(args.static_dir)
    t0 = time.time()
    if args.fill_test:
        from tests.test_madrl import _fill
        replay = ReplayBuffer(cfg, 2000, store)
        _fill(cfg, replay, store, agent, episodes=((3, 200), (3, 200)))
    else:
        replay = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store)
        replay.load(args.replay, cfg.schema_versions())
    print(f"replay {len(replay)} records in {time.time() - t0:.0f}s", flush=True)
    sample = lambda: multi_batch(replay, B, ns, g_step, rng)

    # Target scale from the checkpoint's own target critics (1-step).
    ys = []
    with torch.no_grad():
        for k in range(8):
            b = to_dev(multi_batch(replay, B, [1], g_step, np.random.default_rng(40_000 + k)), dev)
            v2 = agent._expected_next_value(b["n"][1]["next_obs"], b["n"][1]["next_mask"], per_robot=True)
            d = b["n"][1]
            y = d["r_team"][:, None] + d["r_own"] + d["boot"][:, None] * v2
            ys.append(y[b["mask"] > 0])
    y = torch.cat(ys)
    lo, hi, y_mean = float(y.min()), float(y.max()), float(y.mean())
    span = max(hi - lo, 1.0)
    support = (lo - 0.1 * span, hi + 0.1 * span, int(args.bins))
    print(f"targets mean {y_mean:.1f} range [{lo:.1f}, {hi:.1f}] support {support}", flush=True)

    V = []
    for kind, n in variants:
        q1 = T.make_critic(cfg, kind, support, ck["q1"], dev)
        q2 = T.make_critic(cfg, kind, support, ck["q2"], dev)
        for q in (q1, q2):
            if not q.dist:
                with torch.no_grad():
                    if q.dueling:
                        q.v_head[-1].bias.fill_(y_mean)
                    else:
                        q.head[-1].bias.fill_(y_mean)
        q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
        for p in list(q1t.parameters()) + list(q2t.parameters()):
            p.requires_grad_(False)
        opt = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()),
                                lr=float(cfg.LR), weight_decay=float(cfg.WD_Q))
        V.append(dict(name=f"{kind}:{n}", kind=kind, n=n, q1=q1, q2=q2,
                      q1t=q1t, q2t=q2t, opt=opt))

    probes = T.load_probes(args.probes)
    print(f"{len(probes)} probe states", flush=True)
    held = [to_dev(multi_batch(replay, B, ns, g_step, np.random.default_rng(10_000 + k)), dev)
            for k in range(4)]
    ab = to_dev(multi_batch(replay, 128, [1], g_step, np.random.default_rng(20_000)), dev)
    _r = random.Random(30_000)
    rand = torch.as_tensor(np.stack([[ra.random_action(_r) for _ in range(16)]
                                     for _ in range(128)]), dtype=torch.float32, device=dev)
    agree = (ab["obs"], ab["mask"], ab["action"], rand)

    log = open(args.out, "a")

    def write(r):
        log.write(json.dumps(r) + "\n")
        log.flush()
        print({k: (round(x, 4) if isinstance(x, float) else x)
               for k, x in r.items() if k != "probe_rows"}, flush=True)

    ref1 = T.make_critic(cfg, "mse", None, ck["q1"], dev)
    ref2 = T.make_critic(cfg, "mse", None, ck["q2"], dev)
    ref1.load_state_dict(ck["q1"])
    ref2.load_state_dict(ck["q2"])
    ref1.eval(), ref2.eval()
    write({"update": -1, "variant": "checkpoint", **T.agree_metrics(ref1, ref2, agree, dev),
           **T.probe_metrics(ref1, ref2, probes, dev)})
    del ref1, ref2
    agent.q1 = agent.q2 = None
    ck = None

    def evaluate(u, elapsed):
        for v in V:
            v["q1"].eval(), v["q2"].eval()
            td = []
            with torch.no_grad():
                for b in held:
                    yv = target(agent, v, b["n"][v["n"]])
                    q = torch.min(v["q1"](b["obs"], b["action"], b["mask"]),
                                  v["q2"](b["obs"], b["action"], b["mask"]))
                    m = b["mask"]
                    td.append(float((((q - yv) ** 2) * m).sum() / m.sum()))
            write({"update": u, "variant": v["name"], "td_mse": float(np.mean(td)),
                   "elapsed_s": elapsed,
                   **T.agree_metrics(v["q1"], v["q2"], agree, dev),
                   **T.probe_metrics(v["q1"], v["q2"], probes, dev)})
            v["q1"].train(), v["q2"].train()

    def save(u):
        os.makedirs(args.save_dir, exist_ok=True)
        for v in V:
            torch.save({"kind": v["kind"], "n": v["n"], "support": support, "update": u,
                        "q1": v["q1"].state_dict(), "q2": v["q2"].state_dict()},
                       os.path.join(args.save_dir, f"{v['kind']}_n{v['n']}_u{u}.pt"))

    pf = T.Prefetch(sample)
    t_start = time.time()
    tau = 0.995
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            evaluate(u, time.time() - t_start)
        if args.save_dir and u > 0 and (u % args.save_every == 0 or u == args.updates):
            save(u)
        if u == args.updates:
            break
        b = permute(to_dev(pf.get(), dev))
        for v in V:
            yv = target(agent, v, b["n"][v["n"]])
            v["opt"].zero_grad()
            # One critic's graph at a time: same gradients, half the peak memory.
            for q in (v["q1"], v["q2"]):
                q.loss(q.encode_observation(b["obs"]), b["action"], b["mask"], yv).backward()
            v["opt"].step()
            with torch.no_grad():
                for net, tgt in ((v["q1"], v["q1t"]), (v["q2"], v["q2t"])):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(tau).add_((1 - tau) * p)
    log.close()


if __name__ == "__main__":
    main()
