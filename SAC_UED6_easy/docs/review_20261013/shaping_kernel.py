"""Calibrate the rew-v6 approach potential's PEOPLE (kappa).

Phi_j = 1 - exp(-S_j / kappa); S_j does not depend on kappa, so one pass per
(level, policy) records S_j every step and every kappa is evaluated offline.

    python3 docs/review_20261013/shaping_kernel.py record LEVEL POLICY   # shuttle|random|still
    python3 docs/review_20261013/shaping_kernel.py eval
Results of 2026-10-15 (levels 0, 3, 8; 800 steps; w = 20): configs/training/common.py.
"""
import os, sys, random
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
OUT = os.path.join(HERE, "results", "shaping_kernel")
LEVELS = (0, 3, 8)


def record(li, kind):
    from configs import resolve_config
    from learn.zero_shot import validation_levels, build_eval_model
    from sim.rewards import TaskReward
    from sim.robot_action import apply_to, encode
    from validation.shuttle_baseline import ShuttlePolicy
    cfg = resolve_config({"REWARD_VERSION": "rew-v6-pbrs",
                          "REWARD_SHAPE_APPROACH_PEOPLE": 1e9})
    name, level = validation_levels(cfg)[li]
    random.seed(0); np.random.seed(0)
    m = build_eval_model(level, 7); m.should_finish = lambda: False
    task = TaskReward(m, cfg)
    pol = ShuttlePolicy(m, len(m.robots), cfg=cfg)
    rng = np.random.default_rng(0)
    S = []
    for t in range(800):
        if t % 8 == 0:
            for i, rb in enumerate(m.robots):
                if kind == "shuttle":
                    a = pol._act_one(i, rb)
                elif kind == "random":
                    a = encode(tuple(rng.uniform(-2, 2, 2)),
                               rng.choice(["off", "guide"]), speed=1.0)
                else:
                    a = encode((0.0, 0.0), "guide", speed=0.6)
                apply_to(rb, a)
        m.step(); task.step(m)
        S.append(np.array(task._phi_appr) * 1e9)   # kappa = 1e9: Phi = S / kappa
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, f"S_{li}_{kind}.npy"), np.array(S))
    print("done", name, kind)


def evaluate():
    g = 0.99 ** 0.25
    w = 20.0
    for li in LEVELS:
        for kind in ("shuttle", "random", "still"):
            S = np.load(os.path.join(OUT, f"S_{li}_{kind}.npy"))
            print(f"  L{li} {kind:7s} S start={np.round(S[0], 1)} median={np.median(S):.1f} "
                  f"p90={np.percentile(S, 90):.1f} zero-share={(S == 0).mean():.2f}")
    print("kappa | per-4s robot F sd: shuttle random still | moving/still | "
          "max|F| per 4s | share Phi > 0.9")
    for kappa in (2, 3, 5, 10, 20):
        rows = {}
        for kind in ("shuttle", "random", "still"):
            F4, mx, sat = [], 0.0, []
            for li in LEVELS:
                S = np.load(os.path.join(OUT, f"S_{li}_{kind}.npy"))
                phi = -np.expm1(-S / kappa)
                F = w * (g * phi[1:] - phi[:-1])
                n = (len(F) // 8) * 8
                f4 = F[:n].reshape(-1, 8, F.shape[1]).sum(1)
                F4.append(f4.ravel()); mx = max(mx, np.abs(f4).max())
                sat.append((phi > 0.9).mean())
            rows[kind] = (np.concatenate(F4).std(), mx, np.mean(sat))
        sh, ra, st = rows["shuttle"][0], rows["random"][0], rows["still"][0]
        print(f"{kappa:5d} | {sh:6.2f} {ra:6.2f} {st:6.2f} | {sh/st:5.2f} {ra/st:5.2f} | "
              f"{max(r[1] for r in rows.values()):6.2f} | "
              + " ".join(f"{k}:{rows[k][2]:.2f}" for k in rows))


if __name__ == "__main__":
    if sys.argv[1] == "record":
        record(int(sys.argv[2]), sys.argv[3])
    else:
        evaluate()
