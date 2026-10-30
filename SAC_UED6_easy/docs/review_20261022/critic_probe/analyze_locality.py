"""Signal and noise of robot 0's decision when only the people near it count.

From the person-level probe (true_adv_persons.py: per-person returns Rp and
each person's distance from robot 0 at the branch, dist0), per state: the
noise-corrected effect sd over candidates and the interaction-noise sd of the
return summed over people within D m, as analyze_persons.anova.

Usage: analyze_locality.py [RESULTS_DIR]
"""
import glob
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DS = [10, 20, 40, 60, 100, 1e9]


def anova(x):
    C, M = x.shape
    m = x.mean(1)
    resid = x - m[:, None] - x.mean(0)[None, :] + x.mean()
    noise = (resid ** 2).sum() / ((C - 1) * (M - 1))
    return np.sqrt(max(m.var(ddof=1) - noise / M, 0.0)), np.sqrt(noise)


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    for cond in ["persons_det_exo", "persons_det_exo_hold6"]:
        files = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(rdir, cond, "s*_r*.npz"))):
            files[os.path.basename(f).rsplit("_r", 1)[0]].append(np.load(f, allow_pickle=True))
        print(f"== {cond}: effect sd / noise sd (snr), people within D m of robot 0 at the branch")
        agg = defaultdict(list)
        for st, zs in files.items():
            Rp = np.concatenate([z["Rp"] for z in zs], 1).astype(np.float64)
            d = zs[0]["dist0"]
            cells = []
            for D in DS:
                e, n = anova(Rp[:, :, d < D].sum(-1))
                cells.append(f"{e:5.1f}/{n:5.1f} ({e / n if n else 0:.2f})")
                agg[D].append((e, n))
            print(f"   {st:10s} " + " ".join(cells))
        print("   mean       " + " ".join(
            f"{np.mean([e for e, _ in agg[D]]):5.1f}/{np.mean([n for _, n in agg[D]]):5.1f} "
            f"({np.mean([e / n for e, n in agg[D]]):.2f})" for D in DS))


if __name__ == "__main__":
    main()
