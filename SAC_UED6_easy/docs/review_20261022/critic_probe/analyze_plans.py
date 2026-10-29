"""Two-phase plans against single decisions, paired by repeat.

Every condition here is deterministic continuation with person-level common
random numbers and the same seeds, so repeat r of one candidate shares its
crowd's exogenous draws with repeat r of every other candidate, in this run
and in results/persons_det_exo{,_hold3,_hold6}.

Per state:
  value    each candidate's mean return; the paired difference from the
           policy's own decision (policy_det) and from the best single
           decision of any length, mean +- standard error over repeats
  snr      over a candidate set, noise-corrected sd of the candidate means
           over the sd of one return around its candidate mean (the noise a
           critic's regression target carries), as in analyze_persons
Checks that policy_det reproduces the persons_det_exo run exactly.

Usage: analyze_plans.py [RESULTS_DIR]
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_persons as A  # noqa: E402

STATES = ["s2_t320", "s3_t320", "s7_t320", "s3_t80", "s6_t80"]
SINGLE = {"hold1 (~20 s)": "persons_det_exo", "hold3 (~60 s)": "persons_det_exo_hold3",
          "hold6 (~120 s)": "persons_det_exo_hold6"}


def load_plans(rdir, st):
    zs = [np.load(p) for p in sorted(glob.glob(os.path.join(rdir, "plans", f"{st}_r*.npz")))]
    names = [str(n) for n in zs[0]["names"]]
    cat = lambda k: np.concatenate([z[k] for z in zs], axis=1)
    return names, cat("total"), cat("plan_steps"), np.concatenate([z["rep_ids"] for z in zs])


def snr(x):
    a = A.anova(x)
    within = np.sqrt(a["rep_var"] + a["noise_var"])
    return float(np.sqrt(a["effect_var"]) / within) if within > 0 else 0.0


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    pooled = {"plan_vs_det": [], "plan_vs_best_single": [], "best_plan_vs_det": []}
    snrs = {k: [] for k in ["plans (8)", *SINGLE]}
    for st in STATES:
        names, tot, steps, reps = load_plans(rdir, st)
        singles = {lab: A.load_state(rdir, st, sub) for lab, sub in SINGLE.items()}
        h1 = singles["hold1 (~20 s)"]
        assert list(h1["reps"]) == list(reps)
        det_ref = h1["total"][h1["names"].index("policy_det")]
        det = tot[names.index("policy_det")]
        gap = float(np.abs(det - det_ref).max())
        print(f"\n== {st}  (policy_det reproduces persons_det_exo: max gap {gap:.2e})")
        # Best single decision of any length, by mean, excluding the policy's
        # own draws (kept as their own reference).
        best_lab, best_nm, best_x = None, None, None
        for lab, d in singles.items():
            for nm, x in zip(d["names"], d["total"]):
                if nm.startswith("policy"):
                    continue
                if best_x is None or x.mean() > best_x.mean():
                    best_lab, best_nm, best_x = lab, nm, x
        print(f"   {'candidate':24s} {'mean':>8s} {'vs policy_det':>16s} {'vs best single':>16s} {'plan s':>7s}")
        se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
        for nm, x, s in zip(names, tot, steps):
            d1, d2 = x - det, x - best_x
            print(f"   {nm:24s} {x.mean():8.1f} {d1.mean():+8.1f} ±{se(d1):5.1f} {d2.mean():+8.1f} ±{se(d2):5.1f} "
                  f"{np.mean(s[s >= 0]) * 0.5 if (s >= 0).any() else float('nan'):7.1f}")
        print(f"   best single: {best_nm} [{best_lab}] mean {best_x.mean():.1f}")
        shuttle_like = tot[names.index("P_in_off__out_guide")]
        pooled["plan_vs_det"].append(shuttle_like - det)
        pooled["plan_vs_best_single"].append(shuttle_like - best_x)
        plan_idx = [i for i, n in enumerate(names) if n.startswith("P_")]
        bp = max(plan_idx, key=lambda i: tot[i].mean())
        pooled["best_plan_vs_det"].append(tot[bp] - det)
        snrs["plans (8)"].append(snr(tot))
        for lab, d in singles.items():
            snrs[lab].append(snr(d["total"]))
        print("   snr: " + "  ".join(f"{k} {v[-1]:.2f}" for k, v in snrs.items()))
    print("\n== pooled over states (paired, mean ± se over all state-repeats)")
    for k, v in pooled.items():
        d = np.concatenate(v)
        print(f"   {k:22s} {d.mean():+8.1f} ± {d.std(ddof=1) / np.sqrt(len(d)):5.1f}   "
              f"(states with mean > 0: {sum(x.mean() > 0 for x in v)}/{len(v)})")
    print("   mean snr: " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in snrs.items()))


if __name__ == "__main__":
    main()
