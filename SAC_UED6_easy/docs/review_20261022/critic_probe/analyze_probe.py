"""Summarise the counterfactual probe: how much a single decision of robot 0
changes its return, and how much of that the checkpoint critic sees.

Per state (two-way layout, candidates x repeats with common random numbers):
  true_sd    noise-corrected sd over candidates of the Monte Carlo return,
             sqrt(max(0, (MS_cand - MS_resid) / reps))
  noise_sd   sd of a single rollout's return around its candidate and repeat
             means (residual), i.e. the noise one TD sample carries
  critic_sd  sd over candidates of robot 0's min(q1, q2), and of q1, q2 alone
  corr       Pearson correlation across candidates, critic vs MC mean
Pooled: F test of the candidate effect over all states, and the ratio of
root-mean critic variance to root-mean true variance.

Usage: analyze_probe.py [RESULTS_DIR] [--horizon end|200s|400s]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def anova(mc):
    C, M = mc.shape
    g = mc.mean()
    cm, rm = mc.mean(1), mc.mean(0)
    ss_c = M * ((cm - g) ** 2).sum()
    resid = mc - cm[:, None] - rm[None, :] + g
    ss_e = (resid ** 2).sum()
    return ss_c, C - 1, ss_e, (C - 1) * (M - 1), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rdir", nargs="?", default=os.path.join(HERE, "results"))
    ap.add_argument("--horizon", default="end", choices=["end", "200s", "400s"])
    args = ap.parse_args()
    rows, pool = [], dict(ss_c=0.0, df_c=0, ss_e=0.0, df_e=0)
    for path in sorted(glob.glob(os.path.join(args.rdir, "s*_t*.json"))):
        if path.endswith(("_r3.json", "_more.json")):
            continue
        d = json.load(open(path))
        cands = list(d["cands"])
        src = d["mc"] if args.horizon == "end" else d["mc_h"][args.horizon]
        mc = np.asarray([src[c] for c in cands], np.float64)
        ss_c, df_c, ss_e, df_e, cm = anova(mc)
        M = mc.shape[1]
        ms_c, ms_e = ss_c / df_c, (ss_e / df_e if df_e else np.nan)
        for k, v in (("ss_c", ss_c), ("df_c", df_c), ("ss_e", ss_e), ("df_e", df_e)):
            pool[k] += v
        q1 = np.asarray([d["q1"][c] for c in cands])
        q2 = np.asarray([d["q2"][c] for c in cands])
        qm = np.minimum(q1, q2)
        rows.append(dict(
            state=os.path.basename(path)[:-5], scenario=d["scenario"],
            branch_s=d["branch_step"] * 0.5, inside=d["inside_at_branch"],
            n_ref=d["n_ref"], robot0_sd_m=d["robot0_signed_distance"],
            true_sd=float(np.sqrt(max(0.0, (ms_c - ms_e) / M))),
            noise_sd=float(np.sqrt(ms_e)), F=float(ms_c / ms_e),
            p=float(stats.f.sf(ms_c / ms_e, df_c, df_e)),
            range_true=float(cm.max() - cm.min()),
            critic_sd=float(qm.std(ddof=1)), q1_sd=float(q1.std(ddof=1)),
            q2_sd=float(q2.std(ddof=1)), range_critic=float(qm.max() - qm.min()),
            corr=float(np.corrcoef(qm, cm)[0, 1]),
            corr_q1q2=float(np.corrcoef(q1, q2)[0, 1]),
            spearman=float(stats.spearmanr(qm, cm).correlation),
            best_true=cands[int(cm.argmax())], best_critic=cands[int(qm.argmax())],
            mc_mean={c: float(v) for c, v in zip(cands, cm)},
            q_min={c: float(v) for c, v in zip(cands, qm)},
            first_hold={c: d["first_hold"][c] for c in cands}))
    if not rows:
        print("no results")
        return
    F = (pool["ss_c"] / pool["df_c"]) / (pool["ss_e"] / pool["df_e"])
    p = stats.f.sf(F, pool["df_c"], pool["df_e"])
    tv = np.mean([r["true_sd"] ** 2 for r in rows])
    cv = np.mean([r["critic_sd"] ** 2 for r in rows])
    summary = dict(
        horizon=args.horizon, states=len(rows),
        pooled_F=float(F), pooled_df=(pool["df_c"], pool["df_e"]), pooled_p=float(p),
        mean_true_sd=float(np.mean([r["true_sd"] for r in rows])),
        mean_noise_sd=float(np.mean([r["noise_sd"] for r in rows])),
        mean_critic_sd=float(np.mean([r["critic_sd"] for r in rows])),
        ratio_rms=float(np.sqrt(cv / tv)) if tv > 0 else None,
        mean_corr=float(np.mean([r["corr"] for r in rows])),
        positive_corr=int(sum(r["corr"] > 0 for r in rows)),
        mean_spearman=float(np.mean([r["spearman"] for r in rows])),
        mean_corr_q1q2=float(np.mean([r["corr_q1q2"] for r in rows])),
        best_agree=int(sum(r["best_true"] == r["best_critic"] for r in rows)))
    print(json.dumps(summary, indent=1))
    print(f"\n{'state':10s} {'scenario':24s} {'t(s)':>5s} {'in/Nref':>9s} {'r0 sd':>6s} "
          f"{'true':>6s} {'noise':>6s} {'p':>7s} {'critic':>7s} {'corr':>6s} {'q1q2':>6s}  best(true/critic)")
    for r in rows:
        print(f"{r['state']:10s} {r['scenario']:24s} {r['branch_s']:5.0f} "
              f"{r['inside']:4d}/{r['n_ref']:<4.0f} {r['robot0_sd_m']:6.1f} {r['true_sd']:6.1f} "
              f"{r['noise_sd']:6.1f} {r['p']:7.1e} {r['critic_sd']:7.2f} {r['corr']:+6.2f} "
              f"{r['corr_q1q2']:+6.2f}  {r['best_true']}/{r['best_critic']}")
    out = os.path.join(args.rdir, f"summary_{args.horizon}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
