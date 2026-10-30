"""Score testbed_spatial.py critics against the probe ground truth.

Per probe state (8 candidates for robot 0, Monte Carlo returns per repeat):

  reliability  how well the Monte Carlo means can rank the candidates at
               all: the split-half correlation of the candidate means over
               random halves of the repeats, Spearman-Brown corrected. Its
               square root is the correlation a perfect critic would reach
               on average (the noise ceiling).
  corr         Pearson correlation of the critic's values with the candidate
               means
  regret       best candidate mean minus the mean of the critic's top
               candidate, over the same for a uniformly random pick (0 = picks
               the best, 1 = no better than chance)

  pair accuracy  over candidate pairs whose return difference is significant
               (paired over repeats, which share their random numbers; t test
               p < 0.05, states with >= 6 repeats), the share the critic
               orders the same way; confidence interval by resampling states

A critic's values at one update are noisy (earlier testbed: checkpoint to
checkpoint variation as large as between variants), so each variant is
scored on its values averaged over the updates in --late (default: the last
half of training), and per update for the curve.

Usage: analyze_spatial.py TESTBED.jsonl [--late 10000] [--min-rel 0.5]
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ground_truth(dirs):
    gt = {}
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "s*_t*.json"))):
            base = os.path.basename(path)[:-5]
            if base.endswith(("_r3", "_more")) or not os.path.exists(path[:-5] + "_state.npz"):
                continue
            js = json.load(open(path))
            z = np.load(path[:-5] + "_state.npz")
            cands = [str(c) for c in z["cands"]]
            gt[base] = np.asarray([js["mc"][c] for c in cands], np.float64)   # (C, reps)
    return gt


def reliability(mc, splits=200, rng=None):
    rng = rng or np.random.default_rng(0)
    C, M = mc.shape
    if M < 4:
        return float("nan")
    rs = []
    for _ in range(splits):
        p = rng.permutation(M)
        a, b = mc[:, p[:M // 2]].mean(1), mc[:, p[M // 2:]].mean(1)
        if a.std() > 0 and b.std() > 0:
            rs.append(np.corrcoef(a, b)[0, 1])
    r = float(np.mean(rs)) if rs else 0.0
    return 2 * r / (1 + r) if r > -1 else float("nan")


def scores(q, mc):
    m = mc.mean(1)
    q = np.asarray(q, np.float64)
    corr = float(np.corrcoef(q, m)[0, 1]) if q.std() > 0 else 0.0
    best, rand = m.max(), m.mean()
    regret = float((best - m[int(np.argmax(q))]) / (best - rand)) if best > rand else 0.0
    return corr, regret


T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201}


def significant_pairs(mc):
    """[(a, b)] with candidate a's return significantly above b's."""
    C, M = mc.shape
    if M < 6:
        return []
    out = []
    for a in range(C):
        for b in range(C):
            if a == b:
                continue
            d = mc[a] - mc[b]
            se = d.std(ddof=1) / np.sqrt(M)
            if se > 0 and d.mean() / se > T95.get(M - 1, 2.0):
                out.append((a, b))
    return out


def pair_accuracy(qs, pairs, states, boots=2000, rng=None):
    """Share of significant pairs ordered correctly, and a 95% interval
    from resampling states."""
    rng = rng or np.random.default_rng(0)
    per = []
    for s in states:
        q = qs[s]
        per.append((sum(q[a] > q[b] for a, b in pairs[s]), len(pairs[s])))
    per = np.asarray(per, float)
    if per[:, 1].sum() == 0:
        return float("nan"), (float("nan"), float("nan"))
    acc = per[:, 0].sum() / per[:, 1].sum()
    bs = []
    for _ in range(boots):
        i = rng.integers(0, len(per), len(per))
        n = per[i, 1].sum()
        if n > 0:
            bs.append(per[i, 0].sum() / n)
    return float(acc), (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--late", type=int, default=None, help="first update of the averaged window")
    ap.add_argument("--min-rel", type=float, default=0.5)
    ap.add_argument("--pair", default="spatial,vector",
                    help="A,B: report A minus B, pairing A<suffix> with B<suffix> over seeds")
    ap.add_argument("--dirs", default=",".join([os.path.join(HERE, "results"),
                                                os.path.join(HERE, "results", "gen")]))
    args = ap.parse_args()
    gt = ground_truth(args.dirs.split(","))
    rows = [json.loads(l) for l in open(args.jsonl)]
    last = max(r["update"] for r in rows)
    late = args.late if args.late is not None else last // 2
    rel = {s: reliability(mc) for s, mc in gt.items()}
    group = {}
    by = defaultdict(lambda: defaultdict(list))          # variant -> state -> [q over late updates]
    curve = defaultdict(lambda: defaultdict(list))       # variant -> update -> [corr on reliable states]
    for r in rows:
        for p in r["probe"]:
            s = p["state"]
            if s not in gt:
                continue
            group[s] = p["group"]
            if r["update"] == -1 or r["update"] >= late:
                by[r["variant"]][s].append(p["q"])
            if rel[s] >= args.min_rel:
                curve[r["variant"]][r["update"]].append(scores(p["q"], gt[s])[0])

    print(f"probe states: {len(gt)}; reliability >= {args.min_rel}: "
          f"{sum(v >= args.min_rel for v in rel.values())}")
    print(f"{'state':34s} {'group':11s} {'reps':>4s} {'rel':>5s} {'ceil':>5s}")
    for s in sorted(gt, key=lambda s: (group.get(s, ""), s)):
        print(f"{s:34s} {group.get(s, '?'):11s} {gt[s].shape[1]:4d} {rel[s]:5.2f} "
              f"{np.sqrt(max(rel[s], 0)):5.2f}")

    print(f"\n== values averaged over updates >= {late} (checkpoint: its own values)")
    groups = sorted(set(group.values()))
    variants = list(by)
    table = {}
    for v in variants:
        for s, qs in by[v].items():
            table[v, s] = scores(np.mean(qs, 0), gt[s])
    for sel_name, sel in (("reliable", lambda s: rel[s] >= args.min_rel), ("all", lambda s: True)):
        print(f"\n-- {sel_name} states: mean corr (ceiling) / mean regret, per group")
        for g in groups + ["ALL"]:
            ss = [s for s in gt if (g == "ALL" or group.get(s) == g) and sel(s)]
            if not ss:
                continue
            ceil = np.nanmean([np.sqrt(max(rel[s], 0)) if rel[s] == rel[s] else np.nan for s in ss])
            line = f"   {g:11s} n={len(ss):2d} ceil {ceil:4.2f} |"
            for v in variants:
                c = [table[v, s][0] for s in ss if (v, s) in table]
                rg = [table[v, s][1] for s in ss if (v, s) in table]
                line += f" {v} {np.mean(c):+.2f}/{np.mean(rg):.2f}"
            print(line)
    # Pairs trained from the same initialisation seed, averaged over seeds.
    A_, B_ = args.pair.split(",")
    seeds = [v[len(A_):] for v in variants
             if v.startswith(A_) and (v[len(A_):] == "" or v[len(A_):].startswith("_s"))
             and B_ + v[len(A_):] in variants]
    if seeds:
        print(f"\n-- {A_} minus {B_}, corr, mean over {len(seeds)} seed pair(s), +- se over states")
        for g in groups + ["ALL"]:
            for sel_name, sel in (("reliable", lambda s: rel[s] >= args.min_rel), ("all", lambda s: True)):
                d = np.asarray([np.mean([table[A_ + k, s][0] - table[B_ + k, s][0] for k in seeds])
                                for s in gt if (g == "ALL" or group.get(s) == g) and sel(s)
                                and all((A_ + k, s) in table and (B_ + k, s) in table for k in seeds)])
                if len(d) > 1:
                    print(f"   {g:11s} {sel_name:8s} n={len(d):2d} {d.mean():+.2f} +- "
                          f"{d.std(ddof=1) / np.sqrt(len(d)):.2f}  ({A_} better in {(d > 0).sum()}/{len(d)})")
    pairs = {st: significant_pairs(mc) for st, mc in gt.items()}
    print(f"\n== pair accuracy (significant pairs, values averaged over updates >= {late})")
    for g in groups + ["ALL"]:
        ss = [st for st in gt if (g == "ALL" or group.get(st) == g) and pairs[st]]
        n = sum(len(pairs[st]) for st in ss)
        line = f"   {g:11s} {len(ss):2d} states {n:3d} pairs |"
        for v in variants:
            if not all((v, st) in table for st in ss):
                continue
            qs = {st: np.mean(by[v][st], 0) for st in ss}
            acc, (lo, hi) = pair_accuracy(qs, pairs, ss)
            line += f" {v} {acc:.2f} [{lo:.2f}, {hi:.2f}]"
        print(line)
    print("\n== per update: mean corr on reliable states")
    ups = sorted({u for v in curve for u in curve[v]})
    print("   update " + " ".join(f"{v:>9s}" for v in variants))
    for u in ups:
        print(f"   {u:6d} " + " ".join(f"{np.mean(curve[v][u]):+9.2f}" if curve[v].get(u) else f"{'':9s}"
                                    for v in variants))


if __name__ == "__main__":
    main()
