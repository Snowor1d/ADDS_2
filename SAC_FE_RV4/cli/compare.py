"""Compare policies from several evaluation runs on the same scenarios.

    python3 -m cli.compare ~/RV4_eval/accel-1025-1200 ~/RV4_eval/dr-1025-1300 \
        --labels accel dr --suites test

Each run directory is the output of cli.evaluate. Its policy condition
(policy_det by default, --condition to change) becomes one column; the
baselines are taken from the first run. Every pair of policies, and every
policy against the best baseline and against humans when present, is
compared with the paired cluster bootstrap of evaluation/stats.py, on the
scenarios both have (leaked scenarios are dropped per run).

Several training seeds of one method: give their runs the same label; their
per-scenario times are averaged into one column before comparing.
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def _load(run_dir, suite):
    path = os.path.join(run_dir, f"{suite}_results.jsonl")
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "error" not in r and not r.get("leaked"):
                out.append(r)
    return out


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--suites", nargs="*", default=None)
    ap.add_argument("--condition", default="policy_det")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    from configs import cfg
    from evaluation import stats
    if args.suites is None:
        args.suites = list(cfg.EVAL_HEADLINE_SUITES)
    from evaluation.report import LABELS

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match the runs")
    dt = float(cfg.AGENT_TIME_STEP)
    B = int(cfg.EVAL_BOOTSTRAP_SAMPLES)
    lines = [f"# Policy comparison ({args.condition})", ""]
    for suite in args.suites:
        merged = []
        per_label = defaultdict(lambda: defaultdict(list))
        meta = {}
        for i, (run, lab) in enumerate(zip(args.runs, labels)):
            rs = _load(run, suite)
            for r in rs:
                if r["condition"] == args.condition:
                    per_label[lab][r["id"]].append(r)
                    meta[r["id"]] = (r["map_key"], r["group"])
                elif i == 0 and r["condition"] != args.condition:
                    merged.append(r)
        for lab, by_id in per_label.items():
            for i, rs in by_id.items():
                merged.append({"id": i, "map_key": meta[i][0], "group": meta[i][1],
                               "condition": lab,
                               "evac_time_100": sum(x["evac_time_100"] for x in rs) / len(rs),
                               "success": sum(x["success"] for x in rs) / len(rs)})
        base = [c for c in ("shuttle_random", "shuttle_nearest") if any(r["condition"] == c for r in merged)]
        if base:
            merged += stats.best_of(merged, base, "best_baseline")
        conds = list(dict.fromkeys(list(per_label) + [c for c in ("best_baseline", "shuttle_random",
                                                                    "shuttle_nearest", "human")
                                                       if any(r["condition"] == c for r in merged)]))
        lines += [f"## {suite}", "", "| | n | mean (s) | 95% CI (s) | median (s) | success |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for c in conds:
            s = stats.summarize(merged, c, B=B)
            if s:
                lines.append(f"| {LABELS.get(c, c)} | {s['n']:.0f} | {s['mean'] * dt:.1f} | "
                             f"{s['ci_low'] * dt:.1f}-{s['ci_high'] * dt:.1f} | {s['median'] * dt:.1f} | "
                             f"{s['success_rate'] * 100:.1f}% |")
        lines += ["", "| A vs B | n | mean diff (s) | 95% CI (s) | A faster on | time ratio |",
                  "| --- | --- | --- | --- | --- | --- |"]
        labs = list(per_label)
        pairs = [(a, b) for i, a in enumerate(labs) for b in labs[i + 1:]]
        pairs += [(a, b) for a in labs for b in ("best_baseline", "human") if b in conds]
        for a, b in pairs:
            p = stats.paired(merged, a, b, B=B)
            if p:
                lines.append(f"| {LABELS.get(a, a)} vs {LABELS.get(b, b)} | {p['n']:.0f} | "
                             f"{p['mean_diff'] * dt:+.1f} | {p['diff_ci_low'] * dt:+.1f} to "
                             f"{p['diff_ci_high'] * dt:+.1f} | {p['win_rate'] * 100:.0f}% | "
                             f"{p['ratio_gmean']:.3f} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
