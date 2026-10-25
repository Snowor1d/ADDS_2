"""Tables, plots and W&B upload for an evaluation run."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from evaluation import stats

LABELS = {
    "no_robot": "No robot", "exit_robot": "Direct-to-exit robot",
    "shuttle_random": "Shuttle (random person)", "shuttle_nearest": "Shuttle (nearest person)",
    "best_baseline": "Best baseline per scenario", "policy": "Policy", "policy_det": "Policy (mean)",
    "policy_stoch": "Policy (sampled)", "human": "Human",
}
ORDER = ("no_robot", "exit_robot", "shuttle_random", "shuttle_nearest", "best_baseline",
         "human", "policy_stoch", "policy_det", "policy")


def _order(conds):
    return [c for c in ORDER if c in conds] + sorted(c for c in conds if c not in ORDER)


def build_summary(results: List[dict], policy_conds: Sequence[str], refs: Sequence[str],
                  dt: float, B: int, ci: float) -> dict:
    conds = _order({r["condition"] for r in results if "error" not in r})
    out = {"conditions": {}, "paired": {}, "groups": {}}
    for c in conds:
        s = stats.summarize(results, c, B=B, ci=ci)
        if s:
            s["mean_s"] = s["mean"] * dt
            out["conditions"][c] = s
    for pc in policy_conds:
        for ref in refs:
            if pc in out["conditions"] and ref in out["conditions"] and pc != ref:
                out["paired"][f"{pc}__vs__{ref}"] = stats.paired(results, pc, ref, B=B, ci=ci)
    for g, rs in sorted(stats.by_group(results).items()):
        out["groups"][g] = {c: stats.summarize(rs, c, B=max(200, B // 10), ci=ci)
                            for c in conds if any(r["condition"] == c for r in rs)}
    return out


def markdown(name: str, suite_summaries: Dict[str, dict], dt: float, meta: dict) -> str:
    L = [f"# Evaluation: {name}", ""]
    for k, v in meta.items():
        L.append(f"- {k}: {v}")
    L.append("")
    L.append(f"Evacuation time = steps until everyone is out (x {dt} s per step); unfinished "
             "episodes count as the step cap. Intervals: 95% cluster bootstrap over maps.")
    for suite, S in suite_summaries.items():
        L += ["", f"## {suite}", "",
              "| condition | n | maps | mean (s) | 95% CI (s) | median (s) | IQM (s) | success |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c, s in S["conditions"].items():
            L.append(f"| {LABELS.get(c, c)} | {s['n']:.0f} | {s['n_maps']:.0f} | {s['mean'] * dt:.1f} | "
                     f"{s['ci_low'] * dt:.1f}-{s['ci_high'] * dt:.1f} | {s['median'] * dt:.1f} | "
                     f"{s['iqm'] * dt:.1f} | {s['success_rate'] * 100:.1f}% |")
        if S["paired"]:
            L += ["", "| comparison | mean diff (s) | 95% CI (s) | faster on | time ratio (gmean) |",
                  "| --- | --- | --- | --- | --- |"]
            for k, p in S["paired"].items():
                a, b = k.split("__vs__")
                L.append(f"| {LABELS.get(a, a)} vs {LABELS.get(b, b)} | {p['mean_diff'] * dt:+.1f} | "
                         f"{p['diff_ci_low'] * dt:+.1f} to {p['diff_ci_high'] * dt:+.1f} | "
                         f"{p['win_rate'] * 100:.0f}% ({p['win_ci_low'] * 100:.0f}-{p['win_ci_high'] * 100:.0f}) | "
                         f"{p['ratio_gmean']:.3f} ({p['ratio_ci_low']:.3f}-{p['ratio_ci_high']:.3f}) |")
        if len(S["groups"]) > 1:
            conds = list(S["conditions"])
            L += ["", "Mean (s) by map group:", "",
                  "| group | " + " | ".join(LABELS.get(c, c) for c in conds) + " |",
                  "| --- |" + " --- |" * len(conds)]
            for g, gs in S["groups"].items():
                cells = [f"{gs[c]['mean'] * dt:.1f}" if gs.get(c) else "-" for c in conds]
                L.append(f"| {g} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def write_csv(path: str, results: List[dict]) -> None:
    keys = ["id", "map_key", "group", "condition", "evac_time_100", "evac_time_80", "success",
            "steps", "step_cap", "total_lifetime", "freeflow_steps", "robot_path_m",
            "robot_collision_steps", "trips", "approach", "escort", "duration_s", "error"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)


def plots(out_dir: str, suite: str, results: List[dict], summary: dict, dt: float,
          policy_cond: Optional[str]) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    conds = list(summary["conditions"])
    if conds:
        fig, ax = plt.subplots(figsize=(1.6 * len(conds) + 2, 4))
        m = [summary["conditions"][c]["mean"] * dt for c in conds]
        lo = [m[i] - summary["conditions"][c]["ci_low"] * dt for i, c in enumerate(conds)]
        hi = [summary["conditions"][c]["ci_high"] * dt - m[i] for i, c in enumerate(conds)]
        ax.bar(range(len(conds)), m, yerr=[lo, hi], capsize=4, color="#7389AF", edgecolor="black")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([LABELS.get(c, c) for c in conds], rotation=20, ha="right")
        ax.set_ylabel("100% evacuation time (s)")
        ax.set_title(f"{suite}: mean and 95% CI")
        fig.tight_layout()
        p = os.path.join(out_dir, f"{suite}_bars.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(p)
    if policy_cond and policy_cond in conds:
        ref = "best_baseline" if "best_baseline" in conds else ("shuttle_nearest" if "shuttle_nearest" in conds else None)
        if ref:
            a = stats._by(results, policy_cond)
            b = stats._by(results, ref)
            ids = sorted(set(a) & set(b))
            if ids:
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                x = np.array([b[i] for i in ids]) * dt
                y = np.array([a[i] for i in ids]) * dt
                ax.scatter(x, y, s=10, alpha=0.6)
                top = max(x.max(), y.max()) * 1.05
                ax.plot([0, top], [0, top], "k--", lw=1)
                ax.set_xlabel(f"{LABELS.get(ref, ref)} (s)")
                ax.set_ylabel(f"{LABELS.get(policy_cond, policy_cond)} (s)")
                ax.set_title(f"{suite}: per scenario (below the line = policy faster)")
                fig.tight_layout()
                p = os.path.join(out_dir, f"{suite}_scatter.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                files.append(p)
    return files


def log_wandb(cfg, name: str, meta: dict, suite_summaries: Dict[str, dict],
              all_results: Dict[str, List[dict]], images: List[str], dt: float) -> Optional[str]:
    if cfg.WANDB_MODE == "disabled" or not cfg.EVAL_WANDB:
        return None
    try:
        import wandb
        run = wandb.init(project=cfg.WANDB_PROJECT, entity=cfg.WANDB_ENTITY,
                         group=cfg.WANDB_GROUP or cfg.EXPERIMENT_ID, job_type="eval",
                         name=f"eval-{name}", mode=cfg.WANDB_MODE, config=meta,
                         dir=os.path.join(os.path.expanduser("~"), cfg.EVAL_RESULTS_DIR))
        flat = {}
        for suite, S in suite_summaries.items():
            for c, s in S["conditions"].items():
                flat[f"eval/{suite}/{c}/evac_time_100_mean_s"] = s["mean"] * dt
                flat[f"eval/{suite}/{c}/evac_time_100_median_s"] = s["median"] * dt
                flat[f"eval/{suite}/{c}/success_rate"] = s["success_rate"]
            for k, p in S["paired"].items():
                flat[f"eval/{suite}/{k}/mean_diff_s"] = p["mean_diff"] * dt
                flat[f"eval/{suite}/{k}/win_rate"] = p["win_rate"]
        run.log(flat)
        for k, v in flat.items():
            run.summary[k] = v
        for suite, rs in all_results.items():
            cols = ["id", "map_key", "group", "condition", "evac_time_s", "success"]
            t = wandb.Table(columns=cols, data=[[r["id"], r["map_key"], r["group"], r["condition"],
                                                 r["evac_time_100"] * dt, r["success"]]
                                                for r in rs if "error" not in r])
            run.log({f"eval/{suite}/scenarios": t})
        for p in images:
            run.log({f"eval/plots/{os.path.basename(p)[:-4]}": wandb.Image(p)})
        url = run.url
        run.finish()
        return url
    except Exception as e:
        print(f"[eval] W&B upload failed ({type(e).__name__}: {e}); results are on disk")
        return None
