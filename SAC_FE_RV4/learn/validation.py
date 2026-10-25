"""Periodic validation in its own process.

The trainer writes a policy snapshot and starts `validation_main` on it; the
process evaluates the VALIDATION_SUITE scenarios under the policy and the two
shuttle baselines (cached after the first time) and writes a JSON summary
the trainer logs to W&B and uses for best_validation.pth.
"""

from __future__ import annotations

import json
import os
import time


def summarize_validation(cfg, results, policy_cond: str) -> dict:
    from evaluation import stats
    out = {"time": time.time(), "conditions": {}, "paired": {}}
    for cond in sorted({r["condition"] for r in results if "error" not in r}):
        out["conditions"][cond] = stats.summarize(results, cond, B=1000)
    for ref in ("shuttle_nearest", "shuttle_random"):
        if any(r.get("condition") == ref for r in results):
            out["paired"][f"{policy_cond}_vs_{ref}"] = stats.paired(results, policy_cond, ref, B=1000)
    out["n_errors"] = sum(1 for r in results if "error" in r)
    return out


def validation_main(snapshot: str, out_path: str):
    # Lead a process group so the trainer can stop this process together
    # with its evaluation pool.
    os.setpgrp()
    from configs import cfg
    from evaluation.runner import run_suite
    from evaluation.suites import load_suite
    suite = load_suite(cfg.VALIDATION_SUITE, cfg)
    pc = "policy_det" if cfg.VALIDATION_POLICY_MODE == "deterministic" else "policy_stoch"
    res = run_suite(cfg, suite, [pc, "shuttle_nearest", "shuttle_random"], policy_path=snapshot,
                    workers=int(cfg.VALIDATION_WORKERS), log=lambda m: None)
    summary = summarize_validation(cfg, res, pc)
    summary["policy_condition"] = pc
    summary["snapshot"] = snapshot
    if cfg.VALIDATION_VIDEO:
        try:
            from evaluation.video import record_policy_episode
            vid = os.path.splitext(out_path)[0] + ".mp4"
            r = record_policy_episode(cfg, suite["scenarios"][0], snapshot, vid,
                                      deterministic=(pc == "policy_det"))
            if r is not None:
                summary["video"] = vid
                summary["video_fps"] = 20
        except Exception as e:
            summary["video_error"] = f"{type(e).__name__}: {e}"
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    os.replace(tmp, out_path)


def flatten_for_logging(summary: dict, prefix: str = "eval/val") -> dict:
    out = {}
    for cond, s in summary.get("conditions", {}).items():
        for k in ("mean", "median", "iqm", "success_rate", "ci_low", "ci_high"):
            if k in s:
                out[f"{prefix}/{cond}/evac_time_100_{k}" if k not in ("success_rate",)
                    else f"{prefix}/{cond}/success_rate"] = s[k]
    for name, p in summary.get("paired", {}).items():
        for k in ("mean_diff", "win_rate", "ratio_gmean"):
            if k in p:
                out[f"{prefix}/{name}/{k}"] = p[k]
    pc = summary.get("policy_condition")
    if pc and pc in summary.get("conditions", {}):
        out["key/val_evac_time_100"] = summary["conditions"][pc]["mean"]
        out["key/val_success_rate"] = summary["conditions"][pc]["success_rate"]
        p = summary.get("paired", {}).get(f"{pc}_vs_shuttle_nearest")
        if p:
            out["key/val_win_rate_vs_shuttle_nearest"] = p["win_rate"]
    return out
