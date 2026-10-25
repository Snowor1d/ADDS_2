"""Tune the shuttle baselines' approach and escort speeds.

    python3 -m cli.tune_baselines                       # both shuttles
    python3 -m cli.tune_baselines --only shuttle_nearest --workers 8

Objective: the mean 100% evacuation time on the TUNE_SUITE scenarios (an
unfinished episode counts as the step cap), lower is better. Search:
successive halving over TUNE_APPROACH_SPEEDS x TUNE_ESCORT_SPEEDS. Rung r
evaluates the surviving speed pairs on the first TUNE_FIRST_RUNG_SCENARIOS *
TUNE_ETA**r scenarios (a fixed shuffled order, the same for every pair, so
the comparison is paired), keeps the best 1/TUNE_ETA, and ends with one pair
evaluated on the whole suite. Every episode is cached, so an interrupted
run resumes where it stopped.

The tune suite has its own generator seeds, apart from validation and test,
so the baselines are tuned the way the policy's hyperparameters are: on data
the final comparison does not use.

Writes SHUTTLE_TUNED_FILE; ShuttleController reads it from then on.
"""

import argparse
import datetime as dt
import itertools
import json
import math
import os
import random
import sys


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["shuttle_random", "shuttle_nearest"])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    from configs import cfg
    from evaluation.baselines import ROOT, tuned_key
    from evaluation.runner import run_many, sim_version
    from evaluation.suites import load_suite

    suite = load_suite(cfg.TUNE_SUITE, cfg)
    order = list(suite["scenarios"])
    random.Random(0).shuffle(order)
    grid = [dict(approach=float(a), escort=float(e))
            for a, e in itertools.product(cfg.TUNE_APPROACH_SPEEDS, cfg.TUNE_ESCORT_SPEEDS)]
    eta = int(cfg.TUNE_ETA)
    path = cfg.SHUTTLE_TUNED_FILE if os.path.isabs(cfg.SHUTTLE_TUNED_FILE) else \
        os.path.join(ROOT, cfg.SHUTTLE_TUNED_FILE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)
    except (OSError, ValueError):
        out = {}
    for cond in args.only:
        alive = list(grid)
        n = int(cfg.TUNE_FIRST_RUNG_SCENARIOS)
        rungs = []
        table = {}
        while True:
            n = min(n, len(order))
            scen = order[:n]
            res = run_many(cfg, suite, [(cond, p) for p in alive], scen,
                           workers=args.workers, log=print, tag=f" {cond} rung {len(rungs)}")
            scored = []
            for p in alive:
                rs = [r for r in res[(cond, json.dumps(p, sort_keys=True))] if "error" not in r]
                ids = {r["id"] for r in rs}
                if len(ids) < len(scen):
                    raise SystemExit(f"{cond} {p}: only {len(ids)}/{len(scen)} episodes succeeded")
                mean = sum(r["evac_time_100"] for r in rs) / len(rs)
                succ = sum(r["success"] for r in rs) / len(rs)
                scored.append((mean, p, succ))
                table[json.dumps(p, sort_keys=True)] = dict(**p, n=len(rs), mean=mean, success_rate=succ)
            scored.sort(key=lambda x: x[0])
            rungs.append({"n_scenarios": n, "n_configs": len(alive),
                          "ranking": [dict(**p, mean=m, success_rate=s) for m, p, s in scored]})
            print(f"[tune] {cond} rung {len(rungs) - 1}: {len(alive)} configs x {n} scenarios; "
                  f"best {scored[0][1]} mean {scored[0][0]:.1f} steps", flush=True)
            if len(alive) == 1 and n >= len(order):
                break
            keep = max(1, int(math.ceil(len(alive) / eta)))
            if n >= len(order):
                keep = 1
            alive = [p for _, p, _ in scored[:keep]]
            n = n * eta
        best_mean, best, best_s = scored[0]
        key = tuned_key(cfg, cond.split("_", 1)[1])
        out[key] = {"best": dict(**best, mean=best_mean, success_rate=best_s, n=len(order)),
                     "rungs": rungs, "all": sorted(table.values(), key=lambda r: (r["n"] * -1, r["mean"]))}
        out.setdefault("meta", {})[key] = {"speed_limit": cfg.ROBOT_SPEED_NORM_LIMIT, "suite": suite["name"], "digest": suite["digest"], "sim_version": sim_version(cfg),
                       "objective": "mean evac_time_100 (steps, cap if unfinished)",
                       "updated": dt.datetime.now().isoformat(timespec="seconds"),
                       "grid": {"approach": list(cfg.TUNE_APPROACH_SPEEDS),
                                "escort": list(cfg.TUNE_ESCORT_SPEEDS)}, "eta": eta}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        print(f"[tune] {cond}: best approach {best['approach']} m/s, escort {best['escort']} m/s, "
              f"mean {best_mean:.1f} steps ({best_mean * cfg.AGENT_TIME_STEP:.0f} s) -> {path}",
              flush=True)


if __name__ == "__main__":
    main()
