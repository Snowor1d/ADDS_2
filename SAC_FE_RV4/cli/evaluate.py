"""Evaluate a checkpoint (RV4 or RV3) and the baselines on the test suites.

    python3 -m cli.evaluate --checkpoint ~/Log_SAC_FE_RV4/best_validation.pth
    python3 -m cli.evaluate --checkpoint ~/SOTA_MODELS/300maps.pth --name rv3-300maps \
        --train-maps 1000-1299
    python3 -m cli.evaluate --baselines-only --suites test
    python3 -m cli.evaluate --checkpoint X --suites human --conditions policy_det policy_stoch

Default suites: EVAL_HEADLINE_SUITES (configs/evaluation.py). Default
conditions: EVAL_CONDITIONS.
Baseline episodes are cached, so they run once per suite.

Human results (ADDS_AS_HumanPlay.py in suite mode) are read from
~/<EVAL_RESULTS_DIR>/human/*.jsonl and joined to their scenarios as the
condition "human".

Leakage: scenarios whose map the checkpoint was trained on (its DATASET_MAPS,
or --train-maps for an RV3 file, which carries no record of them) and their
near-copies are reported separately and left out of the headline numbers.

Output: ~/<EVAL_RESULTS_DIR>/<name>/ with results.jsonl, per-scenario CSV,
summary.json, summary.md and plots; and a W&B run (job_type "eval").
"""

import argparse
import datetime as dt
import glob
import json
import os
import sys


def _parse_maps(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def _human_results(cfg, suite):
    from evaluation.runner import results_root
    ids = {s["id"]: s for s in suite["scenarios"]}
    src = {s.get("source_id"): s for s in suite["scenarios"] if s.get("source_id")}
    out = []
    for path in glob.glob(os.path.join(results_root(cfg), "human", "*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                sid = r.get("scenario_id")
                sc = ids.get(sid) or src.get(sid)
                if sc is None:
                    # a human run on a human-suite scenario counts for its source too
                    for s in suite["scenarios"]:
                        if s["id"] == r.get("source_id"):
                            sc = s
                if sc is None:
                    continue
                # Same cap as the other conditions: a run that did not finish
                # (or finished after the cap) counts as the cap.
                cap = float(cfg.EVAL_MAX_STEPS)
                if cfg.EVAL_MAX_STEPS_SCALE:
                    cap = round(cap * max(sc["width"], sc["height"]) / 100.0)
                t = float(r.get("evac_time_100", cap))
                ok = bool(r.get("success")) and t <= cap
                out.append({**r, "id": sc["id"], "map_key": sc["map_key"], "group": sc["group"],
                            "condition": "human", "evac_time_100": t if ok else cap,
                            "success": float(ok), "step_cap": cap})
    # several runs of one scenario: average them
    by = {}
    for r in out:
        by.setdefault(r["id"], []).append(r)
    merged = []
    for i, rs in by.items():
        m = dict(rs[0])
        for k in ("evac_time_100", "success", "evac_time_80", "total_lifetime"):
            vals = [x[k] for x in rs if k in x]
            if vals:
                m[k] = sum(vals) / len(vals)
        m["n_runs"] = len(rs)
        merged.append(m)
    return merged


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--name")
    ap.add_argument("--suites", nargs="*")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit", type=int, help="first N scenarios per suite (smoke tests)")
    ap.add_argument("--train-maps", help="maps the checkpoint trained on, e.g. 1000-1299,105")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args(argv)

    from configs import cfg
    from evaluation import report, stats
    from evaluation.runner import BASELINES, POLICY_CONDITIONS, results_root, run_suite, sim_version
    from evaluation.suites import LeakageIndex, load_suite, raster

    conds = list(args.conditions or cfg.EVAL_CONDITIONS)
    if args.baselines_only:
        conds = [c for c in conds if c in BASELINES]
    elif not args.checkpoint and any(c in POLICY_CONDITIONS for c in conds):
        raise SystemExit("--checkpoint is required unless --baselines-only")
    suites = list(args.suites or cfg.EVAL_HEADLINE_SUITES)
    name = args.name or (os.path.splitext(os.path.basename(args.checkpoint))[0]
                         if args.checkpoint else "baselines")
    name = f"{name}-{dt.datetime.now().strftime('%m%d-%H%M')}"
    out_dir = os.path.join(results_root(cfg), name)
    os.makedirs(out_dir, exist_ok=True)

    meta = {"name": name, "checkpoint": args.checkpoint, "conditions": conds,
            "sim_version": sim_version(cfg), "created": dt.datetime.now().isoformat(timespec="seconds")}
    train_maps = None
    if args.checkpoint:
        from learn.policy_io import load_policy
        _, pcfg, pmeta = load_policy(args.checkpoint)
        meta.update({"policy_format": pmeta.get("format"), "policy_encoder": pcfg.NET_ENCODER,
                     "policy_episode": pmeta.get("global_episode")})
        tm = pmeta.get("train_maps") or {}
        train_maps = tm.get("maps")
    if args.train_maps:
        train_maps = _parse_maps(args.train_maps)
    meta["train_maps"] = (f"{len(train_maps)} maps" if train_maps else "none recorded")

    leak = LeakageIndex(sorted(set(train_maps)), cfg) if train_maps else None
    dtime = float(cfg.AGENT_TIME_STEP)
    summaries, all_res, images = {}, {}, []
    policy_conds = [c for c in conds if c in POLICY_CONDITIONS]
    for sname in suites:
        suite = load_suite(sname, cfg)
        meta[f"suite_{sname}_digest"] = suite["digest"]
        res = run_suite(cfg, suite, conds, policy_path=args.checkpoint, workers=args.workers,
                        limit=args.limit)
        res += _human_results(cfg, suite)
        if args.limit:
            keep = {s["id"] for s in suite["scenarios"][: args.limit]}
            res = [r for r in res if r["id"] in keep]
        # leakage against what this checkpoint trained on
        dirty = set()
        if leak is not None:
            for sc in suite["scenarios"]:
                if sc.get("map_num") in set(train_maps):
                    dirty.add(sc["id"])
                    continue
                if sc.get("level"):
                    g = sc["level"]
                    d, _ = leak.nearest(raster(g["obstacles"], g["exits"], g["width"], g["height"]))
                else:
                    import sim_maps
                    g = sim_maps.map_geometry(sc["map_num"], cfg)
                    d, _ = leak.nearest(raster(g["obstacles"], g["exits"], g["width"], g["height"]))
                if d < float(cfg.LEAKAGE_PIXEL_THRESHOLD):
                    dirty.add(sc["id"])
        if dirty:
            print(f"[eval] {sname}: {len(dirty)} scenarios use maps this checkpoint trained on; "
                  "excluded from the summary", flush=True)
        clean = [r for r in res if r["id"] not in dirty]
        if any(c.startswith("shuttle") for c in conds):
            clean += stats.best_of(clean, [c for c in ("shuttle_random", "shuttle_nearest") if c in conds],
                                   "best_baseline")
        refs = [c for c in ("no_robot", "exit_robot", "shuttle_random", "shuttle_nearest",
                            "best_baseline", "human")
                if any(r["condition"] == c for r in clean)]
        S = report.build_summary(clean, policy_conds, refs, dtime, int(cfg.EVAL_BOOTSTRAP_SAMPLES),
                                 float(cfg.EVAL_CI))
        S["n_excluded_for_leakage"] = len(dirty)
        summaries[sname] = S
        all_res[sname] = res
        with open(os.path.join(out_dir, f"{sname}_results.jsonl"), "w", encoding="utf-8") as fh:
            for r in res:
                fh.write(json.dumps({**r, "leaked": r["id"] in dirty}) + "\n")
        report.write_csv(os.path.join(out_dir, f"{sname}_scenarios.csv"), res)
        images += report.plots(out_dir, sname, clean, S, dtime,
                               policy_conds[0] if policy_conds else None)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "suites": summaries}, fh, indent=1)
    md = report.markdown(name, summaries, dtime, meta)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)
    if not args.no_wandb:
        url = report.log_wandb(cfg, name, meta, summaries, all_res, images, dtime)
        if url:
            print(f"[eval] W&B: {url}")
    print(f"[eval] results in {out_dir}")


if __name__ == "__main__":
    main()
