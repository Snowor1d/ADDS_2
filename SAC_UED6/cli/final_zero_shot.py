"""The final outdoor zero-shot evaluation (docs/outdoor_madrl_redesign.md §7).

Two steps, deliberately separate:

    python3 -m cli.final_zero_shot check
        Draws every pre-registered hazard seed on the pre-registered crop,
        applies the pre-evaluation checks (walkable ground inside, a walking
        route to safety from all of it, effective density) and writes the
        accepted and rejected seeds with their reasons. Needs no policy, so it
        can and should be run before any policy is evaluated.

    python3 -m cli.final_zero_shot run --checkpoint PATH
        Runs the accepted hazard seeds x crowd seeds x robot counts, each
        paired with the zero-command signal-off control. The checkpoint must
        be the model already selected on validation; its SHA-256 is recorded,
        and a second run with a different checkpoint is refused unless
        --register-new-model is given, which is itself recorded, so the final
        map cannot quietly become a model-selection set.

Results go to <LOG_DIR>/final_zero_shot/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time


def _out_dir(cfg) -> str:
    d = os.path.join(os.path.expanduser("~"), cfg.LOG_DIR, "final_zero_shot")
    os.makedirs(d, exist_ok=True)
    return d


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(cfg) -> dict:
    from learn.zero_shot import draw_final_hazard, final_base_level

    base, crowd = final_base_level(cfg)
    accepted, checks = [], []
    for seed in cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS:
        lv, rec = draw_final_hazard(base, int(seed), cfg)
        checks.append(rec)
        if lv is not None:
            accepted.append(int(seed))
    report = {
        "site": cfg.FINAL_ZERO_SHOT_SITE, "size_m": cfg.FINAL_ZERO_SHOT_SIZE_M,
        "crowd": crowd, "hazard_seeds": list(cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS),
        "crowd_seeds": list(cfg.FINAL_ZERO_SHOT_CROWD_SEEDS),
        "robot_counts": list(cfg.FINAL_ZERO_SHOT_ROBOT_COUNTS),
        "rejection_rules": {
            "min_walkable_inside_m2": cfg.FINAL_ZERO_SHOT_MIN_WALKABLE_M2,
            "route_to_safety": "every walkable navmesh triangle inside",
            "density_band": list(cfg.FINAL_ZERO_SHOT_DENSITY_BAND)},
        "checks": checks, "accepted_hazard_seeds": accepted,
        "schema_versions": cfg.schema_versions(),
        "config_fingerprint": cfg.fingerprint, "time": time.time(),
    }
    path = os.path.join(_out_dir(cfg), "preregistration_check.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"{len(accepted)} of {len(checks)} hazard seeds accepted -> {path}")
    for rec in checks:
        print(f"  seed {rec['seed']}: {rec['shape']:6s} "
              f"area {rec['area_fraction']:.3f} walkable-in "
              f"{rec['walkable_inside_m2']:.0f} m2 density "
              f"{rec['effective_density']:.4f} "
              f"{'REJECTED ' + ','.join(rec['rejected']) if rec['rejected'] else 'ok'}")
    return report


def run(cfg, checkpoint: str, register_new: bool, max_steps=None) -> None:
    from learn.sac import SACAgent
    from learn.zero_shot import (append_jsonl, draw_final_hazard,
                                 final_base_level, paired_records, summarise)

    out = _out_dir(cfg)
    lock_path = os.path.join(out, "model_lock.json")
    digest = _sha256(checkpoint)
    if os.path.exists(lock_path):
        with open(lock_path) as fh:
            lock = json.load(fh)
        if lock["sha256"] != digest and not register_new:
            raise SystemExit(
                f"the final evaluation was already run with {lock['checkpoint']} "
                f"(sha256 {lock['sha256'][:12]}). Running another model on the "
                "final map would make it a selection set. Use "
                "--register-new-model only for a deliberate, reported re-run.")
    history = []
    if os.path.exists(lock_path):
        with open(lock_path) as fh:
            history = json.load(fh).get("history", [])
    history.append({"checkpoint": os.path.abspath(checkpoint),
                    "sha256": digest, "time": time.time(),
                    "registered_new_model": bool(register_new)})
    with open(lock_path, "w") as fh:
        json.dump({"checkpoint": os.path.abspath(checkpoint), "sha256": digest,
                   "history": history}, fh, indent=2)

    report = check(cfg)
    if not report["accepted_hazard_seeds"]:
        raise SystemExit("no pre-registered hazard seed passed the checks")
    agent = SACAgent(cfg, device="cpu")
    agent.load(checkpoint, policy_only=True)
    agent.policy.eval()
    base, crowd = final_base_level(cfg)
    scenarios = []
    for seed in report["accepted_hazard_seeds"]:
        lv, _ = draw_final_hazard(base, seed, cfg)
        scenarios.append((f"final_{cfg.FINAL_ZERO_SHOT_SITE}_"
                          f"{cfg.FINAL_ZERO_SHOT_SIZE_M}m_h{seed}", lv))
    records = paired_records(
        agent, scenarios, cfg, episode=-1,
        seeds_for=lambda name: tuple(cfg.FINAL_ZERO_SHOT_CROWD_SEEDS),
        robot_counts=cfg.FINAL_ZERO_SHOT_ROBOT_COUNTS, max_steps=max_steps)
    for r in records:
        r.update({"checkpoint_sha256": digest,
                  "low_density": crowd["low_density"]})
    append_jsonl(os.path.join(out, "final_metrics.jsonl"), records)
    summary = summarise(records, "final")
    with open(os.path.join(out, "final_summary.json"), "w") as fh:
        json.dump({"summary": summary, "crowd": crowd,
                   "checkpoint_sha256": digest,
                   "schema_versions": cfg.schema_versions()}, fh, indent=2)
    for k in sorted(summary):
        if "/paired/" in k or k.endswith("held_clear_success"):
            print(f"{k}: {summary[k]:.4f}")


def main():
    from configs import resolve_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--register-new-model", action="store_true")
    r.add_argument("--max-steps", type=int, default=None,
                   help="shorter horizon for a smoke test; never for results")
    args = ap.parse_args()
    cfg = resolve_config()
    if args.cmd == "check":
        check(cfg)
    else:
        run(cfg, args.checkpoint, args.register_new_model, args.max_steps)


if __name__ == "__main__":
    main()
