"""Reproduce this review from saved scalar histories, without network access."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = ("z44h9d77", "4pr4wxci", "vsa06hsg")
LABELS = ("A20 / P0", "A0 / P150", "A20 / P150")
PREFIX = "eval/validation/200m/robots_3/"
CONDITIONS = ("off_zero_command", "shuttle", "policy_deterministic")
METRICS = ("hazard_person_steps", "activity_dwell_hazard_person_steps",
           "reentries_after_clear", "robot_inside_share", "robot_depth_m",
           "robot_speed_mean", "followers_per_robot", "held_clear_success", "n")


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def average(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.mean(values)) if values else None


def deduplicate(rows, marker, identity):
    result = {}
    duplicates = 0
    for row in rows:
        if marker not in row:
            continue
        key = tuple(row[k] for k in identity)
        if key in result:
            # Upload replay can repeat identical data at a different W&B step.
            prior = result[key]
            for field in prior.keys() & row.keys():
                if not field.startswith("_"):
                    assert prior[field] == row[field], (key, field)
            duplicates += 1
        result[key] = row
    return sorted(result.values(), key=lambda r: tuple(r[k] for k in identity)), duplicates


def main():
    inventory = {r["id"]: r for r in load_inventory()}
    output, histories = {}, {}
    for rid, label in zip(RUNS, LABELS):
        rows = load(ROOT / f"{rid}.jsonl.gz")
        episodes, ep_dups = deduplicate(rows, "episode/total_reward", ("global_episode",))
        evaluations, ev_dups = deduplicate(
            rows, PREFIX + "policy_deterministic/hazard_person_steps", ("global_episode",))
        learner, tr_dups = deduplicate(rows, "train/alpha", ("global_update", "global_episode"))
        histories[rid] = episodes, evaluations, learner
        item = dict(label=label, url=inventory[rid]["url"], state=inventory[rid]["state"],
                    last_episode=max(r["global_episode"] for r in rows),
                    duplicate_rows=dict(episode=ep_dups, validation=ev_dups, learner=tr_dups),
                    validation=[], training_bins=[])
        for r in evaluations:
            item["validation"].append(dict(
                episode=r["global_episode"], update=r["global_update"],
                paired_vs_off=r.get("eval/validation/paired/person_steps_reduction_vs_off"),
                paired_vs_shuttle=r.get("eval/validation/paired/policy_deterministic/person_steps_reduction_vs_shuttle"),
                **{c: {k: r.get(PREFIX + c + "/" + k) for k in METRICS} for c in CONDITIONS}))
        for low, high in ((1, 500), (501, 1000), (1001, 1500), (1501, 2000), (2001, 2500)):
            selected = [r for r in episodes if low <= r["global_episode"] <= high]
            item["training_bins"].append(dict(low=low, high=high, n=len(selected), **{
                k: average(selected, k) for k in (
                    "episode/total_reward", "episode/hazard_person_steps",
                    "episode/reward/collision", "episode/mode_share/guide",
                    "episode/robots/robot_inside_share")}))
        recent = [r for r in learner if 1500 <= r["global_episode"] <= 2000]
        item["learner_1500_2000"] = {k: average(recent, k) for k in (
            "train/alpha", "train/entropy", "train/target_entropy", "train/loss_q",
            "train/q_team", "train/target")}
        floor = inventory[rid]["config"]["ALPHA_MIN"]
        floor_rows = [r for r in learner if r["train/alpha"] <= floor * 1.00001]
        item["first_alpha_floor_episode"] = floor_rows[0]["global_episode"] if floor_rows else None
        item["stochastic_validation_logged"] = any(
            PREFIX + "policy_stochastic/hazard_person_steps" in r for r in rows)
        output[rid] = item

    # Fixed baselines must match across all runs and checkpoints.
    for c in CONDITIONS[:2]:
        for k in METRICS:
            values = [v[c][k] for rid in RUNS for v in output[rid]["validation"]
                      if v[c][k] is not None]
            assert np.allclose(values, values[0]), (c, k)

    records = load(ROOT / "progress_validation_records.jsonl")
    ep = max(r["episode"] for r in records)
    latest = [r for r in records if r["episode"] == ep]
    by_key = {(r["scenario"], r["seed"], r["condition"]): r for r in latest}
    validation = next(v for v in output["4pr4wxci"]["validation"] if v["episode"] == ep)
    pairs = []
    for r in latest:
        if r["condition"] != "policy_deterministic":
            continue
        off = by_key[r["scenario"], r["seed"], "off_zero_command"]["hazard_person_steps"]
        shuttle = by_key[r["scenario"], r["seed"], "shuttle"]["hazard_person_steps"]
        pairs.append(dict(scenario=r["scenario"], seed=r["seed"], off=off, shuttle=shuttle,
                          policy=r["hazard_person_steps"],
                          vs_off=1-r["hazard_person_steps"]/off,
                          vs_shuttle=1-r["hazard_person_steps"]/shuttle))
    assert len(pairs) == 9
    assert np.isclose(np.mean([p["policy"] for p in pairs]),
                      validation["policy_deterministic"]["hazard_person_steps"])
    assert np.isclose(np.mean([p["vs_off"] for p in pairs]), validation["paired_vs_off"])
    output["progress_latest_pairs"] = pairs
    output["progress_latest_condition_means"] = {
        c: {k: average([r for r in latest if r["condition"] == c], k)
            for k in METRICS + ("activity_cancelled_robot_follow", "mode_share_guide")}
        for c in CONDITIONS}
    output["matched_episode_2000"] = {
        rid: next(v for v in output[rid]["validation"] if v["episode"] == 2000)
        for rid in RUNS}
    (ROOT / "analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    plot(histories)
    print(json.dumps({"matched_2000": output["matched_episode_2000"],
                      "progress_pairs": pairs,
                      "duplicates": {rid: output[rid]["duplicate_rows"] for rid in RUNS}}, indent=2))


def load_inventory():
    return json.loads((ROOT / "inventory.json").read_text())


def plot(histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for rid, label in zip(RUNS, LABELS):
        episodes, evaluations, _ = histories[rid]
        x = [r["global_episode"] for r in evaluations]
        for ax, metric in ((axes[0, 0], "hazard_person_steps"),
                           (axes[0, 1], "activity_dwell_hazard_person_steps"),
                           (axes[1, 0], "robot_inside_share")):
            ax.plot(x, [r[PREFIX + "policy_deterministic/" + metric] for r in evaluations],
                    marker="o", markersize=3, label=label)
        bins = range(1, 2501, 100)
        y = [average([r for r in episodes if low <= r["global_episode"] < low+100],
                     "episode/total_reward") for low in bins]
        axes[1, 1].plot([low+49.5 for low in bins], y, label=label)
    reference = histories[RUNS[0]][1][-1]
    for ax, metric in ((axes[0, 0], "hazard_person_steps"),
                       (axes[0, 1], "activity_dwell_hazard_person_steps"),
                       (axes[1, 0], "robot_inside_share")):
        for condition, color in (("off_zero_command", "black"), ("shuttle", "gray")):
            ax.axhline(reference[PREFIX + condition + "/" + metric], color=color,
                       linestyle="--", alpha=.7, label=condition)
    axes[0, 0].set(title="Deterministic validation: total hazard exposure", ylabel="Person-steps")
    axes[0, 1].set(title="Validation: activity dwell hazard exposure", ylabel="Person-steps")
    axes[1, 0].set(title="Validation: robot time inside hazard", ylabel="Fraction")
    axes[1, 1].set(title="Training task reward (100-episode means)", ylabel="Reward (higher is better)")
    for ax in axes.flat:
        ax.set_xlabel("Episode")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("W&B snapshot 2026-10-20 | same base reward | 9 validation tasks")
    fig.savefig(ROOT / "learning_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
