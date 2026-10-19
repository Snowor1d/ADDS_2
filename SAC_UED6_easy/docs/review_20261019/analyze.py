"""Analyze downloaded scalar W&B histories; never connects to W&B."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PREFIX = "eval/validation/200m/robots_3/"
RUN_IDS = ("vsa06hsg", "d774mrrb", "cx8bbli5", "fi2kcteu")


def numeric(x):
    return isinstance(x, (int, float)) and np.isfinite(x)


def mean(rows, key):
    values = [r[key] for r in rows if numeric(r.get(key))]
    return float(np.mean(values)) if values else None


def load(path):
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def analyze():
    inventory = {r["id"]: r for r in json.loads((ROOT / "inventory.json").read_text())}
    output = {}
    histories = {}
    for rid in RUN_IDS:
        rows = load(ROOT / f"{rid}.jsonl")
        episodes = [r for r in rows if numeric(r.get("episode/total_reward"))]
        learner = [r for r in rows if numeric(r.get("train/loss_q"))]
        # Some W&B pages include keys with null values: presence is insufficient.
        evaluations = [r for r in rows if numeric(r.get(
            PREFIX + "policy_deterministic/hazard_person_steps"))]
        histories[rid] = (episodes, learner, evaluations)
        result = dict(name=inventory[rid]["name"], url=inventory[rid]["url"],
                      snapshot_episode=inventory[rid]["summary"].get("global_episode"),
                      resumed_episode=inventory[rid]["config"].get("run/resumed_episode"),
                      validation=[], training_bins=[])
        for row in evaluations:
            result["validation"].append({
                "episode": row["global_episode"],
                "paired_vs_off": row.get("eval/validation/paired/person_steps_reduction_vs_off"),
                "paired_vs_shuttle": row.get("eval/validation/paired/policy_deterministic/person_steps_reduction_vs_shuttle"),
                **{c: {key: row.get(PREFIX + c + "/" + key) for key in (
                    "hazard_person_steps", "activity_dwell_hazard_person_steps",
                    "robot_inside_share", "robot_depth_m", "robot_speed_mean",
                    "followers_per_robot", "held_clear_success", "n")}
                   for c in ("off_zero_command", "shuttle", "policy_deterministic")},
            })
        for low, high in ((1, 500), (501, 1000), (1001, 2000), (2001, 3000),
                          (3001, 4000), (4001, 5000), (5001, 6000)):
            chosen = [r for r in episodes if low <= r["global_episode"] <= high]
            if not chosen:
                continue
            result["training_bins"].append(dict(
                low=low, high=high, n=len(chosen),
                **{key: mean(chosen, key) for key in (
                    "episode/total_reward", "episode/hazard_person_steps",
                    "episode/robots/robot_inside_share", "episode/robots/robot_depth_m",
                    "episode/mode_share/guide", "episode/robot_speed_mean")}))
        recent = [r for r in learner if r["global_episode"] >= 4000]
        floor = inventory[rid]["config"]["ALPHA_MIN"]
        result["recent_learner"] = {key: mean(recent, key) for key in (
            "train/alpha", "train/entropy", "train/target_entropy", "train/loss_q")}
        result["alpha_floor_fraction"] = float(np.mean([
            r["train/alpha"] <= floor * 1.00001 for r in learner])) if learner else None
        floor_rows = [r for r in learner if r["train/alpha"] <= floor * 1.00001]
        result["first_floor_episode"] = floor_rows[0]["global_episode"] if floor_rows else None
        result["stochastic_validation_logged"] = any(
            numeric(r.get(PREFIX + "policy_stochastic/hazard_person_steps")) for r in rows)
        output[rid] = result
    local = ROOT / "latest_validation_records.jsonl"
    if local.exists():
        records = load(local)
        latest_ep = max(r["episode"] for r in records)
        records = [r for r in records if r["episode"] == latest_ep]
        policy = [r for r in records if r["condition"] == "policy_deterministic"]
        expected = next(e for e in output["vsa06hsg"]["validation"] if e["episode"] == latest_ep)
        assert np.isclose(mean(policy, "hazard_person_steps"),
                          expected["policy_deterministic"]["hazard_person_steps"])
        by_key = {(r["scenario"], r["seed"], r["condition"]): r for r in records}
        paired = []
        for row in policy:
            off = by_key[(row["scenario"], row["seed"], "off_zero_command")]
            shuttle = by_key[(row["scenario"], row["seed"], "shuttle")]
            paired.append(dict(scenario=row["scenario"], seed=row["seed"],
                               off=off["hazard_person_steps"],
                               shuttle=shuttle["hazard_person_steps"],
                               policy=row["hazard_person_steps"],
                               vs_off=1-row["hazard_person_steps"]/off["hazard_person_steps"],
                               vs_shuttle=1-row["hazard_person_steps"]/shuttle["hazard_person_steps"]))
        assert len(paired) == 9
        output["latest_pairs"] = paired
        output["latest_condition_means"] = {
            c: {k: mean([r for r in records if r["condition"] == c], k)
                for k in ("hazard_person_steps", "activity_dwell_hazard_person_steps",
                          "reentries_after_clear", "robot_inside_share", "robot_depth_m",
                          "robot_speed_mean", "followers_per_robot", "held_clear_success")}
            for c in ("off_zero_command", "shuttle", "policy_deterministic")}
        assert np.isclose(np.mean([p["vs_off"] for p in paired]), expected["paired_vs_off"])
    (ROOT / "analysis.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    plot(histories)
    print(json.dumps({
        "latest_validation": output["vsa06hsg"]["validation"][-1],
        "latest_pairs": output.get("latest_pairs"),
        "condition_means": output.get("latest_condition_means"),
        "alpha": {k: output[k]["recent_learner"] for k in RUN_IDS},
        "first_floor_episode": output["vsa06hsg"]["first_floor_episode"],
        "floor_fraction": output["vsa06hsg"]["alpha_floor_fraction"],
    }, ensure_ascii=False, indent=2))


def plot(histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = {"vsa06hsg": "Approach20 + Progress150 (latest)",
              "d774mrrb": "Approach20 (different task reward)",
              "cx8bbli5": "Progress50 (resume/config caveat)",
              "fi2kcteu": "Event decisions (different schema)"}
    for rid, (episodes, learner, evaluations) in histories.items():
        x = [r["global_episode"] for r in evaluations]
        axes[0, 0].plot(x, [r[PREFIX+"policy_deterministic/hazard_person_steps"]
                            for r in evaluations], marker=".", label=labels[rid])
    last = histories["vsa06hsg"][2][-1]
    for c, color in (("off_zero_command", "black"), ("shuttle", "gray")):
        axes[0, 0].axhline(last[PREFIX+c+"/hazard_person_steps"], color=color,
                          linestyle="--", label=c+" (latest run)")
    axes[0, 0].set(title="Fixed validation: exposure (lower is better)", ylabel="Person-steps")
    axes[0, 0].legend(fontsize=7)
    ep, tr, ev = histories["vsa06hsg"]
    for c in ("policy_deterministic", "shuttle"):
        axes[0, 1].plot([r["global_episode"] for r in ev],
                        [r[PREFIX+c+"/robot_inside_share"] for r in ev],
                        marker=".", label=c)
    axes[0, 1].set(title="Latest run: robot time inside hazard", ylabel="Fraction")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot([r["global_episode"] for r in tr], [r["train/entropy"] for r in tr],
                    alpha=.45, linewidth=.6, label="Logged entropy")
    axes[1, 0].axhline(-3, color="red", linestyle="--", label="Target entropy")
    axes[1, 0].set(title="Latest run: entropy remains above target")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot([r["global_episode"] for r in tr], [r["train/alpha"] for r in tr], linewidth=.8)
    axes[1, 1].axhline(.05, color="red", linestyle="--", label="Alpha floor = 0.05")
    axes[1, 1].set(title="Latest run: temperature is floor-limited", ylabel="Alpha")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("Episode")
        ax.grid(alpha=.2)
    fig.suptitle("W&B snapshot 2026-10-19 | validation through episode 5000")
    fig.savefig(ROOT / "learning_diagnostics.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
