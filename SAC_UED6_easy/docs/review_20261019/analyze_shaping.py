"""Compare shaping variants from the fixed W&B snapshot; no rollouts/training."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze import ROOT, PREFIX, load, mean, numeric


IDS = ("vsa06hsg", "d774mrrb", "cx8bbli5", "fi2kcteu")
LABELS = {"vsa06hsg": "A20 + P150; task path=0, reentry=0",
          "d774mrrb": "A20; task path=1, reentry=2",
          "cx8bbli5": "P50; resume/config caveat",
          "fi2kcteu": "No PBRS; different schema/estimator/UPT"}


def main():
    histories = {rid: load(ROOT / (rid + ".jsonl")) for rid in IDS}
    vkey = PREFIX + "policy_deterministic/"
    evaluations = {
        rid: {r["global_episode"]: r for r in rows
              if numeric(r.get(vkey + "hazard_person_steps"))}
        for rid, rows in histories.items()}
    a, b = evaluations["vsa06hsg"], evaluations["d774mrrb"]
    common = sorted(ep for ep in set(a) & set(b) if ep >= 1000)
    metrics = ("hazard_person_steps", "activity_dwell_hazard_person_steps",
               "reentries_after_clear", "robot_inside_share", "robot_depth_m",
               "robot_speed_mean", "followers_per_robot")
    matched = {}
    for metric in metrics:
        am = float(np.mean([a[ep][vkey + metric] for ep in common]))
        bm = float(np.mean([b[ep][vkey + metric] for ep in common]))
        matched[metric] = dict(combined=am, approach=bm,
                               combined_relative_to_approach=am / bm - 1)
    episodes = {rid: [r for r in rows if numeric(r.get("episode/total_reward"))]
                for rid, rows in histories.items()}
    decomp = {}
    for rid in ("vsa06hsg", "d774mrrb"):
        rr = [r for r in episodes[rid] if 3001 <= r["global_episode"] <= 4000]
        common_r = mean(rr, "episode/reward/person_time") + mean(rr, "episode/reward/collision")
        decomp[rid] = dict(n=len(rr), reported=mean(rr, "episode/total_reward"),
                           common_task_reward=common_r,
                           remaining_path=mean(rr, "episode/reward/remaining_path"),
                           reentry=mean(rr, "episode/reward/reentry"),
                           shape_progress=mean(rr, "episode/reward/shape_progress"),
                           shape_approach=mean(rr, "episode/reward/shape_approach"))
        assert np.isclose(decomp[rid]["reported"], common_r
                          + decomp[rid]["remaining_path"] + decomp[rid]["reentry"])
    old, new = decomp["d774mrrb"], decomp["vsa06hsg"]
    gain = new["reported"] - old["reported"]
    decomp["gain"] = dict(reported=gain,
                          comparable_component=new["common_task_reward"]-old["common_task_reward"],
                          removed_terms=-(old["remaining_path"]+old["reentry"]),
                          removed_terms_fraction=-(old["remaining_path"]+old["reentry"])/gain)
    # Use the current production potential implementation on a synthetic geometry.
    path = ROOT.parents[1] / "sim" / "rewards.py"
    spec = importlib.util.spec_from_file_location("shaping_review_rewards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward = object.__new__(module.TaskReward)
    reward.approach_cap, reward.approach_kappa = 50.0, 10.0
    robot = SimpleNamespace(xy=[0.0, 0.0])
    person = SimpleNamespace(xy=[10.0, 0.0], type=1,
                             informed_by_robot=False, hazard_memory=[])
    phi0 = reward._phi_approach(None, [robot], [person])[0]
    robot.xy = [5.0, 0.0]
    phi1 = reward._phi_approach(None, [robot], [person])[0]
    person.type = 0
    phi2 = reward._phi_approach(None, [robot], [person])[0]
    g = .99 ** (1 / 4)
    probe = dict(phi_at_10m=phi0, phi_at_5m=phi1, phi_after_follow_accept=phi2,
                 move_5m_in_4s_shaping=20 * (g ** 8 * phi1 - phi0),
                 follow_accept_one_step_shaping=20 * (g * phi2 - phi1),
                 note="Synthetic potential probe only, not a simulated or feasible full trajectory.")
    out = dict(common_episodes=common, matched_metrics=matched,
               training_reward_decomposition_3001_4000=decomp,
               synthetic_approach_probe=probe)
    (ROOT / "shaping_analysis.json").write_text(json.dumps(out, indent=2))
    plot(evaluations, vkey)
    print(json.dumps(out, indent=2))


def plot(evaluations, vkey):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric, title in zip(axes.flat,
                                ("hazard_person_steps", "activity_dwell_hazard_person_steps",
                                 "robot_inside_share", "reentries_after_clear"),
                                ("Hazard exposure (lower is better)", "Dwelling-person exposure (lower is better)",
                                 "Robot time inside hazard (diagnostic only)", "Reentries after clear (lower is better)")):
        for rid, rows in evaluations.items():
            xs = sorted(ep for ep, row in rows.items() if numeric(row.get(vkey + metric)))
            ax.plot(xs, [rows[ep][vkey + metric] for ep in xs], marker=".", label=LABELS[rid])
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Shaping comparison | descriptive, not a controlled ablation | 2026-10-19 snapshot")
    fig.savefig(ROOT / "shaping_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
