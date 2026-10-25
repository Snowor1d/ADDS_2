"""Reproduce this review from saved scalar histories, without network access.

Control 4pr4wxci (A0/P150, shared heads, gamma 0.99) against two single-factor
changes: 5uhnt1nz (ACTOR_CONT_HEADS=per_mode) and ul6kds66 (gamma 0.995).
vsa06hsg (A20/P150) is kept as a reference with a different reward.
"""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = {
    "4pr4wxci": ("control A0/P150", ROOT / "4pr4wxci.jsonl.gz"),
    "5uhnt1nz": ("per_mode", ROOT / "5uhnt1nz.jsonl.gz"),
    "ul6kds66": ("gamma 0.995", ROOT / "ul6kds66.jsonl.gz"),
    "vsa06hsg": ("ref A20/P150", ROOT.parent / "review_20261020" / "vsa06hsg.jsonl.gz"),
}
PREFIX = "eval/validation/200m/robots_3/"
DET = PREFIX + "policy_deterministic/"
VAL_METRICS = ("hazard_person_steps", "activity_dwell_hazard_person_steps",
               "reentries_after_clear", "robot_inside_share", "robot_depth_m",
               "robot_speed_mean", "followers_per_robot", "command_norm_mean",
               "mode_switch_rate", "held_clear_success")
EP_METRICS = ("episode/total_reward", "episode/hazard_person_steps",
              "episode/mode_share/guide", "episode/mode_switch_rate",
              "episode/robots/robot_inside_share", "episode/robots/robot_depth_m",
              "episode/robot_speed_mean", "episode/command_norm_mean",
              "episode/robots/followers_per_robot", "episode/reward/shape_progress",
              "episode/steps")
TR_METRICS = ("train/alpha", "train/entropy", "train/loss_q", "train/loss_pi",
              "train/q_team", "train/target")
BINS = [(lo, lo + 499) for lo in range(1, 5501, 500)]


def load(path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return float(np.mean(v)) if v else None


def dedup(rows, marker, identity):
    out, dups = {}, 0
    for row in rows:
        if marker not in row:
            continue
        key = tuple(row.get(k) for k in identity)
        if key in out:
            for f in out[key].keys() & row.keys():
                if not f.startswith("_"):
                    assert out[key][f] == row[f], (key, f)
            dups += 1
        out[key] = row
    return sorted(out.values(), key=lambda r: tuple(r.get(k) for k in identity)), dups


def main():
    result, hist = {}, {}
    for rid, (label, path) in RUNS.items():
        rows = load(path)
        eps, d1 = dedup(rows, "episode/total_reward", ("global_episode",))
        val, d2 = dedup(rows, DET + "hazard_person_steps", ("global_episode",))
        tr, d3 = dedup(rows, "train/alpha", ("global_update", "global_episode"))
        hist[rid] = eps, val, tr
        item = dict(label=label, last_episode=max(r["global_episode"] for r in eps),
                    last_update=max((r.get("global_update") or 0) for r in rows),
                    duplicates=dict(episode=d1, validation=d2, learner=d3),
                    validation=[], bins=[])
        for r in val:
            item["validation"].append(dict(
                episode=r["global_episode"], update=r.get("global_update"),
                vs_off=r.get("eval/validation/paired/policy_deterministic/person_steps_reduction_vs_off"),
                vs_shuttle=r.get("eval/validation/paired/policy_deterministic/person_steps_reduction_vs_shuttle"),
                wins_vs_off=r.get("eval/validation/paired/policy_deterministic/relative_pairs_vs_off"),
                **{m: r.get(DET + m) for m in VAL_METRICS}))
        for lo, hi in BINS:
            e = [r for r in eps if lo <= r["global_episode"] <= hi]
            t = [r for r in tr if lo <= r["global_episode"] <= hi]
            if not e:
                continue
            item["bins"].append(dict(lo=lo, hi=hi, n=len(e),
                                     **{k: mean(e, k) for k in EP_METRICS},
                                     **{k: mean(t, k) for k in TR_METRICS}))
        result[rid] = item

    # Baselines must be identical across runs and checkpoints.
    rows = [r for rid in RUNS for r in hist[rid][1]]
    for cond in ("off_zero_command", "shuttle"):
        for m in ("hazard_person_steps", "activity_dwell_hazard_person_steps"):
            v = [r[PREFIX + cond + "/" + m] for r in rows if PREFIX + cond + "/" + m in r]
            assert np.allclose(v, v[0]), (cond, m)
            result.setdefault("baselines", {}).setdefault(cond, {})[m] = v[0]

    # Window means of the deterministic validation, per run.
    def window(rid, lo, hi):
        v = [x for x in result[rid]["validation"] if lo <= x["episode"] <= hi]
        h = [x["hazard_person_steps"] for x in v]
        return dict(n=len(h), mean=float(np.mean(h)) if h else None,
                    sd=float(np.std(h, ddof=1)) if len(h) > 1 else None,
                    best=min(h) if h else None,
                    best_ep=min(v, key=lambda x: x["hazard_person_steps"])["episode"] if v else None)
    result["windows"] = {rid: {f"{lo}-{hi}": window(rid, lo, hi)
                               for lo, hi in ((500, 1500), (2000, 3000), (3500, 5000), (500, 6000))}
                         for rid in RUNS}
    (ROOT / "analysis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    plot(hist, result)
    report(result)


def report(res):
    ids = list(RUNS)
    print("last episode / update:", {r: (res[r]["last_episode"], res[r]["last_update"]) for r in ids})
    print("duplicates:", {r: res[r]["duplicates"] for r in ids})
    print("baselines:", res["baselines"])
    eps = sorted({v["episode"] for r in ids for v in res[r]["validation"]})
    print("\nDET hazard_person_steps | vs_off | vs_shuttle | updates")
    for e in eps:
        cells = []
        for r in ids:
            v = next((x for x in res[r]["validation"] if x["episode"] == e), None)
            cells.append(f"{v['hazard_person_steps']:8.0f} {v['vs_off']:+.3f} {v['vs_shuttle']:+.3f} {v['update']:>7}"
                         if v else " " * 35)
        print(f"{e:5d} | " + " | ".join(cells))
    print("\nwindows:", json.dumps(res["windows"], indent=0))
    for r in ids:
        print(f"\n== {r} {res[r]['label']}  latest validation details")
        for v in res[r]["validation"][-3:]:
            print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
    print("\nTRAINING BINS")
    for k in EP_METRICS + TR_METRICS:
        print(f"\n{k}")
        for r in ids:
            print(f"  {res[r]['label']:<16}", " ".join(
                f"{b[k]:9.3f}" if b.get(k) is not None else "     None" for b in res[r]["bins"]))


def plot(hist, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    colors = {"4pr4wxci": "#1f77b4", "5uhnt1nz": "#d62728", "ul6kds66": "#2ca02c", "vsa06hsg": "#999999"}
    for rid, (label, _) in RUNS.items():
        eps, val, tr = hist[rid]
        x = [r["global_episode"] for r in val]
        style = dict(color=colors[rid], label=label, lw=1.4 if rid != "vsa06hsg" else 1,
                     ls="-" if rid != "vsa06hsg" else ":")
        ax[0, 0].plot(x, [r[DET + "hazard_person_steps"] for r in val], marker="o", ms=3, **style)
        ax[0, 1].plot(x, [r[DET + "robot_inside_share"] for r in val], marker="o", ms=3, **style)
        ax[0, 2].plot(x, [r[DET + "robot_speed_mean"] for r in val], marker="o", ms=3, **style)
        for a, k in ((ax[1, 0], "episode/total_reward"), (ax[1, 1], "episode/mode_share/guide")):
            b = range(1, 6001, 100)
            y = [mean([r for r in eps if lo <= r["global_episode"] < lo + 100], k) for lo in b]
            a.plot([lo + 49.5 for lo in b], y, **style)
        b = range(1, 6001, 100)
        ax[1, 2].plot([lo + 49.5 for lo in b],
                      [mean([r for r in tr if lo <= r["global_episode"] < lo + 100], "train/entropy") for lo in b],
                      **style)
    for a, m in ((ax[0, 0], "hazard_person_steps"), (ax[0, 1], "robot_inside_share"),
                 (ax[0, 2], "robot_speed_mean")):
        r0 = hist["4pr4wxci"][1][-1]
        for cond, c in (("off_zero_command", "black"), ("shuttle", "gray")):
            a.axhline(r0[PREFIX + cond + "/" + m], color=c, ls="--", alpha=.6, label=cond)
    titles = ("Validation hazard person-steps (lower better)", "Validation robot inside share",
              "Validation robot speed (m/s)", "Training task reward (100-ep mean)",
              "Training guide mode share", "Actor entropy (100-ep mean)")
    for a, t in zip(ax.flat, titles):
        a.set_title(t, fontsize=10)
        a.set_xlabel("Episode")
        a.grid(alpha=.2)
        a.legend(fontsize=7)
    fig.suptitle("W&B snapshot 2026-10-22 | control A0/P150 vs per_mode vs gamma 0.995 | 9 validation tasks")
    fig.savefig(ROOT / "learning_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
