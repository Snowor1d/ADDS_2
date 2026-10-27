"""fg5jloea (RD60, UPT4) against the A0/P150 control and the per_mode run,
from saved W&B scalar histories, without network access."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RUNS = {
    "4pr4wxci": ("control A0/P150", ROOT.parent / "review_20261022" / "4pr4wxci.jsonl.gz", 2),
    "fg5jloea": ("RD60 UPT4", ROOT / "fg5jloea.jsonl.gz", 4),
    "5uhnt1nz": ("per_mode", ROOT / "5uhnt1nz.jsonl.gz", 2),
    "vsa06hsg": ("ref A20/P150", ROOT.parent / "review_20261020" / "vsa06hsg.jsonl.gz", 2),
}
DET = "eval/validation/200m/robots_3/policy_deterministic/"
PAIR = "eval/validation/paired/policy_deterministic/"
EP = ("episode/total_reward", "episode/hazard_person_steps", "episode/mode_share/guide",
      "episode/mode_switch_rate", "episode/robots/robot_inside_share",
      "episode/robots/robot_depth_m", "episode/robot_speed_mean",
      "episode/robots/followers_per_robot")
TR = ("train/alpha", "train/entropy", "train/loss_q", "train/q_team")


def load(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def mean(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return float(np.mean(v)) if v else None


def dedup(rows, marker, ident):
    out = {}
    for r in rows:
        if marker in r:
            out[tuple(r.get(k) for k in ident)] = r
    return [out[k] for k in sorted(out)]


hist = {}
for rid, (label, path, upt) in RUNS.items():
    rows = load(path)
    eps = dedup(rows, "episode/total_reward", ("global_episode",))
    val = dedup(rows, DET + "hazard_person_steps", ("global_episode",))
    tr = dedup(rows, "train/alpha", ("global_update", "global_episode"))
    hist[rid] = (eps, val, tr, upt)

print("== validation (deterministic, 9 tasks): hazard person-steps | paired vs off / vs shuttle | update")
allep = sorted({v["global_episode"] for rid in RUNS for v in hist[rid][1]})
print("ep     " + "".join(f"{RUNS[r][0]:>34s}" for r in RUNS))
for e in allep:
    cells = []
    for rid in RUNS:
        v = next((x for x in hist[rid][1] if x["global_episode"] == e), None)
        cells.append(f"{v[DET+'hazard_person_steps']:9.0f} {v.get(PAIR+'person_steps_reduction_vs_off', float('nan')):+.3f} "
                     f"{v.get(PAIR+'person_steps_reduction_vs_shuttle', float('nan')):+.3f} {v.get('global_update', 0)/1000:6.0f}k"
                     if v else " " * 34)
    print(f"{e:5d}  " + "".join(f"{c:>34s}" for c in cells))

print("\n== decisions per episode (from updates per episode / UPT), 500-episode bins")
for rid in RUNS:
    eps, val, tr, upt = hist[rid]
    pts = sorted((r["global_episode"], r["global_update"]) for r in tr if r.get("global_update"))
    out = []
    for lo in range(500, 13000, 500):
        a = [p for p in pts if lo - 250 <= p[0] < lo + 250]
        if len(a) > 2:
            (e0, u0), (e1, u1) = a[0], a[-1]
            if e1 > e0:
                out.append(f"{lo}:{(u1 - u0) / (e1 - e0) / upt:.0f}")
    print(f"  {RUNS[rid][0]:16s} " + " ".join(out[:12]))

print("\n== training, 500-episode bins")
bins = [(lo, lo + 499) for lo in range(1, 12501, 500)]
for k in EP + TR:
    print(f"\n{k}")
    for rid in RUNS:
        eps, val, tr, upt = hist[rid]
        src = eps if k.startswith("episode/") else tr
        vals = []
        for lo, hi in bins:
            sel = [r for r in src if lo <= r["global_episode"] <= hi]
            m = mean(sel, k)
            if m is None:
                if sel:
                    vals.append(float("nan"))
                continue
            vals.append(m)
        print(f"  {RUNS[rid][0]:16s} " + " ".join(f"{v:8.3f}" for v in vals[:14]))
