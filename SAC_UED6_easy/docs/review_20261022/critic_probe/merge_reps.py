"""Join extra repeats (results/<state>_more.json) into results/<state>.json.

The extra run must start from the same branch: same step, robot position,
candidate actions and critic values. The 3-repeat original is kept as
<state>_r3.json.
"""
import glob, json, os, shutil, sys
import numpy as np

rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
for more in sorted(glob.glob(os.path.join(rdir, "s*_t*_more.json"))):
    base = more.replace("_more.json", ".json")
    a, b = json.load(open(base)), json.load(open(more))
    assert a["branch_step"] == b["branch_step"] and np.allclose(a["robot0_xy"], b["robot0_xy"])
    for c in a["cands"]:
        assert np.allclose(a["cands"][c], b["cands"][c]), c
        if b["q1"]:  # only recorded by a run that includes repeat 0
            assert abs(a["q1"][c] - b["q1"][c]) < 1e-4, c
    ids_a = a.get("rep_ids", list(range(a["reps"])))
    assert not set(ids_a) & set(b["rep_ids"]), "overlapping repeats"
    if not os.path.exists(base[:-5] + "_r3.json"):
        shutil.copy(base, base[:-5] + "_r3.json")
    for key in ("mc", "first_hold", "steps_run"):
        for c in a["cands"]:
            a[key][c] += b[key][c]
    for h in a["mc_h"]:
        for c in a["cands"]:
            a["mc_h"][h][c] += b["mc_h"][h][c]
    a["rep_ids"] = ids_a + b["rep_ids"]
    a["reps"] = len(a["rep_ids"])
    json.dump(a, open(base, "w"))
    os.remove(more)
    print("merged", os.path.basename(base), "reps", a["reps"])
