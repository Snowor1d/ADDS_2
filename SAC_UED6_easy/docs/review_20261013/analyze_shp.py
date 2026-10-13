import os
import json, glob, numpy as np, collections
recs = []
for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/shp/*.json"))):
    recs += json.load(open(fn))
by = collections.defaultdict(list)
for r in recs:
    by[(r["scenario"], r["t0"])].append(r)
acts = ["policy_det", "in_off", "in_guide", "out_guide", "out_off", "still_guide"]
summ = {k: [] for k in ("R8", "F_prog", "F_appr")}
for key, rs in sorted(by.items()):
    print(f"== {key[0]} t0={key[1]} inside={rs[0]['inside']} n_ref={rs[0]['n_ref']:.0f} robot0_inside={rs[0]['robot0_inside']} phi_appr0={rs[0]['phi0_appr']:.2f} followers0={rs[0]['follow0_before']}")
    for metric in ("R8", "F_prog", "F_appr"):
        mat = np.array([[next(x[metric] for x in rs if x["action"] == a and x["rep"] == rep) for rep in range(3)] for a in acts])  # (A, M)
        between = mat.mean(1).std(ddof=1)          # spread of action means
        within = mat.std(1, ddof=1).mean()          # replicate noise per action
        crn = (mat - mat.mean(0, keepdims=True)).std(ddof=1)  # action effect after removing replicate common part
        summ[metric].append((between, within, crn))
        print(f"   {metric:7s} " + " ".join(f"{a}={mat[i].mean():+7.3f}" for i, a in enumerate(acts)) + f" | action spread {between:.3f} noise {within:.3f}")
    fol = {a: np.mean([x['follow0_after'] for x in rs if x['action'] == a]) for a in acts}
    print("   followers of robot0 after 4 s:", {a: round(v, 1) for a, v in fol.items()})
print()
for m, v in summ.items():
    v = np.array(v)
    print(f"{m:7s} mean action-spread {v[:,0].mean():.3f} | mean replicate noise {v[:,1].mean():.3f} | ratio {v[:,0].mean()/max(v[:,1].mean(),1e-9):.2f}")
