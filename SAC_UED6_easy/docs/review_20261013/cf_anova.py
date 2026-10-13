import os
import json, glob, numpy as np
acts = ["policy_det", "in_off", "in_guide", "out_guide", "out_off", "still_guide"]
out = []
for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/adv/*.json"))):
    d = json.load(open(f))
    X = np.array([d["mc"][a] for a in acts])        # (A, M)
    A, M = X.shape
    grand = X.mean()
    am = X.mean(1, keepdims=True); rm = X.mean(0, keepdims=True)
    ms_a = M * ((am - grand) ** 2).sum() / (A - 1)
    resid = X - am - rm + grand                       # action x replicate interaction (CRN pairing removes rep main effect)
    ms_e = (resid ** 2).sum() / ((A - 1) * (M - 1))
    var_a = max(0.0, (ms_a - ms_e) / M)
    F = ms_a / ms_e
    q = np.array([d["qpred"][a] for a in acts])
    out.append((np.sqrt(var_a), np.sqrt(ms_e), q.std(ddof=1), F))
    print(f"{d['scenario']:26s} t0={d['t0_dec']:2d}: noise-corrected action sd {np.sqrt(var_a):6.1f} | paired residual sd {np.sqrt(ms_e):6.1f} | F={F:5.1f} | critic sd {q.std(ddof=1):.2f}")
o = np.array(out)
print(f"mean: noise-corrected action sd {o[:,0].mean():.1f}, paired residual sd {o[:,1].mean():.1f}, critic sd {o[:,2].mean():.2f}, ratio 1/{o[:,0].mean()/o[:,2].mean():.0f}")
