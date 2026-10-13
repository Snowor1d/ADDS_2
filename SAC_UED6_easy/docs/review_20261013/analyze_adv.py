import os
import json, glob, numpy as np
rows = []
for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/adv/*.json"))):
    d = json.load(open(fn))
    mc = {k: np.array(v) for k, v in d["mc"].items()}
    names = list(mc)
    means = np.array([mc[k].mean() for k in names])
    sds = np.array([mc[k].std(ddof=1) for k in names])
    q = np.array([d["qpred"][k] for k in names])
    # paired (CRN) differences vs policy_det per replicate
    base = mc["policy_det"]
    print(f"== {d['scenario']} t0={d['t0_dec']} inside={d['inside_at_branch']} n_ref={d['n_ref']:.0f} robot0_inside={d['robot0_inside']} H={d['H']} M={d['M']}")
    print(f"   {'action':12s} {'MC mean':>9s} {'MC sd':>7s} {'paired diff vs det (per rep)':>34s} {'critic Q':>9s} {'critic dQ':>9s}")
    for k, m_, s_, qq in zip(names, means, sds, q):
        diff = mc[k] - base
        print(f"   {k:12s} {m_:9.2f} {s_:7.2f}   {np.array2string(diff, precision=1):>32s} {qq:9.2f} {qq - d['qpred']['policy_det']:9.2f}")
    spread_mc = means.std(ddof=1); noise = np.sqrt(np.mean(sds ** 2)); spread_q = q.std(ddof=1)
    # within-replicate (CRN) spread across actions
    mat = np.stack([mc[k] for k in names])            # (A, M)
    crn_spread = mat.std(axis=0, ddof=1).mean()
    rows.append((spread_mc, noise, spread_q, crn_spread, np.corrcoef(means, q)[0, 1]))
    print(f"   spread of MC means across actions = {spread_mc:.2f} | MC noise sd (per rollout) = {noise:.2f} | CRN within-rep spread = {crn_spread:.2f} | critic spread = {spread_q:.2f} | corr(MC, Q) = {rows[-1][4]:+.2f}")
r = np.array(rows)
print("\nOVERALL mean: MC action spread %.2f | rollout noise sd %.2f | CRN spread %.2f | critic spread %.2f | mean corr %.2f" % (r[:,0].mean(), r[:,1].mean(), r[:,3].mean(), r[:,2].mean(), r[:,4].mean()))
