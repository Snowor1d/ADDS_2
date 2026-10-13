import os
import json, glob, numpy as np
acts = ["policy_det", "in_off", "in_guide", "out_guide", "out_off", "still_guide"]
rows, stats = [], []
order = {"val_dotonbori_200m_0": 0, "val_covent_garden_200m_0": 1, "val_hongdae_200m_2": 2}
files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/adv/*.json")), key=lambda f: (order[json.load(open(f))["scenario"]], json.load(open(f))["t0_dec"]))
print("| 상태 | 위험구역 인원 / N_ref | 로봇0 위치 | " + " | ".join(acts) + " | 실제 sd | 반복 잡음 sd | critic sd | 상관 |")
print("|---|---|---|" + "---:|" * (len(acts) + 4))
for f in files:
    d = json.load(open(f))
    mc = {k: np.array(v) for k, v in d["mc"].items()}
    base = mc["policy_det"]
    means = np.array([mc[a].mean() for a in acts]); q = np.array([d["qpred"][a] for a in acts])
    rel = means - means[0]
    sds = np.array([mc[a].std(ddof=1) for a in acts])
    spread, noise, qs = means.std(ddof=1), np.sqrt((sds ** 2).mean()), q.std(ddof=1)
    c = np.corrcoef(means, q)[0, 1]
    stats.append((spread, noise, qs, c, np.ptp(means), np.ptp(q)))
    name = d["scenario"].replace("val_", "").replace("_200m", "")
    cells = " | ".join(f"{r:+.0f}" for r in rel)
    print(f"| {name}, t={d['t0_dec']*4} s | {d['inside_at_branch']} / {d['n_ref']:.0f} | {'안' if d['robot0_inside'] else '밖'} | {cells} | {spread:.1f} | {noise:.1f} | {qs:.2f} | {c:+.2f} |")
s = np.array(stats)
print()
print(f"평균: 실제 sd {s[:,0].mean():.1f}, 반복 잡음 sd {s[:,1].mean():.1f}, critic sd {s[:,2].mean():.2f}, 비율 1/{s[:,0].mean()/s[:,2].mean():.0f}, 상관 {s[:,3].mean():+.2f}, 실제 최대-최소 평균 {s[:,4].mean():.1f}, critic 최대-최소 평균 {s[:,5].mean():.2f}, n={len(s)}")
