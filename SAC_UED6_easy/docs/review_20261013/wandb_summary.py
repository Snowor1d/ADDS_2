import json, glob, collections, statistics as st, sys
import numpy as np
P = "eval/validation/200m/robots_3/"
def load(fn):
    rows = [json.loads(l) for l in open(fn)]
    return rows
for fn in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_dump/") + "adds-sac-ued6-261011ENV__*.jsonl")) + sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb_dump/") + "adds-sac-ued6__*.jsonl")):
    cfgj = json.load(open(fn.replace(".jsonl", ".config.json")))
    rows = load(fn)
    print("="*100)
    print(fn.split("__")[-1], cfgj["name"], cfgj["state"], cfgj["created_at"])
    # validation
    evm = collections.OrderedDict()
    for r in rows:
        ks = [k for k in r if k.startswith(P) and r[k] is not None]
        if ks:
            d = evm.setdefault(r.get("global_episode"), {"global_episode": r.get("global_episode")})
            for k in ks: d[k] = r[k]
    ev = [d for d in evm.values() if (P + "policy_deterministic/hazard_person_steps") in d]
    if ev:
        print(" validation (episode: off / shuttle / det  hazard_person_steps; reward off/shuttle/det; det guide? n)")
        for r in ev:
            ge = r.get("global_episode")
            g = lambda c, m: (r.get(P + c + "/" + m) if r.get(P + c + "/" + m) is not None else float("nan"))
            print(f"  ep {ge}: hps off={g('off_zero_command','hazard_person_steps'):.0f} shut={g('shuttle','hazard_person_steps'):.0f} det={g('policy_deterministic','hazard_person_steps'):.0f} | R off={g('off_zero_command','total_reward'):.1f} shut={g('shuttle','total_reward'):.1f} det={g('policy_deterministic','total_reward'):.1f} | det speed={g('policy_deterministic','robot_speed_mean'):.2f} foll_dep={g('policy_deterministic','followed_departures')} shut_foll={g('shuttle','followed_departures')} n={g('policy_deterministic','n')}")
    eps = [r for r in rows if "episode/total_reward" in r]
    tr = [r for r in rows if "train/loss_q" in r]
    if eps:
        E = np.array([r["global_episode"] for r in eps])
        def col(k):
            return np.array([r.get(k, np.nan) for r in eps], dtype=float)
        R = col("episode/total_reward"); H = col("episode/hazard_person_steps"); G = col("episode/mode_share/guide")
        S = col("episode/robot_speed_mean"); C = col("episode/command_norm_mean"); EPS = col("episode/epsilon")
        F = col("episode/followed_departures"); I = col("episode/initial_inside"); PT = col("episode/reward/person_time"); RP = col("episode/reward/remaining_path"); RE = col("episode/reward/reentry"); CO = col("episode/reward/collision"); PR=col("episode/reward/projection")
        bins = list(range(0, int(E.max()) + 500, 500))
        print(" training episodes: bin | n | R | hps | hps/init_inside | guide | speed | cmdnorm | eps | followed_dep | PT RP RE CO PR")
        for a, b in zip(bins[:-1], bins[1:]):
            m = (E > a) & (E <= b)
            if m.sum() == 0: continue
            rat = np.nanmean(H[m] / np.maximum(I[m], 1))
            print(f"  {a+1:5d}-{b:5d} | {m.sum():4d} | {np.nanmean(R[m]):8.1f} | {np.nanmean(H[m]):8.0f} | {rat:6.1f} | {np.nanmean(G[m]):.2f} | {np.nanmean(S[m]):.2f} | {np.nanmean(C[m]):.2f} | {np.nanmean(EPS[m]):.2f} | {np.nanmean(F[m]):6.1f} | {np.nanmean(PT[m]):7.1f} {np.nanmean(RP[m]):6.1f} {np.nanmean(RE[m]):6.1f} {np.nanmean(CO[m]):6.1f} {np.nanmean(PR[m]):6.1f}")
    if tr:
        U = np.array([r["global_update"] for r in tr]); Ep = np.array([r.get("global_episode", np.nan) for r in tr], dtype=float)
        def tcol(k): return np.array([r.get(k, np.nan) for r in tr], dtype=float)
        A = tcol("train/alpha"); EN = tcol("train/entropy"); Q = tcol("train/q_team"); T = tcol("train/target"); LQ = tcol("train/loss_q"); LP = tcol("train/loss_pi"); TE = tcol("train/target_entropy")
        n = len(tr); idx = np.linspace(0, n - 1, min(12, n)).astype(int)
        print(" learner: update | ep | alpha | entropy (target) | q_team | target | loss_q | loss_pi")
        for i in idx:
            print(f"  {U[i]:7d} | {Ep[i]:5.0f} | {A[i]:.4f} | {EN[i]:6.2f} ({TE[i]:.1f}) | {Q[i]:8.2f} | {T[i]:8.2f} | {LQ[i]:8.3f} | {LP[i]:8.2f}")
