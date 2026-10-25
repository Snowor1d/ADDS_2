"""Where a decision's effect and the return's noise sit, person by person.

For each state (candidates x repeats, common random numbers within a repeat):
  component       a part of robot 0's return: the persons who interacted
                  with robot 0 after the branch (followed it or were exposed
                  to its signal, in any rollout), the other persons present
                  at the branch, persons who arrived later, robot 0's own
                  terms, and all of it
  effect_sd       noise-corrected sd over candidates of the component's mean
  noise_sd        sd of one rollout around its candidate and repeat means
  share           the component's part of the total effect, <mu_X, mu>/|mu|^2
                  (sums to 1 over a partition)
  snr             effect_sd / noise_sd
Per person: the same two-way layout for each person present at the branch,
and how independent the persons' noises are: var(sum of noises) against the
sum of their variances.

Usage: analyze_persons.py [RESULTS_DIR]
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def anova(x):
    """x (C, M) -> effect var (noise-corrected), noise var, F, p, centred means."""
    C, M = x.shape
    g = x.mean()
    cm, rm = x.mean(1), x.mean(0)
    ss_c = M * ((cm - g) ** 2).sum()
    resid = x - cm[:, None] - rm[None, :] + g
    ms_c = ss_c / (C - 1)
    ms_e = (resid ** 2).sum() / ((C - 1) * (M - 1))
    F = ms_c / ms_e if ms_e > 0 else np.inf
    ms_r = C * ((rm - g) ** 2).sum() / (M - 1)
    rep_var = max(0.0, (ms_r - ms_e) / C)
    return dict(effect_var=max(0.0, (ms_c - ms_e) / M), noise_var=ms_e, F=F,
                rep_var=rep_var, rep_share=rep_var / (rep_var + ms_e) if rep_var + ms_e > 0 else 0.0,
                p=float(stats.f.sf(F, C - 1, (C - 1) * (M - 1))) if ms_e > 0 else 0.0,
                mu=cm - cm.mean(), resid=resid)


def load_state(rdir, state, sub="persons"):
    parts = sorted(glob.glob(os.path.join(rdir, sub, f"{state}_r*.npz")))
    zs = [np.load(p) for p in parts]
    assert all((z["ids"] == zs[0]["ids"]).all() for z in zs)
    cat = lambda k: np.concatenate([z[k] for z in zs], axis=1)
    d = dict(names=[str(n) for n in zs[0]["names"]], Rp=cat("Rp").astype(np.float64),
             R_inflow=cat("R_inflow"), R_robot=cat("R_robot"), total=cat("total"),
             follow=cat("follow"), expo=cat("expo"), dist0=zs[0]["dist0"],
             inside0=zs[0]["inside0"], reps=np.concatenate([z["rep_ids"] for z in zs]))
    # A stochastic continuation must reproduce the probe's Monte Carlo
    # returns (same seeds); a deterministic one is a different rollout.
    d["max_ref_gap"] = float("nan")
    cont = str(zs[0]["continuation"]) if "continuation" in zs[0].files else "stoch"
    d["continuation"] = cont
    if cont == "stoch":
        js = json.load(open(os.path.join(rdir, f"{state}.json")))
        ref = np.asarray([[js["mc"][c][r] for r in d["reps"]] for c in d["names"]])
        d["max_ref_gap"] = float(np.abs(ref - d["total"]).max())
    return d


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
    sub = sys.argv[2] if len(sys.argv) > 2 else "persons"
    states = sorted({os.path.basename(p).rsplit("_r", 1)[0]
                     for p in glob.glob(os.path.join(rdir, sub, "s*_r*.npz"))})
    report = {}
    for st in states:
        d = load_state(rdir, st, sub)
        Rp = d["Rp"]
        C, M, P = Rp.shape
        inter = (d["follow"].any((0, 1))) | ((d["expo"] > 0).any((0, 1)))
        near = d["dist0"] <= 20.0
        comps = {
            "total": d["total"],
            "interacted": Rp[:, :, inter].sum(-1),
            "other_present": Rp[:, :, ~inter].sum(-1),
            "inflow": d["R_inflow"],
            "robot": d["R_robot"],
            "near20m": Rp[:, :, near].sum(-1),
            "interacted+robot": Rp[:, :, inter].sum(-1) + d["R_robot"],
        }
        tot = anova(d["total"])
        mu_t = tot["mu"]
        rows = {}
        for k, x in comps.items():
            a = anova(x)
            rows[k] = dict(effect_sd=float(np.sqrt(a["effect_var"])),
                           noise_sd=float(np.sqrt(a["noise_var"])),
                           snr=float(np.sqrt(a["effect_var"]) / np.sqrt(a["noise_var"])) if a["noise_var"] > 0 else None,
                           share=float(a["mu"] @ mu_t / (mu_t @ mu_t)),
                           p=a["p"])
        # Per person.
        per = [anova(Rp[:, :, i]) for i in range(P)]
        ev = np.asarray([a["effect_var"] for a in per])
        nv = np.asarray([a["noise_var"] for a in per])
        pp = np.asarray([a["p"] for a in per])
        resid = np.stack([a["resid"] for a in per], -1)          # (C, M, P)
        dof = (C - 1) * (M - 1)
        var_sum = float(((resid.sum(-1)) ** 2).sum() / dof)
        sum_var = float(nv.sum())
        var_sum_int = float(((resid[:, :, inter].sum(-1)) ** 2).sum() / dof)
        sum_var_int = float(nv[inter].sum())
        report[st] = dict(
            persons=P, interacted=int(inter.sum()), near20=int(near.sum()),
            inside_at_branch=int(d["inside0"].sum()), reps=M,
            max_ref_gap=d["max_ref_gap"], components=rows,
            persons_p05=int((pp < 0.05).sum()), persons_p05_interacted=int((pp[inter] < 0.05).sum()),
            noise_independence_all=var_sum / sum_var if sum_var > 0 else None,
            noise_independence_interacted=var_sum_int / sum_var_int if sum_var_int > 0 else None,
            top10_effect_share=float(np.sort(ev)[::-1][:10].sum() / ev.sum()) if ev.sum() > 0 else None,
            median_person_snr_interacted=float(np.median(np.sqrt(ev[inter]) / np.sqrt(np.maximum(nv[inter], 1e-12)))) if inter.any() else None)
    json.dump(report, open(os.path.join(rdir, f"{sub}_summary.json"), "w"), indent=1)
    for st, r in report.items():
        print(f"\n== {st}: persons {r['persons']} (inside {r['inside_at_branch']}), interacted with robot 0 {r['interacted']}, "
              f"reps {r['reps']}, gap to probe {r['max_ref_gap']:.1e}")
        print(f"   {'component':18s} {'effect':>7s} {'noise':>7s} {'snr':>6s} {'share':>6s} {'p':>8s}")
        for k, c in r["components"].items():
            print(f"   {k:18s} {c['effect_sd']:7.2f} {c['noise_sd']:7.2f} {c['snr'] or 0:6.2f} {c['share']:+6.2f} {c['p']:8.1e}")
        print(f"   persons with p<0.05: {r['persons_p05']} (of them interacted: {r['persons_p05_interacted']}); "
              f"top-10 share of per-person effect variance {r['top10_effect_share']}")
        print(f"   noise var(sum)/sum(var): all {r['noise_independence_all']:.2f}, interacted {r['noise_independence_interacted']}")


if __name__ == "__main__":
    main()
