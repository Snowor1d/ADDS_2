"""Statistics for comparing conditions on a suite.

Scenarios of one map are not independent (same layout), so uncertainty is a
cluster bootstrap over maps: resample maps with replacement, average each
map's scenarios, average the maps. Comparisons are paired by scenario
(same layout and starting crowd).

Reported per condition: mean, median, IQM (interquartile mean, Agarwal et
al. 2021), success rate, and the 95% bootstrap interval of the mean. Per pair
(A against B): mean difference A - B with its interval, the share of
scenarios A finishes faster (ties count half), the geometric mean of A / B,
and the probability of improvement with its interval.

Evacuation time of an unfinished episode is the step cap, which understates
how bad it was; the success rate is reported beside it for that reason.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

METRIC = "evac_time_100"


def _by(results, cond, metric=METRIC) -> Dict[str, float]:
    return {r["id"]: float(r[metric]) for r in results
            if r.get("condition") == cond and "error" not in r and metric in r}


def _maps(results) -> Dict[str, str]:
    return {r["id"]: r["map_key"] for r in results if "map_key" in r}


def iqm(x: Sequence[float]) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def _cluster_boot(values: Dict[str, float], id2map: Dict[str, str], B: int, rng,
                  stat=np.mean) -> np.ndarray:
    groups = defaultdict(list)
    for i, v in values.items():
        groups[id2map[i]].append(v)
    keys = sorted(groups)
    per = np.array([np.mean(groups[k]) for k in keys])
    if len(per) == 0:
        return np.array([np.nan])
    idx = rng.integers(0, len(per), size=(B, len(per)))
    return stat(per[idx], axis=1)


def summarize(results: List[dict], cond: str, B: int = 5000, ci: float = 0.95,
              seed: int = 0) -> Dict[str, float]:
    vals = _by(results, cond)
    if not vals:
        return {}
    succ = _by(results, cond, "success")
    x = np.array(list(vals.values()))
    rng = np.random.default_rng(seed)
    boot = _cluster_boot(vals, _maps(results), B, rng)
    a = (1 - ci) / 2
    return {"n": float(len(x)), "n_maps": float(len({_maps(results)[i] for i in vals})),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "iqm": iqm(x),
            "success_rate": float(np.mean(list(succ.values()))) if succ else float("nan"),
            "ci_low": float(np.quantile(boot, a)), "ci_high": float(np.quantile(boot, 1 - a))}


def paired(results: List[dict], cond_a: str, cond_b: str, B: int = 5000, ci: float = 0.95,
           seed: int = 0) -> Dict[str, float]:
    va, vb = _by(results, cond_a), _by(results, cond_b)
    ids = sorted(set(va) & set(vb))
    if not ids:
        return {}
    id2map = _maps(results)
    d = {i: va[i] - vb[i] for i in ids}
    win = {i: 1.0 if va[i] < vb[i] else (0.5 if va[i] == vb[i] else 0.0) for i in ids}
    ratio = {i: np.log(max(va[i], 1.0) / max(vb[i], 1.0)) for i in ids}
    rng = np.random.default_rng(seed)
    a = (1 - ci) / 2
    bd = _cluster_boot(d, id2map, B, rng)
    bw = _cluster_boot(win, id2map, B, rng)
    br = _cluster_boot(ratio, id2map, B, rng)
    return {"n": float(len(ids)),
            "mean_diff": float(np.mean(list(d.values()))),
            "diff_ci_low": float(np.quantile(bd, a)), "diff_ci_high": float(np.quantile(bd, 1 - a)),
            "win_rate": float(np.mean(list(win.values()))),
            "win_ci_low": float(np.quantile(bw, a)), "win_ci_high": float(np.quantile(bw, 1 - a)),
            "ratio_gmean": float(np.exp(np.mean(list(ratio.values())))),
            "ratio_ci_low": float(np.exp(np.quantile(br, a))),
            "ratio_ci_high": float(np.exp(np.quantile(br, 1 - a)))}


def best_of(results: List[dict], conds: Sequence[str], name: str) -> List[dict]:
    """A virtual condition: per scenario, the best (lowest time) of `conds`
    ("the best baseline on each map", an oracle choice)."""
    out = []
    by = defaultdict(list)
    for r in results:
        if r.get("condition") in conds and "error" not in r:
            by[r["id"]].append(r)
    for i, rs in by.items():
        if len(rs) == len(conds):
            b = min(rs, key=lambda r: r[METRIC])
            out.append({**b, "condition": name, "best_of": b["condition"]})
    return out


def by_group(results: List[dict]) -> Dict[str, List[dict]]:
    g = defaultdict(list)
    for r in results:
        g[r.get("group", "?")].append(r)
    return dict(g)
