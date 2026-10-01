"""Behaviour-layer checks, P1-P8 of docs/behavior_model_design.md.

Levels come from the same path training uses (learn.training_maps), so what
is measured is what the policy trains against. Each run is one episode; the
report pools runs across sites, seeds and hazard perceptibility.

    python3 -m validation.behaviour --out validation/results/behaviour.json

Like the rest of validation/, these are measurements, not unit tests: they
take tens of minutes and a failure is a finding.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np

STEPS = 800
PERCEPTIBILITIES = (0.2, 0.3, 0.5, 0.9)   # 0.2: below the sensory floor
SEEDS = (1, 2)
DENSITY_RADIUS_M = 1.5


def _level(cfg, site: str, seed: int, perceptibility: Optional[float]):
    from learn.training_maps import OsmTrainingMaps
    maps = OsmTrainingMaps(cfg)
    key = (site, int(cfg.DATASET_SIZES_M[0]))
    rng = random.Random(seed * 1000 + list(cfg.DATASET_SITES).index(site))
    level = None
    for _ in range(10):
        level = maps.make_level(key, rng)
        if level is not None:
            break
    if level is None:
        return None
    if perceptibility is not None:
        level.perceptibility = float(perceptibility)
    return level


def _local_density(xy: np.ndarray, r: float = DENSITY_RADIUS_M) -> np.ndarray:
    """People per square metre within r of each person, self included."""
    if len(xy) == 0:
        return np.zeros(0)
    cell = {}
    keys = np.floor(xy / r).astype(int)
    for i, (cx, cy) in enumerate(keys):
        cell.setdefault((cx, cy), []).append(i)
    out = np.zeros(len(xy))
    for i, (cx, cy) in enumerate(keys):
        n = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in cell.get((cx + dx, cy + dy), ()):
                    if (xy[i, 0] - xy[j, 0]) ** 2 + (xy[i, 1] - xy[j, 1]) ** 2 <= r * r:
                        n += 1
        out[i] = n / (math.pi * r * r)
    return out


def _scripted_act(model, cfg):
    """Robots walk into the hazard centre guiding, then out along spokes,
    then stop signalling: the pattern that exposes hand-offs and release."""
    from sim import robot_action as ra
    z = model.danger_zone
    cx, cy = float(z.cx), float(z.cy)
    n = len(model.robots)
    spokes = [(cx + 60 * math.cos(2 * math.pi * i / max(1, n)),
               cy + 60 * math.sin(2 * math.pi * i / max(1, n))) for i in range(n)]
    third = STEPS // 3

    def move(rb, tx, ty):
        dx, dy = tx - rb.xy[0], ty - rb.xy[1]
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1.0 else (0.0, 0.0)

    def act(obs, rec):
        k = int(model.step_count)
        rows = []
        for i, rb in enumerate(model.robots):
            if k < third:
                rows.append(ra.encode(move(rb, cx, cy), "guide"))
            elif k < 2 * third:
                tx = min(max(spokes[i][0], 2), model.width - 2)
                ty = min(max(spokes[i][1], 2), model.height - 2)
                rows.append(ra.encode(move(rb, tx, ty), "guide"))
            else:
                rows.append(ra.encode((0.0, 0.0), "off"))
        return np.stack(rows)
    return act


def run_one(job) -> Dict:
    site, seed, perceptibility, condition = job
    from configs import resolve_config
    cfg = resolve_config(check_data=False)
    import sim.agent as A
    import sim.model as M
    from learn.rollout import run_episode

    level = _level(cfg, site, seed, perceptibility)
    if level is None:
        return {"site": site, "seed": seed, "error": "no level"}
    random.seed(seed)
    np.random.seed(seed)
    model = M.FightingModel(int(level.crowd_size), int(level.width),
                            int(level.height), robot="Q", level=level)
    zone = model.danger_zone
    r_a = float(A.HAZARD_MEMORY_RADIUS_M)

    def people():
        return [a for a in model.crowds if not a.dead and a.type != 3]

    cohort = {a.unique_id for a in people()
              if zone.signed_distance(float(a.xy[0]), float(a.xy[1])) < 0}
    n0 = len(cohort)
    cohort_curve, inside_curve = [], []
    premove, prev_aw, prev_in_belief, prev_mem = [], {}, {}, {}
    self_reentry = 0
    reentry_ctx = []
    near = defaultdict(list)
    dens_all, dens_hazard = [], []
    decisions = []

    # P3: log each choice event with the alignment the person saw.
    if condition == "scripted":
        orig_step = A.CrowdAgent._robot_step

        def logged(self, neighbors, perceived, new_ids):
            leader = self.following_robot_id if self.type == 0 else None
            lost = leader is not None and leader not in self._known_robots
            if new_ids or lost:
                opts = [rb for rb in perceived
                        if not self._robot_leads_back_in(rb)]
                al = [self._crowd_alignment(
                    neighbors, self._instruction_direction(rb)) for rb in opts]
                us = [self._robot_utility(rb, neighbors, leader) for rb in opts]
                before = self.following_robot_id if self.type == 0 else None
                orig_step(self, neighbors, perceived, new_ids)
                after = self.following_robot_id if self.type == 0 else None
                decisions.append({
                    "n_options": len(opts), "alignment": al,
                    "p_follow_pred": (sum(math.exp(u) for u in us)
                                      / (1.0 + sum(math.exp(u) for u in us))
                                      if us else 0.0),
                    "followed": after is not None,
                    "switched": (before is not None and after is not None
                                 and after != before),
                    "handoff": lost})
                return
            orig_step(self, neighbors, perceived, new_ids)
        A.CrowdAgent._robot_step = logged

    def on_step(m, k):
        nonlocal self_reentry
        ps = people()
        for a in ps:
            uid = a.unique_id
            st = a.awareness
            if prev_aw.get(uid) == "milling" and st == "acting" and a.cued_at is not None:
                premove.append(((k - a.cued_at) * float(A.AGENT_TIME_STEP),
                                a.cue_source, a.type == 0))
            prev_aw[uid] = st
            x, y = float(a.xy[0]), float(a.xy[1])
            near[uid].append(abs(zone.signed_distance(x, y)) < 3.0)
            if len(near[uid]) > 300:
                near[uid].pop(0)
            # P4: walking back into believed danger by its own choice.
            if a.hazard_memory and st == "acting" and a.type != 0:
                dmin = min(math.hypot(x - hx, y - hy) for hx, hy in a.hazard_memory)
                deep = dmin < r_a - 1.0
                if (deep and prev_in_belief.get(uid) is False
                        and prev_mem.get(uid) == a._memory_version):
                    self_reentry += 1
                    g = a.now_goal or [x, y]
                    reentry_ctx.append({
                        "dwelling": bool(a._dwelling), "intent": a.post_safe_intent,
                        "goal_in_belief": a._believes_dangerous(float(g[0]), float(g[1])),
                        "responding": bool(a.responding),
                        "speed": round(math.hypot(a.vel[0], a.vel[1]), 2)})
                prev_in_belief[uid] = dmin < r_a
                prev_mem[uid] = a._memory_version
        if k % 25 == 0:
            inside_curve.append(sum(zone.signed_distance(float(a.xy[0]), float(a.xy[1])) < 0 for a in ps))
            cohort_curve.append(sum(1 for a in ps if a.unique_id in cohort
                                    and zone.signed_distance(float(a.xy[0]), float(a.xy[1])) < 0))
        if k % 50 == 0 and ps:
            xy = np.array([[float(a.xy[0]), float(a.xy[1])] for a in ps])
            d = _local_density(xy)
            dens_all.extend(d.tolist())
            ins = np.array([zone.signed_distance(p[0], p[1]) < 0 for p in xy])
            dens_hazard.extend(d[ins].tolist())

    act = _scripted_act(model, cfg) if condition == "scripted" else None
    t0 = time.perf_counter()
    res = run_episode(model, cfg, act, gamma=0.99, max_steps=STEPS,
                      on_step=on_step, needs_obs=False)
    wall = time.perf_counter() - t0

    def first_below(frac):
        for i, c in enumerate(cohort_curve):
            if n0 and c <= frac * n0:
                return i * 25
        return None

    alive = people()
    lingering = sum(1 for a in alive
                    if len(near[a.unique_id]) >= 300
                    and sum(near[a.unique_id]) >= 240
                    and not getattr(a, "_dwelling", False))
    waiting = sum(1 for a in alive if getattr(a, "_dwelling", False)
                  and a.hazard_memory and a.awareness == "acting")
    ever_cued = [a for a in model.crowds if a.type != 3 and a.cued_at is not None]
    told_left = sum(1 for a in model.crowds if a.type != 3 and a.dead
                    and a.outflow_reason == "evacuation_departure"
                    and not a.hazard_memory)
    reasons = Counter(a.outflow_reason for a in model.crowds
                      if a.type != 3 and a.dead)
    return {
        "site": site, "seed": seed, "perceptibility": perceptibility,
        "condition": condition, "crowd": int(level.crowd_size),
        "n0_inside": n0, "cohort_curve": cohort_curve,
        "inside_curve": inside_curve,
        "t50": first_below(0.5), "t90": first_below(0.1),
        "cohort_end_frac": (cohort_curve[-1] / n0) if n0 else None,
        "lingering": lingering, "waiting_acting": waiting,
        "self_reentry": self_reentry, "reentry_ctx": reentry_ctx,
        "told_departed": told_left,
        "outflow_reasons": dict(reasons),
        "premovement": premove,
        "nonresponsive_frac": (sum(a.awareness == "nonresponsive" for a in ever_cued)
                               / max(1, len(ever_cued))),
        "awareness_end": model.awareness_counts(),
        "density_p50": float(np.percentile(dens_all, 50)) if dens_all else None,
        "density_p95": float(np.percentile(dens_all, 95)) if dens_all else None,
        "density_p99": float(np.percentile(dens_all, 99)) if dens_all else None,
        "density_max": float(np.max(dens_all)) if dens_all else None,
        "density_hazard_p95": float(np.percentile(dens_hazard, 95)) if dens_hazard else None,
        "sim_ms_per_step": res.sim_ms_per_step, "wall_s": wall,
        "decisions": decisions,
    }


def jobs(sites) -> List:
    out = [(s, seed, p, "none") for s in sites for seed in SEEDS
           for p in PERCEPTIBILITIES]
    out += [(s, 1, 0.5, "scripted") for s in sites]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/results/behaviour.json")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--sites", nargs="*")
    args = ap.parse_args()
    from configs import resolve_config
    cfg = resolve_config(check_data=False)
    sites = args.sites or list(cfg.DATASET_SITES)
    todo = jobs(sites)
    results = []
    with Pool(args.procs) as pool:
        for r in pool.imap_unordered(run_one, todo):
            results.append(r)
            print(r.get("site"), r.get("condition"), r.get("perceptibility"),
                  r.get("seed"), "t50", r.get("t50"), "linger", r.get("lingering"),
                  "reentry", r.get("self_reentry"), flush=True)
    with open(args.out, "w") as f:
        json.dump({"behavior_model_version": cfg.BEHAVIOR_MODEL_VERSION,
                   "steps": STEPS, "runs": results}, f)


if __name__ == "__main__":
    main()
