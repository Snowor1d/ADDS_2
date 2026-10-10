"""A hand-written shuttle policy, against robots off, on the training maps.

    python3 -m validation.shuttle_baseline --out validation/results/shuttle.jsonl

Each robot repeats one trip: walk into the hazard with its signal off, to its
own entry point; switch to guide and lead back out, at walking pace, to the
nearest ground outside the hazard by walking distance; wait there a moment so
followers arrive; go back in. Robots move along the navmesh, not in straight
lines, so walls do not stop them.

Its purpose is a question about the environment rather than about learning:
can robots acting sensibly move the task reward clearly beyond what happens
with them switched off? If this policy cannot, a learned one has little to
find. Every (map, seed) is run under both conditions with the same level and
random seeds, and the comparison is paired.

The policy uses the map and the hazard's extent, which the robots are given
as static knowledge, and nothing about where people are.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# Walking pace while guiding, as a fraction of ROBOT_SPEED_MAX (2 m/s):
# 0.6 is 1.2 m/s, a little under the crowd's 1.5 m/s mean, so followers keep up.
GUIDE_SPEED_FRACTION = 0.6
# Measured in simulation seconds, independent of decision duration.
EXIT_WAIT_SECONDS = 10.0
# Entry points sit this far from the hazard centre, as a share of its radius.
ENTRY_RADIUS_SHARE = 0.5
# Ground counts as outside once its triangle centre is this far past the edge.
EXIT_MARGIN_M = 3.0
ARRIVE_M = 2.5
# No progress over this many decisions: give up the leg and start the next.
STALL_SECONDS = 20.0


def _centre(t) -> Tuple[float, float]:
    return ((t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0)


class ShuttlePolicy:
    """act(obs, record) for run_episode; one state machine per robot."""

    def __init__(self, model, n_robots: int, cfg=None):
        if cfg is None:
            from configs import resolve_config
            cfg = resolve_config(check_data=False)
        self.cfg = cfg
        from sim import robot_action as ra
        self.ra = ra
        self.m = model
        self.zone = model.danger_zone
        self.inside = [t for t in model.pure_mesh
                       if self.zone.signed_distance(*_centre(t)) < -1.0]
        self.entries = [self._entry_mesh(i, n_robots) for i in range(n_robots)]
        self.state = [{"phase": "in", "target": None, "wait_until": 0.0,
                       "best": math.inf, "last_progress": None}
                      for _ in range(n_robots)]

    # ------------------------------------------------------------- targets

    def _entry_mesh(self, i: int, n: int):
        """A walkable triangle inside the hazard near this robot's own
        point on a ring round the centre, so robots spread out."""
        if not self.inside:
            return None
        r = float(getattr(self.zone, "radius", 0.0)
                  or min(getattr(self.zone, "half_w", 0.0),
                         getattr(self.zone, "half_h", 0.0)))
        ang = 2.0 * math.pi * i / max(1, n)
        px = float(self.zone.cx) + ENTRY_RADIUS_SHARE * r * math.cos(ang)
        py = float(self.zone.cy) + ENTRY_RADIUS_SHARE * r * math.sin(ang)
        return min(self.inside, key=lambda t: math.hypot(
            _centre(t)[0] - px, _centre(t)[1] - py))

    def _exit_mesh(self, start):
        """The nearest triangle by walking distance whose centre is
        EXIT_MARGIN_M outside the hazard."""
        import heapq
        if start is None:
            return None
        dist = {start: 0.0}
        heap = [(0.0, 0, start)]
        tie = 1
        while heap:
            d, _, t = heapq.heappop(heap)
            if d > dist.get(t, math.inf):
                continue
            if self.zone.signed_distance(*_centre(t)) >= EXIT_MARGIN_M:
                return t
            ct = _centre(t)
            for nb in self.m.adjacent_mesh.get(t, ()):
                cn = _centre(nb)
                nd = d + math.hypot(cn[0] - ct[0], cn[1] - ct[1])
                if nd < dist.get(nb, math.inf):
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, tie, nb))
                    tie += 1
        return None

    # ---------------------------------------------------------- navigation

    def _heading(self, rb, target) -> Tuple[float, float]:
        """Unit direction to the next point on the navmesh route to
        `target`, or to its centre once in it."""
        from sim.agent import CrowdAgent
        x, y = float(rb.xy[0]), float(rb.xy[1])
        now = self.m.find_mesh((x, y))
        goal = _centre(target)
        if now is not None and now != target:
            nxt = self.m.next_mesh_from_to(now, target)
            if nxt is not None:
                goal = tuple(CrowdAgent._portal_waypoint(now, nxt))
        dx, dy = goal[0] - x, goal[1] - y
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-6 else (0.0, 0.0)

    # ------------------------------------------------------------- acting

    def __call__(self, obs, rec) -> np.ndarray:
        rows = []
        for i, rb in enumerate(self.m.robots):
            rows.append(self._act_one(i, rb))
        return np.stack(rows)

    def _act_one(self, i, rb) -> np.ndarray:
        ra = self.ra
        s = self.state[i]
        x, y = float(rb.xy[0]), float(rb.xy[1])
        now = float(self.m.step_count) * float(self.cfg.AGENT_TIME_STEP)

        if s["phase"] == "wait":
            if now < s["wait_until"]:
                return ra.encode((0.0, 0.0), "guide")
            self._start(s, "in", self.entries[i])

        if s["target"] is None:
            if s["phase"] == "in":
                self._start(s, "in", self.entries[i])
            if s["target"] is None:
                return ra.encode((0.0, 0.0), "off")

        tx, ty = _centre(s["target"])
        d = math.hypot(tx - x, ty - y)
        if d < s["best"] - 0.5:
            s["best"], s["last_progress"] = d, now
        stalled = (s["last_progress"] is not None
                   and now - s["last_progress"] >= STALL_SECONDS)
        arrived = d < ARRIVE_M or (
            self.m.find_mesh((x, y)) == s["target"] and d < 6.0)
        if arrived or stalled:
            if s["phase"] == "in":
                self._start(s, "out", self._exit_mesh(self.m.find_mesh((x, y))))
            else:
                s["phase"], s["wait_until"], s["target"] = (
                    "wait", now + EXIT_WAIT_SECONDS, None)
                return ra.encode((0.0, 0.0), "guide")
            if s["target"] is None:
                return ra.encode((0.0, 0.0), "off")

        return self._command(rb, s["target"],
                             "off" if s["phase"] == "in" else "guide")

    def _command(self, rb, target, mode):
        speed = GUIDE_SPEED_FRACTION if mode == "guide" else 1.0
        if self.cfg.ROBOT_ACTION_MODE == "waypoint":
            tx, ty = _centre(target)
            dx, dy = tx - float(rb.xy[0]), ty - float(rb.xy[1])
            limit = float(self.cfg.ROBOT_WAYPOINT_RANGE_M)
            scale = max(1.0, abs(dx) / limit, abs(dy) / limit)
            move = (2.0 * dx / (limit * scale), 2.0 * dy / (limit * scale))
            return self.ra.encode(move, mode, speed=speed)
        hx, hy = self._heading(rb, target)
        return self.ra.encode((hx * speed, hy * speed), mode)

    @staticmethod
    def _start(s, phase, target):
        s["phase"], s["target"] = phase, target
        s["best"], s["last_progress"] = math.inf, None


# ------------------------------------------------------------------ runner

def run_one(job) -> Dict:
    site, seed, condition, perceptibility, steps = job
    from configs import resolve_config
    from learn.rollout import run_episode
    from learn.zero_shot import EpisodeMetrics
    import sim.model as M
    from validation.behaviour import _level

    cfg = resolve_config(check_data=False)
    level = _level(cfg, site, seed, perceptibility)
    if level is None:
        return {"site": site, "seed": seed, "condition": condition,
                "error": "no level"}
    random.seed(seed)
    np.random.seed(seed)
    model = M.FightingModel(int(level.crowd_size), int(level.width),
                            int(level.height), robot="Q", level=level)
    act = ShuttlePolicy(model, len(model.robots)) if condition == "shuttle" else None
    inside = []
    zone = model.danger_zone

    def on_step(m, k):
        if k % 25 == 0:
            inside.append(sum(
                1 for a in m.crowds if not a.dead and a.type != 3
                and zone.signed_distance(float(a.xy[0]), float(a.xy[1])) < 0))

    tm = EpisodeMetrics().start(model)
    res = run_episode(model, cfg, act, gamma=cfg.gamma(), max_steps=steps,
                      needs_obs=False, on_step=on_step, task_metrics=tm)
    task = res.task or {}
    return {
        "site": site, "seed": seed, "condition": condition,
        "perceptibility": perceptibility, "steps": res.steps,
        "total_reward": res.total_reward,
        "components": res.components,
        "inside_curve": inside,
        "initial_inside": task.get("initial_inside"),
        "hazard_person_steps": task.get("hazard_person_steps"),
        "reentries_after_clear": task.get("reentries_after_clear"),
        "outflows": task.get("outflows"),
        "mode_share_guide": res.mode_share.get("guide"),
        "robot_speed_mean": res.robot_speed_mean,
        "followers_end": sum(1 for a in model.crowds
                             if not a.dead and a.type == 0),
        # How far the robots reached people, over everybody who was ever in
        # the crop (inflow included).
        "people": sum(1 for a in model.crowds if a.type != 3),
        "ever_followed": sum(1 for a in model.crowds if a.type != 3
                             and getattr(a, "is_effected_by_robot", 0)),
        "informed_by_robot": sum(1 for a in model.crowds if a.type != 3
                                 and getattr(a, "informed_by_robot", False)),
        "cued_by_robot": sum(1 for a in model.crowds if a.type != 3
                             and getattr(a, "cue_source", None) == "robot"),
        "behavior_model_version": cfg.BEHAVIOR_MODEL_VERSION,
    }


def summarise(rows: List[Dict]) -> Dict:
    """Paired shuttle - off differences, with a 95% interval."""
    by = {(r["site"], r["seed"], r["condition"]): r for r in rows
          if "error" not in r}
    pairs = [(by[(s, sd, "shuttle")], by[(s, sd, "off")])
             for (s, sd, c) in by if c == "shuttle" and (s, sd, "off") in by]

    def stats(vals):
        a = np.asarray(vals, float)
        n = len(a)
        if n == 0:
            return None
        se = a.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        return {"mean": float(a.mean()), "ci95": float(1.96 * se),
                "n": n, "better": int((a > 0).sum())}

    out = {"pairs": len(pairs)}
    keys = ["total_reward"] + [f"components.{c}" for c in
                               ("person_time", "remaining_path", "reentry",
                                "collision")]
    for k in keys:
        def val(r):
            return r["components"][k.split(".")[1]] if "." in k else r[k]
        out[k] = stats([val(sh) - val(of) for sh, of in pairs])
    # Hazard person-steps: lower is better, so off - shuttle.
    out["hazard_person_steps_reduction"] = stats(
        [of["hazard_person_steps"] - sh["hazard_person_steps"]
         for sh, of in pairs if of["hazard_person_steps"] is not None])
    out["relative_person_time_change"] = stats(
        [(sh["components"]["person_time"] - of["components"]["person_time"])
         / abs(of["components"]["person_time"])
         for sh, of in pairs if of["components"]["person_time"]])
    off_sd = np.std([of["total_reward"] for _, of in pairs], ddof=1)
    out["off_total_reward_sd_across_pairs"] = float(off_sd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="validation/results/shuttle.jsonl")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--sites", nargs="*")
    ap.add_argument("--perceptibility", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    from multiprocessing import Pool
    from configs import resolve_config
    cfg = resolve_config(check_data=False)
    sites = args.sites or list(cfg.DATASET_SITES)
    steps = args.steps or int(cfg.MAX_STEPS)
    jobs = [(s, sd, c, args.perceptibility, steps)
            for s in sites for sd in args.seeds for c in ("off", "shuttle")]
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["site"], r["seed"], r["condition"]))
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    with Pool(args.procs) as pool, open(args.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run_one, todo)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(i + 1, "/", len(todo), r.get("site"), r.get("seed"),
                  r.get("condition"), round(r.get("total_reward", 0.0), 1),
                  flush=True)
    rows = [json.loads(l) for l in open(args.out)]
    summary = summarise(rows)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
