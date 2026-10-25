"""Hand-written robot controllers the learned policy is compared against.

    no_robot          the model built with robot='N' (RV3 'N'); no controller
    exit_robot        RV3 'T': every step, head for the nearest exit along the
                      navmesh at 1 m/s and stop within 2 m of it. Unchanged,
                      except that a mesh lookup that would raise (a rounded
                      position inside an obstacle triangle) falls back to the
                      nearest walkable triangle instead of crashing.
    shuttle_random    pick one person still inside at random, go to them,
    shuttle_nearest   lead to the nearest exit, wait, repeat; "nearest" picks
                      the person closest to the robot by walking distance.

The shuttles decide every BASELINE_DECISION_INTERVAL steps (the learned
policy's interval) and command a velocity of the tuned speed along the
heading, clipped per axis to [-2, 2] like the policy's action. They move
along the navmesh: the route is the chain of triangles from
next_vertex_matrix, and the robot heads for the farthest point of that chain
it can see in a straight line with its body clear of walls, so it does not
zig-zag between triangle centres.
"""

from __future__ import annotations

import json
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.prepared import prep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _centre(t) -> Tuple[float, float]:
    return ((t[0][0] + t[1][0] + t[2][0]) / 3.0, (t[0][1] + t[1][1] + t[2][1]) / 3.0)


def _clip_velocity(vx: float, vy: float, limit: float = 2.0) -> Tuple[float, float]:
    return max(-limit, min(limit, vx)), max(-limit, min(limit, vy))


class Navigator:
    """Navmesh routing with straight-line shortcuts."""

    def __init__(self, env, clearance: float, lookahead: int = 12):
        self.env = env
        self.lookahead = int(lookahead)
        polys = list(env._obstacle_polys)
        W, H = env.width, env.height
        from shapely.geometry import box
        border = box(-50, -50, W + 50, H + 50).difference(box(0, 0, W, H))
        self.blocked = prep(unary_union(polys + [border]).buffer(max(0.0, clearance)))

    def _mesh(self, xy):
        env = self.env
        key = (int(round(xy[0])), int(round(xy[1])))
        m = env.match_grid_to_mesh.get(key)
        if m is not None and m in env.next_vertex_matrix and m not in env.obstacle_mesh:
            return m
        try:
            return env.crowds[0].choice_safe_mesh(xy) if env.crowds else None
        except Exception:
            best, bd = None, math.inf
            for t in env.pure_mesh:
                c = _centre(t)
                d = (c[0] - xy[0]) ** 2 + (c[1] - xy[1]) ** 2
                if d < bd:
                    best, bd = t, d
            return best

    def visible(self, a, b) -> bool:
        if a[0] == b[0] and a[1] == b[1]:
            return True
        return not self.blocked.intersects(LineString([tuple(a), tuple(b)]))

    def heading(self, xy, goal) -> Tuple[float, float]:
        """Unit vector from xy towards the next point to aim for on the way
        to goal."""
        env = self.env
        aim = None
        if self.visible(xy, goal):
            aim = goal
        else:
            now = self._mesh(xy)
            gm = self._mesh(goal)
            route = []
            if now is not None and gm is not None:
                m = now
                for _ in range(len(env.mesh_list)):
                    nxt = env.next_vertex_matrix.get(m, {}).get(gm)
                    if nxt is None or nxt == m:
                        break
                    route.append(nxt)
                    m = nxt
                    if m == gm:
                        break
            pts = [_centre(t) for t in route[: self.lookahead]]
            for p in reversed(pts):
                if self.visible(xy, p):
                    aim = p
                    break
            if aim is None:
                aim = pts[0] if pts else goal
        dx, dy = aim[0] - xy[0], aim[1] - xy[1]
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return 0.0, 0.0
        return dx / n, dy / n

    def distance(self, a, b) -> float:
        try:
            d = self.env.crowds[0].point_to_point_distance(a, b)
        except Exception:
            d = math.inf
        if not math.isfinite(d) or d > 1e8:
            d = math.hypot(a[0] - b[0], a[1] - b[1]) * 10.0
        return d


class ExitRobotController:
    """RV3 'T' (robot_policy_going_exit), run every step."""

    needs_obs = False
    decision_interval = 1

    def reset(self, env, ob):
        self.nav = Navigator(env, clearance=0.0)

    def act(self, env, ob, stack, state5):
        rb = env.robot
        try:
            rb.robot_policy_going_exit()
            return None  # receive_action was called inside
        except (KeyError, TypeError, IndexError):
            _, q, _ = env.nearest_exit(rb.xy)
            if math.hypot(q[0] - rb.xy[0], q[1] - rb.xy[1]) < 2:
                return np.array([0.0, 0.0])
            ux, uy = self.nav.heading(rb.xy, q)
            return np.array([ux, uy])


def tuned_key(cfg, mode: str) -> str:
    """Tuning results are kept per robot speed limit: "shuttle_random" for no
    limit, "shuttle_random@2.0" under ROBOT_SPEED_NORM_LIMIT = 2.0."""
    lim = cfg.ROBOT_SPEED_NORM_LIMIT
    return f"shuttle_{mode}" if lim is None else f"shuttle_{mode}@{float(lim):g}"


def load_tuned(cfg, mode: str) -> Tuple[float, float, str]:
    """(approach, escort, source) for shuttle_<mode> under the current speed
    limit; the untuned defaults when that limit has not been tuned."""
    path = cfg.SHUTTLE_TUNED_FILE
    if not os.path.isabs(path):
        path = os.path.join(ROOT, path)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        best = data[tuned_key(cfg, mode)]["best"]
        return float(best["approach"]), float(best["escort"]), f"tuned:{os.path.basename(path)}"
    except (OSError, KeyError, ValueError):
        return float(cfg.SHUTTLE_APPROACH_SPEED), float(cfg.SHUTTLE_ESCORT_SPEED), "default"


class ShuttleController:
    """Fetch one person, escort to the nearest exit, wait, repeat."""

    needs_obs = False

    def __init__(self, cfg, mode: str, approach: Optional[float] = None,
                 escort: Optional[float] = None, seed: Optional[int] = None):
        assert mode in ("random", "nearest")
        self.cfg = cfg
        self.mode = mode
        a, e, src = load_tuned(cfg, mode)
        self.approach = float(a if approach is None else approach)
        self.escort = float(e if escort is None else escort)
        self.speed_source = src if (approach is None and escort is None) else "explicit"
        self.decision_interval = int(cfg.BASELINE_DECISION_INTERVAL)
        self.seed = seed

    def reset(self, env, ob):
        self.rng = random.Random(self.seed)
        self.nav = Navigator(env, clearance=float(self.cfg.ROBOT_BODY_RADIUS))
        self.phase = "select"
        self.target = None
        self.exit_goal = None
        self.timer = 0
        self.trips = 0

    # -- choices -----------------------------------------------------------

    def _candidates(self, env) -> List:
        near = float(self.cfg.SHUTTLE_IGNORE_NEAR_EXIT_M)
        alive = [a for a in env.crowds if not a.dead
                 and (near <= 0 or env.distance_to_exit(a.xy) > near)]
        if self.cfg.SHUTTLE_TARGET_FILTER == "uninformed":
            sel = [a for a in alive if a.exit_belief is None and a.type != 0]
            return sel or alive
        return alive

    def _select(self, env):
        cands = self._candidates(env)
        if not cands:
            return None
        if self.mode == "random":
            return self.rng.choice(cands)
        rb = env.robot.xy
        if self.cfg.SHUTTLE_DISTANCE == "euclidean":
            key = lambda a: math.hypot(a.xy[0] - rb[0], a.xy[1] - rb[1])
        else:
            key = lambda a: self.nav.distance(rb, a.xy)
        return min(cands, key=key)

    def _nearest_exit_point(self, env, xy):
        best, bd = None, math.inf
        for i in range(len(env._exit_polys)):
            q, d0 = env.nearest_point_on_exit(i, xy)
            d = 0.0 if d0 == 0 else self.nav.distance(xy, q)
            if d < bd:
                best, bd = q, d
        return best

    # -- control -----------------------------------------------------------

    def _move(self, env, goal, speed, stop_within: float = 0.0):
        """Head for goal at `speed`, slowing so as not to pass within one
        decision the point `stop_within` short of it (at 2.8 m/s a 2 s
        decision covers 5.6 m, more than the pickup diameter)."""
        rb = env.robot.xy
        ux, uy = self.nav.heading(rb, goal)
        horizon = self.decision_interval * float(self.cfg.ROBOT_TIME_STEP)
        gap = max(0.0, math.hypot(goal[0] - rb[0], goal[1] - rb[1]) - stop_within)
        v = min(float(speed), gap / horizon) if horizon > 0 else float(speed)
        vx, vy = _clip_velocity(v * ux, v * uy)
        return np.array([vx, vy])

    def act(self, env, ob, stack, state5):
        rb = env.robot.xy
        c = self.cfg
        D = self.decision_interval
        if self.phase == "wait":
            self.timer -= D
            if self.timer > 0:
                return np.array([0.0, 0.0])
            self.phase = "select"
        if self.phase == "select":
            self.target = self._select(env)
            if self.target is None:
                # Everyone left is at an exit: step out of the way.
                centre = (env.width / 2.0, env.height / 2.0)
                if env.distance_to_exit(rb) < float(c.SHUTTLE_IGNORE_NEAR_EXIT_M) + 3.0:
                    return self._move(env, centre, self.approach)
                return np.array([0.0, 0.0])
            self.phase = "approach"
            self.timer = 0
        if self.phase == "approach":
            t = self.target
            self.timer += D
            if t is None or t.dead or self.timer > int(c.SHUTTLE_APPROACH_TIMEOUT_STEPS):
                self.phase = "select"
                return self.act(env, ob, stack, state5)
            if math.hypot(t.xy[0] - rb[0], t.xy[1] - rb[1]) <= float(c.SHUTTLE_PICKUP_RADIUS_M):
                self.phase = "escort"
                self.exit_goal = self._nearest_exit_point(env, rb)
                self.trips += 1
            else:
                return self._move(env, t.xy, self.approach,
                                  stop_within=0.5 * float(c.SHUTTLE_PICKUP_RADIUS_M))
        if self.phase == "escort":
            g = self.exit_goal
            if g is None or math.hypot(g[0] - rb[0], g[1] - rb[1]) <= float(c.SHUTTLE_EXIT_REACH_M):
                self.phase = "wait"
                self.timer = int(c.SHUTTLE_EXIT_WAIT_STEPS)
                return np.array([0.0, 0.0])
            return self._move(env, g, self.escort,
                              stop_within=0.5 * float(c.SHUTTLE_EXIT_REACH_M))
        return np.array([0.0, 0.0])


def make_controller(condition: str, cfg, seed: Optional[int] = None, **kw):
    """Controller and the robot version the model must be built with."""
    if condition == "no_robot":
        return None, "N"
    if condition == "exit_robot":
        return ExitRobotController(), "Q"
    if condition == "shuttle_random":
        return ShuttleController(cfg, "random", seed=seed, **kw), "Q"
    if condition == "shuttle_nearest":
        return ShuttleController(cfg, "nearest", seed=seed, **kw), "Q"
    raise ValueError(f"unknown baseline {condition!r}")
