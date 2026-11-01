"""Waypoint actions (ROBOT_ACTION_MODE = "waypoint").

The action is three numbers in [-2, 2]: a target offset per axis
(+-ROBOT_WAYPOINT_RANGE_M, from where the robot stands when it decides) and a
speed ([0, 1] of ROBOT_WAYPOINT_SPEED_MAX). A target inside an obstacle or
outside the map is moved to the nearest walkable point. Every simulation step
the follower commands a velocity toward the target along the navmesh with the
shuttle baselines' Navigator (straight when the body-clear line is free,
otherwise toward the farthest visible point of the triangle route), slowing so
as not to pass the target within a step, and stops on arrival. The simulator
is unchanged: the follower only issues the per-step velocity commands a
velocity policy would.

A decision is held until an ACTION_SCALE boundary (2 s) at which
    - ROBOT_DECISION_MAX_S_WAYPOINT has passed, or
    - with ROBOT_DECISION_ON_EVENTS_WAYPOINT: the robot has arrived, or moved
      less than half of what it could have in the last 2 s (blocked by a wall
      or the crowd), or
    - the commanded speed is (almost) zero: standing still is a 2 s decision,
      so the policy can wait without committing for long,
or the episode ends.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

STAND_M = 0.5        # a decision that would move less than this in 2 s is a wait


def decode(action, cfg) -> Tuple[float, float, float]:
    """(dx m, dy m, speed m/s) of a 3-number action."""
    a = np.clip(np.asarray(action, dtype=np.float64)[:3], -2.0, 2.0)
    r = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    speed = (a[2] + 2.0) / 4.0 * float(cfg.ROBOT_WAYPOINT_SPEED_MAX)
    return a[0] / 2.0 * r, a[1] / 2.0 * r, speed


class WaypointFollower:
    """One per episode: holds the current target and turns it into per-step
    velocity commands."""

    def __init__(self, env, cfg):
        from shapely.geometry import box
        from shapely.ops import unary_union
        from evaluation.baselines import Navigator

        self.env = env
        self.cfg = cfg
        clearance = float(cfg.ROBOT_BODY_RADIUS)
        self.nav = Navigator(env, clearance=clearance)
        self.dt = float(cfg.ROBOT_TIME_STEP)
        self.tick_s = int(cfg.ACTION_SCALE) * float(cfg.AGENT_TIME_STEP)
        self.arrival = float(cfg.ROBOT_WAYPOINT_ARRIVAL_M)
        self.on_events = bool(cfg.ROBOT_DECISION_ON_EVENTS_WAYPOINT)
        self.max_ticks = int(cfg.decision_max_ticks())
        # Where a target may be: where the robot's body can actually stand.
        # The simulator's outer frame is at 0 and at W-1 / H-1 (not W / H), and
        # wall repulsion keeps the body about half a metre further off than its
        # radius, so targets keep clearance + 0.5 m from both; shrunk a little
        # more so a projected target is strictly inside (a point on the
        # boundary is never "visible" to the Navigator).
        m = clearance + 0.5
        free = box(m, m, float(env.width) - 1.0 - m, float(env.height) - 1.0 - m)
        polys = list(env._obstacle_polys)
        if polys:
            free = free.difference(unary_union(polys).buffer(m))
        inner = free.buffer(-0.2)
        self.free = inner if not inner.is_empty else free
        self.target = None
        self.requested = None
        self.speed = 0.0
        self.ticks = 0
        self._tick_xy = None
        self._tick_gap = 0.0

    # ------------------------------------------------------------ decisions

    def _project(self, x: float, y: float) -> Tuple[float, float]:
        from shapely.geometry import Point
        from shapely.ops import nearest_points
        p = Point(x, y)
        if self.free.is_empty or self.free.contains(p):
            return float(x), float(y)
        q = nearest_points(self.free, p)[0]
        return float(q.x), float(q.y)

    def set(self, action) -> None:
        """Start a decision."""
        dx, dy, speed = decode(action, self.cfg)
        x, y = float(self.env.robot.xy[0]), float(self.env.robot.xy[1])
        self.requested = (x + dx, y + dy)
        self.target = self._project(*self.requested)
        self.speed = float(speed)
        self.ticks = 0
        self._tick_xy = (x, y)
        self._tick_gap = self.gap()

    def gap(self) -> float:
        if self.target is None:
            return 0.0
        x, y = self.env.robot.xy[0], self.env.robot.xy[1]
        return math.hypot(self.target[0] - x, self.target[1] - y)

    def arrived(self) -> bool:
        return self.gap() < self.arrival

    # ---------------------------------------------------------------- steps

    def command(self) -> Tuple[float, float]:
        """Velocity for the next simulation step."""
        if self.target is None or self.speed <= 1e-6:
            return 0.0, 0.0
        gap = self.gap()
        if gap < self.arrival:
            return 0.0, 0.0
        x, y = float(self.env.robot.xy[0]), float(self.env.robot.xy[1])
        ux, uy = self.nav.heading((x, y), self.target)
        v = min(self.speed, gap / self.dt)
        vx = max(-2.0, min(2.0, v * ux))
        vy = max(-2.0, min(2.0, v * uy))
        return vx, vy

    def end_of_tick(self, episode_over: bool) -> bool:
        """Call at every ACTION_SCALE boundary; True if the decision ends here."""
        self.ticks += 1
        x, y = float(self.env.robot.xy[0]), float(self.env.robot.xy[1])
        moved = math.hypot(x - self._tick_xy[0], y - self._tick_xy[1])
        could = min(self.speed * self.tick_s, self._tick_gap)
        self._tick_xy = (x, y)
        self._tick_gap = self.gap()
        if episode_over or self.ticks >= self.max_ticks:
            return True
        if self.speed * self.tick_s < STAND_M:
            return True
        if self.on_events:
            if self.arrived():
                return True
            if could >= STAND_M and moved < 0.5 * could:
                return True
        return False
