"""A curriculum level and the checks every level must pass.

A level is the map (size, obstacle polygons, exit polygons) plus the crowd
size and, optionally, a fixed initial-placement seed. Geometry is stored in
its own orientation; the D4 augmentation is chosen per episode and applied
by the simulator. Levels pickle and also round-trip through JSON, which is
how suites store generated maps.

`check_level` rejects a level the simulator could not finish: every walkable
1 m cell (with the agents' 0.5 m body kept clear of walls) must connect to an
exit, exits must sit on the boundary without overlapping obstacles, and the
obstacles may not cover more than UED_MAX_OBSTACLE_DENSITY of the map.
"""

from __future__ import annotations

import itertools
import json
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

_ids = itertools.count(1)
_ids_lock = threading.Lock()


def next_level_id() -> int:
    with _ids_lock:
        return next(_ids)


def reserve_level_ids(start: int) -> None:
    """Make new ids start at `start` or above (after restoring a population)."""
    global _ids
    with _ids_lock:
        cur = next(_ids)
        _ids = itertools.count(max(cur, int(start)))


@dataclass
class Level:
    level_id: int
    width: int
    height: int
    obstacles: List[List[List[float]]]
    exits: List[List[Tuple[float, float]]]
    crowd: int
    source: str = "gen"                 # "gen" | "mut"
    seed: Optional[int] = None          # generator seed
    difficulty: Optional[int] = None
    generation: int = 0
    parent_id: Optional[int] = None
    edits: Tuple[str, ...] = ()
    placement_seed: Optional[int] = None
    map_num: Optional[int] = None       # never set for generated levels

    def obstacle_density(self) -> float:
        polys = [Polygon(p).buffer(0) for p in self.obstacles if len(p) >= 3]
        if not polys:
            return 0.0
        area = unary_union(polys).intersection(box(0, 0, self.width, self.height)).area
        return float(area) / float(self.width * self.height)

    def to_json(self) -> dict:
        d = asdict(self)
        d["exits"] = [[list(p) for p in e] for e in self.exits]
        d["edits"] = list(self.edits)
        return d

    @staticmethod
    def from_json(d: dict) -> "Level":
        d = dict(d)
        d["exits"] = [[tuple(p) for p in e] for e in d["exits"]]
        d["edits"] = tuple(d.get("edits", ()))
        return Level(**d)

    def digest(self) -> str:
        import hashlib
        blob = json.dumps([self.width, self.height, self.obstacles,
                           [[list(p) for p in e] for e in self.exits]], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# validity
# ----------------------------------------------------------------------------

def exit_side(poly, W, H) -> Optional[str]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    if min(xs) <= 0:
        return "left"
    if max(xs) >= W:
        return "right"
    if min(ys) <= 0:
        return "bottom"
    if max(ys) >= H:
        return "top"
    return None


def check_level(level: Level, max_density: float = 0.55, body: float = 0.5,
                exit_gap: float = 1.0) -> Tuple[bool, str]:
    W, H = int(level.width), int(level.height)
    if W < 20 or H < 20:
        return False, "map too small"
    if not level.exits:
        return False, "no exit"
    if not 1 <= int(level.crowd) <= 1000:
        return False, "crowd size"
    frame = box(0, 0, W, H)
    exits = []
    for e in level.exits:
        P = Polygon(e)
        if not P.is_valid or P.area <= 0:
            return False, "invalid exit"
        if exit_side(e, W, H) is None:
            return False, "exit not on the boundary"
        if not frame.buffer(1e-6).contains(P):
            return False, "exit outside the map"
        exits.append(P)
    for a, b in itertools.combinations(exits, 2):
        if a.intersects(b):
            return False, "exits overlap"
    polys = []
    for o in level.obstacles:
        if len(o) < 3:
            return False, "degenerate obstacle"
        P = Polygon(o)
        if not P.is_valid:
            P = P.buffer(0)
        if P.is_empty or P.area <= 0:
            return False, "empty obstacle"
        if not frame.buffer(1e-6).contains(P):
            return False, "obstacle outside the map"
        polys.append(P)
    blocked = unary_union(polys) if polys else Polygon()
    ex_union = unary_union(exits)
    if polys and blocked.distance(ex_union) < exit_gap:
        return False, "obstacle touches an exit"
    if level.obstacle_density() > max_density:
        return False, "too dense"

    # Connectivity on a 1 m grid. A cell is walkable when its centre is more
    # than `body` from any obstacle and inside the map margin.
    xs = np.arange(W) + 0.5
    ys = np.arange(H) + 0.5
    X, Y = np.meshgrid(xs, ys)          # [row = y, col = x]
    if polys:
        grown = blocked.buffer(body)
        wall = shapely.contains_xy(grown, X, Y) | shapely.intersects_xy(grown, X, Y)
    else:
        wall = np.zeros((H, W), bool)
    free = ~wall
    exit_cells = shapely.intersects_xy(ex_union.buffer(0.75), X, Y) & free
    if not exit_cells.any():
        return False, "exit unreachable"
    seen = exit_cells.copy()
    q = deque(zip(*np.nonzero(exit_cells)))
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and free[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    stranded = int((free & ~seen).sum())
    if stranded > 0:
        return False, f"{stranded} m2 of floor cannot reach an exit"
    if free.sum() < 0.3 * W * H:
        return False, "too little floor"
    return True, "ok"


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------

class GenerationTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise GenerationTimeout()


def generate_level(seed: int, size_m: int, difficulty: int, crowd: int,
                   timeout_s: float = 20.0, max_density: float = 0.55,
                   placement_seed: Optional[int] = None) -> Optional[Level]:
    """One random_map level, or None when it timed out or failed the check.
    The timeout uses SIGALRM, so call it in a process's main thread (the
    producer processes do)."""
    import random as _random
    import signal
    from random_map import RandomMapSpec, generate_map

    state = _random.getstate()
    use_alarm = timeout_s and threading.current_thread() is threading.main_thread()
    old = None
    if use_alarm:
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        data = generate_map(RandomMapSpec(width=int(size_m), height=int(size_m),
                                          difficulty=int(difficulty), seed=int(seed)))
    except (GenerationTimeout, RuntimeError):
        return None
    finally:
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        _random.setstate(state)  # generate_map reseeds the global generator
    lv = Level(level_id=0, width=int(data.width), height=int(data.height),
               obstacles=[[[int(x), int(y)] for x, y in o] for o in data.obstacles],
               exits=[[(int(x), int(y)) for x, y in e] for e in data.exits],
               crowd=int(crowd), source="gen", seed=int(seed), difficulty=int(difficulty),
               placement_seed=placement_seed)
    ok, _ = check_level(lv, max_density=max_density)
    return lv if ok else None


def seed_is_reserved(seed: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in ranges)
