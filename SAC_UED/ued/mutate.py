"""Level editing, the part of ACCEL that makes it more than prioritized replay.

Random generation explores the level space broadly but shallowly. Editing works
the neighbourhood of levels the agent is already on the edge of solving, so the
population's difficulty tracks the policy as it improves.

Every validity rule here is the one `random_map.generate_map` already enforces;
nothing is reimplemented. Connectivity in particular is not optional: a
pedestrian spawned in a region that cannot reach an exit never evacuates, and
the episode burns all MAX_STEPS steps. Edits are the easiest way to create that.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from shapely.affinity import rotate as _sh_rotate
from shapely.affinity import scale as _sh_scale
from shapely.affinity import translate as _sh_translate
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from config import (
    UED_CROWD_RANGE,
    UED_MUTATE_MAX_TRIES,
    UED_MUTATIONS_PER_CHILD,
)
from random_map import (
    _clip_to_bounds,
    _has_enclosed_free_pocket_grid,
    _has_exit_disconnected_free_region_grid,
    _params_from_difficulty,
    _passes_exit_rules,
    _passes_wall_rules_strict_by_side,
    _pick_large_obstacle,
    _pick_small_obstacle,
    _poly_to_points_int,
    _random_exit_on_side,
    _too_close_or_intersect,
    _valid_polygon,
)
from ued.level import Level

# Grid resolution for the two flood-fill checks. These dominate the cost of
# validating a child, so the value is a deliberate speed/precision trade.
VALIDATION_GRID_STEP = 2

# Minimum clearance an inherited or edited obstacle must leave around an exit.
# Chosen to match the closest spacing the generator itself produces (its exit
# flanks land 3-4 units out) and to stay well clear of the robot's diameter.
EXIT_CLEARANCE_FLOOR = 3.0

_DEFAULT_DIFFICULTY = 3


class MutationFailed(RuntimeError):
    """Raised when no valid child was produced within the retry budget."""


# ---------------------------------------------------------------------------
# geometry conversion
# ---------------------------------------------------------------------------

def _to_polys(rings: Sequence[Sequence[Sequence[float]]]) -> List[Polygon]:
    out = []
    for ring in rings:
        if len(ring) < 3:
            continue
        poly = Polygon([(float(p[0]), float(p[1])) for p in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.geom_type != "Polygon" or poly.is_empty:
                continue
        out.append(poly)
    return out


def _exit_points(poly: Polygon) -> List[Tuple[int, int]]:
    return [(int(x), int(y)) for x, y in _poly_to_points_int(poly)]


# ---------------------------------------------------------------------------
# individual edits
# ---------------------------------------------------------------------------

def _op_translate(obstacles: List[Polygon], W: int, H: int, rng: random.Random) -> Optional[str]:
    if not obstacles:
        return None
    i = rng.randrange(len(obstacles))
    span = 0.15 * min(W, H)
    dx = rng.uniform(-span, span)
    dy = rng.uniform(-span, span)
    obstacles[i] = _sh_translate(obstacles[i], xoff=dx, yoff=dy)
    return "translate"


def _op_rotate(obstacles: List[Polygon], W: int, H: int, rng: random.Random) -> Optional[str]:
    if not obstacles:
        return None
    i = rng.randrange(len(obstacles))
    angle = rng.choice([90.0, 180.0, 270.0, rng.uniform(-45.0, 45.0)])
    obstacles[i] = _sh_rotate(obstacles[i], angle, origin="centroid")
    return "rotate"


def _op_resize(obstacles: List[Polygon], W: int, H: int, rng: random.Random) -> Optional[str]:
    if not obstacles:
        return None
    i = rng.randrange(len(obstacles))
    # Kept narrow on purpose. A wide shrink range makes a single op able to
    # halve an obstacle's area, and since shrinking almost always validates
    # while growing often does not, that alone drives the population toward
    # empty maps regardless of what the score function prefers.
    fx = rng.uniform(0.85, 1.35)
    fy = rng.uniform(0.85, 1.35)
    obstacles[i] = _sh_scale(obstacles[i], xfact=fx, yfact=fy, origin="centroid")
    return "resize"


# Deleting always validates, so an unguarded delete competes unfairly with an
# add that has to find free space. Keep a floor on how sparse a level can get.
MIN_OBSTACLES = 3


def _op_delete(obstacles: List[Polygon], W: int, H: int, rng: random.Random) -> Optional[str]:
    if len(obstacles) <= MIN_OBSTACLES:
        return None
    obstacles.pop(rng.randrange(len(obstacles)))
    return "delete"


# How hard `add` tries to find a legal placement. Without this the operator
# fails far more often than delete and the two do not balance.
ADD_PLACEMENT_TRIES = 40


def _op_add(
    obstacles: List[Polygon],
    exits: List[Polygon],
    W: int,
    H: int,
    rng: random.Random,
    params,
) -> Optional[str]:
    """Place one new obstacle, rejecting illegal placements locally.

    Reuses the generator's own shape samplers so bred obstacles come from the
    same family as generated ones, and applies the same spacing rules the quota
    loop applies, so a rejected placement costs one resample rather than the
    whole child.
    """
    min_gap = float(params.get("min_obstacle_gap", 5.0))
    keep_gap = float(params.get("keep_gap_from_exits", 0.0))
    wall_clearance = float(params.get("wall_clearance", 7.0))

    for _ in range(ADD_PLACEMENT_TRIES):
        if rng.random() < 0.3:
            main, extras, _tag = _pick_large_obstacle(W, H, params)
        else:
            main, extras, _tag = _pick_small_obstacle(W, H, params)
        if main is None:
            continue

        merged = unary_union([c for c in [main] + list(extras or []) if c is not None])
        if merged.is_empty or merged.geom_type != "Polygon":
            continue
        cand = _clip_to_bounds(merged, W, H)
        if cand is None or not _valid_polygon(cand):
            continue
        if not _passes_exit_rules(cand, exits, keep_gap):
            continue
        if not _passes_wall_rules_strict_by_side(cand, "generic", W, H, wall_clearance):
            continue
        if _too_close_or_intersect(cand, obstacles, min_gap):
            continue

        obstacles.append(cand)
        return "add"
    return None


def _op_move_exit(exits: List[Polygon], W: int, H: int, rng: random.Random, params) -> Optional[str]:
    if not exits:
        return None
    i = rng.randrange(len(exits))
    side = rng.choice(["left", "right", "bottom", "top"])
    exits[i] = _random_exit_on_side(
        W, H, side,
        along_min=int(params["exit_along_min"]),
        along_max=int(params["exit_along_max"]),
        depth_min=int(params["exit_depth_min"]),
        depth_max=int(params["exit_depth_max"]),
    )
    return "move_exit"


def _op_resize_exit(exits: List[Polygon], W: int, H: int, rng: random.Random, params) -> Optional[str]:
    if not exits:
        return None
    i = rng.randrange(len(exits))
    e = exits[i]
    minx, miny, maxx, maxy = e.bounds
    # Grow or shrink along the wall only, so the exit stays flush against it.
    horizontal = (maxx - minx) >= (maxy - miny)
    factor = rng.uniform(0.7, 1.4)
    if horizontal:
        exits[i] = _sh_scale(e, xfact=factor, yfact=1.0, origin="center")
    else:
        exits[i] = _sh_scale(e, xfact=1.0, yfact=factor, origin="center")
    clipped = _clip_to_bounds(exits[i], W, H)
    if clipped is None:
        return None
    exits[i] = clipped
    return "resize_exit"


def _pick_ops(rng: random.Random, n: int) -> List[str]:
    names = ["translate", "rotate", "resize", "delete", "add", "move_exit", "resize_exit", "crowd"]
    weights = [0.20, 0.10, 0.16, 0.08, 0.26, 0.06, 0.06, 0.08]
    return rng.choices(names, weights=weights, k=n)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _validate(
    obstacles: List[Polygon],
    exits: List[Polygon],
    W: int,
    H: int,
    params,
    fresh: Optional[set] = None,
) -> bool:
    """Apply generate_map's rule set to an edited level.

    `fresh` holds the indices of obstacles this mutation added. The exit gap in
    the difficulty table applies only to obstacles the generator places through
    its quota loop; the exit flanks it seeds the loop with sit deliberately 3-4
    units from an exit. Inheriting an exit flank must not make a level invalid,
    so edited and inherited obstacles only have to leave the exit mouth
    passable, while newly bred ones face the full rule.
    """
    if not exits:
        return False

    fresh = fresh or set()
    min_gap = float(params.get("min_obstacle_gap", 5.0))
    keep_gap_from_exits = float(params.get("keep_gap_from_exits", 0.0))
    wall_clearance = float(params.get("wall_clearance", 7.0))

    cleaned: List[Polygon] = []
    for i, poly in enumerate(obstacles):
        clipped = _clip_to_bounds(poly, W, H)
        if clipped is None or not _valid_polygon(clipped):
            return False
        gap = keep_gap_from_exits if i in fresh else EXIT_CLEARANCE_FLOOR
        if not _passes_exit_rules(clipped, exits, gap):
            return False
        if not _passes_wall_rules_strict_by_side(clipped, "generic", W, H, wall_clearance):
            return False
        if _too_close_or_intersect(clipped, cleaned, min_gap):
            return False
        cleaned.append(clipped)

    obstacles[:] = cleaned

    # Exits must stay flush against the boundary and apart from each other.
    boundary = box(0, 0, W, H).boundary
    min_exit_distance = float(params.get("min_exit_distance", 0.0))
    for i, e in enumerate(exits):
        if not _valid_polygon(e, min_area=4.0):
            return False
        if e.distance(boundary) > 1e-6:
            return False
        for other in exits[i + 1:]:
            if e.intersects(other) or e.distance(other) < min_exit_distance:
                return False

    # Global reachability. The blocked union is obstacles only, matching how
    # generate_map calls these; adding a ring for the outer walls would change
    # the free-space topology and report pockets that are not there.
    #
    # An obstacle-free level is the difficulty-0 tier, not a failure: an open
    # room with boundary exits is trivially reachable everywhere, and it is the
    # level ACCEL is meant to start from and add complexity to.
    if not obstacles:
        return True
    blocked = unary_union(list(obstacles))
    if _has_exit_disconnected_free_region_grid(
        W, H, blocked, exits, grid_step=VALIDATION_GRID_STEP, pinch_cells=1
    ):
        return False
    if _has_enclosed_free_pocket_grid(
        W, H, blocked, grid_step=VALIDATION_GRID_STEP, pinch_cells=1
    ):
        return False
    return True


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def mutate_level(
    parent: Level,
    rng: Optional[random.Random] = None,
    n_ops: Optional[int] = None,
    max_tries: int = UED_MUTATE_MAX_TRIES,
) -> Level:
    """Breed one valid child from `parent`.

    Raises MutationFailed if the retry budget is exhausted, which the caller
    should treat as "skip breeding this episode" rather than as an error.
    """
    rng = rng or random
    W, H = int(parent.width), int(parent.height)
    params = _params_from_difficulty(
        W, H, parent.difficulty if parent.difficulty is not None else _DEFAULT_DIFFICULTY
    )

    lo_ops, hi_ops = UED_MUTATIONS_PER_CHILD
    crowd_lo, crowd_hi = UED_CROWD_RANGE

    for _ in range(max_tries):
        obstacles = _to_polys(parent.obstacles)
        exits = _to_polys(parent.exits)
        crowd = int(parent.crowd_size)
        applied: List[str] = []
        # Tracked by object identity: the edit operators rebuild the polygon
        # they touch, so an edited inherited obstacle is correctly not "fresh".
        fresh_ids = set()

        k = n_ops if n_ops is not None else rng.randint(lo_ops, hi_ops)
        for op in _pick_ops(rng, k):
            if op == "translate":
                tag = _op_translate(obstacles, W, H, rng)
            elif op == "rotate":
                tag = _op_rotate(obstacles, W, H, rng)
            elif op == "resize":
                tag = _op_resize(obstacles, W, H, rng)
            elif op == "delete":
                tag = _op_delete(obstacles, W, H, rng)
            elif op == "add":
                tag = _op_add(obstacles, exits, W, H, rng, params)
                if tag:
                    fresh_ids.add(id(obstacles[-1]))
            elif op == "move_exit":
                tag = _op_move_exit(exits, W, H, rng, params)
            elif op == "resize_exit":
                tag = _op_resize_exit(exits, W, H, rng, params)
            else:
                crowd = max(crowd_lo, min(crowd_hi, crowd + rng.choice([-4, -2, 2, 4])))
                tag = "crowd"
            if tag:
                applied.append(tag)

        if not applied:
            continue
        fresh = {i for i, p in enumerate(obstacles) if id(p) in fresh_ids}
        if not _validate(obstacles, exits, W, H, params, fresh=fresh):
            continue

        return parent.child(
            obstacles=[_poly_to_points_int(p) for p in obstacles],
            exits=[_exit_points(e) for e in exits],
            crowd_size=crowd,
            ops=applied,
        )

    raise MutationFailed(
        f"no valid child for level {parent.level_id} after {max_tries} tries"
    )


def is_playable(level: Level) -> bool:
    """Validate a level that did not come from `mutate_level` (tests, loading)."""
    W, H = int(level.width), int(level.height)
    params = _params_from_difficulty(
        W, H, level.difficulty if level.difficulty is not None else _DEFAULT_DIFFICULTY
    )
    return _validate(_to_polys(level.obstacles), _to_polys(level.exits), W, H, params)
