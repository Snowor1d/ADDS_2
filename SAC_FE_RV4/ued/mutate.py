"""ACCEL edits: small, local changes to a level that keep it valid.

    move_obstacle    translate one obstacle by UED_MUTATE_MOVE_M
    scale_obstacle   scale one obstacle about its centroid by UED_MUTATE_SCALE
    add_obstacle     add an axis-aligned block, sides UED_MUTATE_ADD_SIZE_M
    remove_obstacle  delete one obstacle (never the last)
    move_exit        slide one exit along its wall by UED_MUTATE_EXIT_SHIFT_M
    resize_exit      set one exit's width to UED_MUTATE_EXIT_ALONG_M
    crowd            crowd size +/- UED_MUTATE_CROWD_STEP within UED_CROWD_LIMITS

A child gets UED_MUTATIONS_PER_CHILD edits; every intermediate result must
pass ued.level.check_level, so an edit that strands floor, blocks an exit or
overfills the map is redrawn. Coordinates stay integral like the stored maps.
"""

from __future__ import annotations

import copy
import math
import random
from typing import List, Optional, Tuple

from shapely import affinity
from shapely.geometry import Polygon, box

from ued.level import Level, check_level, exit_side, next_level_id


class MutationFailed(Exception):
    pass


def _poly_pts(P: Polygon) -> List[List[int]]:
    coords = list(P.exterior.coords)[:-1]
    out = []
    for x, y in coords:
        p = [int(round(x)), int(round(y))]
        if not out or out[-1] != p:
            out.append(p)
    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()
    return out


def _inside(P: Polygon, W: int, H: int) -> bool:
    return box(0, 0, W, H).buffer(1e-6).contains(P)


def _move_obstacle(lv: Level, rng: random.Random, cfg) -> bool:
    if not lv.obstacles:
        return False
    i = rng.randrange(len(lv.obstacles))
    d = rng.uniform(*cfg.UED_MUTATE_MOVE_M)
    a = rng.uniform(0, 2 * math.pi)
    P = affinity.translate(Polygon(lv.obstacles[i]), d * math.cos(a), d * math.sin(a))
    if not _inside(P, lv.width, lv.height):
        return False
    lv.obstacles[i] = _poly_pts(P)
    return len(lv.obstacles[i]) >= 3


def _scale_obstacle(lv: Level, rng: random.Random, cfg) -> bool:
    if not lv.obstacles:
        return False
    i = rng.randrange(len(lv.obstacles))
    s = rng.uniform(*cfg.UED_MUTATE_SCALE)
    P = affinity.scale(Polygon(lv.obstacles[i]), s, s, origin="centroid")
    if not _inside(P, lv.width, lv.height) or P.area < 4:
        return False
    lv.obstacles[i] = _poly_pts(P)
    return len(lv.obstacles[i]) >= 3


def _add_obstacle(lv: Level, rng: random.Random, cfg) -> bool:
    lo, hi = cfg.UED_MUTATE_ADD_SIZE_M
    w = rng.randint(int(lo), int(hi))
    h = rng.randint(int(lo), int(hi))
    if w >= lv.width - 2 or h >= lv.height - 2:
        return False
    x0 = rng.randint(1, lv.width - w - 1)
    y0 = rng.randint(1, lv.height - h - 1)
    lv.obstacles.append([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    return True


def _remove_obstacle(lv: Level, rng: random.Random, cfg) -> bool:
    if len(lv.obstacles) < 2:
        return False
    lv.obstacles.pop(rng.randrange(len(lv.obstacles)))
    return True


def _exit_rect(e, W, H):
    side = exit_side(e, W, H)
    xs = [p[0] for p in e]
    ys = [p[1] for p in e]
    return side, min(xs), min(ys), max(xs), max(ys)


def _set_exit(lv: Level, i: int, side: str, start: int, along: int, depth: int) -> bool:
    W, H = lv.width, lv.height
    if side in ("left", "right"):
        if start < 0 or start + along > H:
            return False
        x0, x1 = (0, depth) if side == "left" else (W - depth, W)
        lv.exits[i] = [(x0, start), (x1, start), (x1, start + along), (x0, start + along)]
    else:
        if start < 0 or start + along > W:
            return False
        y0, y1 = (0, depth) if side == "bottom" else (H - depth, H)
        lv.exits[i] = [(start, y0), (start + along, y0), (start + along, y1), (start, y1)]
    return True


def _move_exit(lv: Level, rng: random.Random, cfg) -> bool:
    i = rng.randrange(len(lv.exits))
    side, x0, y0, x1, y1 = _exit_rect(lv.exits[i], lv.width, lv.height)
    if side is None:
        return False
    shift = int(round(rng.uniform(*cfg.UED_MUTATE_EXIT_SHIFT_M))) * rng.choice((-1, 1))
    if side in ("left", "right"):
        return _set_exit(lv, i, side, int(y0) + shift, int(y1 - y0), int(x1 - x0))
    return _set_exit(lv, i, side, int(x0) + shift, int(x1 - x0), int(y1 - y0))


def _resize_exit(lv: Level, rng: random.Random, cfg) -> bool:
    i = rng.randrange(len(lv.exits))
    side, x0, y0, x1, y1 = _exit_rect(lv.exits[i], lv.width, lv.height)
    if side is None:
        return False
    along = rng.randint(*[int(v) for v in cfg.UED_MUTATE_EXIT_ALONG_M])
    if side in ("left", "right"):
        c = (y0 + y1) / 2.0
        return _set_exit(lv, i, side, int(round(c - along / 2)), along, int(x1 - x0))
    c = (x0 + x1) / 2.0
    return _set_exit(lv, i, side, int(round(c - along / 2)), along, int(y1 - y0))


def _crowd(lv: Level, rng: random.Random, cfg) -> bool:
    lo, hi = cfg.UED_CROWD_LIMITS
    step = int(cfg.UED_MUTATE_CROWD_STEP)
    new = max(int(lo), min(int(hi), int(lv.crowd) + rng.choice((-step, step))))
    if new == lv.crowd:
        return False
    lv.crowd = new
    return True


EDITS = {
    "move_obstacle": _move_obstacle,
    "scale_obstacle": _scale_obstacle,
    "add_obstacle": _add_obstacle,
    "remove_obstacle": _remove_obstacle,
    "move_exit": _move_exit,
    "resize_exit": _resize_exit,
    "crowd": _crowd,
}


def mutate_level(parent: Level, cfg, rng: Optional[random.Random] = None) -> Level:
    rng = rng or random.Random()
    names = [k for k, w in cfg.UED_MUTATION_WEIGHTS.items() if w > 0]
    weights = [float(cfg.UED_MUTATION_WEIGHTS[k]) for k in names]
    n_edits = rng.randint(*cfg.UED_MUTATIONS_PER_CHILD)
    child = copy.deepcopy(parent)
    done: List[str] = []
    tries = 0
    while len(done) < n_edits:
        tries += 1
        if tries > int(cfg.UED_MUTATE_MAX_TRIES):
            break
        cand = copy.deepcopy(child)
        name = rng.choices(names, weights=weights, k=1)[0]
        if not EDITS[name](cand, rng, cfg):
            continue
        ok, _ = check_level(cand, max_density=float(cfg.UED_MAX_OBSTACLE_DENSITY))
        if ok:
            child = cand
            done.append(name)
    if not done:
        raise MutationFailed("no valid edit found")
    child.level_id = next_level_id()
    child.source = "mut"
    child.generation = parent.generation + 1
    child.parent_id = parent.level_id
    child.edits = tuple(parent.edits) + tuple(done)
    if parent.placement_seed is not None:
        child.placement_seed = rng.randrange(0, 2 ** 31 - 1)
    return child
