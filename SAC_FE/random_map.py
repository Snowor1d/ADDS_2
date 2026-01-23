#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
random_map.py (Shapely / unary_union 기반)
- edit_map.py의 Random Map Generator 로직을 model.py에서도 재사용하기 위한 모듈

출력 포맷 (json-friendly):
    obstacles: List[List[List[int]]]  # [[[x,y],...], ...]
    exits:     List[List[Tuple[int,int]]] or List[List[List[int]]]
              (여기서는 model.py에서 쓰기 쉬우라고 exits도 int list로 통일 가능)

NOTE:
- 장애물 후보는 (main + extras)를 unary_union으로 합친 뒤, 결과가 Polygon일 때만 채택
- MultiPolygon은 기본적으로 reject (edit_map.py도 대부분 그렇게 처리했지)
- wall_clearance rule 포함:
    - boundary에 "touch" 하거나,
    - boundary와의 거리 >= wall_clearance 여야 함
    - (0 < dist < wall_clearance) 금지
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import random
from collections import deque
from shapely.prepared import prep

# -------- Optional Shapely --------
try:
    from shapely.geometry import Polygon, box, Point as ShPoint
    from shapely.ops import unary_union
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False


Point = Tuple[int, int]


# -----------------------------
# Public dataclasses
# -----------------------------
@dataclass
class MapData:
    width: int
    height: int
    obstacles: List[List[List[int]]]       # [[[x,y],...], ...]
    exits: List[List[Tuple[int, int]]]     # [[(x,y),...], ...]
    seed_used: int


@dataclass
class RandomMapSpec:
    width: int
    height: int
    seed: Optional[int] = None

    # exits
    exit_size_min_range: Tuple[int, int] = (5, 5)
    exit_size_max_range: Tuple[int, int] = (10, 10)
    min_exit_distance_range: Tuple[float, float] = (18.0, 18.0)
    corner_avoid_dist_range: Tuple[float, float] = (14.0, 14.0)
    disallow_same_side: bool = True

    # obstacles count / density
    min_obstacles_range: Tuple[int, int] = (5, 5)
    max_obstacles_range: Tuple[int, int] = (12, 12)
    density_min_range: Tuple[float, float] = (0.10, 0.10)
    density_max_range: Tuple[float, float] = (0.25, 0.25)

    # spacing constraints
    min_obstacle_gap_range: Tuple[float, float] = (5.0, 5.0)       # obstacle-obstacle gap
    keep_gap_from_exits_range: Tuple[float, float] = (0.0, 0.0)    # obstacle-exit gap (optional)
    wall_clearance_range: Tuple[float, float] = (7.0, 7.0)         # boundary clearance rule

    # shape biases (sampled per-map, used in picker)
    main_block_bias_range: Tuple[float, float] = (0.35, 0.35)
    L_shape_bias_range: Tuple[float, float] = (0.35, 0.35)
    U_shape_bias_range: Tuple[float, float] = (0.20, 0.20)
    corridor_bias_range: Tuple[float, float] = (0.25, 0.25)
    deadend_bias_range: Tuple[float, float] = (0.25, 0.25)
    max_corridor_aspect_range: Tuple[float, float] = (6.0, 6.0)

    # attempts
    map_level_max_tries: int = 200
    obstacle_level_max_tries: int = 20000
    exit_max_tries: int = 2000


# -----------------------------
# Internal helpers
# -----------------------------
def _require_shapely():
    if not SHAPELY_OK:
        raise RuntimeError("Shapely not installed. Run: pip install shapely")


def _ri(lo: int, hi: int) -> int:
    lo, hi = int(lo), int(hi)
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _rf(lo: float, hi: float) -> float:
    lo, hi = float(lo), float(hi)
    if lo > hi:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def _clip_to_bounds(poly: "Polygon", W: int, H: int) -> Optional["Polygon"]:
    world = box(0, 0, W, H)
    p = poly.intersection(world)
    if p.is_empty:
        return None
    if p.geom_type == "Polygon":
        return p
    # MultiPolygon이면 가장 큰 것만 (원하면 reject로 바꿔도 됨)
    if p.geom_type == "MultiPolygon":
        largest = max(list(p.geoms), key=lambda g: g.area)
        return largest if largest.geom_type == "Polygon" else None
    return None


def _poly_to_points_int(poly: "Polygon") -> List[List[int]]:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[int(round(x)), int(round(y))] for x, y in coords]


def _valid_polygon(poly: "Polygon", min_area: float = 10.0) -> bool:
    return (poly.is_valid and (not poly.is_empty) and poly.area >= min_area)


def _too_close_or_intersect(new_poly: "Polygon", polys: List["Polygon"], min_gap: float) -> bool:
    for p in polys:
        if new_poly.intersects(p):
            return True
        if min_gap > 0 and new_poly.distance(p) < min_gap:
            return True
    return False


def _corners(W: int, H: int):
    return [(0, 0), (W, 0), (0, H), (W, H)]


def _nearest_corner_id(poly: "Polygon", W: int, H: int):
    c = _corners(W, H)
    dists = [poly.distance(ShPoint(x, y)) for x, y in c]
    i = min(range(4), key=lambda k: dists[k])
    return i, dists[i]


# -----------------------------
# Exit generation (edit_map.py 스타일)
# -----------------------------
def _random_exit_on_side(W: int, H: int, side: str, size_min: int, size_max: int) -> "Polygon":
    ew = _ri(size_min, size_max)
    eh = _ri(size_min, size_max)

    if side == "left":
        x0, x1 = 0, ew
        y0 = _ri(0, max(0, H - eh))
        y1 = y0 + eh
    elif side == "right":
        x1, x0 = W, W - ew
        y0 = _ri(0, max(0, H - eh))
        y1 = y0 + eh
    elif side == "bottom":
        y0, y1 = 0, eh
        x0 = _ri(0, max(0, W - ew))
        x1 = x0 + ew
    elif side == "top":
        y1, y0 = H, H - eh
        x0 = _ri(0, max(0, W - ew))
        x1 = x0 + ew
    else:
        raise ValueError("side must be one of left/right/top/bottom")

    return box(x0, y0, x1, y1)

def _filter_tiny_extras(extras, min_area: float) -> List["Polygon"]:
    if min_area <= 0:
        return extras
    out = []
    for e in extras:
        if (not e.is_empty) and e.area >= min_area:
            out.append(e)
    return out


def generate_two_exits(
    W: int, H: int,
    size_min: int,
    size_max: int,
    min_exit_distance: float,
    corner_avoid_dist: float,
    disallow_same_side: bool,
    max_tries: int,
) -> List["Polygon"]:
    sides = ["left", "right", "bottom", "top"]

    for _ in range(max_tries):
        if disallow_same_side:
            side1, side2 = random.sample(sides, 2)
        else:
            side1, side2 = random.choice(sides), random.choice(sides)

        e1 = _random_exit_on_side(W, H, side1, size_min, size_max)
        e2 = _random_exit_on_side(W, H, side2, size_min, size_max)

        if e1.intersects(e2):
            continue
        if e1.distance(e2) < min_exit_distance:
            continue

        c1, d1 = _nearest_corner_id(e1, W, H)
        c2, d2 = _nearest_corner_id(e2, W, H)
        if c1 == c2 and d1 < corner_avoid_dist and d2 < corner_avoid_dist:
            continue

        return [e1, e2]

    raise RuntimeError("Failed to generate two exits within max_tries")


# -----------------------------
# Obstacle candidates (edit_map.py 스타일)
# -----------------------------
def _random_main_block(W: int, H: int) -> "Polygon":
    cx = _ri(int(0.30 * W), int(0.70 * W))
    cy = _ri(int(0.30 * H), int(0.70 * H))

    bw = _ri(int(0.18 * W), int(0.35 * W))
    bh = _ri(int(0.18 * H), int(0.35 * H))

    x0 = cx - bw // 2
    y0 = cy - bh // 2
    base = box(x0, y0, x0 + bw, y0 + bh)

    # 50%: cutout vs add-on
    if random.random() < 0.5:
        cut_w = _ri(int(0.20 * bw), int(0.45 * bw))
        cut_h = _ri(int(0.20 * bh), int(0.45 * bh))
        side = random.choice(["L", "R", "B", "T"])
        if side == "L":
            cut = box(x0, y0 + _ri(0, max(0, bh - cut_h)), x0 + cut_w, y0 + cut_h)
        elif side == "R":
            cut = box(x0 + bw - cut_w, y0 + _ri(0, max(0, bh - cut_h)), x0 + bw, y0 + cut_h)
        elif side == "B":
            cut = box(x0 + _ri(0, max(0, bw - cut_w)), y0, x0 + cut_w, y0 + cut_h)
        else:
            cut = box(x0 + _ri(0, max(0, bw - cut_w)), y0 + bh - cut_h, x0 + cut_w, y0 + bh)
        poly = base.difference(cut)
    else:
        add_w = _ri(int(0.18 * bw), int(0.40 * bw))
        add_h = _ri(int(0.18 * bh), int(0.40 * bh))
        side = random.choice(["L", "R", "B", "T"])
        if side == "L":
            add = box(x0 - add_w, y0 + _ri(0, max(0, bh - add_h)), x0, y0 + add_h)
        elif side == "R":
            add = box(x0 + bw, y0 + _ri(0, max(0, bh - add_h)), x0 + bw + add_w, y0 + add_h)
        elif side == "B":
            add = box(x0 + _ri(0, max(0, bw - add_w)), y0 - add_h, x0 + add_w, y0)
        else:
            add = box(x0 + _ri(0, max(0, bw - add_w)), y0 + bh, x0 + add_w, y0 + bh + add_h)
        poly = unary_union([base, add])

    p = _clip_to_bounds(poly, W, H)
    return p if p is not None else base


def _random_L_shape(W: int, H: int) -> Optional["Polygon"]:
    cx = _ri(int(0.20 * W), int(0.80 * W))
    cy = _ri(int(0.20 * H), int(0.80 * H))

    arm_th = _ri(4, 10)
    arm_len1 = _ri(int(0.15 * min(W, H)), int(0.40 * min(W, H)))
    arm_len2 = _ri(int(0.15 * min(W, H)), int(0.40 * min(W, H)))

    orient = random.choice(["UR", "UL", "DR", "DL"])
    if orient == "UR":
        a = box(cx, cy, cx + arm_len1, cy + arm_th)
        b = box(cx, cy, cx + arm_th, cy + arm_len2)
    elif orient == "UL":
        a = box(cx - arm_len1, cy, cx, cy + arm_th)
        b = box(cx - arm_th, cy, cx, cy + arm_len2)
    elif orient == "DR":
        a = box(cx, cy - arm_th, cx + arm_len1, cy)
        b = box(cx, cy - arm_len2, cx + arm_th, cy)
    else:
        a = box(cx - arm_len1, cy - arm_th, cx, cy)
        b = box(cx - arm_th, cy - arm_len2, cx, cy)

    u = unary_union([a, b])
    p = _clip_to_bounds(u, W, H)
    if p is None or p.area < 20:
        return None
    return p


def _random_U_shape(W: int, H: int, min_opening: int = 10) -> Optional["Polygon"]:
    cx = _ri(int(0.20 * W), int(0.80 * W))
    cy = _ri(int(0.20 * H), int(0.80 * H))

    th = _ri(4, 10)
    w = _ri(int(0.18 * W), int(0.35 * W))
    h = _ri(int(0.18 * H), int(0.35 * H))

    open_dir = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    # ✅ 오목부(입구) 폭 체크
    if open_dir in ("UP", "DOWN"):
        opening = w - 2 * th
    else:
        opening = h - 2 * th
    if opening < min_opening:
        return None

def _random_corridor_strip(W: int, H: int, margin: int = 1) -> "Polygon":
    thickness = _ri(4, 10)
    length = _ri(int(0.35 * min(W, H)), int(0.9 * min(W, H)))
    horizontal = (random.random() < 0.5)

    if horizontal:
        w, h = length, thickness
    else:
        w, h = thickness, length

    x0 = _ri(margin, max(margin, W - margin - w))
    y0 = _ri(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)


def _random_rect_obstacle(W: int, H: int, margin: int = 1) -> "Polygon":
    w = _ri(4, 30)
    h = _ri(4, 30)
    x0 = _ri(margin, max(margin, W - margin - w))
    y0 = _ri(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)


def _random_convex_obstacle(W: int, H: int, margin: int = 1) -> "Polygon":
    n = _ri(4, 10)
    pts = [( _ri(margin, W - margin), _ri(margin, H - margin) ) for _ in range(n)]
    return Polygon(pts).convex_hull


def _random_deadend_cap_near(strip: "Polygon", W: int, H: int, margin: int = 1) -> "Polygon":
    x0, y0, x1, y1 = strip.bounds
    if (x1 - x0) >= (y1 - y0):  # horizontal-ish
        cap_w = _ri(6, 12)
        cap_h = _ri(6, 14)
        side = random.choice(["L", "R"])
        if side == "L":
            cx1 = int(x0)
            cx0 = max(margin, cx1 - cap_w)
        else:
            cx0 = int(x1)
            cx1 = min(W - margin, cx0 + cap_w)
        cy0 = _ri(max(margin, int(y0 - cap_h)), min(int(y1), H - margin - cap_h))
        cy1 = cy0 + cap_h
        return box(cx0, cy0, cx1, cy1)
    else:  # vertical-ish
        cap_w = _ri(6, 14)
        cap_h = _ri(6, 12)
        side = random.choice(["B", "T"])
        if side == "B":
            cy1 = int(y0)
            cy0 = max(margin, cy1 - cap_h)
        else:
            cy0 = int(y1)
            cy1 = min(H - margin, cy0 + cap_h)
        cx0 = _ri(max(margin, int(x0 - cap_w)), min(int(x1), W - margin - cap_w))
        cx1 = cx0 + cap_w
        return box(cx0, cy0, cx1, cy1)


def _pick_obstacle_candidate(W: int, H: int, params: Dict[str, float]):
    """
    returns: (main_poly, extra_polys, shape_tag)
    shape_tag: "main" | "L" | "U" | "corr" | "rect" | "conv"
    """
    extra = []

    main_block_bias = float(params.get("main_block_bias", 0.35))
    L_shape_bias    = float(params.get("L_shape_bias", 0.35))
    U_shape_bias    = float(params.get("U_shape_bias", 0.20))
    corridor_bias   = float(params.get("corridor_bias", 0.25))
    deadend_bias    = float(params.get("deadend_bias", 0.25))
    max_aspect      = float(params.get("max_corridor_aspect", 6.0))

    palette = [
        ("main", main_block_bias),
        ("L",    L_shape_bias),
        ("U",    U_shape_bias),
        ("corr", corridor_bias),
        ("rect", 0.20),
        ("conv", 0.15),
    ]
    total = sum(w for _, w in palette)
    r = random.random() * (total if total > 0 else 1.0)

    acc = 0.0
    choice = "rect"
    for name, w in palette:
        acc += w
        if r <= acc:
            choice = name
            break

    if choice == "main":
        return _random_main_block(W, H), extra, "main"

    if choice == "L":
        p = _random_L_shape(W, H)
        if p is not None:
            return p, extra, "L"
        # fallback
        return _random_main_block(W, H), extra, "main"

    if choice == "U":
        p = _random_U_shape(W, H, min_opening=int(params.get("min_u_opening", 10)))
        if p is not None:
            return p, extra, "U"
        # fallback
        return _random_main_block(W, H), extra, "main"

    if choice == "corr":
        base = _random_corridor_strip(W, H, margin=1)
        x0, y0, x1, y1 = base.bounds
        w = max(1e-6, (x1 - x0))
        h = max(1e-6, (y1 - y0))
        aspect = max(w / h, h / w)
        if aspect > max_aspect:
            base = _random_rect_obstacle(W, H, margin=1)
            if random.random() < deadend_bias:
                extra.append(_random_deadend_cap_near(base, W, H, margin=1))
            return base, extra, "rect"

        if random.random() < deadend_bias:
            extra.append(_random_deadend_cap_near(base, W, H, margin=1))
        extra = _filter_tiny_extras(extra, float(params.get("min_extra_area", 20.0)))
        return base, extra, "corr"

    if choice == "conv":
        return _random_convex_obstacle(W, H, margin=1), extra, "conv"

    return _random_rect_obstacle(W, H, margin=1), extra, "rect"


# -----------------------------
# Core obstacle generation (density + count)
# -----------------------------
def generate_obstacles_with_density(
    W: int, H: int,
    exits: List["Polygon"],
    min_obstacles: int,
    max_obstacles: int,
    density_min: float,
    density_max: float,
    min_gap: float,
    keep_gap_from_exits: float,
    params: Dict[str, float],
    max_tries: int,
) -> List["Polygon"]:

    map_area = float(W * H)
    target_density = random.uniform(density_min, density_max)
    target_area = target_density * map_area
    target_count = random.randint(min_obstacles, max_obstacles)

    obstacles: List["Polygon"] = []
    total_area = 0.0

    boundary = box(0, 0, W, H).boundary
    wall_clearance = float(params.get("wall_clearance", 7.0))

    for _ in range(max_tries):
        if (len(obstacles) >= target_count) and (total_area >= target_area):
            break

        main, extras, shape_tag = _pick_obstacle_candidate(W, H, params)
        merged = unary_union([main] + extras)

        # union 결과가 Polygon일 때만 채택 (원하면 MultiPolygon 처리 추가 가능)
        if merged.geom_type != "Polygon":
            continue
        cand = merged

        if not _valid_polygon(cand, min_area=20.0):
            continue

        # wall clearance rule
        d_wall = cand.distance(boundary)  # 0 if touching
        if d_wall < wall_clearance and (not cand.touches(boundary)):
            continue

        # exit constraints
        bad = False
        for e in exits:
            if cand.intersects(e):
                bad = True
                break
            if keep_gap_from_exits > 0 and cand.distance(e) < keep_gap_from_exits:
                bad = True
                break
        if bad:
            continue

        # obstacle-obstacle constraints
        if _too_close_or_intersect(cand, obstacles, min_gap=min_gap):
            continue

        # density upper guard
        if (total_area + cand.area) > (density_max * map_area * 1.05):
            continue

                # (특히 ㄴ자/오목형이 벽에 붙을 때 발생)
        world = box(0, 0, W, H)

        # existing obstacles + cand 로 blocked 구성
        # (주의: unary_union이 MultiPolygon을 반환해도 difference는 됨)
        blocked = unary_union(obstacles + [cand])
        if _has_enclosed_free_pocket_grid(W, H, blocked, grid_step=2, pinch_cells=1):
            continue

                # ✅ L/U는 벽 "touch" 자체를 금지 (ㄴ/ㄷ자 벽붙음 방지)
        if shape_tag in ("L", "U"):
            if cand.touches(boundary):
                continue
            # 더 강하게: 벽에서 wall_clearance 이상 떨어져야만 허용하고 싶으면 이거까지 켜
            if d_wall < wall_clearance:
                continue
        else:
            # 기존 룰 유지: touch는 허용, 가까이만(0<d<clearance) 금지
            if d_wall < wall_clearance and (not cand.touches(boundary)):
                continue

        obstacles.append(cand)
        total_area += cand.area



    final_density = total_area / map_area if map_area > 0 else 0.0
    if final_density < density_min:
        raise RuntimeError(f"Density too low: {final_density:.3f} < {density_min}")
    if len(obstacles) < min_obstacles:
        raise RuntimeError(f"Obstacle count too low: {len(obstacles)} < {min_obstacles}")

    return obstacles

def _has_boundary_opening(poly: "Polygon", world_boundary, eps: float = 1e-6) -> bool:
    """
    poly(자유공간 컴포넌트)가 world boundary와 '선분 길이'로 붙어있으면 True.
    점으로만 닿는 건 length=0 이라서 False 처리됨.
    """
    inter = poly.boundary.intersection(world_boundary)
    # Point면 length=0, LineString면 length>0
    return getattr(inter, "length", 0.0) > eps


def _has_enclosed_free_pocket_grid(W: int, H: int, blocked_union, grid_step: int = 2, pinch_cells: int = 1) -> bool:
    """
    grid_step: 검사 해상도 (너 스냅이 2면 2 추천)
    pinch_cells: 얇은 통로 끊기용. 1이면 한 칸짜리 목(실제로 못지나갈 확률 큼)을 pocket으로 취급.
    """
    if blocked_union.is_empty:
        return False

    P = prep(blocked_union)

    nx = W // grid_step + 1
    ny = H // grid_step + 1

    # occupied grid
    occ = [[False] * ny for _ in range(nx)]

    # cell center point-in-geom
    for ix in range(nx):
        x = ix * grid_step
        for iy in range(ny):
            y = iy * grid_step
            if P.contains(ShPoint(x, y)) or P.touches(ShPoint(x, y)):
                occ[ix][iy] = True

    # pinch: 얇은 목을 끊기 위해 occ를 살짝 팽창(모폴로지 dilation)
    # (pinch_cells=1이면 8-neighborhood 1칸 확장)
    if pinch_cells > 0:
        occ2 = [row[:] for row in occ]
        for ix in range(nx):
            for iy in range(ny):
                if not occ[ix][iy]:
                    continue
                for dx in range(-pinch_cells, pinch_cells + 1):
                    for dy in range(-pinch_cells, pinch_cells + 1):
                        jx, jy = ix + dx, iy + dy
                        if 0 <= jx < nx and 0 <= jy < ny:
                            occ2[jx][jy] = True
        occ = occ2

    # flood fill from boundary free cells
    q = deque()
    vis = [[False] * ny for _ in range(nx)]

    def try_push(ix, iy):
        if 0 <= ix < nx and 0 <= iy < ny and (not occ[ix][iy]) and (not vis[ix][iy]):
            vis[ix][iy] = True
            q.append((ix, iy))

    # boundary seeds
    for ix in range(nx):
        try_push(ix, 0)
        try_push(ix, ny - 1)
    for iy in range(ny):
        try_push(0, iy)
        try_push(nx - 1, iy)

    # BFS 4-neighborhood (원하면 8-neighborhood로 더 관대/빡세게 조절)
    while q:
        ix, iy = q.popleft()
        try_push(ix + 1, iy)
        try_push(ix - 1, iy)
        try_push(ix, iy + 1)
        try_push(ix, iy - 1)

    # any unvisited free cell => enclosed pocket exists
    for ix in range(nx):
        for iy in range(ny):
            if (not occ[ix][iy]) and (not vis[ix][iy]):
                return True
    return False

# -----------------------------
# Public API: generate_map(spec)
# -----------------------------
def generate_map(spec: RandomMapSpec) -> MapData:
    _require_shapely()

    # seed
    seed_used = spec.seed if spec.seed is not None else random.randrange(0, 2**31 - 1)
    random.seed(seed_used)

    W, H = int(spec.width), int(spec.height)

    # sample ranges -> concrete params (map-level)
    params: Dict[str, float] = {}

    params["density_min"] = float(_rf(*spec.density_min_range))
    params["density_max"] = float(_rf(*spec.density_max_range))
    if params["density_min"] > params["density_max"]:
        params["density_min"], params["density_max"] = params["density_max"], params["density_min"]

    params["min_obstacles"] = float(_ri(*spec.min_obstacles_range))
    params["max_obstacles"] = float(_ri(*spec.max_obstacles_range))
    if params["min_obstacles"] > params["max_obstacles"]:
        params["min_obstacles"], params["max_obstacles"] = params["max_obstacles"], params["min_obstacles"]

    params["min_obstacle_gap"] = float(_rf(*spec.min_obstacle_gap_range))
    params["keep_gap_from_exits"] = float(_rf(*spec.keep_gap_from_exits_range))
    params["wall_clearance"] = float(_rf(*spec.wall_clearance_range))

    params["main_block_bias"] = float(_rf(*spec.main_block_bias_range))
    params["L_shape_bias"]    = float(_rf(*spec.L_shape_bias_range))
    params["U_shape_bias"]    = float(_rf(*spec.U_shape_bias_range))
    params["corridor_bias"]   = float(_rf(*spec.corridor_bias_range))
    params["deadend_bias"]    = float(_rf(*spec.deadend_bias_range))
    params["max_corridor_aspect"] = float(_rf(*spec.max_corridor_aspect_range))

    exit_size_min = int(_ri(*spec.exit_size_min_range))
    exit_size_max = int(_ri(*spec.exit_size_max_range))
    if exit_size_min > exit_size_max:
        exit_size_min, exit_size_max = exit_size_max, exit_size_min

    min_exit_distance = float(_rf(*spec.min_exit_distance_range))
    corner_avoid_dist = float(_rf(*spec.corner_avoid_dist_range))

    # map-level tries (same as edit_map.py)
    for _ in range(spec.map_level_max_tries):
        exits = generate_two_exits(
            W, H,
            size_min=exit_size_min,
            size_max=exit_size_max,
            min_exit_distance=min_exit_distance,
            corner_avoid_dist=corner_avoid_dist,
            disallow_same_side=spec.disallow_same_side,
            max_tries=spec.exit_max_tries,
        )
        try:
            obstacles = generate_obstacles_with_density(
                W, H,
                exits=exits,
                min_obstacles=int(params["min_obstacles"]),
                max_obstacles=int(params["max_obstacles"]),
                density_min=float(params["density_min"]),
                density_max=float(params["density_max"]),
                min_gap=float(params["min_obstacle_gap"]),
                keep_gap_from_exits=float(params["keep_gap_from_exits"]),
                params=params,
                max_tries=spec.obstacle_level_max_tries,
            )
        except RuntimeError:
            continue

        obstacles_pts = [_poly_to_points_int(o) for o in obstacles]
        exits_pts = []
        for e in exits:
            pts = _poly_to_points_int(e)
            exits_pts.append([(int(x), int(y)) for x, y in pts])

        return MapData(
            width=W,
            height=H,
            obstacles=obstacles_pts,
            exits=exits_pts,
            seed_used=seed_used,
        )

    raise RuntimeError("Failed to generate a valid random map after many attempts.")
