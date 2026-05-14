#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
random_map.py (Shapely / unary_union 기반) - difficulty 단일 입력 + Large/Small obstacle quota + large_conv 포함

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import random
from collections import deque
from shapely.prepared import prep
from shapely.geometry import LineString
import math

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
    difficulty: int = 6
    seed: Optional[int] = None

    # attempts
    map_level_max_tries: int = 500
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
    # strict reject for MultiPolygon by default
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
# Convex quality (min-width) helpers
# -----------------------------
def _min_width_of_polygon(poly: "Polygon") -> float:
    """
    minimum_rotated_rectangle의 4개 변 길이 중 min을 폭으로 근사.
    얇고 길쭉한 convex(hull)을 reject하기 위한 지표.
    """
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 5:
        return 0.0
    lens = []
    for i in range(4):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        lens.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
    return float(min(lens)) if lens else 0.0


def _passes_convex_quality(poly: "Polygon", min_width: float) -> bool:
    if poly.is_empty or (not poly.is_valid) or poly.area < 10:
        return False
    mw = _min_width_of_polygon(poly)
    return mw >= float(min_width)


# -----------------------------
# Exit flank helpers
# -----------------------------
def _exit_side_from_rect(e: "Polygon", W: int, H: int) -> str:
    x0, y0, x1, y1 = e.bounds
    eps = 1e-6
    if abs(x0 - 0) < eps:
        return "left"
    if abs(x1 - W) < eps:
        return "right"
    if abs(y0 - 0) < eps:
        return "bottom"
    if abs(y1 - H) < eps:
        return "top"
    cx, cy = e.centroid.x, e.centroid.y
    d = {
        "left": cx,
        "right": (W - cx),
        "bottom": cy,
        "top": (H - cy),
    }
    return min(d.keys(), key=lambda k: d[k])


def generate_exit_flanks(
    W: int, H: int,
    exits: List["Polygon"],
    bias: float = 0.6,
    gap_min: int = 1,
    gap_max: int = 4,
    flank_len_min: int = 8,
    flank_len_max: int = 18,
    depth_boost_min: int = 0,
    depth_boost_max: int = 8,
    margin: int = 0,

    # ✅ 수직 벽 규칙 파라미터
    perp_detach_min: int = 10,
    eps: float = 1e-6,
) -> List["Polygon"]:
    flanks: List["Polygon"] = []
    bias = max(0.0, min(1.0, float(bias)))
    gmin, gmax = int(gap_min), int(gap_max)
    perp_detach_min = int(perp_detach_min)

    def _touch(v: float, target: float) -> bool:
        return abs(v - target) < eps

    def _perp_rule_ok(p: "Polygon", side: str) -> bool:
        """
        side = exit side (left/right/bottom/top)
        - along-wall은 항상 touch
        - 수직 벽(perp)은 (touch 하나) OR (양쪽 dist>=perp_detach_min)
        """
        x0, y0, x1, y1 = p.bounds

        if side == "left":
            if not _touch(x0, 0.0):
                return False
            touch_bottom = _touch(y0, 0.0)
            touch_top = _touch(y1, float(H))
            if touch_bottom or touch_top:
                return True
            return (y0 >= perp_detach_min) and ((H - y1) >= perp_detach_min)

        if side == "right":
            if not _touch(x1, float(W)):
                return False
            touch_bottom = _touch(y0, 0.0)
            touch_top = _touch(y1, float(H))
            if touch_bottom or touch_top:
                return True
            return (y0 >= perp_detach_min) and ((H - y1) >= perp_detach_min)

        if side == "bottom":
            if not _touch(y0, 0.0):
                return False
            touch_left = _touch(x0, 0.0)
            touch_right = _touch(x1, float(W))
            if touch_left or touch_right:
                return True
            return (x0 >= perp_detach_min) and ((W - x1) >= perp_detach_min)

        # top
        if not _touch(y1, float(H)):
            return False
        touch_left = _touch(x0, 0.0)
        touch_right = _touch(x1, float(W))
        if touch_left or touch_right:
            return True
        return (x0 >= perp_detach_min) and ((W - x1) >= perp_detach_min)

    for e in exits:
        if random.random() > bias:
            continue

        x0, y0, x1, y1 = e.bounds
        side = _exit_side_from_rect(e, W, H)

        depth = int(round(min((x1 - x0), (y1 - y0))))
        flank_depth = max(4, depth + _ri(depth_boost_min, depth_boost_max))

        gap1 = _ri(gmin, gmax)
        gap2 = _ri(gmin, gmax)
        flank_len1 = _ri(flank_len_min, flank_len_max)
        flank_len2 = _ri(flank_len_min, flank_len_max)

        # along-wall은 항상 붙게
        if side == "left":
            a = box(0, max(margin, y0 - gap1 - flank_len1), flank_depth, max(margin, y0 - gap1))
            b = box(0, min(H - margin, y1 + gap2), flank_depth, min(H - margin, y1 + gap2 + flank_len2))
        elif side == "right":
            a = box(max(margin, W - flank_depth), max(margin, y0 - gap1 - flank_len1), W, max(margin, y0 - gap1))
            b = box(max(margin, W - flank_depth), min(H - margin, y1 + gap2), W, min(H - margin, y1 + gap2 + flank_len2))
        elif side == "bottom":
            a = box(max(margin, x0 - gap1 - flank_len1), 0, max(margin, x0 - gap1), flank_depth)
            b = box(min(W - margin, x1 + gap2), 0, min(W - margin, x1 + gap2 + flank_len2), flank_depth)
        else:  # top
            a = box(max(margin, x0 - gap1 - flank_len1), max(margin, H - flank_depth), max(margin, x0 - gap1), H)
            b = box(min(W - margin, x1 + gap2), max(margin, H - flank_depth), min(W - margin, x1 + gap2 + flank_len2), H)

        for cand in (a, b):
            p = _clip_to_bounds(cand, W, H)
            if p is None:
                continue
            if not _valid_polygon(p, min_area=6.0):
                continue
            if not _perp_rule_ok(p, side=side):
                continue
            if any(p.intersects(ex) for ex in exits):
                continue
            flanks.append(p)

    if not flanks:
        return []

    merged = unary_union(flanks)
    if merged.is_empty:
        return []
    if merged.geom_type == "Polygon":
        return [merged]
    if merged.geom_type == "MultiPolygon":
        return [g for g in merged.geoms if g.geom_type == "Polygon"]
    return []


# -----------------------------
# Difficulty table
# -----------------------------
def _params_from_difficulty(W: int, H: int, difficulty: int) -> Dict[str, float]:
    d = int(difficulty)

    def base_common():
        return {
            # exits
            "exit_along_min": 7, "exit_along_max": 10,
            "exit_depth_min": 4, "exit_depth_max": 5,
            "min_exit_distance": 30.0,
            "min_exit_center_distance": 40.0,
            "corner_avoid_dist": 12.0,
            "disallow_same_side": 1.0,

            # density/quota
            "large_count_min": 1, "large_count_max": 2,
            "small_count_min": 1, "small_count_max": 3,
            "density_large_ratio": 0.4,

            # spacing
            "min_obstacle_gap": 7.0,
            "keep_gap_from_exits": 10.0,
            "wall_clearance": 7.0,

            # large shape mix
            "large_block_bias": 0.30,
            "L_shape_bias": 0.05,
            "U_shape_bias": 0.05,
            "wall_rect_bias": 0.25,
            "large_conv_bias": 0.4,

            # small mix
            "small_rect_bias": 0.45,
            "small_corr_bias": 0.45,
            "small_conv_bias": 0.20,
            "deadend_bias": 0.25,
            "max_corridor_aspect": 6.0,

            # wall rect sizes (large)
            "rect_w_min": 25, "rect_w_max": 60,
            "rect_h_min": 25, "rect_h_max": 60,

            # small rect sizes
            "small_rect_w_min": 6, "small_rect_w_max": 15,
            "small_rect_h_min": 6, "small_rect_h_max": 15,

            # large block base size ratios (map 대비)
            "large_bw_min": 0.15, "large_bw_max": 0.50,
            "large_bh_min": 0.15, "large_bh_max": 0.50,

            # large block placement range (맵 전역; 너무 구석만 피하려면 범위 조절)
            "large_cx_min": 0.10, "large_cx_max": 0.90,
            "large_cy_min": 0.10, "large_cy_max": 0.90,

            # ✅ large block attach control
            "large_attach_bias": 0.60,      # add 블록을 붙일 확률
            "large_add1_bias": 0.65,        # 붙을 때 add=1 비중
            "large_add2_bias": 0.20,        # 붙을 때 add=2 비중
            "large_add3_bias": 0.15,        # 붙을 때 add=3 비중
            "add_w_min_ratio": 0.20,
            "add_w_max_ratio": 0.7,
            "add_h_min_ratio": 0.20,
            "add_h_max_ratio": 0.7,

            # large convex params
            "large_conv_scale_min": 0.10,
            "large_conv_scale_max": 0.40,
            "large_conv_n_min": 6,
            "large_conv_n_max": 10,

            # convex gap/quality
            "conv_min_width": 10.0,
            "large_conv_min_width": 10.0,
            "conv_gap_boost": 10.0,

            # exit flanks
            "exit_flank_bias": 0.20,
            "exit_flank_gap_min": 2,
            "exit_flank_gap_max": 4,
            "exit_flank_len_min": 8,
            "exit_flank_len_max": 18,
            "exit_flank_depth_boost_min": 10,
            "exit_flank_depth_boost_max": 30,
            "large_conv_radial_jitter": 0.18,  # 0.10~0.25 추천
            "large_conv_xy_jitter": 0.8,       # 0.5~1.5 추천

        }

    p = base_common()

    if d == 0:
        p.update({
            "density_min": 0, "density_max": 0,
            "large_count_min": 0, "large_count_max": 0,
            "small_count_min": 0, "small_count_max": 0,
        })
        return p

    if d <= 1:
        p.update({
            "density_min": 0.05, "density_max": 0.10,
            "large_count_min": 1, "large_count_max": 2,
            "small_count_min": 1, "small_count_max": 2,
        })
        return p

    if d <= 2:
        p.update({
            "density_min": 0.05, "density_max": 0.15,
            "large_count_min": 1, "large_count_max": 2,
            "small_count_min": 1, "small_count_max": 3,
        })
        return p

    if d <= 3:
        p.update({
            "density_min": 0.1, "density_max": 0.20,
            "large_count_min": 1, "large_count_max": 3,
            "small_count_min": 2, "small_count_max": 4,
        })
        return p

    if d == 4:
        p.update({
            "density_min": 0.15, "density_max": 0.25,
            "large_count_min": 2, "large_count_max": 4,
            "small_count_min": 2, "small_count_max": 5,
        })
        return p

    if d == 5:
        p.update({
            "density_min": 0.20, "density_max": 0.30,
            "large_count_min": 2, "large_count_max": 5,
            "small_count_min": 2, "small_count_max": 6,

        })
        return p

    # HARD
    p.update({
        "density_min": 0.25, "density_max": 0.35,
        "large_count_min": 2, "large_count_max": 5,
        "small_count_min": 3, "small_count_max": 7,
    })
    return p


# -----------------------------
# Exit generation
# -----------------------------
def _random_exit_on_side(W: int, H: int, side: str,
                         along_min: int, along_max: int,
                         depth_min: int, depth_max: int) -> "Polygon":
    along = _ri(along_min, along_max)
    depth = _ri(depth_min, depth_max)

    if side == "left":
        x0, x1 = 0, depth
        y0 = _ri(0, max(0, H - along))
        y1 = y0 + along
    elif side == "right":
        x1, x0 = W, W - depth
        y0 = _ri(0, max(0, H - along))
        y1 = y0 + along
    elif side == "bottom":
        y0, y1 = 0, depth
        x0 = _ri(0, max(0, W - along))
        x1 = x0 + along
    elif side == "top":
        y1, y0 = H, H - depth
        x0 = _ri(0, max(0, W - along))
        x1 = x0 + along
    else:
        raise ValueError("side must be one of left/right/top/bottom")

    return box(x0, y0, x1, y1)


def generate_two_exits(
    W: int, H: int,
    along_min: int, along_max: int,
    depth_min: int, depth_max: int,
    min_exit_distance: float,
    min_exit_center_distance: float,
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

        e1 = _random_exit_on_side(W, H, side1, along_min, along_max, depth_min, depth_max)
        e2 = _random_exit_on_side(W, H, side2, along_min, along_max, depth_min, depth_max)

        if e1.intersects(e2):
            continue
        if e1.distance(e2) < min_exit_distance:
            continue
        if e1.centroid.distance(e2.centroid) < min_exit_center_distance:
            continue

        c1, d1 = _nearest_corner_id(e1, W, H)
        c2, d2 = _nearest_corner_id(e2, W, H)
        if c1 == c2 and d1 < corner_avoid_dist and d2 < corner_avoid_dist:
            continue

        return [e1, e2]

    raise RuntimeError("Failed to generate two exits within max_tries")

def _has_exit_disconnected_free_region_grid(
    W: int,
    H: int,
    blocked_union,
    exits: List["Polygon"],
    grid_step: int = 2,
    pinch_cells: int = 1,
) -> bool:
    """
    ✅ 출구에서 시작해서 도달 가능한 free-space만 '정상'으로 본다.
    - 벽+장애물(main+add 포함)이 만든 closed 공간(벽에 붙어있어도)도 잡힘
    - exits로 연결되지 않는 free component가 하나라도 있으면 True(=reject)
    """
    if blocked_union.is_empty:
        return False

    P = prep(blocked_union)

    nx = W // grid_step + 1
    ny = H // grid_step + 1

    # occupancy grid
    occ = [[False] * ny for _ in range(nx)]
    for ix in range(nx):
        x = ix * grid_step
        for iy in range(ny):
            y = iy * grid_step
            pt = ShPoint(x, y)
            if P.contains(pt) or P.touches(pt):
                occ[ix][iy] = True

    # pinch(=장애물 팽창)로 좁은 틈/거의 닫힌 공간을 더 잘 잡기
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

    vis = [[False] * ny for _ in range(nx)]
    q = deque()

    def try_push(ix, iy):
        if 0 <= ix < nx and 0 <= iy < ny and (not occ[ix][iy]) and (not vis[ix][iy]):
            vis[ix][iy] = True
            q.append((ix, iy))

    # ✅ seed: exits 내부(또는 경계)에서 시작
    # (exit가 blocked와 겹치면 seed가 없을 수 있으니 여러 점을 넣어줌)
    for e in exits:
        ex0, ey0, ex1, ey1 = e.bounds
        ix0 = max(0, int(ex0 // grid_step) - 1)
        ix1 = min(nx - 1, int(ex1 // grid_step) + 1)
        iy0 = max(0, int(ey0 // grid_step) - 1)
        iy1 = min(ny - 1, int(ey1 // grid_step) + 1)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                # exit 내부(또는 touches)인 셀만 seed로
                x = ix * grid_step
                y = iy * grid_step
                pt = ShPoint(x, y)
                if e.contains(pt) or e.touches(pt):
                    try_push(ix, iy)

    # seed가 아예 없으면(드물지만) -> 안전하게 reject 쪽(=연결성 판단 불가)
    if not q:
        return True

    # BFS
    while q:
        ix, iy = q.popleft()
        try_push(ix + 1, iy)
        try_push(ix - 1, iy)
        try_push(ix, iy + 1)
        try_push(ix, iy - 1)

    # exits에서 도달 불가능한 free cell이 있으면 => 닫힌 공간/죽은 공간 존재
    for ix in range(nx):
        for iy in range(ny):
            if (not occ[ix][iy]) and (not vis[ix][iy]):
                return True

    return False



# -----------------------------
# Obstacle primitives (Large / Small)
# -----------------------------
def _random_large_block(W: int, H: int, params: Dict[str, float]) -> Optional["Polygon"]:
    """
    Large block = base rect + (optional) 1~3 add blocks

    ✅ 규칙(요청 반영, 명확 버전)
    - base(main)는 벽에 닿아도 됨. (기존 strict-by-side 규칙 유지)
    - add(돌출 블록)는 "어떤 벽에도 닿으면 안 됨"
      => add는 항상 모든 벽에서 wall_clearance 이상 떨어진 내부 영역에 완전히 포함되어야 함
      => 즉 add.bounds가 [c, W-c]×[c, H-c] 밖으로 나가면 무조건 reject
    """
    # placement range (map-wide)
    cx = _ri(int(float(params.get("large_cx_min", 0.10)) * W),
             int(float(params.get("large_cx_max", 0.90)) * W))
    cy = _ri(int(float(params.get("large_cy_min", 0.10)) * H),
             int(float(params.get("large_cy_max", 0.90)) * H))

    bw = _ri(int(float(params.get("large_bw_min", 0.22)) * W),
             int(float(params.get("large_bw_max", 0.60)) * W))
    bh = _ri(int(float(params.get("large_bh_min", 0.22)) * H),
             int(float(params.get("large_bh_max", 0.60)) * H))

    bw = max(6, bw)
    bh = max(6, bh)

    x0 = cx - bw // 2
    y0 = cy - bh // 2
    base_raw = box(x0, y0, x0 + bw, y0 + bh)

    wall_clearance = float(params.get("wall_clearance", 7.0))

    # base clip + 최소 검증
    base = _clip_to_bounds(base_raw, W, H)
    if base is None or base.geom_type != "Polygon" or base.area < 20:
        return None

    # ✅ base는 기존 룰 유지: band 안이면 해당 벽 touch 해야 함(=떠있는 상태 금지)
    if not _passes_wall_rules_strict_by_side(base, "small", W, H, wall_clearance):
        return None

    # base bounds 재계산 (clip 반영)
    x0, y0, x1, y1 = base.bounds
    bw = (x1 - x0)
    bh = (y1 - y0)

    # base가 벽에서 얼마나 떨어졌는지 (add side 후보 필터용)
    distL = x0
    distR = W - x1
    distB = y0
    distT = H - y1

    # add가 “충분히 돌출”되도록 최소 돌출 길이
    min_protrude = float(params.get("add_protrude_min", wall_clearance))

    attach_bias = float(params.get("large_attach_bias", 0.70))
    attach_bias = max(0.0, min(1.0, attach_bias))

    # add를 안 붙이면 base만 반환
    if random.random() > attach_bias:
        return base

    # --- add 개수(1~3) 분포 선택 ---
    w1 = float(params.get("large_add1_bias", 0.55))
    w2 = float(params.get("large_add2_bias", 0.30))
    w3 = float(params.get("large_add3_bias", 0.15))
    if (w1 + w2 + w3) <= 0:
        k = 1
    else:
        k = random.choices([1, 2, 3], weights=[w1, w2, w3], k=1)[0]

    sides = random.sample(["L", "R", "B", "T"], k=k)

    add_w_min_ratio = float(params.get("add_w_min_ratio", 0.30))
    add_w_max_ratio = float(params.get("add_w_max_ratio", 1.00))
    add_h_min_ratio = float(params.get("add_h_min_ratio", 0.30))
    add_h_max_ratio = float(params.get("add_h_max_ratio", 1.00))

    adds: List["Polygon"] = []

    c = float(wall_clearance)

    def _add_inside_interior(add_poly: "Polygon") -> bool:
        # ✅ add는 [c, W-c] × [c, H-c] 내부에 완전히 포함
        ax0, ay0, ax1, ay1 = add_poly.bounds
        return (ax0 >= c) and (ay0 >= c) and (ax1 <= (W - c)) and (ay1 <= (H - c))

    for side in sides:
        # base가 해당 방향 벽에 너무 가까우면(돌출 공간이 없으면) skip
        # (돌출 자체는 가능해도 add를 "내부 영역"에 두려면 여유가 더 필요함)
        if side == "L" and distL < (min_protrude + c):
            continue
        if side == "R" and distR < (min_protrude + c):
            continue
        if side == "B" and distB < (min_protrude + c):
            continue
        if side == "T" and distT < (min_protrude + c):
            continue

        add_w = _ri(int(add_w_min_ratio * bw), int(add_w_max_ratio * bw))
        add_h = _ri(int(add_h_min_ratio * bh), int(add_h_max_ratio * bh))
        add_w = max(2, add_w)
        add_h = max(2, add_h)

        # 돌출 길이 최소 반영
        if side in ("L", "R"):
            add_w = max(add_w, int(min_protrude))
        else:
            add_h = max(add_h, int(min_protrude))

        # ✅ add가 "완전히 내부"에 들어가도록 사이즈 상한을 미리 제한
        if side == "L":
            # add의 left edge = x0 - add_w >= c  -> add_w <= x0 - c
            max_w = int(math.floor(x0 - c))
            if max_w < int(min_protrude):
                continue
            add_w = min(add_w, max_w)

        elif side == "R":
            # add의 right edge = x1 + add_w <= W - c -> add_w <= (W - c) - x1
            max_w = int(math.floor((W - c) - x1))
            if max_w < int(min_protrude):
                continue
            add_w = min(add_w, max_w)

        elif side == "B":
            # add의 bottom edge = y0 - add_h >= c -> add_h <= y0 - c
            max_h = int(math.floor(y0 - c))
            if max_h < int(min_protrude):
                continue
            add_h = min(add_h, max_h)

        else:  # "T"
            # add의 top edge = y1 + add_h <= H - c -> add_h <= (H - c) - y1
            max_h = int(math.floor((H - c) - y1))
            if max_h < int(min_protrude):
                continue
            add_h = min(add_h, max_h)

        # ✅ start 범위도 내부 영역을 만족하도록 제한 (특히 위/아래/좌/우 여유)
        if side in ("L", "R"):
            y_lo = int(math.ceil(max(c, y0)))
            y_hi = int(math.floor(min((H - c) - add_h, y1 - add_h)))
            if y_hi < y_lo:
                continue
            y_start = _ri(y_lo, y_hi)

            if side == "L":
                add_raw = box(x0 - add_w, y_start, x0, y_start + add_h)
            else:
                add_raw = box(x1, y_start, x1 + add_w, y_start + add_h)

        else:
            x_lo = int(math.ceil(max(c, x0)))
            x_hi = int(math.floor(min((W - c) - add_w, x1 - add_w)))
            if x_hi < x_lo:
                continue
            x_start = _ri(x_lo, x_hi)

            if side == "B":
                add_raw = box(x_start, y0 - add_h, x_start + add_w, y0)
            else:
                add_raw = box(x_start, y1, x_start + add_w, y1 + add_h)

        add_clip = _clip_to_bounds(add_raw, W, H)
        if add_clip is None or add_clip.geom_type != "Polygon" or add_clip.area < 6:
            continue

        # ✅ 최종 확정 조건: add는 반드시 내부에 완전히 포함(=벽 접촉/근접 불가)
        if not _add_inside_interior(add_clip):
            continue

        adds.append(add_clip)

    poly = unary_union([base] + adds)
    p = _clip_to_bounds(poly, W, H)
    if p is None or p.geom_type != "Polygon" or p.area < 20:
        return None

    return p



def _random_L_shape(W: int, H: int) -> Optional["Polygon"]:
    cx = _ri(int(0.20 * W), int(0.80 * W))
    cy = _ri(int(0.20 * H), int(0.80 * H))

    arm_th = _ri(4, 10)
    arm_len1 = _ri(int(0.18 * min(W, H)), int(0.45 * min(W, H)))
    arm_len2 = _ri(int(0.18 * min(W, H)), int(0.45 * min(W, H)))

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
    if p is None or p.geom_type != "Polygon" or p.area < 20:
        return None
    return p


def _random_U_shape(W: int, H: int, min_opening: int = 10) -> Optional["Polygon"]:
    cx = _ri(int(0.20 * W), int(0.80 * W))
    cy = _ri(int(0.20 * H), int(0.80 * H))

    th = _ri(4, 10)
    w = _ri(int(0.20 * W), int(0.40 * W))
    h = _ri(int(0.20 * H), int(0.40 * H))

    open_dir = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
    opening = (w - 2 * th) if open_dir in ("UP", "DOWN") else (h - 2 * th)
    if opening < min_opening:
        return None

    x0 = cx - w // 2
    y0 = cy - h // 2
    outer = box(x0, y0, x0 + w, y0 + h)

    if open_dir == "UP":
        cut = box(x0 + th, y0 + th, x0 + w - th, y0 + h)
    elif open_dir == "DOWN":
        cut = box(x0 + th, y0, x0 + w - th, y0 + h - th)
    elif open_dir == "LEFT":
        cut = box(x0, y0 + th, x0 + w - th, y0 + h - th)
    else:  # RIGHT
        cut = box(x0 + th, y0 + th, x0 + w, y0 + h - th)

    poly = outer.difference(cut)
    p = _clip_to_bounds(poly, W, H)
    if p is None or p.geom_type != "Polygon" or p.area < 20:
        return None
    return p


def _random_wall_rect_obstacle(W: int, H: int, params: Dict[str, float], margin: int = 1) -> "Polygon":
    w_min = int(params.get("rect_w_min", 12))
    w_max = int(params.get("rect_w_max", 32))
    h_min = int(params.get("rect_h_min", 12))
    h_max = int(params.get("rect_h_max", 32))

    side = random.choice(["left", "right", "bottom", "top"])

    if side in ("left", "right"):
        depth = _ri(w_min, w_max)
        along = _ri(h_min, h_max)
        y0 = _ri(margin, max(margin, H - margin - along))
        y1 = y0 + along
        if side == "left":
            x0, x1 = 0, min(W, depth)
        else:
            x1, x0 = W, max(0, W - depth)
        return box(x0, y0, x1, y1)

    depth = _ri(h_min, h_max)
    along = _ri(w_min, w_max)
    x0 = _ri(margin, max(margin, W - margin - along))
    x1 = x0 + along
    if side == "bottom":
        y0, y1 = 0, min(H, depth)
    else:
        y1, y0 = H, max(0, H - depth)
    return box(x0, y0, x1, y1)

def _random_large_convex_obstacle(W: int, H: int, params: Dict[str, float], margin: int = 1) -> Optional["Polygon"]:
    """
    large convex 생성 개선:
    - 기존: sample box 내부에서 균일 랜덤 점 -> 얇은 hull 빈번
    - 개선: 타원(ellipse) 주변에서 각도 샘플링 + 약간의 radial/xy 노이즈 -> 둥근 hull 확률↑
    """
    smin = float(params.get("large_conv_scale_min", 0.22))
    smax = float(params.get("large_conv_scale_max", 0.45))
    if smin > smax:
        smin, smax = smax, smin

    # ✅ 벽 규칙(7 이내면 touch해야 함) 때문에, large_conv는 애초에 벽에서 clearance만큼 떨어져 생성되게 하는 편이 안전
    wall_clearance = float(params.get("wall_clearance", 7.0))
    safe_margin = max(int(math.ceil(wall_clearance)), int(margin))

    sw = int(_rf(smin, smax) * W)
    sh = int(_rf(smin, smax) * H)
    sw = max(10, min(W - 2 * safe_margin, sw))
    sh = max(10, min(H - 2 * safe_margin, sh))
    if sw < 10 or sh < 10:
        return None

    # sample box center (맵 전역 허용하되, safe_margin 고려)
    cx = _ri(safe_margin + sw // 2, W - safe_margin - sw // 2)
    cy = _ri(safe_margin + sh // 2, H - safe_margin - sh // 2)

    # ellipse radii (박스 대비 조금 안쪽)
    rx = max(5.0, 0.45 * sw)
    ry = max(5.0, 0.45 * sh)

    nmin = int(params.get("large_conv_n_min", 6))
    nmax = int(params.get("large_conv_n_max", 12))
    n = _ri(nmin, nmax)

    # 노이즈 크기 (너무 크면 다시 찌그러짐)
    # - radial jitter: 반지름을 ±j 만큼 흔듦
    # - xy jitter: 좌표에 작은 가우시안 노이즈
    radial_jitter = float(params.get("large_conv_radial_jitter", 0.18))  # 0~0.3 권장
    xy_jitter = float(params.get("large_conv_xy_jitter", 0.8))           # 0~2 정도 권장

    pts = []
    for _ in range(n):
        theta = random.random() * (2.0 * math.pi)

        # radial factor: 1 ± jitter
        rfac = 1.0 + random.uniform(-radial_jitter, radial_jitter)

        x = cx + (rx * rfac) * math.cos(theta) + random.gauss(0.0, xy_jitter)
        y = cy + (ry * rfac) * math.sin(theta) + random.gauss(0.0, xy_jitter)

        # bounds clamp
        x = max(safe_margin, min(W - safe_margin, x))
        y = max(safe_margin, min(H - safe_margin, y))
        pts.append((x, y))

    hull = Polygon(pts).convex_hull
    p = _clip_to_bounds(hull, W, H)
    if p is None or p.geom_type != "Polygon":
        return None

    minw = float(params.get("large_conv_min_width", 14.0))
    if not _passes_convex_quality(p, min_width=minw):
        return None

    return p

def _iter_polys(g):
    if g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if g.geom_type == "MultiPolygon":
        return [p for p in g.geoms if p.geom_type == "Polygon"]
    return []

def _passes_wall_rules_strict_by_side(cand: "Polygon", shape_tag: str, W: int, H: int, wall_clearance: float) -> bool:
    c = float(wall_clearance)

    # 4개 벽 line
    left_wall   = LineString([(0, 0), (0, H)])
    right_wall  = LineString([(W, 0), (W, H)])
    bottom_wall = LineString([(0, 0), (W, 0)])
    top_wall    = LineString([(0, H), (W, H)])

    checks = [
        ("left",   box(0, 0, c, H),         left_wall),
        ("right",  box(W - c, 0, W, H),     right_wall),
        ("bottom", box(0, 0, W, c),         bottom_wall),
        ("top",    box(0, H - c, W, H),     top_wall),
    ]

    # wall_rect: 어떤 벽엔가 붙어있어야 함
    if shape_tag == "wall_rect":
        return cand.touches(left_wall) or cand.touches(right_wall) or cand.touches(bottom_wall) or cand.touches(top_wall)

    # L/U: 벽에 닿으면 안 되고, 모든 벽에서 clearance 이상
    if shape_tag in ("L", "U"):
        for _, band, wall in checks:
            if cand.intersects(band):  # band 안으로 들어오면(=dist < c)
                return False
            if cand.touches(wall):
                return False
        return True

    # default: band 안에 들어온 "각 컴포넌트"는 반드시 그 벽을 touches 해야 함
    for _, band, wall in checks:
        inter = cand.intersection(band)
        comps = _iter_polys(inter)
        if not comps:
            continue

        for comp in comps:
            # band 안에 있는 조각이 벽을 안 touches 하면 = main이 애매하게 떠있는 케이스
            if not comp.touches(wall):
                return False

    return True

def _random_small_rect_obstacle(W: int, H: int, params: Dict[str, float], margin: int = 1) -> "Polygon":
    w = _ri(int(params.get("small_rect_w_min", 6)), int(params.get("small_rect_w_max", 16)))
    h = _ri(int(params.get("small_rect_h_min", 6)), int(params.get("small_rect_h_max", 16)))
    x0 = _ri(margin, max(margin, W - margin - w))
    y0 = _ri(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)


def _random_corridor_strip(W: int, H: int, margin: int = 1) -> "Polygon":
    thickness = _ri(4, 10)
    length = _ri(int(0.30 * min(W, H)), int(0.80 * min(W, H)))
    horizontal = (random.random() < 0.5)

    if horizontal:
        w, h = length, thickness
    else:
        w, h = thickness, length

    x0 = _ri(margin, max(margin, W - margin - w))
    y0 = _ri(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)


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


def _random_convex_obstacle(W: int, H: int, margin: int = 1) -> "Polygon":
    n = _ri(4, 9)
    pts = [(_ri(margin, W - margin), _ri(margin, H - margin)) for _ in range(n)]
    return Polygon(pts).convex_hull


# -----------------------------
# Large/Small pickers (quota-based)
# -----------------------------
def _pick_large_obstacle(W: int, H: int, params: Dict[str, float]) -> Tuple[Optional["Polygon"], List["Polygon"], str]:
    """
    returns: (main_poly, extras, shape_tag)
    shape_tag: "large_block" | "L" | "U" | "wall_rect" | "large_conv"
    """
    extras: List["Polygon"] = []

    names = ["large_block", "L", "U", "wall_rect", "large_conv"]
    weights = [
        float(params.get("large_block_bias", 0.40)),
        float(params.get("L_shape_bias", 0.25)),
        float(params.get("U_shape_bias", 0.10)),
        float(params.get("wall_rect_bias", 0.15)),
        float(params.get("large_conv_bias", 0.10)),
    ]

    if sum(weights) <= 0:
        return _random_large_block(W, H, params), extras, "large_block"

    choice = random.choices(names, weights=weights, k=1)[0]

    if choice == "large_block":
        return _random_large_block(W, H, params), extras, "large_block"
    if choice == "L":
        return _random_L_shape(W, H), extras, "L"
    if choice == "U":
        return _random_U_shape(W, H, min_opening=10), extras, "U"
    if choice == "wall_rect":
        return _random_wall_rect_obstacle(W, H, params, margin=1), extras, "wall_rect"

    return _random_large_convex_obstacle(W, H, params, margin=1), extras, "large_conv"


def _pick_small_obstacle(W: int, H: int, params: Dict[str, float]) -> Tuple["Polygon", List["Polygon"], str]:
    """
    returns: (main_poly, extras, shape_tag)
    shape_tag: "s_rect" | "corr" | "conv"
    """
    extras: List["Polygon"] = []

    weights = [
        float(params.get("small_rect_bias", 0.50)),
        float(params.get("small_corr_bias", 0.30)),
        float(params.get("small_conv_bias", 0.20)),
    ]
    names = ["s_rect", "corr", "conv"]
    choice = random.choices(names, weights=weights, k=1)[0]

    if choice == "s_rect":
        return _random_small_rect_obstacle(W, H, params, margin=1), extras, "s_rect"

    if choice == "corr":
        base = _random_corridor_strip(W, H, margin=1)

        x0, y0, x1, y1 = base.bounds
        w = max(1e-6, (x1 - x0))
        h = max(1e-6, (y1 - y0))
        aspect = max(w / h, h / w)
        if aspect > float(params.get("max_corridor_aspect", 6.0)):
            return _random_small_rect_obstacle(W, H, params, margin=1), extras, "s_rect"

        if random.random() < float(params.get("deadend_bias", 0.2)):
            extras.append(_random_deadend_cap_near(base, W, H, margin=1))
        return base, extras, "corr"

    # conv: 너무 얇은 hull reject + fallback
    minw = float(params.get("conv_min_width", 12.0))
    for _ in range(40):
        p = _random_convex_obstacle(W, H, margin=1)
        if _passes_convex_quality(p, min_width=minw):
            return p, extras, "conv"
    return _random_small_rect_obstacle(W, H, params, margin=1), extras, "s_rect"


# -----------------------------
# Enclosed pocket check (grid flood fill)
# -----------------------------
def _has_enclosed_free_pocket_grid(W: int, H: int, blocked_union, grid_step: int = 2, pinch_cells: int = 1) -> bool:
    if blocked_union.is_empty:
        return False

    P = prep(blocked_union)

    nx = W // grid_step + 1
    ny = H // grid_step + 1

    occ = [[False] * ny for _ in range(nx)]

    for ix in range(nx):
        x = ix * grid_step
        for iy in range(ny):
            y = iy * grid_step
            pt = ShPoint(x, y)
            if P.contains(pt) or P.touches(pt):
                occ[ix][iy] = True

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

    q = deque()
    vis = [[False] * ny for _ in range(nx)]

    def try_push(ix, iy):
        if 0 <= ix < nx and 0 <= iy < ny and (not occ[ix][iy]) and (not vis[ix][iy]):
            vis[ix][iy] = True
            q.append((ix, iy))

    for ix in range(nx):
        try_push(ix, 0)
        try_push(ix, ny - 1)
    for iy in range(ny):
        try_push(0, iy)
        try_push(nx - 1, iy)

    while q:
        ix, iy = q.popleft()
        try_push(ix + 1, iy)
        try_push(ix - 1, iy)
        try_push(ix, iy + 1)
        try_push(ix, iy - 1)

    for ix in range(nx):
        for iy in range(ny):
            if (not occ[ix][iy]) and (not vis[ix][iy]):
                return True
    return False


# -----------------------------
# Constraint checks
# -----------------------------
def _passes_wall_rules(cand: "Polygon", shape_tag: str, boundary, wall_clearance: float) -> bool:
    d_wall = cand.distance(boundary)

    if shape_tag == "wall_rect":
        return cand.touches(boundary)

    if shape_tag in ("L", "U"):
        if cand.touches(boundary):
            return False
        if d_wall < wall_clearance:
            return False
        return True

    # default (incl. large_block, large_conv, small)
    if d_wall < wall_clearance and (not cand.touches(boundary)):
        return False
    return True


def _passes_exit_rules(cand: "Polygon", exits: List["Polygon"], keep_gap_from_exits: float) -> bool:
    for e in exits:
        if cand.intersects(e):
            return False
        if keep_gap_from_exits > 0 and cand.distance(e) < keep_gap_from_exits:
            return False
    return True

def _creates_wall_pocket(cand: "Polygon", W: int, H: int, wall_clearance: float) -> bool:
    """
    벽과 장애물로 인해 wall_clearance 밴드 내부에
    '끝(코너 방향)으로 연결되지 않는 고립된 free component'가 생기면 True(=reject).

    - left/right band: free component가 y=0 또는 y=H 중 하나라도 touch해야 통로로 간주
    - bottom/top band: free component가 x=0 또는 x=W 중 하나라도 touch해야 통로로 간주
    """
    eps = 1e-6
    c = float(wall_clearance)

    def _iter_polys(g):
        if g.is_empty:
            return []
        if g.geom_type == "Polygon":
            return [g]
        if g.geom_type == "MultiPolygon":
            return [p for p in g.geoms if p.geom_type == "Polygon"]
        return []

    # 각 band 정의
    bands = {
        "left":   box(0, 0, c, H),
        "right":  box(W - c, 0, W, H),
        "bottom": box(0, 0, W, c),
        "top":    box(0, H - c, W, H),
    }

    for side, band in bands.items():
        # cand가 이 band랑 아예 무관하면 스킵
        if cand.intersects(band) is False:
            continue

        free = band.difference(cand)
        comps = _iter_polys(free)
        if not comps:
            # 밴드 자체가 막힌 경우도 '포켓/차단'이라 보고 reject
            return True

        for comp in comps:
            x0, y0, x1, y1 = comp.bounds

            if side in ("left", "right"):
                touches_end = (abs(y0 - 0.0) < eps) or (abs(y1 - float(H)) < eps)
                if not touches_end:
                    return True

            else:  # bottom/top
                touches_end = (abs(x0 - 0.0) < eps) or (abs(x1 - float(W)) < eps)
                if not touches_end:
                    return True

    return False

def _creates_wall_pocket_deep(
    cand: "Polygon",
    W: int,
    H: int,
    wall_clearance: float,
    depth: Optional[float] = None,
) -> bool:
    """
    main+add+wall 조합에서 생기는 '깊은 pocket'까지 잡기 위한 강화 버전.

    - 각 벽마다 '두꺼운 band(깊이 depth)'를 잡고,
      band 안의 free-space component 중에서
        (A) 코너 방향 끝(end)에 닿지도 않고
        (B) band의 안쪽 경계(inner boundary)에도 닿지 않으면
      => pocket 으로 판단(reject).

    해석:
    - end로 닿으면: 벽을 따라 코너로 빠져나갈 수 있는 통로
    - inner boundary로 닿으면: 실내 쪽(맵 내부)으로 연결되는 통로
    - 둘 다 아니면: 벽-장애물 사이에 갇힌 pocket
    """
    eps = 1e-6
    c = float(wall_clearance)

    # depth 기본값: wall_clearance보다 확실히 크게
    # (pocket이 7보다 깊게 생기는 케이스를 잡아야 하므로)
    if depth is None:
        depth = max(3.0 * c, 18.0)  # 7이면 21, 최소 18
    d = float(depth)

    # 맵이 작으면 band가 말이 안되니까 제한
    d = min(d, (W * 0.45), (H * 0.45))
    if d <= c + 1.0:
        d = c + 2.0

    def _iter_polys(g):
        if g.is_empty:
            return []
        if g.geom_type == "Polygon":
            return [g]
        if g.geom_type == "MultiPolygon":
            return [p for p in g.geoms if p.geom_type == "Polygon"]
        return []

    # band와 inner boundary line
    bands = {
        "left":   (box(0, 0, d, H),            LineString([(d, 0), (d, H)])),
        "right":  (box(W - d, 0, W, H),        LineString([(W - d, 0), (W - d, H)])),
        "bottom": (box(0, 0, W, d),            LineString([(0, d), (W, d)])),
        "top":    (box(0, H - d, W, H),        LineString([(0, H - d), (W, H - d)])),
    }

    for side, (band, inner_line) in bands.items():
        if not cand.intersects(band):
            continue

        free = band.difference(cand)
        comps = _iter_polys(free)
        if not comps:
            # band 자체가 완전히 막힌 경우도 pocket/차단으로 봄
            return True

        for comp in comps:
            x0, y0, x1, y1 = comp.bounds

            # end(코너 방향) 연결성
            if side in ("left", "right"):
                touches_end = (abs(y0 - 0.0) < eps) or (abs(y1 - float(H)) < eps)
            else:
                touches_end = (abs(x0 - 0.0) < eps) or (abs(x1 - float(W)) < eps)

            # band 안쪽 경계(inner boundary) 연결성
            # bounds만으로도 대부분 잡히지만, 안정성을 위해 line 교차도 같이 봄
            touches_inner = comp.intersects(inner_line)

            # "pocket 성분"은 보통 벽에 인접해 있으므로,
            # 벽쪽에 붙어있는 free component만 검사해도 된다.
            # (안쪽에 떠있는 free component는 의미가 약함)
            if side == "left":
                touches_wall = (abs(x0 - 0.0) < eps)
            elif side == "right":
                touches_wall = (abs(x1 - float(W)) < eps)
            elif side == "bottom":
                touches_wall = (abs(y0 - 0.0) < eps)
            else:
                touches_wall = (abs(y1 - float(H)) < eps)

            if touches_wall and (not touches_end) and (not touches_inner):
                return True

    return False


# -----------------------------
# Core obstacle generation (quota + density)
# -----------------------------
def generate_obstacles_with_density_quota(
    W: int, H: int,
    exits: List["Polygon"],
    density_min: float,
    density_max: float,
    params: Dict[str, float],
    max_tries: int,
    initial_obstacles: Optional[List["Polygon"]] = None,
) -> List["Polygon"]:
    map_area = float(W * H)
    target_density = random.uniform(density_min, density_max)
    target_area = target_density * map_area  # (참고용)

    n_large = random.randint(int(params["large_count_min"]), int(params["large_count_max"]))
    n_small = random.randint(int(params["small_count_min"]), int(params["small_count_max"]))

    min_gap = float(params.get("min_obstacle_gap", 5.0))
    conv_gap_boost = float(params.get("conv_gap_boost", 1.5))
    keep_gap_from_exits = float(params.get("keep_gap_from_exits", 0.0))
    wall_clearance = float(params.get("wall_clearance", 7.0))
    density_upper_guard = density_max * map_area * 1.05

    boundary = box(0, 0, W, H).boundary

    obstacles: List["Polygon"] = []
    total_area = 0.0

    if initial_obstacles:
        obstacles = list(initial_obstacles)
        total_area = float(sum(p.area for p in obstacles))

    # ---------- Phase 1: LARGE ----------
    placed_large = 0
    tries = 0
    while placed_large < n_large and tries < max_tries:
        tries += 1

        main, extras, shape_tag = _pick_large_obstacle(W, H, params)
        if main is None:
            continue

        merged = unary_union([main] + extras)
        if merged.geom_type != "Polygon":
            continue
        cand = merged

        if not _valid_polygon(cand, min_area=20.0):
            continue
        if not _passes_wall_rules_strict_by_side(cand, shape_tag, W, H, wall_clearance):
            continue

        if not _passes_wall_rules_strict_by_side(cand, "small", W, H, wall_clearance):
            continue

                # (2) 벽 포켓 금지: 특히 large_block/벽에 붙는 계열에서 중요
        if shape_tag in ("large_block", "wall_rect"):
            pocket_depth = float(params.get("wall_pocket_depth", max(3.0 * wall_clearance, 18.0)))
            if _creates_wall_pocket_deep(cand, W, H, wall_clearance, depth=pocket_depth):
                continue

        if not _passes_exit_rules(cand, exits, keep_gap_from_exits):
            continue

        local_gap = (min_gap * conv_gap_boost) if (shape_tag == "large_conv") else min_gap
        if _too_close_or_intersect(cand, obstacles, min_gap=local_gap):
            continue

        if (total_area + cand.area) > density_upper_guard:
            continue

        # pocket check을 너무 자주 하면 느려져서 샘플링
        if (placed_large % 6 == 0) or (tries % 6 == 0):
            blocked = unary_union(obstacles + [cand])
            if _has_enclosed_free_pocket_grid(W, H, blocked, grid_step=2, pinch_cells=1):
                continue

        obstacles.append(cand)
        total_area += cand.area
        placed_large += 1

    if placed_large < n_large:
        raise RuntimeError(f"Failed to place enough LARGE obstacles: {placed_large}/{n_large}")

    # ---------- Phase 2: SMALL ----------
    placed_small = 0
    tries = 0
    while placed_small < n_small and tries < max_tries:
        tries += 1

        main, extras, shape_tag = _pick_small_obstacle(W, H, params)
        merged = unary_union([main] + extras)
        if merged.geom_type != "Polygon":
            continue
        cand = merged

        if not _valid_polygon(cand, min_area=10.0):
            continue
        if not _passes_wall_rules(cand, "small", boundary, wall_clearance):
            continue
        if not _passes_exit_rules(cand, exits, keep_gap_from_exits):
            continue

        local_gap = (min_gap * conv_gap_boost) if (shape_tag == "conv") else min_gap
        if _too_close_or_intersect(cand, obstacles, min_gap=local_gap):
            continue

        if (total_area + cand.area) > density_upper_guard:
            continue

        blocked = unary_union(obstacles + [cand])
        if _has_enclosed_free_pocket_grid(W, H, blocked, grid_step=2, pinch_cells=1):
            continue

        obstacles.append(cand)
        total_area += cand.area
        placed_small += 1

    if placed_small < n_small:
        raise RuntimeError(f"Failed to place enough SMALL obstacles: {placed_small}/{n_small}")

    final_density = total_area / map_area if map_area > 0 else 0.0
    if final_density < density_min:
        raise RuntimeError(f"Density too low: {final_density:.3f} < {density_min}")

    return obstacles


# -----------------------------
# Public API: generate_map(spec)
# -----------------------------
def generate_map(spec: RandomMapSpec) -> MapData:
    _require_shapely()

    seed_used = spec.seed if spec.seed is not None else random.randrange(0, 2**31 - 1)
    random.seed(seed_used)

    W, H = int(spec.width), int(spec.height)
    params = _params_from_difficulty(W, H, spec.difficulty)

    for _ in range(spec.map_level_max_tries):
        exits = generate_two_exits(
            W, H,
            along_min=int(params["exit_along_min"]),
            along_max=int(params["exit_along_max"]),
            depth_min=int(params["exit_depth_min"]),
            depth_max=int(params["exit_depth_max"]),
            min_exit_distance=float(params["min_exit_distance"]),
            min_exit_center_distance=float(params.get("min_exit_center_distance", 40.0)),
            corner_avoid_dist=float(params["corner_avoid_dist"]),
            disallow_same_side=bool(int(params["disallow_same_side"])),
            max_tries=spec.exit_max_tries,
        )

        exit_flanks = generate_exit_flanks(
            W, H, exits,
            bias=float(params.get("exit_flank_bias", 0.3)),
            gap_min=int(params.get("exit_flank_gap_min", 2)),
            gap_max=int(params.get("exit_flank_gap_max", 4)),
            flank_len_min=int(params.get("exit_flank_len_min", 5)),
            flank_len_max=int(params.get("exit_flank_len_max", 40)),
            depth_boost_min=int(params.get("exit_flank_depth_boost_min", 10)),
            depth_boost_max=int(params.get("exit_flank_depth_boost_max", 30)),
            margin=0,
        )

        try:
            obstacles = generate_obstacles_with_density_quota(
                W, H,
                exits=exits,
                density_min=float(params["density_min"]),
                density_max=float(params["density_max"]),
                params=params,
                max_tries=spec.obstacle_level_max_tries,
                initial_obstacles=exit_flanks
            )
        except RuntimeError:
            continue

        obstacles_pts = [_poly_to_points_int(o) for o in obstacles]

        exits_pts: List[List[Tuple[int, int]]] = []
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


if __name__ == "__main__":
    spec = RandomMapSpec(width=100, height=100, difficulty=2, seed=1234)
    m = generate_map(spec)
    print("seed_used:", m.seed_used)
    print("obstacles:", len(m.obstacles))
    print("exits:", len(m.exits))
