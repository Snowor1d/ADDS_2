#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
random_map.py (Shapely / unary_union 기반) - difficulty 단일 입력 + Large/Small obstacle quota + large_conv 추가 버전

추가 반영:
- ✅ Large에도 convex(large_conv) 추가
  * 파라미터: large_conv_bias
  * 크기(스케일): large_conv_scale_min/max (conv hull 샘플 포인트 생성 범위)
  * convex 생성 시 내부 "샘플 박스"를 두고 그 안에서 점을 뽑아 convex_hull 생성 -> 큰 convex가 안정적으로 나옴

- ✅ 요청 반영: convex로 인해 너무 "좁은 틈"이 생기는 문제 완화
  1) convex 자체의 "최소 폭(min width)" 품질 필터 추가 (minimum_rotated_rectangle 기반)
     - conv_min_width (small conv)
     - large_conv_min_width (large conv)
  2) convex가 선택된 경우에만 min_gap을 더 크게 요구 (conv_gap_boost)

기존 요구사항 유지:
1) Exit: along 5~10, depth 4~5
2) wall_rect_bias
3) RandomMapSpec: (width/height/seed/difficulty)만
4) 장애물 갯수 ↓, 크기 ↑ (difficulty table)
5) Large/Small quota 기반
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
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
    difficulty: int = 6          # 1=easy, 2=medium, 3=hard
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


def _params_from_difficulty(W: int, H: int, difficulty: int) -> Dict[str, float]:
    """
    difficulty에 따라 모든 파라미터가 결정됨.

    ✅ Large convex 추가:
      - large_conv_bias
      - large_conv_scale_min/max
      - large_conv_n_min/max

    ✅ Convex로 생기는 '좁은 틈' 완화:
      - conv_min_width / large_conv_min_width: 너무 얇은 convex reject
      - conv_gap_boost: convex일 때만 min_gap을 더 크게 요구
    """
    d = int(difficulty)

    if d <= 1:  # EASY
        return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.1, "density_max": 0.2,

        "large_count_min": 1, "large_count_max": 2,
        "small_count_min": 2, "small_count_max": 4,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
        }
    
    if d <= 2:  # EASY
        return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.15, "density_max": 0.25,

        "large_count_min": 1, "large_count_max": 2,
        "small_count_min": 2, "small_count_max": 4,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
        }
    
    if d <= 3:  # EASY
        return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.20, "density_max": 0.30,

        "large_count_min": 1, "large_count_max": 3,
        "small_count_min": 2, "small_count_max": 4,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
        }
    


    if d == 4:  # MEDIUM
        return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.25, "density_max": 0.35,

        "large_count_min": 1, "large_count_max": 3,
        "small_count_min": 3, "small_count_max": 6,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
        }
    

    if d == 5:  # MEDIUM
        return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.30, "density_max": 0.40,

        "large_count_min": 1, "large_count_max": 3,
        "small_count_min": 3, "small_count_max": 6,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
        }

    # HARD
    return {
        "exit_along_min": 5, "exit_along_max": 10,
        "exit_depth_min": 4, "exit_depth_max": 5,
        "min_exit_distance": 16.0,
        "corner_avoid_dist": 12.0,
        "disallow_same_side": 1.0,

        "density_min": 0.35, "density_max": 0.50,

        "large_count_min": 1, "large_count_max": 3,
        "small_count_min": 3, "small_count_max": 6,
        "density_large_ratio": 0.4,

        "min_obstacle_gap": 6.0,
        "keep_gap_from_exits": 0.0,
        "wall_clearance": 7.0,

        "main_block_bias": 0.4,
        "L_shape_bias": 0.15,
        "U_shape_bias": 0.1,
        "wall_rect_bias": 0.25,
        "large_conv_bias": 0.3,

        "small_rect_bias": 0.45,
        "small_corr_bias": 0.45,
        "small_conv_bias": 0.20,
        "deadend_bias": 0.25,
        "max_corridor_aspect": 6.0,

        "rect_w_min": 25, "rect_w_max": 60,
        "rect_h_min": 25, "rect_h_max": 60,
        "small_rect_w_min": 6, "small_rect_w_max": 15,
        "small_rect_h_min": 6, "small_rect_h_max": 15,

        "main_bw_min": 0.22, "main_bw_max": 0.60,
        "main_bh_min": 0.22, "main_bh_max": 0.60,

        "large_conv_scale_min": 0.22,
        "large_conv_scale_max": 0.50,
        "large_conv_n_min": 6,
        "large_conv_n_max": 12,

        "conv_min_width": 14.0,
        "large_conv_min_width": 16.0,
        "conv_gap_boost": 4,
    }


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


# -----------------------------
# Obstacle primitives (Large / Small)
# -----------------------------
def _random_main_block(W: int, H: int, params: Dict[str, float]) -> Optional["Polygon"]:
    cx = _ri(int(0.30 * W), int(0.70 * W))
    cy = _ri(int(0.30 * H), int(0.70 * H))

    bw = _ri(int(params.get("main_bw_min", 0.25) * W), int(params.get("main_bw_max", 0.42) * W))
    bh = _ri(int(params.get("main_bh_min", 0.25) * H), int(params.get("main_bh_max", 0.42) * H))

    x0 = cx - bw // 2
    y0 = cy - bh // 2
    base = box(x0, y0, x0 + bw, y0 + bh)

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
    if p is None or p.geom_type != "Polygon":
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
    ✅ Large convex:
    - 지도 중앙 근처에 '샘플 박스'를 하나 잡고, 그 박스 내부에서 점들을 샘플링해 convex_hull.
    - scale_min/max는 샘플 박스 크기 스케일(지도 대비).

    ✅ 추가: 너무 얇은 hull은 reject (large_conv_min_width)
    """
    smin = float(params.get("large_conv_scale_min", 0.22))
    smax = float(params.get("large_conv_scale_max", 0.45))
    if smin > smax:
        smin, smax = smax, smin
    sw = int(_rf(smin, smax) * W)
    sh = int(_rf(smin, smax) * H)
    sw = max(8, min(W - 2 * margin, sw))
    sh = max(8, min(H - 2 * margin, sh))

    # sample box center-ish
    cx = _ri(int(0.25 * W), int(0.75 * W))
    cy = _ri(int(0.25 * H), int(0.75 * H))
    x0 = max(margin, cx - sw // 2)
    y0 = max(margin, cy - sh // 2)
    x1 = min(W - margin, x0 + sw)
    y1 = min(H - margin, y0 + sh)

    if x1 - x0 < 6 or y1 - y0 < 6:
        return None

    nmin = int(params.get("large_conv_n_min", 5))
    nmax = int(params.get("large_conv_n_max", 12))
    n = _ri(nmin, nmax)

    pts = [(_ri(x0, x1), _ri(y0, y1)) for _ in range(n)]
    hull = Polygon(pts).convex_hull
    p = _clip_to_bounds(hull, W, H)
    if p is None or p.geom_type != "Polygon":
        return None

    minw = float(params.get("large_conv_min_width", 14.0))
    if not _passes_convex_quality(p, min_width=minw):
        return None

    return p


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
    shape_tag: "main" | "L" | "U" | "wall_rect" | "large_conv"
    """
    extras: List["Polygon"] = []

    names = ["main", "L", "U", "wall_rect", "large_conv"]
    weights = [
        float(params.get("main_block_bias", 0.40)),
        float(params.get("L_shape_bias", 0.25)),
        float(params.get("U_shape_bias", 0.10)),
        float(params.get("wall_rect_bias", 0.15)),
        float(params.get("large_conv_bias", 0.10)),
    ]

    if sum(weights) <= 0:
        return _random_main_block(W, H, params), extras, "main"

    choice = random.choices(names, weights=weights, k=1)[0]

    if choice == "main":
        return _random_main_block(W, H, params), extras, "main"
    if choice == "L":
        return _random_L_shape(W, H), extras, "L"
    if choice == "U":
        return _random_U_shape(W, H, min_opening=10), extras, "U"
    if choice == "wall_rect":
        return _random_wall_rect_obstacle(W, H, params, margin=1), extras, "wall_rect"

    # large_conv: 생성 실패(얇음 reject 등) 가능 -> None 반환 허용
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

    # default (incl. large_conv, main, small)
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
) -> List["Polygon"]:
    map_area = float(W * H)
    target_density = random.uniform(density_min, density_max)
    target_area = target_density * map_area

    n_large = random.randint(int(params["large_count_min"]), int(params["large_count_max"]))
    n_small = random.randint(int(params["small_count_min"]), int(params["small_count_max"]))

    large_ratio = float(params.get("density_large_ratio", 0.60))
    large_ratio = max(0.0, min(1.0, large_ratio))

    min_gap = float(params.get("min_obstacle_gap", 5.0))
    conv_gap_boost = float(params.get("conv_gap_boost", 1.5))
    keep_gap_from_exits = float(params.get("keep_gap_from_exits", 0.0))
    wall_clearance = float(params.get("wall_clearance", 7.0))
    density_upper_guard = density_max * map_area * 1.05

    boundary = box(0, 0, W, H).boundary

    obstacles: List["Polygon"] = []
    total_area = 0.0

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

        if not _passes_wall_rules(cand, shape_tag, boundary, wall_clearance):
            continue
        if not _passes_exit_rules(cand, exits, keep_gap_from_exits):
            continue

        local_gap = min_gap * conv_gap_boost if shape_tag == "large_conv" else min_gap
        if _too_close_or_intersect(cand, obstacles, min_gap=local_gap):
            continue

        if (total_area + cand.area) > density_upper_guard:
            continue

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

        local_gap = min_gap * conv_gap_boost if shape_tag == "conv" else min_gap
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
            min_exit_center_distance = float(params.get("min_exit_center_distance", 40.0)),
            corner_avoid_dist=float(params["corner_avoid_dist"]),
            disallow_same_side=bool(int(params["disallow_same_side"])),
            max_tries=spec.exit_max_tries,
        )

        try:
            obstacles = generate_obstacles_with_density_quota(
                W, H,
                exits=exits,
                density_min=float(params["density_min"]),
                density_max=float(params["density_max"]),
                params=params,
                max_tries=spec.obstacle_level_max_tries,
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
