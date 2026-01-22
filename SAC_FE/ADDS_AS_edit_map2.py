#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDS Map Editor (Pygame) + Random Map Generator (Shapely)
- Polygon by clicking points (ENTER finalize)
- Rectangle by drag (auto 4 points)
- Delete tool: click inside polygon to remove
- Undo for add/delete
- Save format matches your style (JSON):
    {
      "width": 100,
      "height": 100,
      "obstacles": [ [[x,y],...], ... ],
      "exits": [ [(x,y),...], ... ]
    }

[NEW]
- Panel controls for random generation:
  * obstacle density min/max (sliders)
  * obstacle count min/max (inputs)
  * corridor bias (slider)
  * dead-end bias (slider)
  * min obstacle gap (input)
  * exit min distance (input)
  * corner avoid distance (input)
- Button: "Generate Random Map" (and hotkey R)
- After generation, press S to save
"""

import os
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import random
import pygame

# -------- Optional Shapely --------
try:
    from shapely.geometry import Polygon, box, Point as ShPoint
    from shapely.ops import unary_union
    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False

MAP_INFO_DIR = os.path.join(os.getcwd(), "map_infos")

def path_for_mapnum(n: int) -> str:
    return os.path.join(MAP_INFO_DIR, f"map_{n}.json")


WINDOW_W, WINDOW_H = 1280, 900
PANEL_W = 420
CANVAS_BG = (245, 245, 245)

GRID_ON = True
GRID_STEP = 2          # snap step in map units
GRID_DRAW_STEP = 2     # grid lines step in map units
POINT_RADIUS = 5

OBSTACLE_FILL_ALPHA = 70
EXIT_FILL_ALPHA = 90

Point = Tuple[int, int]
PolygonPts = List[Point]


@dataclass
class MapData:
    width: int
    height: int
    obstacles: List[List[List[int]]]          # [[[x,y],...], ...]
    exits: List[List[Tuple[int, int]]]        # [[(x,y),...], ...]


# -----------------------------
# Transform
# -----------------------------
class Transform:
    def __init__(self, map_w: int, map_h: int, canvas_rect: pygame.Rect, margin: int = 30):
        self.map_w = map_w
        self.map_h = map_h
        self.canvas = canvas_rect
        self.margin = margin

        avail_w = canvas_rect.width - 2 * margin
        avail_h = canvas_rect.height - 2 * margin
        self.scale = min(avail_w / max(1, map_w), avail_h / max(1, map_h))

        self.origin_x = canvas_rect.x + margin
        self.origin_y = canvas_rect.y + margin

    def map_to_px(self, p: Point) -> Tuple[int, int]:
        x, y = p
        px = int(self.origin_x + x * self.scale)
        py = int(self.origin_y + (self.map_h - y) * self.scale)
        return px, py

    def px_to_map(self, px: int, py: int) -> Point:
        x = (px - self.origin_x) / max(1e-9, self.scale)
        y = self.map_h - (py - self.origin_y) / self.scale
        return int(round(x)), int(round(y))


def clamp_point(p: Point, w: int, h: int) -> Point:
    x, y = p
    x = max(0, min(w, x))
    y = max(0, min(h, y))
    return x, y


def snap_point(p: Point, step: int) -> Point:
    x, y = p
    sx = int(round(x / step) * step)
    sy = int(round(y / step) * step)
    return sx, sy


# -----------------------------
# Geometry
# -----------------------------
def point_in_poly(pt: Point, poly: PolygonPts) -> bool:
    """Ray casting. Works for convex/concave, assumes non-self-intersecting."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def rect_from_drag(a: Point, b: Point) -> PolygonPts:
    x1, y1 = a
    x2, y2 = b
    lo_x, hi_x = sorted([x1, x2])
    lo_y, hi_y = sorted([y1, y2])
    return [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]


# -----------------------------
# Save/Load
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_map_snippet(path: str, data: MapData) -> None:
    obj = {
        "width": data.width,
        "height": data.height,
        "obstacles": data.obstacles,
        "exits": data.exits
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_map_snippet(path: str) -> MapData:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    width = int(obj["width"])
    height = int(obj["height"])

    obstacles_raw = obj.get("obstacles", [])
    exits_raw = obj.get("exits", [])

    obstacles: List[List[List[int]]] = []
    for poly in obstacles_raw:
        if len(poly) >= 3:
            obstacles.append([[int(x), int(y)] for x, y in poly])

    exits: List[List[Tuple[int, int]]] = []
    for poly in exits_raw:
        if len(poly) >= 3:
            exits.append([(int(x), int(y)) for x, y in poly])

    return MapData(width=width, height=height, obstacles=obstacles, exits=exits)


# -----------------------------
# Draw helpers
# -----------------------------
def draw_text(surf, text, x, y, font, color=(20, 20, 20)):
    surf.blit(font.render(text, True, color), (x, y))

def draw_grid(surface: pygame.Surface, tf: Transform, canvas: pygame.Rect, step_map: int):
    if step_map <= 0:
        return
    grid_color = (230, 230, 230)
    x = 0
    while x <= tf.map_w:
        px, _ = tf.map_to_px((x, 0))
        pygame.draw.line(surface, grid_color, (px, canvas.y), (px, canvas.bottom), 1)
        x += step_map
    y = 0
    while y <= tf.map_h:
        _, py = tf.map_to_px((0, y))
        pygame.draw.line(surface, grid_color, (canvas.x, py), (canvas.right, py), 1)
        y += step_map

def draw_polygon_layer(surface: pygame.Surface, tf: Transform, pts: PolygonPts,
                       fill_rgb, outline_rgb, fill_alpha: int):
    pts_px = [tf.map_to_px(p) for p in pts]
    fill = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(fill, (*fill_rgb, fill_alpha), pts_px)
    surface.blit(fill, (0, 0))
    pygame.draw.polygon(surface, outline_rgb, pts_px, width=2)

def draw_points_and_edges(surface: pygame.Surface, tf: Transform, pts: PolygonPts, color):
    if not pts:
        return
    pts_px = [tf.map_to_px(p) for p in pts]
    if len(pts_px) >= 2:
        pygame.draw.lines(surface, color, False, pts_px, width=2)
    for (x, y) in pts_px:
        pygame.draw.circle(surface, color, (x, y), POINT_RADIUS)


# -----------------------------
# Simple Panel UI Widgets
# -----------------------------
class Slider:
    """
    value in [vmin, vmax]
    """
    def __init__(self, rect: pygame.Rect, vmin: float, vmax: float, value: float, label: str):
        self.rect = rect
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.value = float(value)
        self.label = label
        self.dragging = False

    def _clamp(self, x: float) -> float:
        return max(self.vmin, min(self.vmax, x))

    def value_from_mouse(self, mx: int) -> float:
        t = (mx - self.rect.x) / max(1, self.rect.w)
        v = self.vmin + t * (self.vmax - self.vmin)
        return self._clamp(v)

    def handle_event(self, ev: pygame.event.Event) -> bool:
        changed = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.dragging = True
                self.value = self.value_from_mouse(ev.pos[0])
                changed = True
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self.value = self.value_from_mouse(ev.pos[0])
            changed = True
        return changed

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        # label
        draw_text(surf, f"{self.label}: {self.value:.3f}", self.rect.x, self.rect.y - 20, font, (30, 30, 30))
        # track
        pygame.draw.rect(surf, (235, 235, 235), self.rect, border_radius=6)
        pygame.draw.rect(surf, (190, 190, 190), self.rect, 1, border_radius=6)
        # thumb
        t = (self.value - self.vmin) / max(1e-9, (self.vmax - self.vmin))
        tx = int(self.rect.x + t * self.rect.w)
        thumb = pygame.Rect(tx - 6, self.rect.y - 2, 12, self.rect.h + 4)
        pygame.draw.rect(surf, (120, 120, 120), thumb, border_radius=6)

class TextBox:
    """
    Numeric input box. Stores text and parses to int/float on commit.
    """
    def __init__(self, rect: pygame.Rect, text: str, label: str, is_float: bool = False):
        self.rect = rect
        self.text = text
        self.label = label
        self.is_float = is_float
        self.active = False
        self._last_good = text

    def set_text(self, s: str):
        self.text = s
        self._last_good = s

    def handle_event(self, ev: pygame.event.Event) -> bool:
        changed = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.active = self.rect.collidepoint(ev.pos)
            changed = True
        elif ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_RETURN:
                # commit attempt
                if self._parse_ok(self.text):
                    self._last_good = self.text
                else:
                    self.text = self._last_good
                self.active = False
                changed = True
            elif ev.key == pygame.K_ESCAPE:
                self.text = self._last_good
                self.active = False
                changed = True
            elif ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                changed = True
            else:
                ch = ev.unicode
                allowed = "0123456789.-" if self.is_float else "0123456789-"
                if ch and (ch in allowed):
                    self.text += ch
                    changed = True
        return changed

    def _parse_ok(self, s: str) -> bool:
        try:
            if self.is_float:
                float(s)
            else:
                int(s)
            return True
        except Exception:
            return False

    def value(self, default):
        try:
            return float(self.text) if self.is_float else int(self.text)
        except Exception:
            return default

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        draw_text(surf, self.label, self.rect.x, self.rect.y - 20, font, (30, 30, 30))
        bg = (255, 245, 235) if self.active else (245, 245, 245)
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, (180, 180, 180), self.rect, 1, border_radius=6)
        draw_text(surf, self.text, self.rect.x + 10, self.rect.y + 7, font, (20, 20, 20))

class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label

    def clicked(self, ev: pygame.event.Event) -> bool:
        return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos))

    def draw(self, surf: pygame.Surface, font: pygame.font.Font, fg=(20, 80, 20), bg=(235, 245, 235), border=(120, 160, 120)):
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, border, self.rect, width=2, border_radius=8)
        draw_text(surf, self.label, self.rect.x + 12, self.rect.y + 8, font, fg)


# -----------------------------
# Random Map Generator (Shapely)
# -----------------------------
def _clip_to_bounds(poly: "Polygon", W: int, H: int) -> Optional["Polygon"]:
    world = box(0, 0, W, H)
    p = poly.intersection(world)
    if p.is_empty:
        return None
    if p.geom_type == "Polygon":
        return p
    # MultiPolygon이면 가장 큰 것만
    if p.geom_type == "MultiPolygon":
        largest = max(list(p.geoms), key=lambda g: g.area)
        return largest if largest.geom_type == "Polygon" else None
    return None

def _random_main_block(W: int, H: int) -> "Polygon":
    """
    큰 '메인 폴리곤'을 중앙 근처에 생성 (건물/홀 같은 느낌)
    - 너무 길쭉하지 않게
    """
    # 중앙 근처로 샘플링
    cx = random.randint(int(0.30 * W), int(0.70 * W))
    cy = random.randint(int(0.30 * H), int(0.70 * H))

    bw = random.randint(int(0.18 * W), int(0.35 * W))
    bh = random.randint(int(0.18 * H), int(0.35 * H))

    x0 = cx - bw // 2
    y0 = cy - bh // 2
    base = box(x0, y0, x0 + bw, y0 + bh)

    # 살짝 '깨진' 느낌 주기: 모서리 쪽에 작은 컷아웃(빼기) or 부착(더하기)
    # 50% 확률로 컷아웃
    if random.random() < 0.5:
        cut_w = random.randint(int(0.20 * bw), int(0.45 * bw))
        cut_h = random.randint(int(0.20 * bh), int(0.45 * bh))
        side = random.choice(["L", "R", "B", "T"])
        if side == "L":
            cut = box(x0, y0 + random.randint(0, max(0, bh - cut_h)), x0 + cut_w, y0 + cut_h)
        elif side == "R":
            cut = box(x0 + bw - cut_w, y0 + random.randint(0, max(0, bh - cut_h)), x0 + bw, y0 + cut_h)
        elif side == "B":
            cut = box(x0 + random.randint(0, max(0, bw - cut_w)), y0, x0 + cut_w, y0 + cut_h)
        else:
            cut = box(x0 + random.randint(0, max(0, bw - cut_w)), y0 + bh - cut_h, x0 + cut_w, y0 + bh)
        poly = base.difference(cut)
    else:
        add_w = random.randint(int(0.18 * bw), int(0.40 * bw))
        add_h = random.randint(int(0.18 * bh), int(0.40 * bh))
        side = random.choice(["L", "R", "B", "T"])
        if side == "L":
            add = box(x0 - add_w, y0 + random.randint(0, max(0, bh - add_h)), x0, y0 + add_h)
        elif side == "R":
            add = box(x0 + bw, y0 + random.randint(0, max(0, bh - add_h)), x0 + bw + add_w, y0 + add_h)
        elif side == "B":
            add = box(x0 + random.randint(0, max(0, bw - add_w)), y0 - add_h, x0 + add_w, y0)
        else:
            add = box(x0 + random.randint(0, max(0, bw - add_w)), y0 + bh, x0 + add_w, y0 + bh + add_h)
        poly = unary_union([base, add])

    p = _clip_to_bounds(poly, W, H)
    return p if p is not None else base

def _random_L_shape(W: int, H: int) -> Optional["Polygon"]:
    """
    ㄴ/ㄱ 형태: 두 직사각형이 직각으로 붙는 형태
    """
    # 기준점(중앙 근처)
    cx = random.randint(int(0.20 * W), int(0.80 * W))
    cy = random.randint(int(0.20 * H), int(0.80 * H))

    arm_th = random.randint(4, 10)
    arm_len1 = random.randint(int(0.15 * min(W, H)), int(0.40 * min(W, H)))
    arm_len2 = random.randint(int(0.15 * min(W, H)), int(0.40 * min(W, H)))

    # 방향 선택
    orient = random.choice(["UR", "UL", "DR", "DL"])  # arm이 뻗는 방향
    if orient == "UR":
        a = box(cx, cy, cx + arm_len1, cy + arm_th)     # right
        b = box(cx, cy, cx + arm_th, cy + arm_len2)     # up
    elif orient == "UL":
        a = box(cx - arm_len1, cy, cx, cy + arm_th)     # left
        b = box(cx - arm_th, cy, cx, cy + arm_len2)     # up
    elif orient == "DR":
        a = box(cx, cy - arm_th, cx + arm_len1, cy)     # right
        b = box(cx, cy - arm_len2, cx + arm_th, cy)     # down
    else:  # DL
        a = box(cx - arm_len1, cy - arm_th, cx, cy)     # left
        b = box(cx - arm_th, cy - arm_len2, cx, cy)     # down

    u = unary_union([a, b])
    p = _clip_to_bounds(u, W, H)
    if p is None or p.area < 20:
        return None
    return p

def _random_U_shape(W: int, H: int) -> Optional["Polygon"]:
    """
    U/C 형태: 세 개의 직사각형으로 둘러싸는 구조 (방/코너 느낌)
    """
    cx = random.randint(int(0.20 * W), int(0.80 * W))
    cy = random.randint(int(0.20 * H), int(0.80 * H))

    th = random.randint(4, 10)
    w = random.randint(int(0.18 * W), int(0.35 * W))
    h = random.randint(int(0.18 * H), int(0.35 * H))

    # U의 열린 방향
    open_dir = random.choice(["UP", "DOWN", "LEFT", "RIGHT"])

    x0 = cx - w // 2
    y0 = cy - h // 2
    x1 = x0 + w
    y1 = y0 + h

    # 세 변 만들기
    if open_dir == "UP":
        left  = box(x0, y0, x0 + th, y1)
        right = box(x1 - th, y0, x1, y1)
        bottom= box(x0, y0, x1, y0 + th)
        parts = [left, right, bottom]
    elif open_dir == "DOWN":
        left  = box(x0, y0, x0 + th, y1)
        right = box(x1 - th, y0, x1, y1)
        top   = box(x0, y1 - th, x1, y1)
        parts = [left, right, top]
    elif open_dir == "LEFT":
        top   = box(x0, y1 - th, x1, y1)
        bot   = box(x0, y0, x1, y0 + th)
        right = box(x1 - th, y0, x1, y1)
        parts = [top, bot, right]
    else:  # RIGHT
        top  = box(x0, y1 - th, x1, y1)
        bot  = box(x0, y0, x1, y0 + th)
        left = box(x0, y0, x0 + th, y1)
        parts = [top, bot, left]

    u = unary_union(parts)
    p = _clip_to_bounds(u, W, H)
    if p is None or p.area < 20:
        return None
    return p

def _require_shapely():
    if not SHAPELY_OK:
        raise RuntimeError("Shapely not installed. Run: pip install shapely")

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
        if new_poly.distance(p) < min_gap:
            return True
    return False

def _corners(W: int, H: int):
    return [(0,0), (W,0), (0,H), (W,H)]

def _nearest_corner_id(poly: "Polygon", W: int, H: int):
    c = _corners(W, H)
    dists = [poly.distance(ShPoint(x, y)) for x, y in c]
    i = min(range(4), key=lambda k: dists[k])
    return i, dists[i]

def _random_exit_on_side(W: int, H: int, side: str, size_min: int, size_max: int) -> "Polygon":
    ew = random.randint(size_min, size_max)
    eh = random.randint(size_min, size_max)

    if side == "left":
        x0, x1 = 0, ew
        y0 = random.randint(0, H - eh)
        y1 = y0 + eh
    elif side == "right":
        x1, x0 = W, W - ew
        y0 = random.randint(0, H - eh)
        y1 = y0 + eh
    elif side == "bottom":
        y0, y1 = 0, eh
        x0 = random.randint(0, W - ew)
        x1 = x0 + ew
    elif side == "top":
        y1, y0 = H, H - eh
        x0 = random.randint(0, W - ew)
        x1 = x0 + ew
    else:
        raise ValueError("side must be one of left/right/top/bottom")

    return box(x0, y0, x1, y1)

def generate_two_exits(
    W: int, H: int,
    size_min: int = 5,
    size_max: int = 10,
    min_exit_distance: float = 15.0,
    corner_avoid_dist: float = 12.0,
    disallow_same_side: bool = True,
    max_tries: int = 2000,
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

def _random_corridor_strip(W: int, H: int, margin: int = 1) -> "Polygon":
    thickness = random.randint(4, 10)
    length = random.randint(int(0.35 * min(W, H)), int(0.9 * min(W, H)))
    horizontal = (random.random() < 0.5)

    if horizontal:
        w, h = length, thickness
    else:
        w, h = thickness, length

    x0 = random.randint(margin, max(margin, W - margin - w))
    y0 = random.randint(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)

def _random_rect_obstacle(W: int, H: int, margin: int = 1) -> "Polygon":
    w = random.randint(4, 30)
    h = random.randint(4, 30)
    x0 = random.randint(margin, max(margin, W - margin - w))
    y0 = random.randint(margin, max(margin, H - margin - h))
    return box(x0, y0, x0 + w, y0 + h)

def _random_convex_obstacle(W: int, H: int, margin: int = 1) -> "Polygon":
    n = random.randint(4, 10)
    pts = []
    for _ in range(n):
        pts.append((random.randint(margin, W - margin), random.randint(margin, H - margin)))
    return Polygon(pts).convex_hull

def _random_deadend_cap_near(strip: "Polygon", W: int, H: int, margin: int = 1) -> "Polygon":
    x0, y0, x1, y1 = strip.bounds
    if (x1 - x0) >= (y1 - y0):  # horizontal-ish
        cap_w = random.randint(6, 12)
        cap_h = random.randint(6, 14)
        side = random.choice(["L", "R"])
        if side == "L":
            cx1 = int(x0)
            cx0 = max(margin, cx1 - cap_w)
        else:
            cx0 = int(x1)
            cx1 = min(W - margin, cx0 + cap_w)
        cy0 = int(random.randint(max(margin, int(y0 - cap_h)), min(int(y1), H - margin - cap_h)))
        cy1 = cy0 + cap_h
        return box(cx0, cy0, cx1, cy1)
    else:  # vertical-ish
        cap_w = random.randint(6, 14)
        cap_h = random.randint(6, 12)
        side = random.choice(["B", "T"])
        if side == "B":
            cy1 = int(y0)
            cy0 = max(margin, cy1 - cap_h)
        else:
            cy0 = int(y1)
            cy1 = min(H - margin, cy0 + cap_h)
        cx0 = int(random.randint(max(margin, int(x0 - cap_w)), min(int(x1), W - margin - cap_w)))
        cx1 = cx0 + cap_w
        return box(cx0, cy0, cx1, cy1)
def _pick_obstacle_candidate(W: int, H: int, params: Dict[str, float]):
    """
    현실형 맵 bias:
      - main block (central)
      - L-shape / U-shape
      - corridors (shorter / limited aspect)
      - generic rect/convex
    returns: (main_poly, extra_polys)
    """
    extra = []

    main_block_bias = float(params.get("main_block_bias", 0.35))
    L_shape_bias    = float(params.get("L_shape_bias", 0.35))
    U_shape_bias    = float(params.get("U_shape_bias", 0.20))
    corridor_bias   = float(params.get("corridor_bias", 0.25))
    deadend_bias    = float(params.get("deadend_bias", 0.25))
    max_aspect      = float(params.get("max_corridor_aspect", 6.0))

    # 정규화 (합이 1이 아니어도 됨)
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
        return _random_main_block(W, H), extra

    if choice == "L":
        p = _random_L_shape(W, H)
        if p is not None:
            return p, extra
        # fallback
        return _random_main_block(W, H), extra

    if choice == "U":
        p = _random_U_shape(W, H)
        if p is not None:
            return p, extra
        return _random_main_block(W, H), extra

    if choice == "corr":
        base = _random_corridor_strip(W, H, margin=1)

        # ✅ 너무 긴 복도 억제 (aspect 제한)
        x0, y0, x1, y1 = base.bounds
        w = max(1e-6, (x1 - x0))
        h = max(1e-6, (y1 - y0))
        aspect = max(w / h, h / w)
        if aspect > max_aspect:
            # 짧은 rect로 대체
            base = _random_rect_obstacle(W, H, margin=1)

        # dead-end cap
        if random.random() < deadend_bias:
            extra.append(_random_deadend_cap_near(base, W, H, margin=1))
        return base, extra

    if choice == "conv":
        return _random_convex_obstacle(W, H, margin=1), extra

    # rect default
    return _random_rect_obstacle(W, H, margin=1), extra


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
    max_tries: int = 20000,
) -> List["Polygon"]:


    map_area = float(W * H)
    target_density = random.uniform(density_min, density_max)
    target_area = target_density * map_area
    target_count = random.randint(min_obstacles, max_obstacles)

    obstacles: List["Polygon"] = []
    total_area = 0.0


    for _ in range(max_tries):
        if (len(obstacles) >= target_count) and (total_area >= target_area):
            break

        main, extras = _pick_obstacle_candidate(W, H, params)
        merged = unary_union([main] + extras)

        if merged.geom_type != "Polygon":
            continue
        cand = merged

        if not _valid_polygon(cand, min_area=20.0):
            continue

        # -------------------------
        # Wall clearance rule:
        #   - obstacle touches boundary  OR
        #   - obstacle is at least wall_clearance away from boundary
        #   - (0 < dist < wall_clearance) is NOT allowed
        # -------------------------
        wall_clearance = float(params.get("wall_clearance", 7.0))
        boundary = box(0, 0, W, H).boundary  # map outer boundary

        d_wall = cand.distance(boundary)  # 0 if touching
        if d_wall < wall_clearance and (not cand.touches(boundary)):
            continue

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

        if _too_close_or_intersect(cand, obstacles, min_gap=min_gap):
            continue

        # density upper guard (slight slack)
        if (total_area + cand.area) > (density_max * map_area * 1.05):
            continue

        obstacles.append(cand)
        total_area += cand.area

    final_density = total_area / map_area if map_area > 0 else 0.0
    if final_density < density_min:
        raise RuntimeError(f"Density too low: {final_density:.3f} < {density_min}")

    if len(obstacles) < min_obstacles:
        raise RuntimeError(f"Obstacle count too low: {len(obstacles)} < {min_obstacles}")

    return obstacles

def generate_random_map_obj(
    width: int,
    height: int,
    params: Dict[str, float],
    seed: Optional[int] = None,
    map_level_max_tries: int = 200,
) -> Dict[str, Any]:
    _require_shapely()
    if seed is not None:
        random.seed(seed)

    # sanitize
    density_min = float(params["density_min"])
    density_max = float(params["density_max"])
    if density_min > density_max:
        density_min, density_max = density_max, density_min

    min_obstacles = int(params["min_obstacles"])
    max_obstacles = int(params["max_obstacles"])
    if min_obstacles > max_obstacles:
        min_obstacles, max_obstacles = max_obstacles, min_obstacles

    corridor_bias = float(params["corridor_bias"])
    deadend_bias = float(params["deadend_bias"])
    corridor_bias = max(0.0, min(1.0, corridor_bias))
    deadend_bias = max(0.0, min(1.0, deadend_bias))

    for _ in range(map_level_max_tries):
        exits = generate_two_exits(
            width, height,
            size_min=int(params["exit_size_min"]),
            size_max=int(params["exit_size_max"]),
            min_exit_distance=float(params["min_exit_distance"]),
            corner_avoid_dist=float(params["corner_avoid_dist"]),
            disallow_same_side=True,
        )
        try:
            obstacles = generate_obstacles_with_density(
                width, height,
                exits=exits,
                min_obstacles=min_obstacles,
                max_obstacles=max_obstacles,
                density_min=density_min,
                density_max=density_max,
                min_gap=float(params["min_obstacle_gap"]),
                keep_gap_from_exits=float(params["keep_gap_from_exits"]),
                params=params,
            )
        except RuntimeError:
            continue

        return {
            "width": width,
            "height": height,
            "obstacles": [_poly_to_points_int(o) for o in obstacles],
            "exits": [_poly_to_points_int(e) for e in exits],
        }

    raise RuntimeError("Failed to generate a valid random map after many attempts.")


# -----------------------------
# CLI prompt
# -----------------------------
def prompt_int(msg: str, default: int) -> int:
    try:
        s = input(f"{msg} (default {default}) : ").strip()
        return default if not s else int(s)
    except Exception:
        return default


# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(MAP_INFO_DIR)

    print("=== ADDS Map Editor ===")
    map_num = 1
    map_w = prompt_int("Map width", 100)
    map_h = prompt_int("Map height", 100)

    data = MapData(width=map_w, height=map_h, obstacles=[], exits=[])

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("ADDS Map Editor (Polygon/Rect/Delete) + Random Generator")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 18)
    font_big = pygame.font.SysFont("consolas", 22, bold=True)

    panel = pygame.Rect(WINDOW_W - PANEL_W, 0, PANEL_W, WINDOW_H)
    canvas = pygame.Rect(0, 0, WINDOW_W - PANEL_W, WINDOW_H)
    tf = Transform(map_w, map_h, canvas, margin=30)

    tool = "poly_ob"
    editing_mapnum = False
    mapnum_text = str(map_num)

    current_pts: PolygonPts = []
    dragging_rect = False
    drag_start: Optional[Point] = None
    drag_now: Optional[Point] = None

    history: List[Tuple[str, Any]] = []

    status = "Ready"
    status_t = time.time()

    def set_status(msg: str):
        nonlocal status, status_t
        status = msg
        status_t = time.time()

    def map_click_to_point(mx: int, my: int) -> Point:
        p = tf.px_to_map(mx, my)
        p = clamp_point(p, data.width, data.height)
        if GRID_ON:
            p = snap_point(p, GRID_STEP)
            p = clamp_point(p, data.width, data.height)
        return p

    def commit_obstacle_poly(pts: PolygonPts):
        poly = [[int(x), int(y)] for (x, y) in pts]
        data.obstacles.append(poly)
        history.append(("add_ob", poly))
        set_status(f"Added obstacle #{len(data.obstacles)}")

    def commit_exit_poly(pts: PolygonPts):
        poly = [(int(x), int(y)) for (x, y) in pts]
        data.exits.append(poly)
        history.append(("add_ex", poly))
        set_status(f"Added exit #{len(data.exits)}")

    def try_delete_at(pt: Point):
        for i in range(len(data.exits) - 1, -1, -1):
            poly = data.exits[i]
            pts = [(p[0], p[1]) for p in poly]
            if point_in_poly(pt, pts):
                removed = data.exits.pop(i)
                history.append(("del_ex", removed, i))
                set_status(f"Deleted exit at index {i}")
                return
        for i in range(len(data.obstacles) - 1, -1, -1):
            poly = data.obstacles[i]
            pts = [(p[0], p[1]) for p in poly]
            if point_in_poly(pt, pts):
                removed = data.obstacles.pop(i)
                history.append(("del_ob", removed, i))
                set_status(f"Deleted obstacle at index {i}")
                return
        set_status("Delete: nothing hit")

    def undo():
        if not history:
            set_status("Undo: nothing")
            return
        act = history.pop()
        kind = act[0]
        if kind == "add_ob":
            poly = act[1]
            for i in range(len(data.obstacles) - 1, -1, -1):
                if data.obstacles[i] == poly:
                    data.obstacles.pop(i)
                    set_status("Undo: removed last added obstacle")
                    return
            set_status("Undo: obstacle not found")
        elif kind == "add_ex":
            poly = act[1]
            for i in range(len(data.exits) - 1, -1, -1):
                if data.exits[i] == poly:
                    data.exits.pop(i)
                    set_status("Undo: removed last added exit")
                    return
            set_status("Undo: exit not found")
        elif kind == "del_ob":
            poly, idx = act[1], act[2]
            idx = min(max(0, idx), len(data.obstacles))
            data.obstacles.insert(idx, poly)
            set_status("Undo: restored deleted obstacle")
        elif kind == "del_ex":
            poly, idx = act[1], act[2]
            idx = min(max(0, idx), len(data.exits))
            data.exits.insert(idx, poly)
            set_status("Undo: restored deleted exit")

    # ---- Random generator params (panel-controlled) ----
    rand_params = {
        "density_min": 0.10,
        "density_max": 0.25,
        "min_obstacles": 5,
        "max_obstacles": 12,
        "corridor_bias": 0.65,
        "deadend_bias": 0.45,
        "min_obstacle_gap": 5.0,
        "keep_gap_from_exits": 0.0,
        "exit_size_min": 5,
        "exit_size_max": 10,
        "min_exit_distance": 18.0,
        "corner_avoid_dist": 14.0,
        "wall_clearance" : 7.0
    }

    def apply_random_map():
        #print("[Random] generated OK")
        nonlocal data, tf, current_pts, dragging_rect, drag_start, drag_now, history, rand_params
        try:
            if not SHAPELY_OK:
                set_status("Random: Shapely missing. pip install shapely")
                return

            obj = generate_random_map_obj(
                width=data.width,
                height=data.height,
                params=rand_params,
                seed=None
            )

            data.obstacles = obj["obstacles"]
            data.exits = [[(int(x), int(y)) for x, y in poly] for poly in obj["exits"]]

            current_pts = []
            dragging_rect = False
            drag_start = None
            drag_now = None
            history.clear()

            tf = Transform(data.width, data.height, canvas, margin=30)
            set_status("Random map generated. Press S to save.")

        except Exception as e:
            set_status(f"Random failed: {e}")

    running = True
    last_path: Optional[str] = None

    # Panel clickables (will be created each frame)
    mapnum_rect: Optional[pygame.Rect] = None
    gen_btn: Optional[Button] = None

    # UI widgets (initialized once, rects updated each frame)
    # Sliders
    density_min_slider = Slider(pygame.Rect(0,0,0,0), 0.00, 0.40, rand_params["density_min"], "Density min")
    density_max_slider = Slider(pygame.Rect(0,0,0,0), 0.00, 0.40, rand_params["density_max"], "Density max")
    corridor_slider    = Slider(pygame.Rect(0,0,0,0), 0.00, 1.00, rand_params["corridor_bias"], "Corridor bias")
    deadend_slider     = Slider(pygame.Rect(0,0,0,0), 0.00, 1.00, rand_params["deadend_bias"], "Dead-end bias")

    # Text boxes
    minobs_box   = TextBox(pygame.Rect(0,0,0,0), str(rand_params["min_obstacles"]), "Min obstacles", is_float=False)
    maxobs_box   = TextBox(pygame.Rect(0,0,0,0), str(rand_params["max_obstacles"]), "Max obstacles", is_float=False)
    mingap_box   = TextBox(pygame.Rect(0,0,0,0), str(rand_params["min_obstacle_gap"]), "Min obstacle gap", is_float=True)
    exitdist_box = TextBox(pygame.Rect(0,0,0,0), str(rand_params["min_exit_distance"]), "Exit min distance", is_float=True)
    corner_box   = TextBox(pygame.Rect(0,0,0,0), str(rand_params["corner_avoid_dist"]), "Corner avoid dist", is_float=True)

    # Helper: if any textbox active, don't use hotkeys that would type in tools
    def any_textbox_active() -> bool:
        return any([
            minobs_box.active, maxobs_box.active, mingap_box.active,
            exitdist_box.active, corner_box.active
        ])

    while running:
        clock.tick(60)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # Panel widget events first (so typing goes into textboxes)
            # Sliders + Textboxes should still respond even if mouse is on panel
            changed = False
            changed |= density_min_slider.handle_event(ev)
            changed |= density_max_slider.handle_event(ev)
            changed |= corridor_slider.handle_event(ev)
            changed |= deadend_slider.handle_event(ev)

            changed |= minobs_box.handle_event(ev)
            changed |= maxobs_box.handle_event(ev)
            changed |= mingap_box.handle_event(ev)
            changed |= exitdist_box.handle_event(ev)
            changed |= corner_box.handle_event(ev)

            if changed:
                # sync widget values -> rand_params
                rand_params["density_min"] = float(density_min_slider.value)
                rand_params["density_max"] = float(density_max_slider.value)
                # keep min<=max visually by gently clamping
                if rand_params["density_min"] > rand_params["density_max"]:
                    rand_params["density_max"] = rand_params["density_min"]
                    density_max_slider.value = rand_params["density_max"]

                rand_params["corridor_bias"] = float(corridor_slider.value)
                rand_params["deadend_bias"]  = float(deadend_slider.value)

                rand_params["min_obstacles"] = int(minobs_box.value(rand_params["min_obstacles"]))
                rand_params["max_obstacles"] = int(maxobs_box.value(rand_params["max_obstacles"]))
                if rand_params["min_obstacles"] > rand_params["max_obstacles"]:
                    rand_params["max_obstacles"] = rand_params["min_obstacles"]
                    maxobs_box.set_text(str(rand_params["max_obstacles"]))

                rand_params["min_obstacle_gap"] = float(mingap_box.value(rand_params["min_obstacle_gap"]))
                rand_params["min_exit_distance"] = float(exitdist_box.value(rand_params["min_exit_distance"]))
                rand_params["corner_avoid_dist"] = float(corner_box.value(rand_params["corner_avoid_dist"]))

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Button click (if created)
                if gen_btn is not None and gen_btn.clicked(ev):
                    apply_random_map()
                    continue

            if ev.type == pygame.KEYDOWN:
                # If a textbox is active, we only allow general escape/quit
                if any_textbox_active():
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    continue

                # Map num editing mode
                if editing_mapnum:
                    if ev.key == pygame.K_RETURN:
                        try:
                            new_num = int(mapnum_text) if mapnum_text.strip() else map_num
                            map_num = max(0, new_num)
                            mapnum_text = str(map_num)
                            set_status(f"Map num set to {map_num}")
                        except Exception:
                            set_status("Invalid map number")
                            mapnum_text = str(map_num)
                        editing_mapnum = False
                        continue

                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        editing_mapnum = False
                        mapnum_text = str(map_num)
                        set_status("Map num edit canceled")
                        continue

                    if ev.key == pygame.K_BACKSPACE:
                        mapnum_text = mapnum_text[:-1]
                        continue

                    if ev.unicode.isdigit():
                        mapnum_text += ev.unicode
                        continue

                    continue

                # Normal hotkeys
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                elif ev.key == pygame.K_1:
                    tool = "poly_ob"
                    current_pts = []
                    dragging_rect = False
                    set_status("Tool: Polygon OBSTACLE")

                elif ev.key == pygame.K_2:
                    tool = "poly_ex"
                    current_pts = []
                    dragging_rect = False
                    set_status("Tool: Polygon EXIT")

                elif ev.key == pygame.K_3:
                    tool = "rect_ob"
                    current_pts = []
                    dragging_rect = False
                    set_status("Tool: Rect OBSTACLE (drag)")

                elif ev.key == pygame.K_4:
                    tool = "rect_ex"
                    current_pts = []
                    dragging_rect = False
                    set_status("Tool: Rect EXIT (drag)")

                elif ev.key == pygame.K_d:
                    tool = "delete"
                    current_pts = []
                    dragging_rect = False
                    set_status("Tool: DELETE (click inside polygon)")

                elif ev.key == pygame.K_BACKSPACE:
                    if current_pts:
                        current_pts.pop()
                        set_status("Removed last point")

                elif ev.key == pygame.K_n:
                    current_pts = []
                    dragging_rect = False
                    drag_start = None
                    drag_now = None
                    set_status("Canceled current draw")

                elif ev.key == pygame.K_u:
                    undo()

                elif ev.key == pygame.K_c:
                    data.obstacles.clear()
                    data.exits.clear()
                    history.clear()
                    current_pts = []
                    dragging_rect = False
                    set_status("Cleared all")

                elif ev.key == pygame.K_RETURN:
                    if tool in ("poly_ob", "poly_ex"):
                        if len(current_pts) < 3:
                            set_status("Need >= 3 points")
                        else:
                            if tool == "poly_ob":
                                commit_obstacle_poly(current_pts)
                            else:
                                commit_exit_poly(current_pts)
                            current_pts = []

                elif ev.key == pygame.K_s:
                    try:
                        path = path_for_mapnum(map_num)
                        save_map_snippet(path, data)
                        last_path = path
                        set_status(f"Saved: {path}")
                    except Exception as e:
                        set_status(f"Save failed: {e}")

                elif ev.key == pygame.K_l:
                    try:
                        path = path_for_mapnum(map_num)
                        loaded = load_map_snippet(path)
                        data = loaded
                        tf = Transform(data.width, data.height, canvas, margin=30)
                        current_pts = []
                        dragging_rect = False
                        drag_start = None
                        drag_now = None
                        history.clear()
                        last_path = path
                        set_status(f"Loaded: {path}")
                    except Exception as e:
                        set_status(f"Load failed: {e}")

                elif ev.key == pygame.K_m:
                    editing_mapnum = True
                    mapnum_text = ""
                    set_status("Editing map num... type digits, ENTER apply")

                elif ev.key == pygame.K_r:
                    apply_random_map()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos

                # Panel click: mapnum rect toggles edit
                if panel.collidepoint(mx, my):
                    if mapnum_rect is not None and mapnum_rect.collidepoint(mx, my):
                        editing_mapnum = True
                        mapnum_text = ""
                        set_status("Editing map num... type digits, ENTER apply")
                    continue

                if not canvas.collidepoint(mx, my):
                    continue

                p = map_click_to_point(mx, my)

                if tool in ("poly_ob", "poly_ex"):
                    current_pts.append(p)
                    set_status(f"Point: {p}")

                elif tool in ("rect_ob", "rect_ex"):
                    dragging_rect = True
                    drag_start = p
                    drag_now = p
                    set_status(f"Rect start: {p}")

                elif tool == "delete":
                    try_delete_at(p)

            elif ev.type == pygame.MOUSEMOTION:
                if dragging_rect and drag_start is not None:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my):
                        drag_now = map_click_to_point(mx, my)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if dragging_rect and drag_start is not None and drag_now is not None:
                    pts = rect_from_drag(drag_start, drag_now)
                    if pts[0] == pts[1] or pts[1] == pts[2]:
                        set_status("Rect too small, ignored")
                    else:
                        if tool == "rect_ob":
                            commit_obstacle_poly(pts)
                        elif tool == "rect_ex":
                            commit_exit_poly(pts)
                    dragging_rect = False
                    drag_start = None
                    drag_now = None

        # ---------------- Draw ----------------
        screen.fill((255, 255, 255))
        pygame.draw.rect(screen, CANVAS_BG, canvas)

        if GRID_ON:
            draw_grid(screen, tf, canvas, step_map=GRID_DRAW_STEP)

        # obstacles
        for poly in data.obstacles:
            pts = [(p[0], p[1]) for p in poly]
            draw_polygon_layer(screen, tf, pts,
                               fill_rgb=(70, 70, 70),
                               outline_rgb=(30, 30, 30),
                               fill_alpha=OBSTACLE_FILL_ALPHA)

        # exits
        for poly in data.exits:
            pts = [(p[0], p[1]) for p in poly]
            draw_polygon_layer(screen, tf, pts,
                               fill_rgb=(50, 120, 255),
                               outline_rgb=(20, 80, 200),
                               fill_alpha=EXIT_FILL_ALPHA)

        # current polygon preview
        if tool in ("poly_ob", "poly_ex"):
            col = (0, 0, 0) if tool == "poly_ob" else (20, 80, 200)
            draw_points_and_edges(screen, tf, current_pts, color=col)

        # current rect preview
        if dragging_rect and drag_start is not None and drag_now is not None:
            pts = rect_from_drag(drag_start, drag_now)
            col = (0, 0, 0) if tool == "rect_ob" else (20, 80, 200)
            pts_px = [tf.map_to_px(p) for p in pts]
            pygame.draw.polygon(screen, col, pts_px, width=2)

        pygame.draw.rect(screen, (210, 210, 210), canvas, width=2)

        # panel
        panel = pygame.Rect(WINDOW_W - PANEL_W, 0, PANEL_W, WINDOW_H)
        pygame.draw.rect(screen, (250, 250, 250), panel)
        pygame.draw.line(screen, (220, 220, 220), (panel.x, 0), (panel.x, WINDOW_H), 2)

        y = 20
        draw_text(screen, "ADDS Map Editor", panel.x + 20, y, font_big); y += 40

                # ---- FIXED Status Bar (always visible) ----
        status_bar = pygame.Rect(panel.x + 20, 62, PANEL_W - 40, 26)
        pygame.draw.rect(screen, (255, 245, 245), status_bar, border_radius=6)
        pygame.draw.rect(screen, (220, 180, 180), status_bar, 1, border_radius=6)
        msg = status if (time.time() - status_t < 5.0) else "(idle)"
        draw_text(screen, f"Status: {msg}", status_bar.x + 8, status_bar.y + 4, font, color=(120, 0, 0))

        mapnum_y = y
        mapnum_display = mapnum_text if editing_mapnum else str(map_num)
        draw_text(screen, f"Map num : {mapnum_display}", panel.x + 20, y, font); y += 22
        mapnum_rect = pygame.Rect(panel.x + 20, mapnum_y, PANEL_W - 40, 22)
        if editing_mapnum:
            pygame.draw.rect(screen, (255, 230, 230), mapnum_rect, border_radius=4)
        else:
            pygame.draw.rect(screen, (240, 240, 240), mapnum_rect, 1, border_radius=4)

        draw_text(screen, f"Size    : {data.width} x {data.height}", panel.x + 20, y, font); y += 28

        tool_name = {
            "poly_ob": "Polygon OBSTACLE",
            "poly_ex": "Polygon EXIT",
            "rect_ob": "Rect OBSTACLE",
            "rect_ex": "Rect EXIT",
            "delete":  "DELETE",
        }[tool]
        draw_text(screen, f"Tool: {tool_name}", panel.x + 20, y, font_big,
                  color=(120, 0, 0) if tool == "delete" else (20, 20, 20))
        y += 34

        draw_text(screen, f"Obstacles: {len(data.obstacles)}", panel.x + 20, y, font); y += 20
        draw_text(screen, f"Exits    : {len(data.exits)}", panel.x + 20, y, font); y += 20
        draw_text(screen, f"Undo stack: {len(history)}", panel.x + 20, y, font); y += 22

        # ---- Random controls ----
        draw_text(screen, "Random Map Controls", panel.x + 20, y, font_big); y += 30
        if not SHAPELY_OK:
            draw_text(screen, "Shapely missing: pip install shapely", panel.x + 20, y, font, color=(140, 0, 0))
            y += 22

        # sliders layout
        slider_w = PANEL_W - 60
        slider_h = 14

        # density min slider
        density_min_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        density_min_slider.draw(screen, font)
        y += 54

        # density max slider
        density_max_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        density_max_slider.draw(screen, font)
        y += 54

        # corridor bias slider
        corridor_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        corridor_slider.draw(screen, font)
        y += 54

        # dead-end bias slider
        deadend_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        deadend_slider.draw(screen, font)
        y += 54

        # text boxes (two columns)
        box_w = (PANEL_W - 60) // 2
        box_h = 28

        minobs_box.rect = pygame.Rect(panel.x + 30, y + 18, box_w, box_h)
        maxobs_box.rect = pygame.Rect(panel.x + 30 + box_w + 10, y + 18, box_w, box_h)
        minobs_box.draw(screen, font)
        maxobs_box.draw(screen, font)
        y += 66

        mingap_box.rect = pygame.Rect(panel.x + 30, y + 18, box_w, box_h)
        exitdist_box.rect = pygame.Rect(panel.x + 30 + box_w + 10, y + 18, box_w, box_h)
        mingap_box.draw(screen, font)
        exitdist_box.draw(screen, font)
        y += 66

        corner_box.rect = pygame.Rect(panel.x + 30, y + 18, box_w, box_h)
        corner_box.draw(screen, font)
        y += 66

        # Generate button
        gen_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 38), "R / Click: Generate Random Map")
        gen_btn.draw(screen, font)
        y += 54

        # Keys help
        draw_text(screen, "Keys", panel.x + 20, y, font_big); y += 26
        keys = [
            "1: poly obstacle   | 2: poly exit",
            "3: rect obstacle   | 4: rect exit",
            "D: delete tool (click inside)",
            "ENTER: finalize poly (poly tools)",
            "BACKSP: remove last vertex",
            "N: cancel current draw",
            "U: undo (add/delete)",
            "C: clear all",
            "S: save map_<num>.json",
            "L: load map_<num>.json",
            "M: edit map num",
            "R: random generate",
            "Q/ESC: quit",
        ]
        for k in keys:
            draw_text(screen, k, panel.x + 20, y, font, color=(40, 40, 40))
            y += 20

        y += 10
        if time.time() - status_t < 3.0:
            draw_text(screen, f"Status: {status}", panel.x + 20, y, font, color=(120, 0, 0))
        else:
            draw_text(screen, "Status: (idle)", panel.x + 20, y, font, color=(120, 120, 120))
        y += 24

        if last_path:
            p = last_path
            if len(p) > 50:
                p = "..." + p[-47:]
            draw_text(screen, f"Last: {p}", panel.x + 20, y, font, color=(80, 80, 80))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
