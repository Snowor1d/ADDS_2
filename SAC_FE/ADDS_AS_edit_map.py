#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDS Map Editor (Pygame)
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
- Panel controls for random generation (delegates to random_map.py):
  * obstacle density min/max (sliders)
  * obstacle count min/max (inputs)
  * corridor bias (slider)
  * dead-end bias (slider)
  * min obstacle gap (input)
  * exit min distance (input)
  * corner avoid distance (input)
- Button: "Generate Random Map" (and hotkey R)
- After generation, press S to save

REQUIREMENT:
- random_map.py must be importable (same folder or in PYTHONPATH)
  and must provide:
    - RandomMapSpec
    - generate_map(spec) -> MapData
"""

import os
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import pygame
from random import Random

# -----------------------------
# Import random_map.py generator
# -----------------------------
try:
    import random_map as rm  # your Shapely-based module
    RANDOM_MAP_OK = True
except Exception:
    rm = None
    RANDOM_MAP_OK = False

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
    """value in [vmin, vmax]"""
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
        draw_text(surf, f"{self.label}: {self.value:.3f}", self.rect.x, self.rect.y - 20, font, (30, 30, 30))
        pygame.draw.rect(surf, (235, 235, 235), self.rect, border_radius=6)
        pygame.draw.rect(surf, (190, 190, 190), self.rect, 1, border_radius=6)
        t = (self.value - self.vmin) / max(1e-9, (self.vmax - self.vmin))
        tx = int(self.rect.x + t * self.rect.w)
        thumb = pygame.Rect(tx - 6, self.rect.y - 2, 12, self.rect.h + 4)
        pygame.draw.rect(surf, (120, 120, 120), thumb, border_radius=6)


class TextBox:
    """Numeric input box. Stores text and parses to int/float on commit."""
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

    def draw(self, surf: pygame.Surface, font: pygame.font.Font,
             fg=(20, 80, 20), bg=(235, 245, 235), border=(120, 160, 120)):
        pygame.draw.rect(surf, bg, self.rect, border_radius=8)
        pygame.draw.rect(surf, border, self.rect, width=2, border_radius=8)
        draw_text(surf, self.label, self.rect.x + 12, self.rect.y + 8, font, fg)


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
    # These are "UI values". We'll convert them into rm.RandomMapSpec ranges inside apply_random_map().
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
        "wall_clearance": 7.0,
    }

    def _mk_spec_from_panel() -> "rm.RandomMapSpec":
        """Convert UI params -> RandomMapSpec with fixed ranges."""
        assert rm is not None

        # sanitize here (UI already clamps some, but keep it safe)
        dmin = float(rand_params["density_min"])
        dmax = float(rand_params["density_max"])
        if dmin > dmax:
            dmin, dmax = dmax, dmin

        omin = int(rand_params["min_obstacles"])
        omax = int(rand_params["max_obstacles"])
        if omin > omax:
            omin, omax = omax, omin

        exit_min = int(rand_params["exit_size_min"])
        exit_max = int(rand_params["exit_size_max"])
        if exit_min > exit_max:
            exit_min, exit_max = exit_max, exit_min

        # NOTE: this editor only controls corridor/deadend bias.
        # We'll map them into the spec bias ranges (fixed per-map).
        corridor_bias = float(rand_params["corridor_bias"])
        deadend_bias = float(rand_params["deadend_bias"])
        corridor_bias = max(0.0, min(1.0, corridor_bias))
        deadend_bias = max(0.0, min(1.0, deadend_bias))

        spec = rm.RandomMapSpec(
            width=int(data.width),
            height=int(data.height),
            seed=None,

            exit_size_min_range=(exit_min, exit_min),
            exit_size_max_range=(exit_max, exit_max),
            min_exit_distance_range=(float(rand_params["min_exit_distance"]), float(rand_params["min_exit_distance"])),
            corner_avoid_dist_range=(float(rand_params["corner_avoid_dist"]), float(rand_params["corner_avoid_dist"])),
            disallow_same_side=True,

            min_obstacles_range=(omin, omin),
            max_obstacles_range=(omax, omax),
            density_min_range=(dmin, dmin),
            density_max_range=(dmax, dmax),

            min_obstacle_gap_range=(float(rand_params["min_obstacle_gap"]), float(rand_params["min_obstacle_gap"])),
            keep_gap_from_exits_range=(float(rand_params["keep_gap_from_exits"]), float(rand_params["keep_gap_from_exits"])),
            wall_clearance_range=(float(rand_params["wall_clearance"]), float(rand_params["wall_clearance"])),

            # Biases: keep original "main/L/U" defaults from random_map.py by not overriding them
            corridor_bias_range=(corridor_bias, corridor_bias),
            deadend_bias_range=(deadend_bias, deadend_bias),
        )
        return spec

    def apply_random_map():
        nonlocal data, tf, current_pts, dragging_rect, drag_start, drag_now, history
        try:
            if not RANDOM_MAP_OK or rm is None:
                set_status("Random: random_map.py import failed (check file path / module name).")
                return

            spec = _mk_spec_from_panel()
            out = rm.generate_map(spec)  # rm.MapData

            # rm.MapData is json-friendly already.
            data.obstacles = out.obstacles
            # exits in rm.MapData are List[List[Tuple[int,int]]]
            data.exits = [[(int(x), int(y)) for (x, y) in poly] for poly in out.exits]

            current_pts = []
            dragging_rect = False
            drag_start = None
            drag_now = None
            history.clear()

            tf = Transform(data.width, data.height, canvas, margin=30)
            set_status(f"Random map generated (seed={out.seed_used}). Press S to save.")

        except Exception as e:
            set_status(f"Random failed: {e}")

    running = True
    last_path: Optional[str] = None

    # Panel clickables (will be created each frame)
    mapnum_rect: Optional[pygame.Rect] = None
    gen_btn: Optional[Button] = None

    # UI widgets (initialized once, rects updated each frame)
    density_min_slider = Slider(pygame.Rect(0, 0, 0, 0), 0.00, 0.40, rand_params["density_min"], "Density min")
    density_max_slider = Slider(pygame.Rect(0, 0, 0, 0), 0.00, 0.40, rand_params["density_max"], "Density max")
    corridor_slider    = Slider(pygame.Rect(0, 0, 0, 0), 0.00, 1.00, rand_params["corridor_bias"], "Corridor bias")
    deadend_slider     = Slider(pygame.Rect(0, 0, 0, 0), 0.00, 1.00, rand_params["deadend_bias"], "Dead-end bias")

    minobs_box   = TextBox(pygame.Rect(0, 0, 0, 0), str(rand_params["min_obstacles"]), "Min obstacles", is_float=False)
    maxobs_box   = TextBox(pygame.Rect(0, 0, 0, 0), str(rand_params["max_obstacles"]), "Max obstacles", is_float=False)
    mingap_box   = TextBox(pygame.Rect(0, 0, 0, 0), str(rand_params["min_obstacle_gap"]), "Min obstacle gap", is_float=True)
    exitdist_box = TextBox(pygame.Rect(0, 0, 0, 0), str(rand_params["min_exit_distance"]), "Exit min distance", is_float=True)
    corner_box   = TextBox(pygame.Rect(0, 0, 0, 0), str(rand_params["corner_avoid_dist"]), "Corner avoid dist", is_float=True)

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

            # Panel widget events
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
                rand_params["density_min"] = float(density_min_slider.value)
                rand_params["density_max"] = float(density_max_slider.value)
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
                if gen_btn is not None and gen_btn.clicked(ev):
                    apply_random_map()
                    continue

            if ev.type == pygame.KEYDOWN:
                if any_textbox_active():
                    if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    continue

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

        # status bar
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
        if not RANDOM_MAP_OK:
            draw_text(screen, "random_map.py import failed", panel.x + 20, y, font, color=(140, 0, 0))
            y += 22

        slider_w = PANEL_W - 60
        slider_h = 14

        density_min_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        density_min_slider.draw(screen, font)
        y += 54

        density_max_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        density_max_slider.draw(screen, font)
        y += 54

        corridor_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        corridor_slider.draw(screen, font)
        y += 54

        deadend_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        deadend_slider.draw(screen, font)
        y += 54

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
