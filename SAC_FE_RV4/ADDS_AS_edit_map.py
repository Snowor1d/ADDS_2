#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDS Map Editor (Pygame)

[UPDATE for new random_map.py structure]
- random_map.RandomMapSpec now takes:
    width, height, difficulty, seed(optional)
- random generation panel:
    * Difficulty slider (1..3)
    * Wall-rect bias slider (0..1)  [optional override]
- (옵션) random_map.py에 아래 함수가 있으면 UI에서 즉시 반영됨:
    - set_difficulty_overrides(difficulty:int, overrides:dict)  -> None
  없으면 그냥 difficulty만 전달하고, 나머지는 random_map.py 기본 테이블을 사용.

REQUIREMENT:
- random_map.py must be importable and provide:
    - RandomMapSpec
    - generate_map(spec) -> MapData
"""

import os
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict
import pygame

# -----------------------------
# Import random_map.py generator
# -----------------------------
try:
    import random_map as rm  # Shapely-based module
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

class TextBox:
    def __init__(self, rect: pygame.Rect, text: str = "", label: str = ""):
        self.rect = rect
        self.text = text
        self.label = label
        self.active = False

    def handle_event(self, ev: pygame.event.Event) -> bool:
        changed = False
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.active = self.rect.collidepoint(ev.pos)
            changed = True
        elif ev.type == pygame.KEYDOWN and self.active:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                changed = True
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                # Enter는 main loop에서 처리하게 두고 여기선 변경 없음
                pass
            else:
                # 숫자만 허용 (map id)
                if ev.unicode.isdigit():
                    self.text += ev.unicode
                    changed = True
        return changed

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        if self.label:
            draw_text(surf, self.label, self.rect.x, self.rect.y - 20, font, (30, 30, 30))

        bg = (255, 255, 255) if self.active else (245, 245, 245)
        border = (80, 120, 220) if self.active else (180, 180, 180)

        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, border, self.rect, 2, border_radius=6)

        show = self.text if self.text else ""
        draw_text(surf, show, self.rect.x + 10, self.rect.y + 6, font, (20, 20, 20))
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

    # -----------------------------
    # Random generator params (NEW)
    # -----------------------------
    rand_params = {
        "difficulty": 5,          # 1..3
        "wall_rect_bias": 0.20,   # optional override (only if rm supports overrides)
    }

    # optional override API (we'll detect at runtime)
    HAS_OVERRIDE_API = RANDOM_MAP_OK and (rm is not None) and hasattr(rm, "set_difficulty_overrides")

    def _apply_overrides_if_supported():
        if not HAS_OVERRIDE_API:
            return
        try:
            rm.set_difficulty_overrides(int(rand_params["difficulty"]), {
                "wall_rect_bias": float(rand_params["wall_rect_bias"]),
            })
        except Exception:
            # ignore override errors; fall back to difficulty-only
            pass

    def apply_random_map():
        nonlocal data, tf, current_pts, dragging_rect, drag_start, drag_now, history
        try:
            if not RANDOM_MAP_OK or rm is None:
                set_status("Random: random_map.py import failed (check file path / module name).")
                return

            _apply_overrides_if_supported()

            spec = rm.RandomMapSpec(
                width=int(data.width),
                height=int(data.height),
                difficulty=int(rand_params["difficulty"]),
                seed=None,
            )
            out = rm.generate_map(spec)

            data.obstacles = out.obstacles
            data.exits = [[(int(x), int(y)) for (x, y) in poly] for poly in out.exits]

            current_pts = []
            dragging_rect = False
            drag_start = None
            drag_now = None
            history.clear()

            tf = Transform(data.width, data.height, canvas, margin=30)
            set_status(f"Random map generated (difficulty={rand_params['difficulty']}, seed={out.seed_used}). Press S to save.")

        except Exception as e:
            set_status(f"Random failed: {e}")

    running = True
    last_path: Optional[str] = None

    # Panel clickables (will be created each frame)
    mapnum_rect: Optional[pygame.Rect] = None
    gen_btn: Optional[Button] = None
    open_btn: Optional[Button] = None
    open_id_box = TextBox(pygame.Rect(0, 0, 0, 0), text=str(map_num), label="Open map # (e.g., 200)")
    open_by_id_btn: Optional[Button] = None

    # Widgets
    difficulty_slider = Slider(pygame.Rect(0, 0, 0, 0), 1.0, 6.0, float(rand_params["difficulty"]), "Difficulty (1..6)")
    wallrect_slider   = Slider(pygame.Rect(0, 0, 0, 0), 0.0, 1.0, float(rand_params["wall_rect_bias"]), "Wall-rect bias")

    def any_widget_dragging() -> bool:
        return difficulty_slider.dragging or wallrect_slider.dragging

    while running:
        clock.tick(60)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # panel sliders
            changed = False
            changed |= difficulty_slider.handle_event(ev)
            if HAS_OVERRIDE_API:
                changed |= wallrect_slider.handle_event(ev)

            open_id_box.handle_event(ev)

            if changed:
                # difficulty is discrete 1..3
                rand_params["difficulty"] = int(round(difficulty_slider.value))
                difficulty_slider.value = float(rand_params["difficulty"])
                if HAS_OVERRIDE_API:
                    rand_params["wall_rect_bias"] = float(wallrect_slider.value)

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if gen_btn is not None and gen_btn.clicked(ev):
                    apply_random_map()
                    continue

            if open_btn is not None and open_btn.clicked(ev):
                # O 키와 동일 동작
                try:
                    default_path = path_for_mapnum(map_num)
                    s = input(f"Open map path (ENTER = {default_path}) : ").strip()
                    path = default_path if not s else s

                    loaded = load_map_snippet(path)
                    data = loaded
                    tf = Transform(data.width, data.height, canvas, margin=30)

                    current_pts = []
                    dragging_rect = False
                    drag_start = None
                    drag_now = None
                    history.clear()

                    last_path = path
                    set_status(f"Opened for edit: {path}")
                except Exception as e:
                    set_status(f"Open failed: {e}")
                continue


            if ev.type == pygame.KEYDOWN:
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

                elif ev.key in (pygame.K_RETURN, pygame.K_TAB):
                    if open_id_box.active:
                        try:
                            n = int(open_id_box.text) if open_id_box.text.strip() else map_num
                            path = path_for_mapnum(n)
                            loaded = load_map_snippet(path)
                            data = loaded
                            tf = Transform(data.width, data.height, canvas, margin=30)

                            current_pts = []
                            dragging_rect = False
                            drag_start = None
                            drag_now = None
                            history.clear()

                            last_path = path
                            map_num = n
                            mapnum_text = str(map_num)
                            set_status(f"Opened for edit: {path}")
                        except Exception as e:
                            set_status(f"Open failed: {e}")
                        continue

                    # 아니면 기존 polygon finalize 동작
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

                elif ev.key == pygame.K_o:
                    try:
                        # 1) 우선 map_num 기반 기본 경로를 보여주고
                        default_path = path_for_mapnum(map_num)
                        s = input(f"Open map path (ENTER = {default_path}) : ").strip()
                        path = default_path if not s else s

                        loaded = load_map_snippet(path)
                        data = loaded
                        tf = Transform(data.width, data.height, canvas, margin=30)

                        current_pts = []
                        dragging_rect = False
                        drag_start = None
                        drag_now = None
                        history.clear()

                        last_path = path
                        set_status(f"Opened for edit: {path}")
                    except Exception as e:
                        set_status(f"Open failed: {e}")

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if gen_btn is not None and gen_btn.clicked(ev):
                    apply_random_map()
                    continue

                if open_by_id_btn is not None and open_by_id_btn.clicked(ev):
                    try:
                        n = int(open_id_box.text) if open_id_box.text.strip() else map_num
                        path = path_for_mapnum(n)
                        loaded = load_map_snippet(path)
                        data = loaded
                        tf = Transform(data.width, data.height, canvas, margin=30)

                        current_pts = []
                        dragging_rect = False
                        drag_start = None
                        drag_now = None
                        history.clear()

                        last_path = path
                        map_num = n
                        mapnum_text = str(map_num)
                        set_status(f"Opened for edit: {path}")
                    except Exception as e:
                        set_status(f"Open failed: {e}")
                    continue
                mx, my = ev.pos

                if panel.collidepoint(mx, my):
                    if mapnum_rect is not None and mapnum_rect.collidepoint(mx, my):
                        editing_mapnum = True
                        mapnum_text = ""
                        set_status("Editing map num... type digits, ENTER apply")
                    continue

                if not canvas.collidepoint(mx, my):
                    continue

                # ignore canvas drawing while dragging UI sliders
                if any_widget_dragging():
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

        # status bar (y-flow)
        status_bar = pygame.Rect(panel.x + 20, y, PANEL_W - 40, 28)
        pygame.draw.rect(screen, (255, 245, 245), status_bar, border_radius=6)
        pygame.draw.rect(screen, (220, 180, 180), status_bar, 1, border_radius=6)
        msg = status if (time.time() - status_t < 5.0) else "(idle)"
        draw_text(screen, f"Status: {msg}", status_bar.x + 8, status_bar.y + 5, font, color=(120, 0, 0))
        y += 44  # status bar 아래 여백 포함

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
        draw_text(screen, f"Undo stack: {len(history)}", panel.x + 20, y, font); y += 26

        # ---- Random controls ----
        draw_text(screen, "Random / Open Controls", panel.x + 20, y, font_big); y += 30
        if not RANDOM_MAP_OK:
            draw_text(screen, "random_map.py import failed", panel.x + 20, y, font, color=(140, 0, 0))
            y += 22

        slider_w = PANEL_W - 60
        slider_h = 14

        difficulty_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
        difficulty_slider.draw(screen, font)
        y += 54

        if HAS_OVERRIDE_API:
            wallrect_slider.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, slider_h)
            wallrect_slider.draw(screen, font)
            y += 54
        else:
            draw_text(screen, "Wall-rect bias: (difficulty table only)", panel.x + 30, y, font, color=(80, 80, 80))
            y += 28

        # Generate button
        gen_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 38), "R / Click: Generate Random Map")
        gen_btn.draw(screen, font)
        y += 54

        # Open-by-id textbox + button (GUI)
        open_id_box.rect = pygame.Rect(panel.x + 30, y + 18, slider_w, 28)
        open_id_box.draw(screen, font)
        y += 62

        open_by_id_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 38), "Open for Edit (by map #)")
        open_by_id_btn.draw(screen, font, fg=(20, 20, 80), bg=(235, 235, 245), border=(120, 120, 160))
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
            "O: open existing map (edit)",
            "TAB: finalize poly (poly tools)",
            
        ]
        for k in keys:
            draw_text(screen, k, panel.x + 20, y, font, color=(40, 40, 40))
            y += 20

        y += 10
        if last_path:
            p = last_path
            if len(p) > 50:
                p = "..." + p[-47:]
            draw_text(screen, f"Last: {p}", panel.x + 20, y, font, color=(80, 80, 80))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
