#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ADDS Map Editor (Pygame)

[DISABLED]
- Random map generation UI/import is commented out for now.

[NEW] Background image layer
- Load semi-transparent background image
- BG Edit Mode: move / rotate / zoom / alpha
"""

import os
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

import pygame

# -----------------------------
# Random map generator is disabled for now.
# 원래 random_map.py를 import하던 부분은 임시로 주석 처리합니다.
# -----------------------------
# try:
#     import random_map as rm  # Shapely-based module
#     RANDOM_MAP_OK = True
# except Exception:
#     rm = None
#     RANDOM_MAP_OK = False
rm = None
RANDOM_MAP_OK = False
RANDOM_MAP_FEATURE_ENABLED = False

MAP_INFO_DIR = os.path.join(os.getcwd(), "map_infos")


def path_for_mapnum(n: int) -> str:
    return os.path.join(MAP_INFO_DIR, f"map_{n}.json")


WINDOW_W, WINDOW_H = 1280, 900
PANEL_W = 420

# -----------------------------
# Visual theme
# -----------------------------
THEME = {
    # Minimal neutral palette
    "app_bg": (246, 247, 249),
    "canvas_bg": (250, 250, 250),
    "canvas_border": (210, 214, 220),
    "panel_bg": (255, 255, 255),
    "panel_border": (228, 231, 235),
    "card_bg": (255, 255, 255),
    "card_border": (228, 231, 235),
    "text": (32, 36, 41),
    "muted": (120, 126, 134),
    "primary": (64, 92, 150),
    "primary_soft": (238, 242, 248),
    "warning": (166, 100, 48),
    "warning_soft": (252, 245, 238),
    "danger": (170, 70, 70),
    "danger_soft": (250, 238, 238),
    "obstacle": (76, 82, 92),
    "exit": (64, 92, 150),
}
CANVAS_BG = THEME["canvas_bg"]

GRID_ON = True
GRID_STEP = 1          # snap step in map units
GRID_DRAW_STEP = 1     # grid lines step in map units
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
        self.base_scale = min(avail_w / max(1, map_w), avail_h / max(1, map_h))

        self.origin_x = canvas_rect.x + margin
        self.origin_y = canvas_rect.y + margin

        # View transform for editor zoom/pan.
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    @property
    def scale(self) -> float:
        return self.base_scale * self.view_zoom

    def map_to_px(self, p: Point) -> Tuple[int, int]:
        x, y = p
        px = int(self.origin_x + self.pan_x + x * self.scale)
        py = int(self.origin_y + self.pan_y + (self.map_h - y) * self.scale)
        return px, py

    def px_to_map_float(self, px: int, py: int) -> Tuple[float, float]:
        x = (px - self.origin_x - self.pan_x) / max(1e-9, self.scale)
        y = self.map_h - (py - self.origin_y - self.pan_y) / max(1e-9, self.scale)
        return x, y

    def px_to_map(self, px: int, py: int) -> Point:
        x, y = self.px_to_map_float(px, py)
        return int(round(x)), int(round(y))

    def zoom_at_px(self, px: int, py: int, factor: float):
        # Cursor 기준 zoom: 확대 전 커서 아래 map 좌표를 확대 후에도 같은 화면 위치에 가깝게 유지
        before_x, before_y = self.px_to_map_float(px, py)
        self.view_zoom = max(0.2, min(30.0, self.view_zoom * factor))
        after_px = self.origin_x + self.pan_x + before_x * self.scale
        after_py = self.origin_y + self.pan_y + (self.map_h - before_y) * self.scale
        self.pan_x += px - after_px
        self.pan_y += py - after_py

    def pan_by(self, dx: float, dy: float):
        self.pan_x += dx
        self.pan_y += dy

    def reset_view(self):
        self.view_zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0


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


def polygon_area(poly: PolygonPts) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def sanitize_polygon(poly: PolygonPts) -> PolygonPts:
    """Remove duplicated consecutive points and drop degenerate polygons."""
    cleaned: PolygonPts = []
    for x, y in poly:
        p = (int(x), int(y))
        if not cleaned or cleaned[-1] != p:
            cleaned.append(p)

    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    # 점이 3개 미만이면 polygon으로 저장하지 않음.
    if len(cleaned) < 3:
        return []

    # 서로 다른 점이 3개 미만이면 polygon으로 저장하지 않음.
    if len(set(cleaned)) < 3:
        return []

    # 면적이 거의 0이면 선/점으로 본다.
    if polygon_area(cleaned) < 1e-6:
        return []

    return cleaned


def _clip_edge(poly: List[Tuple[float, float]], edge: str, value: float) -> List[Tuple[float, float]]:
    if not poly:
        return []

    def inside(p: Tuple[float, float]) -> bool:
        x, y = p
        if edge == "left":
            return x >= value
        if edge == "right":
            return x <= value
        if edge == "bottom":
            return y >= value
        if edge == "top":
            return y <= value
        return True

    def intersect(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        if edge in ("left", "right"):
            t = 0.0 if abs(dx) < 1e-12 else (value - ax) / dx
            return (value, ay + t * dy)
        else:
            t = 0.0 if abs(dy) < 1e-12 else (value - ay) / dy
            return (ax + t * dx, value)

    out: List[Tuple[float, float]] = []
    prev = poly[-1]
    prev_in = inside(prev)

    for cur in poly:
        cur_in = inside(cur)
        if cur_in:
            if not prev_in:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_in:
            out.append(intersect(prev, cur))
        prev, prev_in = cur, cur_in

    return out


def clip_polygon_to_rect(poly: PolygonPts, x_min: int, y_min: int, x_max: int, y_max: int) -> PolygonPts:
    """Axis-aligned rectangular clipping. Keeps partial polygons inside selected region."""
    work: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in poly]
    work = _clip_edge(work, "left", float(x_min))
    work = _clip_edge(work, "right", float(x_max))
    work = _clip_edge(work, "bottom", float(y_min))
    work = _clip_edge(work, "top", float(y_max))

    # Remove consecutive duplicates and round to map-unit integer coordinates.
    out: PolygonPts = []
    for x, y in work:
        p = (int(round(x)), int(round(y)))
        if not out or out[-1] != p:
            out.append(p)
    if len(out) >= 2 and out[0] == out[-1]:
        out.pop()

    # Degenerate polygon 제거: 점 3개 미만/중복점/면적 0 polygon은 버림.
    return sanitize_polygon(out)


def crop_map_data(data: MapData, a: Point, b: Point) -> MapData:
    """Crop selected rectangle and translate it to a new local coordinate system."""
    x1, y1 = a
    x2, y2 = b
    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    x_min = max(0, min(data.width, x_min))
    x_max = max(0, min(data.width, x_max))
    y_min = max(0, min(data.height, y_min))
    y_max = max(0, min(data.height, y_max))

    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)
    if new_w <= 0 or new_h <= 0:
        raise ValueError("Selected crop area is too small.")

    new_obstacles: List[List[List[int]]] = []
    for poly in data.obstacles:
        pts = [(int(p[0]), int(p[1])) for p in poly]
        clipped = clip_polygon_to_rect(pts, x_min, y_min, x_max, y_max)
        if clipped:
            shifted_pts = [(int(x - x_min), int(y - y_min)) for x, y in clipped]
            shifted = sanitize_polygon(shifted_pts)
            if shifted:
                new_obstacles.append([[int(x), int(y)] for x, y in shifted])

    new_exits: List[List[Tuple[int, int]]] = []
    for poly in data.exits:
        pts = [(int(p[0]), int(p[1])) for p in poly]
        clipped = clip_polygon_to_rect(pts, x_min, y_min, x_max, y_max)
        if clipped:
            shifted_pts = [(int(x - x_min), int(y - y_min)) for x, y in clipped]
            shifted = sanitize_polygon(shifted_pts)
            if shifted:
                new_exits.append([(int(x), int(y)) for x, y in shifted])

    # 최종 저장 직전에도 한 번 더 강제 필터링.
    # crop 과정에서 1~2점짜리 조각이나 선분이 생기면 모두 제거한다.
    new_obstacles = [
        [[int(x), int(y)] for x, y in clean]
        for poly in new_obstacles
        if (clean := sanitize_polygon([(int(p[0]), int(p[1])) for p in poly]))
    ]
    new_exits = [
        [(int(x), int(y)) for x, y in clean]
        for poly in new_exits
        if (clean := sanitize_polygon([(int(p[0]), int(p[1])) for p in poly]))
    ]

    return MapData(width=new_w, height=new_h, obstacles=new_obstacles, exits=new_exits)


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
def draw_text(surf, text, x, y, font, color=None):
    color = THEME["text"] if color is None else color
    surf.blit(font.render(text, True, color), (x, y))


def draw_card(surface: pygame.Surface, rect: pygame.Rect, bg=None, border=None, radius: int = 8):
    bg = THEME["card_bg"] if bg is None else bg
    border = THEME["card_border"] if border is None else border
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)


def draw_section_title(surface: pygame.Surface, text: str, x: int, y: int, font):
    draw_text(surface, text, x, y, font, THEME["text"])


def draw_pill(surface: pygame.Surface, rect: pygame.Rect, text: str, font, fg, bg, border=None):
    border = bg if border is None else border
    pygame.draw.rect(surface, bg, rect, border_radius=rect.h // 2)
    pygame.draw.rect(surface, border, rect, 1, border_radius=rect.h // 2)
    label = font.render(text, True, fg)
    surface.blit(label, label.get_rect(center=rect.center))


def draw_grid(surface: pygame.Surface, tf: Transform, canvas: pygame.Rect, step_map: int):
    if step_map <= 0:
        return
    minor = (238, 240, 243)
    major = (222, 226, 231)
    x = 0
    while x <= tf.map_w:
        px, _ = tf.map_to_px((x, 0))
        color = major if x % 10 == 0 else minor
        width = 1
        pygame.draw.line(surface, color, (px, canvas.y), (px, canvas.bottom), width)
        x += step_map
    y = 0
    while y <= tf.map_h:
        _, py = tf.map_to_px((0, y))
        color = major if y % 10 == 0 else minor
        width = 1
        pygame.draw.line(surface, color, (canvas.x, py), (canvas.right, py), width)
        y += step_map


def draw_polygon_layer(surface: pygame.Surface, tf: Transform, pts: PolygonPts,
                       fill_rgb, outline_rgb, fill_alpha: int):
    pts_px = [tf.map_to_px(p) for p in pts]
    if len(pts_px) < 3:
        return
    fill = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(fill, (*fill_rgb, fill_alpha), pts_px)
    surface.blit(fill, (0, 0))
    pygame.draw.polygon(surface, outline_rgb, pts_px, width=2)
    for px, py in pts_px:
        pygame.draw.circle(surface, (255, 255, 255), (px, py), 4)
        pygame.draw.circle(surface, outline_rgb, (px, py), 4, 1)


def draw_points_and_edges(surface: pygame.Surface, tf: Transform, pts: PolygonPts, color):
    if not pts:
        return
    pts_px = [tf.map_to_px(p) for p in pts]
    if len(pts_px) >= 2:
        pygame.draw.lines(surface, color, False, pts_px, width=3)
    for (x, y) in pts_px:
        pygame.draw.circle(surface, (255, 255, 255), (x, y), POINT_RADIUS + 2)
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
        draw_text(surf, f"{self.label}: {self.value:.0f}", self.rect.x, self.rect.y - 18, font, THEME["muted"])
        track = self.rect
        pygame.draw.rect(surf, (232, 234, 237), track, border_radius=track.h // 2)
        t = (self.value - self.vmin) / max(1e-9, (self.vmax - self.vmin))
        tx = int(track.x + t * track.w)
        pygame.draw.circle(surf, THEME["primary"], (tx, track.centery), 6)


class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label

    def clicked(self, ev: pygame.event.Event) -> bool:
        return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos))

    def draw(self, surf: pygame.Surface, font: pygame.font.Font,
             fg=None, bg=None, border=None):
        fg = THEME["text"] if fg is None else fg
        bg = (255, 255, 255) if bg is None else bg
        border = THEME["card_border"] if border is None else border
        pygame.draw.rect(surf, bg, self.rect, border_radius=7)
        pygame.draw.rect(surf, border, self.rect, width=1, border_radius=7)
        label = font.render(self.label, True, fg)
        surf.blit(label, label.get_rect(center=self.rect.center))

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
                pass
            else:
                if ev.unicode.isdigit():
                    self.text += ev.unicode
                    changed = True
        return changed

    def draw(self, surf: pygame.Surface, font: pygame.font.Font):
        if self.label:
            draw_text(surf, self.label, self.rect.x, self.rect.y - 22, font, THEME["muted"])

        bg = (255, 255, 255) if self.active else (248, 250, 252)
        border = THEME["primary"] if self.active else THEME["card_border"]

        pygame.draw.rect(surf, bg, self.rect, border_radius=10)
        pygame.draw.rect(surf, border, self.rect, 2 if self.active else 1, border_radius=10)

        show = self.text if self.text else ""
        draw_text(surf, show, self.rect.x + 12, self.rect.y + 6, font, THEME["text"])


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
# Background Image Helpers (NEW)
# -----------------------------
def pick_image_path_dialog() -> Optional[str]:
    """
    Open a file dialog to select an image.
    Handles Pygame/Tkinter event loop conflicts to prevent freezing.
    """
    # 1. Pygame의 마우스 제어권 해제 및 이벤트 플러시 (중요)
    #    이것을 하지 않으면 OS가 여전히 Pygame이 입력을 받고 있다고 착각할 수 있음
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)
    pygame.event.pump() 

    path = None
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Tkinter 루트 윈도우 생성
        root = tk.Tk()
        root.withdraw()  # 빈 부모 윈도우 숨기기

        # 2. 윈도우 순서 및 포커스 강제 조정
        #    이 설정들이 없으면 대화상자가 Pygame 창 뒤로 숨거나 클릭이 안 될 수 있음
        root.attributes("-topmost", True)
        root.lift()
        root.focus_force()
        root.update()  # 변경 사항 즉시 적용

        # 3. 파일 대화상자 열기
        path = filedialog.askopenfilename(
            parent=root,
            title="Select background image",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*"),
            ],
        )
        
        # 4. Tkinter 종료
        root.destroy()
        
    except Exception as e:
        print(f"Dialog failed: {e}")
        path = None
    
    # 5. Pygame으로 복귀 전 이벤트 큐 정리
    #    대화상자에서 발생한 잔여 이벤트가 게임에 영향을 주지 않도록 함
    pygame.event.clear()
    pygame.event.pump()
    
    return path if path else None


def load_image_any(path: str) -> pygame.Surface:
    # convert_alpha for transparency support
    surf = pygame.image.load(path)
    return surf.convert_alpha()

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
    pygame.display.set_caption("ADDS Map Editor")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("segoeui", 15) or pygame.font.SysFont("arial", 15)
    font_small = pygame.font.SysFont("segoeui", 13) or pygame.font.SysFont("arial", 13)
    font_big = pygame.font.SysFont("segoeui", 22, bold=True) or pygame.font.SysFont("arial", 22, bold=True)
    font_section = pygame.font.SysFont("segoeui", 16, bold=True) or pygame.font.SysFont("arial", 16, bold=True)

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

    # Crop/export selection state
    crop_dragging = False
    crop_start: Optional[Point] = None
    crop_now: Optional[Point] = None
    selected_crop: Optional[Tuple[Point, Point]] = None

    # View pan state
    view_panning = False
    view_pan_last = (0, 0)

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

    def square_crop_point(start: Point, raw: Point) -> Point:
        """
        Shift를 누른 상태에서 crop select를 할 때 선택 영역을 정사각형으로 보정합니다.
        start 지점에서 raw 지점 방향으로 가능한 한 같은 width/height를 갖도록 만들고,
        map boundary 밖으로 나가지 않도록 side 길이를 제한합니다.
        """
        sx, sy = start
        rx, ry = raw

        dx = rx - sx
        dy = ry - sy

        if dx == 0 and dy == 0:
            return raw

        sign_x = 1 if dx >= 0 else -1
        sign_y = 1 if dy >= 0 else -1

        side = max(abs(dx), abs(dy))

        # 정사각형이 map boundary 밖으로 나가지 않도록 가능한 최대 길이로 제한
        max_x = data.width - sx if sign_x > 0 else sx
        max_y = data.height - sy if sign_y > 0 else sy
        side = max(0, min(side, max_x, max_y))

        return clamp_point((sx + sign_x * side, sy + sign_y * side), data.width, data.height)

    def crop_point_from_mouse(mx: int, my: int) -> Point:
        """
        일반 crop은 mouse 위치를 그대로 map 좌표로 변환하고,
        Shift가 눌린 경우에는 crop_start를 기준으로 정사각형 끝점을 반환합니다.
        """
        p = map_click_to_point(mx, my)
        mods = pygame.key.get_mods()
        if crop_dragging and crop_start is not None and (mods & pygame.KMOD_SHIFT):
            p = square_crop_point(crop_start, p)
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

    def export_selected_crop():
        nonlocal selected_crop, crop_start, crop_now
        if selected_crop is None:
            set_status("Crop export: 먼저 5번 도구로 영역을 드래그해서 선택하세요.")
            return
        try:
            n = int(crop_export_box.text) if crop_export_box.text.strip() else (map_num + 1000)
            out = crop_map_data(data, selected_crop[0], selected_crop[1])
            path = path_for_mapnum(n)
            save_map_snippet(path, out)
            set_status(f"Crop exported: map_{n}.json ({out.width}x{out.height}, ob={len(out.obstacles)}, ex={len(out.exits)})")
        except Exception as e:
            set_status(f"Crop export failed: {e}")

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
    # Random generator params / functions are disabled for now.
    # 필요하면 아래 주석 처리된 구조를 되살려 random_map.py 연동을 복구하면 됩니다.
    # -----------------------------
    rand_params = {
        "difficulty": 5,
        "wall_rect_bias": 0.20,
    }
    HAS_OVERRIDE_API = False

    # def _apply_overrides_if_supported():
    #     ...
    #
    # def apply_random_map():
    #     ...

    def apply_random_map():
        set_status("Random map generation is disabled.")

    # -----------------------------
    # Background image state (NEW)
    # -----------------------------
    bg_path: Optional[str] = None
    bg_orig: Optional[pygame.Surface] = None  # original
    bg_edit_mode = False

    # position is in SCREEN pixels (not map), relative to screen
    bg_pos = [canvas.centerx, canvas.centery]  # center position (px)
    bg_scale = 1.0
    bg_angle = 0.0
    bg_alpha = 120  # 0..255

    # cache to avoid re-rotozoom every frame
    bg_cache_surf: Optional[pygame.Surface] = None
    bg_cache_key = None  # (id(bg_orig), scale, angle, alpha)

    # editing drag
    bg_dragging = False
    bg_drag_last = (0, 0)

    def bg_reset():
        nonlocal bg_pos, bg_scale, bg_angle, bg_cache_surf, bg_cache_key
        bg_pos = [canvas.centerx, canvas.centery]
        bg_scale = 1.0
        bg_angle = 0.0
        bg_cache_surf = None
        bg_cache_key = None

    def load_bg_image():
        nonlocal bg_path, bg_orig, bg_cache_surf, bg_cache_key

        # ✅ 다이얼로그 열기 전에 드래그 상태 리셋 (중요)
        reset_all_drag_states()
        pygame.event.clear()
        pygame.event.pump()

        path = pick_image_path_dialog()

        # ✅ 다이얼로그 닫힌 직후에도 한 번 더 리셋 + 큐 정리 (중요)
        reset_all_drag_states()
        pygame.event.clear()
        pygame.event.pump()

        if not path:
            set_status("BG: canceled")
            return
        try:
            img = load_image_any(path)
            bg_path = path
            bg_orig = img
            bg_reset()
            set_status(f"BG loaded: {os.path.basename(path)} (Edit mode에서 이동/회전/줌 가능)")
        except Exception as e:
            set_status(f"BG load failed: {e}")


    def get_bg_render_surface() -> Optional[pygame.Surface]:
        nonlocal bg_cache_surf, bg_cache_key
        if bg_orig is None:
            return None
        key = (id(bg_orig), round(bg_scale, 4), round(bg_angle, 2), int(bg_alpha))
        if bg_cache_surf is not None and bg_cache_key == key:
            return bg_cache_surf

        # rotozoom: angle in degrees, scale factor
        # NOTE: pygame rotates CCW by degrees
        s = pygame.transform.rotozoom(bg_orig, bg_angle, bg_scale)
        # alpha
        s = s.copy()
        s.set_alpha(int(bg_alpha))

        bg_cache_surf = s
        bg_cache_key = key
        return bg_cache_surf

    # -----------------------------
    # UI widgets
    # -----------------------------
    running = True
    last_path: Optional[str] = None

    mapnum_rect: Optional[pygame.Rect] = None
    gen_btn: Optional[Button] = None
    open_btn: Optional[Button] = None
    open_id_box = TextBox(pygame.Rect(0, 0, 0, 0), text=str(map_num), label="Open map # (e.g., 200)")
    open_by_id_btn: Optional[Button] = None

    crop_export_box = TextBox(pygame.Rect(0, 0, 0, 0), text=str(map_num + 1000), label="Export selected as map #")
    crop_export_btn: Optional[Button] = None

    # background UI buttons/sliders (NEW)
    bg_load_btn: Optional[Button] = None
    bg_edit_btn: Optional[Button] = None
    bg_alpha_slider = Slider(pygame.Rect(0, 0, 0, 0), 0.0, 255.0, float(bg_alpha), "BG Alpha (0..255)")

    difficulty_slider = Slider(pygame.Rect(0, 0, 0, 0), 1.0, 6.0, float(rand_params["difficulty"]), "Difficulty (1..6)")
    wallrect_slider   = Slider(pygame.Rect(0, 0, 0, 0), 0.0, 1.0, float(rand_params["wall_rect_bias"]), "Wall-rect bias")

    def reset_all_drag_states():
    # 슬라이더 드래그 고착 방지
        difficulty_slider.dragging = False
        wallrect_slider.dragging = False
        bg_alpha_slider.dragging = False

        # BG / crop / view 드래그 상태도 리셋
        nonlocal bg_dragging, crop_dragging, view_panning
        bg_dragging = False
        crop_dragging = False
        view_panning = False

    def any_widget_dragging() -> bool:
        return bg_alpha_slider.dragging

    while running:
        clock.tick(60)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # panel sliders
            changed = False

            # bg alpha slider
            changed |= bg_alpha_slider.handle_event(ev)

            open_id_box.handle_event(ev)
            crop_export_box.handle_event(ev)

            if changed:
                bg_alpha = int(round(bg_alpha_slider.value))
                bg_alpha_slider.value = float(bg_alpha)

            # buttons click
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Random map generation disabled.
                # if gen_btn is not None and gen_btn.clicked(ev):
                #     apply_random_map()
                #     continue

                if bg_load_btn is not None and bg_load_btn.clicked(ev):
                    load_bg_image()
                    continue

                if bg_edit_btn is not None and bg_edit_btn.clicked(ev):
                    bg_edit_mode = not bg_edit_mode
                    set_status("BG Edit Mode: ON" if bg_edit_mode else "BG Edit Mode: OFF")
                    continue

            if open_btn is not None and open_btn.clicked(ev):
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
                # BG edit keybinds (NEW)
                if bg_edit_mode and bg_orig is not None:
                    if ev.key == pygame.K_q:   # rotate -
                        bg_angle -= 2.0
                        set_status(f"BG rotate: {bg_angle:.1f} deg")
                    elif ev.key == pygame.K_e: # rotate +
                        bg_angle += 2.0
                        set_status(f"BG rotate: {bg_angle:.1f} deg")
                    elif ev.key == pygame.K_x: # reset
                        bg_reset()
                        set_status("BG reset")
                    # NOTE: ESC/Q quit 기존 로직과 충돌 가능 -> quit은 아래에서 처리
                    # (여기서는 q/e를 bg에 쓰니까 quit은 ESC로도 가능)

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

                    if ev.key in (pygame.K_ESCAPE,):
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

                if ev.key == pygame.K_ESCAPE:
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

                elif ev.key == pygame.K_5:
                    tool = "crop_select"
                    current_pts = []
                    dragging_rect = False
                    crop_dragging = False
                    crop_start = None
                    crop_now = None
                    selected_crop = None
                    set_status("Tool: CROP SELECT - drag area, then press P or button to export")

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

                elif ev.key == pygame.K_p:
                    export_selected_crop()

                elif ev.key == pygame.K_v:
                    tf.reset_view()
                    set_status("View reset")

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

                # Random map generation disabled.
                # elif ev.key == pygame.K_r:
                #     apply_random_map()

                elif ev.key == pygame.K_b:  # NEW: quick load bg
                    load_bg_image()

            # BG edit mouse controls (NEW)
            if bg_edit_mode and bg_orig is not None:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my) and (not panel.collidepoint(mx, my)) and (not any_widget_dragging()):
                        bg_dragging = True
                        bg_drag_last = (mx, my)

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    bg_dragging = False

                elif ev.type == pygame.MOUSEMOTION and bg_dragging:
                    mx, my = ev.pos
                    dx = mx - bg_drag_last[0]
                    dy = my - bg_drag_last[1]
                    bg_pos[0] += dx
                    bg_pos[1] += dy
                    bg_drag_last = (mx, my)

                # mouse wheel zoom (pygame 2: MOUSEWHEEL event)
                elif ev.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if canvas.collidepoint(mx, my) and (not panel.collidepoint(mx, my)):
                        # zoom
                        z = 1.08 if ev.y > 0 else 1.0 / 1.08
                        bg_scale *= z
                        bg_scale = max(0.05, min(20.0, bg_scale))
                        set_status(f"BG scale: {bg_scale:.2f}")

            # Canvas view zoom/pan controls
            if not bg_edit_mode:
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 2:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my):
                        view_panning = True
                        view_pan_last = (mx, my)
                        continue

                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 2:
                    view_panning = False

                elif ev.type == pygame.MOUSEMOTION and view_panning:
                    mx, my = ev.pos
                    dx = mx - view_pan_last[0]
                    dy = my - view_pan_last[1]
                    tf.pan_by(dx, dy)
                    view_pan_last = (mx, my)
                    continue

                elif ev.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if canvas.collidepoint(mx, my) and (not panel.collidepoint(mx, my)):
                        factor = 1.12 if ev.y > 0 else 1.0 / 1.12
                        tf.zoom_at_px(mx, my, factor)
                        set_status(f"View zoom: {tf.view_zoom:.2f}x")
                        continue

            # Existing drawing / open-by-id button
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # Random map generation disabled.
                # if gen_btn is not None and gen_btn.clicked(ev):
                #     apply_random_map()
                #     continue

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

                if crop_export_btn is not None and crop_export_btn.clicked(ev):
                    export_selected_crop()
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

                if any_widget_dragging():
                    continue

                # if BG edit mode, we don't draw polygons (to avoid conflict)
                if bg_edit_mode:
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

                elif tool == "crop_select":
                    crop_dragging = True
                    crop_start = p
                    crop_now = p
                    selected_crop = None
                    set_status(f"Crop start: {p}  (hold Shift for square)")

                elif tool == "delete":
                    try_delete_at(p)

            elif ev.type == pygame.MOUSEMOTION:
                if dragging_rect and drag_start is not None:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my):
                        drag_now = map_click_to_point(mx, my)

                if crop_dragging and crop_start is not None:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my):
                        crop_now = crop_point_from_mouse(mx, my)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if crop_dragging and crop_start is not None and crop_now is not None:
                    mx, my = ev.pos
                    if canvas.collidepoint(mx, my):
                        crop_now = crop_point_from_mouse(mx, my)

                    x1, y1 = crop_start
                    x2, y2 = crop_now
                    if abs(x2 - x1) <= 0 or abs(y2 - y1) <= 0:
                        set_status("Crop too small, ignored")
                        selected_crop = None
                    else:
                        selected_crop = (crop_start, crop_now)
                        x_min, x_max = sorted([x1, x2])
                        y_min, y_max = sorted([y1, y2])
                        set_status(f"Crop selected: ({x_min},{y_min})~({x_max},{y_max}) => {x_max-x_min}x{y_max-y_min}. Press P to export.")
                    crop_dragging = False

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
        screen.fill(THEME["app_bg"])
        pygame.draw.rect(screen, CANVAS_BG, canvas)

        # draw background image (NEW)
        bg_surf = get_bg_render_surface()
        if bg_surf is not None:
            rect = bg_surf.get_rect(center=(int(bg_pos[0]), int(bg_pos[1])))
            # clip to canvas
            prev_clip = screen.get_clip()
            screen.set_clip(canvas)
            screen.blit(bg_surf, rect.topleft)
            screen.set_clip(prev_clip)

        if GRID_ON:
            draw_grid(screen, tf, canvas, step_map=GRID_DRAW_STEP)

        # obstacles
        for poly in data.obstacles:
            pts = [(p[0], p[1]) for p in poly]
            draw_polygon_layer(screen, tf, pts,
                               fill_rgb=THEME["obstacle"],
                               outline_rgb=(15, 23, 42),
                               fill_alpha=95)

        # exits
        for poly in data.exits:
            pts = [(p[0], p[1]) for p in poly]
            draw_polygon_layer(screen, tf, pts,
                               fill_rgb=THEME["exit"],
                               outline_rgb=(29, 78, 216),
                               fill_alpha=110)

        # current polygon preview
        if tool in ("poly_ob", "poly_ex") and (not bg_edit_mode):
            col = THEME["obstacle"] if tool == "poly_ob" else THEME["primary"]
            draw_points_and_edges(screen, tf, current_pts, color=col)

        # current rect preview
        if dragging_rect and drag_start is not None and drag_now is not None and (not bg_edit_mode):
            pts = rect_from_drag(drag_start, drag_now)
            col = THEME["obstacle"] if tool == "rect_ob" else THEME["primary"]
            pts_px = [tf.map_to_px(p) for p in pts]
            pygame.draw.polygon(screen, col, pts_px, width=2)

        # crop selection preview
        crop_a = crop_start if crop_dragging else (selected_crop[0] if selected_crop else None)
        crop_b = crop_now if crop_dragging else (selected_crop[1] if selected_crop else None)
        if crop_a is not None and crop_b is not None and (not bg_edit_mode):
            pts = rect_from_drag(crop_a, crop_b)
            pts_px = [tf.map_to_px(p) for p in pts]
            pygame.draw.polygon(screen, THEME["warning"], pts_px, width=3)

        pygame.draw.rect(screen, THEME["canvas_border"], canvas, width=2)

        # panel
        panel = pygame.Rect(WINDOW_W - PANEL_W, 0, PANEL_W, WINDOW_H)
        pygame.draw.rect(screen, THEME["panel_bg"], panel)
        pygame.draw.line(screen, THEME["panel_border"], (panel.x, 0), (panel.x, WINDOW_H), 2)

        y = 14
        draw_text(screen, "ADDS Map Editor", panel.x + 22, y, font_big, THEME["text"]); y += 36

        # status bar
        status_bar = pygame.Rect(panel.x + 20, y, PANEL_W - 40, 30)
        is_idle = (time.time() - status_t >= 5.0)
        msg = status if not is_idle else "idle"
        status_bg = THEME["card_bg"] if is_idle else THEME["primary_soft"]
        status_fg = THEME["muted"] if is_idle else THEME["primary"]
        pygame.draw.rect(screen, status_bg, status_bar, border_radius=12)
        pygame.draw.rect(screen, THEME["card_border"], status_bar, 1, border_radius=12)
        draw_text(screen, msg, status_bar.x + 12, status_bar.y + 6, font, color=status_fg)
        y += 36

        mapnum_y = y
        mapnum_display = mapnum_text if editing_mapnum else str(map_num)
        draw_text(screen, f"Map num : {mapnum_display}", panel.x + 20, y, font); y += 20
        mapnum_rect = pygame.Rect(panel.x + 20, mapnum_y, PANEL_W - 40, 22)
        if editing_mapnum:
            pygame.draw.rect(screen, (255, 230, 230), mapnum_rect, border_radius=4)
        else:
            pygame.draw.rect(screen, (240, 240, 240), mapnum_rect, 1, border_radius=4)

        draw_text(screen, f"Size    : {data.width} x {data.height}", panel.x + 20, y, font); y += 20
        draw_text(screen, f"View    : {tf.view_zoom:.2f}x", panel.x + 20, y, font); y += 24

        tool_name = {
            "poly_ob": "Polygon OBSTACLE",
            "poly_ex": "Polygon EXIT",
            "rect_ob": "Rect OBSTACLE",
            "rect_ex": "Rect EXIT",
            "crop_select": "CROP SELECT / EXPORT",
            "delete":  "DELETE",
        }[tool]
        tool_fg = THEME["danger"] if tool == "delete" else THEME["primary"]
        tool_bg = THEME["danger_soft"] if tool == "delete" else THEME["primary_soft"]
        draw_pill(screen, pygame.Rect(panel.x + 20, y, PANEL_W - 40, 30), f"Tool: {tool_name}", font, tool_fg, tool_bg)
        y += 34

        draw_text(screen, f"Obstacles: {len(data.obstacles)}   Exits: {len(data.exits)}   Undo: {len(history)}", panel.x + 20, y, font); y += 24

        # ---- Background controls (NEW) ----
        draw_section_title(screen, "Background Image", panel.x + 20, y, font_section); y += 24

        bg_load_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 30), "Load BG Image (Click / B)")
        bg_load_btn.draw(screen, font, fg=(80, 40, 20), bg=(245, 235, 225), border=(180, 140, 120))
        y += 38

        label = "BG Edit Mode: ON (drag/scroll/QE)" if bg_edit_mode else "BG Edit Mode: OFF"
        bg_edit_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 30), label)
        bg_edit_btn.draw(screen, font, fg=(20, 20, 80), bg=(235, 235, 245), border=(120, 120, 160))
        y += 44

        bg_alpha_slider.rect = pygame.Rect(panel.x + 30, y + 18, PANEL_W - 60, 14)
        bg_alpha_slider.draw(screen, font)
        y += 42

        if bg_path:
            base = os.path.basename(bg_path)
            draw_text(screen, f"BG: {base}", panel.x + 30, y, font, color=THEME["muted"]); y += 20
            draw_text(screen, f"angle {bg_angle:.1f} / scale {bg_scale:.2f}", panel.x + 30, y, font_small, color=THEME["muted"]); y += 18
        else:
            draw_text(screen, "No BG loaded.", panel.x + 30, y, font_small, color=THEME["muted"]); y += 22

        # ---- Open / Export ----
        draw_section_title(screen, "Open / Export", panel.x + 20, y, font_section); y += 22

        slider_w = PANEL_W - 60

        open_id_box.rect = pygame.Rect(panel.x + 30, y + 14, slider_w, 24)
        open_id_box.draw(screen, font)
        y += 46

        open_by_id_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 30), "Open map")
        open_by_id_btn.draw(screen, font)
        y += 38

        crop_export_box.rect = pygame.Rect(panel.x + 30, y + 14, slider_w, 24)
        crop_export_box.draw(screen, font)
        y += 46

        crop_export_btn = Button(pygame.Rect(panel.x + 30, y, PANEL_W - 60, 30), "Export crop")
        crop_export_btn.draw(screen, font)
        y += 34

        if selected_crop is not None:
            a, b = selected_crop
            x_min, x_max = sorted([a[0], b[0]])
            y_min, y_max = sorted([a[1], b[1]])
            draw_text(screen, f"Crop: {x_max-x_min} x {y_max-y_min}", panel.x + 30, y, font_small, color=THEME["muted"])
            y += 18

        draw_section_title(screen, "Shortcuts", panel.x + 20, y, font_section); y += 22
        keys = [
            "1/2 Polygon ob/ex",
            "3/4 Rect ob/ex",
            "5 Crop select",
            "Shift Crop square",
            "D Delete",
            "Enter Finish poly",
            "S Save   L Load",
            "U Undo   C Clear",
            "Wheel Zoom   V Reset",
            "B Load BG",
            "ESC Quit",
        ]
        key_y0 = y
        col_w = (PANEL_W - 44) // 2
        for i, k in enumerate(keys):
            col = i % 2
            row = i // 2
            draw_text(screen, k, panel.x + 20 + col * col_w, key_y0 + row * 17, font_small, color=THEME["muted"])
        y = key_y0 + ((len(keys) + 1) // 2) * 17 + 8

        if last_path:
            p = last_path
            if len(p) > 50:
                p = "..." + p[-47:]
            draw_text(screen, f"Last: {p}", panel.x + 20, y, font, color=THEME["muted"])

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
