#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-series Evacuation-100 Plotter (with hard cutoff at episode 15000)
- Reads all *.txt under ~/evacuation_100s
- Each txt: one number per line (evacuation_100_time per episode index)
- Always truncates series to episodes <= 15000
- Plots all series on ONE figure with optional smoothing (MA/EMA)
- Rich customization: fonts, colors/modes, linewidths, legend, grid, y-clip, raw-trace overlay, aliases, per-series style overrides

Display modes:
- LUMA_ONLY_MODE: grayscale luminance only, solid lines (Ours stays dark-blue solid)
- BLACKWHITE_MODE: grayscale + varied dash patterns
- else: color mode

Markers:
- MARKERS_ENABLE: add markers to plotted lines (legend reflects markers)
- MARKER_MODE: "uniform" or "varied"
- Marker style is print-friendly
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

# --- Legend order (display names after resolve_display_name) ---
LEGEND_ORDER = [
    "Ours",
    "Epsilon = 0",
    "R_danger = 0",
    "R_alived = 0",
    "R_penalty = 0",
    "R_base = 0",
]

# =========================
# Global
# =========================
HOME_DIR   = os.path.expanduser("~")
ROOT_DIR   = os.path.join(HOME_DIR, "evacuation_100s")
OUT_DIR    = ROOT_DIR
SAVE_NAME  = "evac100_multi"
SAVE_DPI   = 300
SAVE_FORMAT= "png"

MAX_EPISODE = 4300

# --- Figure / fonts ---
FIGSIZE = (15, 10)
FONT_SIZES = {"title": 30, "axes": 35, "ticks": 30, "legend": 30}
FONT_MODE = "serif"              # "serif" | "sans" | "mono" | "custom"
FONT_FAMILY = "DejaVu Serif"
FONT_FALLBACKS: List[str] = []

# --- Titles / axes / legend ---
PLOT_TITLE      = ""
XLABEL          = "Training Episode"
YLABEL          = "Time Step"
LEGEND_TITLE    = ""
LEGEND_LOC      = "upper right"

SHOW_GRID       = True
# ★ Grid 커스텀(더 잘 보이게)
GRID_STYLE = dict(color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.7)

# --- Raw trace overlay ---
SHOW_RAW_TRACE      = False
RAW_TRACE_ALPHA     = 0.18
RAW_TRACE_LINEWIDTH = 1

# --- Y clipping ---
CLIP_Y_MIN: Optional[float] = None
CLIP_Y_MAX: Optional[float] = None

# --- Unified smoothing controls ---
SMOOTH_ENABLE    = True
SMOOTH_KIND      = "ma"    # "ma" | "ema"
SMOOTH_STRENGTH  = 0.80    # 0.0 ~ 1.0 (higher → stronger smoothing)

MA_WINDOW_MIN    = 5
MA_WINDOW_MAX    = 301
EMA_ALPHA_MIN    = 0.05
EMA_ALPHA_MAX    = 0.50

# --- Lines ---
DEFAULT_LINEWIDTH = 2
H_BLUE   = 253/360.0
H_ORANGE = 30/360.0
H_GREEN  = 120/360.0
H_RED    = 0/360.0
H_PURPLE = 280/360.0
H_YELLOW = 55/360.0

COLOR_CYCLE_HUES = [H_RED, H_ORANGE, H_PURPLE, H_YELLOW, H_GREEN, H_GREEN, 0.58, 0.33, 0.12, 0.75, 0.5, 0.5]
DEFAULT_S = 0.75
DEFAULT_L = 0.42
GLOBAL_S_SCALE = 0.9
GLOBAL_L_SCALE = 1.00
GLOBAL_H_SHIFT = 0.00

SERIES_STYLE_OVERRIDES: Dict[str, Dict[str, object]] = {"Ours": {"linewidth": 5}}

FILENAME_ALIASES: Dict[str, str] = {
    "eva100_A0":"R_alived = 0",
    "eva100_B0":"R_danger = 0",
    "eva100_P0":"R_penalty = 0",
    "eva100_F0":"R_base = 0",
    "eva100_E0":"Epsilon = 0",
    "eva100_SOTA":"Ours"
}

INCLUDE_REGEX: Optional[str] = r".*\.txt$"
EXCLUDE_REGEX: Optional[str] = None

# --- Display modes ---
LUMA_ONLY_MODE   = False   # grayscale luminance only (solid lines)
BLACKWHITE_MODE  = False   # grayscale + varied dash (ignored if LUMA_ONLY_MODE=True)

# --- Dash patterns (offset, on_off_seq) or string ---
LINE_STYLES: List[Union[str, Tuple[int, Tuple[int, ...]]]] = [
    "solid",
    (0, (5, 5)),
    (0, (2, 4)),
    (0, (8, 6, 2, 6)),
    (0, (2, 2, 10, 2)),
    (0, (1, 3)),
    (0, (10, 2, 2, 2, 2, 2)),
    (0, (3, 3, 1, 3)),
]

# --- Markers ---
MARKERS_ENABLE  = True
MARKER_MODE     = "varied"   # "uniform" | "varied"
MARKER_SIZE_PT  = 15
MARKER_EVERY: Union[int, str] = "auto"  # int (e.g., 50) or "auto"
MARKER_EDGEWIDTH = 1.4
UNIFORM_MARKER  = "o"        # used when MARKER_MODE="uniform"

MARKER_CYCLE = [
    "o", "s", "^", "D", "v", "P", "X", "*", ">", "<", "h", "H", "1", "2", "3", "4"
]

EXCLUDE_MARKER_NAMES = {"Ours"}

# =========================
# Font utils
# =========================
def _installed_family_names() -> set:
    return {f.name for f in fm.fontManager.ttflist}

def _filter_installed(cands: List[str]) -> List[str]:
    inst = _installed_family_names()
    return [c for c in cands if c in inst]

def _base_candidates_for_mode() -> List[str]:
    mode = (FONT_MODE or "").lower()
    if mode in ("serif", "serifed"):
        return ["DejaVu Serif", "Times New Roman", "Times"]
    if mode in ("sans", "sans-serif"):
        return ["DejaVu Sans", "Helvetica", "Arial"]
    if mode in ("mono", "monospace"):
        return ["DejaVu Sans Mono", "Consolas", "Menlo"]
    if FONT_FAMILY and FONT_FAMILY.lower().endswith((".ttf", ".otf")):
        if os.path.isfile(FONT_FAMILY):
            try:
                fm.fontManager.addfont(FONT_FAMILY)
                prop = fm.FontProperties(fname=FONT_FAMILY)
                return [prop.get_name()]
            except Exception:
                pass
        return []
    return [FONT_FAMILY] if FONT_FAMILY else []

def _resolve_font_list() -> List[str]:
    base = _base_candidates_for_mode()
    cands = base + list(FONT_FALLBACKS)
    used = _filter_installed([c for c in cands if c])
    mode = (FONT_MODE or "").lower()
    if not used:
        if mode in ("serif", "serifed", "custom"):
            used = ["DejaVu Serif"]
        elif mode in ("mono", "monospace"):
            used = ["DejaVu Sans Mono"]
        else:
            used = ["DejaVu Sans"]
    seen, out = set(), []
    for name in used:
        if name not in seen:
            out.append(name); seen.add(name)
    return out

def resolve_display_name(fn: str) -> str:
    base = os.path.splitext(fn)[0]
    if fn in FILENAME_ALIASES:
        return FILENAME_ALIASES[fn]
    if base in FILENAME_ALIASES:
        return FILENAME_ALIASES[base]
    return base

def _math_fontset_for_mode() -> str:
    mode = (FONT_MODE or "").lower()
    if mode in ("serif", "serifed", "custom"):
        return "dejavuserif"
    return "dejavusans"

def apply_font_settings():
    families = _resolve_font_list()
    mode = (FONT_MODE or "").lower()
    if mode in ("serif", "serifed", "custom"):
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = families
    elif mode in ("mono", "monospace"):
        mpl.rcParams["font.family"] = "monospace"
        mpl.rcParams["font.monospace"] = families
    else:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = families

    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"]  = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]
    mpl.rcParams["mathtext.fontset"] = _math_fontset_for_mode()
    mpl.rcParams["axes.unicode_minus"] = False
    try:
        fm._rebuild()
    except Exception:
        pass

# =========================
# Colors
# =========================
def hsl_to_rgb_hex(h: float, s: float, l: float) -> str:
    h = (h + GLOBAL_H_SHIFT) % 1.0
    s = max(0.0, min(1.0, s * GLOBAL_S_SCALE))
    l = max(0.0, min(1.0, l * GLOBAL_L_SCALE))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def color_from_cycle(idx: int) -> str:
    h = COLOR_CYCLE_HUES[idx % len(COLOR_CYCLE_HUES)]
    return hsl_to_rgb_hex(h, DEFAULT_S, DEFAULT_L)

# =========================
# Data
# =========================
@dataclass
class Series:
    name: str
    x: np.ndarray
    y: np.ndarray
    y_smooth: Optional[np.ndarray] = None

def read_txt_series(path: str) -> List[float]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    vals: List[float] = []
    for ln in lines:
        tokens = re.split(r"[,\s]+", ln)
        for tok in tokens:
            if tok:
                try:
                    vals.append(float(tok))
                except Exception:
                    pass
    return vals

def load_all_series(root: str) -> List[Series]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")

    inc_re = re.compile(INCLUDE_REGEX) if INCLUDE_REGEX else None
    exc_re = re.compile(EXCLUDE_REGEX) if EXCLUDE_REGEX else None

    files = [fn for fn in os.listdir(root) if os.path.isfile(os.path.join(root, fn))]
    files.sort()

    series_list: List[Series] = []
    for fn in files:
        if inc_re and not inc_re.match(fn):
            continue
        if exc_re and exc_re.match(fn):
            continue

        path = os.path.join(root, fn)
        y_list = read_txt_series(path)
        if not y_list:
            continue

        x = np.arange(1, len(y_list)+1, dtype=float)
        y = np.asarray(y_list, dtype=float)

        mask = x <= MAX_EPISODE
        if not np.any(mask):
            continue
        x = x[mask]
        y = y[mask]

        name = resolve_display_name(fn)
        series_list.append(Series(name=name, x=x, y=y))
    return series_list

# =========================
# Smoothing
# =========================
def _compute_ma_window_from_strength(strength: float) -> int:
    s = float(np.clip(strength, 0.0, 1.0))
    win = int(round(MA_WINDOW_MIN + s * (MA_WINDOW_MAX - MA_WINDOW_MIN)))
    if win % 2 == 0:
        win += 1
    return max(1, win)

def _compute_ema_alpha_from_strength(strength: float) -> float:
    s = float(np.clip(strength, 0.0, 1.0))
    alpha = EMA_ALPHA_MAX - s * (EMA_ALPHA_MAX - EMA_ALPHA_MIN)
    return float(np.clip(alpha, 1e-6, 1.0))

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(y) < 2:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="symmetric")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ypad, kernel, mode="valid")

def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0 or len(y) < 2:
        return y.copy()
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i-1]
    return out

def apply_smoothing(series: Series) -> Series:
    if not SMOOTH_ENABLE:
        series.y_smooth = None
        return series

    kind = (SMOOTH_KIND or "").lower()
    if kind == "ma":
        win = _compute_ma_window_from_strength(SMOOTH_STRENGTH)
        series.y_smooth = moving_average(series.y, win)
    elif kind == "ema":
        alpha = _compute_ema_alpha_from_strength(SMOOTH_STRENGTH)
        series.y_smooth = ema(series.y, alpha)
    else:
        series.y_smooth = None
    return series

# =========================
# Legend ordering
# =========================
def _apply_legend_order(ax, handles: List[Line2D], order: List[str]) -> List[Line2D]:
    last_by_label: Dict[str, Line2D] = {}
    for h in handles:
        lab = h.get_label()
        if not lab or lab == "_nolegend_":
            continue
        last_by_label[lab] = h

    ordered: List[Line2D] = []
    seen = set()
    for lab in order:
        h = last_by_label.get(lab)
        if h is not None:
            ordered.append(h)
            seen.add(lab)

    for h in handles:
        lab = h.get_label()
        if not lab or lab == "_nolegend_" or lab in seen:
            continue
        ordered.append(h)
        seen.add(lab)
    return ordered

# =========================
# Style resolution
# =========================
def _series_color_and_lw(idx: int, name: str) -> Tuple[str, float, Union[str, Tuple[int, Tuple[int, ...]]]]:
    lw = float(SERIES_STYLE_OVERRIDES.get(name, {}).get("linewidth", DEFAULT_LINEWIDTH))
    OURS_COLOR = "#1f3b8b"

    if name == "Ours":
        return OURS_COLOR, lw, "solid"

    if LUMA_ONLY_MODE:
        gray_levels = np.linspace(0.20, 0.75, 10)
        color = f"{gray_levels[idx % len(gray_levels)]:.2f}"
        return color, lw, "solid"

    if BLACKWHITE_MODE:
        gray_levels = np.linspace(0.15, 0.70, 8)
        color = f"{gray_levels[idx % len(gray_levels)]:.2f}"
        dash_pattern = LINE_STYLES[idx % len(LINE_STYLES)]
        return color, lw, dash_pattern

    # Color mode
    color = color_from_cycle(idx)
    if "color" in SERIES_STYLE_OVERRIDES.get(name, {}):
        color = SERIES_STYLE_OVERRIDES[name]["color"]
    return color, lw, "solid"

def _series_marker(idx: int, name: str) -> str:
    if MARKER_MODE == "uniform":
        return UNIFORM_MARKER
    if name == "Ours":
        return "o"
    return MARKER_CYCLE[idx % len(MARKER_CYCLE)]

def _marker_every_for_length(n_points: int) -> int:
    """
    Determine how often to draw markers.
    Larger values → fewer markers.
    """
    if isinstance(MARKER_EVERY, int) and MARKER_EVERY > 0:
        return MARKER_EVERY
    MAX_MARKERS = 15
    spacing = max(1, int(round(n_points / MAX_MARKERS)))
    return spacing

def _marker_face_edge_colors(line_color: str) -> Tuple[str, str]:
    """
    ★ 마커 내부와 테두리를 모두 선 색으로 통일
    (grayscale 문자열에도 동일 적용)
    """
    mfc = line_color
    mec = line_color
    return mfc, mec

# =========================
# Plotting
# =========================
def apply_axes_style(ax, ylim: Optional[Tuple[float, float]]):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        # ★ grid를 라인 위로 올려 더 잘 보이게
        ax.set_axisbelow(False)
        ax.grid(**GRID_STYLE)
        # 필요시 y축만: ax.yaxis.grid(True, **GRID_STYLE); ax.xaxis.grid(False)
    if ylim is not None:
        ax.set_ylim(*ylim)

def plot_all(series_list: List[Series], out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles: List[Line2D] = []

    for i, s in enumerate(series_list):
        color, lw, dash_pattern = _series_color_and_lw(i, s.name)

        # Raw trace (no markers)
        if SHOW_RAW_TRACE:
            ax.plot(
                s.x, s.y,
                color=color,
                linewidth=RAW_TRACE_LINEWIDTH,
                alpha=RAW_TRACE_ALPHA,
                linestyle=dash_pattern
            )

        # Marker config
        marker_kwargs = {}
        if MARKERS_ENABLE and (s.name not in EXCLUDE_MARKER_NAMES):
            marker = _series_marker(i, s.name)
            mfc, mec = _marker_face_edge_colors(color)
            marker_kwargs.update(dict(
                marker=marker,
                markersize=MARKER_SIZE_PT,
                markerfacecolor=mfc,
                markeredgecolor=mec,
                markeredgewidth=MARKER_EDGEWIDTH,
                markevery=_marker_every_for_length(len(s.x))
            ))

        # Smooth & plot
        stmp = apply_smoothing(s)
        ydraw = stmp.y_smooth if (stmp.y_smooth is not None) else s.y
        xarr = np.asarray(s.x, dtype=float)
        yarr = np.asarray(ydraw, dtype=float)
        n = min(len(xarr), len(yarr))
        xarr = xarr[:n]; yarr = yarr[:n]

        if CLIP_Y_MIN is not None:
            yarr = np.maximum(yarr, CLIP_Y_MIN)
        if CLIP_Y_MAX is not None:
            yarr = np.minimum(yarr, CLIP_Y_MAX)

        line, = ax.plot(
            xarr, yarr,
            color=color, linewidth=lw, label=s.name,
            linestyle=dash_pattern,
            **marker_kwargs
        )
        handles.append(line)

    ax.set_xlabel(XLABEL, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL, fontsize=FONT_SIZES["axes"])

    ylim = None
    if (CLIP_Y_MIN is not None) or (CLIP_Y_MAX is not None):
        ymin = CLIP_Y_MIN if CLIP_Y_MIN is not None else ax.get_ylim()[0]
        ymax = CLIP_Y_MAX if CLIP_Y_MAX is not None else ax.get_ylim()[1]
        ylim = (ymin, ymax)
    apply_axes_style(ax, ylim)

    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE, fontsize=FONT_SIZES["title"])

    if handles:
        ordered = _apply_legend_order(ax, handles, LEGEND_ORDER)
        ax.legend(handles=ordered, title=LEGEND_TITLE, frameon=False, loc=LEGEND_LOC)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SAVE_NAME}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {out_path}")

# =========================
# Main
# =========================
def main():
    apply_font_settings()
    series_list = load_all_series(ROOT_DIR)
    if not series_list:
        print(f"[WARN] No valid .txt series (<= {MAX_EPISODE} episodes) found under: {ROOT_DIR}")
        return
    plot_all(series_list, OUT_DIR)

if __name__ == "__main__":
    main()
