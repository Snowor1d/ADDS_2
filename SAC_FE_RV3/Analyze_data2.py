#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import csv
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from matplotlib import font_manager as fm

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import patheffects as pe  # legend/markers halo 효과용 (선택)

# =========================
# 전역 파라미터
# =========================
ROOT_DIR = os.path.expanduser("~/results")
OUT_DIR  = ROOT_DIR

SHOW_EPISODE_LEGEND = True  # episode 플롯 범례 표시 여부

# --- 그림/폰트 ---
FIGSIZE_EVAC   = (15, 7)
FIGSIZE_EPISOD = (10, 8)
#FONT_SIZES = {"title": 38, "axes": 0, "ticks": 38, "legend": 41}
FONT_SIZES = {"title": 25, "axes": 30, "ticks": 28, "legend": 28} 

FONT_MODE = "serif"          # 논문용 로마자 느낌이면 "serif"
FONT_FAMILY = "DejaVu Serif"  # MODE="custom"일 때만 사용 (정확한 폰트 이름)
# 한국어/한자 혼용 대비를 위한 폴백(시스템에 있는 것만 적용됨)
FONT_FALLBACKS = [

]

# --- 제목/축 레이블 ---
TITLE_MAP_FORMAT   = "Map {map_id} - {plot_name}"
TITLE_EVAC_NAME    = ""
TITLE_EPISODE_NAME = "Remained Agents per Step"
XLABEL_EVAC        = ""
YLABEL_EVAC        = "Time (s)"
XLABEL_EPISODE     = "Time (s)"
YLABEL_EPISODE     = "Unevacuated Agents"

# --- 저장 ---
SAVE_DPI    = 220
SAVE_FORMAT = "png"

# --- 밴드(음영) ---
BAND_MODE: Optional[str] = None  # None | "minmax" | "std" | "p25_75"
BAND_ALPHA = 0.18

# --- evac100 표시 제어 ---
EVAC_SHOW_MEAN   = True       # 평균선(hline)
EVAC_SHOW_BAND   = True       # min-max(or 선택 모드) 밴드
EVAC_SHOW_POINTS = True       # 개별 실험 점
EVAC_POINT_MODE  = "valid_only"  # "valid_only"(<MAX_TIMESTEP만) | "all"

# --- evac100: 로봇 버전 필터 ---
EVAC_ROBOT_ALLOW: Optional[List[str]] = None   # None(전체) | ["R", "T"] 등

# --- y축 범위 (None이면 자동) ---
YLIM_EVAC: Optional[Tuple[float, float]]    = None
YLIM_EPISODE: Optional[Tuple[float, float]] = None

# --- timeout 기준 ---
MAX_TIMESTEP = 3000  # 이 값 이상이면 timeout

# --- 맵/방법별 요약 통계 ---
# 반복 실험을 하나의 표본으로 보고 표본 표준편차(ddof=1)를 사용한다.
# 유효 evacuation time은 기존 플롯과 동일하게 MAX_TIMESTEP 미만만 포함한다.
SAVE_SUMMARY_STATS = True
SUMMARY_STATS_FILENAME = "map_robot_summary_stats.csv"
STD_DDOF = 1
PRINT_SUMMARY_STATS = True

# --- Hue (in [0,1]) for Okabe–Ito palette anchors ---
H_BLUE   = 203/360.0  # Okabe–Ito Blue (#0072B2)
H_ORANGE =  41/360.0  # Okabe–Ito Orange (#E69F00)
H_TEAL   = 164/360.0  # Okabe–Ito Green/Teal (#009E73)
H_RED    =  26/360.0  # Okabe–Ito Vermilion (#D55E00)
H_PURPLE = 327/360.0  # Okabe–Ito Reddish Purple (#CC79A7)
H_CYAN   = 195/360.0  # Tol-ish Cyan / Sky-ish

# --- 색상(HSL): (Hue, Saturation, Lightness) ---
# 정확히 #0072B2, #E69F00 톤을 반영하도록 S/L을 미세 조정함
ROBOT_HSL: Dict[str, Tuple[float, float, float]] = {
    "Q": (H_BLUE,   1.00, 0.35),  # RL-Trained (Okabe–Ito Blue, #0072B2)
    "H": (H_ORANGE, 1.00, 0.45),  # Human Control (Okabe–Ito Orange, #E69F00)
    "T": (0.00,     0.00, 0.65),  # Direct-to-Goal (중간 회색)
    "N": (0.00,     0.00, 0.55),  # No Robot (진한 회색)
}

# 컬러 사이클(미지정 로봇 대비)
COLOR_CYCLE_HUES = [0.00, 0.58, 0.33, 0.12, 0.75, 0.46, 0.90, 0.20]
DEFAULT_S = 0.70
DEFAULT_L = 0.48
GLOBAL_S_SCALE = 1.00
GLOBAL_L_SCALE = 1.00
GLOBAL_H_SHIFT = 0.00

# --- 그래프 보조 ---
SHOW_GRID  = True
GRID_STYLE = dict(linestyle="--", alpha=0.3)

# --- 로봇 라벨 & 표기 순서 ---
ROBOT_LABELS = {
    "R": "R (Go-and-Back Guide)",
    "T": "Direct-to-Goal",
    "Q": "RL-Trained",
    "N": "No Robot",
    "H": "Human Control",
}
ROBOT_ORDER: List[str] = ["Q", "H", "T", "N"]

# --- 산포점 옵션 ---
JITTER_POINTS  = False
JITTER_WIDTH   = 0.55

# --- 선 굵기 설정 ---
# evac100: 모델별 선 굵기 (없으면 DEFAULT 사용)
DEFAULT_EVAC_LINEWIDTH = 2.0
EVAC_LINEWIDTHS: Dict[str, float] = {
    "Q": 7.0,   # 강조 라인 더 두껍게
    "H": 7.0,
    "T": 1.4,
    "N": 1.4,
}

# episode 로그: “마커 OFF + 얇은 점/파선” 스타일
DEFAULT_EPISODE_LINEWIDTH = 2.0
EPISODE_LINEWIDTHS: Dict[str, float] = {
    "Q": 7.0,
    "H": 7.0,
    "T": 7.0,
    "N": 7.0,
}

# --- episode: 마커 대신 라인스타일 기반 구분 ---
EPISODE_USE_MARKERS = True      # 마커 끄기
EPISODE_LINESTYLE_FIXED = None   # 실선 고정 해제 → 로봇별 스타일 사용

# 로봇별 라인스타일(얇은 점선/파선)
EPISODE_LINESTYLES: Dict[str, str] = {
    "Q": "-",  # RL-Trained → 점선
    "H": "-",  # Human Control → 파점선
    "T": ":",   # Direct-to-Goal → 점점선
    "N": "-.",  # No Robot → 점선 (원하면 ":" 로 변경 가능)
}

EPISODE_MARKERS: Dict[str, str] = {
    "Q": "",   # 사용 안 함 (마커 OFF)
    "H": "",
    "T": "",
    "N": "",
}
EPISODE_MARKER_SIZE = 7.5
EPISODE_MARKER_EDGE_W = 1.2
EPISODE_MARKER_TARGET_COUNT = 60   # 각 곡선에 약 이 개수만큼 마커가 보이도록 간격 자동 조절

DISPLAY_MAP_MAP = {6: 1, 7: 2, 26: 3, 50: 4, 53: 5, 54: 6, 105:7, 108:8, 128:9}
INVERSE_DISPLAY_MAP = {v: k for k, v in DISPLAY_MAP_MAP.items()}

def display_id_of(map_id: int) -> int:
    # 폴더의 실제 map_id를 표기용 번호(1..6)로 바꿔서 보여줌
    return DISPLAY_MAP_MAP.get(map_id, map_id)

def lw_for_robot_evacu(rv: str) -> float:
    return EVAC_LINEWIDTHS.get(rv, DEFAULT_EVAC_LINEWIDTH)

def lw_for_robot_episode(rv: str) -> float:
    return EPISODE_LINEWIDTHS.get(rv, DEFAULT_EPISODE_LINEWIDTH)

def ls_for_robot_episode(rv: str) -> str:
    # 고정 스타일이 지정되어 있으면 그것을 사용, 아니면 로봇별 스타일
    if EPISODE_LINESTYLE_FIXED:
        return EPISODE_LINESTYLE_FIXED
    return EPISODE_LINESTYLES.get(rv, "--")

# =========================
# 내부 데이터 구조
# =========================
@dataclass
class TestRun:
    evacuation_100_time: float
    all_agents_life_time: float
    episode_log: List[float]

@dataclass
class RobotGroup:
    runs: List[TestRun] = field(default_factory=list)

@dataclass
class MapData:
    robots: Dict[str, RobotGroup] = field(default_factory=dict)

# =========================
# 로딩 유틸
# =========================
METRICS_FILE = "metrics.txt"
EPISODE_LOG_FILE = "episode_log.txt"

# =========================
# 맵 합본(가로 2열: Robot | Human) evac100 패널
# =========================
def _get_evac_values_for_map_robot(maps: Dict[int, MapData], real_map_id: int, robot_ver: str) -> List[float]:
    mdata = maps.get(real_map_id)
    if not mdata:
        return []
    group = mdata.robots.get(robot_ver)
    if not group or not group.runs:
        return []
    vals = [r.evacuation_100_time for r in group.runs]
    return [v/4 for v in vals if v < MAX_TIMESTEP]

def plot_evacuation_single_axes_pairs(
    maps: Dict[int, MapData],
    subset_disp_ids: List[int],
    out_dir: str,
    name_suffix: str,
    robot_codes: Tuple[str, str] = ("Q", "H"),
    pair_gap: float = 0.7,   # 같은 맵 내 (Q↔H) 간격
    map_gap: float = 1.0,   # 맵 사이 간격
    cat_width: float = 0.6,  # 카테고리 평균선/밴드 가로폭
    xtick_style: str = "map",   # "pair"(기본) | "map"
):
    """
    하나의 axes 안에: Map1-Q, Map1-H, [map_gap], Map2-Q, Map2-H, ...
    - pair_gap: 같은 맵(Q,H) 간격
    - map_gap : 맵 사이 간격
    - xtick_style:
        "pair" → Map1-Q, Map1-H ... (기존)
        "map"  → Map1, Map2 ... (중앙에 하나만)
    """
    real_map_ids = [INVERSE_DISPLAY_MAP.get(d, d) for d in subset_disp_ids]
    left_rv, right_rv = robot_codes
    left_label  = label_for_robot(left_rv)
    right_label = label_for_robot(right_rv)

    # === x 포지션 계산 ===
    x_positions = []  # [(disp_id, rv, x), ...]
    xtick_labels_pair = []
    base_x = 0.0
    for disp_id in subset_disp_ids:
        x_q = base_x
        x_h = base_x + pair_gap
        x_positions.extend([(disp_id, left_rv, x_q), (disp_id, right_rv, x_h)])
        xtick_labels_pair.extend([f"Map {disp_id}\n{left_label}", f"Map {disp_id}\n{right_label}"])
        base_x = x_h + map_gap

    # === y축 범위 ===
    all_vals = []
    for disp_id, real_id in zip(subset_disp_ids, real_map_ids):
        all_vals.extend(_get_evac_values_for_map_robot(maps, real_id, left_rv))
        all_vals.extend(_get_evac_values_for_map_robot(maps, real_id, right_rv))
    if YLIM_EVAC is not None:
        y_min, y_max = YLIM_EVAC
    else:
        if all_vals:
            y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
            if y_min == y_max:
                y_min -= 1.0; y_max += 1.0
            pad = 0.05 * (y_max - y_min)
            y_min -= pad; y_max += pad
        else:
            y_min, y_max = 0.0, 1.0

    # === 그림/axes ===
    fig_h = 6.0 if len(subset_disp_ids) <= 3 else 7.0
    fig_w = max(8.0, 1.0 * len(x_positions) + 0.8 * (len(subset_disp_ids)-1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # === 데이터 플롯 ===
    for (disp_id, rv, x_pos) in x_positions:
        real_id = INVERSE_DISPLAY_MAP.get(disp_id, disp_id)
        color = color_for_robot(rv, 0 if rv == left_rv else 1)
        vals = _get_evac_values_for_map_robot(maps, real_id, rv)

        half_w = cat_width / 2.0
        left, right = x_pos - half_w, x_pos + half_w

        if EVAC_SHOW_MEAN and vals:
            vmin, vmean, vmax = np.min(vals), np.mean(vals), np.max(vals)
            ax.hlines(vmean, left, right, color=color, linewidth=lw_for_robot_evacu(rv))
            if EVAC_SHOW_BAND:
                ax.fill_between([left, right], [vmin, vmin], [vmax, vmax], color=color, alpha=BAND_ALPHA)

        if EVAC_SHOW_POINTS and vals:
            if JITTER_POINTS:
                jitter = (np.random.rand(len(vals)) - 0.5) * (cat_width * JITTER_WIDTH)
                ax.scatter(x_pos + jitter, vals, color=color, alpha=0.45, s=18)
            else:
                ax.scatter([x_pos] * len(vals), vals, color=color, alpha=0.45, s=18)

        if not vals:
            ax.text(x_pos, (y_min + y_max) / 2.0, "No data", ha="center", va="center",
                    fontsize=10, alpha=0.6)

    # === 구분 띠 (optional) ===
    if len(subset_disp_ids) > 1:
        sep_alpha = 0.00
        for i in range(len(subset_disp_ids)-1):
            cur_q_x = x_positions[2*i][2]
            cur_h_x = x_positions[2*i+1][2]
            next_q_x = x_positions[2*(i+1)][2]
            mid_left  = (cur_q_x + cur_h_x)/2 + pair_gap/2
            mid_right = (cur_h_x + next_q_x)/2
            if mid_right > mid_left:
                ax.axvspan(mid_left, mid_right, color="#000000", alpha=sep_alpha, zorder=0)

    # === xticks 스타일 적용 ===
    if xtick_style.lower() == "map":
        map_centers = []
        for i in range(len(subset_disp_ids)):
            x_q = x_positions[2*i][2]
            x_h = x_positions[2*i+1][2]
            map_centers.append(((x_q + x_h) * 0.5, f"Map {subset_disp_ids[i]}"))
        xtick_pos = [c for c, _ in map_centers]
        xtick_labels = [lab for _, lab in map_centers]
    else:
        xtick_pos = [xp for (_, _, xp) in x_positions]
        xtick_labels = xtick_labels_pair

    # === 축/레이아웃 ===
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel(YLABEL_EVAC, fontsize=FONT_SIZES["axes"])
    ax.set_xticks(xtick_pos, xtick_labels, fontsize=FONT_SIZES["ticks"])
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.grid(**GRID_STYLE, axis="y")

    # === 범례 추가 ===
    legend_handles = []
    for rv, idx, label in [(left_rv, 0, left_label), (right_rv, 1, right_label)]:
        color = color_for_robot(rv, idx)
        legend_handles.append(
            Line2D([], [], color=color, linewidth=lw_for_robot_evacu(rv), label=label)
        )
    ax.legend(handles=legend_handles,
              fontsize=FONT_SIZES["legend"],
              title="",
              frameon=False,
              loc="upper left")

    subset_str = _format_id_ranges(subset_disp_ids)
    # ax.set_title(f"Maps {subset_str} – {TITLE_EVAC_NAME}", fontsize=FONT_SIZES["title"])
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"evac100_singleaxes_pairs_{name_suffix}.{SAVE_FORMAT}")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {path}")

def normalize_band_mode(mode):
    if mode is None:
        return None
    if isinstance(mode, str):
        s = mode.strip().lower()
        if s in ("none", "no", "off", ""):
            return None
        if s in ("minmax", "std", "p25_75"):
            return s
    if mode is False:
        return None
    return mode

BAND_MODE = normalize_band_mode(BAND_MODE)

def read_metrics(path: str) -> Tuple[Optional[float], Optional[float]]:
    evac, life = None, None
    if not os.path.exists(path):
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("evacuation_100_time="):
                evac = float(line.split("=", 1)[1])
            elif line.startswith("all_agents_life_time="):
                life = float(line.split("=", 1)[1])
    return evac, life

def read_episode_log(path: str) -> List[float]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    try:
        arr = ast.literal_eval(txt)
        return list(arr)
    except Exception:
        try:
            vals = [float(t.strip()) for t in txt.replace(",", "\n").splitlines() if t.strip()]
            return vals
        except Exception:
            return []

def load_data(root: str) -> Dict[int, MapData]:
    maps: Dict[int, MapData] = {}
    map_dir_pat = re.compile(r"^Result_(\d+)$")
    robot_dir_pat = re.compile(r"^Result_(\d+)_([A-Za-z0-9]+)$")
    test_dir_pat  = re.compile(r"^Result_(\d+)_([A-Za-z0-9]+)_(\d+)$")

    if not os.path.isdir(root):
        raise FileNotFoundError(f"root not found: {root}")

    for map_name in sorted(os.listdir(root)):
        m = map_dir_pat.match(map_name)
        if not m:
            continue
        map_id = int(m.group(1))
        maps.setdefault(map_id, MapData())
        map_dir = os.path.join(root, map_name)

        for rv_name in sorted(os.listdir(map_dir)):
            m2 = robot_dir_pat.match(rv_name)
            if not m2:
                continue
            map_id2 = int(m2.group(1))
            if map_id2 != map_id:
                continue
            robot_ver = m2.group(2)
            rv_dir = os.path.join(map_dir, rv_name)
            if not os.path.isdir(rv_dir):
                continue

            maps[map_id].robots.setdefault(robot_ver, RobotGroup())

            for td in sorted(os.listdir(rv_dir)):
                m3 = test_dir_pat.match(td)
                if not m3:
                    continue
                map_id3 = int(m3.group(1))
                robot_ver3 = m3.group(2)
                if map_id3 != map_id or robot_ver3 != robot_ver:
                    continue
                tdir = os.path.join(rv_dir, td)
                if not os.path.isdir(tdir):
                    continue

                evac, life = read_metrics(os.path.join(tdir, METRICS_FILE))
                if evac is None or life is None:
                    continue
                elog = read_episode_log(os.path.join(tdir, EPISODE_LOG_FILE))
                maps[map_id].robots[robot_ver].runs.append(
                    TestRun(evacuation_100_time=evac, all_agents_life_time=life, episode_log=elog)
                )
    return maps

# =========================
# 색상/라벨 유틸
# =========================
def hsl_to_rgb_hex(h: float, s: float, l: float) -> str:
    h = (h + GLOBAL_H_SHIFT) % 1.0
    s = max(0.0, min(1.0, s * GLOBAL_S_SCALE))
    l = max(0.0, min(1.0, l * GLOBAL_L_SCALE))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def color_for_robot(robot: str, idx: int) -> str:
    if robot in ROBOT_HSL:
        h, s, l = ROBOT_HSL[robot]
        return hsl_to_rgb_hex(h, s, l)
    h = COLOR_CYCLE_HUES[idx % len(COLOR_CYCLE_HUES)]
    return hsl_to_rgb_hex(h, DEFAULT_S, DEFAULT_L)

def label_for_robot(robot: str) -> str:
    return ROBOT_LABELS.get(robot, robot)

def ordered_robot_items(robots_dict: Dict[str, RobotGroup]) -> List[Tuple[str, RobotGroup]]:
    existing = set(robots_dict.keys())
    head = [r for r in ROBOT_ORDER if r in existing]
    tail = sorted(list(existing - set(head)))
    ordered_keys = head + tail
    return [(k, robots_dict[k]) for k in ordered_keys]

# =========================
# 통계/밴드 유틸
# =========================
def compute_min_mean_max(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return np.nan, np.nan, np.nan
    arr = np.array(values, dtype=float)
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

def pad_to_len(x: List[float], L: int, pad_value: float = 0.0) -> np.ndarray:
    out = np.full(L, pad_value, dtype=float)
    n = min(len(x), L)
    if n > 0:
        out[:n] = np.array(x[:n], dtype=float)
    return out

def per_step_stats_from_matrix(M: np.ndarray, mode: Optional[str]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    mean = M.mean(axis=0)
    if mode is None:
        return mean, None, None
    if mode == "std":
        std = M.std(axis=0, ddof=0)
        return mean, mean - std, mean + std
    elif mode == "p25_75":
        p25 = np.percentile(M, 25, axis=0)
        p75 = np.percentile(M, 75, axis=0)
        return mean, p25, p75
    else:  # "minmax"
        return mean, M.min(axis=0), M.max(axis=0)

# =========================
# 마커/색상 헬퍼 (episode용)
# =========================
def _hex_to_rgb01(hexstr: str) -> Tuple[float, float, float]:
    hexstr = hexstr.lstrip("#")
    r = int(hexstr[0:2], 16) / 255.0
    g = int(hexstr[2:4], 16) / 255.0
    b = int(hexstr[4:6], 16) / 255.0
    return (r, g, b)

def _darken_hex(hexstr: str, factor: float = 0.75) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb01(hexstr)
    return (max(0, r*factor), max(0, g*factor), max(0, b*factor))

def _lighten_hex(hexstr: str, factor: float = 0.7) -> Tuple[float, float, float]:
    """factor∈(0,1): 1에 가까울수록 더 흰색에 가까움"""
    r, g, b = _hex_to_rgb01(hexstr)
    return (1 - (1 - r)*factor, 1 - (1 - g)*factor, 1 - (1 - b)*factor)

def _with_alpha(rgb: Tuple[float, float, float], alpha: float) -> Tuple[float, float, float, float]:
    return (rgb[0], rgb[1], rgb[2], alpha)

def _marker_for_robot(rv: str, idx: int) -> str:
    cycle = ["o", "s", "^", "D", "v", "P", "*"]
    return EPISODE_MARKERS.get(rv, cycle[idx % len(cycle)])

def _compute_markevery(length: int, target: int = EPISODE_MARKER_TARGET_COUNT) -> int:
    if length <= 0:
        return 1
    step = int(np.ceil(length / float(max(1, target))))
    return max(1, step)

def _legend_proxy_line(marker: str, color: str, lw: float, msize: float, mew: float):
    """라인+마커가 함께 보이는 범례용 프록시 핸들"""
    return Line2D(
        [], [], color=color, linewidth=lw,
        marker=marker, markersize=msize*0.9,
        markerfacecolor="white",  # 속 흰색
        markeredgecolor=color, markeredgewidth=mew
    )

# =========================
# 맵/방법별 요약 통계
# =========================
def _safe_mean(values: List[float]) -> float:
    if not values:
        return np.nan
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_std(values: List[float], ddof: int = STD_DDOF) -> float:
    """값이 2개 미만이면 표본 표준편차를 계산할 수 없으므로 NaN 반환."""
    if len(values) <= ddof:
        return np.nan
    return float(np.std(np.asarray(values, dtype=float), ddof=ddof))


def compute_map_robot_summary(maps: Dict[int, MapData]) -> List[Dict[str, object]]:
    """각 실제 map_id와 robot method별 반복 실험 요약 통계를 계산한다."""
    rows: List[Dict[str, object]] = []

    for map_id, mdata in sorted(maps.items(), key=lambda x: x[0]):
        for robot_ver, group in ordered_robot_items(mdata.robots):
            evac_all = [float(r.evacuation_100_time) for r in group.runs]
            evac_valid = [v for v in evac_all if v < MAX_TIMESTEP]
            life_all = [float(r.all_agents_life_time) for r in group.runs]

            rows.append({
                "map_id": map_id,
                "display_map_id": display_id_of(map_id),
                "robot_method": robot_ver,
                "robot_label": label_for_robot(robot_ver),
                "n_runs": len(group.runs),
                "n_valid_evac": len(evac_valid),
                "n_timeout": len(evac_all) - len(evac_valid),
                # 원본 metric은 step 단위
                "evacuation_100_time_mean_steps": _safe_mean(evac_valid),
                "evacuation_100_time_std_steps": _safe_std(evac_valid),
                # 기존 그래프와 동일하게 4 step = 1 s 변환
                "evacuation_100_time_mean_s": _safe_mean(evac_valid) / 4.0 if evac_valid else np.nan,
                "evacuation_100_time_std_s": _safe_std(evac_valid) / 4.0 if len(evac_valid) > STD_DDOF else np.nan,
                "all_agents_life_time_mean": _safe_mean(life_all),
                "all_agents_life_time_std": _safe_std(life_all),
            })

    return rows


def _fmt_stat(value: object, digits: int = 3) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if np.isnan(x) else f"{x:.{digits}f}"


def print_map_robot_summary(rows: List[Dict[str, object]]):
    print("\n[Map × Robot Method Summary: sample std, ddof=1]")
    for row in rows:
        print(
            f"Map {row['display_map_id']} (real={row['map_id']}), "
            f"{row['robot_label']} [{row['robot_method']}]: "
            f"evacuation={_fmt_stat(row['evacuation_100_time_mean_s'])} "
            f"± {_fmt_stat(row['evacuation_100_time_std_s'])} s, "
            f"valid n={row['n_valid_evac']}/{row['n_runs']}, "
            f"timeouts={row['n_timeout']}; "
            f"all_agents_life_time={_fmt_stat(row['all_agents_life_time_mean'])} "
            f"± {_fmt_stat(row['all_agents_life_time_std'])}"
        )


def save_map_robot_summary_csv(rows: List[Dict[str, object]], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, SUMMARY_STATS_FILENAME)
    fieldnames = [
        "map_id",
        "display_map_id",
        "robot_method",
        "robot_label",
        "n_runs",
        "n_valid_evac",
        "n_timeout",
        "evacuation_100_time_mean_steps",
        "evacuation_100_time_std_steps",
        "evacuation_100_time_mean_s",
        "evacuation_100_time_std_s",
        "all_agents_life_time_mean",
        "all_agents_life_time_std",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Saved] {path}")
    return path

# =========================
# 공통 스타일
# =========================
def apply_axes_style(ax, ylim: Optional[Tuple[float, float]]):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.grid(**GRID_STYLE)
    if ylim is not None:
        ax.set_ylim(*ylim)

def title_for(map_id: int, plot_name: str) -> str:
    disp_id = display_id_of(map_id)  # ← 실제 id → 표시용 id 로 매핑
    return TITLE_MAP_FORMAT.format(map_id=disp_id, plot_name=plot_name)

# =========================
# 플롯: Evacuation 100
# =========================
def plot_evacuation_per_map(map_id: int, mdata: MapData, out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_EVAC)
    # ax.set_title(title_for(map_id, TITLE_EVAC_NAME), fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL_EVAC, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL_EVAC, fontsize=FONT_SIZES["axes"])

    # 정렬된 로봇 항목
    robot_items = ordered_robot_items(mdata.robots)

    # --- 로봇 필터 적용 (evac100만) ---
    if EVAC_ROBOT_ALLOW is not None:
        robot_items = [(rv, group) for rv, group in robot_items if rv in EVAC_ROBOT_ALLOW]

    robot_versions = [rv for rv, _ in robot_items]
    xs = np.arange(len(robot_versions))
    width = 0.6

    for i, (rv, group) in enumerate(robot_items):
        disp_label = label_for_robot(rv)
        color = color_for_robot(rv, i)

        vals_all = [r.evacuation_100_time for r in group.runs]
        if not vals_all:
            continue

        # timeout 여부 / 유효값
        has_timeout = any(v >= MAX_TIMESTEP for v in vals_all)
        vals_valid = [v for v in vals_all if v < MAX_TIMESTEP]

        x_pos = xs[i]
        left, right = x_pos - width/2, x_pos + width/2

        # 평균선/밴드 (timeout이 하나라도 있으면 평균/밴드 생략)
        if EVAC_SHOW_MEAN and (not has_timeout) and vals_valid:
            vmin, vmean, vmax = np.min(vals_valid), np.mean(vals_valid), np.max(vals_valid)
            ax.hlines(
                vmean, left, right,
                color=color,
                linewidth=lw_for_robot_evacu(rv),
                label=f"{disp_label} (mean)"
            )
            if EVAC_SHOW_BAND:
                ax.fill_between([left, right], [vmin, vmin], [vmax, vmax], color=color, alpha=BAND_ALPHA)

        # 점(개별 실험)
        if EVAC_SHOW_POINTS:
            point_vals = vals_valid if EVAC_POINT_MODE == "valid_only" else vals_all
            if point_vals:
                if JITTER_POINTS:
                    jitter = (np.random.rand(len(point_vals)) - 0.5) * (width * JITTER_WIDTH)
                    ax.scatter(x_pos + jitter, point_vals, color=color, alpha=0.45, s=18)
                else:
                    ax.scatter([x_pos] * len(point_vals), point_vals, color=color, alpha=0.45, s=18)

    ax.set_xticks(xs, [label_for_robot(rv) for rv in robot_versions])
    ax.legend(fontsize=FONT_SIZES["legend"])
    apply_axes_style(ax, YLIM_EVAC)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"map_{map_id}_evac100.{SAVE_FORMAT}")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {path}")

# =========================
# 플롯: Episode (얇은 점/파선, 마커 없음)
# =========================
def plot_episode_log_padded(map_id: int, mdata: MapData, out_dir: str, band_mode: Optional[str]):
    title_mode = band_mode if band_mode is not None else "no-band"
    fig, ax = plt.subplots(figsize=FIGSIZE_EPISOD)
    # ax.set_title(title_for(map_id, f"{TITLE_EPISODE_NAME}"), fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL_EPISODE, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL_EPISODE, fontsize=FONT_SIZES["axes"])

    # 범례용 프록시 핸들 (마커 사용 시에만 필요)
    legend_handles, legend_labels = [], []

    for i, (rv, group) in enumerate(ordered_robot_items(mdata.robots)):
        disp_label = label_for_robot(rv)
        color = color_for_robot(rv, i)
        runs = [r.episode_log for r in group.runs]
        if not runs:
            continue

        # x축 컷: MAX_TIMESTEP 초과 구간 제거
        runs_cut = [r[:MAX_TIMESTEP+1] for r in runs]
        L = max(len(r) for r in runs_cut)
        L = min(L, MAX_TIMESTEP + 1)

        M = np.stack([pad_to_len(r, L, pad_value=0.0) for r in runs_cut], axis=0)
        mean_curve, lower, upper = per_step_stats_from_matrix(M, mode=band_mode)

        x = np.arange(L)
        x = x/4

        # 1) 평균 곡선: 얇은 점/파선
        ax.plot(
            x, mean_curve,
            label=disp_label,
            color=color,
            linewidth=lw_for_robot_episode(rv),
            linestyle=ls_for_robot_episode(rv),
            zorder=2
        )

        # 2) 밴드: 외곽선 제거(깔끔하게 채움만)
        if lower is not None and upper is not None:
            ax.fill_between(
                x, lower, upper,
                color=color,
                alpha=BAND_ALPHA,
                edgecolor=None,
                linewidth=0.0,
                zorder=1
            )

        # 3) (마커는 사용하지 않음) — EPISODE_USE_MARKERS=False

    # 범례: 마커 프록시가 없으면 라벨 기반 자동 생성
    if SHOW_EPISODE_LEGEND:
        if legend_handles:
            ax.legend(legend_handles, legend_labels,
                      fontsize=FONT_SIZES["legend"], frameon=False)
        else:
            ax.legend(fontsize=FONT_SIZES["legend"], frameon=False)

    apply_axes_style(ax, YLIM_EPISODE)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"map_{map_id}_episode_{title_mode}.{SAVE_FORMAT}")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {path}")

def _installed_family_names() -> set:
    # 시스템에 등록된 폰트 패밀리 이름 집합
    return {f.name for f in fm.fontManager.ttflist}

def _filter_installed(cands: List[str]) -> List[str]:
    inst = _installed_family_names()
    return [c for c in cands if c in inst]

def _base_candidates_for_mode() -> List[str]:
    mode = (FONT_MODE or "").lower()
    if mode in ("serif", "serifed"):
        # 리눅스 기본이 잘 있는 순서 먼저 배치
        return ["DejaVu Serif", "Times New Roman", "Times"]
    if mode in ("sans", "sans-serif"):
        return ["DejaVu Sans", "Helvetica", "Arial"]
    if mode in ("mono", "monospace"):
        return ["DejaVu Sans Mono", "Consolas", "Menlo"]
    # custom: FONT_FAMILY가 '이름'일 수도, '.ttf 경로'일 수도 있음
    if FONT_FAMILY and FONT_FAMILY.lower().endswith((".ttf", ".otf")):
        # 파일 경로를 넣은 경우
        if os.path.isfile(FONT_FAMILY):
            try:
                fm.fontManager.addfont(FONT_FAMILY)
                prop = fm.FontProperties(fname=FONT_FAMILY)
                return [prop.get_name()]
            except Exception:
                pass
        # 등록 실패 시 아래에서 안전 폴백 사용
        return []
    # 파일 경로가 아니라 '이름'이면 그대로 후보에 넣기
    return [FONT_FAMILY] if FONT_FAMILY else []

def _resolve_font_list() -> List[str]:
    # 1) 모드 기반 기본 후보 + 사용자가 준 폴백을 합치고
    base = _base_candidates_for_mode()
    cands = base + list(FONT_FALLBACKS)
    # 2) 시스템에 실제로 설치된 것만 남긴다
    used = _filter_installed([c for c in cands if c])
    # 3) 하나도 없으면 모드별 안전 폴백(DejaVu 계열) 강제
    mode = (FONT_MODE or "").lower()
    if not used:
        if mode in ("serif", "serifed", "custom"):
            used = ["DejaVu Serif"]
        elif mode in ("mono", "monospace"):
            used = ["DejaVu Sans Mono"]
        else:
            used = ["DejaVu Sans"]
    # 중복 제거(순서 보존)
    seen, out = set(), []
    for name in used:
        if name not in seen:
            out.append(name); seen.add(name)
    return out

def _math_fontset_for_mode() -> str:
    mode = (FONT_MODE or "").lower()
    if mode in ("serif", "serifed", "custom"):
        return "dejavuserif"  # 필요시 "cm"로 교체 가능
    return "dejavusans"

def apply_font_settings():
    families = _resolve_font_list()

    # 가족군 우선순위를 설정
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

    # 크기
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]

    # 수식 렌더링 셋
    mpl.rcParams["mathtext.fontset"] = _math_fontset_for_mode()
    mpl.rcParams["axes.unicode_minus"] = False

    # 라인 렌더링 품질 소폭 개선 (선택)
    mpl.rcParams["lines.antialiased"] = True
    mpl.rcParams["lines.solid_joinstyle"] = "round"
    mpl.rcParams["lines.solid_capstyle"]  = "round"

    # 폰트 캐시 꼬임 방지(필요시 1회)
    try:
        fm._rebuild()
    except Exception:
        pass

def _format_id_ranges(ids: List[int]) -> str:
    """[1,2,3,4,5,6] -> '1~6', [1,2,3] -> '1~3', [4,5,6] -> '4~6'
       비연속 구간이 섞이면 '1~3, 5, 7~9' 이런 식으로 표기"""
    if not ids:
        return ""
    s = sorted(ids)
    ranges = []
    start = prev = s[0]
    for x in s[1:]:
        if x == prev + 1:      # 연속이면 이어가기
            prev = x
            continue
        # 구간 닫기
        ranges.append(f"{start}" if start == prev else f"{start}~{prev}")
        start = prev = x
    # 마지막 구간 닫기
    ranges.append(f"{start}" if start == prev else f"{start}~{prev}")
    return ", ".join(ranges)

# =========================
# 메인
# =========================
def main():
    apply_font_settings()
    maps = load_data(ROOT_DIR)
    if not maps:
        print(f"[WARN] No maps found under: {ROOT_DIR}")
        return

    # 각 맵 × 로봇 방법별 평균 및 표준편차 계산
    summary_rows = compute_map_robot_summary(maps)
    if PRINT_SUMMARY_STATS:
        print_map_robot_summary(summary_rows)
    if SAVE_SUMMARY_STATS:
        save_map_robot_summary_csv(summary_rows, OUT_DIR)

    for map_id, mdata in sorted(maps.items(), key=lambda x: x[0]):
        plot_evacuation_per_map(map_id, mdata, OUT_DIR)
        plot_episode_log_padded(map_id, mdata, OUT_DIR, band_mode=BAND_MODE)
#    print(f("[DONE] Plots saved to: {abs_path}") if (abs_path := os.path.abspath(OUT_DIR)) else "[DONE]")

    # ===== 추가: 단일 축에 (맵×2카테고리) 모두 배치한 버전 3장 =====
    plot_evacuation_single_axes_pairs(maps, [1, 2, 3], OUT_DIR, name_suffix="maps_1_2_3")
    plot_evacuation_single_axes_pairs(maps, [4, 5, 6], OUT_DIR, name_suffix="maps_4_5_6")
    plot_evacuation_single_axes_pairs(maps, [102, 113, 114, 115, 116], OUT_DIR, name_suffix="unseen_maps_1_6")

if __name__ == "__main__":
    main()