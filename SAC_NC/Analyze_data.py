#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import colorsys

# =========================
# 전역 파라미터
# =========================
ROOT_DIR = os.path.expanduser("~/Result_data_Experiment2")
OUT_DIR  = ROOT_DIR

# --- 그림/폰트 ---
FIGSIZE_EVAC   = (8, 5)
FIGSIZE_EPISOD = (8, 5)
FONT_SIZES = {"title": 16, "axes": 13, "ticks": 11, "legend": 11}

# --- 제목/축 레이블 ---
TITLE_MAP_FORMAT   = "Map {map_id} - {plot_name}"
TITLE_EVAC_NAME    = "Time to Evacuate overall Crowds"
TITLE_EPISODE_NAME = "Remained Agents per Step"
XLABEL_EVAC        = "Robot Version"
YLABEL_EVAC        = "Timestep"
XLABEL_EPISODE     = "Step"
YLABEL_EPISODE     = "Remained evacuaee"

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
EVAC_ROBOT_ALLOW: Optional[List[str]] = ["H", "Q"]    # None(전체) | ["R", "T"] 등

# --- y축 범위 (None이면 자동) ---
YLIM_EVAC: Optional[Tuple[float, float]]    = None
YLIM_EPISODE: Optional[Tuple[float, float]] = None

# --- timeout 기준 ---
MAX_TIMESTEP = 1500  # 이 값 이상이면 timeout

H_BLUE   = 202/360.0  # Okabe–Ito Blue
H_ORANGE =  41/360.0  # Okabe–Ito Orange
H_TEAL   = 164/360.0  # Okabe–Ito Green/Teal
H_RED    =  26/360.0  # Okabe–Ito Vermilion (Red-Orange)
H_PURPLE = 327/360.0  # Okabe–Ito Reddish Purple
H_CYAN   = 195/360.0  # Tol-ish Cyan / Sky-ish

# --- 색상(HSL) ---
ROBOT_HSL: Dict[str, Tuple[float, float, float]] = {
    "Q": (H_BLUE, 1.00, 0.35),  # 강조(블루, 기본)
    "H": (H_ORANGE,   1.00, 0.45),  # 회색(연)
    "T": (0.00,   0.00, 0.65),  # 회색(중)
    "N": (0.00,   0.00, 0.55),  # 회색(진)

    # --- 대안 강조색 (Q 라인을 다른 색으로 바꾸고 싶을 때 아래 중 하나로 교체) ---
    # "Q": (H_ORANGE, 1.00, 0.45),  # 오렌지
    # "Q": (H_TEAL,   1.00, 0.31),  # 초록/틸
    # "Q": (H_RED,    1.00, 0.42),  # 빨강(버밀리온)
    # "Q": (H_PURPLE, 0.45, 0.64),  # 보라(채도 낮춤)
    # "Q": (H_CYAN,   0.80, 0.66),  # 시안/스카이블루(연함)
    # "Q": (0.00,     0.00, 0.00),  # 블랙(흑백 인쇄용)
}

# --- 여러 모델을 컬러로 구분해야 할 때(회색 대신 컬러 세트 예시) ---
# ROBOT_HSL = {
#     "Q": (H_BLUE,   1.00, 0.35),
#     "H": (H_ORANGE, 1.00, 0.45),
#     "T": (H_TEAL,   1.00, 0.31),
#     "N": (H_PURPLE, 0.45, 0.64),
# }
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
    # "Human": "Human (Joystick Baseline)",
}
ROBOT_ORDER: List[str] = ["Q", "H", "T", "N"]

# --- 산포점 옵션 ---
JITTER_POINTS  = False
JITTER_WIDTH   = 0.55

# --- 선 굵기 설정 ---
# evac100: 모델별 선 굵기 (없으면 DEFAULT 사용)
DEFAULT_EVAC_LINEWIDTH = 2.0
EVAC_LINEWIDTHS: Dict[str, float] = {
    "Q": 3.0,   # 강조 라인 더 두껍게
    "H": 3.0,
    "T": 1.4,
    "N": 1.4,
}

# episode 로그: 모델별 선 굵기
DEFAULT_EPISODE_LINEWIDTH = 2.2
EPISODE_LINEWIDTHS: Dict[str, float] = {
    "Q": 3,   # 필요시 조정
    "H": 2.5,
    "T": 2.5,
    "N": 2.5,
}

def lw_for_robot_evacu(rv: str) -> float:
    return EVAC_LINEWIDTHS.get(rv, DEFAULT_EVAC_LINEWIDTH)

def lw_for_robot_episode(rv: str) -> float:
    return EPISODE_LINEWIDTHS.get(rv, DEFAULT_EPISODE_LINEWIDTH)

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
# 공통 스타일
# =========================
def apply_axes_style(ax, ylim: Optional[Tuple[float, float]]):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.grid(**GRID_STYLE)
    if ylim is not None:
        ax.set_ylim(*ylim)

def title_for(map_id: int, plot_name: str) -> str:
    return TITLE_MAP_FORMAT.format(map_id=map_id, plot_name=plot_name)

# =========================
# 플롯: Evacuation 100
# =========================
def plot_evacuation_per_map(map_id: int, mdata: MapData, out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_EVAC)
    ax.set_title(title_for(map_id, TITLE_EVAC_NAME), fontsize=FONT_SIZES["title"])
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
                linewidth=lw_for_robot_evacu(rv),     # ✅ 모델별 굵기 적용
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
# 플롯: Episode (0-패딩 + 밴드)
# =========================
def plot_episode_log_padded(map_id: int, mdata: MapData, out_dir: str, band_mode: Optional[str]):
    title_mode = band_mode if band_mode is not None else "no-band"
    fig, ax = plt.subplots(figsize=FIGSIZE_EPISOD)
    ax.set_title(title_for(map_id, f"{TITLE_EPISODE_NAME} ({title_mode})"), fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL_EPISODE, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL_EPISODE, fontsize=FONT_SIZES["axes"])

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
        ax.plot(x, mean_curve, label=disp_label, color=color, linewidth=lw_for_robot_episode(rv))
        if lower is not None and upper is not None:
            ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA)

    ax.legend(fontsize=FONT_SIZES["legend"])
    apply_axes_style(ax, YLIM_EPISODE)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"map_{map_id}_episode_{title_mode}.{SAVE_FORMAT}")
    fig.savefig(path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {path}")

# =========================
# 메인
# =========================
def main():
    maps = load_data(ROOT_DIR)
    if not maps:
        print(f"[WARN] No maps found under: {ROOT_DIR}")
        return
    for map_id, mdata in sorted(maps.items(), key=lambda x: x[0]):
        plot_evacuation_per_map(map_id, mdata, OUT_DIR)
        plot_episode_log_padded(map_id, mdata, OUT_DIR, band_mode=BAND_MODE)
    print(f"[DONE] Plots saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
