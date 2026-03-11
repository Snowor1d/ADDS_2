#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reward Plotter for saved evaluation episodes

Expected directory structure (example):
~/Result_data_0304/vs_human/
    Result_105/
        Result_105_Q/
            reward_A.txt
            reward_B.txt
            reward_penalty.txt
            reward_fixed.txt
            reward_based_farthest_agent_distance.txt
            Result_105_Q_0/
                reward_A.txt
                reward_B.txt
                ...
            Result_105_Q_1/
                ...
            Result_105_Q_2/
                ...

What this script does:
1) For each slot directory (episode), create one graph:
   - reward_plot.png
2) For each map_robot_dir (average reward files), create one graph:
   - reward_plot_avg.png
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm


# =========================================================
# USER CONFIG
# =========================================================
HOME_DIR = os.path.expanduser("~")

# 네 실험 결과 루트
ROOT_DIR = os.path.join(HOME_DIR, "Result_data_0309", "test")
# 저장 주기: reward를 몇 timestep마다 기록했는가
ACTION_SCALE = 4

# map/robot 설정과 맞추고 싶으면 적절히 수정
ROBOT_VERSION = "Q"

# episode 폴더 이름 패턴: Result_{map_id}_{robot_ver}_{slot}
EPISODE_DIR_REGEX = re.compile(r"^Result_(\d+)_([A-Za-z]+)_(\d+)$")

# 평균 reward가 들어있는 map_robot_dir 이름 패턴: Result_{map_id}_{robot_ver}
MAP_ROBOT_DIR_REGEX = re.compile(r"^Result_(\d+)_([A-Za-z]+)$")

# reward 파일 이름 패턴
REWARD_FILE_REGEX = re.compile(r"^reward_[A-Za-z0-9_]+\.txt$")

# 출력 파일명
EPISODE_PLOT_NAME = "reward_plot.png"
AVG_PLOT_NAME = "reward_plot_avg.png"

# Figure
FIGSIZE = (14, 8)
SAVE_DPI = 300

# Font
FONT_MODE = "serif"   # "serif" | "sans" | "mono"
FONT_SIZES = {
    "title": 22,
    "axes": 35,
    "ticks": 30,
    "legend": 13,
}
FONT_FAMILY = "DejaVu Serif"

# Labels / title
EPISODE_TITLE_TEMPLATE = ""
AVG_TITLE_TEMPLATE = "Average Reward Terms over Time (Map {map_id})"
XLABEL = "Time"
YLABEL = "Reward"

# Grid
SHOW_GRID = True
GRID_STYLE = dict(color="#BBBBBB", linestyle="--", linewidth=0.8, alpha=0.7)

# Smoothing
SMOOTH_ENABLE = False
SMOOTH_KIND = "ma"   # "ma" | "ema"
SMOOTH_STRENGTH = 0.35

MA_WINDOW_MIN = 3
MA_WINDOW_MAX = 51
EMA_ALPHA_MIN = 0.05
EMA_ALPHA_MAX = 0.50

# Y clipping
CLIP_Y_MIN: Optional[float] = None
CLIP_Y_MAX: Optional[float] = None

# Legend
LEGEND_LOC = "best"

# Raw trace overlay
SHOW_RAW_TRACE = False
RAW_TRACE_ALPHA = 0.20
RAW_TRACE_LINEWIDTH = 1.0

# Line
DEFAULT_LINEWIDTH = 5

# 이름 보기 좋게 바꾸기
DISPLAY_NAME_ALIASES: Dict[str, str] = {
    "reward_A": "Reward A",
    "reward_B": "Reward B",
    "reward_penalty": "Penalty",
    "reward_fixed": "Fixed Reward",
    "reward_based_farthest_agent_distance": "Farthest-Agent Distance",
    "reward_total": "Total Reward",
}

# 특정 순서로 legend를 보여주고 싶으면
LEGEND_ORDER = [
    "Reward A",
    "Reward B",
    "Farthest-Agent Distance",
    "Penalty",
    "Fixed Reward",
    "Total Reward",
]

# 색상 커스텀
COLOR_MAP: Dict[str, str] = {
    "Reward A": "#1f77b4",
    "Reward B": "#ff7f0e",
    "Farthest-Agent Distance": "#2ca02c",
    "Penalty": "#d62728",
    "Fixed Reward": "#9467bd",
    "Total Reward": "#111111",
}


# =========================================================
# Font utils
# =========================================================
def apply_font_settings():
    mode = FONT_MODE.lower()
    if mode == "serif":
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.serif"] = [FONT_FAMILY, "Times New Roman", "DejaVu Serif"]
        mpl.rcParams["mathtext.fontset"] = "dejavuserif"
    elif mode == "mono":
        mpl.rcParams["font.family"] = "monospace"
        mpl.rcParams["font.monospace"] = [FONT_FAMILY, "DejaVu Sans Mono"]
        mpl.rcParams["mathtext.fontset"] = "dejavusans"
    else:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = [FONT_FAMILY, "Arial", "DejaVu Sans"]
        mpl.rcParams["mathtext.fontset"] = "dejavusans"

    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]
    mpl.rcParams["axes.unicode_minus"] = False

    try:
        fm._rebuild()
    except Exception:
        pass


# =========================================================
# Data structures
# =========================================================
@dataclass
class RewardSeries:
    name: str
    x: np.ndarray
    y: np.ndarray
    y_smooth: Optional[np.ndarray] = None


# =========================================================
# IO utils
# =========================================================
def read_series_txt(path: str) -> List[float]:
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except Exception:
                pass
    return vals


def resolve_display_name(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return DISPLAY_NAME_ALIASES.get(base, base)


def load_reward_series_from_dir(target_dir: str) -> List[RewardSeries]:
    if not os.path.isdir(target_dir):
        return []

    series_list: List[RewardSeries] = []

    files = sorted(os.listdir(target_dir))
    for fn in files:
        if not REWARD_FILE_REGEX.fullmatch(fn):
            continue

        path = os.path.join(target_dir, fn)
        if not os.path.isfile(path):
            continue

        y_list = read_series_txt(path)
        if not y_list:
            continue

        # reward는 ACTION_SCALE step마다 기록되었으므로
        # x축을 실제 timestep으로 표시
        x = (np.arange(1, len(y_list) + 1, dtype=float) * ACTION_SCALE)/2
        y = np.asarray(y_list, dtype=float)

        name = resolve_display_name(fn)
        series_list.append(RewardSeries(name=name, x=x, y=y))

    return series_list


# =========================================================
# Smoothing
# =========================================================
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
        out[i] = alpha * y[i] + (1.0 - alpha) * out[i - 1]
    return out


def apply_smoothing(series: RewardSeries) -> RewardSeries:
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


# =========================================================
# Plot utils
# =========================================================
def get_series_color(name: str, idx: int) -> str:
    if name in COLOR_MAP:
        return COLOR_MAP[name]
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return default_cycle[idx % len(default_cycle)]


def apply_axes_style(ax):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.set_axisbelow(False)
        ax.grid(**GRID_STYLE)

    if CLIP_Y_MIN is not None or CLIP_Y_MAX is not None:
        ymin, ymax = ax.get_ylim()
        if CLIP_Y_MIN is not None:
            ymin = CLIP_Y_MIN
        if CLIP_Y_MAX is not None:
            ymax = CLIP_Y_MAX
        ax.set_ylim(ymin, ymax)


def order_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    last_by_label = {}
    for h, lab in zip(handles, labels):
        if lab and lab != "_nolegend_":
            last_by_label[lab] = h

    ordered_handles = []
    ordered_labels = []
    used = set()

    for lab in LEGEND_ORDER:
        if lab in last_by_label:
            ordered_handles.append(last_by_label[lab])
            ordered_labels.append(lab)
            used.add(lab)

    for h, lab in zip(handles, labels):
        if lab and lab != "_nolegend_" and lab not in used:
            ordered_handles.append(h)
            ordered_labels.append(lab)
            used.add(lab)

    # if ordered_handles:
    #     ax.legend(ordered_handles, ordered_labels, loc=LEGEND_LOC, frameon=False)


def plot_reward_series(
    series_list: List[RewardSeries],
    save_path: str,
    title: str,
):
    if not series_list:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, s in enumerate(series_list):
        s = apply_smoothing(s)
        ydraw = s.y_smooth if s.y_smooth is not None else s.y
        xarr = np.asarray(s.x, dtype=float)
        yarr = np.asarray(ydraw, dtype=float)

        if CLIP_Y_MIN is not None:
            yarr = np.maximum(yarr, CLIP_Y_MIN)
        if CLIP_Y_MAX is not None:
            yarr = np.minimum(yarr, CLIP_Y_MAX)

        color = get_series_color(s.name, i)

        if SHOW_RAW_TRACE and s.y_smooth is not None:
            ax.plot(
                s.x, s.y,
                color=color,
                linewidth=RAW_TRACE_LINEWIDTH,
                alpha=RAW_TRACE_ALPHA,
                label="_nolegend_",
            )

        ax.plot(
            xarr,
            yarr,
            color=color,
            linewidth=DEFAULT_LINEWIDTH,
            label=s.name,
        )

    ax.set_xlabel(XLABEL, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL, fontsize=FONT_SIZES["axes"])
    ax.set_title(title, fontsize=FONT_SIZES["title"])

    apply_axes_style(ax)
    order_legend(ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {save_path}")


# =========================================================
# Directory traversal
# =========================================================
def find_map_robot_dirs(root_dir: str) -> List[Tuple[int, str, str]]:
    """
    Returns:
        [(map_id, robot_ver, map_robot_dir), ...]
    """
    found = []

    if not os.path.isdir(root_dir):
        return found

    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue

        # e.g. Result_105
        if not re.fullmatch(r"Result_\d+", name):
            continue

        for sub in sorted(os.listdir(path)):
            subpath = os.path.join(path, sub)
            if not os.path.isdir(subpath):
                continue

            m = MAP_ROBOT_DIR_REGEX.fullmatch(sub)
            if not m:
                continue

            map_id = int(m.group(1))
            robot_ver = m.group(2)
            found.append((map_id, robot_ver, subpath))

    return found


def find_episode_dirs(map_robot_dir: str) -> List[Tuple[int, int, str]]:
    """
    Returns:
        [(map_id, slot, episode_dir), ...]
    """
    found = []
    for name in sorted(os.listdir(map_robot_dir)):
        path = os.path.join(map_robot_dir, name)
        if not os.path.isdir(path):
            continue

        m = EPISODE_DIR_REGEX.fullmatch(name)
        if not m:
            continue

        map_id = int(m.group(1))
        robot_ver = m.group(2)
        slot = int(m.group(3))

        if robot_ver != ROBOT_VERSION:
            continue

        found.append((map_id, slot, path))

    return found


# =========================================================
# Main
# =========================================================
def main():
    apply_font_settings()

    if not os.path.isdir(ROOT_DIR):
        print(f"[ERROR] ROOT_DIR not found: {ROOT_DIR}")
        return

    map_robot_dirs = find_map_robot_dirs(ROOT_DIR)
    if not map_robot_dirs:
        print(f"[WARN] No map_robot_dir found under: {ROOT_DIR}")
        return

    for map_id, robot_ver, map_robot_dir in map_robot_dirs:
        if robot_ver != ROBOT_VERSION:
            continue

        # 1) 평균 reward 그래프
        avg_series = load_reward_series_from_dir(map_robot_dir)
        if avg_series:
            avg_save_path = os.path.join(map_robot_dir, AVG_PLOT_NAME)
            avg_title = AVG_TITLE_TEMPLATE.format(map_id=map_id)
            plot_reward_series(avg_series, avg_save_path, avg_title)

        # 2) 각 episode(slot) reward 그래프
        episode_dirs = find_episode_dirs(map_robot_dir)
        for ep_map_id, slot, episode_dir in episode_dirs:
            ep_series = load_reward_series_from_dir(episode_dir)
            if not ep_series:
                continue

            ep_save_path = os.path.join(episode_dir, EPISODE_PLOT_NAME)
            ep_title = EPISODE_TITLE_TEMPLATE.format(map_id=ep_map_id, slot=slot)
            plot_reward_series(ep_series, ep_save_path, ep_title)

    print("[DONE] Reward plotting finished.")


if __name__ == "__main__":
    main()