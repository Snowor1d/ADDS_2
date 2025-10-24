#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from typing import List, Tuple, Optional

# =========================
# USER CONFIG
# =========================
RUN_DIR_NAME = "Drawing_rewards_242526"
REWARD_FILE = "total_reward.txt"
EVAC_FILE   = "evacuation_100.txt"

# =========================
# FIGURE / SAVE OPTIONS
# =========================
FIGSIZE_COMBINED = (16, 10)
FIGSIZE_SCALE    = 1.0
USE_TIGHT_LAYOUT = True
BBOX_TIGHT       = False
PAD_INCHES       = 0.05
SAVE_FORMAT = "png"
SAVE_DPI    = 300
SAVE_TRANSPARENT = False

# =========================
# FONTS / STYLE
# =========================
FONT_SIZES  = {"title": 30, "axes": 50, "ticks": 40, "legend": 40}
FONT_FAMILY = "serif"
SHOW_GRID   = True
GRID_STYLE  = dict(linestyle="--", alpha=0.3)

TITLE_COMBINED = ""
XLABEL         = "Episode"
YLABEL_LEFT    = "Total Reward"
YLABEL_RIGHT   = "Timestep"

LEGEND_A_SMO = "Reward"
LEGEND_B_SMO = "Evac-100%"

# =========================
# COLORS (명도 대비로 구분, 실선 유지)
# =========================
# Okabe–Ito palette: 흑백에서도 구분되는 색상 조합
COLOR_A_SMOOTH  = "#0072B2"   # 어두운 파랑
COLOR_B_SMOOTH  = "#E69F00"   # 밝은 오렌지
COLOR_A_SHADE   = COLOR_A_SMOOTH
COLOR_B_SHADE   = COLOR_B_SMOOTH
LINESTYLE_A = LINESTYLE_B = "-"  # 실선

# 스무딩 및 밴드
SMOOTH_MODE  = "ma"
MA_WINDOW    = 21
SHADE_MODE   = "std"
SHADE_ALPHA  = 0.12
SMOOTH_LINEWIDTH = 2.2

# =========================
# UTILS
# =========================
def apply_font_settings():
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"]  = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]

def expand_run_dir(name: str) -> str:
    return os.path.expanduser(os.path.join("~", name))

def read_txt_series(path: str) -> list[float]:
    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return []
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    vals.append(float(s))
                except:
                    pass
    return vals

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) == 0:
        return y
    window = max(1, min(window, len(y)))
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window//2, window-1-window//2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")

def rolling_std(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) == 0:
        return y
    window = max(2, min(window, len(y)))
    pad = window // 2
    y_pad = np.pad(y, (pad, window - 1 - pad), mode="edge")
    csum  = np.cumsum(y_pad, dtype=float)
    csum2 = np.cumsum(y_pad**2, dtype=float)
    csum  = np.concatenate([[0.0], csum])
    csum2 = np.concatenate([[0.0], csum2])
    n = window
    sums  = csum[n:]  - csum[:-n]
    sums2 = csum2[n:] - csum2[:-n]
    means = sums / n
    vars_ = (sums2 / n) - (means**2)
    np.maximum(vars_, 0.0, out=vars_)
    return np.sqrt(vars_)

def smooth_series(y: np.ndarray):
    if len(y) == 0:
        return y, None
    sm = moving_average(y, MA_WINDOW)
    if SHADE_MODE == "std":
        band = rolling_std(y, MA_WINDOW)
    else:
        band = None
    return sm, band

def _scaled_size(size: Tuple[float, float]) -> Tuple[float, float]:
    return (size[0] * FIGSIZE_SCALE, size[1] * FIGSIZE_SCALE)

# =========================
# Plot function
# =========================
def plot_dual_curve(y_left, y_right, save_path):
    y_left  = np.asarray(y_left, dtype=float)
    y_right = np.asarray(y_right, dtype=float)
    L = min(len(y_left), len(y_right))
    if L == 0:
        print("[WARN] nothing to plot.")
        return
    x = np.arange(1, L+1)
    yl_s, yl_band = smooth_series(y_left[:L])
    yr_s, yr_band = smooth_series(y_right[:L])

    fig, axL = plt.subplots(figsize=_scaled_size(FIGSIZE_COMBINED))
    axL.set_xlabel(XLABEL, fontsize=FONT_SIZES["axes"])
    axL.set_ylabel(YLABEL_LEFT, fontsize=FONT_SIZES["axes"])
    axL.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

    axR = axL.twinx()
    axR.set_ylabel(YLABEL_RIGHT, fontsize=FONT_SIZES["axes"])
    axR.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

    # Reward (좌측)
    axL.plot(x, yl_s, color=COLOR_A_SMOOTH, linewidth=SMOOTH_LINEWIDTH,
             linestyle=LINESTYLE_A, label=LEGEND_A_SMO)
    if yl_band is not None:
        axL.fill_between(x, yl_s - yl_band, yl_s + yl_band,
                         color=COLOR_A_SHADE, alpha=SHADE_ALPHA)

    # Evac (우측)
    axR.plot(x, yr_s, color=COLOR_B_SMOOTH, linewidth=SMOOTH_LINEWIDTH,
             linestyle=LINESTYLE_B, label=LEGEND_B_SMO)
    if yr_band is not None:
        axR.fill_between(x, yr_s - yr_band, yr_s + yr_band,
                         color=COLOR_B_SHADE, alpha=SHADE_ALPHA)

    if SHOW_GRID:
        axL.grid(**GRID_STYLE)

    # --- 범례: 그래프 내부 하단 중앙, 가로 한 줄 ---
    legend_handles = [
        Line2D([0], [0], color=COLOR_A_SMOOTH, lw=4, label=LEGEND_A_SMO),
        Line2D([0], [0], color=COLOR_B_SMOOTH, lw=4, label=LEGEND_B_SMO),
    ]
    leg = axL.legend(
        handles=legend_handles,
        loc="lower center",  # 그래프 안쪽 하단 중앙
        ncol=2,              # 가로로 배치
        frameon=True,
        framealpha=0.9,      # 살짝 투명한 배경
        handlelength=2.8,
        borderpad=0.5,
    )

    if USE_TIGHT_LAYOUT:
        fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    print(f"[Saved] {save_path}")

# =========================
# MAIN
# =========================
def main():
    apply_font_settings()
    run_dir = expand_run_dir(RUN_DIR_NAME)
    if not os.path.isdir(run_dir):
        print(f"[ERROR] directory not found: {run_dir}")
        return

    reward_path = os.path.join(run_dir, REWARD_FILE)
    evac_path   = os.path.join(run_dir, EVAC_FILE)
    rewards = read_txt_series(reward_path)[:15000]
    evacs   = read_txt_series(evac_path)[:15000]

    if not rewards or not evacs:
        print("[STOP] need both files.")
        return

    out_combined = os.path.join(run_dir, f"curve_reward+evac_sm_{SMOOTH_MODE}.{SAVE_FORMAT}")
    plot_dual_curve(rewards, evacs, out_combined)
    print("[DONE]")

if __name__ == "__main__":
    main()
