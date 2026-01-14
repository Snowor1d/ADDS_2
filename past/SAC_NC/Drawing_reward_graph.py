#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Tuple, Optional

# =========================
# USER CONFIG
# =========================
# 홈 디렉토리 아래의 폴더 이름만 지정하세요. (~/<이_폴더> 안에서 txt를 읽음)
RUN_DIR_NAME = "Drawing_rewards_678"   # ← 변경

# 파일명 (폴더 안에 다음 이름의 파일이 있어야 함)
REWARD_FILE = "total_reward.txt"
EVAC_FILE   = "evacuation_100.txt"

# =========================
# FIGURE / SAVE OPTIONS
# =========================
# ▶ 그림 크기(인치). 그래프별로 따로 지정 가능
FIGSIZE_REWARD = (15, 10)   # (width, height)
FIGSIZE_EVAC   = (15, 10)

# ▶ 전역 스케일(배율). 1.2면 가로/세로 모두 20% 커짐
FIGSIZE_SCALE = 1.0

# ▶ 레이아웃/여백
USE_TIGHT_LAYOUT = True          # True면 tight_layout 사용
BBOX_TIGHT       = False         # True면 bbox_inches='tight'로 저장(여백 최소화)
PAD_INCHES       = 0.05          # BBOX_TIGHT가 True일 때 여백(inches)

# ▶ 저장 옵션
SAVE_FORMAT = "png"
SAVE_DPI    = 220
SAVE_TRANSPARENT = False         # True면 투명 배경으로 저장

# =========================
# FONTS / STYLE
# =========================
FONT_SIZES = {"title": 30, "axes": 35, "ticks": 35, "legend": 35}
FONT_FAMILY = "serif"   # ← 여기서 로마자(serif) 선택
SHOW_GRID  = True
GRID_STYLE = dict(linestyle="--", alpha=0.3)

# 라벨/제목/범례
TITLE_REWARD = ""
TITLE_EVAC   = ""
XLABEL       = "Episode"
YLABEL_REW   = "Total Reward"
YLABEL_EVAC  = "Timestep"
LEGEND_TITLE = ""
LEGEND_RAW   = "Raw"
LEGEND_SMO   = ""

# 원시 데이터 표시 방식: "points" | "line" | "none"
DRAW_RAW_SERIES = "none"   # ← 점/선 모두 끄고, 스무딩+음영만 보이게
RAW_POINT_SIZE  = 10       # (points일 때만 사용)
RAW_LINE_ALPHA  = 0.35     # (line일 때만 사용)
COLOR_RAW       = "#9aa0a6"

# 색상
COLOR_RAW    = "#9aa0a6"   # 연회색 (원 데이터 점/선)
COLOR_SMOOTH = "#1f77b4"   # 파랑 (스무딩 선)
COLOR_SHADE  = "#1f77b4"   # 파랑 톤 음영

# 스무딩 설정
# mode: "ma"(이동평균) | "ema"(지수이동평균) | None
SMOOTH_MODE  = "ma"
MA_WINDOW    = 21           # 이동평균 윈도 크기(홀수 권장). 데이터 길이보다 크면 자동 조정
EMA_ALPHA    = 0.6          # 0<alpha<=1, 클수록 덜 매끈하고 반응 빠름

# 음영(band) 설정
# SHADE_MODE: "std" | "iqr" | None
SHADE_MODE   = "std"        # 스무딩 기준으로 ±표준편차 or IQR/2 범위를 밴드로
SHADE_SCALE  = 1.0          # 표준편차(k*std) 또는 iqr*(k*0.5)
SHADE_ALPHA  = 0.2

# 마커/선
DRAW_RAW_POINTS = True
RAW_POINT_SIZE  = 10
SMOOTH_LINEWIDTH = 2

# x축 간격 (너무 빽빽하면 xticks 줄이기)
XTICK_STEP = 0  # 0이면 자동(적응), >0이면 해당 간격으로 눈금 배치

# =========================
# UTILS
# =========================

def apply_font_settings():
    mpl.rcParams["font.family"] = FONT_FAMILY
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]

def expand_run_dir(name: str) -> str:
    return os.path.expanduser(os.path.join("~", name))

def read_txt_series(path: str) -> List[float]:
    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return []
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s))
            except:
                pass
    return vals

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    window = min(window, len(y)) if len(y) > 0 else 1
    if window <= 1:
        return y.copy()
    # same-패딩 이동평균
    kernel = np.ones(window) / window
    y_pad = np.pad(y, (window//2, window-1-window//2), mode="edge")
    return np.convolve(y_pad, kernel, mode="valid")

def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) == 0:
        return y
    alpha = float(alpha)
    alpha = min(max(alpha, 1e-6), 1.0)
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def rolling_std(y: np.ndarray, window: int) -> np.ndarray:
    """centered rolling std (same length as y)."""
    if len(y) == 0:
        return y
    window = max(2, int(window))
    window = min(window, len(y))  # 창이 데이터보다 길면 자르기

    # centered padding
    pad = window // 2
    y_pad = np.pad(y, (pad, window - 1 - pad), mode="edge")  # 길이: len(y) + window - 1

    # 누적합 앞에 0을 붙여 길이 보존(L - n + 1)
    csum  = np.cumsum(y_pad, dtype=float)
    csum2 = np.cumsum(y_pad**2, dtype=float)
    csum  = np.concatenate([[0.0], csum])
    csum2 = np.concatenate([[0.0], csum2])

    n = window
    # 길이: len(y_pad) - n + 1 == len(y)
    sums  = csum[n:]  - csum[:-n]
    sums2 = csum2[n:] - csum2[:-n]
    means = sums / n
    vars_ = (sums2 / n) - (means**2)
    np.maximum(vars_, 0.0, out=vars_)  # 수치오차 보정
    stds  = np.sqrt(vars_)
    return stds

def rolling_iqr(y: np.ndarray, window: int) -> np.ndarray:
    # 간단/빠름: centered window로 percentile 추정 (엣지 edge-pad)
    window = max(3, int(window) | 1)  # 홀수 강제
    pad = window//2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        seg = y_pad[i:i+window]
        q75, q25 = np.percentile(seg, [75, 25])
        out[i] = q75 - q25
    return out

def smooth_series(y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if len(y) == 0 or SMOOTH_MODE is None:
        return y, None
    if SMOOTH_MODE == "ma":
        sm = moving_average(y, MA_WINDOW)
        if SHADE_MODE == "std":
            sd = rolling_std(y, MA_WINDOW)
            band = sd * SHADE_SCALE
        elif SHADE_MODE == "iqr":
            iqr = rolling_iqr(y, MA_WINDOW)
            band = (iqr * 0.5) * SHADE_SCALE
        else:
            band = None
        return sm, band
    elif SMOOTH_MODE == "ema":
        sm = ema(y, EMA_ALPHA)
        if SHADE_MODE == "std":
            sd = rolling_std(y, max(5, int(1.0/EMA_ALPHA)))
            band = sd * SHADE_SCALE
        elif SHADE_MODE == "iqr":
            iqr = rolling_iqr(y, max(5, int(1.0/EMA_ALPHA)|1))
            band = (iqr * 0.5) * SHADE_SCALE
        else:
            band = None
        return sm, band
    else:
        return y, None

def _scaled_size(size: Tuple[float, float]) -> Tuple[float, float]:
    """FIGSIZE_SCALE을 적용한 새로운 크기 반환."""
    return (size[0] * FIGSIZE_SCALE, size[1] * FIGSIZE_SCALE)

def plot_curve(y: List[float], title: str, ylabel: str, save_path: str, figsize: Tuple[float, float]):
    y = np.asarray(y, dtype=float)
    x = np.arange(1, len(y)+1)

    # 스무딩
    y_smooth, band = smooth_series(y)

    fig, ax = plt.subplots(figsize=_scaled_size(figsize))
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["axes"])
    ax.tick_params(labelsize=FONT_SIZES["ticks"])

    # --- RAW(원 데이터) 표시 로직: points / line / none ---
    if len(y) > 0 and DRAW_RAW_SERIES == "points":
        ax.scatter(x, y, s=RAW_POINT_SIZE, color=COLOR_RAW, alpha=0.4,
                   label=LEGEND_RAW, zorder=2)
    elif len(y) > 0 and DRAW_RAW_SERIES == "line":
        ax.plot(x, y, color=COLOR_RAW, alpha=RAW_LINE_ALPHA,
                label=LEGEND_RAW, zorder=2)

    # --- 스무딩 선 + 음영(밴드) ---
    if len(y) > 0:
        ax.plot(x, y_smooth, color=COLOR_SMOOTH, linewidth=SMOOTH_LINEWIDTH,
                label=LEGEND_SMO, zorder=3)

        if band is not None:
            L = min(len(x), len(y_smooth), len(band))
            upper = y_smooth[:L] + band[:L]
            lower = y_smooth[:L] - band[:L]
            ax.fill_between(x[:L], lower, upper, color=COLOR_SHADE,
                            alpha=SHADE_ALPHA, linewidth=0, zorder=1)

    if SHOW_GRID:
        ax.grid(**GRID_STYLE)

    # 범례: raw를 안 그리면 'Smoothed'만 표시됨
    # leg = ax.legend(
    #     title=LEGEND_TITLE,
    #     fontsize=FONT_SIZES["legend"],
    #     loc="lower right",   # ← 원하는 위치로 변경
    #     frameon=True       # (선택) 범례 박스 테두리 없애기
    # )
    # if leg and leg.get_title():
    #     leg.get_title().set_fontsize(FONT_SIZES["legend"])

    if XTICK_STEP > 0 and len(x) > 0:
        ticks = np.arange(1, len(x)+1, XTICK_STEP)
        ax.set_xticks(ticks)

    if USE_TIGHT_LAYOUT:
        fig.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_kwargs = dict(dpi=SAVE_DPI, transparent=SAVE_TRANSPARENT)
    if BBOX_TIGHT:
        save_kwargs.update(bbox_inches="tight", pad_inches=PAD_INCHES)

    fig.savefig(save_path, **save_kwargs)
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

    if not rewards:
        print(f("[WARN] empty or missing: {reward_path}"))
    if not evacs:
        print(f("[WARN] empty or missing: {evac_path}"))

    # 저장 경로
    out_reward = os.path.join(run_dir, f"curve_reward_sm_{SMOOTH_MODE}.{SAVE_FORMAT}")
    out_evac   = os.path.join(run_dir, f"curve_evac100_sm_{SMOOTH_MODE}.{SAVE_FORMAT}")

    # ▶ 여기서 그래프별 figure 크기 전달
    plot_curve(rewards, TITLE_REWARD, YLABEL_REW, out_reward, figsize=FIGSIZE_REWARD)
    plot_curve(evacs,   TITLE_EVAC,   YLABEL_EVAC, out_evac,   figsize=FIGSIZE_EVAC)

    print("[DONE]")

if __name__ == "__main__":
    main()
