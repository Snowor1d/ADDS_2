#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evac-100 Steps 비교 플로터 (스무딩 + X값 클리핑 + 커스텀 범례 + 마커 제어)
- ~/evac100_steps 안의 evac100_max*.txt 파일 자동 탐색
- 각 파일은 "x y" 형식(공백/쉼표 허용)
- y > 1000 은 캡핑
- X값이 2.2×10^6을 초과하는 데이터는 모두 잘라냄
- 이동평균(MA) 또는 지수이동평균(EMA) 스무딩 지원
- 범례 이름 직접 지정(LEGEND_NAME_MAP)
- 마커 모양 통일/다양화 선택(MARKER_MODE: "uniform" | "varied")
"""

import os
import re
from typing import List, Tuple, Dict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# =========================
# 경로/출력
# =========================
HOME_DIR   = os.path.expanduser("~")
ROOT_DIR   = os.path.join(HOME_DIR, "evac100_steps")
OUT_DIR    = ROOT_DIR
SAVE_NAME  = "evac100_steps_compare"
SAVE_DPI   = 300
SAVE_FORMAT= "png"

INCLUDE_REGEX = r"^evac100_max\d+\.txt$"

# =========================
# 표시/스타일 옵션
# =========================
FIGSIZE = (15, 15)
FONT_SIZES = {"title": 30, "axes": 45, "ticks": 45, "legend": 40}
FONT_PREFER = ["DejaVu Serif", "Times New Roman", "Times"]
SHOW_GRID = True
GRID_STYLE = dict(color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7)

XLABEL = "Learning Step"
YLABEL = "Time step"
PLOT_TITLE = ""
LEGEND_LOC = "best"

# 라인/마커
MARKERS_ENABLE         = True
MARKER_SIZE_PT         = 23
MARKER_EDGEWIDTH       = 1.2
MAX_MARKERS_PER_SERIES = 20
DEFAULT_LINEWIDTH      = 3.5

# 색상 팔레트 (Okabe-Ito)
OKABE_ITO = ["#000000","#E69F00","#56B4E9","#009E73",
             "#F0E442","#0072B2","#D55E00","#CC79A7"]

# 선 스타일 (모두 실선 유지하고 싶으면 아래처럼)
LINE_STYLES = ["solid", "solid", "solid", "solid"]

SHOW_RAW_TRACE      = False
RAW_TRACE_ALPHA     = 0.2
RAW_TRACE_LINEWIDTH = 1.2

# =========================
# 값 제한
# =========================
Y_CAP = 1000.0
X_CLIP_MAX = 2.2e6   # ★ X값 클리핑 기준

# =========================
# 스무딩
# =========================
SMOOTH_ENABLE   = True
SMOOTH_KIND     = "ema"        # "ma" | "ema"
SMOOTH_STRENGTH = 20           # 내부에서 0~1로 clip됨

MA_WINDOW_MIN = 3
MA_WINDOW_MAX = 101
EMA_ALPHA_MIN = 0.005          # 더 부드럽게 하고 싶어 낮춰둠
EMA_ALPHA_MAX = 0.4

# =========================
# 범례 이름 커스텀
#   - 키: 파일명(정확히), 값: 원하는 표시 이름
#   - 예시로 한두 개만 적어둠. 필요에 따라 추가/수정하면 됨.
# =========================
LEGEND_NAME_MAP: Dict[str, str] = {
    "evac100_max1000.txt": "Max Step = 1000",
    "evac100_max2000.txt": "Max Step = 2000",
    "evac100_max4000.txt": "Max Step = 4000",
    "evac100_max6000.txt": "Max Step = 6000"
}

# =========================
# 마커 모드/주기
# =========================
MARKER_MODE = "varied"    # "uniform" | "varied"
UNIFORM_MARKER = "o"
MARKER_CYCLE = ["o", "s", "^", "D"]  # 시리즈별 다른 마커 (필요시 늘려도 됨)

# =========================
# 스무딩 함수
# =========================
def _compute_ma_window_from_strength(s: float) -> int:
    s = float(np.clip(s, 0.0, 1.0))
    w = int(round(MA_WINDOW_MIN + s * (MA_WINDOW_MAX - MA_WINDOW_MIN)))
    if w % 2 == 0:
        w += 1
    return max(1, w)

def _compute_ema_alpha_from_strength(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return float(np.clip(EMA_ALPHA_MAX - s * (EMA_ALPHA_MAX - EMA_ALPHA_MIN), 1e-6, 1.0))

def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")

def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if len(y) < 2:
        return y.copy()
    out = np.empty_like(y, dtype=float)
    out[0] = y[0]
    for i in range(1, len(y)):
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]
    return out

def smooth_series(y: np.ndarray) -> np.ndarray:
    if not SMOOTH_ENABLE or len(y) < 3:
        return y
    if SMOOTH_KIND.lower() == "ma":
        win = _compute_ma_window_from_strength(SMOOTH_STRENGTH)
        return moving_average(y, win)
    elif SMOOTH_KIND.lower() == "ema":
        alpha = _compute_ema_alpha_from_strength(SMOOTH_STRENGTH)
        return ema(y, alpha)
    return y

# =========================
# 유틸
# =========================
def apply_font_settings():
    installed = {f.name for f in fm.fontManager.ttflist}
    fams = [f for f in FONT_PREFER if f in installed]
    if not fams:
        fams = ["DejaVu Serif"]
    mpl.rcParams["font.family"] = fams[0]
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"]  = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]
    mpl.rcParams["axes.unicode_minus"] = False

def list_target_files(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Directory not found: {root}")
    cre = re.compile(INCLUDE_REGEX)
    files = [fn for fn in os.listdir(root)
             if os.path.isfile(os.path.join(root, fn)) and cre.match(fn)]
    files.sort()
    return files

def read_xy_pairs(path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = re.split(r"[,\s]+", line)
            if len(toks) < 2:
                continue
            try:
                x = float(toks[0])
                y = min(float(toks[1]), Y_CAP)  # y 캡핑
            except Exception:
                continue
            xs.append(x)
            ys.append(y)
    if not xs:
        return np.array([]), np.array([])
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(xs)
    return xs[order], ys[order]

def display_name_from_filename(fn: str) -> str:
    # 사용자 지정 이름이 있으면 우선 사용
    if fn in LEGEND_NAME_MAP:
        return LEGEND_NAME_MAP[fn]
    # 기본: evac100_maxXXXX.txt -> "Max XXXX"
    m = re.match(r"^evac100_max(\d+)\.txt$", fn)
    return f"Max {m.group(1)}" if m else os.path.splitext(fn)[0]

def marker_every(n_points: int) -> int:
    return max(1, int(round(n_points / MAX_MARKERS_PER_SERIES)))

def marker_for_index(i: int) -> str:
    if MARKER_MODE == "uniform":
        return UNIFORM_MARKER
    return MARKER_CYCLE[i % len(MARKER_CYCLE)]

# =========================
# 플로팅
# =========================
def plot_compare(root: str, out_dir: str):
    apply_font_settings()
    files = list_target_files(root)
    if not files:
        print(f"[WARN] No target files under: {root}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = []

    for i, fn in enumerate(files):
        path = os.path.join(root, fn)
        x, y = read_xy_pairs(path)
        if len(x) == 0:
            print(f"[SKIP] Empty/invalid: {fn}")
            continue

        # ★ X 클리핑
        mask = x <= X_CLIP_MAX
        x = x[mask]
        y = y[mask]
        if len(x) == 0:
            print(f"[SKIP] {fn} has no data ≤ {X_CLIP_MAX}")
            continue

        # 스무딩 (cap된 값에 적용)
        y_smooth = smooth_series(y)

        color = OKABE_ITO[i % len(OKABE_ITO)]
        ls    = LINE_STYLES[i % len(LINE_STYLES)]
        lbl   = display_name_from_filename(fn)

        # 원시 오버레이
        if SHOW_RAW_TRACE:
            ax.plot(x, y, color=color, linewidth=RAW_TRACE_LINEWIDTH,
                    alpha=RAW_TRACE_ALPHA, linestyle=ls)

        mk_kwargs = {}
        if MARKERS_ENABLE:
            mk_kwargs = dict(
                marker=marker_for_index(i),
                markersize=MARKER_SIZE_PT,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=MARKER_EDGEWIDTH,
                markevery=marker_every(len(x)),
            )

        line, = ax.plot(
            x, y_smooth,
            label=lbl,
            color=color,
            linewidth=DEFAULT_LINEWIDTH,
            linestyle=ls,
            **mk_kwargs
        )
        handles.append(line)

    ax.set_xlabel(XLABEL)
    ax.set_ylabel(YLABEL)
    if PLOT_TITLE:
        ax.set_title(PLOT_TITLE)
    if SHOW_GRID:
        ax.set_axisbelow(False)
        ax.grid(**GRID_STYLE)
    if handles:
        ax.legend(loc=LEGEND_LOC, frameon=False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{SAVE_NAME}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {out_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    plot_compare(ROOT_DIR, OUT_DIR)
