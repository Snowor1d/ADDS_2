#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-series Evacuation & Reward Plotter
- Nested Folder Structure 지원 (Map -> Ablation -> Target -> N개의 txt 파일)
- 다중 시드(Multiple txt files) 병합: 평균(Mean) 및 음영(Shade - Std, Min-Max, SE) 지원
- 각각 개별 그래프(eva100s, rewards) 및 합쳐진 그래프(Combined) 자동 생성
- B&W(흑백) 환경에서도 구분 가능하도록 선 스타일(Line styles), 마커 지원
- 기존 스무딩(MA/EMA), 폰트, Y-Clip 등의 기능 유지
- 그래프별(eva100s, rewards) Y축 가시 범위(ylim_min, ylim_max) 개별 조절 기능 추가
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# =========================
# 1. Directory & Structure Config
# =========================
HOME_DIR = os.path.expanduser("~")
ROOT_DIR = os.path.join(HOME_DIR, "ablations3") # 실제 데이터가 있는 최상위 경로로 수정하세요
OUT_DIR  = os.path.join(ROOT_DIR, "plots")

# 탐색할 폴더 리스트 (순서대로 범례/그래프에 적용됨)
MAP_NUMBERS  = ["3maps", "10maps", "30maps", "100maps", "300maps"]
ABLATIONS    = ["Ours", "NotEgo", "NotEpsilon", "NotFiLM"]
GRAPH_TARGETS= ["eva100s", "rewards"]

# =========================
# 2. Data Processing Config
# =========================
MAX_EPISODE = 7000

# 여러 개의 txt 파일이 있을 때 음영을 그리는 방식
# "std" (표준편차), "minmax" (최소-최대), "se" (표준오차), None (음영 없음)
SHADE_MODE  = "std" 
SHADE_ALPHA = 0.2

# Target별 특화 설정 (스케일, Y축 데이터 클리핑, Y축 뷰 범위 지정, Y축 라벨)
TARGET_CONFIGS = {
    "eva100s": {
        "ylabel": "Time (s)",
        "scale": 0.5,           # 기존 코드의 yarr / 2 반영
        "clip_min": 250,        # 데이터를 이 값 이하로 내려가지 않게 자름
        "clip_max": 3500,       # 데이터를 이 값 이상으로 올라가지 않게 자름
        "ylim_min": 501,          # ★ 그래프 y축 뷰의 최소값 (None이면 자동)
        "ylim_max": 2500,       # ★ 그래프 y축 뷰의 최대값 (None이면 자동)
        "yticks_interval" : 500,
        "xticks_interval" : 2000
    },
    "rewards": {
        "ylabel": "Reward",
        "scale": 1.0,
        "clip_min": None,       
        "clip_max": None,
        "ylim_min": None,        # ★ 필요에 따라 변경하세요 (예: -50)
        "ylim_max": None,         # ★ 필요에 따라 변경하세요 (예: 200)
        "yticks_interval" : 500
    }
}

# =========================
# 3. Style & Graph Config
# =========================
SAVE_DPI    = 300
SAVE_FORMAT = "png"
FIGSIZE_SINGLE = (12, 6.5)
FIGSIZE_COMBINED = (20, 8)

FONT_SIZES = {"title": 0, "axes": 35, "ticks": 35, "legend": 0}
FONT_FAMILY = "DejaVu Serif"

SHOW_GRID  = True
GRID_STYLE = dict(color="#AAAAAA", linestyle="--", linewidth=0.8, alpha=0.7)

# --- Smoothing ---
SMOOTH_ENABLE   = True
SMOOTH_KIND     = "ma"    # "ma" | "ema"
SMOOTH_STRENGTH = 1   # 0.0 ~ 1.0

MA_WINDOW_MIN = 5
MA_WINDOW_MAX = 500
EMA_ALPHA_MIN = 0.05
EMA_ALPHA_MAX = 0.50

# --- Raw trace (Mean line 렌더링 시 노이즈 그대로 표시 여부) ---
SHOW_RAW_TRACE      = False
RAW_TRACE_ALPHA     = 0.15
RAW_TRACE_LINEWIDTH = 1

# --- Markers ---
MARKERS_ENABLE   = True
MARKER_SIZE_PT   = 14
MARKER_EVERY     = "auto" # int (e.g., 500) or "auto"
MARKER_EDGEWIDTH = 1.2

# --- Ablation 요소별 고정 스타일 세팅 ---
# color: 선 및 마커 색상
# ls: 흑백 호환을 위한 선 스타일 ("-": 실선, "--": 파선, "-.": 1점쇄선, ":": 점선)
# marker: 각 Ablation을 명확히 구분하기 위한 모양
ABLATION_STYLES = {
    "Ours":       {"color": "#5d85f1", "ls": "-",  "lw": 3.5, "marker": "o"},
    "NotEgo":     {"color": "#e21717", "ls": "-", "lw": 3.5, "marker": "s"},
    "NotEpsilon": {"color": "#e27227", "ls": "-", "lw": 3.5, "marker": "^"},
    "NotFiLM":    {"color": "#723e16", "ls": "-",  "lw": 3.5, "marker": "D"}
}

# 정의되지 않은 Ablation이 들어올 경우를 대비한 기본 사이클
FALLBACK_COLORS = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]
FALLBACK_LS     = ["-", "--", "-.", ":"]
FALLBACK_MARKER = ["v", "<", ">", "P", "X", "h"]


# =========================
# Core Data Classes
# =========================
@dataclass
class AggregatedSeries:
    name: str
    x: np.ndarray
    mean_y: np.ndarray
    upper_y: Optional[np.ndarray]
    lower_y: Optional[np.ndarray]
    raw_mean_y: np.ndarray # 스무딩 이전의 mean 값 (raw trace용)

# =========================
# Font utils
# =========================
def apply_font_settings():
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY, "Times New Roman", "DejaVu Serif"]
    mpl.rcParams["axes.titlesize"]  = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"]  = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]
    mpl.rcParams["axes.unicode_minus"] = False

# =========================
# Smoothing utils
# =========================
def _compute_ma_window(strength: float) -> int:
    s = float(np.clip(strength, 0.0, 1.0))
    win = int(round(MA_WINDOW_MIN + s * (MA_WINDOW_MAX - MA_WINDOW_MIN)))
    return max(1, win + 1 if win % 2 == 0 else win)

def _compute_ema_alpha(strength: float) -> float:
    s = float(np.clip(strength, 0.0, 1.0))
    return float(np.clip(EMA_ALPHA_MAX - s * (EMA_ALPHA_MAX - EMA_ALPHA_MIN), 1e-6, 1.0))

def apply_smoothing(y: np.ndarray) -> np.ndarray:
    if not SMOOTH_ENABLE or len(y) < 2:
        return y.copy()
    
    if SMOOTH_KIND == "ma":
        window = _compute_ma_window(SMOOTH_STRENGTH)
        if window == 1: return y.copy()
        pad = window // 2
        ypad = np.pad(y, (pad, pad), mode="symmetric")
        kernel = np.ones(window, dtype=float) / window
        return np.convolve(ypad, kernel, mode="valid")
    
    elif SMOOTH_KIND == "ema":
        alpha = _compute_ema_alpha(SMOOTH_STRENGTH)
        out = np.empty_like(y, dtype=float)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = alpha * y[i] + (1.0 - alpha) * out[i-1]
        return out
    
    return y.copy()

# =========================
# Data Loading & Aggregation
# =========================
def read_txt_series(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    vals = []
    for ln in raw.splitlines():
        for tok in re.split(r"[,\s]+", ln.strip()):
            if tok:
                try: vals.append(float(tok))
                except ValueError: pass
    return np.array(vals, dtype=float)
def load_ablation_data(map_dir: str, ablation: str, target: str) -> Optional[AggregatedSeries]:
    target_dir = os.path.join(ROOT_DIR, map_dir, ablation, target)
    if not os.path.isdir(target_dir):
        return None

    # txt 파일들 전부 읽기
    txt_files = [f for f in os.listdir(target_dir) if f.endswith(".txt")]
    if not txt_files:
        return None
    
    scale = TARGET_CONFIGS.get(target, {}).get("scale", 1.0)
    
    all_y = []
    for f in txt_files:
        y_arr = read_txt_series(os.path.join(target_dir, f)) * scale
        if len(y_arr) > 0:
            all_y.append(y_arr)
            
    if not all_y: return None

    # 데이터 길이가 다를 경우 가장 짧은 길이에 맞춤 (에피소드 통일)
    min_len = min(len(arr) for arr in all_y)
    min_len = min(min_len, MAX_EPISODE) # MAX_EPISODE로 자르기
    
    all_y_cropped = np.array([arr[:min_len] for arr in all_y])
    x = np.arange(1, min_len + 1, dtype=float)
    
    # Raw mean 계산 (Raw trace 오버레이용 - 스무딩 전 순수 원본의 평균)
    raw_mean_y = np.mean(all_y_cropped, axis=0)

    # ★ 핵심 변경점: 개별 시드 데이터를 먼저 모두 스무딩 처리합니다!
    all_y_smoothed = np.array([apply_smoothing(arr) for arr in all_y_cropped])
    
    # 스무딩된 데이터로 평균 및 음영(분산) 계산
    mean_y = np.mean(all_y_smoothed, axis=0)
    upper_y, lower_y = None, None
    
    if len(all_y_smoothed) > 1 and SHADE_MODE is not None:
        if SHADE_MODE == "std":
            std_y = np.std(all_y_smoothed, axis=0)
            upper_y = mean_y + std_y
            lower_y = mean_y - std_y
        elif SHADE_MODE == "minmax":
            upper_y = np.max(all_y_smoothed, axis=0)
            lower_y = np.min(all_y_smoothed, axis=0)
        elif SHADE_MODE == "se":
            se_y = np.std(all_y_smoothed, axis=0) / np.sqrt(len(all_y_smoothed))
            upper_y = mean_y + se_y
            lower_y = mean_y - se_y
    
    # Y Clip 적용
    c_min = TARGET_CONFIGS.get(target, {}).get("clip_min")
    c_max = TARGET_CONFIGS.get(target, {}).get("clip_max")
    
    def clip_arr(arr):
        if arr is None: return None
        if c_min is not None: arr = np.maximum(arr, c_min)
        if c_max is not None: arr = np.minimum(arr, c_max)
        return arr

    return AggregatedSeries(
        name=ablation, x=x, raw_mean_y=clip_arr(raw_mean_y),
        mean_y=clip_arr(mean_y), upper_y=clip_arr(upper_y), lower_y=clip_arr(lower_y)
    )

# =========================
# Plotting Logics
# =========================
def get_style(idx: int, name: str) -> dict:
    if name in ABLATION_STYLES:
        return ABLATION_STYLES[name]
    return {
        "color": FALLBACK_COLORS[idx % len(FALLBACK_COLORS)],
        "ls": FALLBACK_LS[idx % len(FALLBACK_LS)],
        "lw": 3,
        "marker": FALLBACK_MARKER[idx % len(FALLBACK_MARKER)]
    }

def draw_on_axis(ax, series_list: List[AggregatedSeries], target: str):
    for idx, series in enumerate(series_list):
        style = get_style(idx, series.name)
        
        # 1. Raw trace
        if SHOW_RAW_TRACE:
            ax.plot(series.x, series.raw_mean_y, color=style["color"],
                    linewidth=RAW_TRACE_LINEWIDTH, alpha=RAW_TRACE_ALPHA, linestyle=style["ls"])
            
        # 2. Shade (Variance/MinMax)
        if series.upper_y is not None and series.lower_y is not None:
            ax.fill_between(series.x, series.lower_y, series.upper_y,
                            color=style["color"], alpha=SHADE_ALPHA, edgecolor="none")
            
        # 3. Marker interval setup
        marker_kwargs = {}
        if MARKERS_ENABLE:
            me = MARKER_EVERY if isinstance(MARKER_EVERY, int) else max(1, len(series.x) // 15)
            marker_kwargs = dict(
                marker=style["marker"], markersize=MARKER_SIZE_PT,
                markerfacecolor=style["color"], markeredgecolor=style["color"],
                markeredgewidth=MARKER_EDGEWIDTH, markevery=me
            )
            
        # 4. Main Line
        ax.plot(series.x, series.mean_y, label=series.name,
                color=style["color"], linewidth=style["lw"], linestyle=style["ls"],
                **marker_kwargs)

    ax.set_xlabel("Training Episode", fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(TARGET_CONFIGS.get(target, {}).get("ylabel", "Value"), fontsize=FONT_SIZES["axes"])
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    
    # ★ 추가: X축 여백 제거 및 0부터 최대 에피소드까지 꽉 채우기
    if series_list:
        max_x = max(series.x[-1] for series in series_list)
        ax.set_xlim(0, max_x)
    
    # ==========================================
    # ★ 여기가 빠져있었습니다! 이 세 줄을 추가해 주세요.
    y_step = TARGET_CONFIGS.get(target, {}).get("yticks_interval")
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
    x_step = TARGET_CONFIGS.get(target, {}).get("xticks_interval")
    if x_step is not None: 
        ax.xaxis.set_major_locator(MultipleLocator(x_step))
    # ==========================================

    # 5. 개별 그래프의 Y축 가시 범위(Lim) 적용
    ylim_min = TARGET_CONFIGS.get(target, {}).get("ylim_min")
    ylim_max = TARGET_CONFIGS.get(target, {}).get("ylim_max")
    if ylim_min is not None or ylim_max is not None:
        ax.set_ylim(bottom=ylim_min, top=ylim_max)
    
    # Grid
    if SHOW_GRID:
        ax.set_axisbelow(False)
        ax.grid(**GRID_STYLE)
        
    # Legend
    leg = ax.legend(frameon=False, loc="best")  
    for line in leg.get_lines():
        line.set_marker("None")

def create_single_plot(map_name: str, target: str, series_list: List[AggregatedSeries]):
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    draw_on_axis(ax, series_list, target)
    
    ax.set_title(f"[{map_name}] {target.capitalize()}", fontsize=FONT_SIZES["title"])
    fig.tight_layout()
    
    out_path = os.path.join(OUT_DIR, f"{map_name}_{target}.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved Single] {out_path}")

def create_combined_plot(map_name: str, dict_series: Dict[str, List[AggregatedSeries]]):
    # 만약 타겟들 중 데이터가 하나라도 없으면 결합 그래프 생성 취소
    if not all(t in dict_series and dict_series[t] for t in GRAPH_TARGETS):
        return

    fig, axes = plt.subplots(1, len(GRAPH_TARGETS), figsize=FIGSIZE_COMBINED)
    if len(GRAPH_TARGETS) == 1: axes = [axes]
    
    for ax, target in zip(axes, GRAPH_TARGETS):
        draw_on_axis(ax, dict_series[target], target)
        ax.set_title(target.capitalize(), fontsize=FONT_SIZES["title"])
        
    fig.suptitle(f"[{map_name}] Combined Performance", fontsize=FONT_SIZES["title"], y=1.02)
    fig.tight_layout()
    
    out_path = os.path.join(OUT_DIR, f"{map_name}_combined.{SAVE_FORMAT}")
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved Combined] {out_path}")

# =========================
# Main Execution
# =========================
def main():
    apply_font_settings()
    os.makedirs(OUT_DIR, exist_ok=True)
    
    for map_name in MAP_NUMBERS:
        dict_series_for_map = {}
        
        for target in GRAPH_TARGETS:
            series_list = []
            for ablation in ABLATIONS:
                data = load_ablation_data(map_name, ablation, target)
                if data is not None:
                    series_list.append(data)
            
            if series_list:
                dict_series_for_map[target] = series_list
                # 1. 개별 타겟 그래프 생성 (e.g., eva100s 전용, rewards 전용)
                create_single_plot(map_name, target, series_list)
            else:
                print(f"[WARN] No data found for {map_name} -> {target}")
                
        # 2. 합쳐진 그래프 생성 (subplots 1 x N)
        if dict_series_for_map:
            create_combined_plot(map_name, dict_series_for_map)

if __name__ == "__main__":
    main()