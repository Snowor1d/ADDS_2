#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Result_data_{EXP_NAME} 폴더를 분석해 논문용 그래프를 생성합니다.

폴더 구조 가정:
ROOT_DIR/
  └─ Result_{map_num}/
       └─ Result_{map_num}_{robot_version}/
            ├─ Result_{map}_{robot}_{test_i}/
            │    ├─ metrics.txt        # evacuation_100_time, all_agents_life_time
            │    └─ episode_log.txt    # [alive_0, alive_1, ...]
            └─ ...

출력:
- {ROOT_DIR}/map_{id}_evac100.png
- {ROOT_DIR}/map_{id}_episode_{band}.png
"""

import os
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import colorsys

# =========================
# 전역 파라미터 (모두 여기서 조절)
# =========================
ROOT_DIR = os.path.expanduser("~/Result_data_experiment_250903")  # 데이터 최상위 폴더
OUT_DIR  = ROOT_DIR                                  # 저장 폴더 (요청: ROOT_DIR 바로 아래)

# --- 그림/폰트 ---
FIGSIZE_EVAC   = (8, 5)     # (width, height) inches
FIGSIZE_EPISOD = (8, 5)
FONT_SIZES = {
    "title": 16,
    "axes":  13,
    "ticks": 11,
    "legend": 11,
}

ROBOT_LABELS = {
    "R": "R (Go-and-Back Guide)",      # 예: 가장 먼 agent와 출구 사이 왕복
    "T": "Direct-to-Goal",         # 예: 바로 출구 지향
    "Q": "RL-Trained",         # 예: 학습 모델
    "H": "Human-Control",
    "N": "No Robot"
    # 필요 시 더 추가: "Human": "Human (Joystick Baseline)" 등
}

# --- 제목/축 레이블 포맷 ---
TITLE_MAP_FORMAT      = "Map {map_id} - {plot_name}"   # 그래프 제목 포맷
TITLE_EVAC_NAME       = "Time Step for 100'%' Evacuation"
TITLE_EPISODE_NAME    = "Remained Agents"
XLABEL_EVAC           = "Robot Version"
YLABEL_EVAC           = "Evacuation 100 Time (steps)"
XLABEL_EPISODE        = "Step"
YLABEL_EPISODE        = "Number of Remained Agents"

# --- 저장 옵션 ---
SAVE_DPI     = 220
SAVE_FORMAT  = "png"  # "png" | "pdf" 등

# --- 밴드(음영) ---
BAND_MODE: Optional[str] = "p25_75"  # None | "minmax" | "std" | "p25_75"
BAND_ALPHA = 0.18

# --- evac100 표시 옵션 ---
EVAC_SHOW_BAND = True      # True면 min–max 밴드, False면 밴드 없이 선+점만
JITTER_POINTS  = False     # 점 산포: 같은 x축에서 좌우로 흩뿌리기
JITTER_WIDTH   = 0.55      # 산포 폭 (카테고리 폭의 비율 0~1 권장)

# --- y축 범위 (None이면 자동) ---
YLIM_EVAC: Optional[Tuple[float, float]]    = None   # e.g., (0, 3000)
YLIM_EPISODE: Optional[Tuple[float, float]] = None   # e.g., (0, 20)

# --- 색상(HSL) 컨트롤 ---
# 로봇 버전별 HSL(0~1) 지정. 지정하지 않으면 COLOR_CYCLE_HUES에서 순환
ROBOT_HSL: Dict[str, Tuple[float, float, float]] = {
    # 예) "R": (0.01, 0.75, 0.45),  "T": (0.58, 0.7, 0.5),  "Q": (0.33, 0.7, 0.45)
}
# 기본 색상 순환용 H(색상) 값 (0~1)
COLOR_CYCLE_HUES = [0.00, 0.58, 0.33, 0.12, 0.75, 0.46, 0.90, 0.20]
DEFAULT_S = 0.70   # 기본 채도
DEFAULT_L = 0.48   # 기본 명도

# 전역 스케일(명도/채도/색상) 조절 (모든 색에 곱/가감)
GLOBAL_S_SCALE = 1.00   # 1.2면 채도 20% 증가
GLOBAL_L_SCALE = 1.00   # 0.9면 전체적으로 약간 어둡게
GLOBAL_H_SHIFT = 0.00   # 0.05면 모든 색상 H를 0.05만큼 회전

# 보조 스타일
SHOW_GRID = True
GRID_STYLE = dict(linestyle="--", alpha=0.3)

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
    robots: Dict[str, RobotGroup] = field(default_factory=dict)  # robot_version -> RobotGroup

# =========================
# 로딩 유틸
# =========================
METRICS_FILE = "metrics.txt"
EPISODE_LOG_FILE = "episode_log.txt"

def normalize_band_mode(mode):
    """문자열 'none', 'off' 등은 None으로 정규화"""
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
    # 리스트 문자열 형태 우선
    try:
        arr = ast.literal_eval(txt)
        return list(arr)
    except Exception:
        # 라인별/쉼표 혼합 허용
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
# 색상 유틸 (HSL → RGB)
# =========================
def hsl_to_rgb_hex(h: float, s: float, l: float) -> str:
    """HSL(0~1)을 글로벌 스케일/시프트 반영 후 RGB hex로 변환."""
    h = (h + GLOBAL_H_SHIFT) % 1.0
    s = max(0.0, min(1.0, s * GLOBAL_S_SCALE))
    l = max(0.0, min(1.0, l * GLOBAL_L_SCALE))
    r, g, b = colorsys.hls_to_rgb(h, l, s)  # colorsys: HLS(순서 주의: H,L,S)
    return "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

def color_for_robot(robot: str, idx: int) -> str:
    if robot in ROBOT_HSL:
        h, s, l = ROBOT_HSL[robot]
        return hsl_to_rgb_hex(h, s, l)
    # cycle hue
    h = COLOR_CYCLE_HUES[idx % len(COLOR_CYCLE_HUES)]
    return hsl_to_rgb_hex(h, DEFAULT_S, DEFAULT_L)

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
    """0-패딩된 [R,L] 행렬에서 평균·밴드 계산. mode=None이면 평균만."""
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
# 플롯 함수
# =========================
def apply_axes_style(ax, ylim: Optional[Tuple[float, float]]):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.grid(**GRID_STYLE)
    if ylim is not None:
        ax.set_ylim(*ylim)

def title_for(map_id: int, plot_name: str) -> str:
    return TITLE_MAP_FORMAT.format(map_id=map_id, plot_name=plot_name)

def plot_evacuation_per_map(map_id: int, mdata: MapData, out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_EVAC)
    ax.set_title(title_for(map_id, TITLE_EVAC_NAME), fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL_EVAC, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL_EVAC, fontsize=FONT_SIZES["axes"])

    robot_versions = sorted(mdata.robots.keys())
    xs = np.arange(len(robot_versions))
    width = 0.6

    for i, rv in enumerate(robot_versions):
        disp_label = label_for_robot(rv)   # ← 추가
        color = color_for_robot(rv, i)
        runs = mdata.robots[rv].runs
        vals = [r.evacuation_100_time for r in runs]
        if not vals:
            continue
        vmin, vmean, vmax = compute_min_mean_max(vals)

        left, right = xs[i] - width/2, xs[i] + width/2
        ax.hlines(vmean, left, right, color=color, linewidth=2.8, label=f"{disp_label} (mean)")
        if EVAC_SHOW_BAND:
            ax.fill_between([left, right], [vmin, vmin], [vmax, vmax], color=color, alpha=BAND_ALPHA)

        if JITTER_POINTS:
            jitter = (np.random.rand(len(vals)) - 0.5) * (width * JITTER_WIDTH)
            ax.scatter(xs[i] + jitter, vals, color=color, alpha=0.4, s=18)
        else:
            ax.scatter([xs[i]] * len(vals), vals, color=color, alpha=0.4, s=18)

    ax.set_xticks(xs, robot_versions)
    ax.legend(fontsize=FONT_SIZES["legend"])
    apply_axes_style(ax, YLIM_EVAC)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"map_{map_id}_evac100.{SAVE_FORMAT}")
    fig.savefig(save_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {save_path}")

def plot_episode_log_padded(map_id: int, mdata: MapData, out_dir: str, band_mode: Optional[str]):
    title_mode = band_mode if band_mode is not None else "no-band"
    fig, ax = plt.subplots(figsize=FIGSIZE_EPISOD)
    ax.set_title(title_for(map_id, f"{TITLE_EPISODE_NAME} ({title_mode})"), fontsize=FONT_SIZES["title"])
    ax.set_xlabel(XLABEL_EPISODE, fontsize=FONT_SIZES["axes"])
    ax.set_ylabel(YLABEL_EPISODE, fontsize=FONT_SIZES["axes"])

    for i, (rv, group) in enumerate(sorted(mdata.robots.items(), key=lambda x: x[0])):
        disp_label = label_for_robot(rv)   # ← 추가
        color = color_for_robot(rv, i)
        runs = [r.episode_log for r in group.runs]
        if not runs:
            continue

        L = max(len(r) for r in runs)
        M = np.stack([pad_to_len(r, L, pad_value=0.0) for r in runs], axis=0)
        mean_curve, lower, upper = per_step_stats_from_matrix(M, mode=band_mode)

        x = np.arange(L)
        ax.plot(x, mean_curve, label=disp_label, color=color, linewidth=2.2)
        if lower is not None and upper is not None:
            ax.fill_between(x, lower, upper, color=color, alpha=BAND_ALPHA)

    ax.legend(fontsize=FONT_SIZES["legend"])
    apply_axes_style(ax, YLIM_EPISODE)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"map_{map_id}_episode_{title_mode}.{SAVE_FORMAT}")
    fig.savefig(save_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"[Saved] {save_path}")

def label_for_robot(robot: str) -> str:
    return ROBOT_LABELS.get(robot, robot)

    

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
