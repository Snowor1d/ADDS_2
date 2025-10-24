#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-series Evacuation-100 Plotter (with hard cutoff at episode 15000)
- Reads all *.txt under ~/evacuation_100s
- Each txt: one number per line (evacuation_100_time per episode index)
- Always truncates series to episodes <= 15000
- Plots all series on ONE figure with optional smoothing (MA/EMA)
- Rich customization: fonts, colors (HSL), linewidths, legend, grid, y-clip, raw-trace overlay, aliases, per-series style overrides
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import colorsys
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D

# --- 범례 순서(표시명 기준; resolve_display_name 적용 후의 name) ---
LEGEND_ORDER = [
    "Ours",
    "Epsilon = 0",
    "R_danger = 0",
    "R_alived = 0",
    "R_penalty = 0",
    "R_base = 0",
]

# =========================
# 전역 설정
# =========================
HOME_DIR   = os.path.expanduser("~")
ROOT_DIR   = os.path.join(HOME_DIR, "evacuation_100s")   # 입력 폴더
OUT_DIR    = ROOT_DIR                                    # 저장 폴더(동일)
SAVE_NAME  = "evac100_multi"                             # 저장 파일명 prefix
SAVE_DPI   = 300
SAVE_FORMAT= "png"

# --- 하드 컷오프(항상 적용) ---
MAX_EPISODE = 15000  # 15000 초과는 항상 잘라냄

# --- 그림/폰트 ---
FIGSIZE = (15, 10)
FONT_SIZES = {"title": 30, "axes": 30, "ticks": 30, "legend": 30}
FONT_MODE = "serif"              # "serif" | "sans" | "mono" | "custom"
FONT_FAMILY = "DejaVu Serif"     # MODE="custom"일 때만 사용 (또는 .ttf 경로)
FONT_FALLBACKS: List[str] = []   # ["Noto Sans CJK KR", "NanumGothic"] 등 설치 시 추가

# --- 제목/축/범례 ---
PLOT_TITLE      = ""
XLABEL          = "Training Episode"
YLABEL          = "Time Step"
LEGEND_TITLE    = ""
LEGEND_LOC      = "upper right"  # "best", "upper left", ...
SHOW_GRID       = True
GRID_STYLE      = dict(linestyle="--", alpha=0.3)

# --- 데이터 표시 옵션 ---
# 원본(raw)도 희미하게 같이 그릴지(스무딩 선 대비)
SHOW_RAW_TRACE      = False
RAW_TRACE_ALPHA     = 0.18
RAW_TRACE_LINEWIDTH = 1.2

# Y축 클립(보기 좋게 제한하고 싶을 때 설정. None이면 자동)
CLIP_Y_MIN: Optional[float] = None
CLIP_Y_MAX: Optional[float] = None

# --- 스무딩 옵션 ---
# method: "ma"(이동평균) | "ema"(지수이동평균) | None
SMOOTH_METHOD   = "ma"
MA_WINDOW       = 101          # 이동평균 창 크기 (>=1, 클수록 더 매끈)
EMA_ALPHA       = 0.1         # 지수이동평균 알파(0~1, 작을수록 더 매끈)

# --- 라인 스타일/색 ---
DEFAULT_LINEWIDTH = 3.2
H_BLUE   = 253/360.0  # Okabe–Ito Blue (그대로)
H_ORANGE = 30/360.0   # 선명한 오렌지
H_GREEN  = 120/360.0  # 그린
H_RED    = 0/360.0    # 레드
H_PURPLE = 280/360.0  # 보라
H_YELLOW = 55/360.0   # 노랑

COLOR_CYCLE_HUES = [H_RED, H_ORANGE, H_PURPLE, H_YELLOW, H_GREEN, H_GREEN, 0.58, 0.33, 0.12, 0.75, 0.5, 0.5]
DEFAULT_S = 0.90
DEFAULT_L = 0.42
GLOBAL_S_SCALE = 1.00
GLOBAL_L_SCALE = 1.00
GLOBAL_H_SHIFT = 0.00

# 개별 시리즈 스타일 오버라이드(파일 표시명 기준)
# e.g., {"RL-Trained": {"linewidth": 4.5, "color": "#3366cc"}}
SERIES_STYLE_OVERRIDES: Dict[str, Dict[str, object]] = {"Ours": {"linewidth": 5, "color": "#3366cc"}}

# --- 파일명 → 표시명 매핑 ---
# 비워두면 파일명(확장자 제외)을 그대로 사용
# 예: {"Q_set.txt":"RL-Trained", "H_set.txt":"Human Control"}
FILENAME_ALIASES: Dict[str, str] = {"eva100_A0":"R_alived = 0", "eva100_B0":"R_danger = 0", "eva100_P0" : "R_penalty = 0", "eva100_F0" : "R_base = 0", "eva100_E0":"Epsilon = 0", "eva100_SOTA" : "Ours"}

# --- 파일 필터 ---
INCLUDE_REGEX: Optional[str] = r".*\.txt$"  # 포함 정규식
EXCLUDE_REGEX: Optional[str] = None          # 제외 정규식

# =========================
# 폰트 유틸
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
    # 1) 확장자 포함 키 우선
    if fn in FILENAME_ALIASES:
        return FILENAME_ALIASES[fn]
    # 2) 확장자 제거 키
    if base in FILENAME_ALIASES:
        return FILENAME_ALIASES[base]
    # 3) 기본: 확장자 제거 이름
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
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
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
# 색상 유틸
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
# 데이터 구조/로딩
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
        # 숫자 한 개/여러 개 모두 지원
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

    files = [fn for fn in os.listdir(root)
             if os.path.isfile(os.path.join(root, fn))]
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

        # === 항상 15000 초과를 잘라냄 ===
        mask = x <= MAX_EPISODE
        if not np.any(mask):
            # 전부 컷오프 밖이면 스킵
            continue
        x = x[mask]
        y = y[mask]

        #name = FILENAME_ALIASES.get(fn, os.path.splitext(fn)[0])
        name = resolve_display_name(fn)
        series_list.append(Series(name=name, x=x, y=y))
    return series_list

# =========================
# 스무딩
# =========================
def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(y) < 2:
        return y.copy()
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
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
    m = (SMOOTH_METHOD or "").lower()
    if m in ("", "none", "off", None):
        series.y_smooth = None
        return series
    if m == "ma":
        series.y_smooth = moving_average(series.y, MA_WINDOW)
    elif m == "ema":
        series.y_smooth = ema(series.y, EMA_ALPHA)
    else:
        series.y_smooth = None
    return series

def _apply_legend_order(ax, handles: List[Line2D], order: List[str]) -> List[Line2D]:
    """
    handles를 주어진 라벨 순서(order)에 맞춰 재배열하고,
    같은 라벨이 여러 번 있으면 마지막 것만 남김.
    order에 없는 라벨은 원래 등장 순서를 유지하여 뒤에 붙임.
    """
    # 1) 같은 라벨 중복 제거(마지막 handle이 살아남도록)
    last_by_label: Dict[str, Line2D] = {}
    for h in handles:
        lab = h.get_label()
        if not lab or lab == "_nolegend_":
            continue
        last_by_label[lab] = h

    # 2) 우선순위 사전
    prio = {lab: i for i, lab in enumerate(order)}

    # 3) 우선순위 핸들(존재하는 것만)
    ordered: List[Line2D] = []
    seen = set()
    for lab in order:
        h = last_by_label.get(lab)
        if h is not None:
            ordered.append(h)
            seen.add(lab)

    # 4) 나머지(원래 handles 등장 순서 유지)
    for h in handles:
        lab = h.get_label()
        if not lab or lab == "_nolegend_" or lab in seen:
            continue
        ordered.append(h)
        seen.add(lab)

    return ordered


# =========================
# 플로팅
# =========================
def apply_axes_style(ax, ylim: Optional[Tuple[float, float]]):
    ax.tick_params(labelsize=FONT_SIZES["ticks"])
    if SHOW_GRID:
        ax.grid(**GRID_STYLE)
    if ylim is not None:
        ax.set_ylim(*ylim)

def _series_color_and_lw(idx: int, name: str) -> Tuple[str, float]:
    color = color_from_cycle(idx)
    lw = DEFAULT_LINEWIDTH
    if name in SERIES_STYLE_OVERRIDES:
        if "color" in SERIES_STYLE_OVERRIDES[name]:
            color = SERIES_STYLE_OVERRIDES[name]["color"]  # type: ignore
        if "linewidth" in SERIES_STYLE_OVERRIDES[name]:
            lw = float(SERIES_STYLE_OVERRIDES[name]["linewidth"])  # type: ignore
    return color, lw

def _diagnose_and_plot(ax, s, color, lw):
    """
    문제 시리즈를 잡아내기 위한 안전 그리기 + 상세 로그
    - 길이 불일치 자동 보정
    - 로그 스케일에서 비양수 자동 처리(옵션)
    - NaN/inf 마스킹 후 잔여 개수 보고
    """
    ADD_EPS_FOR_LOG = True     # 로그 스케일에서 전부 ≤0이면 ε로 shift
    LOG_EPS = 1e-12

    # 1) 스무딩 적용
    stmp = apply_smoothing(s)
    ydraw = stmp.y_smooth if (stmp.y_smooth is not None) else s.y

    # 2) 길이 맞추기
    xarr = np.asarray(s.x, dtype=float)
    yarr = np.asarray(ydraw, dtype=float)
    n = min(len(xarr), len(yarr))
    if len(xarr) != len(yarr):
        print(f"[diag] '{s.name}': len(x)={len(xarr)}, len(y)={len(yarr)} → {n}로 맞춤")
    xarr = xarr[:n]
    yarr = yarr[:n]

    # 3) 클리핑
    if CLIP_Y_MIN is not None:
        yarr = np.maximum(yarr, CLIP_Y_MIN)
    if CLIP_Y_MAX is not None:
        yarr = np.minimum(yarr, CLIP_Y_MAX)

    # 4) 로그 스케일 보호
    use_log = (hasattr(ax, "get_yscale") and ax.get_yscale() == "log")
    if use_log:
        pos_mask = yarr > 0
        if not np.any(pos_mask) and ADD_EPS_FOR_LOG:
            shift = (np.nanmax(yarr) if np.isfinite(np.nanmax(yarr)) else 0.0)
            # 전부 ≤0이면 ε로 치환
            yarr = np.full_like(yarr, LOG_EPS if shift <= 0 else max(shift*1e-6, LOG_EPS))
            print(f"[diag] '{s.name}': log-scale 비양수만 존재 → ε로 치환하여 강제 표시")
        else:
            yarr = np.where(pos_mask, yarr, np.nan)

    # 5) 유효성 마스크
    finite_mask = np.isfinite(yarr)
    valid = np.count_nonzero(finite_mask)

    # 6) 요약 통계
    def _safe_minmax(v):
        vv = v[np.isfinite(v)]
        return (float(np.min(vv)), float(np.max(vv))) if vv.size else (np.nan, np.nan)
    ymin, ymax = _safe_minmax(yarr)

    print(f"[diag] '{s.name}': n={n}, finite={valid}, "
          f"min={ymin:.4g}, max={ymax:.4g}, "
          f"raw_minmax=({_safe_minmax(np.asarray(s.y, float))})")

    if valid == 0:
        print(f"[warn] '{s.name}': 유효 y가 없어 skip")
        return None

    # 7) 안전 그리기(마커도 찍어서 보이는지 확인)
    line, = ax.plot(xarr[finite_mask], yarr[finite_mask],
                    color=color, linewidth=lw, label=s.name)
    ax.scatter(xarr[finite_mask], yarr[finite_mask], s=6, alpha=0.35, label="_nolegend_")
    return line


def plot_all(series_list: List[Series], out_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    handles: List[Line2D] = []

    for i, s in enumerate(series_list):
        color, lw = _series_color_and_lw(i, s.name)

        # 원본 trace (희미)
        if SHOW_RAW_TRACE:
            
            ax.plot(s.x, s.y, color=color, linewidth=RAW_TRACE_LINEWIDTH, alpha=RAW_TRACE_ALPHA)

        # --- 문제 진단 + 안전 그리기 ---
        line = _diagnose_and_plot(ax, s, color, lw)
        if line is not None:
            handles.append(line)
        else:
            # 마지막 수단: y가 상수이면 아주 작은 jitter를 넣어 선을 보이게
            try:
                y = np.asarray(s.y, float)
                if np.isfinite(y).any() and (np.nanmax(y) == np.nanmin(y)):
                    x = np.asarray(s.x, float)
                    n = min(len(x), len(y))
                    if n > 1:
                        yj = y[:n] + (np.linspace(0, 1e-9, n))
                        line2, = ax.plot(x[:n], yj, color=color, linewidth=lw, label="_nolegend_")
                        handles.append(line2)
                        print(f"[diag] '{s.name}': 상수 시퀀스 → jitter로 강제 표시")
            except Exception as e:
                print(f"[diag] '{s.name}': fallback 실패 - {e}")

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
    # 교체:
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
# 메인
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
