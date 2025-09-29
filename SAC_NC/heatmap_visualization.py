#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# =========================
# 전역 파라미터
# =========================
IN_DIR   = os.path.expanduser("~/heatmaps")      # 입력 폴더
OUT_DIR  = os.path.expanduser("~/heatmaps")      # 출력 폴더

RECURSIVE      = True      # 하위 폴더까지 검색 여부
TRANSPOSE      = True      # imshow 전에 배열 전치
ORIGIN_LOWER   = True      # origin="lower" (False면 "upper")
CLIP_PERCENT   = 99.5      # 일반 스케일 vmax 클리핑 퍼센타일
DPI            = 220       # 저장 해상도

# ── 컬러맵 설정 ─────────────────────────────────────────
# 기본 컬러맵(Non-agent 파일)
CMAP              = "jet"          # 예: "Reds", "viridis", "jet"...
# agent_* 파일 전용 컬러맵(0=검정, >0는 흰→빨강)
CMAP_AGENT        = "viridis"    # "black_red" 또는 Matplotlib 내장 이름

# ── 제목/폰트/스타일 설정 ───────────────────────────────
TITLE_PREFIX      = ""             # 제목 앞에 붙일 문자열 (예: "Heatmap · ")
TITLE_FONT_FAMILY = "serif"  # 폰트 패밀리
TITLE_FONT_SIZE   = 14             # 제목 크기
TITLE_FONT_WEIGHT = "bold"         # 'normal' | 'bold'
TICK_FONT_SIZE    = 15              # 축 눈금 폰트 크기
CBAR_FONT_SIZE    = 15             # 컬러바 label 폰트 크기

# =========================
# 유틸
# =========================

def make_black_to_red_cmap() -> LinearSegmentedColormap:
    """0은 검정, 양수는 흰→빨강 그라디언트."""
    colors = [
        (0.0,  "#000000"),  # 0 → black
        (1e-6, "#ffffff"),  # 아주 작은 양의 값부터 흰색
        (1.0,  "#ff0000"),  # 최대 → pure red
    ]
    return LinearSegmentedColormap.from_list("black_to_red", colors)

def get_cmap(name_or_obj):
    """'black_red'면 커스텀 컬러맵 반환, 그 외는 그대로 사용."""
    if isinstance(name_or_obj, str) and name_or_obj.lower() == "black_red":
        return make_black_to_red_cmap()
    return name_or_obj

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def infer_vmax(arr: np.ndarray, clip_pct: float) -> float:
    """상위 퍼센타일을 vmax로 사용 (outlier 방지)"""
    if not np.isfinite(arr).any():
        return 1.0
    return float(np.nanpercentile(arr, clip_pct))

def set_text_styles(ax, title_text: str):
    """제목/틱/폰트 스타일 일괄 적용"""
    # ax.set_title(
    #     TITLE_PREFIX + title_text,
    #     fontfamily=TITLE_FONT_FAMILY,
    #     fontsize=TITLE_FONT_SIZE,
    #     fontweight=TITLE_FONT_WEIGHT,
    # )
    ax.tick_params(labelsize=TICK_FONT_SIZE)

def draw_colorbar(im, label: str):
    cb = plt.colorbar(im)
    cb.set_label(label, fontsize=CBAR_FONT_SIZE, fontfamily=TITLE_FONT_FAMILY)
    cb.ax.tick_params(labelsize=TICK_FONT_SIZE)

def cmap_for_filename(base_name: str):
    """파일명이 'agent'로 시작하면 CMAP_AGENT, 아니면 CMAP."""
    if base_name.lower().startswith("agent"):
        return get_cmap(CMAP_AGENT)
    return get_cmap(CMAP)

def plot_and_save(arr: np.ndarray, title: str, out_png: str, cmap_obj, log_scale: bool = False):
    """히트맵 저장 (일반 / 로그) + 제목/폰트 스타일 적용"""
    A = arr.T if TRANSPOSE else arr
    fig, ax = plt.subplots()

    if log_scale:
        # 0을 보존하면서 로그 스케일: +1 후 LogNorm 사용
        im = ax.imshow(
            A + 1,
            origin="lower" if ORIGIN_LOWER else "upper",
            interpolation="nearest",
            norm=LogNorm(),
            cmap=cmap_obj
        )
        set_text_styles(ax, title + " (log)")
        draw_colorbar(im, "log(visits+1)")
    else:
        vmax = infer_vmax(A, CLIP_PERCENT)
        im = ax.imshow(
            A,
            origin="lower" if ORIGIN_LOWER else "upper",
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
            cmap=cmap_obj
        )
        set_text_styles(ax, title)
        draw_colorbar(im, "visits")

    plt.tight_layout()
    plt.savefig(out_png, dpi=DPI)
    plt.close(fig)

# =========================
# 메인
# =========================
def main():
    ensure_dir(OUT_DIR)

    # npy 파일 수집
    npy_paths = []
    if RECURSIVE:
        for root, _, files in os.walk(IN_DIR):
            for f in files:
                if f.lower().endswith(".npy"):
                    npy_paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(IN_DIR):
            if f.lower().endswith(".npy"):
                npy_paths.append(os.path.join(IN_DIR, f))

    if not npy_paths:
        print(f"[WARN] .npy 파일이 없습니다: {IN_DIR}")
        return

    print(f"[INFO] 발견된 .npy 파일: {len(npy_paths)}개 (기본 cmap={CMAP}, agent cmap={CMAP_AGENT})")

    for path in npy_paths:
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as e:
            print(f"[SKIP] 로드 실패 {path}: {e}")
            continue

        if arr.ndim != 2:
            print(f"[SKIP] 2D 배열 아님 {path} (shape={arr.shape})")
            continue

        rel = os.path.relpath(path, IN_DIR)
        rel_base, _ = os.path.splitext(rel)
        out_base_dir = os.path.join(OUT_DIR, os.path.dirname(rel_base))
        ensure_dir(out_base_dir)

        base_name = os.path.basename(rel_base)  # 파일명(확장자 제외)
        cmap_obj = cmap_for_filename(base_name) # 파일명에 따른 컬러맵 선택

        out_png_norm = os.path.join(out_base_dir, base_name + ".png")
        out_png_log  = os.path.join(out_base_dir, base_name + "_log.png")

        plot_and_save(arr, base_name, out_png_norm, cmap_obj, log_scale=False)
        plot_and_save(arr, base_name, out_png_log,  cmap_obj, log_scale=True)

        print(f"[OK] {path} -> {out_png_norm}, {out_png_log}")

    print(f"[DONE] 출력 폴더: {OUT_DIR}")

if __name__ == "__main__":
    main()
