#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create one ablation-comparison graph for each training-map set.

Expected directory structure::

    ~/ablations3/
      3maps/{Ours,NotEgo,NotEpsilon,NotFiLM}/eva100s/{1,2,3}.txt
      30maps/{Ours,NotEgo,NotEpsilon,NotFiLM}/eva100s/{1,2,3}.txt
      100maps/{Ours,NotEgo,NotEpsilon,NotFiLM}/eva100s/{1,2,3}.txt
      300maps/{Ours,NotEgo,NotEpsilon,NotFiLM}/eva100s/{1,2,3}.txt

Each file contains one value per training episode. Independent runs are
aligned to their common length, smoothed separately, and then aggregated.
For each map set, the lines compare the four model variants. The line is the
episode-wise mean and the shaded region is the sample SD across runs.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# =========================================================
# User configuration
# =========================================================
ROOT_DIR = os.path.expanduser("~/ablations3")
OUT_DIR = os.path.join(ROOT_DIR, "plots")

MAP_SERIES = ["3maps", "30maps", "100maps", "300maps"]
MODEL_FOLDERS = ["Ours", "NotEgo", "NotEpsilon", "NotFiLM"]
GRAPH_TARGETS = ["eva100s"]  # 필요하면 "rewards" 추가

MAX_EPISODE: Optional[int] = 6000
STD_DDOF = 1                 # independent runs -> sample standard deviation
SHADE_MODE: Optional[str] = "std"  # "std" | "minmax" | "se" | None
SHADE_ALPHA = 0.20

TARGET_CONFIGS: Dict[str, Dict[str, object]] = {
    "eva100s": {
        "ylabel": "Time (s)",
        "scale": 0.5,       # 1 simulation step = 0.5 s
    },
    "rewards": {
        "ylabel": "Reward",
        "scale": 1.0,
    },
}

# y축은 mean과 shade 전체가 보이도록 그래프별로 자동 계산한다.
AUTO_Y_MARGIN = 0.05
X_TICK_INTERVAL: Optional[float] = 2000

SMOOTH_ENABLE = True
SMOOTH_KIND = "ma"           # "ma" | "ema"
SMOOTH_STRENGTH = 0.5        # 0.0 ~ 1.0
MA_WINDOW_MIN = 5
MA_WINDOW_MAX = 500
EMA_ALPHA_MIN = 0.05
EMA_ALPHA_MAX = 0.50

SHOW_RAW_MEAN = False
RAW_MEAN_ALPHA = 0.15
RAW_MEAN_LINEWIDTH = 1.0

MARKERS_ENABLE = True
MARKER_SIZE = 14
MARKER_EVERY: object = "auto"
MARKER_EDGEWIDTH = 1.2

SHOW_GRID = True
GRID_STYLE = {
    "color": "#AAAAAA",
    "linestyle": "--",
    "linewidth": 0.8,
    "alpha": 0.7,
}
SHOW_LEGEND = False
LEGEND_LOC = "best"

FIGSIZE_SINGLE = (15, 10)
FIGSIZE_COMBINED = (20, 8)
SAVE_DPI = 300
SAVE_FORMAT = "png"
SHOW_TITLE = False

FONT_FAMILY = "DejaVu Serif"
FONT_SIZES = {"title": 24, "axes": 42, "ticks": 42, "legend": 30}

SERIES_STYLES = {
    "Ours": {"color": "#3769f3", "ls": "-",  "lw": 4.5, "marker": "o"},
    "NotEgo": {"color": "#f04a4a", "ls": "--", "lw": 4.5, "marker": "s"},
    "NotEpsilon": {"color": "#e27227", "ls": "-.", "lw": 4.5, "marker": "^"},
    "NotFiLM": {"color": "#723e16", "ls": ":",  "lw": 4.5, "marker": "D"},
}


@dataclass
class AggregatedSeries:
    name: str
    x: np.ndarray
    raw_mean: np.ndarray
    mean: np.ndarray
    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    n_runs: int


def apply_plot_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY, "Times New Roman", "Times"]
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]
    mpl.rcParams["axes.unicode_minus"] = False


def natural_key(text: str):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def map_display_name(map_folder: str) -> str:
    return map_folder.replace("maps", " maps")


def read_numeric_series(path: str) -> np.ndarray:
    """Read whitespace- or comma-separated numeric values from one run."""
    values: List[float] = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                for token in re.split(r"[,\s]+", line.strip()):
                    if not token:
                        continue
                    try:
                        value = float(token)
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        values.append(value)
    except OSError as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
    return np.asarray(values, dtype=float)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1 or len(values) < 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="symmetric")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def smooth(values: np.ndarray) -> np.ndarray:
    if not SMOOTH_ENABLE:
        return values.copy()

    strength = float(np.clip(SMOOTH_STRENGTH, 0.0, 1.0))
    if SMOOTH_KIND.lower() == "ma":
        window = round(MA_WINDOW_MIN + strength * (MA_WINDOW_MAX - MA_WINDOW_MIN))
        return moving_average(values, window)
    if SMOOTH_KIND.lower() == "ema":
        alpha = EMA_ALPHA_MAX - strength * (EMA_ALPHA_MAX - EMA_ALPHA_MIN)
        return exponential_moving_average(values, alpha)

    raise ValueError(f"Unsupported SMOOTH_KIND: {SMOOTH_KIND}")


def discover_run_files(target_dir: str) -> List[str]:
    """Return every regular run file, including names such as '3txt'."""
    if not os.path.isdir(target_dir):
        return []
    return sorted(
        (
            entry.path
            for entry in os.scandir(target_dir)
            if entry.is_file() and not entry.name.startswith(".")
        ),
        key=lambda path: natural_key(os.path.basename(path)),
    )


def aggregate_runs(
    map_folder: str,
    model_folder: str,
    target: str,
) -> Optional[AggregatedSeries]:
    target_dir = os.path.join(ROOT_DIR, map_folder, model_folder, target)
    run_files = discover_run_files(target_dir)
    if not run_files:
        print(f"[WARN] No run files: {target_dir}")
        return None

    scale = float(TARGET_CONFIGS.get(target, {}).get("scale", 1.0))
    runs: List[np.ndarray] = []
    for path in run_files:
        values = read_numeric_series(path)
        if values.size:
            runs.append(values * scale)
        else:
            print(f"[WARN] Empty run ignored: {path}")

    if not runs:
        return None

    common_length = min(len(run) for run in runs)
    if MAX_EPISODE is not None:
        common_length = min(common_length, MAX_EPISODE)
    if common_length <= 0:
        return None

    aligned = np.stack([run[:common_length] for run in runs], axis=0)
    smoothed = np.stack([smooth(run) for run in aligned], axis=0)
    raw_mean = aligned.mean(axis=0)
    mean = smoothed.mean(axis=0)

    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    if SHADE_MODE is not None and len(runs) > 1:
        mode = SHADE_MODE.lower()
        if mode == "std":
            spread = smoothed.std(axis=0, ddof=STD_DDOF)
            lower, upper = mean - spread, mean + spread
        elif mode == "se":
            spread = smoothed.std(axis=0, ddof=STD_DDOF) / np.sqrt(len(runs))
            lower, upper = mean - spread, mean + spread
        elif mode == "minmax":
            lower, upper = smoothed.min(axis=0), smoothed.max(axis=0)
        else:
            raise ValueError(f"Unsupported SHADE_MODE: {SHADE_MODE}")

    return AggregatedSeries(
        name=model_folder,
        x=np.arange(1, common_length + 1, dtype=float),
        raw_mean=raw_mean,
        mean=mean,
        lower=lower,
        upper=upper,
        n_runs=len(runs),
    )


def marker_interval(length: int) -> int:
    if isinstance(MARKER_EVERY, int) and MARKER_EVERY > 0:
        return MARKER_EVERY
    return max(1, length // 15)


def set_auto_y_limits(ax, visible_values: List[np.ndarray]) -> None:
    """Set a separate y range that contains every line and shaded region."""
    finite_parts = [values[np.isfinite(values)] for values in visible_values]
    finite_parts = [values for values in finite_parts if values.size]
    if not finite_parts:
        return

    y_min = min(float(values.min()) for values in finite_parts)
    y_max = max(float(values.max()) for values in finite_parts)
    span = y_max - y_min
    if span <= 0:
        span = max(abs(y_min), 1.0)
    padding = span * AUTO_Y_MARGIN
    ax.set_ylim(y_min - padding, y_max + padding)


def draw_target(ax, series_list: List[AggregatedSeries], target: str) -> None:
    visible_values: List[np.ndarray] = []

    for series in series_list:
        style = SERIES_STYLES[series.name]

        if SHOW_RAW_MEAN:
            ax.plot(
                series.x,
                series.raw_mean,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=RAW_MEAN_LINEWIDTH,
                alpha=RAW_MEAN_ALPHA,
                zorder=1,
            )
            visible_values.append(series.raw_mean)

        if series.lower is not None and series.upper is not None:
            ax.fill_between(
                series.x,
                series.lower,
                series.upper,
                color=style["color"],
                alpha=SHADE_ALPHA,
                edgecolor="none",
                zorder=2,
            )
            visible_values.extend([series.lower, series.upper])

        marker_options = {}
        if MARKERS_ENABLE:
            marker_options = {
                "marker": style["marker"],
                "markersize": MARKER_SIZE,
                "markerfacecolor": style["color"],
                "markeredgecolor": style["color"],
                "markeredgewidth": MARKER_EDGEWIDTH,
                "markevery": marker_interval(len(series.x)),
            }

        ax.plot(
            series.x,
            series.mean,
            label=series.name,
            color=style["color"],
            linestyle=style["ls"],
            linewidth=style["lw"],
            zorder=3,
            **marker_options,
        )
        visible_values.append(series.mean)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel(str(TARGET_CONFIGS.get(target, {}).get("ylabel", "Value")))
    ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])

    if series_list:
        ax.set_xlim(0, max(series.x[-1] for series in series_list))
    if X_TICK_INTERVAL is not None:
        ax.xaxis.set_major_locator(MultipleLocator(X_TICK_INTERVAL))

    set_auto_y_limits(ax, visible_values)

    if SHOW_GRID:
        ax.set_axisbelow(True)
        ax.grid(**GRID_STYLE)
    if SHOW_LEGEND and series_list:
        ax.legend(frameon=False, loc=LEGEND_LOC)


def create_plot(
    map_folder: str,
    series_by_target: Dict[str, List[AggregatedSeries]],
) -> str:
    targets = [target for target in GRAPH_TARGETS if series_by_target.get(target)]
    if not targets:
        raise RuntimeError("No valid series were loaded")

    figsize = FIGSIZE_SINGLE if len(targets) == 1 else FIGSIZE_COMBINED
    fig, axes = plt.subplots(1, len(targets), figsize=figsize, squeeze=False)
    for ax, target in zip(axes[0], targets):
        draw_target(ax, series_by_target[target], target)
        if SHOW_TITLE:
            ax.set_title(f"{map_display_name(map_folder)} — {target}")

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = targets[0] if len(targets) == 1 else "combined"
    output_path = os.path.join(
        OUT_DIR,
        f"{map_folder}_{suffix}.{SAVE_FORMAT}",
    )
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {output_path}")
    return output_path


def main() -> None:
    apply_plot_style()
    saved_count = 0

    for map_folder in MAP_SERIES:
        series_by_target: Dict[str, List[AggregatedSeries]] = {}
        for target in GRAPH_TARGETS:
            target_series: List[AggregatedSeries] = []
            for model_folder in MODEL_FOLDERS:
                series = aggregate_runs(map_folder, model_folder, target)
                if series is None:
                    continue
                target_series.append(series)
                print(
                    f"[Loaded] {map_folder}/{model_folder}/{target}: "
                    f"runs={series.n_runs}, episodes={len(series.x)}"
                )
            if target_series:
                series_by_target[target] = target_series

        if not series_by_target:
            print(f"[WARN] No valid data for map set: {map_folder}")
            continue
        create_plot(map_folder, series_by_target)
        saved_count += 1

    if saved_count == 0:
        print(f"[WARN] No valid data found under: {ROOT_DIR}")
    else:
        print(f"[Done] Saved {saved_count} map-set graphs")


if __name__ == "__main__":
    main()
