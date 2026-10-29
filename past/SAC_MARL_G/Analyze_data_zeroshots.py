#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plot zero-shot evacuation time with run-level standard deviation.

Expected directory structure::

    ~/zero_shot_compare/
      3maps/
        Result_1500/
          Result_1500_Q/
            Result_1500_Q_0/metrics.txt
            Result_1500_Q_1/metrics.txt
            ...

Each training-map setting is compared separately for every evaluation map.
The line shows the mean and the shaded band shows sample standard deviation.
"""

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# User configuration
# =========================================================
ROOT_DIR = os.path.expanduser("~/zero_shot_compare")
SAVE_DIR = os.path.join(ROOT_DIR, "plots")

TRAINING_SET_ORDER = ["3maps", "30maps", "100maps", "300maps"]
MODEL_CODE = "Q"

METRIC_NAME = "evacuation_100_time"
SECONDS_PER_STEP = 0.5
MAX_TIMESTEP = 3000
STD_DDOF = 1

FIGSIZE = (18, 9)
SAVE_DPI = 300
SAVE_FORMAT = "png"
FONT_FAMILY = "DejaVu Serif"
FONT_SIZES = {"title": 25, "axes": 38, "ticks": 30, "legend": 24}

Y_LABEL = "Time(s)"
SHOW_TITLE = True
SHOW_GRID = True
SHOW_INDIVIDUAL_RUNS = False
SHOW_MEAN_VALUES = False
SHOW_LEGEND = False
Y_LIM_MIN: Optional[float] = 0.0
Y_LIM_MAX: Optional[float] = None

LINE_COLOR = "#3769f3"
LINE_WIDTH = 4.0
LINE_MARKER = "o"
LINE_MARKER_SIZE = 12
STD_BAND_ALPHA = 0.20
POINT_COLOR = "#202020"
POINT_ALPHA = 0.55
POINT_SIZE = 24

SUMMARY_CSV_NAME = "zero_shot_summary.csv"


TRAINING_SET_RE = re.compile(r"^(\d+)maps$")
MAP_DIR_RE = re.compile(r"^Result_(\d+)$")


@dataclass
class RunGroup:
    training_set: str
    evaluation_map: int
    values_s: List[float]

    @property
    def mean_s(self) -> float:
        return float(np.mean(self.values_s))

    @property
    def std_s(self) -> float:
        if len(self.values_s) <= STD_DDOF:
            return float("nan")
        return float(np.std(self.values_s, ddof=STD_DDOF))

    @property
    def timeout_count(self) -> int:
        timeout_s = MAX_TIMESTEP * SECONDS_PER_STEP
        return sum(value >= timeout_s for value in self.values_s)


def apply_plot_style() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = [FONT_FAMILY, "Times New Roman", "Times"]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["axes.titlesize"] = FONT_SIZES["title"]
    mpl.rcParams["axes.labelsize"] = FONT_SIZES["axes"]
    mpl.rcParams["xtick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["ytick.labelsize"] = FONT_SIZES["ticks"]
    mpl.rcParams["legend.fontsize"] = FONT_SIZES["legend"]


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def read_metric(path: str, metric_name: str = METRIC_NAME) -> Optional[float]:
    """Read one numeric key from a metrics.txt file."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                if "=" not in line:
                    continue
                key, raw_value = line.split("=", 1)
                if key.strip() != metric_name:
                    continue
                value = float(raw_value.strip())
                return value if np.isfinite(value) else None
    except (OSError, ValueError) as exc:
        print(f"[WARN] Cannot read {path}: {exc}")
    return None


def discover_training_sets(root: str) -> List[str]:
    discovered = []
    for entry in os.scandir(root):
        if entry.is_dir() and TRAINING_SET_RE.fullmatch(entry.name):
            discovered.append(entry.name)

    preferred = [name for name in TRAINING_SET_ORDER if name in discovered]
    remaining = sorted(
        (name for name in discovered if name not in TRAINING_SET_ORDER),
        key=natural_key,
    )
    return preferred + remaining


def collect_run_groups(root: str) -> List[RunGroup]:
    """Collect individual run metrics using the current nested folder layout."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root directory not found: {root}")

    groups: List[RunGroup] = []
    for training_set in discover_training_sets(root):
        training_path = os.path.join(root, training_set)
        map_entries = sorted(os.scandir(training_path), key=lambda entry: natural_key(entry.name))

        for map_entry in map_entries:
            match = MAP_DIR_RE.fullmatch(map_entry.name)
            if not map_entry.is_dir() or not match:
                continue

            evaluation_map = int(match.group(1))
            model_dir_name = f"Result_{evaluation_map}_{MODEL_CODE}"
            model_path = os.path.join(map_entry.path, model_dir_name)
            if not os.path.isdir(model_path):
                print(f"[WARN] Missing model directory: {model_path}")
                continue

            run_dir_re = re.compile(
                rf"^Result_{evaluation_map}_{re.escape(MODEL_CODE)}_(\d+)$"
            )
            values_s: List[float] = []
            run_entries = sorted(os.scandir(model_path), key=lambda entry: natural_key(entry.name))
            for run_entry in run_entries:
                if not run_entry.is_dir() or not run_dir_re.fullmatch(run_entry.name):
                    continue
                metric_steps = read_metric(os.path.join(run_entry.path, "metrics.txt"))
                if metric_steps is not None:
                    values_s.append(metric_steps * SECONDS_PER_STEP)

            if values_s:
                groups.append(
                    RunGroup(
                        training_set=training_set,
                        evaluation_map=evaluation_map,
                        values_s=values_s,
                    )
                )
            else:
                print(f"[WARN] No valid runs found under: {model_path}")

    return groups


def save_summary_csv(groups: List[RunGroup]) -> str:
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, SUMMARY_CSV_NAME)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "training_set",
                "evaluation_map",
                "n_runs",
                "mean_s",
                "std_s",
                "n_timeout",
            ],
        )
        writer.writeheader()
        for group in sorted(
            groups,
            key=lambda item: (item.evaluation_map, natural_key(item.training_set)),
        ):
            writer.writerow(
                {
                    "training_set": group.training_set,
                    "evaluation_map": group.evaluation_map,
                    "n_runs": len(group.values_s),
                    "mean_s": group.mean_s,
                    "std_s": group.std_s,
                    "n_timeout": group.timeout_count,
                }
            )
    print(f"[Saved] {path}")
    return path


def groups_by_evaluation_map(groups: List[RunGroup]) -> Dict[int, List[RunGroup]]:
    grouped: Dict[int, List[RunGroup]] = {}
    order_index = {name: index for index, name in enumerate(TRAINING_SET_ORDER)}
    for group in groups:
        grouped.setdefault(group.evaluation_map, []).append(group)
    for map_groups in grouped.values():
        map_groups.sort(
            key=lambda item: (
                order_index.get(item.training_set, len(order_index)),
                natural_key(item.training_set),
            )
        )
    return grouped


def plot_evaluation_map(evaluation_map: int, groups: List[RunGroup]) -> str:
    labels = [group.training_set.replace("maps", " maps") for group in groups]
    means = np.asarray([group.mean_s for group in groups], dtype=float)
    stds = np.asarray([group.std_s for group in groups], dtype=float)
    stds_for_plot = np.nan_to_num(stds, nan=0.0)
    x = np.arange(len(groups), dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    lower = means - stds_for_plot
    if Y_LIM_MIN is not None:
        lower = np.maximum(lower, Y_LIM_MIN)
    upper = means + stds_for_plot

    # 평균 ± 1 표준편차를 반투명 음영으로 표시한다.
    ax.fill_between(
        x,
        lower,
        upper,
        color=LINE_COLOR,
        alpha=STD_BAND_ALPHA,
        linewidth=0,
        label="Mean ± 1 SD",
        zorder=1,
    )
    ax.plot(
        x,
        means,
        color=LINE_COLOR,
        linewidth=LINE_WIDTH,
        marker=LINE_MARKER,
        markersize=LINE_MARKER_SIZE,
        markerfacecolor="white",
        markeredgecolor=LINE_COLOR,
        markeredgewidth=2.2,
        label="Mean",
        zorder=3,
    )

    if SHOW_INDIVIDUAL_RUNS:
        for index, group in enumerate(groups):
            offsets = np.linspace(-0.12, 0.12, len(group.values_s))
            ax.scatter(
                index + offsets,
                group.values_s,
                color=POINT_COLOR,
                alpha=POINT_ALPHA,
                s=POINT_SIZE,
                zorder=3,
            )

    if SHOW_MEAN_VALUES:
        for index, value in enumerate(means):
            ax.annotate(
                f"{value:.1f}",
                (index, value),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
            )

    if SHOW_TITLE:
        ax.set_title(f"Evaluation Map {evaluation_map}")
    ax.set_xlabel("Training Map Set")
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="both", labelsize=FONT_SIZES["ticks"])
    if SHOW_LEGEND:
        ax.legend(frameon=False, loc="best")

    if SHOW_GRID:
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    upper_candidates = [
        max(group.values_s + [group.mean_s + (0.0 if np.isnan(group.std_s) else group.std_s)])
        for group in groups
    ]
    data_upper = max(upper_candidates)
    ylim_top = Y_LIM_MAX if Y_LIM_MAX is not None else data_upper * 1.08
    ax.set_ylim(bottom=Y_LIM_MIN, top=ylim_top)

    fig.tight_layout()
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(
        SAVE_DIR,
        f"zero_shot_map_{evaluation_map}_mean_std.{SAVE_FORMAT}",
    )
    fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")
    return path


def main() -> None:
    apply_plot_style()
    groups = collect_run_groups(ROOT_DIR)
    if not groups:
        print(f"[WARN] No valid run data found under: {ROOT_DIR}")
        return

    print(f"[INFO] Loaded {sum(len(group.values_s) for group in groups)} runs")
    for group in groups:
        print(
            f"  {group.training_set}, map {group.evaluation_map}: "
            f"{group.mean_s:.2f} ± {group.std_s:.2f} s "
            f"(n={len(group.values_s)}, timeout={group.timeout_count})"
        )

    save_summary_csv(groups)
    for evaluation_map, map_groups in sorted(groups_by_evaluation_map(groups).items()):
        plot_evaluation_map(evaluation_map, map_groups)


if __name__ == "__main__":
    main()
