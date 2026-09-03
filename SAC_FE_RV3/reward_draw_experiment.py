#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run reward-component experiments, aggregate them, and draw a graph.

The reward calculation follows ``ADDS_AS_reinforcement.worker_process``:

* collision rewards are accumulated between ACTION_SCALE boundaries;
* shaping rewards are emitted at the same boundaries;
* a terminal reward is emitted when the game finishes after ACTION_SCALE;
* reward_f is logged, but is not included in total_reward because the current
  training code also omits it from the total.

Each episode is written as a tab-separated TXT file. Episodes of different
lengths can either retain the original reward-event alignment or be divided
into equal normalized-progress bins. In the normalized mode, every episode is
averaged within a bin before the cross-episode mean is calculated, giving short
and long episodes equal weight.
"""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
import os
import random
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

# Some headless experiment servers have a read-only ~/.config.  Keeping the
# Matplotlib cache in /tmp also prevents every spawned worker from warning.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "reward_draw_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

import config as cfg


# ---------------------------------------------------------------------------
# Settings that can be edited directly (command-line options override these).
# ---------------------------------------------------------------------------
MAP_LIST = [105]
NUMBER_OF_AGENTS = 30
EPISODES_PER_MAP = 10
MAX_STEP_NUM = 3000
ROBOT_VERSION = "Q"  # T: rule policy, Q: learned policy
ROBOT_LEARNED_MODEL = "3maps.pth"  # resolved by the existing SAC loader
MAX_WORKERS = 3  # Q models can use a lot of memory; increase with care.
BASE_SEED = 2025
TIME_PER_STEP_SECONDS = 0.5

SCRIPT_DIR = Path(__file__).resolve().parent

# 결과는 코드 폴더가 아니라 홈 디렉터리 아래 전용 폴더에 저장됩니다.
RESULT_FOLDER_NAME = "Reward_draw_experiment_results2"
DEFAULT_OUTPUT_DIR = Path.home() / RESULT_FOLDER_NAME


# ---------------------------------------------------------------------------
# Graph style settings (edit these values to change every generated graph).
# ---------------------------------------------------------------------------
# "combined": total reward와 개별 reward를 하나의 축에 함께 표시
# "separate": 기존처럼 위/아래의 두 축으로 분리
GRAPH_LAYOUT = "separate"

# Episode alignment for the x-axis:
# - "absolute_time": keep the existing reward-event alignment and plot mean time.
#   Episodes that have already finished no longer contribute to later points.
# - "normalized_progress": divide every episode into equally sized 0--100%
#   progress bins. Each episode is averaged within a bin first, so short and
#   long episodes have equal weight throughout the graph.
EPISODE_ALIGNMENT_MODE = "normalized_progress"
NORMALIZED_PROGRESS_BINS = 20
YLABEL_COMBINED = "Mean Weighted Reward"
YLABEL_COMPONENT = "Mean Weighted Reward Component"
YLABEL_STEP_REWARD = "Mean Reward per Decision Interval"

FIGURE_SIZE = (14, 10)
# 위쪽 개별 reward 패널과 아래쪽 total reward 패널의 상대 높이.
COMPONENT_PANEL_HEIGHT = 2
TOTAL_PANEL_HEIGHT = 1.15
# separate 레이아웃의 위/아래 패널 간격. 0에 가까울수록 서로 붙습니다.
SUBPLOT_VERTICAL_SPACE = 0.06
# step reward의 표준편차 영역이 위/아래 경계에 붙지 않도록 주는 여백.
TOTAL_PANEL_Y_MARGIN = 0.08
# Episode 간 평균 ± 1 표준편차 음영을 표시할지 여부.
SHOW_COMPONENT_STD = True
SHOW_TOTAL_STD = True
SAVE_DPI = 300

# Times New Roman이 설치되어 있지 않으면 모양이 유사한 Nimbus Roman 사용.
FONT_FAMILY = "Times New Roman"
FONT_FALLBACK = "Nimbus Roman"
FONT_SIZE_DEFAULT = 30
FONT_SIZE_TITLE = 22
FONT_SIZE_AXIS_LABEL = 18
FONT_SIZE_TICK = 30
FONT_SIZE_LEGEND = 13

COMPONENT_LINE_WIDTH = 4.0
TOTAL_LINE_WIDTH = 4
ZERO_LINE_WIDTH = 0.9
GRID_LINE_WIDTH = 0.8
MARKER_SIZE = 10
MARKER_EDGE_WIDTH = 1.2
MAX_MARKERS_PER_LINE = 25
TOTAL_MARKER = "o"
LINE_MARKERS = (
    "o",  # circle
    "^",  # triangle up s
    "s",  # square d
    "D",  # diamond h
    "v",  # triangle down
    "h",  # hexagon
    "X",  # filled x
    "<",  # triangle left
    ">",  # triangle right
    "h",  # hexagon
    "p",  # pentagon
    "*",  # star
)
COLOR_MAP = "tab10"
# "auto": 각 선의 색으로 채움, "white": 흰색 내부, "none": 투명 내부
MARKER_FACE_COLOR = "auto"
COMPONENT_STD_BAND_ALPHA = 0.10
STD_BAND_ALPHA = 0.4
GRID_ALPHA = 0.30
LEGEND_COLUMNS = 2

# 선이 겹칠 때 그려지는 앞뒤 순서입니다. 숫자가 작을수록 뒤쪽,
# 클수록 앞쪽에 표시됩니다. 뒤로 보내고 싶은 reward의 숫자를 낮추세요.
DEFAULT_LINE_ZORDER = 10
LINE_ZORDER = {
    "reward_a_alive": 10,
    "reward_b_danger": 10,
    "reward_c_gain": 10,
    "reward_d_penalty": 11,
    "reward_e_evacuated_with_robot": 10,
    "reward_f_nearest_distance": 10,
    "reward_g_nearest_distance_gain": 10,
    "reward_h_gain_time_bonus": 10,
    "reward_i_alive_root": 10,
    "reward_j_danger_root": 10,
    "reward_k_collision": 4,
    "reward_l_farthest_distance": 10,
    "reward_fixed": 5,
    "finished_bonus": 10,
    "total_reward": 20,
}
STD_BAND_ZORDER = 2
ZERO_LINE_ZORDER = 1


COMPONENT_KEYS = (
    "reward_a_alive",
    "reward_b_danger",
    "reward_c_gain",
    "reward_d_penalty",
    "reward_e_evacuated_with_robot",
    "reward_f_nearest_distance",
    "reward_g_nearest_distance_gain",
    "reward_h_gain_time_bonus",
    "reward_i_alive_root",
    "reward_j_danger_root",
    "reward_k_collision",
    "reward_l_farthest_distance",
    "reward_fixed",
    "finished_bonus",
)

# Keep this identical to the sum in ADDS_AS_reinforcement.py.  reward_f is
# calculated and logged there, but is currently absent from the training sum.
TOTAL_INCLUDED_KEYS = (
    "reward_a_alive",
    "reward_b_danger",
    "reward_c_gain",
    "reward_d_penalty",
    "reward_e_evacuated_with_robot",
    "reward_g_nearest_distance_gain",
    "reward_h_gain_time_bonus",
    "reward_i_alive_root",
    "reward_j_danger_root",
    "reward_k_collision",
    "reward_l_farthest_distance",
    "reward_fixed",
    "finished_bonus",
)

EPISODE_COLUMNS = (
    "episode",
    "map_id",
    "reward_step",
    "sim_step",
    "sim_time_seconds",
    "alive_agents",
    *COMPONENT_KEYS,
    "total_reward",
    "cumulative_total_reward",
)

SERIES_KEYS = (
    "alive_agents",
    *COMPONENT_KEYS,
    "total_reward",
    "cumulative_total_reward",
)

DISPLAY_NAMES = {
    "reward_a_alive": "A: alive",
    "reward_b_danger": "B: danger",
    "reward_c_gain": "C: gain",
    "reward_d_penalty": "D: penalty",
    "reward_e_evacuated_with_robot": "E: evacuated with robot",
    "reward_f_nearest_distance": "F: nearest distance",
    "reward_g_nearest_distance_gain": "G: distance gain",
    "reward_h_gain_time_bonus": "H: gain + time bonus",
    "reward_i_alive_root": "I: alive root",
    "reward_j_danger_root": "J: danger root",
    "reward_k_collision": "K: collision",
    "reward_l_farthest_distance": "L: farthest distance",
    "reward_fixed": "fixed",
    "finished_bonus": "finished bonus",
    "total_reward": "total reward",
}


def _float_or_zero(value) -> float:
    """Reward helpers occasionally return None; treat that as zero."""
    return 0.0 if value is None else float(value)


def _weighted(weight: float, reward_fn: Callable[[], float]) -> float:
    if not weight:
        return 0.0
    return _float_or_zero(reward_fn()) * float(weight)


def calculate_shaping_components(env, collision_reward: float, zero_step: int,
                                 max_steps: int) -> dict[str, float]:
    """Calculate the weighted components at one training reward boundary."""
    values = {
        "reward_a_alive": _weighted(cfg.REWARD_A, env.reward_based_alived),
        "reward_b_danger": _weighted(
            cfg.REWARD_B, env.reward_based_all_agents_danger
        ),
        "reward_c_gain": _weighted(cfg.REWARD_C, env.reward_based_gain),
        "reward_d_penalty": _weighted(cfg.REWARD_D, env.reward_penalty),
        "reward_e_evacuated_with_robot": _weighted(
            cfg.REWARD_E, env.reward_based_evacuated_with_robot
        ),
        "reward_f_nearest_distance": _weighted(
            cfg.REWARD_F, env.reward_based_distance_from_near_agents
        ),
        "reward_g_nearest_distance_gain": _weighted(
            cfg.REWARD_G, env.reward_based_distance_from_near_agent_gain
        ),
        "reward_h_gain_time_bonus": _weighted(
            cfg.REWARD_H, env.reward_based_gain_with_time_bonus
        ),
        "reward_i_alive_root": _weighted(
            cfg.REWARD_I, env.reward_based_alived_root
        ),
        "reward_j_danger_root": _weighted(
            cfg.REWARD_J, env.reward_based_all_agents_danger_root
        ),
        "reward_k_collision": float(collision_reward),
        "reward_l_farthest_distance": _weighted(
            cfg.REWARD_L, env.reward_based_farthest_agent_distance
        ),
        "reward_fixed": float(cfg.REWARD_FIXED),
        "finished_bonus": 0.0,
    }
    if env.robot.is_game_finished:
        values["finished_bonus"] = float(cfg.FINISHED_BONUS) * (
            1.0 - zero_step / float(max_steps)
        )
    return values


def _episode_file(map_dir: Path, episode: int) -> Path:
    return map_dir / "episodes" / f"episode_{episode:04d}_rewards.txt"


def _write_tsv_atomic(path: Path, columns: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temp_path, path)


def _release_training_only_objects(env) -> None:
    """Match ADDS_AS_experiment.py's memory reduction for policy evaluation."""
    if not hasattr(env, "sac_agent"):
        return
    for attr in (
        "replay_buffer",
        "q1",
        "q2",
        "q1_target",
        "q2_target",
        "q1_optimizer",
        "q2_optimizer",
        "policy_optimizer",
        "alpha_optimizer",
        "log_alpha",
    ):
        if hasattr(env.sac_agent, attr):
            setattr(env.sac_agent, attr, None)
    gc.collect()


def run_one_episode(
    map_id: int,
    episode: int,
    output_file: str,
    number_of_agents: int,
    max_steps: int,
    robot_version: str,
    learned_model: str,
    seed: int,
) -> dict:
    """Run and atomically save one episode.  Safe to call in a worker."""
    # FightingModel loads generated maps from the relative ``map_infos`` path.
    # This also makes running the script from the repository root reliable.
    os.chdir(SCRIPT_DIR)

    import torch
    import model

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = None
    rows: list[dict] = []
    collision_accumulator = 0.0
    cumulative_total = 0.0

    try:
        env = model.FightingModel(
            number_of_agents,
            width=cfg.MAP_W,
            height=cfg.MAP_H,
            model_num=map_id,
            robot=robot_version,
        )
        if robot_version == "Q":
            env.use_model(learned_model)
            env.sac_agent.policy.eval()
            _release_training_only_objects(env)

        for zero_step in range(max_steps):
            env.step()
            sim_step = zero_step + 1
            alive = int(env.alived_agents())

            if cfg.REWARD_K:
                collision_accumulator += (
                    _float_or_zero(env.reward_penalty_collision())
                    * float(cfg.REWARD_K)
                )

            finished = bool(env.robot.is_game_finished)
            reward_boundary = (
                zero_step % cfg.ACTION_SCALE == cfg.ACTION_SCALE - 1
                and zero_step > cfg.ACTION_SCALE
            )
            terminal_boundary = finished and zero_step > cfg.ACTION_SCALE

            if reward_boundary or terminal_boundary:
                values = calculate_shaping_components(
                    env,
                    collision_reward=collision_accumulator,
                    zero_step=zero_step,
                    max_steps=max_steps,
                )
                total = float(sum(values[key] for key in TOTAL_INCLUDED_KEYS))
                cumulative_total += total
                reward_step = len(rows) + 1
                row = {
                    "episode": episode,
                    "map_id": map_id,
                    "reward_step": reward_step,
                    "sim_step": sim_step,
                    "sim_time_seconds": sim_step * TIME_PER_STEP_SECONDS,
                    "alive_agents": alive,
                    **values,
                    "total_reward": total,
                    "cumulative_total_reward": cumulative_total,
                }
                rows.append(row)
                collision_accumulator = 0.0

            if alive < 1 or sim_step >= max_steps:
                break

        _write_tsv_atomic(Path(output_file), EPISODE_COLUMNS, rows)
        return {
            "ok": True,
            "map_id": map_id,
            "episode": episode,
            "events": len(rows),
            "sim_steps": sim_step,
        }
    except Exception as exc:
        return {
            "ok": False,
            "map_id": map_id,
            "episode": episode,
            "error": repr(exc),
        }
    finally:
        if env is not None:
            del env
        gc.collect()


def read_episode_rewards(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not set(EPISODE_COLUMNS).issubset(reader.fieldnames):
                return []
            for raw in reader:
                rows.append({key: float(raw[key]) for key in EPISODE_COLUMNS})
    except (OSError, TypeError, ValueError):
        return []
    return rows


def is_complete_episode_file(path: Path) -> bool:
    """A valid header is enough; very short episodes may have no reward event."""
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader)
        return set(EPISODE_COLUMNS).issubset(header)
    except (OSError, StopIteration):
        return False


def aggregate_episode_rewards(
    episode_logs: list[list[dict[str, float]]],
) -> list[dict[str, float]]:
    """Average matching reward events among episodes that reached each event."""
    nonempty = [log for log in episode_logs if log]
    if not nonempty:
        return []

    output: list[dict[str, float]] = []
    max_events = max(len(log) for log in nonempty)
    for event_index in range(max_events):
        bucket = [log[event_index] for log in nonempty if len(log) > event_index]
        row: dict[str, float] = {
            "reward_step": event_index + 1,
            "sim_step_mean": float(np.mean([item["sim_step"] for item in bucket])),
            "sim_time_seconds_mean": float(
                np.mean([item["sim_time_seconds"] for item in bucket])
            ),
            "count": len(bucket),
        }
        for key in SERIES_KEYS:
            values = np.asarray([item[key] for item in bucket], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=0))
        output.append(row)
    return output


def aggregate_episode_rewards_by_progress(
    episode_logs: list[list[dict[str, float]]],
    bin_count: int,
) -> list[dict[str, float]]:
    """Average progress bins with equal weight for every episode.

    Values are first averaged within each episode/bin and only then averaged
    across episodes. This prevents longer episodes, which contain more reward
    events, from receiving more weight than shorter episodes.
    """
    if bin_count < 1:
        raise ValueError("bin_count must be at least 1")

    nonempty = [log for log in episode_logs if log]
    if not nonempty:
        return []

    episode_bins: list[list[dict[str, float] | None]] = []
    valid_logs: list[list[dict[str, float]]] = []
    for log in nonempty:
        final_time = float(log[-1]["sim_time_seconds"])
        if final_time <= 0.0:
            continue
        valid_logs.append(log)

        buckets: list[list[dict[str, float]]] = [[] for _ in range(bin_count)]
        # Keep the exact terminal event separate. Otherwise the point nearest
        # 100% represents the whole final bin rather than episode completion.
        for item in log[:-1]:
            progress = np.clip(
                float(item["sim_time_seconds"]) / final_time, 0.0, 1.0
            )
            bin_index = min(int(progress * bin_count), bin_count - 1)
            buckets[bin_index].append(item)

        per_episode: list[dict[str, float] | None] = []
        for bucket in buckets:
            if not bucket:
                per_episode.append(None)
                continue
            summary = {
                "sim_time_seconds": float(
                    np.mean([item["sim_time_seconds"] for item in bucket])
                )
            }
            for key in SERIES_KEYS:
                summary[key] = float(np.mean([item[key] for item in bucket]))
            per_episode.append(summary)
        episode_bins.append(per_episode)

    output: list[dict[str, float]] = []
    for bin_index in range(bin_count):
        bucket = [
            per_episode[bin_index]
            for per_episode in episode_bins
            if per_episode[bin_index] is not None
        ]
        if not bucket:
            continue

        row: dict[str, float] = {
            "progress_bin": bin_index + 1,
            "progress_percent": (bin_index + 0.5) * 100.0 / bin_count,
            "sim_time_seconds_mean": float(
                np.mean([item["sim_time_seconds"] for item in bucket])
            ),
            "count": len(bucket),
        }
        for key in SERIES_KEYS:
            values = np.asarray([item[key] for item in bucket], dtype=np.float64)
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=0))
        output.append(row)

    if valid_logs:
        terminal_items = [log[-1] for log in valid_logs]
        terminal_row: dict[str, float] = {
            "progress_bin": bin_count + 1,
            "progress_percent": 100.0,
            "sim_time_seconds_mean": float(
                np.mean([item["sim_time_seconds"] for item in terminal_items])
            ),
            "count": len(terminal_items),
        }
        for key in SERIES_KEYS:
            values = np.asarray(
                [item[key] for item in terminal_items], dtype=np.float64
            )
            terminal_row[f"{key}_mean"] = float(np.mean(values))
            terminal_row[f"{key}_std"] = float(np.std(values, ddof=0))
        output.append(terminal_row)
    return output


def mean_columns() -> list[str]:
    columns = [
        "reward_step",
        "sim_step_mean",
        "sim_time_seconds_mean",
        "count",
    ]
    for key in SERIES_KEYS:
        columns.extend((f"{key}_mean", f"{key}_std"))
    return columns


def progress_mean_columns() -> list[str]:
    columns = [
        "progress_bin",
        "progress_percent",
        "sim_time_seconds_mean",
        "count",
    ]
    for key in SERIES_KEYS:
        columns.extend((f"{key}_mean", f"{key}_std"))
    return columns


def write_mean_rewards(path: Path, rows: list[dict[str, float]]) -> None:
    _write_tsv_atomic(path, mean_columns(), rows)


def write_progress_mean_rewards(path: Path, rows: list[dict[str, float]]) -> None:
    _write_tsv_atomic(path, progress_mean_columns(), rows)


def read_mean_rewards(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = set(mean_columns())
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return []
            for raw in reader:
                rows.append({key: float(raw[key]) for key in mean_columns()})
    except (OSError, TypeError, ValueError):
        return []
    return rows


def read_progress_mean_rewards(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = set(progress_mean_columns())
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                return []
            for raw in reader:
                rows.append(
                    {key: float(raw[key]) for key in progress_mean_columns()}
                )
    except (OSError, TypeError, ValueError):
        return []
    return rows


def enabled_component_keys() -> list[str]:
    weights = {
        "reward_a_alive": cfg.REWARD_A,
        "reward_b_danger": cfg.REWARD_B,
        "reward_c_gain": cfg.REWARD_C,
        "reward_d_penalty": cfg.REWARD_D,
        "reward_e_evacuated_with_robot": cfg.REWARD_E,
        "reward_f_nearest_distance": cfg.REWARD_F,
        "reward_g_nearest_distance_gain": cfg.REWARD_G,
        "reward_h_gain_time_bonus": cfg.REWARD_H,
        "reward_i_alive_root": cfg.REWARD_I,
        "reward_j_danger_root": cfg.REWARD_J,
        "reward_k_collision": cfg.REWARD_K,
        "reward_l_farthest_distance": cfg.REWARD_L,
        "reward_fixed": cfg.REWARD_FIXED,
        "finished_bonus": cfg.FINISHED_BONUS,
    }
    return [key for key in COMPONENT_KEYS if weights[key] != 0]


def apply_graph_style() -> str:
    """Apply the user-editable graph settings and return the selected font."""
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if FONT_FAMILY in installed_fonts:
        selected_font = FONT_FAMILY
    else:
        selected_font = FONT_FALLBACK
        print(
            f"[FONT] '{FONT_FAMILY}'이 설치되어 있지 않아 "
            f"'{FONT_FALLBACK}'을 사용합니다."
        )

    matplotlib.rcParams.update(
        {
            "font.family": selected_font,
            "font.size": FONT_SIZE_DEFAULT,
            "axes.titlesize": FONT_SIZE_TITLE,
            "axes.labelsize": FONT_SIZE_AXIS_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
        }
    )
    return selected_font


def draw_reward_graph(
    mean_txt: Path,
    output_png: Path,
    map_id: int,
    alignment_mode: str = EPISODE_ALIGNMENT_MODE,
) -> bool:
    """Draw the graph only when a valid mean TXT exists."""
    if alignment_mode == "normalized_progress":
        rows = read_progress_mean_rewards(mean_txt)
        x = np.asarray([row["progress_percent"] for row in rows], dtype=np.float64)
        x_label = "Normalized episode progress (%)"
        title_suffix = "over normalized episode progress"
    elif alignment_mode == "absolute_time":
        rows = read_mean_rewards(mean_txt)
        x = np.asarray(
            [row["sim_time_seconds_mean"] for row in rows], dtype=np.float64
        )
        x_label = "Mean simulation time (s)"
        title_suffix = "during an episode"
    else:
        raise ValueError(f"Unsupported episode alignment mode: {alignment_mode}")

    if not rows:
        print(f"[WARN] 평균 TXT가 없거나 비어 있어 그래프를 건너뜁니다: {mean_txt}")
        return False

    apply_graph_style()
    if GRAPH_LAYOUT == "combined":
        fig, combined_ax = plt.subplots(figsize=FIGURE_SIZE)
        ax_components = combined_ax
        ax_total = combined_ax
    elif GRAPH_LAYOUT == "separate":
        fig, (ax_components, ax_total) = plt.subplots(
            2,
            1,
            figsize=FIGURE_SIZE,
            sharex=True,
            gridspec_kw={
                "height_ratios": (COMPONENT_PANEL_HEIGHT, TOTAL_PANEL_HEIGHT)
            },
        )
    else:
        raise ValueError(
            f"GRAPH_LAYOUT은 'combined' 또는 'separate'여야 합니다: {GRAPH_LAYOUT}"
        )

    colors = plt.get_cmap(COLOR_MAP)
    mark_every = max(1, len(x) // max(1, MAX_MARKERS_PER_LINE))
    for index, key in enumerate(enabled_component_keys()):
        y = np.asarray([row[f"{key}_mean"] for row in rows], dtype=np.float64)
        y_std = np.asarray(
            [row[f"{key}_std"] for row in rows], dtype=np.float64
        )
        color = colors(index % 10)
        if SHOW_COMPONENT_STD:
            ax_components.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=COMPONENT_STD_BAND_ALPHA,
                linewidth=0,
                label="±1 SD" if index == 0 else "_nolegend_",
                zorder=STD_BAND_ZORDER,
            )
        ax_components.plot(
            x,
            y,
            label=DISPLAY_NAMES[key],
            color=color,
            linewidth=COMPONENT_LINE_WIDTH,
            marker=LINE_MARKERS[index % len(LINE_MARKERS)],
            markersize=MARKER_SIZE,
            markerfacecolor=MARKER_FACE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markevery=mark_every,
            zorder=LINE_ZORDER.get(key, DEFAULT_LINE_ZORDER),
        )

    total_mean = np.asarray(
        [row["total_reward_mean"] for row in rows], dtype=np.float64
    )
    total_std = np.asarray(
        [row["total_reward_std"] for row in rows], dtype=np.float64
    )
    ax_total.plot(
        x,
        total_mean,
        color="#111111",
        linewidth=TOTAL_LINE_WIDTH,
        marker=TOTAL_MARKER,
        markersize=MARKER_SIZE,
        markerfacecolor=MARKER_FACE_COLOR,
        markeredgewidth=MARKER_EDGE_WIDTH,
        markevery=mark_every,
        label="mean step reward",
        zorder=LINE_ZORDER.get("total_reward", DEFAULT_LINE_ZORDER),
    )
    if SHOW_TOTAL_STD:
        ax_total.fill_between(
            x,
            total_mean - total_std,
            total_mean + total_std,
            color="#777777",
            alpha=STD_BAND_ALPHA,
            label="±1 SD",
            zorder=STD_BAND_ZORDER,
        )

    if GRAPH_LAYOUT == "combined":
        ax_components.axhline(
            0.0,
            color="#777777",
            linewidth=ZERO_LINE_WIDTH,
            zorder=ZERO_LINE_ZORDER,
        )
        ax_components.set_title(
            f"Reward components and step reward {title_suffix} (map {map_id})"
        )
        ax_components.set_ylabel(YLABEL_COMBINED)
        ax_components.set_xlabel(x_label)
        ax_components.set_axisbelow(True)
        ax_components.grid(
            True, linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA
        )
        ax_components.legend(loc="best", ncol=LEGEND_COLUMNS)
    else:
        ax_components.axhline(
            0.0,
            color="#777777",
            linewidth=ZERO_LINE_WIDTH,
            zorder=ZERO_LINE_ZORDER,
        )
        ax_total.axhline(
            0.0,
            color="#777777",
            linewidth=ZERO_LINE_WIDTH,
            zorder=ZERO_LINE_ZORDER,
        )
        ax_components.set_title(
            f"Reward components {title_suffix} (map {map_id})"
        )
        ax_components.set_ylabel(YLABEL_COMPONENT)
        ax_total.set_ylabel(YLABEL_STEP_REWARD)
        ax_total.set_xlabel(x_label)
        ax_total.margins(y=TOTAL_PANEL_Y_MARGIN)
        ax_components.set_axisbelow(True)
        ax_total.set_axisbelow(True)
        ax_components.grid(
            True, linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA
        )
        ax_total.grid(
            True, linestyle="--", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA
        )
        ax_components.legend(loc="best", ncol=LEGEND_COLUMNS)
        ax_total.legend(loc="best")
    fig.tight_layout()
    if GRAPH_LAYOUT == "separate":
        fig.subplots_adjust(hspace=SUBPLOT_VERTICAL_SPACE)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[GRAPH] {output_png}")
    return True


def write_run_config(path: Path, args: argparse.Namespace, completed: int) -> None:
    lines = [
        f"map_id={args.current_map_id}",
        f"robot_version={args.robot_version}",
        f"learned_model={args.model}",
        f"number_of_agents={args.agents}",
        f"requested_episodes={args.episodes}",
        f"completed_episodes={completed}",
        f"max_steps={args.max_steps}",
        f"time_per_step_seconds={TIME_PER_STEP_SECONDS}",
        f"action_scale={cfg.ACTION_SCALE}",
        f"font_family={FONT_FAMILY}",
        f"font_fallback={FONT_FALLBACK}",
        f"graph_layout={GRAPH_LAYOUT}",
        f"episode_alignment_mode={EPISODE_ALIGNMENT_MODE}",
        f"normalized_progress_bins={NORMALIZED_PROGRESS_BINS}",
        f"subplot_vertical_space={SUBPLOT_VERTICAL_SPACE}",
        f"component_panel_height={COMPONENT_PANEL_HEIGHT}",
        f"total_panel_height={TOTAL_PANEL_HEIGHT}",
        f"total_panel_y_margin={TOTAL_PANEL_Y_MARGIN}",
        f"show_component_std={SHOW_COMPONENT_STD}",
        f"show_total_std={SHOW_TOTAL_STD}",
        f"component_std_band_alpha={COMPONENT_STD_BAND_ALPHA}",
        f"total_std_band_alpha={STD_BAND_ALPHA}",
        f"component_line_width={COMPONENT_LINE_WIDTH}",
        f"total_line_width={TOTAL_LINE_WIDTH}",
        f"marker_size={MARKER_SIZE}",
        f"save_dpi={SAVE_DPI}",
        f"line_zorder={LINE_ZORDER}",
        f"std_band_zorder={STD_BAND_ZORDER}",
    ]
    for key, value in (
        ("REWARD_A", cfg.REWARD_A),
        ("REWARD_B", cfg.REWARD_B),
        ("REWARD_C", cfg.REWARD_C),
        ("REWARD_D", cfg.REWARD_D),
        ("REWARD_E", cfg.REWARD_E),
        ("REWARD_F", cfg.REWARD_F),
        ("REWARD_G", cfg.REWARD_G),
        ("REWARD_H", cfg.REWARD_H),
        ("REWARD_I", cfg.REWARD_I),
        ("REWARD_J", cfg.REWARD_J),
        ("REWARD_K", cfg.REWARD_K),
        ("REWARD_L", cfg.REWARD_L),
        ("REWARD_FIXED", cfg.REWARD_FIXED),
        ("FINISHED_BONUS", cfg.FINISHED_BONUS),
    ):
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _task_args(args: argparse.Namespace, map_id: int, episode: int, path: Path) -> tuple:
    return (
        map_id,
        episode,
        str(path),
        args.agents,
        args.max_steps,
        args.robot_version,
        args.model,
        args.seed + map_id * 100_000 + episode,
    )


def run_pending_episodes(
    args: argparse.Namespace, map_id: int, map_dir: Path, pending: list[int]
) -> None:
    if not pending:
        return

    if args.workers == 1:
        for episode in pending:
            result = run_one_episode(
                *_task_args(args, map_id, episode, _episode_file(map_dir, episode))
            )
            _print_episode_result(result)
        return

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_one_episode,
                *_task_args(args, map_id, episode, _episode_file(map_dir, episode)),
            ): episode
            for episode in pending
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ok": False,
                    "map_id": map_id,
                    "episode": episode,
                    "error": repr(exc),
                }
            _print_episode_result(result)


def _print_episode_result(result: dict) -> None:
    if result["ok"]:
        print(
            f"[OK] map={result['map_id']} episode={result['episode']} "
            f"sim_steps={result['sim_steps']} reward_events={result['events']}",
            flush=True,
        )
    else:
        print(
            f"[ERROR] map={result['map_id']} episode={result['episode']} "
            f"{result['error']}",
            flush=True,
        )


def process_map(args: argparse.Namespace, map_id: int) -> None:
    map_dir = Path(args.output).resolve() / f"map_{map_id}_{args.robot_version}"
    if EPISODE_ALIGNMENT_MODE == "normalized_progress":
        mean_txt = map_dir / "reward_components_progress_mean.txt"
        graph_png = map_dir / "reward_components_progress_mean.png"
    else:
        mean_txt = map_dir / "reward_components_mean.txt"
        graph_png = map_dir / "reward_components_mean.png"

    if args.plot_only:
        draw_reward_graph(mean_txt, graph_png, map_id, EPISODE_ALIGNMENT_MODE)
        return

    map_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        episode
        for episode in range(args.episodes)
        if args.rerun or not is_complete_episode_file(_episode_file(map_dir, episode))
    ]
    print(
        f"[INFO] map={map_id}, total={args.episodes}, pending={len(pending)}, "
        f"workers={args.workers}"
    )
    run_pending_episodes(args, map_id, map_dir, pending)

    episode_logs = []
    completed = 0
    for episode in range(args.episodes):
        path = _episode_file(map_dir, episode)
        if is_complete_episode_file(path):
            completed += 1
            episode_logs.append(read_episode_rewards(path))

    if EPISODE_ALIGNMENT_MODE == "normalized_progress":
        mean_rows = aggregate_episode_rewards_by_progress(
            episode_logs, NORMALIZED_PROGRESS_BINS
        )
    else:
        mean_rows = aggregate_episode_rewards(episode_logs)
    if mean_rows:
        if EPISODE_ALIGNMENT_MODE == "normalized_progress":
            write_progress_mean_rewards(mean_txt, mean_rows)
        else:
            write_mean_rewards(mean_txt, mean_rows)
        print(f"[MEAN] {mean_txt} ({completed} episodes)")
    else:
        print(f"[WARN] map={map_id}: 평균을 계산할 유효한 reward event가 없습니다.")

    args.current_map_id = map_id
    write_run_config(map_dir / "reward_config.txt", args, completed)
    draw_reward_graph(mean_txt, graph_png, map_id, EPISODE_ALIGNMENT_MODE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="에피소드 진행에 따른 reward component 평균과 그래프를 생성합니다."
    )
    parser.add_argument("--maps", nargs="+", type=int, default=MAP_LIST)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_MAP)
    parser.add_argument("--max-steps", type=int, default=MAX_STEP_NUM)
    parser.add_argument("--agents", type=int, default=NUMBER_OF_AGENTS)
    # Reward D/K/L and several optional components require env.robot.  The
    # existing FightingModel intentionally does not create it in N mode.
    parser.add_argument("--robot-version", choices=("T", "Q"), default=ROBOT_VERSION)
    parser.add_argument("--model", default=ROBOT_LEARNED_MODEL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="실험하지 않고 기존 reward_components_mean.txt로 그래프만 다시 그립니다.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="기존 에피소드 TXT가 있어도 요청한 에피소드를 다시 실행합니다.",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.episodes < 1:
        parser.error("--episodes는 1 이상이어야 합니다.")
    if args.max_steps < 1:
        parser.error("--max-steps는 1 이상이어야 합니다.")
    if args.agents < 1:
        parser.error("--agents는 1 이상이어야 합니다.")
    if args.workers < 1:
        parser.error("--workers는 1 이상이어야 합니다.")
    if EPISODE_ALIGNMENT_MODE not in ("absolute_time", "normalized_progress"):
        parser.error(
            "EPISODE_ALIGNMENT_MODE은 'absolute_time' 또는 "
            "'normalized_progress'여야 합니다."
        )
    if NORMALIZED_PROGRESS_BINS < 1:
        parser.error("NORMALIZED_PROGRESS_BINS는 1 이상이어야 합니다.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    # Resolve before a sequential episode changes cwd to SCRIPT_DIR.
    args.output = Path(args.output).expanduser().resolve()
    for map_id in args.maps:
        process_map(args, map_id)


if __name__ == "__main__":
    main()
