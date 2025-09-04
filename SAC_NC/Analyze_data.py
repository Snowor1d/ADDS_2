#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Result_data_{EXP_NAME} 폴더를 분석해 논문용 그래프를 생성합니다.

전역 변수 설정:
- ROOT_DIR   : 분석할 결과 폴더 (예: ~/Result_data_try1)
- OUT_DIR    : 그래프 저장 경로
- CUTOFF     : 몇 스텝 이후 남아있는 인원은 사망으로 간주
"""

import os
import re
import ast
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 전역 파라미터
# =========================
ROOT_DIR = os.path.expanduser("~/Result_data_try1")
OUT_DIR  = os.path.join(ROOT_DIR, "plots")
CUTOFF   = 200   # 이 step 이후 미대피자는 사망으로 간주

# =========================
# 데이터 구조
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

def read_metrics(path: str) -> Tuple[float, float]:
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
    try:
        arr = ast.literal_eval(txt)
        return list(arr)
    except Exception:
        return []

def load_data(root: str) -> Dict[int, MapData]:
    maps: Dict[int, MapData] = {}
    map_dir_pat = re.compile(r"^Result_(\d+)$")
    robot_dir_pat = re.compile(r"^Result_(\d+)_([A-Za-z0-9]+)$")
    test_dir_pat  = re.compile(r"^Result_(\d+)_([A-Za-z0-9]+)_(\d+)$")

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
                metrics_path = os.path.join(tdir, METRICS_FILE)
                elog_path = os.path.join(tdir, EPISODE_LOG_FILE)

                evac, life = read_metrics(metrics_path)
                if evac is None or life is None:
                    continue
                elog = read_episode_log(elog_path)
                maps[map_id].robots[robot_ver].runs.append(
                    TestRun(evacuation_100_time=evac, all_agents_life_time=life, episode_log=elog)
                )
    return maps

# =========================
# 통계 유틸
# =========================
def compute_min_mean_max(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return np.nan, np.nan, np.nan
    arr = np.array(values, dtype=float)
    return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

def mean_curve_truncated(runs: List[List[float]]) -> np.ndarray:
    if not runs:
        return np.array([])
    L = max(len(r) for r in runs)
    means = []
    for i in range(L):
        bucket = [r[i] for r in runs if len(r) > i]
        means.append(np.mean(bucket) if bucket else np.nan)
    return np.array(means, dtype=float)

def mean_curve_padded(runs: List[List[float]], pad_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if not runs:
        return np.array([]), np.array([])
    L = max(len(r) for r in runs)
    M = np.stack([np.pad(r, (0, L-len(r)), constant_values=pad_value) for r in runs], axis=0)
    return M.mean(axis=0), M.std(axis=0)

def deaths_at_cutoff(log: List[float], cutoff: int) -> int:
    if not log:
        return 0
    idx = min(cutoff, len(log)-1)
    return int(round(log[idx]))

# =========================
# 플롯
# =========================
def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def pick_color(idx: int):
    base = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5'])
    return base[idx % len(base)]

def plot_evacuation_per_map(map_id: int, mdata: MapData, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Map {map_id} - Evacuation 100")
    ax.set_ylabel("Evacuation 100 Time (steps)")
    ax.set_xlabel("Robot Version")

    robot_versions = sorted(mdata.robots.keys())
    xs = np.arange(len(robot_versions))
    width = 0.6

    for i, rv in enumerate(robot_versions):
        color = pick_color(i)
        runs = mdata.robots[rv].runs
        vals = [r.evacuation_100_time for r in runs]
        vmin, vmean, vmax = compute_min_mean_max(vals)

        left, right = xs[i] - width/2, xs[i] + width/2
        ax.hlines(vmean, left, right, color=color, linewidth=2.5)
        ax.fill_between([left, right], [vmin, vmin], [vmax, vmax], color=color, alpha=0.2)
        ax.scatter(xs[i]*np.ones(len(vals)), vals, color=color, alpha=0.3)

    ax.set_xticks(xs, robot_versions)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    ensure_out(out_dir)
    fig.savefig(os.path.join(out_dir, f"map_{map_id}_evac100.png"), dpi=200)
    plt.close(fig)

def plot_episode_log_padded(map_id: int, mdata: MapData, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Map {map_id} - Episode Log (padded)")
    ax.set_xlabel("Step")
    ax.set_ylabel("# Alive agents")

    for i, (rv, group) in enumerate(sorted(mdata.robots.items())):
        color = pick_color(i)
        runs = [r.episode_log for r in group.runs]
        mean_curve, _ = mean_curve_padded(runs, pad_value=0.0)
        if mean_curve.size == 0: continue
        ax.plot(mean_curve, label=rv, color=color, linewidth=2)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_out(out_dir)
    fig.savefig(os.path.join(out_dir, f"map_{map_id}_episode_padded.png"), dpi=200)
    plt.close(fig)

def plot_episode_log_with_cutoff(map_id: int, mdata: MapData, out_dir: str, cutoff: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Map {map_id} - Episode Log with Cutoff={cutoff}")
    ax.set_xlabel("Step")
    ax.set_ylabel("# Alive agents")

    for i, (rv, group) in enumerate(sorted(mdata.robots.items())):
        color = pick_color(i)
        runs = [r.episode_log for r in group.runs]
        if not runs: continue

        truncated_runs = [r[:cutoff+1] if len(r) > cutoff+1 else r for r in runs]
        mean_curve = mean_curve_truncated(truncated_runs)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, label=f"{rv} mean", color=color, linewidth=2)

        for r in truncated_runs:
            if not r: continue
            x_end = min(cutoff, len(r)-1)
            y_end = r[x_end]
            ax.scatter([x_end], [y_end], color=color, s=16, alpha=0.65)
            ax.text(x_end, y_end, f"{int(y_end)}", color=color, fontsize=8)

    ax.set_xlim(0, cutoff)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_out(out_dir)
    fig.savefig(os.path.join(out_dir, f"map_{map_id}_episode_cutoff{cutoff}.png"), dpi=200)
    plt.close(fig)

# =========================
# 메인
# =========================
def main():
    maps = load_data(ROOT_DIR)
    for map_id, mdata in sorted(maps.items()):
        plot_evacuation_per_map(map_id, mdata, OUT_DIR)
        plot_episode_log_padded(map_id, mdata, OUT_DIR)
        plot_episode_log_with_cutoff(map_id, mdata, OUT_DIR, cutoff=CUTOFF)
    print(f"[DONE] Plots saved to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
