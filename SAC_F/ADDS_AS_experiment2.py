#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADDS_AS_experiment.py (rebuilt from run_sim.py style)

Batch evaluation runner:
- loops over map_list and test_num
- creates env (FightingModel) in a stable order
- optional: load RL model
- logs:
  - metrics.txt (evacuation_100_time, all_agents_life_time)
  - episode_log.txt (alive agents over time)
  - avg_metrics.txt + episode_log_{mean,min,max}.txt (aggregated)
- visualization OFF by default (to reduce segfault risk)

Note:
- segfault cannot be caught by Python try/except. If it still occurs,
  consider running each episode/map in a separate process.
"""

import os
import re
import ast
import json
import time
import shutil
import numpy as np
import faulthandler
faulthandler.enable()

# ---- reduce BLAS/OMP thread issues (must be before heavy imports) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from config import *  # MAP_W, MAP_H, etc.

# =========================
# Experiment config
# =========================
EXP_NAME = "0113/unseen_test"

RUN_ITERATION = 1
TEST_NUM = 1
MAP_LIST = [102, 1]

NUMBER_OF_AGENTS = 30
MAX_STEPS = 1500

# Robot / policy
ROBOT_VERSION = "Q"          # "N","T","Q"
ROBOT_CONTROL_MODE = "RL"  # "None" or "RL"
MODEL_NAME = "RE_10.pth" # used if ROBOT_CONTROL_MODE == "RL"

# Visualization
VIS_MODE = "off"             # "off" | "cont_png" | "cont_png_every" | "cont_mp4"
VIS_SAVE_EVERY = 5
MP4_FPS = 20
MP4_BITRATE = 4000

# Renderer settings (kept minimal)
USE_RENDERER = (VIS_MODE != "off")

# =========================
# FS utils
# =========================
def home_path(*parts):
    return os.path.join(os.path.expanduser("~"), *parts)

def init_result_root(exp_name: str) -> str:
    root = home_path(f"Result_data_{exp_name}")
    os.makedirs(root, exist_ok=True)
    return root

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_txt(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_list_txt(path: str, values):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(list(values)))

def save_step_series(path: str, series):
    with open(path, "w", encoding="utf-8") as f:
        for v in series:
            f.write(f"{v}\n")

def save_continuous_mp4(frames_rgb, out_path, fps=20, bitrate=4000):
    if not frames_rgb:
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    h, w, _ = frames_rgb[0].shape
    dpi = 100

    fig = plt.figure(
        figsize=(w / dpi, h / dpi),
        dpi=dpi,
        facecolor="black",
        frameon=False
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    im = ax.imshow(frames_rgb[0], interpolation="nearest")

    writer = FFMpegWriter(
        fps=fps,
        bitrate=bitrate,
        codec="libx264",
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black"
        ]
    )

    with writer.saving(fig, out_path, dpi=dpi):
        for fr in frames_rgb:
            im.set_data(fr)
            writer.grab_frame()

    plt.close(fig)

def aggregate_episode_logs(logs):
    if not logs:
        return [], [], []
    max_len = max(len(log) for log in logs)
    means, mins, maxs = [], [], []
    for i in range(max_len):
        bucket = [log[i] for log in logs if len(log) > i]
        means.append(float(np.mean(bucket)))
        mins.append(int(np.min(bucket)))
        maxs.append(int(np.max(bucket)))
    return means, mins, maxs

# -------------------------
# scan existing tests
# -------------------------
def _test_dir_pattern(map_id: int, robot_ver: str):
    return re.compile(rf"^Result_{map_id}_{re.escape(robot_ver)}_(\d+)$")

def list_existing_tests(map_robot_dir: str, map_id: int, robot_ver: str):
    patt = _test_dir_pattern(map_id, robot_ver)
    out = []
    if not os.path.isdir(map_robot_dir):
        return out
    for name in os.listdir(map_robot_dir):
        full = os.path.join(map_robot_dir, name)
        if not os.path.isdir(full):
            continue
        m = patt.match(name)
        if m:
            out.append((int(m.group(1)), full))
    out.sort(key=lambda x: x[0])
    return out

def get_next_test_index(map_robot_dir: str, map_id: int, robot_ver: str) -> int:
    tests = list_existing_tests(map_robot_dir, map_id, robot_ver)
    return 0 if not tests else (tests[-1][0] + 1)

def read_metrics(metrics_path: str):
    evac = None
    life = None
    if not os.path.exists(metrics_path):
        return None, None
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            try:
                if k == "evacuation_100_time":
                    evac = float(v)
                elif k == "all_agents_life_time":
                    life = float(v)
            except Exception:
                pass
    return evac, life

def read_episode_log(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    try:
        data = ast.literal_eval(s)
        if isinstance(data, list):
            return [int(x) for x in data]
    except Exception:
        return None
    return None

def load_all_existing_logs(map_robot_dir: str, map_id: int, robot_ver: str):
    evac_times, life_times, episode_logs = [], [], []
    for n, test_dir in list_existing_tests(map_robot_dir, map_id, robot_ver):
        evac, life = read_metrics(os.path.join(test_dir, "metrics.txt"))
        ep_log = read_episode_log(os.path.join(test_dir, "episode_log.txt"))
        if evac is not None: evac_times.append(evac)
        if life is not None: life_times.append(life)
        if ep_log is not None: episode_logs.append(ep_log)
    return evac_times, life_times, episode_logs

# =========================
# Renderer creation
# =========================
def make_renderer(world_w, world_h):
    # Import renderer lazily AFTER env creation to reduce segfault risk.
    from continuous_renderer import ContinuousRenderer
    return ContinuousRenderer(
        world_size=(float(world_w), float(world_h)),
        crowd_colors={0:"#4e79a7",1:"#4e79a7",2:"#4e79a7"},
        robot_color="#e15759",
        show_agent_heading=False,
        show_robot_heading=False,
        robot_heading_scale=3.0,
        trail_target="none",
        trail_style="persist",
        max_trail=2000,
        single_color_edges=True,
        exit_size=5.0,
        snap_exit_to_boundary=True,
    )

# =========================
# Visualization saves (optional)
# =========================
def save_last_png(rgb, out_png_path):
    import matplotlib.pyplot as plt
    plt.imsave(out_png_path, rgb)

# =========================
# Episode runner
# =========================
def create_env(map_id: int):
    # Import model lazily too (helps isolate C-extension init order)
    from model import FightingModel
    env = FightingModel(
        number_agents=NUMBER_OF_AGENTS,
        width=MAP_W,
        height=MAP_H,
        model_num=map_id,
        robot=ROBOT_VERSION
    )
    if ROBOT_CONTROL_MODE == "RL":
        env.use_model(MODEL_NAME)
    return env

def run_one_episode(map_id: int):
    env = create_env(map_id)

    renderer = None
    if USE_RENDERER:
        renderer = make_renderer(MAP_W, MAP_H)

    step_num = 0
    episode_log = []
    frames_rgb = []
    last_rgb = None

    try:
        while True:
            env.step()
            step_num += 1

            alive = int(env.alived_agents())
            episode_log.append(alive)

            if USE_RENDERER and (VIS_MODE == "cont_png"):
                last_rgb = renderer.draw(env, step=step_num)
            
            if USE_RENDERER and (VIS_MODE == "cont_mp4"):
                rgb = renderer.draw(env, step=step_num)
                frames_rgb.append(rgb)

            if alive < 1:
                evacuated_all = True
                all_life = float(env.calculate_all_agents_life_time())
                return step_num, evacuated_all, all_life, episode_log, last_rgb, frames_rgb

            if step_num >= MAX_STEPS:
                evacuated_all = False
                all_life = float(env.calculate_all_agents_life_time())
                return step_num, evacuated_all, all_life, episode_log, last_rgb, frames_rgb
    finally:
        # Make sure references are dropped
        try:
            del renderer
        except Exception:
            pass
        del env

# =========================
# Main
# =========================
def main():
    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    # Save a small run config snapshot
    write_txt(os.path.join(result_root, "run_config.json"), json.dumps({
        "EXP_NAME": EXP_NAME,
        "RUN_ITERATION": RUN_ITERATION,
        "TEST_NUM": TEST_NUM,
        "MAP_LIST": MAP_LIST,
        "NUMBER_OF_AGENTS": NUMBER_OF_AGENTS,
        "MAX_STEPS": MAX_STEPS,
        "ROBOT_VERSION": ROBOT_VERSION,
        "ROBOT_CONTROL_MODE": ROBOT_CONTROL_MODE,
        "MODEL_NAME": MODEL_NAME,
        "VIS_MODE": VIS_MODE,
    }, indent=2))

    for j in range(RUN_ITERATION):
        print(f"\n=== Iteration {j+1}/{RUN_ITERATION} ===")

        for map_id in MAP_LIST:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{ROBOT_VERSION}")
            ensure_dir(map_robot_dir)

            # Load existing stats
            evac_times, life_times, episode_logs = load_all_existing_logs(map_robot_dir, map_id, ROBOT_VERSION)
            start_index = get_next_test_index(map_robot_dir, map_id, ROBOT_VERSION)
            print(f"[Map {map_id}] next_test_index={start_index} (existing={len(list_existing_tests(map_robot_dir,map_id,ROBOT_VERSION))})")

            for t in range(TEST_NUM):
                unique_test_i = start_index + t
                test_dirname = f"Result_{map_id}_{ROBOT_VERSION}_{unique_test_i}"
                test_dir = os.path.join(map_robot_dir, test_dirname)

                if os.path.exists(test_dir):
                    raise RuntimeError(f"Test dir already exists: {test_dir}")
                os.makedirs(test_dir, exist_ok=False)

                print(f"[Map {map_id}] Test {unique_test_i} start...")
                steps, evacuated_all, all_life, ep_log, last_rgb, frames_rgb = run_one_episode(map_id)

                evacuation_100_time = steps if evacuated_all else MAX_STEPS
                all_agents_life_time = all_life

                evac_times.append(float(evacuation_100_time))
                life_times.append(float(all_agents_life_time))
                episode_logs.append(ep_log)

                write_txt(
                    os.path.join(test_dir, "metrics.txt"),
                    f"evacuation_100_time={evacuation_100_time}\n"
                    f"all_agents_life_time={all_agents_life_time}\n"
                )
                write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

                # optional last png
                if USE_RENDERER and VIS_MODE == "cont_png" and (last_rgb is not None):
                    save_last_png(last_rgb, os.path.join(test_dir, f"continuous_last_{steps:05d}.png"))

                if USE_RENDERER and VIS_MODE == "cont_mp4":
                    out_mp4 = os.path.join(test_dir, "continuous.mp4")
                    save_continuous_mp4(
                        frames_rgb,
                        out_mp4,
                        fps=MP4_FPS,
                        bitrate=MP4_BITRATE
                    )

                print(f"[Map {map_id}] Test {unique_test_i} done: steps={steps}, evacuated_all={evacuated_all}")

            # Write summary across ALL episodes
            avg_evac = float(np.mean(evac_times)) if evac_times else float("nan")
            avg_life = float(np.mean(life_times)) if life_times else float("nan")
            write_txt(
                os.path.join(map_robot_dir, "avg_metrics.txt"),
                f"avg_evacuation_100_time={avg_evac}\n"
                f"avg_all_agents_life_time={avg_life}\n"
                f"num_episodes={len(evac_times)}\n"
            )

            mean_series, min_series, max_series = aggregate_episode_logs(episode_logs)
            save_step_series(os.path.join(map_robot_dir, "episode_log_mean.txt"), mean_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_min.txt"), min_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_max.txt"), max_series)

            print(f"[Map {map_id}] Summary updated at {map_robot_dir}")

if __name__ == "__main__":
    main()
