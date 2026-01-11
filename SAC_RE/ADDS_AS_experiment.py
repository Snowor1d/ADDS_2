#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel experiment runner (fixed slots per map):

- map별로 정확히 test_num개의 결과 폴더만 생성:
    Result_{map_id}_{robot_ver}_{slot}  where slot = 0..test_num-1

- 어떤 슬롯이든 에러/크래시가 나면 그 슬롯 폴더를 정리한 뒤 같은 슬롯을 재시도
  => 최종적으로 "성공 폴더"가 map당 test_num개 보장
  => 시도 횟수는 늘어도, 폴더 개수는 늘지 않음

- worker 프로세스가 죽어서 BrokenProcessPool이 발생해도 풀 재생성 후 계속

주의:
- robot_version='Q'이고 GPU를 쓰면 병렬을 크게 늘리면 CUDA 컨텍스트/메모리 문제 가능.
  안전하게는 max_workers=1 추천.
"""

import os
import re
import ast
import shutil
import traceback
import numpy as np
import faulthandler
faulthandler.enable()

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from continuous_renderer import ContinuousRenderer
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from config import *  # MAP_W, MAP_H, EGO_MAP_SIZE, DOWNSAMPLE_MAP_SIZE 등이 있다고 가정
import model  # FightingModel

import torch
import torch.nn.functional as F


# =========================
# 실험 파라미터 (원본 유지)
# =========================
VIS_SAVE_EVERY = 5

visualization_mode  = 'off'   # 'off', 'cont_mp4', 'cont_png_every', 'cont_png'
run_iteration       = 1
number_of_agents    = 30
max_step_num        = 3000
robot_version       = 'Q'     # 'N','T','Q'
robot_learned_model = 'FU1_10maps_K3.pth'
test_num            = 10
map_list            = [102, 113, 114, 115, 116]

EXP_NAME = "0111/FU1_10maps_K3"

CROWD_COLOR = "#0000FF"
ROBOT_COLOR = "#FF0000"
SINGLE_COLOR_EDGES = True
SHOW_AGENT_HEADING = False
SHOW_ROBOT_HEADING = False
ROBOT_HEADING_SCALE = 1.2

TRAIL_TARGET = "robot"  # robot / crowd / none
TRAIL_STYLE  = "persist"
MAX_TRAIL    = 2000
ROBOT_STYLE  = "circle"
ROBOT_IMAGE_PATH  = "assets/robot.png"
ROBOT_IMAGE_SCALE = 5
EXIT_SIZE = 5.0
SNAP_EXIT_TO_BOUNDARY = False

ANNOTATE_ROBOT_PATH = False
ANNOTATE_MODE       = "every_n"
ANNOTATE_EVERY      = 50
ANNOTATE_STYLE      = "subway"
ANNOTATE_FONTSIZE   = 20


# =========================
# utils
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

def list_slot_dirs(map_robot_dir: str, map_id: int, robot_ver: str):
    """
    고정 슬롯: Result_{map_id}_{robot_ver}_{slot} where slot=0..test_num-1
    """
    out = []
    for slot in range(test_num):
        name = f"Result_{map_id}_{robot_ver}_{slot}"
        out.append((slot, os.path.join(map_robot_dir, name)))
    return out

def is_slot_success(test_dir: str) -> bool:
    """
    성공 판정: metrics.txt + episode_log.txt 존재(최소 조건)
    """
    m = os.path.join(test_dir, "metrics.txt")
    e = os.path.join(test_dir, "episode_log.txt")
    return os.path.exists(m) and os.path.exists(e)

def load_all_slot_logs(map_robot_dir: str, map_id: int, robot_ver: str):
    evac_times = []
    life_times = []
    episode_logs = []
    for slot, test_dir in list_slot_dirs(map_robot_dir, map_id, robot_ver):
        if not is_slot_success(test_dir):
            continue
        evac, life = read_metrics(os.path.join(test_dir, "metrics.txt"))
        ep_log = read_episode_log(os.path.join(test_dir, "episode_log.txt"))
        if evac is not None:
            evac_times.append(evac)
        if life is not None:
            life_times.append(life)
        if ep_log is not None:
            episode_logs.append(ep_log)
    return evac_times, life_times, episode_logs


# =========================
# visualization helpers
# =========================
def save_continuous_mp4(frames_rgb, out_path, fps=20):
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi, facecolor="black", edgecolor="black", frameon=True)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    ax.set_axis_off()

    im = ax.imshow(frames_rgb[0], interpolation="nearest")
    im.set_zorder(10)

    writer = FFMpegWriter(
        fps=fps,
        bitrate=4000,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black"]
    )

    with writer.saving(fig, out_path, dpi=dpi):
        for fr in frames_rgb:
            im.set_data(fr)
            fig.canvas.draw()
            writer.grab_frame(facecolor="black")
    plt.close(fig)

def upscale_gray(img01: np.ndarray, scale: int = 8) -> np.ndarray:
    img01 = np.asarray(img01)
    if img01.ndim == 3:
        img01 = img01[0]
    img01 = np.clip(img01, 0.0, 1.0).astype(np.float32)
    return np.repeat(np.repeat(img01, scale, axis=0), scale, axis=1)

def _to_gray_png(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[0]
    if x.dtype.kind == "f":
        x = np.clip(x, 0.0, 1.0)
    x = np.flip(np.flip(x, axis=-1), axis=-2)
    x = np.flip(x, axis=1)
    x = upscale_gray(x, scale=8)
    return x

def ego_crop_from_full_map(full_map: np.ndarray,
                           robot_xy_px: tuple[int, int],
                           ego_size: int,
                           pad_value: int = 50) -> np.ndarray:
    H, W = full_map.shape
    cx, cy = robot_xy_px
    half = ego_size // 2

    x0, x1 = cx - half, cx - half + ego_size
    y0, y1 = cy - half, cy - half + ego_size

    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)

    crop = np.full((ego_size, ego_size), pad_value, dtype=full_map.dtype)
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = full_map[sy0:sy1, sx0:sx1]
    return crop

def downsample_full_map(full_map: np.ndarray, target: int) -> np.ndarray:
    x = torch.from_numpy(full_map).float().unsqueeze(0).unsqueeze(0)
    y = F.adaptive_max_pool2d(x, (target, target))
    return y.squeeze(0).squeeze(0).byte().numpy()

def _robot_world_to_px(env):
    rx, ry = env.robot.xy
    ix = int(np.clip(rx / env.width  * MAP_W, 0, MAP_W - 1))
    iy = int(np.clip(ry / env.height * MAP_H, 0, MAP_H - 1))
    return ix, iy

def build_ego_global_frames(env):
    full_map_u8 = env.return_current_image(MAP_H, MAP_W)
    ix, iy = _robot_world_to_px(env)
    ego_u8  = ego_crop_from_full_map(full_map_u8, (ix, iy), EGO_MAP_SIZE, pad_value=50)
    glob_u8 = downsample_full_map(full_map_u8, DOWNSAMPLE_MAP_SIZE)
    ego_f  = ego_u8.astype(np.float32) / 255.0
    glob_f = glob_u8.astype(np.float32) / 255.0
    return ego_f, glob_f


# =========================
# episode runner
# =========================
def run_one_episode(map_id: int):
    step_num = 0
    env = model.FightingModel(
        number_of_agents,
        width=MAP_W,
        height=MAP_H,
        model_num=map_id,
        robot=robot_version
    )

    if robot_version == 'Q':
        env.use_model(robot_learned_model)

    renderer = ContinuousRenderer(
        world_size=(float(MAP_W), float(MAP_H)),
        crowd_colors={0: CROWD_COLOR, 1: CROWD_COLOR, 2: CROWD_COLOR},
        robot_color=ROBOT_COLOR,
        single_color_edges=SINGLE_COLOR_EDGES,
        show_agent_heading=SHOW_AGENT_HEADING,
        show_robot_heading=SHOW_ROBOT_HEADING,
        robot_heading_scale=ROBOT_HEADING_SCALE,
        trail_target=TRAIL_TARGET,
        trail_style=TRAIL_STYLE,
        max_trail=MAX_TRAIL,
        robot_style=ROBOT_STYLE,
        robot_image_path=ROBOT_IMAGE_PATH,
        robot_image_scale=ROBOT_IMAGE_SCALE,
        exit_size=EXIT_SIZE,
        snap_exit_to_boundary=SNAP_EXIT_TO_BOUNDARY,
        annotate_robot_path=ANNOTATE_ROBOT_PATH,
        annotate_mode=ANNOTATE_MODE,
        annotate_every=ANNOTATE_EVERY,
        annotate_style=ANNOTATE_STYLE,
        annotate_fontsize=ANNOTATE_FONTSIZE,

        agent_heading_scale=1.8,
        agent_heading_color="#000000",
        agent_heading_linewidth=1.5,
        agent_heading_mutation_scale=9.0
    )

    save_rgb_every   = (visualization_mode == 'cont_png_every')
    save_mp4         = (visualization_mode == 'cont_mp4')
    save_last_png    = (visualization_mode == 'cont_png')
    collected_frames = []

    episode_log = []
    try:
        while True:
            env.step()
            step_num += 1
            alive = env.alived_agents()
            episode_log.append(alive)

            if visualization_mode != 'off':
                if save_mp4:
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(rgb)

                elif save_rgb_every and (step_num % VIS_SAVE_EVERY == 0 or alive < 1):
                    rgb = renderer.draw(env, step=step_num)
                    ego_f, glob_f = build_ego_global_frames(env)
                    collected_frames.append(("PNG_EVERY", step_num, rgb, ego_f, glob_f))

                elif save_last_png:
                    rgb = renderer.draw(env, step=step_num)
                    ego_f, glob_f = build_ego_global_frames(env)
                    collected_frames = [("LAST", step_num, rgb, ego_f, glob_f)]

            if alive < 1:
                evacuated_all = True
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames

            if step_num >= max_step_num:
                evacuated_all = False
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames

    finally:
        del env


# =========================
# worker: "고정 슬롯" 하나를 성공할 때까지 재시도
# =========================
def worker_run_slot_until_success(map_id: int, slot: int, test_dir: str, max_retries: int = 10_000):
    """
    한 슬롯 디렉토리를 '성공' 상태로 만들 때까지 반복.
    - 실패하면 test_dir을 깨끗이 지운 뒤 재시도
    - segfault(-9)처럼 프로세스가 죽는 경우는 이 함수 레벨에서 처리 불가:
      메인에서 BrokenProcessPool로 감지 후 이 슬롯을 다시 제출함.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            # 이전 실패 흔적 있으면 삭제
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir, ignore_errors=True)
            os.makedirs(test_dir, exist_ok=False)

            steps, evacuated_all, all_life, ep_log, vis_frames = run_one_episode(map_id)

            evacuation_100_time = steps if evacuated_all else max_step_num
            all_agents_life_time = all_life

            write_txt(
                os.path.join(test_dir, "metrics.txt"),
                f"evacuation_100_time={evacuation_100_time}\n"
                f"all_agents_life_time={all_agents_life_time}\n"
            )
            write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

            # visualization 저장
            if visualization_mode == 'cont_mp4':
                out_mp4 = os.path.join(test_dir, "continuous.mp4")
                save_continuous_mp4(vis_frames, out_mp4, fps=20)

            elif visualization_mode == 'cont_png_every':
                png_dir  = os.path.join(test_dir, "continuous_pngs")
                ego_dir  = os.path.join(test_dir, "ego_pngs")
                glob_dir = os.path.join(test_dir, "global_pngs")
                os.makedirs(png_dir,  exist_ok=True)
                os.makedirs(ego_dir,  exist_ok=True)
                os.makedirs(glob_dir, exist_ok=True)

                for item in vis_frames:
                    if item[0] != "PNG_EVERY":
                        continue
                    _, step_n, rgb, ego_f, glob_f = item
                    plt.imsave(os.path.join(png_dir,  f"frame_{step_n:05d}.png"), rgb)
                    plt.imsave(os.path.join(ego_dir,  f"ego_{step_n:05d}.png"),
                               _to_gray_png(ego_f), cmap="gray", vmin=0.0, vmax=1.0)
                    plt.imsave(os.path.join(glob_dir, f"glob_{step_n:05d}.png"),
                               _to_gray_png(glob_f), cmap="gray", vmin=0.0, vmax=1.0)

            elif visualization_mode == 'cont_png':
                if vis_frames:
                    _, step_id, rgb, ego_f, glob_f = vis_frames[-1]
                    plt.imsave(os.path.join(test_dir, f"continuous_last_{step_id:05d}.png"), rgb)
                    plt.imsave(os.path.join(test_dir, f"ego_last_{step_id:05d}.png"),
                               _to_gray_png(ego_f), cmap="gray", vmin=0.0, vmax=1.0)
                    plt.imsave(os.path.join(test_dir, f"glob_last_{step_id:05d}.png"),
                               _to_gray_png(glob_f), cmap="gray", vmin=0.0, vmax=1.0)

            return {"ok": True, "map_id": map_id, "slot": slot, "test_dir": test_dir, "attempt": attempt}

        except Exception as e:
            tb = traceback.format_exc()
            # 같은 슬롯 안에 error.txt 기록 후, 폴더를 지우고 재시도
            try:
                os.makedirs(test_dir, exist_ok=True)
                write_txt(os.path.join(test_dir, "error.txt"), f"[attempt={attempt}] {repr(e)}\n\n{tb}\n")
            except Exception:
                pass

            if attempt >= max_retries:
                return {"ok": False, "map_id": map_id, "slot": slot, "test_dir": test_dir,
                        "attempt": attempt, "error": repr(e)}

            # 다음 루프에서 rmtree 후 재시도


# =========================
# main: slot 단위 병렬 실행 + BrokenProcessPool 복구
# =========================
def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    # GPU(Q)면 안전하게 1 권장
    if robot_version == 'Q':
        max_workers =  min(6, os.cpu_count() or 1)
    else:
        max_workers = min(6, os.cpu_count() or 1)

    for it in range(run_iteration):
        print(f"\n=== Iteration {it+1}/{run_iteration} ===")

        # map별 고정 슬롯 디렉토리 생성
        slots = []  # (map_id, slot, test_dir)
        map_robot_dirs = {}

        for map_id in map_list:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{robot_version}")
            ensure_dir(map_robot_dir)
            map_robot_dirs[map_id] = map_robot_dir

            for slot in range(test_num):
                test_dir = os.path.join(map_robot_dir, f"Result_{map_id}_{robot_version}_{slot}")
                slots.append((map_id, slot, test_dir))

        # 이미 성공한 슬롯은 건너뜀
        pending = [(mid, s, td) for (mid, s, td) in slots if not is_slot_success(td)]
        done = {(mid, s): is_slot_success(td) for (mid, s, td) in slots}

        print(f"[INFO] total slots = {len(slots)}, pending = {len(pending)}")

        # pending 슬롯이 없어질 때까지 반복
        while pending:
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futures = {}
                    # 현재 라운드에서 최대 max_workers만 제출
                    batch = pending[:max_workers]
                    pending = pending[max_workers:]

                    for (mid, s, td) in batch:
                        fut = ex.submit(worker_run_slot_until_success, mid, s, td)
                        futures[fut] = (mid, s, td)

                    for fut in as_completed(futures):
                        mid, s, td = futures[fut]

                        try:
                            res = fut.result()
                        except Exception as e:
                            # worker 프로세스가 죽었거나(pickle/환경 문제 포함)
                            print(f"[CRASH][Map {mid}][slot {s}] -> {repr(e)}")
                            # 같은 슬롯을 다시 pending에 넣음
                            pending.append((mid, s, td))
                            continue

                        if res.get("ok", False):
                            done[(mid, s)] = True
                            print(f"[OK][Map {mid}][slot {s}] attempt={res.get('attempt')}")
                        else:
                            print(f"[GIVEUP][Map {mid}][slot {s}] attempt={res.get('attempt')} error={res.get('error')}")
                            # giveup이면 일단 pending에 다시 넣지 않음 (원하면 넣어도 됨)

            except BrokenProcessPool:
                print("[WARN] BrokenProcessPool detected. Recreating executor and continuing...")
                # 풀 깨짐: 아직 완료 안 된 슬롯들을 다시 pending에 넣고 계속
                still = []
                for (mid, s, td) in slots:
                    if not is_slot_success(td):
                        still.append((mid, s, td))
                pending = still
                continue

        # map별 요약파일 갱신 (슬롯 성공 폴더들만 기준)
        for map_id in map_list:
            map_robot_dir = map_robot_dirs[map_id]
            evac_times, life_times, episode_logs = load_all_slot_logs(map_robot_dir, map_id, robot_version)

            avg_evac = float(np.mean(evac_times)) if evac_times else float('nan')
            avg_life = float(np.mean(life_times)) if life_times else float('nan')

            write_txt(
                os.path.join(map_robot_dir, "avg_metrics.txt"),
                f"avg_evacuation_100_time={avg_evac}\n"
                f"avg_all_agents_life_time={avg_life}\n"
                f"num_episodes={len(evac_times)}\n"
            )

            mean_series, min_series, max_series = aggregate_episode_logs(episode_logs)
            save_step_series(os.path.join(map_robot_dir, "episode_log_mean.txt"), mean_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_min.txt"),  min_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_max.txt"),  max_series)

            print(f"[Map {map_id}] Summary updated at {map_robot_dir}")

        print(f"[ITER {it+1}] Done.")


if __name__ == "__main__":
    main()
