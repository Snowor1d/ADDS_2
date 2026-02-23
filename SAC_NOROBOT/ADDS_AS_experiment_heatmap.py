#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
robot_version       = 'N'     # 'N','T','Q'
robot_learned_model = 'FE_105_108_128.pth'
test_num            =  10
#map_list = [1239]
map_list = [105]
EXP_NAME = "0223/NOROBOT/get_heatmap"

CROWD_COLOR = "#0000FF"
ROBOT_COLOR = "#FF0000"
SINGLE_COLOR_EDGES = True
SHOW_AGENT_HEADING = False
SHOW_ROBOT_HEADING = False
ROBOT_HEADING_SCALE = 1.2

TRAIL_TARGET = "robot"  # robot / crowd / none
TRAIL_STYLE  = "persist"
MAX_TRAIL    = 2000
ROBOT_STYLE  = "image"
ROBOT_IMAGE_PATH  = "images/robot.png"
EXIT_SIZE = 5.0
SNAP_EXIT_TO_BOUNDARY = False

ANNOTATE_ROBOT_PATH = False
ANNOTATE_MODE       = "every_n"
ANNOTATE_EVERY      = 50
ANNOTATE_STYLE      = "subway"
ANNOTATE_FONTSIZE   = 20


# =========================
# [NEW] heatmap options
# =========================
ENABLE_HEATMAP = True          # heatmap 생성 여부
HEATMAP_W = 200                # heatmap x 해상도
HEATMAP_H = 200                # heatmap y 해상도
HEATMAP_BLACK_OBSTACLES = True
HEATMAP_OBS_THRESHOLD = 1
HEATMAP_EVERY = 1              # 몇 step마다 누적할지 (1이면 매 step)
HEATMAP_LOG_SCALE = True      # png 저장 시 log 스케일 적용 여부
HEATMAP_CMAP = "turbo"         # blue->red 느낌: turbo/jet 추천


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
# [NEW] heatmap helpers
# =========================
def _world_to_heat_px(x: float, y: float, env) -> tuple[int, int]:
    """
    env 좌표계를 heatmap (HEATMAP_W, HEATMAP_H) 픽셀로 변환
    """
    ix = int(np.clip(x / env.width  * HEATMAP_W, 0, HEATMAP_W - 1))
    iy = int(np.clip(y / env.height * HEATMAP_H, 0, HEATMAP_H - 1))
    return ix, iy

def _accumulate_points(heat: np.ndarray, points_xy: list[tuple[float, float]], env, w: float = 1.0):
    """
    heat[y, x] += w for each point
    heat shape: (HEATMAP_H, HEATMAP_W)
    """
    if heat is None or (not points_xy):
        return
    for (x, y) in points_xy:
        ix, iy = _world_to_heat_px(x, y, env)
        heat[iy, ix] += w

def _get_crowd_positions(env) -> list[tuple[float, float]]:
    """
    crowd agent 위치를 env.agents에서 수집.
    - agent.dead == 1 이면 제외
    - robot은 env.robot로 분리되어 있으므로 여기서는 crowd만 모으는 용도
    """
    out: list[tuple[float, float]] = []

    agents = getattr(env, "agents", None)
    if agents is None:
        return out

    for a in agents:
        try:
            # 죽은 agent 제외
            if getattr(a, "dead", 0) == 1:
                continue

            # 위치
            if not hasattr(a, "xy"):
                continue
            x, y = a.xy
            out.append((float(x), float(y)))
        except Exception:
            # 하나가 이상해도 전체는 계속
            continue

    return out
def _save_heatmap_png(
    heat: np.ndarray,
    out_png: str,
    log_scale: bool = False,
    cmap: str = "turbo",
    obstacle_mask: np.ndarray | None = None,
    cbar_label: str = None
):
    """
    heat를 png로 저장 (blue->red colormap) + colorbar 포함
    + obstacle_mask가 있으면 해당 영역을 검정색으로 표시

    - log_scale=True면 log1p 스케일로 시각화
    """
    if heat is None:
        return

    arr = heat.astype(np.float32)

    # 보기 좋게 y축 뒤집기
    arr = np.flipud(arr)

    # 시각화용 변환
    disp = np.log1p(arr) if log_scale else arr

    vmax = float(disp.max()) if disp.size else 0.0
    if vmax <= 0:
        vmax = 1.0

    # obstacle mask도 flip 맞춰줌
    if obstacle_mask is not None:
        mask = np.flipud(obstacle_mask.astype(bool))
        # mask 위치는 "검정"으로 보이도록 0으로 강제
        disp = disp.copy()
        disp[mask] = 0.0
    else:
        mask = None

    fig = plt.figure(figsize=(6.4, 5.4), dpi=200)
    ax = fig.add_subplot(111)

    im = ax.imshow(disp, cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()

    # obstacle은 진짜 검정으로 확실히 덮고 싶으면, 검정 레이어를 한 번 더 얹기
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)  # RGBA
        overlay[..., 3] = mask.astype(np.float32)              # alpha=1 on obstacles
        ax.imshow(overlay, interpolation="nearest")            # 검정색(0,0,0,1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    if cbar_label is not None:
        label_text = cbar_label
    else:
        label_text = "log(1 + visit count)" if log_scale else "visit count"

    cbar.set_label(label_text)
    fig.tight_layout(pad=0.05)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def load_slot_heatmap(test_dir: str):
    cpath = os.path.join(test_dir, "crowd_heat.npy")
    rpath = os.path.join(test_dir, "robot_heat.npy")
    if not (os.path.exists(cpath) and os.path.exists(rpath)):
        return None, None
    try:
        return np.load(cpath), np.load(rpath)
    except Exception:
        return None, None

def _downsample_u8_to_heatmap_mask(full_u8: np.ndarray) -> np.ndarray:
    """
    full_u8: (MAP_H, MAP_W) uint8 지도 이미지 (env.return_current_image로 획득)
    return: (HEATMAP_H, HEATMAP_W) bool mask, True=obstacle
    """
    x = torch.from_numpy(full_u8).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y = F.adaptive_max_pool2d(x, (HEATMAP_H, HEATMAP_W))             # (1,1,HEATMAP_H,HEATMAP_W)
    m = y.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)       # (HEATMAP_H,HEATMAP_W)
    # obstacle 판정 (임계치 이상이면 obstacle)
    return (m >= HEATMAP_OBS_THRESHOLD) & (m < 100)
# =========================
# episode runner
# =========================
def run_one_episode(map_id: int):
    
    crowd_points = 0
    heat_steps = 0

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

    # heatmap buffers
    crowd_heat = None
    robot_heat = None
    if ENABLE_HEATMAP:
        crowd_heat = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
        robot_heat = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)

    obstacle_mask = None
    if ENABLE_HEATMAP and HEATMAP_BLACK_OBSTACLES:
        full_map_u8 = env.return_current_image(MAP_H, MAP_W)
        obstacle_mask = _downsample_u8_to_heatmap_mask(full_map_u8)

    try:
        while True:
            env.step()
            step_num += 1
            alive = env.alived_agents()
            episode_log.append(alive)

            # [NEW] heatmap accumulate
            if ENABLE_HEATMAP and (step_num % HEATMAP_EVERY == 0):
                heat_steps += 1

                # robot: step 기준으로 1점
                # if hasattr(env, "robot"):
                #     rx, ry = env.robot.xy
                #     _accumulate_points(robot_heat, [(float(rx), float(ry))], env, w=1.0)


                # crowd: 죽은 애 제외된 위치들
                crowd_xy = _get_crowd_positions(env)
                _accumulate_points(crowd_heat, crowd_xy, env, w=1.0)
                crowd_points += len(crowd_xy)

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
                return step_num, evacuated_all, all_life, episode_log, collected_frames, crowd_heat, robot_heat, crowd_points, heat_steps, obstacle_mask

            if step_num >= max_step_num:
                evacuated_all = False
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames, crowd_heat, robot_heat, crowd_points, heat_steps, obstacle_mask

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
            # obstacle_mask = None
            # if ENABLE_HEATMAP and HEATMAP_BLACK_OBSTACLES:
            #     # 장애물 마스크는 해당 맵에서 고정이므로 1회 생성
            #     full_map_u8 = model.return_current_image(MAP_H, MAP_W)
            #     obstacle_mask = _downsample_u8_to_heatmap_mask(full_map_u8)

            steps, evacuated_all, all_life, ep_log, vis_frames, crowd_heat, robot_heat, crowd_points, heat_steps, obstacle_mask = run_one_episode(map_id)

            evacuation_100_time = steps if evacuated_all else max_step_num
            all_agents_life_time = all_life

            write_txt(
                os.path.join(test_dir, "metrics.txt"),
                f"evacuation_100_time={evacuation_100_time}\n"
                f"all_agents_life_time={all_agents_life_time}\n"
            )
            write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

            # [NEW] heatmap 저장
            if ENABLE_HEATMAP:
                np.save(os.path.join(test_dir, "crowd_heat.npy"), crowd_heat)
                np.save(os.path.join(test_dir, "robot_heat.npy"), robot_heat)

                # 분모 저장
                write_txt(os.path.join(test_dir, "heatmap_denoms.txt"),
                          f"crowd_points={crowd_points}\nheat_steps={heat_steps}\n")

                # (A) 점 기준 density
                crowd_density_points = crowd_heat / float(max(1, crowd_points))
                #robot_density_steps  = robot_heat / float(max(1, heat_steps))  # robot은 step당 1점이니까 이게 자연스러움

                np.save(os.path.join(test_dir, "crowd_density_points.npy"), crowd_density_points)
                #np.save(os.path.join(test_dir, "robot_density_steps.npy"), robot_density_steps)

                _save_heatmap_png(crowd_density_points, os.path.join(test_dir, "crowd_density_points.png"),
                                  log_scale=False, cmap=HEATMAP_CMAP, obstacle_mask = obstacle_mask, cbar_label = "Density (Visit Ratio)")
                # _save_heatmap_png(robot_density_steps, os.path.join(test_dir, "robot_density_steps.png"),
                #                   log_scale=False, cmap=HEATMAP_CMAP, obstacle_mask = obstacle_mask, cbar_label = "Density (Visit Ratio)")

                # 원본 count heatmap도 원하면 같이 (지금처럼)
                _save_heatmap_png(crowd_heat, os.path.join(test_dir, "crowd_heat.png"),
                                  log_scale=HEATMAP_LOG_SCALE, cmap=HEATMAP_CMAP, obstacle_mask = obstacle_mask)
                # _save_heatmap_png(robot_heat, os.path.join(test_dir, "robot_heat.png"),
                #                    log_scale=HEATMAP_LOG_SCALE, cmap=HEATMAP_CMAP, obstacle_mask = obstacle_mask)

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

    # GPU(Q)면 안전하게 1 권장 (필요하면 아래에서 직접 1로 고정해도 됨)
    if robot_version == 'Q':
        max_workers = min(6, os.cpu_count() or 1)
        # max_workers = 1  # <- GPU 터지면 이거 추천
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

            # [NEW] map 단위 heatmap(sum/avg) 생성
            if ENABLE_HEATMAP:
                map_obstacle_mask = None
                if HEATMAP_BLACK_OBSTACLES:
                    temp_env = model.FightingModel(number_of_agents, width=MAP_W, height=MAP_H, model_num=map_id, robot=robot_version)
                    if robot_version == 'Q':
                        temp_env.use_model(robot_learned_model)
                    full_map_u8 = temp_env.return_current_image(MAP_H, MAP_W)
                    map_obstacle_mask = _downsample_u8_to_heatmap_mask(full_map_u8)
                    del temp_env # 마스크만 빼오고 메모리 해제

                crowd_sum = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
                robot_sum = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
                count = 0

                for slot in range(test_num):
                    test_dir = os.path.join(map_robot_dir, f"Result_{map_id}_{robot_version}_{slot}")
                    if not is_slot_success(test_dir):
                        continue
                    c, r = load_slot_heatmap(test_dir)
                    if c is None or r is None:
                        continue
                    crowd_sum += c
                    robot_sum += r
                    count += 1

                np.save(os.path.join(map_robot_dir, "crowd_heat_sum.npy"), crowd_sum)
                #np.save(os.path.join(map_robot_dir, "robot_heat_sum.npy"), robot_sum)
                _save_heatmap_png(
                    crowd_sum,
                    os.path.join(map_robot_dir, "crowd_heat_sum.png"),
                    log_scale=HEATMAP_LOG_SCALE,
                    cmap=HEATMAP_CMAP,
                    obstacle_mask = map_obstacle_mask
                )
                # _save_heatmap_png(
                #     robot_sum,
                #     os.path.join(map_robot_dir, "robot_heat_sum.png"),
                #     log_scale=HEATMAP_LOG_SCALE,
                #     cmap=HEATMAP_CMAP,
                #     obstacle_mask = map_obstacle_mask
                # )

                if count > 0:
                    crowd_avg = crowd_sum / float(count)
                    #robot_avg = robot_sum / float(count)

                    np.save(os.path.join(map_robot_dir, "crowd_heat_avg.npy"), crowd_avg)
                    # np.save(os.path.join(map_robot_dir, "robot_heat_avg.npy"), robot_avg)

                    _save_heatmap_png(
                        crowd_avg,
                        os.path.join(map_robot_dir, "crowd_heat_avg.png"),
                        log_scale=HEATMAP_LOG_SCALE,
                        cmap=HEATMAP_CMAP,
                        obstacle_mask = map_obstacle_mask
                    )
                    # _save_heatmap_png(
                    #     robot_avg,
                    #     os.path.join(map_robot_dir, "robot_heat_avg.png"),
                    #     log_scale=HEATMAP_LOG_SCALE,
                    #     cmap=HEATMAP_CMAP,
                    #     obstacle_mask = map_obstacle_mask
                    # )

            print(f"[Map {map_id}] Summary updated at {map_robot_dir}")

        print(f"[ITER {it+1}] Done.")


if __name__ == "__main__":
    main()