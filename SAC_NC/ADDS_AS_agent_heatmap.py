#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import shutil
import numpy as np
import faulthandler
faulthandler.enable()

from typing import Optional  # ✅ Py3.8 호환: Optional 사용

from continuous_renderer import ContinuousRenderer
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
from state_grid_saver import GridStateSaver

VIS_SAVE_EVERY = 5

import model  # FightingModel 이 여기 있다고 가정

# ------------------------- #
# 실험 파라미터
visualization_mode = 'off' # 'off', 'cont_mp4', 'cont_png_every', 'cont_png'
run_iteration      = 1
number_of_agents   = 20
max_step_num       = 1500
robot_version      = 'N' # 'T' : direct to goal, 'R' : 가장 먼 agent와 출구 사이를 왔다갔다, 'Q' : 학습된 모델 사용, 'N' : No Robot
robot_learned_model = 'sac_checkpoint_ep_15000.pth'
test_num           = 50
map_list           = [6, 7, 8, 24, 25, 26]

MAP_WIDTH  = 50
MAP_HEIGHT = 50

EXP_NAME = "norobot_0918"   # 최상위 결과 폴더 이름

# 색상/렌더 옵션 (그대로 유지)
CROWD_COLOR = "#0000FF"
ROBOT_COLOR = "#FF0000"
SINGLE_COLOR_EDGES = True
SHOW_AGENT_HEADING = True
SHOW_ROBOT_HEADING = False
ROBOT_HEADING_SCALE = 1.2

TRAIL_TARGET = "none" # "none" | "crowd" | "robot" | "both"
TRAIL_STYLE = "persist" # "persist" or "fade"
MAX_TRAIL = 2000
ROBOT_STYLE = "circle" # "circle" or "image"
ROBOT_IMAGE_PATH = "assets/robot.png"
ROBOT_IMAGE_SCALE = 5
EXIT_SIZE = 5.0
SNAP_EXIT_TO_BOUNDARY = True

ANNOTATE_ROBOT_PATH = False
ANNOTATE_MODE = "every_n" #'all', 'endpoints'
ANNOTATE_EVERY = 20
ANNOTATE_STYLE = "subway" # "number" | "subway" | "frame"
ANNOTATE_FONTSIZE = 10

SAVE_GRID_ENABLE = True
SAVE_GRID_EVERY = 4

# === [helper: 안전한 MP4 저장기] ===
def save_continuous_mp4(frames_rgb, out_path, fps=20):
    """
    frames_rgb: np.ndarray(H, W, 3)의 리스트
    ffmpeg이 없으면 PNG 폴더로 fallback.
    """
    if not frames_rgb:
        return
    try:
        # matplotlib 애니메이션으로 저장
        h, w, _ = frames_rgb[0].shape
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = plt.axes([0,0,1,1]); ax.axis('off')
        im = ax.imshow(frames_rgb[0])

        writer = FFMpegWriter(fps=fps, bitrate=4000)
        with writer.saving(fig, out_path, dpi=100):
            for fr in frames_rgb:
                im.set_data(fr)
                writer.grab_frame()
        plt.close(fig)
    except Exception as e:
        # ffmpeg 누락 등 문제면 PNG 시퀀스로 대체 저장
        fallback_dir = out_path + "_pngs"
        os.makedirs(fallback_dir, exist_ok=True)
        for i, fr in enumerate(frames_rgb):
            plt.imsave(os.path.join(fallback_dir, f"frame_{i:05d}.png"), fr)
        print(f"[WARN] MP4 저장 실패({e}). PNG 시퀀스로 대체 저장: {fallback_dir}")

# -------------------------
# Heatmap 유틸
# -------------------------
def init_heatmap(width: int, height: int) -> np.ndarray:
    """
    히트맵을 (width, height)로 만든다.
    기존 코드의 이미지 인덱싱 관례에 맞춰 row <- x, col <- y로 누적할 것이므로
    shape = (MAP_WIDTH, MAP_HEIGHT) 로 만든다.
    """
    return np.zeros((width, height), dtype=np.uint32)

def clip_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))

def accumulate_agent_heatmap(env, heatmap: np.ndarray):
    """
    env의 crowd agent 좌표를 읽어 heatmap에 누적.
    - 좌표계: row = round(x), col = round(y) (너의 기존 이미지 저장 방식과 일치)
    - 죽은 에이전트는 제외 (agent.dead == True 제외)
    """
    # 1) crowd 리스트에서 직접 읽기 시도
    try:
        crowds = getattr(env, "crowds", None)
        if crowds is not None:
            for ag in crowds:
                # type 필터: 네 코드에선 type 1/2, 0 등으로 crowd 구분을 했었음
                # 여기서는 dead 아닌 모든 crowd를 누적 (원하면 type 체크 추가)
                if getattr(ag, "dead", False):
                    continue
                xy = getattr(ag, "xy", None)
                if xy is None or len(xy) != 2:
                    continue
                x, y = float(xy[0]), float(xy[1])
                rr = clip_int(x, 0, heatmap.shape[0]-1)  # row <- x
                cc = clip_int(y, 0, heatmap.shape[1]-1)  # col <- y
                heatmap[rr, cc] += 1
            return
    except Exception:
        pass

    # 2) fallback: env.return_current_image() 를 이용 (값으로 crowd 픽셀 찾기)
    try:
        img = env.return_current_image()  # shape (H, W) or (W, H)일 수 있음
        # 이전 대화 예시에서는 image[x, y]로 세팅했으므로 img.shape가 (W, H)일 가능성↑
        # 안전하게 둘 중 작은 축/큰 축 비교로 스왑 체크하지 않고, (W,H)=heatmap.shape에 맞춰 사용
        # crowd 픽셀 후보 값 (예시: 100, 140)
        crowd_vals = {100, 140}
        W, H = heatmap.shape  # (MAP_WIDTH, MAP_HEIGHT)
        # 이미지 크기가 다르면 가능한 범위에서만 누적
        w_img, h_img = img.shape[0], img.shape[1]
        w_lim = min(W, w_img)
        h_lim = min(H, h_img)
        for rr in range(w_lim):
            # row 기준 루프 (x)
            # crowd 픽셀 찾기
            row_slice = img[rr, :h_lim]
            # 벡터화로 조금 빠르게
            for cc in np.where(np.isin(row_slice, list(crowd_vals)))[0]:
                heatmap[rr, int(cc)] += 1
    except Exception:
        # 더 이상 대체 수단 없음: 조용히 패스
        pass

def save_heatmap_npy_png(out_dir: str, heatmap: np.ndarray, ep_idx: int):
    """
    히트맵을 npy와 png(일반/로그 스케일)로 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"agent_heatmap_ep{ep_idx:04d}")

    # .npy
    np.save(base + ".npy", heatmap)

    # PNG (일반)
    plt.figure()
    plt.imshow(heatmap.T, origin="lower", interpolation="nearest")  # 보기 편하게 transpose + origin lower
    plt.title(f"Agent Heatmap (ep={ep_idx})")
    plt.colorbar(label="visits")
    plt.tight_layout()
    plt.savefig(base + ".png", dpi=200)
    plt.close()

    # PNG (로그 스케일) — 0값 처리를 위해 +1
    plt.figure()
    plt.imshow((heatmap.T + 1), origin="lower", interpolation="nearest", norm=LogNorm())
    plt.title(f"Agent Heatmap (log) (ep={ep_idx})")
    plt.colorbar(label="log(visits+1)")
    plt.tight_layout()
    plt.savefig(base + "_log.png", dpi=200)
    plt.close()

# -------------------------
# 에피소드 러너
# -------------------------
def run_one_episode(map_id: int,
                    saver: Optional[GridStateSaver] = None,
                    heatmap: Optional[np.ndarray] = None):  # ✅ heatmap 주입
    step_num = 0
    env = model.FightingModel(
        number_of_agents,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        model_num=map_id,
        robot=robot_version
    )
    if (robot_version == 'Q'):
        env.use_model(robot_learned_model)

    renderer = ContinuousRenderer(
        world_size=(50.0, 50.0),
        crowd_colors = {0:CROWD_COLOR, 1:CROWD_COLOR, 2:CROWD_COLOR},
        robot_color = ROBOT_COLOR,
        single_color_edges= SINGLE_COLOR_EDGES,
        show_agent_heading= SHOW_AGENT_HEADING,
        show_robot_heading= SHOW_ROBOT_HEADING,
        robot_heading_scale= ROBOT_HEADING_SCALE,
        trail_target= TRAIL_TARGET,
        trail_style= TRAIL_STYLE,
        max_trail= MAX_TRAIL,
        robot_style= ROBOT_STYLE,
        robot_image_path= ROBOT_IMAGE_PATH,
        robot_image_scale= ROBOT_IMAGE_SCALE,
        exit_size= EXIT_SIZE,
        snap_exit_to_boundary = SNAP_EXIT_TO_BOUNDARY,
        annotate_robot_path = ANNOTATE_ROBOT_PATH,
        annotate_mode=ANNOTATE_MODE,
        annotate_every=ANNOTATE_EVERY,
        annotate_style = ANNOTATE_STYLE,
        annotate_fontsize = ANNOTATE_FONTSIZE,

        agent_heading_scale = 1.8,
        agent_heading_color = "#000000",
        agent_heading_linewidth = 1.5,
        agent_heading_mutation_scale = 9.0
    )

    save_rgb_every = (visualization_mode == 'cont_png_every')
    save_mp4 = (visualization_mode == 'cont_mp4')
    save_last_png = (visualization_mode == 'cont_png')
    collected_frames = []  # MP4/PNG용 프레임 버퍼

    episode_log = []
    try:
        while True:
            env.step()
            step_num += 1
            alive = env.alived_agents()
            episode_log.append(alive)

            # --- (A) 50x50 state 그레이 이미지 저장 ---
            if saver is not None:
                saver.maybe_save(step_num, env)

            # --- (B) Heatmap 누적: robot_version == 'N'일 때만 ---
            if heatmap is not None and robot_version == 'N':
                accumulate_agent_heatmap(env, heatmap)

            # --- (C) 연속 렌더 프레임 저장/버퍼링 ---
            if visualization_mode != 'off':
                if save_mp4:
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(rgb)
                elif save_rgb_every and (step_num % VIS_SAVE_EVERY == 0 or alive < 1):
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(('PNG', step_num, rgb))
                elif save_last_png:
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames = [('LAST', step_num, rgb)]  # 마지막만 유지

            # 종료 조건
            if alive < 1:
                evacuated_all = True
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames
            if step_num > max_step_num:
                evacuated_all = False
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames
    finally:
        del env

# -------------------------
# 파일/경로 유틸
# -------------------------
def home_path(*parts):
    return os.path.join(os.path.expanduser("~"), *parts)

def init_result_root(exp_name: str) -> str:
    """최상위 결과 폴더는 이미 있으면 삭제하지 않고 사용"""
    root = home_path(f"Result_data_{exp_name}")
    os.makedirs(root, exist_ok=True)
    return root

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def recreate_dir(path: str):
    """세부 실험 폴더는 있으면 지우고 새로 만든다"""
    if os.path.exists(path):
        shutil.rmtree(path)
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

# -------------------------
# 메인
# -------------------------
def main():
    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    # 맵별 Heatmap 준비 (robot_version == 'N'일 때만)
    heatmap_by_map = {}
    ep_count_by_map = {}  # 각 맵에서 몇 번째 에피소드까지 누적됐는지

    if robot_version == 'N':
        for mid in map_list:
            heatmap_by_map[mid] = init_heatmap(MAP_WIDTH, MAP_HEIGHT)
            ep_count_by_map[mid] = 0

    global_test_index = 0

    for j in range(run_iteration):
        print(f"\n=== Iteration {j+1}/{run_iteration} ===")
        for map_id in map_list:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{robot_version}")
            ensure_dir(map_robot_dir)

            # Heatmap 저장 폴더 (N일 때만)
            if robot_version == 'N':
                heat_out_dir = os.path.join(map_robot_dir, "heatmap_agent")
                ensure_dir(heat_out_dir)

            evac_times, life_times, episode_logs = [], [], []

            for test_i in range(test_num):
                unique_test_i = global_test_index
                test_dirname = f"Result_{map_id}_{robot_version}_{unique_test_i}"
                test_dir = os.path.join(map_robot_dir, test_dirname)

                recreate_dir(test_dir)

                # ★★ state grid 저장 디렉토리 & saver 생성
                if SAVE_GRID_ENABLE:
                    grid_out_dir = os.path.join(test_dir, "state_grid")
                    saver = GridStateSaver(grid_out_dir, every_steps=SAVE_GRID_EVERY)
                else:
                    saver = None

                # === 에피소드 실행 ===
                steps, evacuated_all, all_life, ep_log, vis_frames = run_one_episode(
                    map_id,
                    saver=saver,
                    heatmap=(heatmap_by_map[map_id] if robot_version == 'N' else None)
                )

                evacuation_100_time = steps if evacuated_all else max_step_num
                all_agents_life_time = all_life

                evac_times.append(evacuation_100_time)
                life_times.append(all_agents_life_time)
                episode_logs.append(ep_log)

                write_txt(
                    os.path.join(test_dir, "metrics.txt"),
                    f"evacuation_100_time={evacuation_100_time}\nall_agents_life_time={all_agents_life_time}\n"
                )
                write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

                print(f"[Map {map_id}] Test {unique_test_i} -> steps={steps}, evacuated_all={evacuated_all}")

                # === 연속 렌더 저장 ===
                if visualization_mode == 'cont_mp4':
                    out_mp4 = os.path.join(test_dir, "continuous.mp4")
                    save_continuous_mp4(vis_frames, out_mp4, fps=20)
                elif visualization_mode == 'cont_png_every':
                    png_dir = os.path.join(test_dir, "continuous_pngs")
                    os.makedirs(png_dir, exist_ok=True)
                    for tag, step_n, rgb in vis_frames:
                        if tag != 'PNG': 
                            continue
                        plt.imsave(os.path.join(png_dir, f"frame_{step_n:05d}.png"), rgb)
                elif visualization_mode == 'cont_png':
                    if vis_frames:
                        _, step_id, rgb = vis_frames[-1]
                        out_png = os.path.join(test_dir, f"continuous_last_{step_id:05d}.png")
                        plt.imsave(out_png, rgb)

                print(f"[Map {map_id}] Test {unique_test_i} visualization saved.")

                # === Heatmap 저장 체크 (10, 20, 30, ...) ===
                if robot_version == 'N':
                    ep_count_by_map[map_id] += 1
                    cur_ep = ep_count_by_map[map_id]
                    if cur_ep % 10 == 0:
                        save_heatmap_npy_png(heat_out_dir, heatmap_by_map[map_id], cur_ep)
                        print(f"[Map {map_id}] Saved agent heatmap at episode {cur_ep}")

                global_test_index += 1

            # 맵 요약 저장
            avg_evac = float(np.mean(evac_times)) if evac_times else float('nan')
            avg_life = float(np.mean(life_times)) if life_times else float('nan')
            write_txt(
                os.path.join(map_robot_dir, "avg_metrics.txt"),
                f"avg_evacuation_100_time={avg_evac}\navg_all_agents_life_time={avg_life}\n"
            )

            mean_series, min_series, max_series = aggregate_episode_logs(episode_logs)
            save_step_series(os.path.join(map_robot_dir, "episode_log_mean.txt"), mean_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_min.txt"), min_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_max.txt"), max_series)

            print(f"[Map {map_id}] Summary saved at {map_robot_dir}")

    # (선택) 마지막에 10의 배수가 아니면 최종본도 저장하고 싶다면 아래 주석 해제
    # if robot_version == 'N':
    #     for mid in map_list:
    #         heat_out_dir = os.path.join(result_root, f"Result_{mid}", f"Result_{mid}_N", "heatmap_agent")
    #         cur_ep = ep_count_by_map[mid]
    #         if cur_ep % 10 != 0 and cur_ep > 0:
    #             save_heatmap_npy_png(heat_out_dir, heatmap_by_map[mid], cur_ep)
    #             print(f"[Map {mid}] Saved final (non-multiple-of-10) heatmap at episode {cur_ep}")

if __name__ == "__main__":
    main()
