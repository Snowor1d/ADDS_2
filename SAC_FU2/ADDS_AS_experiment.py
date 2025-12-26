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
from config import *
#from state_grid_saver import GridStateSaver

VIS_SAVE_EVERY = 5

import model  # FightingModel 이 여기 있다고 가정

# ------------------------- #
# 실험 파라미터
visualization_mode = 'cont_mp4' # 'off', 'cont_mp4', 'cont_png_every', 'cont_png'
run_iteration      = 1
number_of_agents   = 30
max_step_num       = 100
robot_version      = 'T' # 'T' : direct to goal, 'R' : 가장 먼 agent와 출구 사이를 왔다갔다, 'Q' : 학습된 모델 사용
robot_learned_model = 'sac_checkpoint_ep_15000.pth'
test_num           = 1
map_list           = [100]


EXP_NAME = "test2"   # 최상위 결과 폴더 이름
# 빨강 : "#FF0000", 파랑 : "#0000FF", 초록 : "#00FF00"
# 노랑 : "#FFFF00", 보라 : "#FF00FF", 청록 : "#00FFFF"
# 주황 : "#FFA500", 갈색 : "#A52A2A", 분홍 : "#FFC0CB"
# 회색 : "#808080", 검정 : "#000000"
# 흰색 : "#FFFFFF" 
CROWD_COLOR = "#0000FF"
ROBOT_COLOR = "#FF0000"
SINGLE_COLOR_EDGES = True # 선/채움 통일
SHOW_AGENT_HEADING = False # 군중 화살표 끄기
SHOW_ROBOT_HEADING = False
ROBOT_HEADING_SCALE = 1.2

TRAIL_TARGET = "none" # "none" | "crowd" | "robot" | "both"
TRAIL_STYLE = "persist" # "persist" or "fade"
MAX_TRAIL = 2000
ROBOT_STYLE = "circle" # "circle" or "image"
ROBOT_IMAGE_PATH = "assets/robot.png"
ROBOT_IMAGE_SCALE = 5
EXIT_SIZE = 5.0
SNAP_EXIT_TO_BOUNDARY = False

ANNOTATE_ROBOT_PATH = False
ANNOTATE_MODE = "every_n" #'all', 'endpoints'
ANNOTATE_EVERY = 20
ANNOTATE_STYLE = "subway" # "number" | "subway" | "frame"
ANNOTATE_FONTSIZE = 10

SAVE_GRID_ENABLE = True
SAVE_GRID_EVERY = 4




#     # --- NEW: 연속 렌더러 준비 ---
#     renderer = ContinuousRenderer(
#         world_size=(50.0, 50.0),
#         # 색상 테마 한 번에 변경
#         crowd_colors={0:"#4e79a7", 1:"#4e79a7", 2:"#76b7b2", "else":"#59a14f"},
#         robot_color="#e15759",
#         single_color_edges=True,      # 선/채움 통일
#         # 방향 화살표
#         show_agent_heading=False,     # 군중 화살표 끄기
#         show_robot_heading=True,
#         robot_heading_scale=1.2,
#         # 궤적
#         trail_target="both",          # "none" | "crowd" | "robot" | "both"
#         trail_style="fade",           # "persist"로 두면 계속 남김
#         max_trail=60,
#         # 로봇 모양
#         robot_style="image",          # "circle" | "image"
#         robot_image_path="assets/robot.png",  # 있으면 사용
#         robot_image_scale=1.6,        # 이미지 크기
#         # 출구 붙이기
#         exit_size=5.0,
#         snap_exit_to_boundary=True,
#     )

# # ------------------------- #

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

# === [run_one_episode 수정: 연속 렌더 저장 로직 삽입] ===
# def run_one_episode(map_id: int):
#     step_num = 0
#     env = model.FightingModel(
#         number_of_agents,
#         width=MAP_WIDTH,
#         height=MAP_HEIGHT,
#         model_num=map_id,
#         robot=robot_version
#     )
#     if (robot_version == 'Q'):
#         env.use_model(robot_learned_model)

#     # --- NEW: 연속 렌더러 준비 ---
#     renderer = ContinuousRenderer(
#         world_size=(50.0, 50.0),
#         # 색상 테마 한 번에 변경
#         crowd_colors = {0:CROWD_COLOR, 1:CROWD_COLOR, 2:CROWD_COLOR},
#         robot_color = ROBOT_COLOR,
#         single_color_edges= SINGLE_COLOR_EDGES,      # 선/채움 통일
#         # 방향 화살표
#         show_agent_heading= SHOW_AGENT_HEADING,     # 군중 화살표 끄기
#         show_robot_heading= SHOW_ROBOT_HEADING,
#         robot_heading_scale= ROBOT_HEADING_SCALE,
#         # 궤적
#         trail_target= TRAIL_TARGET,
#         trail_style= TRAIL_STYLE,           # "persist"로 두면 계속 남김
#         max_trail= MAX_TRAIL,
#         # 로봇 모양
#         robot_style= ROBOT_STYLE,          # "circle" | "image"
#         robot_image_path= ROBOT_IMAGE_PATH,  # 있으면 사용
#         robot_image_scale= ROBOT_IMAGE_SCALE,        # 이미지 크기
#         # 출구 붙이기
#         exit_size= EXIT_SIZE,
#         snap_exit_to_boundary = SNAP_EXIT_TO_BOUNDARY,

#         annotate_robot_path = ANNOTATE_ROBOT_PATH,
#         annotate_mode=ANNOTATE_MODE, #'all', 'endpoints'
#         annotate_every=ANNOTATE_EVERY,
#         annotate_style = ANNOTATE_STYLE, # "number" | "subway" | "frame"
#         annotate_fontsize = ANNOTATE_FONTSIZE
#     )
#     save_rgb_every = (visualization_mode == 'cont_png_every')
#     save_mp4 = (visualization_mode == 'cont_mp4')
#     save_last_png = (visualization_mode == 'cont_png')
#     collected_frames = []  # MP4용 프레임 버퍼

#     # (지연 생성: 첫 draw 시에 생성)
#     def ensure_renderer():
#         nonlocal renderer
#         if renderer is None:
#             renderer = ContinuousRenderer(
#                 world_size=(float(MAP_WIDTH), float(MAP_HEIGHT)),
#                 show_axes=False
#             )

#     episode_log = []
#     try:
#         while True:
#             env.step()
#             step_num += 1
#             alive = env.alived_agents()
#             episode_log.append(alive)

#             # --- NEW: 프레임 수집/저장 ---
#             if visualization_mode != 'off':
#                 ensure_renderer()
#                 if save_mp4:
#                     # 매 스텝 프레임 수집 (필요시 간격 저장으로 바꿀 수 있음)
#                     rgb = renderer.draw(env)
#                     collected_frames.append(rgb)
#                 elif save_rgb_every and (step_num % VIS_SAVE_EVERY == 0 or alive < 1):
#                     # test_dir을 아직 모름: 반환값으로 test_dir 받을 수 없으니 상위(main)에서 저장
#                     # → 대신 run_one_episode가 프레임 묶음을 반환하고, main에서 저장하도록 변경
#                     rgb = renderer.draw(env)
#                     collected_frames.append(('PNG', step_num, rgb))  # PNG 태그로 표시
#                 elif save_last_png:
#                     # 마지막 한 장만 나중에 찍기 위해 매 스텝 갱신
#                     rgb = renderer.draw(env)
#                     collected_frames = [('LAST', step_num, rgb)]  # 항상 마지막으로 덮어쓰기

#             if alive < 1:
#                 evacuated_all = True
#                 all_life = env.calculate_all_agents_life_time()
#                 # --- 반환값에 프레임 묶음 추가 ---
#                 return step_num, evacuated_all, all_life, episode_log, collected_frames
#             if step_num > max_step_num:
#                 evacuated_all = False
#                 all_life = env.calculate_all_agents_life_time()
#                 return step_num, evacuated_all, all_life, episode_log, collected_frames
#     finally:
#         del env

def run_one_episode(map_id: int):  # ✅ Py3.8 호환
    step_num = 0
    env = model.FightingModel(
        number_of_agents,
        width=MAP_W,
        height=MAP_H,
        model_num=map_id,
        robot=robot_version
    )
    if (robot_version == 'Q'):
        env.use_model(robot_learned_model)

    renderer = ContinuousRenderer(
        world_size=(MAP_H, MAP_W),
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
        annotate_every=ANNOTATE_EVERY,            # if saver is not None:
            #     saver.maybe_save(step_num, env)
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

            # --- NEW: 50x50 state 그레이 이미지 저장 ---


            # --- 연속 렌더 프레임 수집/저장 버퍼링 ---
            if visualization_mode != 'off':
                if save_mp4:
                    # 메모리 아끼려면 VIS_SAVE_EVERY 간격으로만 수집해도 OK
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(rgb)
                elif save_rgb_every and (step_num % VIS_SAVE_EVERY == 0 or alive < 1):
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(('PNG', step_num, rgb))
                elif save_last_png:
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames = [('LAST', step_num, rgb)]  # 마지막만 유지

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

# def main():
#     result_root = init_result_root(EXP_NAME)
#     print(f"[INFO] Result root at: {result_root}")
#     ...

def main():
    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    global_test_index = 0

    for j in range(run_iteration):
        print(f"\n=== Iteration {j+1}/{run_iteration} ===")
        for map_id in map_list:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{robot_version}")
            ensure_dir(map_robot_dir)

            evac_times, life_times, episode_logs = [], [], []

            for test_i in range(test_num):
                unique_test_i = global_test_index
                test_dirname = f"Result_{map_id}_{robot_version}_{unique_test_i}"
                test_dir = os.path.join(map_robot_dir, test_dirname)

                recreate_dir(test_dir)

                # ★★ state grid 저장 디렉토리 & saver 생성
                if SAVE_GRID_ENABLE:
                    grid_out_dir = os.path.join(test_dir, "state_grid")
                    #saver = GridStateSaver(grid_out_dir, every_steps=SAVE_GRID_EVERY)
                else:
                    saver = None

                # 에피소드 실행 (saver 주입!)
                steps, evacuated_all, all_life, ep_log, vis_frames = run_one_episode(map_id)

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
                    # ❗ 버그 픽스: step_id -> step_n (수집 시 변수와 일치)
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

                global_test_index += 1

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

if __name__ == "__main__":
    main()
