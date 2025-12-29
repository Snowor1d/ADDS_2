import os
import re
import ast
import shutil
import numpy as np
import faulthandler
faulthandler.enable()

from continuous_renderer import ContinuousRenderer
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from config import *  # MAP_W, MAP_H 등이 있다고 가정
import model  # FightingModel


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

VIS_SAVE_EVERY = 5

# ------------------------- #
# 실험 파라미터
visualization_mode = 'cont_png_every'  # 'off', 'cont_mp4', 'cont_png_every', 'cont_png'
run_iteration      = 1
number_of_agents   = 30
max_step_num       = 100
robot_version      = 'Q'  # 'T','R','Q'
robot_learned_model = 'sac_checkpoint_ep_5000.pth'
test_num           = 1
map_list           = [100]

EXP_NAME = "test_100"

CROWD_COLOR = "#0000FF"
ROBOT_COLOR = "#FF0000"
SINGLE_COLOR_EDGES = True
SHOW_AGENT_HEADING = False
SHOW_ROBOT_HEADING = False
ROBOT_HEADING_SCALE = 1.2

TRAIL_TARGET = "none"
TRAIL_STYLE = "persist"
MAX_TRAIL = 2000
ROBOT_STYLE = "circle"
ROBOT_IMAGE_PATH = "assets/robot.png"
ROBOT_IMAGE_SCALE = 5
EXIT_SIZE = 5.0
SNAP_EXIT_TO_BOUNDARY = False

ANNOTATE_ROBOT_PATH = False
ANNOTATE_MODE = "every_n"
ANNOTATE_EVERY = 20
ANNOTATE_STYLE = "subway"
ANNOTATE_FONTSIZE = 10


# ------------------------- #
# utils
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


def _to_gray_png(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    # (C,H,W) → (H,W) : 첫 채널만 사용
    if x.ndim == 3:
        x = x[0]

    # float → [0,1] 클립
    if x.dtype.kind == "f":
        x = np.clip(x, 0.0, 1.0)

    # 180도 회전 (debug 기준 정렬)
    x = np.flip(np.flip(x, axis=-1), axis=-2)

    return x


# ------------------------- #
# 폴더 스캔/로딩
def _test_dir_pattern(map_id: int, robot_ver: str):
    # Result_{map_id}_{robot_ver}_{n}
    return re.compile(rf"^Result_{map_id}_{re.escape(robot_ver)}_(\d+)$")

def list_existing_tests(map_robot_dir: str, map_id: int, robot_ver: str):
    """
    map_robot_dir 아래에 있는 Result_{map_id}_{robot_ver}_{n} 폴더들을 찾아
    (n, full_path) 리스트로 반환.
    """
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
            n = int(m.group(1))
            out.append((n, full))
    out.sort(key=lambda x: x[0])
    return out

def get_next_test_index(map_robot_dir: str, map_id: int, robot_ver: str) -> int:
    tests = list_existing_tests(map_robot_dir, map_id, robot_ver)
    if not tests:
        return 0
    return tests[-1][0] + 1

def read_metrics(metrics_path: str):
    """
    metrics.txt:
      evacuation_100_time=...
      all_agents_life_time=...
    """
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
    """
    episode_log.txt는 write_list_txt로 저장된 "list(...) 문자열".
    ast.literal_eval로 파싱.
    """
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
    """
    map_robot_dir 아래 기존 test 폴더들 전부 읽어서
    evac_times, life_times, episode_logs 반환.
    """
    evac_times = []
    life_times = []
    episode_logs = []

    for n, test_dir in list_existing_tests(map_robot_dir, map_id, robot_ver):
        evac, life = read_metrics(os.path.join(test_dir, "metrics.txt"))
        ep_log = read_episode_log(os.path.join(test_dir, "episode_log.txt"))

        if evac is not None:
            evac_times.append(evac)
        if life is not None:
            life_times.append(life)
        if ep_log is not None:
            episode_logs.append(ep_log)

    return evac_times, life_times, episode_logs


# ------------------------- #
# MP4 저장
def save_continuous_mp4(frames_rgb, out_path, fps=20):
    if not frames_rgb:
        return
    try:
        h, w, _ = frames_rgb[0].shape
        fig = plt.figure(figsize=(w/100, h/100), dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        ax.axis('off')
        im = ax.imshow(frames_rgb[0])

        writer = FFMpegWriter(fps=fps, bitrate=4000)
        with writer.saving(fig, out_path, dpi=100):
            for fr in frames_rgb:
                im.set_data(fr)
                writer.grab_frame()
        plt.close(fig)
    except Exception as e:
        fallback_dir = out_path + "_pngs"
        os.makedirs(fallback_dir, exist_ok=True)
        for i, fr in enumerate(frames_rgb):
            plt.imsave(os.path.join(fallback_dir, f"frame_{i:05d}.png"), fr)
        print(f"[WARN] MP4 저장 실패({e}). PNG 시퀀스로 대체 저장: {fallback_dir}")

 
def ego_crop_from_full_map(full_map: np.ndarray,
                           robot_xy_px: tuple[int, int],
                           ego_size: int,
                           pad_value: int = 50) -> np.ndarray:
    """
    full_map: (H, W) uint8
    robot_xy_px: (ix, iy) in pixel coords (0..W-1, 0..H-1)
    return: (ego_size, ego_size) uint8
    """
    H, W = full_map.shape
    cx, cy = robot_xy_px
    half = ego_size // 2

    # 원하는 crop 좌표(맵 좌표 기준)
    x0, x1 = cx - half, cx - half + ego_size
    y0, y1 = cy - half, cy - half + ego_size

    # 맵과 겹치는 부분
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)

    crop = np.full((ego_size, ego_size), pad_value, dtype=full_map.dtype)

    # crop 안에서 어디에 붙일지 offset
    dx0 = sx0 - x0
    dy0 = sy0 - y0

    crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = full_map[sy0:sy1, sx0:sx1]
    return crop


def downsample_full_map(full_map: np.ndarray, target: int) -> np.ndarray:
    """
    full_map: (H, W) uint8
    return: (target, target) uint8 (adaptive pool)
    """
    x = torch.from_numpy(full_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y = F.adaptive_max_pool2d(x, (target, target))
    return y.squeeze(0).squeeze(0).byte().numpy()

def _robot_world_to_px(env):
    rx, ry = env.robot.xy  # world coords
    ix = int(np.clip(rx / env.width  * MAP_W, 0, MAP_W - 1))
    iy = int(np.clip(ry / env.height * MAP_H, 0, MAP_H - 1))
    return ix, iy

def build_ego_global_frames(env):
    full_map_u8 = env.return_current_image(MAP_H, MAP_W)
    ix, iy = _robot_world_to_px(env)

    ego_u8 = ego_crop_from_full_map(full_map_u8, (ix, iy), EGO_MAP_SIZE, pad_value=50)     # (EGO,EGO)
    glob_u8 = downsample_full_map(full_map_u8, DOWNSAMPLE_MAP_SIZE)                       # (DOWN,DOWN)

    ego_f = ego_u8.astype(np.float32) / 255.0
    glob_f = glob_u8.astype(np.float32) / 255.0
    return ego_f, glob_f

def _to_png_img(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)

    # (C,H,W) -> (H,W,C) 로 변환 (C=1/3/4 모두 가능)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))

    # float이면 [0,1] or [0,255] 형태일 수 있으니 클립
    if x.dtype.kind == "f":
        # 보통 네 파이프라인은 0~1일 가능성이 높음
        x = np.clip(x, 0.0, 1.0)

    x = np.flip(np.flip(x, axis=-1), axis=-2)

    # uint8이면 그대로 OK, float이면 imsave가 알아서 처리함
    return x



# ------------------------- #
# episode
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
        world_size=(float(MAP_W), float(MAP_H)),  # ✅ 보통 (W,H)가 자연스러움
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

    save_rgb_every = (visualization_mode == 'cont_png_every')
    save_mp4 = (visualization_mode == 'cont_mp4')
    save_last_png = (visualization_mode == 'cont_png')
    collected_frames = []

    episode_log = []
    try:
        while True:
            env.step()
            step_num += 1
            alive = env.alived_agents()
            episode_log.append(alive)

            # 시각화 프레임 수집
            if visualization_mode != 'off':
                if save_mp4:
                    rgb = renderer.draw(env, step=step_num)
                    collected_frames.append(rgb)
                elif save_rgb_every and (step_num % VIS_SAVE_EVERY == 0 or alive < 1):
                    rgb = renderer.draw(env, step=step_num)
                    ego_f, glob_f = build_ego_global_frames(env)
                    collected_frames.append(("PNG_EVERY", step_num, rgb, ego_f, glob_f))

                    #collected_frames.append(('PNG', step_num, rgb))
                elif save_last_png:
                    rgb = renderer.draw(env, step=step_num)
                    ego_f, glob_f = build_ego_global_frames(env)
                    collected_frames = [('LAST', step_num, rgb, ego_f, glob_f)]

            if alive < 1:
                evacuated_all = True
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames

            # ✅ max_step_num에서 멈추고 싶으면 >= 가 직관적
            if step_num >= max_step_num:
                evacuated_all = False
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log, collected_frames
    finally:
        del env


# ------------------------- #
# main
def main():
    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    for j in range(run_iteration):
        print(f"\n=== Iteration {j+1}/{run_iteration} ===")

        for map_id in map_list:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{robot_version}")
            ensure_dir(map_robot_dir)

            # ✅ (핵심) 기존 폴더 전부 읽어서 통계에 포함
            evac_times, life_times, episode_logs = load_all_existing_logs(map_robot_dir, map_id, robot_version)

            # ✅ (핵심) 기존 폴더에서 다음 index부터 시작
            start_index = get_next_test_index(map_robot_dir, map_id, robot_version)
            print(f"[Map {map_id}] next_test_index = {start_index} (existing={len(list_existing_tests(map_robot_dir, map_id, robot_version))})")

            for t in range(test_num):
                unique_test_i = start_index + t
                test_dirname = f"Result_{map_id}_{robot_version}_{unique_test_i}"
                test_dir = os.path.join(map_robot_dir, test_dirname)

                # ✅ 덮어쓰기 방지: 기존 있으면 에러 내고 넘어가거나, 더 증가시키는 방식도 가능
                if os.path.exists(test_dir):
                    raise RuntimeError(f"Test dir already exists: {test_dir}")

                os.makedirs(test_dir, exist_ok=False)

                steps, evacuated_all, all_life, ep_log, vis_frames = run_one_episode(map_id)

                evacuation_100_time = steps if evacuated_all else max_step_num
                all_agents_life_time = all_life

                # 이번 결과도 누적 리스트에 추가 (통계 재계산용)
                evac_times.append(float(evacuation_100_time))
                life_times.append(float(all_agents_life_time))
                episode_logs.append(ep_log)

                write_txt(
                    os.path.join(test_dir, "metrics.txt"),
                    f"evacuation_100_time={evacuation_100_time}\n"
                    f"all_agents_life_time={all_agents_life_time}\n"
                )
                write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

                print(f"[Map {map_id}] Test {unique_test_i} -> steps={steps}, evacuated_all={evacuated_all}")

                # 시각화 저장
                if visualization_mode == 'cont_mp4':
                    out_mp4 = os.path.join(test_dir, "continuous.mp4")
                    save_continuous_mp4(vis_frames, out_mp4, fps=20)

                elif visualization_mode == 'cont_png_every':
                    png_dir = os.path.join(test_dir, "continuous_pngs")
                    ego_dir = os.path.join(test_dir, "ego_pngs")
                    glob_dir = os.path.join(test_dir, "global_pngs")
                    os.makedirs(png_dir, exist_ok=True)
                    os.makedirs(ego_dir, exist_ok=True)
                    os.makedirs(glob_dir, exist_ok=True)

                    for item in vis_frames:
                        tag = item[0]
                        if tag != "PNG_EVERY":
                            continue

                        _, step_n, rgb, ego_f, glob_f = item

                        plt.imsave(os.path.join(png_dir,  f"frame_{step_n:05d}.png"), rgb)

                        # ego_f / glob_f가 (H,W)든 (C,H,W)든 저장되게 처리
                        plt.imsave(
                            os.path.join(ego_dir, f"ego_{step_n:05d}.png"),
                            _to_gray_png(ego_f),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0
                        )

                        plt.imsave(
                            os.path.join(glob_dir, f"glob_{step_n:05d}.png"),
                            _to_gray_png(glob_f),
                            cmap="gray",
                            vmin=0.0,
                            vmax=1.0
)
                elif visualization_mode == 'cont_png':
                    if vis_frames:
                        _, step_id, rgb, ego_f, glob_f = vis_frames[-1]

                        out_png = os.path.join(test_dir, f"continuous_last_{step_id:05d}.png")
                        plt.imsave(out_png, rgb)

                        out_ego  = os.path.join(test_dir, f"ego_last_{step_id:05d}.png")
                        out_glob = os.path.join(test_dir, f"glob_last_{step_id:05d}.png")
                    plt.imsave(
                        os.path.join(ego_dir, f"ego_{step_n:05d}.png"),
                        _to_gray_png(ego_f),
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0
                    )

                    plt.imsave(
                        os.path.join(glob_dir, f"glob_{step_n:05d}.png"),
                        _to_gray_png(glob_f),
                        cmap="gray",
                        vmin=0.0,
                        vmax=1.0
                    )

                print(f"[Map {map_id}] Test {unique_test_i} visualization saved.")

            # ✅ (핵심) 폴더 내 “전체 테스트” 기준으로 요약 파일 재작성
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
            save_step_series(os.path.join(map_robot_dir, "episode_log_min.txt"), min_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_max.txt"), max_series)

            print(f"[Map {map_id}] Summary updated (ALL episodes) at {map_robot_dir}")


if __name__ == "__main__":
    main()
