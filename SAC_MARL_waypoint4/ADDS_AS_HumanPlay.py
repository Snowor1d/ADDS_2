import pygame
import sys
import math
import os
import pickle
import time
import shutil
import re
import numpy as np

# model.py, ADDS_AS_reinforcement.py (동일 폴더 or 경로 맞춰 수정)
from model import FightingModel
from ADDS_AS_reinforcement import ReplayBuffer
from config import *
from continuous_renderer import ContinuousRenderer

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

############################
# 실험/기록 파라미터
############################
MAP_NUM_FOR_RUN = 1503        # 원하는 맵 번호(-1은 내부 랜덤 로직)
ROBOT_VERSION_FOR_MODEL = 'Q' # 모델에는 'Q'로 넘기되, 사람이 직접 action 전달
ROBOT_VERSION_FOR_LOG   = 'H' # 결과 폴더명에는 'H'로 기록해 비교군 명확화
EXP_NAME = "0201/seheon_play"
MAX_STEPS  = 3000
MAX_EPISODES = 3
ROBOT_NUM = 3

############################
# 상단 설정 (시뮬/렌더 타이밍)
############################
TARGET_SIM_FPS    = 10
TARGET_RENDER_FPS = 30
RENDER_EVERY      = max(1, TARGET_SIM_FPS // TARGET_RENDER_FPS)

############################
# 레이아웃 / 화면
############################
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 1000

CELL_SIZE = 7.5
MAP_CW = int(MAP_W * CELL_SIZE)
MAP_CH = int(MAP_H * CELL_SIZE)

PANEL_RIGHT_WIDTH = 200
PADDING = 10

MAP_OFFSET_X = PADDING
MAP_OFFSET_Y = (SCREEN_HEIGHT - MAP_CH) // 2

JOYSTICK_RADIUS = 60
KNOB_RADIUS = 15
MAX_MOVE = 2.0
MAX_ROBOTS_UI = 3   # UI에서 지원할 최대 로봇 수

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY  = (200, 200, 200)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW= (255, 255, 0)
DARK_GREY = (150, 150, 150)

############################
# PyGame 초기화
############################
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.joystick.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ADDS Imitation Learning - Multi Robot")
clock = pygame.time.Clock()

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)   # 첫 번째 장치
    joystick.init()
    print(f"[INFO] Joystick connected: {joystick.get_name()}")
else:
    print("[INFO] No physical joystick detected. Fallback to on-screen UI.")

pygame.display.set_caption("ADDS Imitation Learning - Multi Robot")
clock = pygame.time.Clock()

def apply_deadzone(val, dz=0.12):
    if abs(val) < dz:
        return 0.0
    return val

############################
# 리플레이 & 리워드 로그
############################
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_guide_game_imitation")
os.makedirs(log_dir, exist_ok=True)
reward_log_file = os.path.join(log_dir, "total_reward_imitation.txt")

############################
# 연속 공간 렌더 옵션
############################
USE_CONTINUOUS_RENDERER = True
CONT_VIS_MODE = "live"             # "live" | "mp4" | "png_every" | "png_last"
CONT_SAVE_EVERY = 1
CONT_FPS = 20
CONT_OUT_DPI = 200
CONT_BITRATE = 8000

CONT_CROWD_COLOR = "#0000FF"
CONT_ROBOT_COLOR = "#FF0000"
CONT_SINGLE_COLOR_EDGES = True
CONT_SHOW_AGENT_HEADING = False
CONT_SHOW_ROBOT_HEADING = True
CONT_ROBOT_HEADING_SCALE = 3
CONT_TRAIL_TARGET = "none"         # "none"|"crowd"|"robot"|"both"
CONT_TRAIL_STYLE = "persist"       # "persist"|"fade"
CONT_MAX_TRAIL = 2000
CONT_ROBOT_STYLE = "image"         # "circle"|"image"
CONT_ROBOT_IMAGE_PATH = "images/robot.png"
CONT_EXIT_SIZE = 5.0
CONT_SNAP_EXIT_TO_BOUNDARY = True
CONT_ANNOTATE_PATH = False
CONT_ANNOTATE_MODE = "every_n"     # "all"|"endpoints"|"every_n"
CONT_ANNOTATE_EVERY = 10
CONT_ANNOTATE_STYLE = "subway"     # "number"|"subway"|"frame"
CONT_ANNOTATE_FONTSIZE = 12

############################
# 결과 기록 유틸
############################
def result_root():
    root = os.path.join(home_dir, f"Result_data_{EXP_NAME}")
    os.makedirs(root, exist_ok=True)
    return root

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def recreate_dir(path: str):
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

def list_test_dirs(map_id: int, robot_ver: str):
    base = os.path.join(result_root(), f"Result_{map_id}", f"Result_{map_id}_{robot_ver}")
    if not os.path.isdir(base):
        return [], base
    pattern = re.compile(rf"^Result_{map_id}_{robot_ver}_(\d+)$")
    subdirs = []
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and pattern.match(name):
            subdirs.append(full)
    return sorted(subdirs), base

def read_metrics_from_dir(test_dir: str):
    path = os.path.join(test_dir, "metrics.txt")
    evac, life = None, None
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("evacuation_100_time="):
                    evac = float(line.strip().split("=", 1)[1])
                elif line.startswith("all_agents_life_time="):
                    life = float(line.strip().split("=", 1)[1])
    return evac, life

def read_episode_log_from_dir(test_dir: str):
    path = os.path.join(test_dir, "episode_log.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    import ast
    try:
        arr = ast.literal_eval(txt)
        return list(arr)
    except Exception:
        return []

def aggregate_over_existing_tests(map_id: int, robot_ver: str):
    test_dirs, base = list_test_dirs(map_id, robot_ver)
    evac_vals, life_vals, logs = [], [], []
    for td in test_dirs:
        evac, life = read_metrics_from_dir(td)
        if evac is not None:
            evac_vals.append(evac)
        if life is not None:
            life_vals.append(life)
        logs.append(read_episode_log_from_dir(td))

    avg_evac = float(np.mean(evac_vals)) if evac_vals else float("nan")
    avg_life = float(np.mean(life_vals)) if life_vals else float("nan")
    write_txt(
        os.path.join(base, "avg_metrics.txt"),
        f"avg_evacuation_100_time={avg_evac}\navg_all_agents_life_time={avg_life}\n"
    )

    if logs:
        max_len = max(len(l) for l in logs)
        mean_series, min_series, max_series = [], [], []
        for i in range(max_len):
            bucket = [l[i] for l in logs if len(l) > i]
            if bucket:
                mean_series.append(float(np.mean(bucket)))
                min_series.append(int(np.min(bucket)))
                max_series.append(int(np.max(bucket)))
        save_step_series(os.path.join(base, "episode_log_mean.txt"), mean_series)
        save_step_series(os.path.join(base, "episode_log_min.txt"),  min_series)
        save_step_series(os.path.join(base, "episode_log_max.txt"),  max_series)

def get_next_test_index(map_id: int, robot_ver: str):
    test_dirs, base = list_test_dirs(map_id, robot_ver)
    if not test_dirs:
        return 0
    nums = []
    pat = re.compile(rf"^Result_{map_id}_{robot_ver}_(\d+)$")
    for d in test_dirs:
        name = os.path.basename(d)
        m = pat.match(name)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 0

############################
# UI / 그리기 유틸
############################
def draw_text(surface, text, x, y, color=(0, 0, 0), font_size=20):
    font = pygame.font.SysFont("Arial", font_size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

def draw_joystick(surface, center, radius, knob_pos, label=None, selected=False):
    base_color = (180, 180, 180) if not selected else (140, 200, 255)
    knob_color = RED if not selected else (255, 100, 100)

    pygame.draw.circle(surface, base_color, center, radius)
    pygame.draw.circle(surface, BLACK, center, radius, 2)
    pygame.draw.circle(surface, knob_color, (int(knob_pos[0]), int(knob_pos[1])), KNOB_RADIUS)

    if label is not None:
        draw_text(surface, label, center[0] - 40, center[1] - radius - 28, color=BLACK, font_size=18)

def clamp_knob_to_circle(cx, cy, knob_x, knob_y, radius):
    dx = knob_x - cx
    dy = knob_y - cy
    dist = math.hypot(dx, dy)
    if dist > radius:
        scale = radius / dist
        dx *= scale
        dy *= scale
    return cx + dx, cy + dy

def get_joystick_action(joystick_center, knob_pos, max_move=MAX_MOVE):
    cx, cy = joystick_center
    kx, ky = knob_pos
    dx = kx - cx
    dy = ky - cy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0
    ratio = dist / JOYSTICK_RADIUS
    scaled_dx = (dx / dist) * ratio * max_move
    scaled_dy = (dy / dist) * ratio * max_move
    return scaled_dx, scaled_dy

def get_joystick_centers(n_robots):
    """
    우측 패널 안에 로봇 수만큼 조이스틱을 세로 배치
    최대 3대 기준
    """
    n = min(n_robots, MAX_ROBOTS_UI)
    centers = []
    panel_x_center = SCREEN_WIDTH - PANEL_RIGHT_WIDTH // 2

    if n <= 0:
        return centers
    elif n == 1:
        ys = [SCREEN_HEIGHT - 120]
    elif n == 2:
        ys = [SCREEN_HEIGHT - 260, SCREEN_HEIGHT - 110]
    else:
        ys = [SCREEN_HEIGHT - 410, SCREEN_HEIGHT - 260, SCREEN_HEIGHT - 110]

    for y in ys[:n]:
        centers.append((panel_x_center, y))
    return centers

def draw_environment(surface, env_map):
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                continue
            rect = (MAP_OFFSET_X + y * CELL_SIZE, MAP_OFFSET_Y + x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if val == 0:
                color = WHITE
            elif val == 60:
                color = BLUE
            elif val == 100:
                color = GREEN
            elif val == 140:
                color = YELLOW
            elif val == 200:
                color = RED
            else:
                color = DARK_GREY
            pygame.draw.rect(surface, color, rect)

    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                rect = (MAP_OFFSET_X + y * CELL_SIZE, MAP_OFFSET_Y + x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, BLACK, rect)

def save_continuous_mp4(frames_rgb, out_path, fps=20, dpi=200, bitrate=8000):
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    fig = plt.figure(figsize=(w / MAP_W, h / MAP_H), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.axis('off')
    im = ax.imshow(frames_rgb[0])
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    with writer.saving(fig, out_path, dpi=dpi):
        for fr in frames_rgb:
            im.set_data(fr)
            writer.grab_frame()
    plt.close(fig)

def np_rgb_to_surface(rgb):
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

def draw_side_panel(surface):
    panel_rect = (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0, PANEL_RIGHT_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(surface, (245, 245, 245), panel_rect)
    pygame.draw.line(
        surface,
        (200, 200, 200),
        (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0),
        (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, SCREEN_HEIGHT),
        2
    )

####################################
# 메인 함수
####################################
def main():
    # replay_buffer = ReplayBuffer(capacity=int(1e5))
    episode_count = 0
    running = True

    root = result_root()
    map_dir = os.path.join(root, f"Result_{MAP_NUM_FOR_RUN}")
    ensure_dir(map_dir)
    map_robot_dir = os.path.join(map_dir, f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}")
    ensure_dir(map_robot_dir)

    while running and episode_count < MAX_EPISODES:
        env_model = FightingModel(
            number_agents=CROWD_NUMBER_MIN,
            width=int(MAP_W),
            height=int(MAP_H),
            model_num=MAP_NUM_FOR_RUN,
            robot=ROBOT_VERSION_FOR_MODEL,
            robot_num = ROBOT_NUM
        )

        state = np.array(env_model.return_current_image(H=MAP_H, W=MAP_W), dtype=np.float32)

        # ===== 멀티 로봇 가져오기 =====
        robots = list(getattr(env_model, "robots", []))
        n_robots = len(robots)
        if n_robots == 0:
            print("[WARNING] env_model.robots is empty.")

        # UI는 최대 3대 기준
        ui_robot_count = min(n_robots, MAX_ROBOTS_UI)
        joystick_centers = get_joystick_centers(ui_robot_count)
        knob_positions = [list(center) for center in joystick_centers]
        dragging_idx = None
        selected_robot_idx = 0 if ui_robot_count > 0 else None
        robot_actions = [(0.0, 0.0) for _ in range(n_robots)]

        # 연속 렌더러
        renderer = None
        collected_frames = []
        if USE_CONTINUOUS_RENDERER:
            renderer = ContinuousRenderer(
                world_size=(MAP_H, MAP_W),
                crowd_colors={0: CONT_CROWD_COLOR, 1: CONT_CROWD_COLOR, 2: CONT_CROWD_COLOR},

                show_agent_heading=CONT_SHOW_AGENT_HEADING,
                agent_heading_scale=1.8,
                agent_heading_color="#000000",
                agent_heading_linewidth=1.5,
                agent_heading_mutation_scale=9.0,

                robot_color=CONT_ROBOT_COLOR,
                single_color_edges=CONT_SINGLE_COLOR_EDGES,
                show_robot_heading=CONT_SHOW_ROBOT_HEADING,
                robot_heading_scale=CONT_ROBOT_HEADING_SCALE,
                trail_target=CONT_TRAIL_TARGET,
                trail_style=CONT_TRAIL_STYLE,
                max_trail=CONT_MAX_TRAIL,
                robot_style=CONT_ROBOT_STYLE,
                robot_image_path=CONT_ROBOT_IMAGE_PATH,
                exit_size=CONT_EXIT_SIZE,
                snap_exit_to_boundary=CONT_SNAP_EXIT_TO_BOUNDARY,
                annotate_robot_path=CONT_ANNOTATE_PATH,
                annotate_mode=CONT_ANNOTATE_MODE,
                annotate_every=CONT_ANNOTATE_EVERY,
                annotate_style=CONT_ANNOTATE_STYLE,
                annotate_fontsize=CONT_ANNOTATE_FONTSIZE,
            )

        step_count = 0
        total_reward = 0.0
        done = False
        episode_count += 1
        episode_log = []

        sim_acc = 0.0
        last_time = time.perf_counter()
        sim_dt = 1.0 / TARGET_SIM_FPS

        while running and not done:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            sim_acc += dt

            # ===== 이벤트 처리 =====
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    dragging_idx = None
                    for i, (kx, ky) in enumerate(knob_positions):
                        if math.hypot(mx - kx, my - ky) <= KNOB_RADIUS + 5:
                            dragging_idx = i
                            selected_robot_idx = i
                            break

                elif event.type == pygame.MOUSEBUTTONUP:
                    if dragging_idx is not None:
                        knob_positions[dragging_idx] = list(joystick_centers[dragging_idx])
                    dragging_idx = None

                elif event.type == pygame.MOUSEMOTION and dragging_idx is not None:
                    mx, my = pygame.mouse.get_pos()
                    cx, cy = joystick_centers[dragging_idx]
                    new_x, new_y = clamp_knob_to_circle(cx, cy, mx, my, JOYSTICK_RADIUS)
                    knob_positions[dragging_idx] = [new_x, new_y]

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_1 and ui_robot_count >= 1:
                        selected_robot_idx = 0
                    elif event.key == pygame.K_2 and ui_robot_count >= 2:
                        selected_robot_idx = 1
                    elif event.key == pygame.K_3 and ui_robot_count >= 3:
                        selected_robot_idx = 2

            if not running:
                break

            # ===== 선택된 조이스틱을 키보드로 조작 =====
            keys = pygame.key.get_pressed()
            move_step = 5.0

            if selected_robot_idx is not None and 0 <= selected_robot_idx < ui_robot_count:
                kx, ky = knob_positions[selected_robot_idx]
                cx, cy = joystick_centers[selected_robot_idx]

                if keys[pygame.K_UP]:
                    ky -= move_step
                if keys[pygame.K_DOWN]:
                    ky += move_step
                if keys[pygame.K_LEFT]:
                    kx -= move_step
                if keys[pygame.K_RIGHT]:
                    kx += move_step

                kx, ky = clamp_knob_to_circle(cx, cy, kx, ky, JOYSTICK_RADIUS)
                knob_positions[selected_robot_idx] = [kx, ky]

            # ===== 각 로봇 action 계산 =====
            robot_actions = [(0.0, 0.0) for _ in range(n_robots)]
            
            # 1) 기본은 기존 UI 조이스틱 값 반영
            for i in range(ui_robot_count):
                cx, cy = joystick_centers[i]
                kx, ky = knob_positions[i]
                user_dx, user_dy = get_joystick_action((cx, cy), (kx, ky))
                env_dx, env_dy = user_dx, -user_dy
                robot_actions[i] = (env_dx, env_dy)

            # 2) 물리 조이스틱이 있으면 Robot 1(action index 0)을 덮어씀
            if joystick is not None and n_robots > 0:
                pygame.event.pump()  # 조이스틱 상태 갱신

                axis_x = apply_deadzone(joystick.get_axis(0))   # 좌우
                axis_y = apply_deadzone(joystick.get_axis(1))   # 상하

                env_dx = axis_x * MAX_MOVE
                env_dy = -axis_y * MAX_MOVE   # 기존 코드와 좌표계 맞춤

                robot_actions[0] = (env_dx, env_dy)

                # 화면 조이스틱도 같이 움직여 보이게 하고 싶으면
                if ui_robot_count >= 1:
                    cx, cy = joystick_centers[0]
                    knob_positions[0] = [
                        cx + axis_x * JOYSTICK_RADIUS,
                        cy + axis_y * JOYSTICK_RADIUS
                    ]

            # ui_robot_count보다 많은 로봇은 0 action
            for i in range(ui_robot_count, n_robots):
                robot_actions[i] = (0.0, 0.0)

            # ui_robot_count보다 많은 로봇이 있으면 나머지는 0 action
            for i in range(ui_robot_count, n_robots):
                robot_actions[i] = (0.0, 0.0)

            # ===== 고정 스텝 시뮬레이션 =====
            did_step = False
            while sim_acc >= sim_dt and not done:
                # 각 로봇에 action 전달
                for rb, (env_dx, env_dy) in zip(robots, robot_actions):
                    rb.receive_action([env_dx, env_dy])

                current_state = state
                env_model.step()

                img_gray = env_model.return_current_image(H=MAP_H, W=MAP_W)
                reward = 0

                alive = env_model.alived_agents()
                episode_log.append(alive)

                if alive <= 0:
                    done = True
                    reward += 10.0

                next_state = np.array(img_gray, dtype=np.float32)

                # replay_buffer.push(
                #     current_state,
                #     np.array(robot_actions, np.float32),
                #     reward,
                #     next_state,
                #     float(done)
                # )

                total_reward += reward
                step_count += 1
                state = next_state
                sim_acc -= sim_dt
                did_step = True

                if step_count >= MAX_STEPS:
                    done = True

            # ===== 렌더링 =====
            do_render = did_step and (step_count % RENDER_EVERY == 0)
            if do_render:
                screen.fill(WHITE)
                draw_side_panel(screen)

                if USE_CONTINUOUS_RENDERER:
                    rgb = renderer.draw(env_model, step=step_count)
                    surf = np_rgb_to_surface(rgb)
                    surf = pygame.transform.scale(surf, (MAP_CW, MAP_CH))
                    screen.blit(surf, (MAP_OFFSET_X, MAP_OFFSET_Y))

                    if CONT_VIS_MODE == "mp4":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(rgb)
                    elif CONT_VIS_MODE == "png_every":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(("PNG", step_count, rgb))
                    elif CONT_VIS_MODE == "png_last":
                        collected_frames = [("LAST", step_count, rgb)]
                else:
                    draw_environment(screen, np.rot90(img_gray, k=1))

                # ===== HUD =====
                txt_x = SCREEN_WIDTH - PANEL_RIGHT_WIDTH + 10
                draw_text(screen, f"Episode: {episode_count}", txt_x, 10, color=BLACK, font_size=22)
                draw_text(screen, f"Step: {step_count}", txt_x, 35, color=BLACK, font_size=22)
                draw_text(screen, f"Alive: {alive}", txt_x, 60, color=BLACK, font_size=22)
                draw_text(screen, f"EpiTotal: {total_reward:.3f}", txt_x, 85, color=BLACK, font_size=22)
                draw_text(screen, "ESC to quit", txt_x, 110, color=(128, 0, 0), font_size=18)

                draw_text(screen, f"Robots: {n_robots}", txt_x, 140, color=BLACK, font_size=18)
                for i, (adx, ady) in enumerate(robot_actions[:MAX_ROBOTS_UI]):
                    mark = "*" if selected_robot_idx == i else " "
                    draw_text(
                        screen,
                        f"{mark}R{i+1}: ({adx:.2f}, {ady:.2f})",
                        txt_x,
                        165 + i * 22,
                        color=BLACK,
                        font_size=16
                    )

                draw_text(screen, "1/2/3: select robot", txt_x, 240, color=(70, 70, 70), font_size=15)
                draw_text(screen, "Arrows: move selected", txt_x, 260, color=(70, 70, 70), font_size=15)
                draw_text(screen, "Mouse: drag joystick", txt_x, 280, color=(70, 70, 70), font_size=15)

                # ===== 여러 조이스틱 그리기 =====
                for i, center in enumerate(joystick_centers):
                    label = f"Robot {i+1}"
                    draw_joystick(
                        screen,
                        center,
                        JOYSTICK_RADIUS,
                        knob_positions[i],
                        label=label,
                        selected=(selected_robot_idx == i)
                    )

                pygame.display.flip()

            clock.tick(TARGET_RENDER_FPS)

        # ===== 에피소드 종료 기록 =====
        if done:
            print(f"[Episode {episode_count} finished] steps={step_count}, total_reward={total_reward:.2f}")
            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

            test_i = get_next_test_index(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)
            test_dirname = f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}_{test_i}"
            test_dir = os.path.join(map_robot_dir, test_dirname)
            recreate_dir(test_dir)

            evacuation_100_time = step_count if (len(episode_log) > 0 and episode_log[-1] <= 0) else MAX_STEPS
            all_agents_life_time = env_model.calculate_all_agents_life_time()

            write_txt(
                os.path.join(test_dir, "metrics.txt"),
                f"evacuation_100_time={evacuation_100_time}\n"
                f"all_agents_life_time={all_agents_life_time}\n"
            )
            write_list_txt(os.path.join(test_dir, "episode_log.txt"), episode_log)

            if USE_CONTINUOUS_RENDERER:
                if CONT_VIS_MODE == "mp4" and collected_frames:
                    stride = max(1, CONT_SAVE_EVERY)
                    eff_fps = max(1, int(CONT_FPS / stride))
                    out_mp4 = os.path.join(test_dir, "continuous.mp4")
                    save_continuous_mp4(
                        collected_frames,
                        out_mp4,
                        fps=eff_fps,
                        dpi=CONT_OUT_DPI,
                        bitrate=CONT_BITRATE
                    )
                elif CONT_VIS_MODE == "png_every" and collected_frames:
                    png_dir = os.path.join(test_dir, "continuous_pngs")
                    os.makedirs(png_dir, exist_ok=True)
                    for tag, st, rgb in collected_frames:
                        if tag != "PNG":
                            continue
                        plt.imsave(os.path.join(png_dir, f"frame_{st:05d}.png"), rgb, dpi=CONT_OUT_DPI)
                elif CONT_VIS_MODE == "png_last" and collected_frames:
                    _, st, rgb = collected_frames[-1]
                    out_png = os.path.join(test_dir, f"continuous_last_{st:05d}.png")
                    plt.imsave(out_png, rgb, dpi=CONT_OUT_DPI)

            aggregate_over_existing_tests(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)

    pygame.quit()
    print(f"Reward log saved to {reward_log_file}")
    print("Bye ~")

if __name__ == "__main__":
    main()