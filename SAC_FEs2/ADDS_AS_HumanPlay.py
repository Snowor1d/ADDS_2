import pygame
import sys
import math
import os
import time
import shutil
import re
import numpy as np

# model.py, ADDS_AS_reinforcement.py (동일 폴더 or 경로 맞춰 수정)
from model import FightingModel
from config import *
from continuous_renderer import ContinuousRenderer

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

############################
# 실험/기록 파라미터
############################
MAP_NUM_FOR_RUN = 4001        # 원하는 맵 번호(-1은 내부 랜덤 로직)
ROBOT_VERSION_FOR_MODEL = 'Q' # 모델에는 'Q'로 넘기되, 사람이 직접 action 전달
ROBOT_VERSION_FOR_LOG   = 'H' # 결과 폴더명에는 'H'로 기록해 비교군 명확화
EXP_NAME = "0201/seheon_play"
MAX_STEPS  = 3000
MAX_EPISODES = 3

############################
# 타이밍/성능 파라미터
############################
TARGET_SIM_FPS    = 5     # 시뮬 목표 Hz (느려져도 입력은 먹게)
TARGET_RENDER_FPS = 20    # 렌더/입력 루프는 더 자주
MAX_SIM_STEPS_PER_FRAME = 1  # 프레임당 시뮬 step 상한 (입력 responsiveness 핵심)
MAX_ACCUM_SEC = 0.25         # death spiral 방지

############################
# 화면 레이아웃
############################
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 1000

PANEL_RIGHT_WIDTH = 200
PADDING = 10

MAP_AREA_W = SCREEN_WIDTH - PANEL_RIGHT_WIDTH - 2*PADDING
MAP_AREA_H = SCREEN_HEIGHT - 2*PADDING
MAP_OFFSET_X = PADDING
MAP_OFFSET_Y = PADDING

# 조이스틱(우측 패널)
JOYSTICK_CENTER = (SCREEN_WIDTH - PANEL_RIGHT_WIDTH // 2, SCREEN_HEIGHT - 120)
JOYSTICK_RADIUS = 60
KNOB_RADIUS = 15
MAX_MOVE = 2.0
KEY_MOVE_STEP = 6.0  # 키로 knob 움직이는 픽셀

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY  = (200, 200, 200)
RED   = (255, 0, 0)

############################
# 연속 렌더 옵션
############################
USE_CONTINUOUS_RENDERER = True
CONT_VIS_MODE = "live"  # "live" | "mp4" | "png_every" | "png_last"
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
CONT_TRAIL_TARGET = "none"
CONT_TRAIL_STYLE = "persist"
CONT_MAX_TRAIL = 2000
CONT_ROBOT_STYLE = "image"
CONT_ROBOT_IMAGE_PATH = "images/robot.png"
CONT_EXIT_SIZE = 5.0
CONT_SNAP_EXIT_TO_BOUNDARY = True
CONT_ANNOTATE_PATH = False
CONT_ANNOTATE_MODE = "every_n"
CONT_ANNOTATE_EVERY = 10
CONT_ANNOTATE_STYLE = "subway"
CONT_ANNOTATE_FONTSIZE = 12

############################
# PyGame 초기화
############################
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ADDS Imitation Learning")
clock = pygame.time.Clock()

############################
# 로그/결과 저장
############################
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_guide_game_imitation")
os.makedirs(log_dir, exist_ok=True)
reward_log_file = os.path.join(log_dir, "total_reward_imitation.txt")

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
                    evac = float(line.strip().split("=",1)[1])
                elif line.startswith("all_agents_life_time="):
                    life = float(line.strip().split("=",1)[1])
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
        if evac is not None: evac_vals.append(evac)
        if life is not None: life_vals.append(life)
        logs.append(read_episode_log_from_dir(td))

    avg_evac = float(np.mean(evac_vals)) if evac_vals else float("nan")
    avg_life = float(np.mean(life_vals)) if life_vals else float("nan")
    write_txt(os.path.join(base, "avg_metrics.txt"),
              f"avg_evacuation_100_time={avg_evac}\navg_all_agents_life_time={avg_life}\n")

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
    test_dirs, _ = list_test_dirs(map_id, robot_ver)
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
# 렌더/입력 유틸
############################
def draw_text(surface, text, x, y, color=(0,0,0), font_size=20):
    font = pygame.font.SysFont("Arial", font_size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

def draw_side_panel(surface):
    panel_rect = (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0, PANEL_RIGHT_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(surface, (245, 245, 245), panel_rect)
    pygame.draw.line(surface, (200, 200, 200),
                     (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0),
                     (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, SCREEN_HEIGHT), 2)

def draw_joystick(surface, center, radius, knob_pos):
    pygame.draw.circle(surface, GREY, center, radius)
    pygame.draw.circle(surface, RED, knob_pos, KNOB_RADIUS)

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
    scaled_dx = (dx/dist) * ratio * max_move
    scaled_dy = (dy/dist) * ratio * max_move
    return scaled_dx, scaled_dy

def np_rgb_to_surface(rgb):
    return pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))

def save_continuous_mp4(frames_rgb, out_path, fps=20, dpi=200, bitrate=8000):
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    fig = plt.figure(figsize=(w/MAP_W, h/MAP_H), dpi=dpi)
    ax = plt.axes([0,0,1,1]); ax.axis('off')
    im = ax.imshow(frames_rgb[0])
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    with writer.saving(fig, out_path, dpi=dpi):
        for fr in frames_rgb:
            im.set_data(fr)
            writer.grab_frame()
    plt.close(fig)

def handle_events(key_dir, knob_x, knob_y, dragging):
    """
    이벤트를 '소비'해서 큐가 쌓이지 않게 함.
    반환: (running, knob_x, knob_y, dragging)
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False, knob_x, knob_y, dragging

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if math.hypot(mx - knob_x, my - knob_y) <= KNOB_RADIUS + 5:
                dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False
            knob_x, knob_y = JOYSTICK_CENTER

        elif event.type == pygame.MOUSEMOTION and dragging:
            mx, my = pygame.mouse.get_pos()
            knob_x, knob_y = clamp_knob_to_circle(
                JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], mx, my, JOYSTICK_RADIUS
            )

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False, knob_x, knob_y, dragging
            elif event.key == pygame.K_UP:    key_dir["up"] = True
            elif event.key == pygame.K_DOWN:  key_dir["down"] = True
            elif event.key == pygame.K_LEFT:  key_dir["left"] = True
            elif event.key == pygame.K_RIGHT: key_dir["right"] = True

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_UP:    key_dir["up"] = False
            elif event.key == pygame.K_DOWN:  key_dir["down"] = False
            elif event.key == pygame.K_LEFT:  key_dir["left"] = False
            elif event.key == pygame.K_RIGHT: key_dir["right"] = False

    return True, knob_x, knob_y, dragging

############################
# 메인
############################
def main():
    episode_count = 0
    running = True

    root = result_root()
    map_dir = os.path.join(root, f"Result_{MAP_NUM_FOR_RUN}")
    ensure_dir(map_dir)
    map_robot_dir = os.path.join(map_dir, f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}")
    ensure_dir(map_robot_dir)

    key_dir = {"up":False, "down":False, "left":False, "right":False}

    while running and episode_count < MAX_EPISODES:
        env_model = FightingModel(
            number_agents=CROWD_NUMBER_MIN,
            width=int(MAP_W),
            height=int(MAP_H),
            model_num=MAP_NUM_FOR_RUN,
            robot=ROBOT_VERSION_FOR_MODEL
        )

        renderer = None
        collected_frames = []
        if USE_CONTINUOUS_RENDERER:
            renderer = ContinuousRenderer(
                world_size=(MAP_H, MAP_W),
                crowd_colors={0:CONT_CROWD_COLOR, 1:CONT_CROWD_COLOR, 2:CONT_CROWD_COLOR},
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

        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False

        sim_acc = 0.0
        last_time = time.perf_counter()
        sim_dt = 1.0 / TARGET_SIM_FPS

        alive = env_model.alived_agents()

        while running and not done:
            # 시간 누적
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            sim_acc += dt
            sim_acc = min(sim_acc, MAX_ACCUM_SEC)

            # 이벤트/입력 처리 (프레임 시작)
            running, knob_x, knob_y, dragging = handle_events(key_dir, knob_x, knob_y, dragging)
            if not running:
                break

            # 키 입력을 knob에 반영 (프레임마다)
            if key_dir["up"]:    knob_y -= KEY_MOVE_STEP
            if key_dir["down"]:  knob_y += KEY_MOVE_STEP
            if key_dir["left"]:  knob_x -= KEY_MOVE_STEP
            if key_dir["right"]: knob_x += KEY_MOVE_STEP
            knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1],
                                                  knob_x, knob_y, JOYSTICK_RADIUS)

            # 현재 action (항상 최신 knob 기준)
            user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))
            env_dx, env_dy = user_dx, -user_dy

            # 시뮬레이션 step (프레임당 상한)
            did_step = False
            steps_this_frame = 0

            while (sim_acc >= sim_dt) and (not done) and (steps_this_frame < MAX_SIM_STEPS_PER_FRAME):
                # step 사이에도 이벤트를 소비해서 입력 지연을 최소화
                running, knob_x, knob_y, dragging = handle_events(key_dir, knob_x, knob_y, dragging)
                if not running:
                    done = True
                    break

                # step 직전에 키 반영(체감 응답성 ↑)
                if key_dir["up"]:    knob_y -= KEY_MOVE_STEP
                if key_dir["down"]:  knob_y += KEY_MOVE_STEP
                if key_dir["left"]:  knob_x -= KEY_MOVE_STEP
                if key_dir["right"]: knob_x += KEY_MOVE_STEP
                knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1],
                                                      knob_x, knob_y, JOYSTICK_RADIUS)
                user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))
                env_dx, env_dy = user_dx, -user_dy

                env_model.robot.receive_action([env_dx, env_dy])
                env_model.step()

                alive = env_model.alived_agents()
                episode_log.append(alive)
                if alive <= 0:
                    done = True
                    total_reward += 10.0

                step_count += 1
                if step_count >= MAX_STEPS:
                    done = True

                sim_acc -= sim_dt
                did_step = True
                steps_this_frame += 1

            # 렌더
            if did_step:
                screen.fill(WHITE)
                draw_side_panel(screen)

                if USE_CONTINUOUS_RENDERER:
                    rgb = renderer.draw(env_model, step=step_count)
                    surf = np_rgb_to_surface(rgb)
                    surf = pygame.transform.smoothscale(surf, (MAP_AREA_W, MAP_AREA_H))
                    screen.blit(surf, (MAP_OFFSET_X, MAP_OFFSET_Y))

                    if CONT_VIS_MODE == "mp4":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(rgb)
                    elif CONT_VIS_MODE == "png_every":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(("PNG", step_count, rgb))
                    elif CONT_VIS_MODE == "png_last":
                        collected_frames = [("LAST", step_count, rgb)]

                # HUD
                txt_x = SCREEN_WIDTH - PANEL_RIGHT_WIDTH + 10
                draw_text(screen, f"Episode: {episode_count}", txt_x, 10, color=BLACK, font_size=22)
                draw_text(screen, f"Step: {step_count}",       txt_x, 35, color=BLACK, font_size=22)
                draw_text(screen, f"Alive: {alive}",           txt_x, 60, color=BLACK, font_size=22)
                draw_text(screen, f"EpiTotal: {total_reward:.3f}", txt_x, 90, color=BLACK, font_size=20)
                draw_text(screen, "ESC to quit", txt_x, 115, color=(128,0,0), font_size=18)

                draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (int(knob_x), int(knob_y)))
                pygame.display.flip()

            clock.tick(TARGET_RENDER_FPS)

        # === 에피소드 종료 기록 ===
        if not running:
            break

        print(f"[Episode {episode_count} finished] steps={step_count}, total_reward={total_reward:.2f}")
        with open(reward_log_file, "a", encoding='utf-8') as f:
            f.write(f"{total_reward}\n")

        test_i = get_next_test_index(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)
        test_dirname = f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}_{test_i}"
        test_dir = os.path.join(map_robot_dir, test_dirname)
        recreate_dir(test_dir)

        evacuation_100_time = step_count if (len(episode_log) > 0 and episode_log[-1] <= 0) else MAX_STEPS
        all_agents_life_time = env_model.calculate_all_agents_life_time()
        write_txt(os.path.join(test_dir, "metrics.txt"),
                  f"evacuation_100_time={evacuation_100_time}\nall_agents_life_time={all_agents_life_time}\n")
        write_list_txt(os.path.join(test_dir, "episode_log.txt"), episode_log)

        if USE_CONTINUOUS_RENDERER:
            if CONT_VIS_MODE == "mp4" and collected_frames:
                stride = max(1, CONT_SAVE_EVERY)
                eff_fps = max(1, int(CONT_FPS / stride))
                out_mp4 = os.path.join(test_dir, "continuous.mp4")
                save_continuous_mp4(collected_frames, out_mp4, fps=eff_fps,
                                    dpi=CONT_OUT_DPI, bitrate=CONT_BITRATE)
            elif CONT_VIS_MODE == "png_every" and collected_frames:
                png_dir = os.path.join(test_dir, "continuous_pngs")
                os.makedirs(png_dir, exist_ok=True)
                for tag, st, rgb in collected_frames:
                    if tag != "PNG": continue
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
