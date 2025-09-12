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
from Start_training import *
from continuous_renderer import ContinuousRenderer

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

############################
# 실험/기록 파라미터
############################
MAP_NUM_FOR_RUN = 26          # 원하는 맵 번호(-1은 내부 랜덤 로직)
ROBOT_VERSION_FOR_MODEL = 'Q' # 모델에는 'Q'로 넘기되, 사람이 직접 action 전달
ROBOT_VERSION_FOR_LOG   = 'H' # 결과 폴더명에는 'H'로 기록해 비교군 명확화
EXP_NAME   = "human_play_image_making"        # 최상위 결과 폴더 접미사: Result_data_{EXP_NAME}
MAX_STEPS  = 3000             # 실패 시 스텝 상한
MAX_EPISODES = 1


############################
# 상단 설정 (시뮬/렌더 타이밍)
############################
TARGET_SIM_FPS    = 10      # 시뮬레이션 60Hz
TARGET_RENDER_FPS = 30      # 화면 갱신 30Hz
RENDER_EVERY      = max(1, TARGET_SIM_FPS // TARGET_RENDER_FPS)  # 2

############################
# 레이아웃 / 화면

############################
SCREEN_WIDTH  = 1000
SCREEN_HEIGHT = 1000

CELL_SIZE = 15
MAP_W = 50 * CELL_SIZE
MAP_H = 50 * CELL_SIZE

PANEL_RIGHT_WIDTH = 200   # 우측 HUD/조이스틱 패널 폭
PADDING = 10              # 좌측 여백

# 맵은 좌측 정렬, 우측엔 패널 공간
MAP_OFFSET_X = PADDING
MAP_OFFSET_Y = (SCREEN_HEIGHT - MAP_H) // 2

# 조이스틱은 우측 패널 중앙 하단
JOYSTICK_CENTER = (SCREEN_WIDTH - PANEL_RIGHT_WIDTH // 2, SCREEN_HEIGHT - 120)
JOYSTICK_RADIUS = 60
KNOB_RADIUS = 15
MAX_MOVE = 2.0

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
pygame.display.set_caption("ADDS Imitation Learning")
clock = pygame.time.Clock()

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
USE_CONTINUOUS_RENDERER = True     # False면 격자 렌더(draw_environment)
CONT_VIS_MODE = "mp4"             # "live" | "mp4" | "png_every" | "png_last"
CONT_SAVE_EVERY = 1                # png_every/mp4 수집 간격(스텝)
CONT_FPS = 20                      # mp4 저장 FPS(프레임 stride 보정과 함께 사용)
CONT_OUT_DPI = 200                 # mp4/PNG 저장 해상도
CONT_BITRATE = 8000                # mp4 인코딩 비트레이트

# 색/스타일
CONT_CROWD_COLOR = "#4e79a7"
CONT_ROBOT_COLOR = "#e15759"
CONT_SINGLE_COLOR_EDGES = True
CONT_SHOW_AGENT_HEADING = True
CONT_SHOW_ROBOT_HEADING = False
CONT_ROBOT_HEADING_SCALE = 1.2
CONT_TRAIL_TARGET = "robot"        # "none"|"crowd"|"robot"|"both"
CONT_TRAIL_STYLE = "persist"       # "persist"|"fade"
CONT_MAX_TRAIL = 2000
CONT_ROBOT_STYLE = "circle"        # "circle"|"image"
CONT_ROBOT_IMAGE_PATH = "assets/robot.png"
CONT_ROBOT_IMAGE_SCALE = 4
CONT_EXIT_SIZE = 5.0
CONT_SNAP_EXIT_TO_BOUNDARY = True
CONT_ANNOTATE_PATH = True
CONT_ANNOTATE_MODE = "every_n"     # "all"|"endpoints"|"every_n"
CONT_ANNOTATE_EVERY = 10
CONT_ANNOTATE_STYLE = "subway"     # "number"|"subway"|"frame"
CONT_ANNOTATE_FONTSIZE = 12

############################
# 결과 기록 유틸
############################
def result_root():
    root = os.path.join(home_dir, f"Result_data_{EXP_NAME}")
    os.makedirs(root, exist_ok=True)   # 상위는 누적, 삭제하지 않음
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
    """기존 테스트들 전체를 스캔해 평균/스텝별 집계를 재계산"""
    test_dirs, base = list_test_dirs(map_id, robot_ver)
    evac_vals, life_vals, logs = [], [], []
    for td in test_dirs:
        evac, life = read_metrics_from_dir(td)
        if evac is not None: evac_vals.append(evac)
        if life is not None: life_vals.append(life)
        logs.append(read_episode_log_from_dir(td))

    # 평균
    avg_evac = float(np.mean(evac_vals)) if evac_vals else float("nan")
    avg_life = float(np.mean(life_vals)) if life_vals else float("nan")
    write_txt(os.path.join(base, "avg_metrics.txt"),
              f"avg_evacuation_100_time={avg_evac}\navg_all_agents_life_time={avg_life}\n")

    # 스텝별 mean/min/max
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
# 그리기 유틸
############################
def draw_text(surface, text, x, y, color=(0,0,0), font_size=20):
    font = pygame.font.SysFont("Arial", font_size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

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
    dist = math.sqrt(dx*dx + dy*dy)
    if dist == 0:
        return 0.0, 0.0
    ratio = dist / JOYSTICK_RADIUS
    scaled_dx = (dx/dist) * ratio * max_move
    scaled_dy = (dy/dist) * ratio * max_move
    return scaled_dx, scaled_dy

def draw_environment(surface, env_map):
    # non-wall pass
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                continue
            rect = (MAP_OFFSET_X + y*CELL_SIZE, MAP_OFFSET_Y + x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if val == 0:   color = WHITE
            elif val == 60: color = BLUE
            elif val == 100: color = GREEN
            elif val == 140: color = YELLOW
            elif val == 200: color = RED
            else:           color = DARK_GREY
            pygame.draw.rect(surface, color, rect)
    # wall pass
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                rect = (MAP_OFFSET_X + y*CELL_SIZE, MAP_OFFSET_Y + x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, BLACK, rect)

def save_continuous_mp4(frames_rgb, out_path, fps=20, dpi=200, bitrate=8000):
    if not frames_rgb:
        return
    h, w, _ = frames_rgb[0].shape
    fig = plt.figure(figsize=(w/100, h/100), dpi=dpi)
    ax = plt.axes([0,0,1,1]); ax.axis('off')
    im = ax.imshow(frames_rgb[0])
    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    with writer.saving(fig, out_path, dpi=dpi):
        for fr in frames_rgb:
            im.set_data(fr)
            writer.grab_frame()
    plt.close(fig)

def np_rgb_to_surface(rgb):
    """(H,W,3) uint8 → pygame.Surface"""
    return pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2)))

def draw_side_panel(surface):
    # 우측 패널 배경
    panel_rect = (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0, PANEL_RIGHT_WIDTH, SCREEN_HEIGHT)
    pygame.draw.rect(surface, (245, 245, 245), panel_rect)
    # 패널 분리선
    pygame.draw.line(surface, (200, 200, 200), (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, 0),
                     (SCREEN_WIDTH - PANEL_RIGHT_WIDTH, SCREEN_HEIGHT), 2)

####################################
# 메인 함수
####################################
def main():
    replay_buffer = ReplayBuffer(capacity=int(1e5))
    episode_count = 0
    running = True

    # 폴더 기본 구조(상위/중간) 준비
    root = result_root()
    map_dir = os.path.join(root, f"Result_{MAP_NUM_FOR_RUN}")
    ensure_dir(map_dir)
    map_robot_dir = os.path.join(map_dir, f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}")
    ensure_dir(map_robot_dir)

    while running and episode_count < MAX_EPISODES:
        # 새로운 episode 시작
        env_model = FightingModel(
            number_agents=20,
            width=50,
            height=50,
            model_num=MAP_NUM_FOR_RUN,
            robot=ROBOT_VERSION_FOR_MODEL   # 내부 로직은 'Q'로, 사람이 직접 action을 보냄
        )
        state = np.array(env_model.return_current_image(), dtype=np.float32)

        # 연속 렌더러
        renderer = None
        collected_frames = []
        if USE_CONTINUOUS_RENDERER:
            renderer = ContinuousRenderer(
                world_size=(50.0, 50.0),
                crowd_colors={0:CONT_CROWD_COLOR, 1:CONT_CROWD_COLOR, 2:CONT_CROWD_COLOR},

                # ▼ 화살표 보이기 + 크기
                show_agent_heading=True,
                agent_heading_scale=1.8,              # 화살표 길이(벡터 스케일)

                # ▼ 화살표 색: 전체 공통 색 또는 타입별 dict 가능
                # 예1) 전체를 보라색으로
                agent_heading_color="#000000",
                # 예2) 타입별(원한다면)
                # agent_heading_color={0:"#7e57c2", 1:"#26a69a", 2:"#ef5350"},

                # ▼ 선 두께/화살촉 크기
                agent_heading_linewidth=1.5,          # 선 두께
                agent_heading_mutation_scale=9.0,    # 화살촉(머리) 크기

                # (나머지 기존 옵션 그대로)
                robot_color=CONT_ROBOT_COLOR,
                single_color_edges=CONT_SINGLE_COLOR_EDGES,
                show_robot_heading=CONT_SHOW_ROBOT_HEADING,
                robot_heading_scale=CONT_ROBOT_HEADING_SCALE,
                trail_target=CONT_TRAIL_TARGET,
                trail_style=CONT_TRAIL_STYLE,
                max_trail=CONT_MAX_TRAIL,
                robot_style=CONT_ROBOT_STYLE,
                robot_image_path=CONT_ROBOT_IMAGE_PATH,
                robot_image_scale=CONT_ROBOT_IMAGE_SCALE,
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
        episode_log = []  # 매 스텝 살아있는 인원 기록

        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False

        # 타이밍 초기화 (에피소드 단위)
        sim_acc = 0.0
        last_time = time.perf_counter()
        sim_dt = 1.0 / TARGET_SIM_FPS

        # === 프레임 루프 ===
        while running and not done:

            # ----- 타이밍 누적 -----
            now = time.perf_counter()
            dt  = now - last_time
            last_time = now
            sim_acc += dt

            # ----- 이벤트(항상) -----
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            if not running:
                break

            # ----- 입력 업데이트 -----
            keys = pygame.key.get_pressed()
            move_step = 5.0
            if keys[pygame.K_UP]:    knob_y -= move_step
            if keys[pygame.K_DOWN]:  knob_y += move_step
            if keys[pygame.K_LEFT]:  knob_x -= move_step
            if keys[pygame.K_RIGHT]: knob_x += move_step
            knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], knob_x, knob_y, JOYSTICK_RADIUS)

            user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))
            env_dx, env_dy = user_dx, -user_dy

            # ----- 고정 스텝 시뮬레이션 -----
            did_step = False
            while sim_acc >= sim_dt and not done:
                env_model.robot.receive_action([env_dx, env_dy])

                current_state = state
                env_model.step()

                # 한 번만 이미지 뽑아 캐시
                img_gray = env_model.return_current_image()
                reward = 0
                alive = env_model.alived_agents()
                episode_log.append(alive)
                if alive <= 0:
                    done = True
                    reward += 10.0

                next_state = np.array(img_gray, dtype=np.float32)
                # replay_buffer.push(current_state, np.array([env_dx, env_dy], np.float32),
                #                    reward, next_state, float(done))

                total_reward += reward
                step_count += 1
                state = next_state
                sim_acc -= sim_dt
                did_step = True

                if step_count >= MAX_STEPS:
                    done = True

            # ----- 렌더링 (스킵 적용) -----
            do_render = did_step and (step_count % RENDER_EVERY == 0)
            if do_render:
                screen.fill(WHITE)
                draw_side_panel(screen)

                if USE_CONTINUOUS_RENDERER:
                    rgb = renderer.draw(env_model, step=step_count)
                    surf = np_rgb_to_surface(rgb)
                    # 빠른 스케일러 (smoothscale보다 가벼움)
                    surf = pygame.transform.scale(surf, (MAP_W, MAP_H))
                    screen.blit(surf, (MAP_OFFSET_X, MAP_OFFSET_Y))

                    # 기록 수집
                    if CONT_VIS_MODE == "mp4":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(rgb)
                    elif CONT_VIS_MODE == "png_every":
                        if (step_count % CONT_SAVE_EVERY == 0) or (alive <= 0):
                            collected_frames.append(("PNG", step_count, rgb))
                    elif CONT_VIS_MODE == "png_last":
                        collected_frames = [("LAST", step_count, rgb)]
                else:
                    # 격자 렌더 (캐시한 img_gray 사용)
                    draw_environment(screen, np.rot90(img_gray, k=1))

                # HUD
                txt_x = SCREEN_WIDTH - PANEL_RIGHT_WIDTH + 10
                draw_text(screen, f"Episode: {episode_count}",     txt_x, 10, color=BLACK, font_size=22)
                draw_text(screen, f"Step: {step_count}",           txt_x, 35, color=BLACK, font_size=22)
                draw_text(screen, f"EpiTotal: {total_reward:.3f}", txt_x, 85, color=BLACK, font_size=22)
                draw_text(screen, "ESC to quit",                   txt_x,110, color=(128,0,0), font_size=18)
                draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))

                pygame.display.flip()

            # 렌더 FPS에 맞춰 쉼 (busy loop 방지)
            #clock.tick(TARGET_RENDER_FPS)

        # === 에피소드 종료: 결과 기록 ===
        if done:
            print(f"[Episode {episode_count} finished] steps={step_count}, total_reward={total_reward:.2f}")
            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

            # 1) 세부 폴더 결정/생성 (먼저!)
            test_i = get_next_test_index(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)
            test_dirname = f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}_{test_i}"
            test_dir = os.path.join(map_robot_dir, test_dirname)
            recreate_dir(test_dir)

            # 2) metrics 저장
            evacuation_100_time = step_count if (len(episode_log) > 0 and episode_log[-1] <= 0) else MAX_STEPS
            all_agents_life_time = env_model.calculate_all_agents_life_time()
            write_txt(os.path.join(test_dir, "metrics.txt"),
                      f"evacuation_100_time={evacuation_100_time}\nall_agents_life_time={all_agents_life_time}\n")
            write_list_txt(os.path.join(test_dir, "episode_log.txt"), episode_log)

            # 3) 연속 렌더 기록 저장 (폴더 만든 후)
            if USE_CONTINUOUS_RENDERER:
                if CONT_VIS_MODE == "mp4" and collected_frames:
                    stride = max(1, CONT_SAVE_EVERY)
                    eff_fps = max(1, int(CONT_FPS / stride))  # 시간 축 보정
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

            # 4) 집계
            aggregate_over_existing_tests(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)

    # 종료 처리
    pygame.quit()
    print(f"Reward log saved to {reward_log_file}")
    print("Bye ~")

if __name__ == "__main__":
    main()
