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

############################
# 화면 및 조이스틱 설정
############################
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

CELL_SIZE = 10
MAP_OFFSET_X = (SCREEN_WIDTH - 50*CELL_SIZE) // 2
MAP_OFFSET_Y = (SCREEN_HEIGHT - 50*CELL_SIZE) // 2

JOYSTICK_CENTER = (120, 700)
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
# 실험/기록 파라미터
############################
# 맵/로봇 버전 (모델에는 'Q'로 넘기되, 폴더명 표기는 'H'로 구분)
MAP_NUM_FOR_RUN = 8          # 원하는 맵 번호(-1은 내부 랜덤 로직)
ROBOT_VERSION_FOR_MODEL = 'Q' # 인간 조작이지만 모델 로직은 'Q'로 유지(수동 action 전달)
ROBOT_VERSION_FOR_LOG   = 'H' # 결과 폴더명에만 'H'로 기록해 비교군 명확화

EXP_NAME = "ImitationRun"     # 최상위 결과 폴더 접미사: Result_data_{EXP_NAME}
MAX_STEPS = 3000              # evac 실패 시 기록에 사용할 상한
MAX_EPISODES = 2
############################
# === 결과 기록 유틸 ===
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
    # 텍스트에 저장된 파이썬 리스트 그대로 들어있으므로 literal_eval 사용
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
    # 폴더명에서 숫자만 추출해 다음 인덱스 산출
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

        step_count = 0
        total_reward = 0.0
        done = False
        episode_count += 1
        episode_log = []  # 매 스텝 살아있는 인원 기록

        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False

        while not done and running:
            clock.tick(15)  # FPS

            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    dist = math.hypot(mx - knob_x, my - knob_y)
                    if dist <= KNOB_RADIUS+5:
                        dragging = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging = False
                    knob_x, knob_y = JOYSTICK_CENTER
                elif event.type == pygame.MOUSEMOTION and dragging:
                    mx, my = pygame.mouse.get_pos()
                    knob_x, knob_y = clamp_knob_to_circle(
                        JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], mx, my, JOYSTICK_RADIUS
                    )
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break
            if not running:
                break

            # 키보드로 knob 이동
            keys = pygame.key.get_pressed()
            move_step = 5.0
            if keys[pygame.K_UP]:    knob_y -= move_step
            if keys[pygame.K_DOWN]:  knob_y += move_step
            if keys[pygame.K_LEFT]:  knob_x -= move_step
            if keys[pygame.K_RIGHT]: knob_x += move_step
            knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], knob_x, knob_y, JOYSTICK_RADIUS)

            # 조이스틱 → 액션
            user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))
            env_dx = user_dx
            env_dy = -user_dy
            env_model.robot.receive_action([env_dx, env_dy])

            current_state = state
            action = np.array([env_dx, env_dy], dtype=np.float32)

            env_model.step()

            # 리워드(원래 로직 그대로)
            reward = 0.0
            reward += env_model.reward_based_alived()
            reward += env_model.reward_penalty()
            reward += env_model.reward_penalty_collision()

            alive = env_model.alived_agents()
            episode_log.append(alive)
            if alive <= 0:
                done = True
                reward += 10.0

            next_state = np.array(env_model.return_current_image(), dtype=np.float32)

            replay_buffer.push(current_state, action, reward, next_state, float(done))

            total_reward += reward
            step_count += 1
            state = next_state

            # 타임아웃(실패) 처리
            if step_count >= MAX_STEPS:
                done = True

            # 화면
            screen.fill(WHITE)
            draw_environment(screen, np.rot90(env_model.return_current_image(), k=1))
            draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))
            draw_text(screen, f"Episode: {episode_count}", 10, 10, color=BLACK, font_size=22)
            draw_text(screen, f"Step: {step_count}", 10, 35, color=BLACK, font_size=22)
            draw_text(screen, f"Reward: {reward:.3f}", 10, 60, color=BLACK, font_size=22)
            draw_text(screen, f"EpiTotal: {total_reward:.3f}", 10, 85, color=BLACK, font_size=22)
            draw_text(screen, "ESC to quit", 10, 110, color=(128,0,0), font_size=18)
            pygame.display.flip()

        # === 에피소드 종료: 결과 기록 ===
        if done:
            print(f"[Episode {episode_count} finished] steps={step_count}, total_reward={total_reward:.2f}")
            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

            # 1) 세부 폴더 결정(같은 이름 있으면 새로 만듦)
            test_i = get_next_test_index(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)
            test_dirname = f"Result_{MAP_NUM_FOR_RUN}_{ROBOT_VERSION_FOR_LOG}_{test_i}"
            test_dir = os.path.join(map_robot_dir, test_dirname)
            recreate_dir(test_dir)

            # 2) metrics 저장
            # 전원 탈출이면 evacuation_100_time=step_count, 아니면 MAX_STEPS
            evacuation_100_time = step_count if (len(episode_log) > 0 and episode_log[-1] <= 0) else MAX_STEPS
            all_agents_life_time = env_model.calculate_all_agents_life_time()
            write_txt(os.path.join(test_dir, "metrics.txt"),
                      f"evacuation_100_time={evacuation_100_time}\nall_agents_life_time={all_agents_life_time}\n")
            write_list_txt(os.path.join(test_dir, "episode_log.txt"), episode_log)

            # 3) 집계 재계산(해당 맵×로봇버전 범위 전체 스캔)
            aggregate_over_existing_tests(MAP_NUM_FOR_RUN, ROBOT_VERSION_FOR_LOG)

    # 종료 처리
    pygame.quit()
    print(f"Reward log saved to {reward_log_file}")
    print("Bye ~")

if __name__ == "__main__":
    main()
