"""
ADDS_AS_ImitationGame.py

PyGame으로 사람이 직접 로봇(정책)을 조종하여
(s, a, r, next_state, done) 형태로 데이터를 수집하고
imitation_dataset.pkl 파일에 저장
"""

import pygame
import sys
import math
import os
import pickle
import time
import numpy as np
from model import FightingModel
import re

##########################
# 화면 설정
##########################
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

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ADDS Imitation Learning")
clock = pygame.time.Clock()

##########################
# 로그 및 파일 경로
##########################
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_guide_game_sac_gail")
os.makedirs(log_dir, exist_ok=True)
reward_log_file = os.path.join(log_dir, "total_reward_ImitationGame.txt")

##########################
# 유틸 함수
##########################
def draw_text(surface, text, x, y, color=(0,0,0), font_size=20):
    font = pygame.font.SysFont("Arial", font_size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

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
    """
    조이스틱 위치(knob_pos)를 (-max_move, +max_move) 범위의 (dx, dy)로 매핑
    """
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

def draw_environment(surface, env_map):
    # (1) 먼저 벽(20)이 아닌 타일 그리기
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                continue  # 벽은 뒤에서 그림
            rect = (
                MAP_OFFSET_X + y*CELL_SIZE,
                MAP_OFFSET_Y + x*CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
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

    # (2) 벽(20)만 덮어씌우기
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                rect = (
                    MAP_OFFSET_X + y*CELL_SIZE,
                    MAP_OFFSET_Y + x*CELL_SIZE,
                    CELL_SIZE, CELL_SIZE
                )
                pygame.draw.rect(surface, BLACK, rect)

def draw_joystick(surface, center, radius, knob_pos):
    pygame.draw.circle(surface, GREY, center, radius)
    pygame.draw.circle(surface, RED, knob_pos, KNOB_RADIUS)

##########################
# 메인
##########################
def main():
    # 사람 플레이로 만든 (s,a,r,next_s,done) 데이터를 저장할 리스트
    expert_data = []

    episode_count = 0
    running = True

    while running:
        # FightingModel 환경 생성
        env_model = FightingModel(
            number_agents=20,
            width=50,
            height=50,
            model_num=2,
            robot='Q'
        )
        # 초기 상태
        state = env_model.return_current_image()  # 50x50 2D
        state = np.array(state, dtype=np.float32)

        total_reward = 0.0
        step_count = 0
        done = False
        episode_count += 1

        # 조이스틱 knob
        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False

        while not done and running:
            clock.tick(20)

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
                    knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], mx, my, JOYSTICK_RADIUS)

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

            if not running:
                break

            # 키보드 화살표로도 knob 조작
            keys = pygame.key.get_pressed()
            move_step = 5.0
            if keys[pygame.K_UP]:
                knob_y -= move_step
            if keys[pygame.K_DOWN]:
                knob_y += move_step
            if keys[pygame.K_LEFT]:
                knob_x -= move_step
            if keys[pygame.K_RIGHT]:
                knob_x += move_step
            knob_x, knob_y = clamp_knob_to_circle(JOYSTICK_CENTER[0], JOYSTICK_CENTER[1], knob_x, knob_y, JOYSTICK_RADIUS)

            # 조이스틱 -> (dx, dy)
            raw_dx, raw_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y), MAX_MOVE)
            # FightingModel 내부에서 로봇에 (dx, dy) 적용
            # 필요 시 방향 뒤집힘 보정
            dx = raw_dy
            dy = raw_dx

            current_state = state
            action = np.array([dx, dy], dtype=np.float32)

            # 액션 적용
            env_model.robot.receive_action([dx, dy])
            env_model.step()

            # 보상, done
            r = 0.0
            r += env_model.reward_based_alived()
            r += env_model.reward_penalty()
            # etc...
            done = (env_model.alived_agents()<=1)
            if done:
                r += 10.0

            next_state = env_model.return_current_image()
            next_state = np.array(next_state, dtype=np.float32)

            # (s,a,r,ns,done) 형태로 저장
            expert_data.append((current_state, action, r, next_state, float(done)))

            total_reward += r
            step_count += 1
            state = next_state

            # 화면 표시
            screen.fill(WHITE)
            draw_environment(screen, env_model.return_current_image())
            draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))
            draw_text(screen, f"Episode: {episode_count}", 10, 10, color=BLACK, font_size=22)
            draw_text(screen, f"Step: {step_count}", 10, 35, color=BLACK, font_size=22)
            draw_text(screen, f"Reward: {r:.3f}", 10, 60, color=BLACK, font_size=22)
            draw_text(screen, f"Total: {total_reward:.3f}", 10, 85, color=BLACK, font_size=22)
            pygame.display.flip()

        if done:
            print(f"[Episode {episode_count}] total_reward={total_reward}")
            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

    pygame.quit()

    pattern = r"^imitation_dataset_(\d+)\.pkl$"
    existing_files = [f for f in os.listdir(log_dir) if f.startswith("imitation_dataset_") and f.endswith(".pkl")]
    max_index = -1
    for ef in existing_files:
        match = re.match(pattern, ef)
        if match:
            idx = int(match.group(1))
            if idx > max_index:
                max_index = idx
    next_index = max_index + 1  # 다음 파일 번호

    save_path = os.path.join(log_dir, f"imitation_dataset_{next_index}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(expert_data, f)

    print(f"Expert data saved to {save_path}, size={len(expert_data)}")
    print(f"Reward log saved to {reward_log_file}")

if __name__ == "__main__":
    main()
