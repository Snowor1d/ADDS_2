import pygame
import sys
import math
import os
import pickle
import time

import numpy as np

# model.py, ADDS_AS_reinforcement.py (동일 폴더 or 경로 맞춰 수정)
from model import FightingModel
from ADDS_AS_reinforcement import ReplayBuffer

############################
# 화면 및 조이스틱 설정
############################
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# 맵 70x70을 그릴 때의 셀 크기
CELL_SIZE = 10
# 맵을 화면 중간에 배치하기 위한 오프셋
MAP_OFFSET_X = (SCREEN_WIDTH - 70*CELL_SIZE) // 2
MAP_OFFSET_Y = (SCREEN_HEIGHT - 70*CELL_SIZE) // 2

# 조이스틱 UI
JOYSTICK_CENTER = (120, 700)  # 화면 아래쪽 근처
JOYSTICK_RADIUS = 60
KNOB_RADIUS = 15
MAX_MOVE = 2.0  # 환경에서 허용하는 로봇 이동 범위가 -2~2

# 색상
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
# 로깅(리워드 파일 저장) 설정
############################
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_guide_game_imitation")
os.makedirs(log_dir, exist_ok=True)
reward_log_file = os.path.join(log_dir, "total_reward_imitation.txt")

############################
# 유틸 함수들
############################
def draw_text(surface, text, x, y, color=(0,0,0), font_size=20):
    font = pygame.font.SysFont("Arial", font_size)
    img = font.render(text, True, color)
    surface.blit(img, (x, y))

def draw_joystick(surface, center, radius, knob_pos):
    # 조이스틱 밑판
    pygame.draw.circle(surface, GREY, center, radius)
    # 핸들
    pygame.draw.circle(surface, RED, knob_pos, KNOB_RADIUS)

def clamp_knob_to_circle(cx, cy, knob_x, knob_y, radius):
    """knob이 조이스틱 원 범위 벗어나지 않도록 clamp."""
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
    조이스틱에서 (-max_move ~ +max_move) 범위의 (dx, dy) 계산
    """
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
    """
    2-pass로 그리드 그림:
      1) 벽(20) 이외의 것
      2) 벽(20) => 최상단
    """
    # (1) 먼저 벽 아닌 것(0, 60, 100, 140, 200 등) 그리기
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            if val == 20:
                # 벽은 두 번째 pass에서 그릴 것
                continue

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
                # 예외 코드
                color = DARK_GREY

            pygame.draw.rect(surface, color, rect)

    # (2) 벽(20)만 별도로 덮어 그림
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

####################################
# 메인 함수
####################################
def main():
    # -------------------------
    # 여러 에피소드 진행을 위해
    # -------------------------
    replay_buffer = ReplayBuffer(capacity=int(1e5))
    episode_count = 0
    running = True

    while running:
        # 새로운 episode 시작
        env_model = FightingModel(
            number_agents=30,
            width=70,
            height=70,
            model_num=2,   # 원하는 맵 번호
            robot='Q'
        )
        state = np.array(env_model.return_current_image(), dtype=np.float32)

        # episode별 변수
        step_count = 0
        total_reward = 0.0
        done = False
        episode_count += 1

        # 조이스틱 knob 초기 위치
        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False

        while not done and running:
            clock.tick(15)  # 초당 30프레임

            # -------------------------
            # 이벤트 처리
            # -------------------------
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
                    # 손 떼면 중앙 복귀
                    knob_x, knob_y = JOYSTICK_CENTER

                elif event.type == pygame.MOUSEMOTION and dragging:
                    mx, my = pygame.mouse.get_pos()
                    knob_x, knob_y = clamp_knob_to_circle(
                        JOYSTICK_CENTER[0], JOYSTICK_CENTER[1],
                        mx, my,
                        JOYSTICK_RADIUS
                    )

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break

            if not running:
                break

            # -------------------------
            # 키보드 방향키 입력 읽기
            # -------------------------
            keys = pygame.key.get_pressed()
            # 방향키를 누르면 knob 움직이기
            move_step = 5.0  # 한 프레임당 knob 이동 픽셀
            if keys[pygame.K_UP]:
                knob_y -= move_step
            if keys[pygame.K_DOWN]:
                knob_y += move_step
            if keys[pygame.K_LEFT]:
                knob_x -= move_step
            if keys[pygame.K_RIGHT]:
                knob_x += move_step

            # 키보드 누른 뒤에도 knob이 조이스틱 범위 안에 있어야 함
            knob_x, knob_y = clamp_knob_to_circle(
                JOYSTICK_CENTER[0], JOYSTICK_CENTER[1],
                knob_x, knob_y,
                JOYSTICK_RADIUS
            )

            # -------------------------
            # 조이스틱 -> 로봇 액션
            # -------------------------
            user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))
            # 방향 뒤집힘 보정: (env_dx, env_dy) = (dy, dx)
            env_dx = user_dy
            env_dy = user_dx

            env_model.robot.receive_action([env_dx, env_dy])

            # -------------------------
            # 환경 스텝
            # -------------------------
            current_state = state
            action = np.array([env_dx, env_dy], dtype=np.float32)

            env_model.step()

            reward = 0.0
            reward += env_model.reward_based_alived()
            reward += env_model.reward_penalty()

            if env_model.alived_agents() <= 0:
                done = True
                reward += 10.0

            next_state = np.array(env_model.return_current_image(), dtype=np.float32)

            # 버퍼에 저장
            replay_buffer.push(
                current_state,
                action,
                reward,
                next_state,
                float(done)
            )

            total_reward += reward
            step_count += 1
            state = next_state

            # -------------------------
            # 화면 그리기
            # -------------------------
            screen.fill(WHITE)

            draw_environment(screen, env_model.return_current_image())
            draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))

            draw_text(screen, f"Episode: {episode_count}", 10, 10, color=BLACK, font_size=22)
            draw_text(screen, f"Step: {step_count}", 10, 35, color=BLACK, font_size=22)
            draw_text(screen, f"Reward: {reward:.3f}", 10, 60, color=BLACK, font_size=22)
            draw_text(screen, f"EpiTotal: {total_reward:.3f}", 10, 85, color=BLACK, font_size=22)
            draw_text(screen, "ESC to quit", 10, 110, color=(128,0,0), font_size=18)

            pygame.display.flip()

        # 한 episode 끝난 뒤 => total_reward 로그에 기록
        if done:
            print(f"[Episode {episode_count} finished] steps={step_count}, total_reward={total_reward:.2f}")

            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

    # while문 종료 => ESC or Quit
    pygame.quit()

    # 전체 종료 시점에 ReplayBuffer 저장
    save_path = os.path.join(os.path.dirname(__file__), "imitation_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(replay_buffer.buffer, f)

    print(f"Replay buffer saved to {save_path}, size={len(replay_buffer.buffer)}")
    print(f"Reward log saved to {reward_log_file}")
    print("Bye ~")

if __name__ == "__main__":
    main()
