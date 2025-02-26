import pygame
import sys
import math
import os
import pickle
import time

import numpy as np

# model.py, ADDS_AS_reinforcement.py 경로를 맞춰주세요.
from model import FightingModel
from ADDS_AS_reinforcement import ReplayBuffer

############################
# 화면 및 조이스틱 설정
############################
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800

# 맵 70x70을 그릴 때의 셀 크기
CELL_SIZE = 10
# 맵 전체 픽셀 크기 = 70 * CELL_SIZE = 700
# 화면이 800x800이므로, 조금 여유있게 중앙정렬하기 위해 오프셋을 줍니다.
MAP_OFFSET_X = (SCREEN_WIDTH - 70*CELL_SIZE) // 2  # 대략 50
MAP_OFFSET_Y = (SCREEN_HEIGHT - 70*CELL_SIZE) // 2 # 대략 50

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
    pygame.draw.circle(surface, GREY, center, radius)
    pygame.draw.circle(surface, RED, knob_pos, KNOB_RADIUS)

def get_joystick_action(joystick_center, knob_pos, max_move=MAX_MOVE):
    """조이스틱에서 (-max_move ~ +max_move) 범위의 (dx, dy)를 구합니다."""
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
    env_map[x][y] 2D 배열(70x70)을 화면에 그림.
      - 0: 빈공간, 20: 벽, 60: 출구, 100/140: agent, 200: 로봇
      - 그릴 때 (x,y) => 실제 픽셀 위치는 (MAP_OFFSET_X + y*CELL_SIZE, MAP_OFFSET_Y + x*CELL_SIZE)
    """
    for x in range(len(env_map)):
        for y in range(len(env_map[0])):
            val = env_map[x][y]
            rect = (
                MAP_OFFSET_X + y*CELL_SIZE,
                MAP_OFFSET_Y + x*CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )
            if val == 0:
                color = WHITE
            elif val == 20:
                color = BLACK
            elif val == 60:
                color = BLUE
            elif val == 100:
                color = GREEN
            elif val == 140:
                color = YELLOW
            elif val == 200:
                color = RED
            else:
                color = (150, 150, 150)
            pygame.draw.rect(surface, color, rect)


def main():
    # -------------------------
    # (1) 환경 초기화
    # -------------------------
    env_model = FightingModel(
        number_agents=30,
        width=70,
        height=70,
        model_num=2,      # 랜덤맵 or 특정맵
        robot='Q'
    )

    # ReplayBuffer (동일 형식)
    replay_buffer = ReplayBuffer(capacity=int(1e5))

    # 초기 state
    state = np.array(env_model.return_current_image(), dtype=np.float32)

    # 조이스틱 knob 초기 위치
    knob_x, knob_y = JOYSTICK_CENTER
    dragging = False

    running = True
    step_count = 0
    total_reward = 0.0

    # 에피소드 단위 (원하면 여러 에피소드 가능)
    episode_done = False

    while running:
        clock.tick(30)  # 초당 30프레임

        # -------------------------
        # 이벤트 처리
        # -------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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
                dx = mx - JOYSTICK_CENTER[0]
                dy = my - JOYSTICK_CENTER[1]
                dist = math.hypot(dx, dy)
                if dist > JOYSTICK_RADIUS:
                    scale = JOYSTICK_RADIUS / dist
                    dx *= scale
                    dy *= scale
                knob_x = JOYSTICK_CENTER[0] + dx
                knob_y = JOYSTICK_CENTER[1] + dy

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        # -------------------------
        # 조이스틱 -> 로봇 액션
        # -------------------------
        user_dx, user_dy = get_joystick_action(JOYSTICK_CENTER, (knob_x, knob_y))

        # (중요) 방향 뒤집힘을 해결하기 위해, 실제 env는 action=[dy, dx] 로 전달
        env_dx = user_dy
        env_dy = user_dx

        # 로봇에 액션 주기
        env_model.robot.receive_action([env_dx, env_dy])

        # 현재 state
        current_state = state

        # action (dx, dy) -> float array
        action = np.array([env_dx, env_dy], dtype=np.float32)

        # -------------------------
        # 환경 한 스텝
        # -------------------------
        env_model.step()

        # reward, done
        reward = 0.0
        reward += env_model.reward_based_alived()
        reward += env_model.reward_penalty()

        done = False
        if env_model.alived_agents() <= 0:
            done = True
            reward += 10.0

        # 다음 상태
        next_state = np.array(env_model.return_current_image(), dtype=np.float32)

        # ReplayBuffer에 추가
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

        if done and not episode_done:
            # 에피소드 끝
            episode_done = True
            print(f"[Episode finished] step={step_count}, total_reward={total_reward:.2f}")
            # 로그 파일에 기록
            with open(reward_log_file, "a", encoding='utf-8') as f:
                f.write(f"{total_reward}\n")

            # 여기서는 바로 종료(원하면 재시작 가능)
            running = False

        # -------------------------
        # 화면 그리기
        # -------------------------
        screen.fill(WHITE)

        # 맵 그리기
        draw_environment(screen, env_model.return_current_image())

        # 조이스틱
        draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))

        # 텍스트
        draw_text(screen, f"Step: {step_count}", 10, 10, color=BLACK, font_size=22)
        draw_text(screen, f"Reward: {reward:.3f}", 10, 35, color=BLACK, font_size=22)
        draw_text(screen, f"EpiTotal: {total_reward:.3f}", 10, 60, color=BLACK, font_size=22)
        draw_text(screen, "ESC to quit", 10, 85, color=(128,0,0), font_size=18)

        pygame.display.flip()

    # -------------------------
    # 종료 처리
    # -------------------------
    pygame.quit()

    # 마지막으로 ReplayBuffer 저장
    save_path = os.path.join(os.path.dirname(__file__), "imitation_dataset.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(replay_buffer.buffer, f)
    print(f"Replay buffer saved to {save_path}, size={len(replay_buffer.buffer)}")
    print(f"Reward log saved to {reward_log_file}")

if __name__ == "__main__":
    main()
