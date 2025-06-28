"""
ADDS Imitation-Learning 데이터 수집 UI
------------------------------------
• PyGame 조이스틱으로 로봇을 수동 조종하며 (dx, dy) 액션을 생성
• 매 프레임(또는 ACTION_SCALE 주기)마다 (s, a, r, s', done)을 ReplayBuffer에 push
• 여러 에피소드를 한 세션에서 계속 진행해도 버퍼는 메모리에 누적
• 프로그램을 종료할 때만 디스크의 기존 expert_dataset.pkl 과 병합하여 저장
"""
import pygame
import sys, math, os, pickle, time
import numpy as np
from pathlib import Path

# ---------- 외부 모듈 ----------
from model import FightingModel
from ADDS_AS_reinforcement import ReplayBuffer          # 동일 폴더
from Start_training import *                            # REWARD_A …, ACTION_SCALE …

# ---------- 화면/조이스틱 파라미터 ----------
SCREEN_WIDTH  = 800
SCREEN_HEIGHT = 800
CELL_SIZE     = 10
MAP_OFFSET_X  = (SCREEN_WIDTH  - 50*CELL_SIZE) // 2
MAP_OFFSET_Y  = (SCREEN_HEIGHT - 50*CELL_SIZE) // 2
JOYSTICK_CENTER = (120, 700)
JOYSTICK_RADIUS = 60
KNOB_RADIUS     = 15
MAX_MOVE        = 2.0

WHITE = (255,255,255); BLACK=(0,0,0); GREY=(200,200,200)
RED   = (255,0,0); BLUE=(0,0,255); GREEN=(0,255,0)
YELLOW=(255,255,0); DARK_GREY=(150,150,150)

# ---------- 경로 ----------
HOME_DIR = Path.home()
LOG_DIR  = HOME_DIR / LOG_DIR        # LOG_DIR 상수는 Start_training.py 쪽에서 정의
LOG_DIR.mkdir(parents=True, exist_ok=True)
REWARD_LOG = LOG_DIR / "total_reward_imitation.txt"
BUFFER_FNAME = LOG_DIR / "expert_dataset.npz"           # ← 기존 파일과 병합

# =================================================================== #
#                       PyGame 초기화 & 유틸                           #
# =================================================================== #
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("ADDS Imitation Learning")
clock = pygame.time.Clock()

def draw_text(surf, text, x, y, color=BLACK, size=20):
    font = pygame.font.SysFont("Arial", size)
    surf.blit(font.render(text, True, color), (x, y))

def draw_joystick(surf, center, radius, knob_pos):
    pygame.draw.circle(surf, GREY, center, radius)
    pygame.draw.circle(surf, RED,  knob_pos, KNOB_RADIUS)

def clamp_knob(cx, cy, kx, ky, radius):
    dx, dy = kx-cx, ky-cy
    dist = math.hypot(dx, dy)
    if dist > radius:
        dx *= radius/dist; dy *= radius/dist
    return cx+dx, cy+dy

def get_action(center, knob, max_move=MAX_MOVE):
    cx, cy = center; kx, ky = knob
    dx, dy = kx-cx, ky-cy
    dist = math.hypot(dx, dy)
    if dist == 0: return 0.0, 0.0
    ratio = dist / JOYSTICK_RADIUS
    scaled_dx = (dx/dist) * ratio * max_move
    scaled_dy = (dy/dist) * ratio * max_move
    return scaled_dx, scaled_dy       # (dx, dy)

def draw_env(surf, env_map):
    # pass-1: 배경 / 에이전트 / 목표
    for i,row in enumerate(env_map):
        for j,val in enumerate(row):
            if val == 20:      # 벽은 두 번째 패스
                continue
            rect = (MAP_OFFSET_X + j*CELL_SIZE,
                    MAP_OFFSET_Y + i*CELL_SIZE,
                    CELL_SIZE, CELL_SIZE)
            color = {
                0:WHITE, 60:BLUE, 100:GREEN, 140:YELLOW, 200:RED
            }.get(val, DARK_GREY)
            pygame.draw.rect(surf, color, rect)
    # pass-2: 벽
    for i,row in enumerate(env_map):
        for j,val in enumerate(row):
            if val == 20:
                rect = (MAP_OFFSET_X + j*CELL_SIZE,
                        MAP_OFFSET_Y + i*CELL_SIZE,
                        CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surf, BLACK, rect)

# =================================================================== #
#                               main                                  #
# =================================================================== #
def main():
    replay_buffer = ReplayBuffer(capacity=int(1e5),
                                 state_shape=(50,50),
                                 action_dim=2,
                                 device="cpu")

    episode_count = 0
    running = True

    # ------------------------ 메인 루프 ------------------------ #
    while running:
        env = FightingModel(number_agents=20, width=50, height=50,
                             model_num=1, robot="Q")
        state = np.array(env.return_current_image(), dtype=np.float32)
        done = False; step = 0; total_reward = 0.0
        episode_count += 1

        knob_x, knob_y = JOYSTICK_CENTER
        dragging = False
        frame_cnt = 0

        # -------------- 1 개 에피소드 루프 -------------- #
        while not done and running:
            clock.tick(15)                         # FPS ≃ 15

            # ──── 이벤트 ────
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = pygame.mouse.get_pos()
                    if math.hypot(mx-knob_x, my-knob_y) <= KNOB_RADIUS+5:
                        dragging = True
                elif ev.type == pygame.MOUSEBUTTONUP:
                    dragging = False; knob_x,knob_y = JOYSTICK_CENTER
                elif ev.type == pygame.MOUSEMOTION and dragging:
                    mx,my = pygame.mouse.get_pos()
                    knob_x,knob_y = clamp_knob(*JOYSTICK_CENTER, mx,my, JOYSTICK_RADIUS)
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    running = False
            if not running: break

            # ──── 키보드 → knob 이동 ────
            keys = pygame.key.get_pressed()
            move = 5
            if keys[pygame.K_UP]   : knob_y -= move
            if keys[pygame.K_DOWN] : knob_y += move
            if keys[pygame.K_LEFT] : knob_x -= move
            if keys[pygame.K_RIGHT]: knob_x += move
            knob_x, knob_y = clamp_knob(*JOYSTICK_CENTER, knob_x, knob_y, JOYSTICK_RADIUS)

            # ──── 조이스틱 → 액션 ────
            user_dx, user_dy = get_action(JOYSTICK_CENTER, (knob_x, knob_y))
            env_dx, env_dy   = user_dy, user_dx          # 축 뒤집기
            env.robot.receive_action([env_dx, env_dy])

            # ──── 환경 스텝 ────
            cur_state = state
            action = np.array([env_dx, env_dy], dtype=np.float32)

            env.step(); frame_cnt += 1
            reward = 0.0
            if REWARD_A: reward += env.reward_based_alived() * REWARD_A
            if REWARD_B: reward += env.reward_based_all_agents_danger() * REWARD_B
            if REWARD_C: reward += env.reward_based_gain() * REWARD_C
            if REWARD_D: reward += env.reward_penalty() * REWARD_D
            if REWARD_E: reward += env.reward_based_evacuated_with_robot() * REWARD_E
            if REWARD_F: reward += env.reward_based_distance_from_near_agents() * REWARD_F
            if REWARD_G: reward += env.reward_based_distance_from_near_agent_gain() * REWARD_G
            if REWARD_H: reward += env.reward_based_gain_with_time_bonus() * REWARD_H
            if REWARD_I: reward += env.reward_based_alived_root() * REWARD_I
            if REWARD_J: reward += env.reward_based_all_agents_danger_log() * REWARD_J
            reward += REWARD_FIXED

            if env.alived_agents() <= 0:
                done = True; reward += 10.0

            next_state = np.array(env.return_current_image(), dtype=np.float32)

            # 버퍼 push (ACTION_SCALE 주기)
            if frame_cnt % ACTION_SCALE == 0:
                replay_buffer.push(cur_state, action, reward, next_state, float(done))

            total_reward += reward
            step += 1; state = next_state

            # ──── 화면 그리기 ────
            screen.fill(WHITE)
            draw_env(screen, env.return_current_image())
            draw_joystick(screen, JOYSTICK_CENTER, JOYSTICK_RADIUS, (knob_x, knob_y))
            draw_text(screen, f"Episode {episode_count}", 10,10, size=22)
            draw_text(screen, f"Step    {step}", 10,35, size=22)
            draw_text(screen, f"Reward  {reward:7.3f}", 10,60, size=22)
            draw_text(screen, f"EpiTotal{total_reward:7.3f}", 10,85, size=22)
            draw_text(screen, "ESC to quit", 10,110, color=(128,0,0), size=18)
            pygame.display.flip()

        # ──── 에피소드 종료 후 로그 ────
        if done:
            print(f"[Episode {episode_count}] steps={step}, total_reward={total_reward:.2f}")
            with open(REWARD_LOG, "a", encoding="utf-8") as f:
                f.write(f"{total_reward}\n")

    # --------------- 메인 루프 끝 (사용자 종료) --------------- #
    pygame.quit()

    # ──── 종료 시점: ReplayBuffer ←→ 디스크 병합 저장 ────
    if BUFFER_FNAME.exists():
        print(f"Loading existing buffer: {BUFFER_FNAME}")
        # (1) 기존 버퍼를 같은 capacity 로 새 객체에 로드
        existing = ReplayBuffer(capacity=int(1e5),
                                state_shape=(50,50),
                                action_dim=2,
                                device="cpu")
        existing.load(BUFFER_FNAME)          # ← npz 로드

        print("Appending new transitions …")
        for i in range(replay_buffer.size):
            existing.push(replay_buffer.states[i],
                        replay_buffer.actions[i],
                        replay_buffer.rewards[i],
                        replay_buffer.next_states[i],
                        replay_buffer.dones[i])
        final_buf = existing
    else:
        final_buf = replay_buffer

    # (2) 병합(또는 신규) 버퍼를 저장
    final_buf.save(BUFFER_FNAME)             # → expert_dataset.npz (np.savez_compressed)
    print(f"Saved merged buffer ▶ {BUFFER_FNAME}  (size={len(final_buf):,})")
    print(f"Reward log ▶ {REWARD_LOG}")
    print("Bye 👋")

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
