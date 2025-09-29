#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_astar.py  –  A* baseline 성능 측정 & TensorBoard 기록

✓ 기존 run_astar_eval.py 의 파일-로그 유지
✓ 추가로 TensorBoard(SummaryWriter) 로 evac-time / reward / 성공률 저장
"""

import os, random, numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import model                                   # FightingModel
from Start_training import (LOG_DIR, ACTION_SCALE, MAX_STEPS,
                            CROWD_NUMBER_MIN, CROWD_NUMBER_MAX,
                            MAP_NUM, MAP_NUM_RANDOM, FINISHED_BONUS)

# ──────────────────────────────────────────────────────
# 로그 디렉터리 준비
home_log = os.path.join(os.path.expanduser("~"), LOG_DIR)
os.makedirs(home_log, exist_ok=True)

writer = SummaryWriter(os.path.join(home_log, "tb_astar"),
                       flush_secs=1)

# 텍스트 로그 (SAC 과 동일 폴더·형식)
reward_log   = open(os.path.join(home_log, "total_reward_astar.txt"),   "w", buffering=1)
evac80_log   = open(os.path.join(home_log, "evacuation_80_astar.txt"),  "w", buffering=1)
evac100_log  = open(os.path.join(home_log, "evacuation_100_astar.txt"), "w", buffering=1)

# ──────────────────────────────────────────────────────
def select_map() -> int:
    """MAP_NUM == -2 → 고정 / -1 → 무작위 / 그 외 → 지정"""
    if MAP_NUM == -2:                  # 고정 (Start_training 에서 사용하지 않음)
        return 1
    if MAP_NUM == -1:                  # 무작위
        return random.choice(MAP_NUM_RANDOM)
    return MAP_NUM

# ──────────────────────────────────────────────────────
def run_episode(ep_idx: int):
    """A* 로봇으로 1 에피소드 수행 & 로그"""
    # Crowd 수 무작위
    n_agents = CROWD_NUMBER_MIN if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX \
               else random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)

    env  = model.FightingModel(n_agents, 50, 50,
                               model_num = select_map(),
                               robot     = 'A')   # ← A* 모드
    total_reward = 0
    evac80 = evac100 = MAX_STEPS

    for step in range(MAX_STEPS):
        env.step()                         # RobotAgent.robot_policy_A 내부 호출

        if env.alived_agents() < env.total_agents * 0.20 and evac80  == MAX_STEPS:
            evac80  = step
        if env.alived_agents() < 1                      and evac100 == MAX_STEPS:
            evac100 = step

        if getattr(env.robot, "is_game_finished", False):
            # 기존 규칙: 탈출 완료 시 추가 보너스
            total_reward += FINISHED_BONUS * (1 - step / MAX_STEPS)
            break

    # ─ 텍스트 로그
    reward_log.write(f"{total_reward}\n")
    evac80_log.write(f"{evac80}\n")
    evac100_log.write(f"{evac100}\n")

    # ─ TensorBoard 로그
    writer.add_scalar("evac_time/80_percent",  evac80,        ep_idx)
    writer.add_scalar("evac_time/100_percent", evac100,       ep_idx)
    writer.add_scalar("reward/total",          total_reward,  ep_idx)
    writer.add_scalar("success/100_percent",
                      int(evac100 < MAX_STEPS), ep_idx)

    print(f"[EP {ep_idx:04d}] reward={total_reward:.1f}  "
          f"80%={evac80} 100%={evac100}")

# ──────────────────────────────────────────────────────
def main(max_episodes: int = 200):
    for ep in range(max_episodes):
        run_episode(ep)

    # 파일 close & TensorBoard flush
    reward_log.close(); evac80_log.close(); evac100_log.close()
    writer.flush()
    print(f"\n✔ 완료!  TensorBoard:  tensorboard --logdir {home_log}/tb_astar")

# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
