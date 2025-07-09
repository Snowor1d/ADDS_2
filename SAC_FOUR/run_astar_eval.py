#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_astar_eval.py  –  A* baseline 성능 측정

• 동일한 환경 / 리워드 / 로그 구조 재사용
• 결과 파일: Log_SAC_FOUR/total_reward_astar.txt, evacuation_*.txt
"""

import os, random, time, numpy as np
import model                                   # FightingModel
from Start_training import (LOG_DIR, ACTION_SCALE, MAX_STEPS,
                            CROWD_NUMBER_MIN, CROWD_NUMBER_MAX,
                            MAP_NUM, MAP_NUM_RANDOM, FINISHED_BONUS)

log_dir = os.path.join(os.path.expanduser("~"), LOG_DIR)
os.makedirs(log_dir, exist_ok=True)

def main(max_episodes=200):
    reward_log = open(os.path.join(log_dir, "total_reward_astar.txt"), "w")
    evac80_log = open(os.path.join(log_dir, "evacuation_80_astar.txt"), "w")
    evac100_log= open(os.path.join(log_dir, "evacuation_100_astar.txt"), "w")

    for ep in range(max_episodes):
        n_agents = CROWD_NUMBER_MIN if CROWD_NUMBER_MIN==CROWD_NUMBER_MAX \
                   else random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
        env = model.FightingModel(n_agents, 50, 50, model_num=MAP_NUM, robot='A')
        total_reward = 0
        evac80 = evac100 = MAX_STEPS

        for step in range(MAX_STEPS):
            env.step()                         # RobotAgent.robot_policy_A 가 내부에서 호출됨
            if env.alived_agents() < env.total_agents*0.2 and evac80==MAX_STEPS:
                evac80 = step
            if env.alived_agents() < 1 and evac100==MAX_STEPS:
                evac100 = step
            if env.robot.is_game_finished:
                total_reward += FINISHED_BONUS * (1-step/MAX_STEPS)
                break

        # 로그 기록
        reward_log.write(f"{total_reward}\n")
        evac80_log.write(f"{evac80}\n")
        evac100_log.write(f"{evac100}\n")
        print(f"[EP {ep:04d}] reward={total_reward:.1f}  80%={evac80} 100%={evac100}")

    reward_log.close(); evac80_log.close(); evac100_log.close()

if __name__ == "__main__":
    main()
