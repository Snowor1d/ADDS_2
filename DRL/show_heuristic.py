#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_heuristic.py
────────────────────────────────────────────────────────
HeuristicPolicy의 움직임을 GIF로 시각화해 저장합니다.

결과 파일
  • heuristic.gif   : 애니메이션 (30 fps)
  • trajectory.png  : 로봇 이동 궤적 스냅숏
둘 다  ~/LOG_DIR/  아래에 생성됩니다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")             # GUI 없이 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import animation

from Start_training import LOG_DIR          # 프로젝트 공통 로그폴더
import model
from make_dql_dataset import HeuristicPolicy, ACTION_SCALE   # 휴리스틱·스케일 재사용

# ───────────────────── 사용자 설정 ───────────────────── #
DEBUG_PRINT = True        # True → 프레임별 state 로그 출력
CROWD_SIZE  = 20
MAP_SIZE    = 50
MAX_FRAMES  = 600           # 600 frame ≒ 20 s @ 30 fps
# ─────────────────────────────────────────────────────── #

def main() -> None:
    # 저장 폴더 준비 -------------------------------------------------
    home     = Path.home()
    save_dir = home / LOG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    gif_path  = save_dir / "heuristic.gif"
    traj_path = save_dir / "trajectory.png"

    # 환경 & 정책 ---------------------------------------------------
    env    = model.FightingModel(CROWD_SIZE, MAP_SIZE, MAP_SIZE,
                                 model_num=-1, robot="Q")
    policy = HeuristicPolicy(env)

    # Matplotlib Figure --------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    im      = ax.imshow(env.return_current_image(), cmap="gray",
                        vmin=0, vmax=255, animated=True)
    ax.axis("off")

    # 로봇 궤적 저장용
    traj_x, traj_y = [], []

    # 업데이트 함수 -------------------------------------------------
    def update(frame_idx: int):
        if env.robot.is_game_finished:
            ani.event_source.stop()          # ▶︎ 녹화 중단
            if DEBUG_PRINT:
                print(f"[{frame_idx}] GAME FINISHED")
            return (im,)

        # 4 step마다 실제 행동
        if frame_idx % ACTION_SCALE == 0:
            a = policy.select_action(None)
            env.robot.receive_action(a)

            if DEBUG_PRINT:
                print(f"[{frame_idx:4d}] state={policy.state:<8s} "
                      f"robot={env.robot.xy} "
                      f"target={getattr(policy.target_agent, 'xy', None)}")

        env.step()

        # 화면 & 궤적 갱신
        im.set_array(env.return_current_image())
        traj_x.append(env.robot.xy[0]); traj_y.append(env.robot.xy[1])

        return (im,)

    # 애니메이션 ----------------------------------------------------
    ani = animation.FuncAnimation(fig, update,
                                  frames=MAX_FRAMES,
                                  blit=True, interval=33)

    # GIF 저장 ------------------------------------------------------
    print(f"[Save] {gif_path} 생성 중 (다소 시간이 걸립니다)…")
    ani.save(gif_path, writer=animation.ImageMagickWriter(fps=30))
    print("✅  heuristic.gif 저장 완료!")

    # 궤적 PNG 저장 -------------------------------------------------
    plt.figure(figsize=(5, 5))
    plt.plot(traj_x, traj_y, '-', lw=1.2, color='red')
    plt.scatter(*policy.exit_xy, marker='*', s=150, color='yellow',
                edgecolor='black', linewidths=0.5, label='Exit')
    plt.gca().invert_yaxis(); plt.axis('equal')
    plt.title("Robot trajectory (HeuristicPolicy)")
    plt.legend(); plt.tight_layout()
    plt.savefig(traj_path, dpi=150)
    print(f"✅  trajectory.png 저장 완료!  ({traj_path})")

# -----------------------------------------------------------------
if __name__ == "__main__":
    main()
