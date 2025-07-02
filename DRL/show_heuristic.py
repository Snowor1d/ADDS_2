#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_heuristic.py
────────────────────────────────────────────────────────
HeuristicPolicy 동작 시각화:
  • heuristic.gif   (30 fps)
  • trajectory.png
둘 다  ~/LOG_DIR/  아래에 저장
"""

from __future__ import annotations
import os, subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from Start_training import LOG_DIR, ACTION_SCALE
import model
from make_dql_dataset import HeuristicPolicy
from matplotlib.animation import PillowWriter, ImageMagickWriter

# ---------- 사용자 매개변수 ----------
DEBUG_PRINT = True
CROWD_SIZE  = 20
MAP_SIZE    = 50
<<<<<<< Updated upstream
MAX_FRAMES  = 4_000            # 충분히 길게
# ------------------------------------

def alive_agents(env: model.FightingModel) -> int:
    return sum(not ag.dead for ag in env.crowds)


def save_gif(anim, path, fps=30):
    try:
        anim.save(path, writer=PillowWriter(fps=fps))
        return
    except Exception as e:
        print("[WARN] PillowWriter 실패 → ImageMagick 시도:", e)
    anim.save(path, writer=ImageMagickWriter(fps=fps))
=======
MAX_FRAMES  = 3000           # 600 frame ≒ 20 s @ 30 fps
# ─────────────────────────────────────────────────────── #
>>>>>>> Stashed changes

def main() -> None:
    # ─ 경로 ----------
    save_dir = Path.home() / LOG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    gif_path  = save_dir / "heuristic.gif"
    traj_path = save_dir / "trajectory.png"

    # ─ Env & Policy ─
    env    = model.FightingModel(CROWD_SIZE, MAP_SIZE, MAP_SIZE,
                                 model_num=-1, robot="Q")
    policy = HeuristicPolicy(env)

    # ─ Figure ───────
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(env.return_current_image(),
                   cmap="gray", vmin=0, vmax=255,
                   origin="lower", animated=True)
    ax.axis("off")

    traj_x, traj_y = [], []

<<<<<<< Updated upstream
    # ─ 업데이트 ──────
    def update(f_idx: int):
        if alive_agents(env) == 0 or f_idx >= MAX_FRAMES-1:
            ani.event_source.stop()
=======
    # 업데이트 함수 -------------------------------------------------
    def update(frame_idx: int):
        if env.robot.is_game_finished:
            #ani.event_source.stop()          # ▶︎ 녹화 중단
>>>>>>> Stashed changes
            if DEBUG_PRINT:
                print(f"[{f_idx}] ✅  ALL AGENTS ESCAPED")
            return (im,)

        if f_idx % ACTION_SCALE == 0:
            raw = policy.select_action(None)
            env.robot.receive_action(raw)

            if DEBUG_PRINT:
                print(f"[{f_idx:4d}] {policy.state:<7} "
                      f"remain={alive_agents(env)}  "
                      f"robot={env.robot.xy!s:<20}")

        env.step()
        im.set_array(env.return_current_image())
        traj_x.append(env.robot.xy[0]); traj_y.append(env.robot.xy[1])
        return (im,)

    ani = animation.FuncAnimation(fig, update,
                                  frames=MAX_FRAMES, blit=True, interval=33)

    print(f"[Save] {gif_path}")
    save_gif(ani, gif_path, fps=30)   # <<<<<< 변경
    print("✅  heuristic.gif 저장 완료!")

    # trajectory.png  (invert_yaxis 제거)
    plt.figure(figsize=(5, 5))
    plt.plot(traj_x, traj_y, '-', lw=1.2, color='red')
    plt.scatter(*policy.exit_xy, marker='*', s=150,
                color='yellow', edgecolor='black')
    plt.axis('equal'); plt.xlim(0, MAP_SIZE); plt.ylim(0, MAP_SIZE)
    plt.tight_layout(); plt.savefig(traj_path, dpi=150)

if __name__ == "__main__":
    main()
