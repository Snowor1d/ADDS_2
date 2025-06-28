#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_heuristic.py
HeuristicPolicy 동작 시각화 → heuristic.gif 로 저장
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")          # GUI 없이 그림 저장
import matplotlib.pyplot as plt
from matplotlib import animation

import model
from make_dql_dataset import HeuristicPolicy       # 이미 정의돼 있음


def main():
    # ─ Env & Policy ────────────────────────────────────────────────
    env     = model.FightingModel(20, 50, 50, model_num=-1, robot="Q")
    policy  = HeuristicPolicy(env)
    max_len = 600                 # 600 step ≒ 20 sec @ 30 fps

    # ─ matplotlib 초기화 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    im      = ax.imshow(env.return_current_image(), cmap="gray", vmin=0, vmax=255)
    ax.axis("off")

    # ─ 애니메이션 한 프레임 그리기 ──────────────────────────────────
    def update(_frame):
        if env.robot.is_game_finished:
            return im,
        if _frame % 4 == 0:                      # ACTION_SCALE = 4
            raw = policy.select_action(None)
            env.robot.receive_action(raw)
        env.step()
        im.set_array(env.return_current_image())
        return im,

    ani = animation.FuncAnimation(fig, update,
                                  frames=max_len, blit=True, interval=33)

    # ─ GIF 저장 ────────────────────────────────────────────────────
    print("[Save] heuristic.gif 생성 중 (조금 걸립니다)…")
    ani.save("heuristic.gif",
             writer=animation.ImageMagickWriter(fps=30))
    print("✅  heuristic.gif 저장 완료 — 로컬에서 열어 확인하세요!")


if __name__ == "__main__":
    main()
