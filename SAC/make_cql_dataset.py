#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_cql_dataset.py
────────────────────────────────────────────────────────
• FightingModel 시뮬레이터에서 단일-프레임 transition 을 수집해
  CQL / BC 학습용 offline_dataset.npz 를 생성
• 저장 위치는 기존 로그 폴더( Start_training.LOG_DIR )와 동일

기본 실행
─────────
$ python make_cql_dataset.py

예시 옵션
─────────
$ python make_cql_dataset.py --capacity 300000 --r-random 0.1
"""

# ───────────────────────── 기본 설정 ───────────────────────── #
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Callable, List

import numpy as np
import torch

from Start_training import LOG_DIR, ACTION_SCALE          # 기존 파라미터 재사용
import model                                              # 시뮬레이터
from ADDS_AS_reinforcement import SACAgent                # (선택) SAC 정책

# 로그 폴더 및 기본 저장 경로 -------------------------------------------------
HOME_DIR = os.path.expanduser("~")
LOG_DIR_PATH = os.path.join(HOME_DIR, LOG_DIR)
os.makedirs(LOG_DIR_PATH, exist_ok=True)

# argparse 기본값 -------------------------------------------------------------
DEFAULTS: dict[str, object] = {
    "capacity":    500_000,      # 총 transition 개수
    "r_random":    0.20,         # 랜덤 데이터 비율
    "r_algo":      0.80,         # 휴리스틱 / SAC 데이터 비율
    "r_human":     0.00,         # (미사용) 인간 플레이 비율
    "map_num":     -1,           # FightingModel 맵 (-1 = 무작위)
    "crowd_size":  20,           # 군중 수
    "max_steps":   3_000,        # 1 episode 최대 스텝
    "seed":        42,
    "algo_ckpt":   None,         # SAC 모델을 정책으로 쓰고 싶을 때 경로
    "save_path":   os.path.join(LOG_DIR_PATH, "offline_dataset.npz"),
}

# ╔════════════════════════ Heuristic Policy ════════════════════════╗
class HeuristicPolicy:
    """
    로봇이 ➊출구에서 가장 먼 agent 를 찾아 이동 → ➋agent 밀집 지역을
    통과하며 출구로 복귀 → ➌끌고 온 agent 가 탈출할 때까지 대기, 를 반복.

    상태(state)
    ───────────
    • SEARCH   : target agent 선택 전
    • TO_AGENT : target agent 쪽으로 이동
    • TO_EXIT  : agent 근처 → 출구로 이동
    • WAIT     : 출구 도착, 끌고 온 agent 탈출 대기
    """

    def __init__(
        self,
        env: model.FightingModel,
        sight_radius: int = 7,
        arrive_tol: float = 2.0,
        density_coef: float = 8.0,
        obstacle_cost: float = 1_000.0,
    ) -> None:
        self.env = env

        # 파라미터
        self.R = sight_radius          # 로봇 영향 반경
        self.EPS = arrive_tol          # '도착' 허용 오차
        self.DC = density_coef         # 밀집 지역 보너스
        self.OC = obstacle_cost        # 장애물 패널티

        # 내부 상태
        self.state: str = "SEARCH"
        self.target_agent: model.CrowdAgent | None = None
        self.following: set[int] = set()
        self.exit_xy: List[float] = env.exit_point[0]  # 단일 출구 (x, y)

    # ────────────────────── 헬퍼 메서드 ──────────────────────
    def _dist(self, a: List[float], b: List[float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _alive_agents(self):
        return [ag for ag in self.env.crowds if not ag.dead]

    # ────────────────────── 주 행동 함수 ─────────────────────
    def select_action(self, *_unused) -> np.ndarray:
        robot = self.env.robot
        pos = robot.xy

        # 1) 상태 머신 갱신 --------------------------------------------------
        if self.state == "SEARCH":
            agents = self._alive_agents()
            if not agents:
                return np.zeros(2, dtype=np.float32)      # 전원 탈출 완료
            self.target_agent = max(
                agents, key=lambda ag: self._dist(ag.xy, self.exit_xy)
            )
            self.state = "TO_AGENT"

        elif self.state == "TO_AGENT":
            if (
                self.target_agent is None
                or self.target_agent.dead
                or self._dist(pos, self.target_agent.xy) < self.EPS
            ):
                self.state = "TO_EXIT"

        elif self.state == "TO_EXIT":
            if self._dist(pos, self.exit_xy) < self.EPS:
                self.state = "WAIT"
                self.following = {
                    ag.unique_id
                    for ag in self._alive_agents()
                    if self._dist(ag.xy, pos) < self.R
                }

        elif self.state == "WAIT":
            still_left = [
                ag
                for ag in self._alive_agents()
                if ag.unique_id in self.following
            ]
            if not still_left:
                self.state = "SEARCH"
                self.target_agent = None
                self.following.clear()

        # 2) 목표점 결정 ------------------------------------------------------
        if self.state == "TO_AGENT" and self.target_agent is not None:
            goal = self.target_agent.xy
        elif self.state in ("TO_EXIT", "WAIT"):
            goal = self.exit_xy
        else:
            # SEARCH 상태에서는 가만히 있기
            return np.zeros(2, dtype=np.float32)

        # 3) 8-방향 스코어링 ---------------------------------------------------
        base_dir = np.array(goal) - np.array(pos)
        base_dir /= np.linalg.norm(base_dir) or 1.0

        directions = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ]

        best_score = -float("inf")
        best_vec: np.ndarray | None = None

        for vx, vy in directions:
            vec = np.array([vx, vy], dtype=float)
            vec /= np.linalg.norm(vec)
            nxt = (pos[0] + vec[0], pos[1] + vec[1])

            ix, iy = int(round(nxt[0])), int(round(nxt[1]))
            if not self.env.valid_space.get((ix, iy), 0):
                # 장애물: 큰 패널티
                score = -self.OC
            else:
                # 기본 점수: 출구까지 남은 거리 (짧을수록 좋음 → 음수 부호)
                score = -self._dist(nxt, self.exit_xy)

                # agent 밀집 보너스
                density = sum(
                    self._dist(ag.xy, nxt) < self.R * 1.5
                    for ag in self._alive_agents()
                )
                score += self.DC * density

                # 기준 방향(base_dir) 과 각도 차이 보정
                score += 2.0 * np.dot(vec, base_dir)

            if score > best_score:
                best_score = score
                best_vec = vec

        # 4) Δx, Δy 반환 (스케일 [-2, 2])
        assert best_vec is not None
        return (best_vec * 2.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════


# ───────────────────────── Random Policy ───────────────────────── #
class RandomPolicy:
    """Δx, Δy ∈ [-2, 2] 균등 추출"""

    def select_action(self, *_unused) -> np.ndarray:
        return np.random.uniform(-2.0, 2.0, size=2)


# ───────────────────────── Replay Buffer ───────────────────────── #
class ReplayBuffer:
    """단일 프레임(state_shape=(50,50)) 버퍼 + source_id"""

    def __init__(
        self,
        capacity: int,
        state_shape: tuple = (50, 50),
        action_dim: int = 2,
        dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = capacity

        self.states = np.zeros((capacity, *state_shape), dtype=dtype)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.source_id = np.zeros(capacity, dtype=np.uint8)

        self.ptr = 0
        self.size = 0

    # --------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        src_tag: int,
    ) -> None:
        idx = self.ptr

        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.source_id[idx] = src_tag

        self.ptr = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # --------------------------------------------------------------
    def save(self, file_path: str | os.PathLike) -> None:
        np.savez_compressed(
            file_path,
            states=self.states[: self.size],
            next_states=self.next_states[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            dones=self.dones[: self.size],
            source_id=self.source_id[: self.size],
            size=self.size,
            capacity=self.capacity,
            dtype=str(self.states.dtype),
        )


# ────────────────────── Transition 수집 함수 ──────────────────────
def collect_transitions(
    policy_factory: str | Callable[[model.FightingModel], object],
    num_samples: int,
    src_tag: int,
    cfg: dict,
    buffer: ReplayBuffer,
    sac_agent: SACAgent | None = None,
) -> None:
    """
    지정된 정책으로 num_samples 개의 transition 을 수집하여 buffer 에 push.
    """
    samples_collected = 0

    while samples_collected < num_samples:
        env = model.FightingModel(
            cfg["crowd_size"],
            50,
            50,
            model_num=cfg["map_num"],
            robot="Q",
        )

        # 정책 인스턴스 준비
        if policy_factory == "random":
            policy = RandomPolicy()
        elif sac_agent is not None:
            policy = sac_agent  # SACAgent 도 select_action(state) 인터페이스
        else:
            policy = policy_factory(env)

        state = env.return_current_image()

        for step in range(cfg["max_steps"]):
            # ACTION_SCALE 간격으로만 new action
            if step % ACTION_SCALE == 0:
                action_raw = policy.select_action(state)
                action_eff = env.robot.receive_action(action_raw)

            env.step()
            next_state = env.return_current_image()

            done = env.robot.is_game_finished or (step == cfg["max_steps"] - 1)

            # transition 저장 시점: ACTION_SCALE-1 또는 episode 종료
            if step % ACTION_SCALE == ACTION_SCALE - 1 or done:
                buffer.push(
                    state=state,
                    action=np.asarray(action_eff, dtype=np.float32),
                    reward=0.0,
                    next_state=next_state,
                    done=done,
                    src_tag=src_tag,
                )
                samples_collected += 1

                if samples_collected >= num_samples:
                    break

            state = next_state


# ────────────────────────────── main ──────────────────────────────
def main() -> None:
    # argparse -----------------------------------------------------
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)

    cfg = vars(parser.parse_args())

    # 랜덤 시드 -----------------------------------------------------
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # 수집 개수 계산 ----------------------------------------------
    n_random = int(cfg["capacity"] * cfg["r_random"])
    n_algo = int(cfg["capacity"] * cfg["r_algo"])
    n_human = cfg["capacity"] - n_random - n_algo

    # 버퍼 및 (선택) SAC 정책 --------------------------------------
    buffer = ReplayBuffer(cfg["capacity"])
    sac_policy = None

    if cfg["algo_ckpt"] is not None:
        sac_policy = SACAgent(device="cpu")
        sac_policy.load_model(cfg["algo_ckpt"])

    # 수집 ---------------------------------------------------------
    print(f"[Collect] random={n_random}  algo={n_algo}")

    collect_transitions(
        policy_factory="random",
        num_samples=n_random,
        src_tag=0,
        cfg=cfg,
        buffer=buffer,
    )

    collect_transitions(
        policy_factory=HeuristicPolicy,
        num_samples=n_algo,
        src_tag=1,
        cfg=cfg,
        buffer=buffer,
        sac_agent=sac_policy,
    )

    # 저장 ---------------------------------------------------------
    buffer.save(cfg["save_path"])

    counts = np.bincount(buffer.source_id[: buffer.size], minlength=3)
    print(f"\nSaved {buffer.size} transitions  →  {cfg['save_path']}")
    print(
        f" Ratio  random={counts[0] / buffer.size:.2%},"
        f"  algo={counts[1] / buffer.size:.2%},"
        f"  human={counts[2] / buffer.size:.2%}"
    )


if __name__ == "__main__":
    main()
