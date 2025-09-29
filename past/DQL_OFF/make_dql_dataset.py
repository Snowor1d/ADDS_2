#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dql_dataset.py
────────────────────────────────────────────────────────
Diffusion-QL 학습용 오프라인 데이터셋 생성 스크립트
  • 저장 위치 :  <HOME_DIR>/<LOG_DIR>/offline_dataset_dql.npz
  • 정책      : Random / Heuristic / (선택) SAC                              
"""

from __future__ import annotations

# ───────────────────── 기본 라이브러리 ───────────────────── #
import argparse, math, os, random, sys
from typing import Callable, List

import numpy as np
import torch
from tqdm import tqdm                              

# ───────────────────── 프로젝트 모듈 ───────────────────── #
from Start_training import *
import model                                        # FightingModel
from ADDS_AS_reinforcement import DiffusionQLAgent, ReplayBuffer       # 옵션: 선행 학습 모델 불러오기

# ───────────────────── 기본 설정값 ───────────────────── #
DEFAULTS = {
    "capacity"  : 50,          # 총 transition 수
    "r_random"  : 0.20,             # 랜덤 데이터 비율
    "r_algo"    : 0.80,             # 휴리스틱+SAC 비율
    "r_human"   : 0.00,             # 인간 데모 비율(현재 0)
    "map_num"   : -1,
    "crowd_size": 20,
    "max_steps" : 3_000,
    "seed"      : 42,
    "algo_ckpt" : None,             # --algo-ckpt 경로
}
HOME_DIR = os.path.expanduser("~")
SAVE_PATH_DEFAULT = os.path.join(HOME_DIR, LOG_DIR, "offline_dataset_dql.npz")

# ╔════════ Heuristic Policy (내장) ═════════╗
class HeuristicPolicy:
    """
    (1) 출구에서 가장 먼 agent로 이동
    (2) agent 밀집 지역을 통과하며 출구로 복귀
    (3) 끌고 온 agent 전부 탈출 시까지 대기
    ── 위 과정을 반복하는 휴리스틱 정책
    """
    def __init__(self, env: model.FightingModel,
                 sight_radius: int = 7,
                 arrive_tol: float = 2.0,
                 density_coef: float = 8.0,
                 obstacle_cost: float = 1_000.0):
        self.env = env
        self.R   = sight_radius
        self.EPS = arrive_tol
        self.DC  = density_coef
        self.OC  = obstacle_cost
        self.state = "SEARCH"
        self.target_agent = None
        self.following: set[int] = set()
        self.exit_xy: list[float] = env.exit_point[0]

    def _dist(self, a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _alive(self):
        return [ag for ag in self.env.crowds if not ag.dead]

    def _reached(self, a, b) -> bool:
        return self._dist(a, b) < self.EPS

    def _update_state(self, pos):
        if self.state == "SEARCH":
            agents = self._alive()
            if not agents:
                return
            self.target_agent = max(
                agents, key=lambda ag: self._dist(ag.xy, self.exit_xy))
            self.state = "TO_AGENT"

        elif self.state == "TO_AGENT":
            if (self.target_agent is None or self.target_agent.dead
                    or self._reached(pos, self.target_agent.xy)):
                self.state = "TO_EXIT"

        elif self.state == "TO_EXIT":
            if self._reached(pos, self.exit_xy):
                self.following = {ag.unique_id for ag in self._alive()
                                  if self._dist(ag.xy, pos) < self.R}
                self.state = "WAIT"

        elif self.state == "WAIT":
            self.following = {ag.unique_id for ag in self._alive()
                              if self._dist(ag.xy, pos) < self.R
                                 and ag.unique_id in self.following}
            if not self.following:
                self.state, self.target_agent = "SEARCH", None

    def select_action(self, robot=None) -> np.ndarray:
        pos = self.env.robot.xy
        self._update_state(pos)

        if self.state == "TO_AGENT" and self.target_agent:
            goal = self.target_agent.xy
        elif self.state in ("TO_EXIT", "WAIT"):
            goal = self.exit_xy
        else:
            return np.zeros(2, np.float32)

        p = np.asarray(pos, dtype=np.float32)
        g = np.asarray(goal, dtype=np.float32)
        base_dir = g - p
        base_dir /= np.linalg.norm(base_dir) or 1.0

        dirs = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
        best_vec, best_score = None, -float("inf")

        for vx, vy in dirs:
            vec = np.asarray([vx, vy], dtype=np.float32)
            vec /= np.linalg.norm(vec)
            nxt = (p[0] + vec[0], p[1] + vec[1])
            ix, iy = map(round, nxt)
            score = 0
            if not self.env.valid_space.get((ix, iy), 0):
                score = -self.OC * 100
            else:
                score  = -self.env.robot.point_to_point_distance(nxt, self.exit_xy)*0.05

                density = sum(self._dist(ag.xy, nxt) < self.R*1.5
                              for ag in self._alive())
                score += self.DC * density
                score += 2.0 * np.dot(vec, base_dir)

            if score > best_score:
                best_score, best_vec = score, vec

        return (best_vec * 2.0).astype(np.float32)


# ─────── Random Policy ─────── #
class RandomPolicy:
    """Δx, Δy ∈ [-2,2] 균등 추출"""
    def select_action(self, *_, robot=None):
        return np.random.uniform(-2.0, 2.0, 2).astype(np.float32)


# ─────── Transition 수집 ─────── #
def collect(policy_factory: str|Callable[[model.FightingModel],object],
            num_samples:int, src_tag:int, cfg:dict, buf:ReplayBuffer,
            sac_agent:DiffusionQLAgent|None=None):
    src_name = {0:"random",1:"algo",2:"human"}.get(src_tag,f"src{src_tag}")

    with tqdm(total=num_samples, desc=f"[{src_name}]", unit="ts", ncols=70) as pbar:
        collected = 0
        while collected < num_samples:
            env = model.FightingModel(
                cfg["crowd_size"], 50, 50, model_num=cfg["map_num"], robot="Q"
            )

            if policy_factory == "random":
                policy = RandomPolicy()
            elif sac_agent:
                policy = sac_agent
            else:
                policy = policy_factory(env)

            state = env.return_current_image()

            for step in range(cfg["max_steps"]):
                if step % ACTION_SCALE == 0:
                    raw = policy.select_action(state, robot=env.robot)
                    eff = env.robot.receive_action(raw)
                    if isinstance(eff, str) or (isinstance(eff, np.ndarray) and eff.dtype.kind not in 'fiu'):
                        eff = raw

                # 환경 스텝
                env.step()

                # 보상 계산 (imitation 스크립트 로직 재활용)
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
                    reward += 10.0

                next_state = env.return_current_image()
                done = env.robot.is_game_finished or step == cfg["max_steps"]-1

                if step % ACTION_SCALE == ACTION_SCALE-1 or done:
                    eff_arr = np.asarray(eff)
                    if not np.issubdtype(eff_arr.dtype, np.number):
                        continue
                    buf.push(
                        state,
                        np.asarray(eff, np.float32),
                        float(reward),
                        next_state,
                        done,
                        src_tag
                    )

                    collected += 1
                    pbar.update(1); pbar.set_postfix(ts=collected); pbar.refresh()
                    if collected >= num_samples:
                        break

                    if done:
                        break

                state = next_state

                


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in DEFAULTS.items():
        parser.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    parser.add_argument("--save-path", type=str, default=SAVE_PATH_DEFAULT)
    cfg = vars(parser.parse_args())

    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    print(f"[Save-Path] {cfg['save_path']}")

    n_random = int(cfg["capacity"] * cfg["r_random"])
    n_algo   = int(cfg["capacity"] * cfg["r_algo"])
    n_human  = cfg["capacity"] - n_random - n_algo

    human_data = None
    if cfg["r_human"] > 0:
        try:
            human_npz = np.load(os.path.join(LOG_DIR, "human_demo.npz"))
            human_data = human_npz["transitions"]
        except FileNotFoundError:
            print("[Human] 파일 없음 → human=0 으로 처리")
            n_human = 0

    print(f"[Plan] random={n_random}, algo={n_algo}, human={n_human}")

    buf = ReplayBuffer(cfg["capacity"], (50, 50), 2, 'cpu')
    sac = None
    if cfg["algo_ckpt"]:
        sac = DiffusionQLAgent(device="cpu"); sac.load_model(cfg["algo_ckpt"])

    if n_random: collect("random",        n_random, src_tag=0, cfg=cfg, buf=buf)
    if n_algo:   collect(HeuristicPolicy, n_algo,   src_tag=1, cfg=cfg, buf=buf, sac_agent=sac)
    if n_human and human_data is not None:
        for i in range(min(n_human, len(human_data))):
            s,a,ns,d = human_data[i]
            buf.push(s, a.astype(np.float32), 0.0, ns, bool(d), 2)

    buf.save(cfg["save_path"])
    #cnts = np.bincount(buf.source_id[:buf.size], minlength=3)
    print(f"\nSaved {buf.size} transitions → {cfg['save_path']}")
    #print(f"Ratio random={cnts[0]/buf.size:.2%}, algo={cnts[1]/buf.size:.2%}, human={cnts[2]/buf.size:.2%}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] partial dataset NOT saved.", file=sys.stderr)
