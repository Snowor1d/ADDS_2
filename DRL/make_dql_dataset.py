#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dql_dataset.py
─────────────────────────────────────────────────────────
Diffusion-QL 학습용 오프라인 데이터셋(offine_dataset_dql.npz) 생성 스크립트

 • 수집 소스
      0 : RandomPolicy
      1 : HeuristicPolicy  (또는 --algo-ckpt 가 있을 경우 SACAgent)
      2 : Human-log (npz/npz 묶음, --human-data 로 지정)

 • HOME_DIR/<LOG_DIR>/offline_dataset_dql.npz 로 저장
"""

from __future__ import annotations

# ────────────────────────── 파이썬 표준 / 외부 ───────────────────────── #
import argparse
import math
import os
import random
from pathlib import Path
from typing import Callable, Iterable, List
from tqdm.auto import tqdm

import numpy as np
import torch

# ──────────────────────── 프로젝트 내부 모듈 ────────────────────────── #
from Start_training import LOG_DIR, ACTION_SCALE
import model                                      # FightingModel
from ADDS_AS_reinforcement import SACAgent        # 사전학습 정책 (선택)

# ═════════════════════════ Default 파라미터 ══════════════════════════ #
DEFAULTS = {
    "capacity"   : 500_000,          # 총 transition 수
    "r_random"   : 0.20,             # random  비율
    "r_algo"     : 0.80,             # heuristic/SAC 비율
    "r_human"    : 0,             # human   비율
    "map_num"    : -1,
    "crowd_size" : 20,
    "max_steps"  : 3_000,
    "seed"       : 42,
    "algo_ckpt"  : None,             # SAC checkpoint (.pt)
    "human_data" : None,             # human npz 경로(단일·디렉터리)
}
HOME_DIR = os.path.expanduser("~")
SAVE_PATH_DEFAULT = os.path.join(HOME_DIR, LOG_DIR, "offline_dataset_dql.npz")

# ╔═════════════════ Heuristic Policy (내장) ═════════════════╗
import numpy as np

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
    ):
        self.env = env
        self.R = sight_radius ## 로봇 영향 반경
        self.EPS = arrive_tol ## 도착 허용 오차
        self.DC = density_coef ## 밀집도 계수
        self.OC = obstacle_cost ## 장애물 패널티

        self.state = "SEARCH"
        self.target_agent = None
        self.following: set[int] = set()
        self.exit_xy: List[float] = env.exit_point[0]

    # ---------- 유틸 ----------
    def _dist(self, a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _alive_agents(self):
        return [ag for ag in self.env.crowds if not ag.dead]

    # ---------- 메인 ----------
    def select_action(self, _state_img=None) -> np.ndarray:
        robot = self.env.robot
        pos = robot.xy

        # 1) 상태 갱신 ──────────────────────────────────────────
        if self.state == "SEARCH":
            agents = self._alive_agents()
            if not agents:
                return np.zeros(2, np.float32) ## 전원 탈출
            self.target_agent = max(agents, key=lambda ag: self._dist(ag.xy, self.exit_xy))
            self.state = "TO_AGENT"

        elif self.state == "TO_AGENT":
            if (self.target_agent is None 
                or self.target_agent.dead 
                or self._dist(pos, self.target_agent.xy) < self.EPS):
                self.state = "TO_EXIT"

        elif self.state == "TO_EXIT":
            if self._dist(pos, self.exit_xy) < self.EPS:
                self.state = "WAIT"
                self.following = {
                    ag.unique_id for ag in self._alive_agents()
                    if self._dist(ag.xy, pos) < self.R
                }

        elif self.state == "WAIT":
            still_left = [
                ag for ag in self._alive_agents()
                if ag.unique_id in self.following
            ]
            if not still_left:
                self.state = "SEARCH"
                self.target_agent = None
                self.following.clear()

        # 2) 목표점 계산 ────────────────────────────────────────
        if self.state == "TO_AGENT" and self.target_agent:
            goal = self.target_agent.xy
        elif self.state in ("TO_EXIT", "WAIT"):
            goal = self.exit_xy
        else:        # SEARCH
            return np.zeros(2, np.float32)

        # 3) 8-방향 스코어링 ─────────────────────────────────────
        base_dir = np.array(goal) - np.array(pos)
        base_dir /= np.linalg.norm(base_dir) or 1.0

        dirs = [(1, 0), (1, 1), (0, 1), (-1, 1),
                (-1, 0), (-1, -1), (0, -1), (1, -1)]

        best_vec, best_score = None, -float("inf")
        for vx, vy in dirs:
            vec = np.array([vx, vy], float)
            vec /= np.linalg.norm(vec)
            nxt = (pos[0] + vec[0], pos[1] + vec[1])

            ix, iy = map(round, nxt)
            if not self.env.valid_space.get((ix, iy), 0):
                score = -self.OC
            else:
                score = -self._dist(nxt, self.exit_xy)          # 출구까지 거리
                density = sum(self._dist(ag.xy, nxt) < self.R * 1.5
                              for ag in self._alive_agents())
                score += self.DC * density
                score += 2.0 * np.dot(vec, base_dir)            # 진행 방향 일치도

            if score > best_score:
                best_score, best_vec = score, vec

        return (best_vec * 2.0).astype(np.float32)

# ════════════════════════════════════════════════════════════════ #

# ────────────────────── Random Policy ────────────────────────── #
class RandomPolicy: 
    """Δx, Δy ∈ [-2, 2] 균등 추출"""
    def select_action(self, *_):
        return np.random.uniform(-2.0, 2.0, 2).astype(np.float32)

# ────────────────────── Replay Buffer ────────────────────────── #
class ReplayBuffer:
    def __init__(self, capacity, state_shape=(50,50), action_dim=2):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.states      = np.zeros((capacity,*state_shape), np.uint8)
        self.next_states = np.zeros_like(self.states)
        self.actions     = np.zeros((capacity, action_dim), np.float32)
        self.rewards     = np.zeros(capacity, np.float32)
        self.dones       = np.zeros(capacity, bool)
        self.source_id   = np.zeros(capacity, np.uint8)

    def push(self, s,a,r,ns,d,src):
        i = self.ptr
        self.states[i], self.next_states[i] = s, ns
        self.actions[i], self.rewards[i], self.dones[i], self.source_id[i] = a, r, d, src
        self.ptr = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def save(self, path: str):
        np.savez_compressed(
            path,
            states      = self.states[:self.size],
            next_states = self.next_states[:self.size],
            actions     = self.actions[:self.size],
            rewards     = self.rewards[:self.size],
            dones       = self.dones[:self.size],
            source_id   = self.source_id[:self.size],
            size        = self.size,
            capacity    = self.capacity,
        )


# ─────────────────────────── Human 데이터 로드 ─────────────────────── #
def iter_human_npz(files: Iterable[Path]):
    """
    각 npz 는 'states','next_states','actions','rewards','dones' 키를 보유한다고 가정
    """
    for f in files:
        data = np.load(f)
        N = len(data["actions"])
        for i in range(N):
            yield (data["states"][i],
                   data["actions"][i],
                   data["rewards"][i].item() if "rewards" in data else 0.0,
                   data["next_states"][i],
                   data["dones"][i])

def load_human(buf: ReplayBuffer, human_path: str, max_samples: int):
    if human_path is None:
        return 0
    p = Path(human_path)
    files = list(p.glob("*.npz")) if p.is_dir() else [p]
    added = 0
    for trans in iter_human_npz(files):
        if added >= max_samples:
            break
        s,a,r,ns,d = trans
        buf.push(s, a.astype(np.float32), r, ns, bool(d), src=2)
        added += 1
    return added


# ───────────────── Transition 수집 ───────────────────────── #
def _is_numeric_action(x) -> bool:
    """np.ndarray, list, tuple → 수치형(길이 2) 인지 검사"""
    try:
        arr = np.asarray(x, dtype=np.float32)
    except (ValueError, TypeError):
        return False
    return (arr.shape == (2,)) and np.isfinite(arr).all()

def collect(policy_factory: str|Callable[[model.FightingModel], object],
            num_samples: int, src_tag: int, cfg: dict,
            buf: ReplayBuffer, sac_agent: SACAgent|None = None):
    
    collected = 0
    pbar = tqdm(total=num_samples,
                desc=f"[{policy_factory if isinstance(policy_factory,str) else 'heuristic'}]",
                ncols=80)
    
    while collected < num_samples:
        env = model.FightingModel(cfg["crowd_size"], 50, 50,
                                   model_num=cfg["map_num"], robot="Q")

        # 정책 선택 -----------------------------------------------------------
        if policy_factory == "random":
            policy = RandomPolicy()
        elif sac_agent:
            policy = sac_agent
        else:
            policy = policy_factory(env)

        state = env.return_current_image()

        for step in range(cfg["max_steps"]):
            if step % ACTION_SCALE == 0:
                raw = policy.select_action(state)
                eff = env.robot.receive_action(raw)

            env.step()
            next_state = env.return_current_image()
            done = env.robot.is_game_finished or step == cfg["max_steps"] - 1

            if (step % ACTION_SCALE == ACTION_SCALE - 1) or done:
                if not _is_numeric_action(eff):          # ← GUIDE 같은 경우 skip
                    continue
                buf.push(state,
                        np.asarray(eff, np.float32),
                        0.0, next_state, done, src_tag)
                collected += 1
                pbar.update(1)

                if collected >= num_samples:
                    break
            state = next_state
    pbar.close()


# ───────────────────────────── main ───────────────────────────── #
def main():
    # ────────────── CLI ──────────────
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in DEFAULTS.items():
        parser.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    parser.add_argument("--save-path", default=SAVE_PATH_DEFAULT)
    cfg = vars(parser.parse_args())

    # 시드 고정
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    # ────────────── 버퍼 준비 ──────────────
    buf  = ReplayBuffer(cfg["capacity"])
    sac  = None
    if cfg["algo_ckpt"]:
        sac = SACAgent(device="cpu"); sac.load_model(cfg["algo_ckpt"])

    # ──────────── human 우선 → 잔여 정규화 ────────────
    desired_human = int(cfg["capacity"] * cfg["r_human"])
    human_added   = load_human(buf, cfg["human_data"], desired_human)
    print(f"[Human]  requested={desired_human}  loaded={human_added}")

    remain = cfg["capacity"] - human_added
    # 원래 random:algo 비율로 재분배
    if cfg["r_random"]+cfg["r_algo"] == 0:
        r_rand = r_algo = 0.5
    else:
        r_rand = cfg["r_random"] / (cfg["r_random"] + cfg["r_algo"])
        r_algo = 1.0 - r_rand

    n_random = int(remain * r_rand)
    n_algo   = remain - n_random

    print(f"[Plan] random={n_random}, algo={n_algo}, human={human_added}")

    # ────────────── 수집 시작 ──────────────
    collect("random",           n_random, src_tag=0, cfg=cfg, buf=buf)
    collect(HeuristicPolicy,    n_algo,   src_tag=1, cfg=cfg, buf=buf, sac_agent=sac)

    # ────────────── 검증 & 저장 ──────────────
    buf.save(cfg["save_path"])
    print(f"\nSaved {buf.size} transitions → {cfg['save_path']}")

    cnt = np.bincount(buf.source_id[:buf.size], minlength=3)
    total = buf.size
    print(f"Ratio  random={cnt[0]/total:6.2%}  "
          f"algo={cnt[1]/total:6.2%}  human={cnt[2]/total:6.2%}")

    # 핵심 검증 (assertion 실패 시 ValueError 발생) ------------------------
    assert buf.size == cfg["capacity"], "Capacity와 실제 저장 수 mismatch!"
    assert np.all(np.abs(buf.actions[:buf.size]) <= 2.0001), "액션 범위 초과!"
    print("Basic sanity-check PASSED ✔")

# ───────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    main()