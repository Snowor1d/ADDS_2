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
from tqdm import tqdm                               # ★ 진행률 표시

# ───────────────────── 프로젝트 모듈 ───────────────────── #
from Start_training import LOG_DIR, ACTION_SCALE
import model                                        # FightingModel
from ADDS_AS_reinforcement import SACAgent          # 옵션: 선행 학습 모델 불러오기

# ───────────────────── 기본 설정값 ───────────────────── #
DEFAULTS = {
    "capacity"  : 500_000,          # 총 transition 수
    "r_random"  : 0.20,             # 랜덤 데이터 비율
    "r_algo"    : 0.80,             # 휴리스틱+SAC 비율
    "r_human"   : 0.00,             # ★ 인간 데모 비율(현재 0)
    "map_num"   : -1,
    "crowd_size": 20,
    "max_steps" : 3_000,
    "seed"      : 42,
    "algo_ckpt" : None,             # --algo-ckpt 경로
}
HOME_DIR = os.path.expanduser("~")
SAVE_PATH_DEFAULT = os.path.join(HOME_DIR, LOG_DIR, "offline_dataset_dql.npz")

# ╔═══════════════ Heuristic Policy (내장) ═══════════════╗
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
    def __init__(self, env: model.FightingModel,
                 sight_radius=7, arrive_tol=2.0,
                 density_coef=8.0, obstacle_cost=1_000.0):
        self.env = env
        self.R, self.EPS, self.DC, self.OC = sight_radius, arrive_tol, density_coef, obstacle_cost
        self.state, self.target_agent, self.following = "SEARCH", None, set()
        self.exit_xy: List[float] = env.exit_point[0]

    # ---------- 유틸 ----------
    def _dist(self, a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
    def _alive_agents(self):
        return [ag for ag in self.env.crowds if not ag.dead]

    # ---------- 메인 ----------
    def select_action(self, _state_img=None) -> np.ndarray:
        robot, pos = self.env.robot, self.env.robot.xy

        # 1) 상태 머신 -------------------------------------------------
        if self.state == "SEARCH":
            agents = self._alive_agents()
            if not agents: return np.zeros(2, np.float32)   # 전원 탈출
            self.target_agent = max(agents, key=lambda ag: self._dist(ag.xy, self.exit_xy))
            self.state = "TO_AGENT"

        elif self.state == "TO_AGENT":
            if (self.target_agent is None or self.target_agent.dead
                    or self._dist(pos, self.target_agent.xy) < self.EPS):
                self.state = "TO_EXIT"

        elif self.state == "TO_EXIT":
            if self._dist(pos, self.exit_xy) < self.EPS:
                self.state = "WAIT"
                self.following = {ag.unique_id for ag in self._alive_agents()
                                  if self._dist(ag.xy, pos) < self.R}

        elif self.state == "WAIT":
            if not [ag for ag in self._alive_agents() if ag.unique_id in self.following]:
                self.state, self.target_agent, self.following = "SEARCH", None, set()

        # 2) 목표점 ----------------------------------------------------
        if self.state == "TO_AGENT" and self.target_agent:
            goal = self.target_agent.xy
        elif self.state in ("TO_EXIT", "WAIT"):
            goal = self.exit_xy
        else:                         # SEARCH
            return np.zeros(2, np.float32)

        # 3) 8-방향 스코어링 -------------------------------------------
        p = np.asarray(pos , dtype=np.float32)
        g = np.asarray(goal, dtype=np.float32)
        base_dir = g - p

        norm = np.linalg.norm(base_dir) or 1.0
        base_dir /= norm

        dirs = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
        best_vec, best_score = None, -float("inf")
        for vx, vy in dirs:
            vec = np.array([vx, vy], dtype=np.float32)
            vec /= np.linalg.norm(vec)

            nxt = (pos[0] + vec[0], pos[1] + vec[1])
            ix, iy = map(round, nxt)

            if not self.env.valid_space.get((ix, iy), 0):
                score = -self.OC
            else:
                score  = -self._dist(nxt, self.exit_xy)                # 출구까지 거리
                density = sum(self._dist(ag.xy, nxt) < self.R*1.5 for ag in self._alive_agents())
                score += self.DC * density                             # agent 밀집 보너스
                score += 2.0 * np.dot(vec, base_dir)                   # 진행 방향 일치도
            if score > best_score:
                best_score, best_vec = score, vec

        return (best_vec * 2.0).astype(np.float32)

# ════════════════════════════════════════════════════════════════ #

# ─────────────────── Random Policy ─────────────────── #
class RandomPolicy:
    """Δx, Δy ∈ [-2,2] 균등 추출"""
    def select_action(self, *_):  # noqa: D401
        return np.random.uniform(-2.0, 2.0, 2).astype(np.float32)

# ─────────────────── Replay Buffer ─────────────────── #
class ReplayBuffer:
    def __init__(self, capacity, state_shape=(50,50), action_dim=2):
        self.capacity = capacity; self.ptr = 0; self.size = 0
        self.states      = np.zeros((capacity, *state_shape), np.uint8)
        self.next_states = np.zeros_like(self.states)
        self.actions     = np.zeros((capacity, action_dim), np.float32)
        self.rewards     = np.zeros(capacity, np.float32)
        self.dones       = np.zeros(capacity, bool)
        self.source_id   = np.zeros(capacity, np.uint8)

    def push(self, s,a,r,ns,done,src):
        i = self.ptr
        self.states[i], self.next_states[i] = s, ns
        self.actions[i], self.rewards[i]    = a, r
        self.dones[i],  self.source_id[i]   = done, src
        self.ptr  = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states[:self.size], next_states=self.next_states[:self.size],
            actions=self.actions[:self.size], rewards=self.rewards[:self.size],
            dones=self.dones[:self.size],     source_id=self.source_id[:self.size],
            size=self.size, capacity=self.capacity,
        )

# ─────────────────── Transition 수집 ─────────────────── #
def collect(policy_factory: str|Callable[[model.FightingModel],object],
            num_samples:int, src_tag:int, cfg:dict, buf:ReplayBuffer,
            sac_agent:SACAgent|None=None):
    src_name = {0:"random",1:"algo",2:"human"}.get(src_tag,f"src{src_tag}")

    with tqdm(total=num_samples, desc=f"[{src_name}]", unit="ts", ncols=70) as pbar:
        collected = 0
        while collected < num_samples:
            env = model.FightingModel(
                cfg["crowd_size"], 50, 50, model_num=cfg["map_num"], robot="Q"
            )

            # ---------- 정책 준비 ----------
            if policy_factory == "random":
                policy = RandomPolicy()
            elif sac_agent:                       # 사전학습 SAC
                policy = sac_agent
            else:                                # Heuristic
                policy = policy_factory(env)

            state = env.return_current_image()

            for step in range(cfg["max_steps"]):
                # ACTION_SCALE 마다 한 번만 새로운 액션
                if step % ACTION_SCALE == 0:
                    raw = policy.select_action(state)
                    eff = env.robot.receive_action(raw)

                    # ★ GUIDE 문자열이면 raw 그대로 사용
                    if isinstance(eff, str) or (isinstance(eff, np.ndarray) and eff.dtype.kind not in 'fiu'):
                        eff = raw

                env.step()
                next_state = env.return_current_image()
                done = env.robot.is_game_finished or step == cfg["max_steps"]-1

                if step % ACTION_SCALE == ACTION_SCALE-1 or done:
                    # 숫자형 아니라면 skip
                    eff_arr = np.asarray(eff)
                    is_numeric = np.issubdtype(eff_arr.dtype, np.number)
                    if not is_numeric:
                        continue
                    buf.push(state, np.asarray(eff, np.float32), 0.0,
                             next_state, done, src_tag)

                    collected += 1
                    pbar.update(1); pbar.set_postfix(ts=collected); pbar.refresh()
                    if collected >= num_samples:
                        break

                state = next_state
            # end episode loop
    # end tqdm context

# ───────────────────────────── main ───────────────────────────── #
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for k,v in DEFAULTS.items():
        parser.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    parser.add_argument("--save-path", type=str, default=SAVE_PATH_DEFAULT)
    cfg = vars(parser.parse_args())

    # ---------- 시드 ----------
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    # ---------- 저장 경로 출력 ----------
    print(f"[Save-Path] {cfg['save_path']}")

    # ---------- 샘플 수 할당 ----------
    n_random = int(cfg["capacity"] * cfg["r_random"])
    n_algo   = int(cfg["capacity"] * cfg["r_algo"])
    n_human  = cfg["capacity"] - n_random - n_algo            # 누적 오차 보정 ★

    # ---------- Human NPZ 로드 ----------
    human_data: np.ndarray|None = None
    if cfg["r_human"] > 0:
        try:
            human_npz = np.load(os.path.join(LOG_DIR, "human_demo.npz"))
            human_data = human_npz["transitions"]
        except FileNotFoundError:
            print("[Human]  파일 없음 → human=0 으로 처리")
            n_human = 0

    print(f"[Plan] random={n_random}, algo={n_algo}, human={n_human}")

    # ---------- 버퍼/에이전트 ----------
    buf = ReplayBuffer(cfg["capacity"])
    sac = None
    if cfg["algo_ckpt"]:
        sac = SACAgent(device="cpu"); sac.load_model(cfg["algo_ckpt"])

    # ---------- 수집 ----------
    if n_random: collect("random",           n_random, src_tag=0, cfg=cfg, buf=buf)
    if n_algo:   collect(HeuristicPolicy,    n_algo,   src_tag=1, cfg=cfg, buf=buf, sac_agent=sac)
    if n_human and human_data is not None:
        # human_data 는 (N, state, action, next_state, done) 구조라고 가정
        for i in range(min(n_human, len(human_data))):
            s,a,ns,d = human_data[i]
            buf.push(s, a.astype(np.float32), 0.0, ns, bool(d), 2)

    # ---------- 저장 ----------
    buf.save(cfg["save_path"])
    cnts = np.bincount(buf.source_id[:buf.size], minlength=3)
    print(f"\nSaved {buf.size} transitions → {cfg['save_path']}")
    print(f"Ratio random={cnts[0]/buf.size:.2%}, algo={cnts[1]/buf.size:.2%}, "
          f"human={cnts[2]/buf.size:.2%}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted] partial dataset NOT saved.", file=sys.stderr)
