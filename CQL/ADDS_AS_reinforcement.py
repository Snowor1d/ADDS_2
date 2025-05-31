"""
train_sac_cql.py
──────────────────────────────────────────────────────────────────────────
• NumPy 링-버퍼 ReplayBuffer
• Conservative Q-Learning(CQL) 패널티 옵션
• OFFLINE_MODE ↔ 온라인 파인튜닝 전환
• 텐서보드 실시간 스트리밍
"""

# ╔═════════════════════╗
# ║ 0. 표준 라이브러리  ║
# ╚═════════════════════╝
import os, time, random, subprocess, webbrowser, threading
from typing import Union
import numpy as np

# ╔═════════════════════╗
# ║ 0-1. PyTorch        ║
# ╚═════════════════════╝
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# ╔═════════════════════╗
# ║ 0-2. 프로젝트 공용  ║
# ╚═════════════════════╝
from timer_utils import Timer
from config import ENABLE_TIMER
from Start_training import *                 # 모든 하이퍼파라미터/플래그
from heat_map import HeatMapLogger           # (필요 시)

# ─────────────────────────────────────────────────────────────────────────
# 1. TensorBoard 도우미
# ─────────────────────────────────────────────────────────────────────────
def launch_tensorboard(logdir: str, port: int = 6006):
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", logdir, "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}")
    return proc

def tail_txt_to_tb(txt_path: str, tag: str, tb_dir: str):
    writer = SummaryWriter(tb_dir)
    step = 0
    while not os.path.exists(txt_path):
        time.sleep(1)
    with open(txt_path) as f:
        while True:
            ln = f.readline()
            if ln:
                try:
                    writer.add_scalar(tag, float(ln.strip()), step)
                    step += 1
                except ValueError:
                    pass
            else:
                time.sleep(0.5)

# ─────────────────────────────────────────────────────────────────────────
# 2. ReplayBuffer – NumPy 고정 배열 링 버퍼
# ─────────────────────────────────────────────────────────────────────────
class ReplayBuffer:
    """메모리 절약형 NumPy 링 버퍼"""
    def __init__(self,
                 capacity: int,
                 state_shape=(1, 50, 50),
                 action_dim: int = 2,
                 device: str = "cpu",
                 dtype=np.uint8):
        self.cap = capacity
        self.dev = torch.device(device)
        self.dtype = dtype

        self.s = np.zeros((capacity, *state_shape), dtype=dtype)
        self.ns = np.zeros_like(self.s)
        self.a = np.zeros((capacity, action_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=bool)

        self.ptr = 0
        self.size = 0

    # --------------------------------------------------
    def push(self, s, a, r, ns, done):
        i = self.ptr
        self.s[i] = s.astype(self.dtype, copy=False)
        self.ns[i] = ns.astype(self.dtype, copy=False)
        self.a[i] = a
        self.r[i] = r
        self.d[i] = done
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    # --------------------------------------------------
    def sample(self, batch: int):
        idx = np.random.choice(self.size, batch, replace=False)
        s = torch.from_numpy(self.s[idx].astype(np.float32)).to(self.dev)
        ns = torch.from_numpy(self.ns[idx].astype(np.float32)).to(self.dev)
        a = torch.from_numpy(self.a[idx]).to(self.dev)
        r = torch.from_numpy(self.r[idx]).to(self.dev)
        d = torch.from_numpy(self.d[idx].astype(np.float32)).to(self.dev)
        return s, a, r, ns, d

    # --------------------------------------------------
    def save(self, path: Union[str, os.PathLike]):
        np.savez_compressed(
            path,
            s=self.s[:self.size], ns=self.ns[:self.size],
            a=self.a[:self.size], r=self.r[:self.size], d=self.d[:self.size],
            size=self.size, ptr=self.ptr, cap=self.cap,
            dtype=np.dtype(self.dtype).name
        )

    def load(self, path: Union[str, os.PathLike]):
        data = np.load(path, allow_pickle=False)
        self.size = int(data["size"]); self.ptr = int(data["ptr"])
        self.s[:self.size] = data["s"]; self.ns[:self.size] = data["ns"]
        self.a[:self.size] = data["a"]; self.r[:self.size] = data["r"]
        self.d[:self.size] = data["d"]

    # --------------------------------------------------
    def __len__(self): return self.size

# ─────────────────────────────────────────────────────────────────────────
# 3. 네트워크 정의
# ─────────────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, input_shape=(50, 50), action_dim=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.01, True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.01, True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.01, True)
        )
        with torch.no_grad():
            flat = int(np.prod(self.enc(torch.zeros(1, 1, *input_shape)).shape[1:]))
        self.head = nn.Sequential(
            nn.Linear(flat + action_dim, 512), nn.LeakyReLU(0.01, True),
            nn.Linear(512, 256), nn.LeakyReLU(0.01, True),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        f = self.enc(s).view(s.size(0), -1)
        return self.head(torch.cat([f, a], 1))

class PolicyNetwork(nn.Module):
    def __init__(self, input_shape=(50, 50)):
        super().__init__()
        self.lmin, self.lmax = -5.0, 2.0
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.BatchNorm2d(32), nn.LeakyReLU(0.01, True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.01, True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.01, True)
        )
        with torch.no_grad():
            flat = int(np.prod(self.enc(torch.zeros(1, 1, *input_shape)).shape[1:]))
        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.LeakyReLU(0.01, True),
            nn.Linear(512, 256), nn.LeakyReLU(0.01, True),
            nn.Linear(256, 64), nn.LeakyReLU(0.01, True)
        )
        self.mean = nn.Linear(64, 2)
        self.log_std = nn.Linear(64, 2)

    # --------------------------------------------------
    def _dist(self, s):
        f = self.fc(self.enc(s).view(s.size(0), -1))
        m = self.mean(f)
        ls = torch.clamp(self.log_std(f), self.lmin, self.lmax)
        return m, ls

    # --------------------------------------------------
    def sample_action(self, s, temperature: float = 1.0):
        m, ls = self._dist(s)
        std = ls.exp()
        u = m + std * torch.randn_like(m) * temperature
        sig = torch.sigmoid(u)
        a = 4 * sig - 2                                      # [-2, 2]
        logp_u = -0.5 * (((u - m) / (std + 1e-8))**2 + 2*ls + np.log(2*np.pi)).sum(1)
        jac = torch.log(4 * sig * (1 - sig) + 1e-8).sum(1)
        return a, (logp_u - jac)

# ─────────────────────────────────────────────────────────────────────────
# 4. SAC + CQL 에이전트
# ─────────────────────────────────────────────────────────────────────────
class SAC_CQL_Agent:
    def __init__(self):
        # 디바이스
        self.dev = torch.device(DEVICE)

        # 하이퍼파라미터
        self.gamma = GAMMA_START
        self.alpha = ALPHA_START
        self.cql_alpha = CQL_ALPHA_START
        self.tau = 0.995
        self.batch_size = BATCH_SIZE

        # 네트워크
        self.q1 = QNetwork().to(self.dev)
        self.q2 = QNetwork().to(self.dev)
        self.q1_t = QNetwork().to(self.dev)
        self.q2_t = QNetwork().to(self.dev)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.pi = PolicyNetwork().to(self.dev)

        # 옵티마이저
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=LR)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=LR)
        self.pi_opt = optim.Adam(self.pi.parameters(), lr=LR)

        # 버퍼
        self.replay = ReplayBuffer(BUFFER_SIZE, device=self.dev)

    # --------------------------------------------------
    def soft(self, src: nn.Module, tgt: nn.Module):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * tp.data + (1 - self.tau) * p.data)

    # --------------------------------------------------
    def cql_regularizer(self, s, a):
        with torch.no_grad():
            rand_a = torch.empty((s.size(0) * CQL_K, 2), device=self.dev).uniform_(-2, 2)
            s_rep = s.unsqueeze(1).repeat(1, CQL_K, 1, 1, 1).view(-1, 1, 50, 50)
        q_rand = self.q1(s_rep, rand_a).view(s.size(0), CQL_K)
        logsumexp = torch.logsumexp(q_rand, 1)
        q_in = self.q1(s, a).squeeze(-1)
        return (logsumexp - q_in).mean() * self.cql_alpha

    def step_alpha(self, global_step: int):
        if global_step >= CQL_ALPHA_DECAY_STEPS:
            self.cql_alpha = CQL_ALPHA_END
        else:
            t = global_step / CQL_ALPHA_DECAY_STEPS
            self.cql_alpha = CQL_ALPHA_START + t * (CQL_ALPHA_END - CQL_ALPHA_START)

    # --------------------------------------------------
    def update(self, offline: bool = False):
        if len(self.replay) < self.batch_size * START_BATCH_TIMES:
            return
        s, a, r, ns, d = self.replay.sample(self.batch_size)

        # ---- target Q ----
        with torch.no_grad():
            na, nlp = self.pi.sample_action(ns)
            q1n = self.q1_t(ns, na)
            q2n = self.q2_t(ns, na)
            qn = torch.min(q1n, q2n).squeeze(-1)
            tgt = r + self.gamma * (1 - d) * (qn - self.alpha * nlp)

        # ---- Q1 & Q2 ----
        q1v = self.q1(s, a).squeeze(-1)
        q2v = self.q2(s, a).squeeze(-1)
        loss_q1 = F.mse_loss(q1v, tgt)
        if offline or ONLINE_USE_CQL:
            loss_q1 += self.cql_regularizer(s, a)
        loss_q2 = F.mse_loss(q2v, tgt)

        self.q1_opt.zero_grad(); loss_q1.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); loss_q2.backward(); self.q2_opt.step()

        # ---- Policy ----
        na, lp = self.pi.sample_action(s)
        qmix = torch.min(self.q1(s, na), self.q2(s, na)).squeeze(-1)
        loss_pi = (self.alpha * lp - qmix).mean()

        self.pi_opt.zero_grad(); loss_pi.backward(); self.pi_opt.step()

        # ---- Soft update ----
        self.soft(self.q1, self.q1_t); self.soft(self.q2, self.q2_t)

# ─────────────────────────────────────────────────────────────────────────
# 5. 보상 계산 함수
# ─────────────────────────────────────────────────────────────────────────
def compute_reward(env) -> float:
    """환경 객체(env_model)에서 보상을 계산해 반환."""
    r = REWARD_FIXED
    # (필요에 따라) try/except 로 개별 보상 항목이 없는 경우 무시
    try:
        if REWARD_A: r += env.reward_based_alived() * REWARD_A
        if REWARD_B: r += env.reward_based_all_agents_danger() * REWARD_B
        if REWARD_C: r += env.reward_based_gain() * REWARD_C
        if REWARD_D: r += env.reward_penalty() * REWARD_D
        if REWARD_E: r += env.reward_based_evacuated_with_robot() * REWARD_E
        if REWARD_F: r += env.reward_based_distance_from_near_agents() * REWARD_F
        if REWARD_G: r += env.reward_based_distance_from_near_agent_gain() * REWARD_G
        if REWARD_H: r += env.reward_based_gain_with_time_bonus() * REWARD_H
        if REWARD_I: r += env.reward_based_alived_root() * REWARD_I
        if REWARD_J: r += env.reward_based_all_agents_danger_log() * REWARD_J
        if REWARD_K: r += env.reward_penalty_collision() * REWARD_K
        if REWARD_L: r += env.reward_based_heatmap() * REWARD_L
    except AttributeError:
        pass
    return r

# ─────────────────────────────────────────────────────────────────────────
# 6. 오프라인 프리트레이닝 / 온라인 파인튜닝
# ─────────────────────────────────────────────────────────────────────────
def offline_train(agent: SAC_CQL_Agent, tb_dir: str):
    agent.replay.load(OFFLINE_DATASET)
    agent.batch_size = OFFLINE_BATCH_SIZE
    total_steps = OFFLINE_EPOCHS * (len(agent.replay) // OFFLINE_BATCH_SIZE)
    gstep = 0
    for epoch in range(OFFLINE_EPOCHS):
        steps = len(agent.replay) // OFFLINE_BATCH_SIZE
        for _ in range(steps):
            agent.update(offline=True)
            agent.step_alpha(gstep); gstep += 1
            
        if (epoch + 1) % 10 == 0:
            torch.save(agent.pi.state_dict(), os.path.join(LOG_DIR, f"offline_pi_ep{epoch+1}.pth"))
        print(f"[Offline] Epoch {epoch+1}/{OFFLINE_EPOCHS} 완료")

def online_train(agent: SAC_CQL_Agent, tb_dir: str):
    import model  # 시뮬레이터

    # 로그 파일 경로
    rew_txt = os.path.join(LOG_DIR, "total_reward.txt")
    evac_txt = os.path.join(LOG_DIR, "evac_100.txt")
    os.makedirs(LOG_DIR, exist_ok=True)

    # TensorBoard 스트리머
    threading.Thread(target=tail_txt_to_tb, args=(rew_txt, "Reward", tb_dir), daemon=True).start()
    threading.Thread(target=tail_txt_to_tb, args=(evac_txt, "Evac100", tb_dir), daemon=True).start()

    sim_t, learn_t = Timer(), Timer()
    for ep in range(9999999):
        n_people = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
        env = model.FightingModel(n_people, 50, 50, model_num=MAP_NUM, robot='Q')
        state = env.return_current_image()[None, ...]  # (1,50,50)
        ep_ret, evac100 = 0.0, MAX_STEPS

        for step in range(MAX_STEPS):
            # ---------- 행동 선택 ----------
            action_t, _ = agent.pi.sample_action(
                torch.from_numpy(state).float().unsqueeze(0).to(agent.dev)
            )
            action_np = action_t.cpu().numpy()[0]
            real_action = env.robot.receive_action(action_np)

            # ---------- 환경 진전 ----------
            sim_t.start(); env.step(); sim_t.stop()
            next_state = env.return_current_image()[None, ...]

            # ---------- 보상 ----------
            reward = compute_reward(env)
            if env.robot.is_game_finished:
                reward += FINISHED_BONUS * (1 - step / MAX_STEPS)

            done = env.robot.is_game_finished or step == MAX_STEPS-1
            agent.replay.push(state, real_action, reward, next_state, done)
            ep_ret += reward
            state = next_state

            # ---------- 학습 업데이트 ----------
            if (len(agent.replay) >= BATCH_SIZE * START_BATCH_TIMES and
                step % ACTION_SCALE == ACTION_SCALE - 1 and
                ep >= START_UPDATE_STEP):
                learn_t.start(); agent.update(offline=False); learn_t.stop()

            # ---------- 통계 ----------
            if env.alived_agents() < 1 and evac100 == MAX_STEPS:
                evac100 = step
            if done:
                break

        # ---------- 로그 기록 ----------
        with open(rew_txt, "a") as f: f.write(f"{ep_ret}\n")
        with open(evac_txt, "a") as f: f.write(f"{evac100}\n")

        if ENABLE_TIMER:
            print(f"[EP{ep}] Reward:{ep_ret:.1f}  sim:{sim_t.get_time():.2f}s  "
                  f"learn:{learn_t.get_time():.2f}s")
            sim_t.reset(); learn_t.reset()

        # ---------- 주기적 저장 ----------
        if ep % 50 == 0:
            torch.save(agent.pi.state_dict(), os.path.join(LOG_DIR, f"pi_ep{ep}.pth"))
            agent.replay.save(os.path.join(LOG_DIR, "replay_buffer.npz"))

        if ONLINE_USE_CQL:
            agent.step_alpha(ep)

# ─────────────────────────────────────────────────────────────────────────
# 7. 진입점
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    tb_dir = os.path.join(LOG_DIR, "tb"); os.makedirs(tb_dir, exist_ok=True)
    launch_tensorboard(tb_dir, PORT_NUM)

    agent = SAC_CQL_Agent()

    if OFFLINE_MODE:
        print("=== Offline CQL pre-training ===")
        offline_train(agent, tb_dir)
        agent.replay.save(os.path.join(LOG_DIR, "offline_buffer.npz"))
        torch.save(agent.pi.state_dict(), os.path.join(LOG_DIR, "offline_pi.pth"))
        print("=== Offline 완료 ===\n")

    print("=== Online 학습 (파인튜닝) 시작 ===")
    online_train(agent, tb_dir)
