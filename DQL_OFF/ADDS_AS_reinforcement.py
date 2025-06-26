import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import time
from timer_utils import Timer
from config import ENABLE_TIMER
import pickle
import argparse
import threading
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser
from Start_training import *
from einops import repeat
import model
from typing import Tuple
from pathlib import Path

OFFLINE_TRAIN = True
EVAL_INTERVAL = 200
EVAL_EPISODES = 10
MAX_OFFLINE_EP = 1_000_000



HOME_DIR = Path.home()
LOG_DIR_PATH = HOME_DIR / LOG_DIR
TB_DIR = LOG_DIR_PATH / "tensorboard_logs"
LOG_DIR_PATH.mkdir(parents = True, exist_ok = True)
TB_DIR.mkdir(parents = True, exist_ok = True)

TOTAL_REWARD_TXT = LOG_DIR_PATH / "total_reward.txt"
EVAC80_TXT = LOG_DIR_PATH / "evacuation_80.txt"
EVAC100_TXT = LOG_DIR_PATH / "evacuation_100.txt"

sim_timer = Timer()
learn_timer = Timer()

def launch_tensorboard(port: int = PORT_NUM):  # noqa: F405
    proc = subprocess.Popen([
        "tensorboard", "--logdir", str(TB_DIR), "--port", str(port)
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(5)
    webbrowser.open(f"http://localhost:{port}")
    print(f"TensorBoard ▶ http://localhost:{port}")
    return proc

def load_offline_buffer(agent: "DiffusionQLAgent", fname: str = "offline_buffer.pkl") -> bool:
    path = LOG_DIR_PATH / fname
    if path.exists():
        print(f"Loading offline buffer: {path}")
        agent.buffer.load(str(path))
        return True
    print(f"Offline buffer not found: {path}")
    return False

def monitor_metric(metric_file, metric_name, tb_log_dir):
    """
    metric_file에서 새로운 라인이 추가될 때마다 읽어서,
    metric_name으로 TensorBoard에 기록합니다.
    """
    writer = SummaryWriter(log_dir=tb_log_dir)

    # metric_file이 생성될 때까지 대기
    while not os.path.exists(metric_file):
        print(f"Waiting for {metric_file} to be created...")
        time.sleep(2)

    with open(metric_file, "r") as f:
        # 파일 끝으로 이동 (기존 데이터 무시하고 새로 들어오는 라인부터 읽고 싶다면 아래 주석 제거)
        # f.seek(0, os.SEEK_END)

        episode = 0
        print(f"Start monitoring {metric_file} for new data...")
        try:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            value = float(line)
                            writer.add_scalar(f"{metric_name}", value, episode)
                            #print(f"Episode {episode} - {metric_name} = {value}")
                            episode += 1
                        except ValueError:
                            print(f"Invalid value in {metric_file}: {line}")
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"Monitoring for {metric_file} interrupted by user.")
        finally:
            writer.close()

def alpha_decay_schedule(parameter_start: float,
                   parameter_end: float,
                   decay_steps: int,
                   episode_num: int) -> float:

    # 감쇠가 끝났으면 최종값 고정
    if episode_num >= decay_steps:
        return parameter_end

    # 선형 보간
    progress = episode_num / float(decay_steps)
    return parameter_start + (parameter_end - parameter_start) * progress


def gamma_ascent_schedule(parameter_start: float,
                          parameter_end : float,
                          decay_steps : int,
                          episode_num : int) -> float:
    if episode_num >= decay_steps:
        return parameter_end

    # 음수 에피소드 처리: 초기값 반환
    if episode_num <= 0:
        return parameter_start

    # 선형 보간으로 값 계산
    progress = episode_num / float(decay_steps)
    return parameter_start + (parameter_end - parameter_start) * progress



def evaluate_agent(
    agent: "DiffusionQLAgent",
    episodes: int = EVAL_EPISODES,
    max_steps: int = MAX_STEPS,
    action_scale: int = ACTION_SCALE,   # ⇦ 학습과 동일한 간격으로 행동 수행
) -> Tuple[float, float, float]:
    """Evaluate *episodes* random maps.

    • **action_scale**:  환경이 `state` 4‑step 마다 1회 행동을 처리하므로
      `ACTION_SCALE` 간격으로만 `agent.policy.sample` & `receive_action` 수행.
    • 시뮬 오류 발생 시 재시도하여 *episodes* 회 성공할 때까지 반복.
    """
    rewards, t80s, t100s = [], [], []
    attempts = 0
    while len(rewards) < episodes:
        attempts += 1
        try:
            n_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
            env = model.FightingModel(n_agents, 50, 50, model_num=MAP_NUM, robot="Q")
            state = env.return_current_image()
            t80 = t100 = max_steps
            total_r = 0.0

            # 초기 행동
            action = None

            for step in range(max_steps):
                # ACTION_SCALE 간격으로만 새 행동 샘플링 및 적용
                if step % action_scale == 0:
                    with torch.no_grad():
                        action = agent.policy.sample(
                            torch.from_numpy(state).view(1, -1).float().to(agent.dev)
                        ).cpu().numpy()[0]
                    env.robot.receive_action(action)

                env.step()
                state = env.return_current_image()

                # 80% / 100% 대피시간 기록
                if env.alived_agents() < env.total_agents * 0.2 and t80 == max_steps:
                    t80 = step
                if env.alived_agents() < 1 and t100 == max_steps:
                    t100 = step

                if env.robot.is_game_finished:
                    total_r += FINISHED_BONUS * (1 - step / max_steps)  # noqa: F405
                    break

            # record episode
            rewards.append(total_r)
            t80s.append(t80)
            t100s.append(t100)

        except Exception as e:
            print(f"[Eval] Simulation error (attempt {attempts}): {e} — retrying…")
            time.sleep(0.1)
            continue

    return (
        float(np.mean(rewards)),
        float(np.mean(t80s)),
        float(np.mean(t100s)),
    )



# --------------------------------------------------------------------- #
#                     DiffusionPolicy  (조건부 DDPM)                     #
# --------------------------------------------------------------------- #
class SinPosEmb(nn.Module): #DDPM에서 timestep을 임베딩하는 역할, 확산 과정에서 "현재 몇 번째 타임스텝인지"라는 정수형 시간 정보를 벡터 표현으로 변환
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):                 # t: (B,1)
        freqs = torch.arange(self.dim, device=t.device) / self.dim
        emb = torch.einsum('bi,d->bid', t, freqs) * 2 * torch.pi
        return torch.cat([emb.sin(), emb.cos()], dim=-1).squeeze(1)

class EpsModel(nn.Module): # DDPM에서 εθ(a_i, s, t_idx)를 예측하는 네트워크, 주어진 노이즈가 섞인 행동을 보고 노이즈를 예측
    def __init__(self, a_dim, s_dim, hidden=256):
        super().__init__()
        self.t_emb = SinPosEmb(hidden//2)
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + hidden, hidden),
            nn.Mish(), 
            nn.Linear(hidden, hidden), 
            nn.Mish(),
            nn.Linear(hidden, a_dim)
        )
    def forward(self, a_i, s, t_idx):     # (B,a) (B,s) (B,1)
        emb = self.t_emb(t_idx)
        x = torch.cat([a_i, s, emb], dim=-1)
        return self.net(x)

class DiffusionPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, N=5, beta_min=.1, beta_max=10.):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        self.register_buffer("betas", self._make_beta(N, beta_min, beta_max))
        self.eps_model = EpsModel(a_dim, s_dim)
        self.alphas   = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas = self.alphas.to(self.betas.device)
        self.alphas_bar = self.alphas_bar.to(self.betas.device)

                
        #self.betas -> 각 timesteps에서 추가되는 노이즈의 분산
        #self.alphas -> 각 timesteps에서 노이즈가 제거되는 비율
        #self.alphas_bar -> 누적된 노이즈 제거 비율

    # ------------ BC 손실 (Alg.1 ③, 식 2) ------------
    def bc_loss(self, s_flat, a0): # s_flat은 state, a0은 전문가가 취한 행동, 노이즈를 얼마나 잘 제거할 수 있냐에 집중
        B = s_flat.size(0)
        i = torch.randint(1, self.N+1, (B,1), device=s_flat.device)   # (B,1)
        eps = torch.randn(B, self.a_dim, device=s_flat.device)
        alpha_bar = self.alphas_bar.to(s_flat.device)
        alpha_bar_i = alpha_bar[i-1]            # (B,1)
        ai = (alpha_bar_i.sqrt() * a0 +
              (1-alpha_bar_i).sqrt() * eps) # 행동 a_0에 노이즈를 넣어 a_i 생성하는 식
        eps_hat = self.eps_model(ai, s_flat, i.float()) # 조건 s와 a_i를 보고 실제로 들어간 노이즈를 예측

        return F.mse_loss(eps_hat, eps)

    # ------------ 샘플링 (Alg.1 ①②) ------------
    @torch.no_grad()
    def sample(self, s_flat):
        B = s_flat.size(0)
        a = torch.randn(B, self.a_dim, device=s_flat.device)
        for i in reversed(range(1, self.N+1)):
            beta, alpha = self.betas[i-1], self.alphas[i-1]
            alpha_bar = self.alphas_bar[i-1]
            eps_hat = self.eps_model(a, s_flat,
                                     torch.full((B,1), i, device=s_flat.device))
            coef1 = 1/alpha.sqrt()
            coef2 = beta/((1-alpha_bar).sqrt())
            a = coef1 * (a - coef2 * eps_hat) # sampling, 위 식을 반복하여 노이즈 상태를 점점 원래 행동으로 되돌림
            if i > 1:
                a += beta.sqrt() * torch.randn_like(a)
        return torch.tanh(a)*2          # [-2,2]

    # ------------ forward dummy (안쓰지만 torch.compile 호환) ------------
    def forward(self, x): return x

    def _make_beta(self, N, bmin, bmax):
        idx = torch.arange(1, N+1)
        return 1 - torch.exp(-bmin/N - 0.5*(bmax-bmin)*(2*idx-1)/N**2)
    
    

##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    """메모리·디스크 용량을 최소화한 링 버퍼"""
    def __init__(self,
                 capacity     : int,
                 state_shape  : Tuple[int, int],   # (H,W)
                 action_dim   : int,
                 device       : torch.device):
        self.cap   = capacity
        self.ptr   = 0          # 다음에 쓸 위치
        self.size  = 0
        self.dev   = device

        H, W = state_shape
        self.states      = np.zeros((capacity, H, W), dtype=np.uint8)
        self.next_states = np.zeros_like(self.states)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float16)
        self.rewards     = np.zeros((capacity,),            dtype=np.float16)
        self.dones       = np.zeros((capacity,),            dtype=np.bool_)

    # ------------------------------------------------------------------ #
    #                           push / sample                            #
    # ------------------------------------------------------------------ #
    def push(self, state, action, reward, next_state, done):
        self.states     [self.ptr] = state            # uint8
        self.next_states[self.ptr] = next_state
        self.actions    [self.ptr] = action           # float32 → float16 OK
        self.rewards    [self.ptr] = reward
        self.dones      [self.ptr] = done

        self.ptr  = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        s      = torch.from_numpy(self.states[idx].astype(np.float32) / 255.0) \
                       .unsqueeze(1).to(self.dev)          # (B,1,H,W)
        s_next = torch.from_numpy(self.next_states[idx].astype(np.float32) / 255.0) \
                       .unsqueeze(1).to(self.dev)
        a      = torch.from_numpy(self.actions[idx].astype(np.float32)).to(self.dev)
        r      = torch.from_numpy(self.rewards[idx].astype(np.float32)).to(self.dev)
        d      = torch.from_numpy(self.dones[idx].astype(np.float32)).to(self.dev)
        return s, a, r, s_next, d

    def __len__(self):
        return self.size

    # ------------------------------------------------------------------ #
    #                       디스크 저장 / 로드 (zip)                      #
    # ------------------------------------------------------------------ #
    def save(self, file: str | Path):
        file = Path(file)
        np.savez_compressed(
            file,
            states      = self.states[:self.size],
            next_states = self.next_states[:self.size],
            actions     = self.actions[:self.size],
            rewards     = self.rewards[:self.size],
            dones       = self.dones[:self.size],
            size        = np.array([self.size], dtype=np.int32)
        )
        print(f"[ReplayBuffer] saved {self.size:,} transitions → {file}")

    def load(self, file: str | Path):
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(file)
        data = np.load(file, allow_pickle=False)
        n    = int(data["size"][0])
        if n > self.cap:
            raise ValueError(f"file contains {n} samples but capacity={self.cap}")

        # 기존 메모리 재사용
        self.states[:n]       = data["states"]
        self.next_states[:n]  = data["next_states"]
        self.actions[:n]      = data["actions"]
        self.rewards[:n]      = data["rewards"]
        self.dones[:n]        = data["dones"]

        self.size = n
        self.ptr  = n % self.cap
        print(f"[ReplayBuffer] loaded {self.size:,} transitions from {file}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # stride=1로만 운영 (downsample하지 않음)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # in/out 채널 다르면 skip 연결에 1×1 Conv
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.leaky_relu(out, negative_slope=0.01)
        
    
##########################################################################
# 3) Critic (Q) Network
##########################################################################
class QNetwork(nn.Module):
    def __init__(self, input_shape=(50,50), action_dim=2):
        super(QNetwork, self).__init__()
        # CNN feature extractor
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        # conv_out_size 계산 (flatten 전 feature 크기)
        conv_out_size = self._get_conv_out(input_shape)
        
        # FC layers
        self.fc1 = nn.Linear(conv_out_size + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)  # (batch, channel=1, H, W)
        o = F.leaky_relu(self.bn1(self.conv1(dummy)), negative_slope=0.01)
        o = F.leaky_relu(self.bn2(self.conv2(o)), negative_slope=0.01)
        o = F.leaky_relu(self.bn3(self.conv3(o)), negative_slope=0.01)
        return int(np.prod(o.size()[1:]))

    def forward(self, state, action):
        x = F.leaky_relu(self.bn1(self.conv1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        # 행동 정보를 이미지 feature와 concat
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        q_val = self.q_out(x)
        return q_val


##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape=(50,50)):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

        # --- 여기서 conv1, bn1 선언 ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        # fc_backbone
        conv_out_size = self._get_conv_out(input_shape)
        self.fc_backbone = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01, inplace=True)
        )
        
        # 최종 출력: (dx, dy)의 mean, log_std
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        x = F.leaky_relu(self.bn1(self.conv1(dummy)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        return int(np.prod(x.size()[1:]))

    def backbone(self, state):
        x = F.leaky_relu(self.bn1(self.conv1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        feat = self.fc_backbone(x)
        return feat

    def forward(self, state):
        feat = self.backbone(state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, state, temperature=1.0):
        B = state.size(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mean) * temperature
        u = mean + std * eps
        sigma = torch.sigmoid(u)
        action = 4 * sigma - 2  # [-2,2] 범위 매핑
        
        # 가우시안 로그확률
        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        log_prob_u = log_prob_u.sum(dim=1)
        
        # 시그모이드 야코비안 보정
        jacobian = torch.log(4*sigma*(1 - sigma) + 1e-8).sum(dim=1)
        log_prob = log_prob_u - jacobian
        return action, log_prob


class DiffusionQLAgent:
    def __init__(self, device="cpu", batch_size=BATCH_SIZE):
        self.dev = torch.device(device)
        self.batch = batch_size
        self.buffer = ReplayBuffer(BUFFER_SIZE, (50, 50), 2, DEVICE)

        self.q1 = QNetwork().to(self.dev)
        self.q2 = QNetwork().to(self.dev)
        self.q1_t = QNetwork().to(self.dev); 
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t = QNetwork().to(self.dev); 
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.q_optim = optim.Adam(
            list(self.q1.parameters())+list(self.q2.parameters()), lr=LR)

        self.policy = DiffusionPolicy(
            s_dim=50*50, a_dim=2, N=DIFF_STEPS,
            beta_min=BETA_MINMAX[0], beta_max=BETA_MINMAX[1]).to(self.dev)
        self.p_optim = optim.Adam(self.policy.parameters(), lr=LR)

        self.gamma = GAMMA_START
        self.tau   = .995
        self.eps   = 0.05            # ε-greedy (아주 작게)

    # -------------- 행동 선택 --------------
    def select_action(self, state_np):
        if np.random.rand() < self.eps:
            return np.random.uniform(-2,2, size=(2,)) , True
        s_flat = torch.from_numpy(state_np).view(1,-1).float().to(self.dev)
        with torch.no_grad():
            a = self.policy.sample(s_flat)
        return a.cpu().numpy()[0] , False

    # -------------- 학습 스텝 --------------
    def update(self):
        if len(self.buffer) < self.batch*START_BATCH_TIMES: 
            return
        s,a,r,s2,d = self.buffer.sample(self.batch)
        B = s.size(0)
        s_f  = s.view(B,-1); s2_f = s2.view(B,-1)

        # --- ① Q 손실 & 업데이트 (Alg.1 ④⑤) ---
        with torch.no_grad():
            a2 = self.policy.sample(s2_f)
            qt = torch.min(self.q1_t(s2,a2), self.q2_t(s2,a2)).squeeze(-1)
            y  = r + self.gamma*(1-d)*qt
        q1_loss = F.mse_loss(self.q1(s,a).squeeze(-1), y)
        q2_loss = F.mse_loss(self.q2(s,a).squeeze(-1), y)

        self.q_optim.zero_grad()
        (q1_loss+q2_loss).backward()
        torch.nn.utils.clip_grad_norm_( list(self.q1.parameters())+list(self.q2.parameters()), 1.0)
        self.q_optim.step()

        # --- ② 정책 손실 (Ld + η·Lq) ---
        Ld = self.policy.bc_loss(s_f, a)                 # 식 2
        a0 = self.policy.sample(s_f)
        q_min = torch.min(self.q1(s,a0), self.q2(s,a0)).squeeze(-1)
        Lq = (-q_min).mean()
        L_total = Ld + ETA * Lq

        self.p_optim.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
        self.p_optim.step()

        # --- ③ 타깃 네트워크 소프트 업데이트 ---
        with torch.no_grad():
            for p,pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(self.tau).add_((1-self.tau)*p.data)
            for p,pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(self.tau).add_((1-self.tau)*p.data)
        
        q_loss = (q1_loss + q2_loss).item()

        losses = {
            "Ld" : Ld.item(),
            "Lq" : Lq.item(),
            "q_loss" : q_loss
        }

        return losses

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def save_replay_buffer(self, f):
        path = LOG_DIR_PATH / f
        self.buffer.save(str(path))
    # def load_replay_buffer(self, f):
    #     self.buffer.load(os.path.join(LOG_DIR, f))

    # -------------- 버퍼 저장·로드 --------------
    def save_model(self, path):
        torch.save({
            "q1":self.q1.state_dict(),"q2":self.q2.state_dict(),
            "q1_t":self.q1_t.state_dict(),"q2_t":self.q2_t.state_dict(),
            "policy":self.policy.state_dict(),
            "q_opt":self.q_optim.state_dict(),"p_opt":self.p_optim.state_dict()
        }, path)
    def load_model(self,path):
        ckpt = torch.load(path,map_location=self.dev)
        self.q1.load_state_dict(ckpt["q1"]); self.q2.load_state_dict(ckpt["q2"])
        self.q1_t.load_state_dict(ckpt["q1_t"]); self.q2_t.load_state_dict(ckpt["q2_t"])
        self.policy.load_state_dict(ckpt["policy"], strict=False)
        self.q_optim.load_state_dict(ckpt["q_opt"]); self.p_optim.load_state_dict(ckpt["p_opt"])


##########################################################################
# TensorBoard 모니터링 함수: total_reward.txt 파일의 새 라인을 지속적으로 읽어 기록
##########################################################################
def monitor_total_reward(total_reward_file, tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    # 파일 생성 대기
    while not os.path.exists(total_reward_file):
        print(f"Waiting for {total_reward_file} to be created...")
        time.sleep(2)
    with open(total_reward_file, "r") as f:
        # 기존 내용 무시를 위해 파일 끝으로 이동
        #f.seek(0, os.SEEK_END)
        episode = 0
        print("Start monitoring total_reward.txt for new rewards...")
        try:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            total_reward = float(line)
                            writer.add_scalar("Total Reward", total_reward, episode)
                            print(f"Episode {episode}: Total Reward = {total_reward}")
                            episode += 1
                        except ValueError:
                            print(f"Invalid value in total_reward.txt: {line}")
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring interrupted by user.")
        finally:
            writer.close()

def main():
    tb = SummaryWriter(TB_DIR.as_posix())
    launch_tensorboard()

    agent = DiffusionQLAgent(device=DEVICE, batch_size=BATCH_SIZE)  # noqa: F405
    offline = load_offline_buffer(agent, OFFLINE_BUFFER_NAME)

    if offline:
        print("▶ Offline training mode")
        step = 0
        for ep in range(1, MAX_OFFLINE_EP + 1):
            print(f"step : {ep}")
            loss_dict = agent.update()
            if loss_dict:
                tb.add_scalar("Loss/BC",        loss_dict["Ld"],     step)
                tb.add_scalar("Loss/Q_guidance", loss_dict["Lq"],     step)
                tb.add_scalar("Loss/Q_critic",   loss_dict["q_loss"], step)
                step += 1
            if ep % EVAL_INTERVAL == 0 and loss_dict:
                R, e80, e100 = evaluate_agent(agent)
                tb.add_scalar("Eval/TotalReward", R,   ep)
                tb.add_scalar("Eval/Evac80",      e80, ep)
                tb.add_scalar("Eval/Evac100",     e100, ep)

                ckpt_path = LOG_DIR_PATH / f"dql_eval_ckpt_ep_{ep}.pth"
                agent.save_model(str(ckpt_path))
                print(f"[Checkpoint] Saved evaluation model at episode {ep}: {ckpt_path}")

        tb.close(); return

    # Online roll‑out fallback … [unchanged] …

if __name__ == "__main__":
    main()
