import os
import time
import random
import pickle
import argparse
import threading
import subprocess
import webbrowser
from pathlib import Path
from collections import deque
from typing import Tuple, Union          # ⇦ ① 추가

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from einops import repeat

from timer_utils import Timer
from config import ENABLE_TIMER
from Start_training import *            # noqa
import model

OFFLINE_TRAIN = True

HOME_DIR = Path.home()
LOG_DIR_PATH = HOME_DIR / LOG_DIR
TB_DIR = LOG_DIR_PATH / "tensorboard_logs"
LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_REWARD_TXT = LOG_DIR_PATH / "total_reward.txt"
EVAC80_TXT = LOG_DIR_PATH / "evacuation_80.txt"
EVAC100_TXT = LOG_DIR_PATH / "evacuation_100.txt"

sim_timer = Timer()
learn_timer = Timer()


def launch_tensorboard(port: int = PORT_NUM):  # noqa: F405
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", str(TB_DIR), "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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

    while not os.path.exists(metric_file):
        print(f"Waiting for {metric_file} to be created...")
        time.sleep(2)

    with open(metric_file, "r") as f:
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
                            writer.add_scalar(metric_name, value, episode)
                            episode += 1
                        except ValueError:
                            print(f"Invalid value in {metric_file}: {line}")
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"Monitoring for {metric_file} interrupted by user.")
        finally:
            writer.close()


def alpha_decay_schedule(
    parameter_start: float, parameter_end: float, decay_steps: int, episode_num: int
) -> float:
    if episode_num >= decay_steps:
        return parameter_end
    progress = episode_num / float(decay_steps)
    return parameter_start + (parameter_end - parameter_start) * progress


def gamma_ascent_schedule(
    parameter_start: float, parameter_end: float, decay_steps: int, episode_num: int
) -> float:
    if episode_num >= decay_steps:
        return parameter_end
    if episode_num <= 0:
        return parameter_start
    progress = episode_num / float(decay_steps)
    return parameter_start + (parameter_end - parameter_start) * progress


def evaluate_agent(
    agent: "DiffusionQLAgent",
    episodes: int = EVAL_EPISODES,
    max_steps: int = MAX_STEPS,
    action_scale: int = ACTION_SCALE,
) -> Tuple[float, float, float]:
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
            action = None

            for step in range(max_steps):
                if step % action_scale == 0:
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).float()/255.0
                        state_t = state_t.to(agent.dev)
                        action = agent.policy.sample(state_t).cpu().numpy()[0]
                    env.robot.receive_action(action)

                    r_a = r_b = r_c = r_d = r_e = r_f = r_g = r_h = r_i = r_j = r_k = 0

                    if REWARD_A:
                        r_a = env.reward_based_alived() * REWARD_A
                    if REWARD_B:
                        r_b = env.reward_based_all_agents_danger() * REWARD_B
                    if REWARD_C:
                        r_c = env.reward_based_gain() * REWARD_C
                    if REWARD_D:
                        r_d = env.reward_penalty() * REWARD_D
                    if REWARD_E:
                        r_e = env.reward_based_evacuated_with_robot() * REWARD_E
                    if REWARD_F:
                        r_f = env.reward_based_distance_from_near_agents() * REWARD_F
                    if REWARD_G:
                        r_g = env.reward_based_distance_from_near_agent_gain() * REWARD_G
                    if REWARD_H:
                        r_h = env.reward_based_gain_with_time_bonus() * REWARD_H
                    if REWARD_I:
                        r_i = env.reward_based_alived_root() * REWARD_I
                    if REWARD_J:
                        r_j = env.reward_based_all_agents_danger_log() * REWARD_J

                    total_r += (
                        r_a
                        + r_b
                        + r_c
                        + r_d
                        + r_e
                        + r_g
                        + r_f
                        + r_h
                        + r_i
                        + r_j
                        + r_k
                        + REWARD_FIXED
                    )

                env.step()
                state = env.return_current_image()

                if env.alived_agents() < env.total_agents * 0.2 and t80 == max_steps:
                    t80 = step
                if env.alived_agents() < 1 and t100 == max_steps:
                    t100 = step

                if env.robot.is_game_finished:
                    total_r += FINISHED_BONUS * (1 - step / max_steps)  # noqa: F405
                    break

            rewards.append(total_r)
            t80s.append(t80)
            t100s.append(t100)

        except Exception as e:
            print(f"[Eval] Simulation error (attempt {attempts}): {e} — retrying…")
            time.sleep(0.1)
            continue

    return float(np.mean(rewards)), float(np.mean(t80s)), float(np.mean(t100s))


# --------------------------------------------------------------------- #
#                     DiffusionPolicy  (조건부 DDPM)                     #
# --------------------------------------------------------------------- #
class SinPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,1)
        freqs = torch.arange(self.dim, device=t.device) / self.dim
        emb = torch.einsum("bi,d->bid", t, freqs) * 2 * torch.pi
        return torch.cat([emb.sin(), emb.cos()], dim=-1).squeeze(1)


class EpsModel(nn.Module):
    def __init__(self, a_dim, s_dim, hidden=256):
        super().__init__()
        self.t_emb = SinPosEmb(hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, a_dim),
        )

    def forward(self, a_i, s, t_idx):  # (B,a) (B,s) (B,1)
        emb = self.t_emb(t_idx)
        x = torch.cat([a_i, s, emb], dim=-1)
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_shape=(1, 50, 50), feature_dim=256):
        super().__init__()
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 5, 2, 2), nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.01),
        )

        # ⬇️  flatten 크기를 자동으로 계산
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            conv_out = self.conv(dummy).view(1, -1).size(1)   # 6272
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, feature_dim),
            nn.LeakyReLU(0.01),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class DiffusionPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, N=5, beta_min=0.1, beta_max=10.0):
        super().__init__()
        
        self.N, self.a_dim = N, a_dim
        self.encoder = ImageEncoder((1, 50, 50), s_dim)
        
        self.register_buffer("betas", self._make_beta(N, beta_min, beta_max))
        self.eps_model = EpsModel(a_dim, s_dim)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas = self.alphas.to(self.betas.device)
        self.alphas_bar = self.alphas_bar.to(self.betas.device)

    def bc_loss(self, s_flat, a0):
        feat = self.encoder(s_flat)
        B = feat.size(0)
        i = torch.randint(1, self.N + 1, (B, 1), device=s_flat.device)
        eps = torch.randn(B, self.a_dim, device=s_flat.device)
        alpha_bar = self.alphas_bar.to(s_flat.device)
        alpha_bar_i = alpha_bar[i - 1]
        ai = alpha_bar_i.sqrt() * a0 + (1 - alpha_bar_i).sqrt() * eps
        eps_hat = self.eps_model(ai, feat, i.float())
        return F.mse_loss(eps_hat, eps)

    @torch.no_grad()
    def sample(self, s_flat):

        feat = self.encoder(s_flat)
        B = s_flat.size(0)
        a = torch.randn(B, self.a_dim, device=s_flat.device)
        for i in reversed(range(1, self.N + 1)):
            beta, alpha = self.betas[i - 1], self.alphas[i - 1]
            alpha_bar = self.alphas_bar[i - 1]

            t_idx = torch.full((B, 1), i, device=s_flat.device, dtype=torch.float32)
            eps_hat = self.eps_model(a, feat, t_idx)

            coef1 = 1 / alpha.sqrt()
            coef2 = beta / (1 - alpha_bar).sqrt()
            a = coef1 * (a - coef2 * eps_hat)
            if i > 1:
                a += beta.sqrt() * torch.randn_like(a)
        return torch.tanh(a) * 2

    def forward(self, x):
        return x

    def _make_beta(self, N, bmin, bmax):
        idx = torch.arange(1, N + 1)
        return 1 - torch.exp(-bmin / N - 0.5 * (bmax - bmin) * (2 * idx - 1) / N ** 2)


##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    """메모리·디스크 용량을 최소화한 링 버퍼"""

    def __init__(
        self, capacity: int, state_shape: Tuple[int, int], action_dim: int, device: torch.device
    ):
        self.cap = capacity
        self.ptr = 0
        self.size = 0
        self.dev = device

        H, W = state_shape
        self.states = np.zeros((capacity, H, W), dtype=np.uint8)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float16)
        self.rewards = np.zeros((capacity,), dtype=np.float16)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    # ------------------------------------------------------------------ #
    #                           push / sample                            #
    # ------------------------------------------------------------------ #
    def push(self, state, action, reward, next_state, done, src_tag=None):
        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)

        s = (
            torch.from_numpy(self.states[idx].astype(np.float32) / 255.0)
            .unsqueeze(1)
            .to(self.dev)
        )
        s_next = (
            torch.from_numpy(self.next_states[idx].astype(np.float32) / 255.0)
            .unsqueeze(1)
            .to(self.dev)
        )
        a = torch.from_numpy(self.actions[idx].astype(np.float32)).to(self.dev)
        r = torch.from_numpy(self.rewards[idx].astype(np.float32)).to(self.dev)
        d = torch.from_numpy(self.dones[idx].astype(np.float32)).to(self.dev)
        return s, a, r, s_next, d

    def __len__(self):
        return self.size

    # ------------------------------------------------------------------ #
    #                       디스크 저장 / 로드 (zip)                      #
    # ------------------------------------------------------------------ #
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            states   = self.states[:self.size],
            next_states = self.next_states[:self.size],
            actions  = self.actions[:self.size],
            rewards  = self.rewards[:self.size],
            dones    = self.dones[:self.size],
            size     = np.array([self.size], np.int32),
            capacity = np.array([self.cap],  np.int32),
        )
        print(f"[ReplayBuffer] saved {self.size} transitions → {path}")
        
    def load(self, file: Union[str, Path]):       # ⇦ ② 수정
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(file)
        data = np.load(file, allow_pickle=False)
        n = int(data["size"][0])
        if n > self.cap:
            raise ValueError(f"file contains {n} samples but capacity={self.cap}")

        self.states[:n] = data["states"]
        self.next_states[:n] = data["next_states"]
        self.actions[:n] = data["actions"]
        self.rewards[:n] = data["rewards"]
        self.dones[:n] = data["dones"]

        self.size = n
        self.ptr = n % self.cap
        print(f"[ReplayBuffer] loaded {self.size:,} transitions from {file}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.leaky_relu(out, 0.01)


##########################################################################
# 3) Critic (Q) Network
##########################################################################
class QNetwork(nn.Module):
    def __init__(self, input_shape=(50, 50), action_dim=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        o = F.leaky_relu(self.bn1(self.conv1(dummy)), 0.01)
        o = F.leaky_relu(self.bn2(self.conv2(o)), 0.01)
        o = F.leaky_relu(self.bn3(self.conv3(o)), 0.01)
        return int(np.prod(o.size()[1:]))

    def forward(self, state, action):
        x = F.leaky_relu(self.bn1(self.conv1(state)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        return self.q_out(x)


##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape=(50, 50)):
        super().__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

        self.conv1 = nn.Conv2d(1, 32, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(128)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc_backbone = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01, inplace=True),
        )

        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        x = F.leaky_relu(self.bn1(self.conv1(dummy)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = x.view(x.size(0), -1)
        return int(np.prod(x.size()[1:]))

    def backbone(self, state):
        x = F.leaky_relu(self.bn1(self.conv1(state)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = x.view(x.size(0), -1)
        return self.fc_backbone(x)

    def forward(self, state):
        feat = self.backbone(state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, state, temperature=1.0):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mean) * temperature
        u = mean + std * eps
        sigma = torch.sigmoid(u)
        action = 4 * sigma - 2

        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob_u = log_prob_u.sum(dim=1)
        jacobian = torch.log(4 * sigma * (1 - sigma) + 1e-8).sum(dim=1)
        log_prob = log_prob_u - jacobian
        return action, log_prob


class DiffusionQLAgent:
    def __init__(self, device="cpu", batch_size=BATCH_SIZE):
        self.dev = torch.device(device)
        self.batch = batch_size
        self.buffer = ReplayBuffer(BUFFER_SIZE, (50, 50), 2, DEVICE)
        self.R_BASE = 4.0

        self.q1 = QNetwork().to(self.dev)
        self.q2 = QNetwork().to(self.dev)
        self.q1_t = QNetwork().to(self.dev)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t = QNetwork().to(self.dev)
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.q_optim = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=LR
        )

        FEATURE_DIM = 256
        self.policy = DiffusionPolicy(
            s_dim=FEATURE_DIM, a_dim=2, N=DIFF_STEPS, beta_min=BETA_MINMAX[0], beta_max=BETA_MINMAX[1]
        ).to(self.dev)
        self.p_optim = optim.Adam(self.policy.parameters(), lr=LR)

        self.gamma = GAMMA_START
        self.tau = 0.995
        self.eps = 0.05

    def select_action(self, state_np):
        if np.random.rand() < self.eps:
            return np.random.uniform(-2, 2, size=(2,)), True

        state_t = torch.from_numpy(state_np).unsqueeze(0).unsqueeze(0).float() / 255.0
        state_t = state_t.to(self.dev)
        with torch.no_grad():
            a = self.policy.sample(state_t)
        return a.cpu().numpy()[0], False

    def update(self):
        if len(self.buffer) < self.batch * START_BATCH_TIMES:
            return

        s, a, r, s2, d = self.buffer.sample(self.batch)
        #r = r+ self.R_BASE #R_BASE 더해야 할까?

        r_mean = r.mean().item()

        with torch.no_grad():
            a2 = self.policy.sample(s2)
            qt = torch.min(self.q1_t(s2, a2), self.q2_t(s2, a2)).squeeze(-1)
            qt_mean = qt.mean().item()
            y = r + self.gamma * (1 - d) * qt
        q1_pred = self.q1(s, a).squeeze(-1)
        q2_pred = self.q2(s, a).squeeze(-1)
        q1_loss = F.mse_loss(self.q1(s, a).squeeze(-1), y)
        q2_loss = F.mse_loss(self.q2(s, a).squeeze(-1), y)

        self.q_optim.zero_grad()
        (q1_loss + q2_loss).backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 1.0)
        self.q_optim.step()

        Ld = self.policy.bc_loss(s, a)
        a0 = self.policy.sample(s)
        q_min = torch.min(self.q1(s, a0), self.q2(s, a0)).squeeze(-1)
        Lq = (-q_min).mean()
        L_total = Ld + ETA * Lq

        self.p_optim.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.p_optim.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(self.tau).add_((1 - self.tau) * p.data)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(self.tau).add_((1 - self.tau) * p.data)

        with torch.no_grad():
            return {
                "Ld"       : Ld.item(),
                "Lq"       : Lq.item(),
                "q_loss"   : (q1_loss + q2_loss).item(),
                "r_mean"   : r_mean,
                "qt_mean"  : qt_mean,
                "y_mean"   : y.mean().item(),
                "q1_mean"  : q1_pred.mean().item(),
                "q2_mean"  : q2_pred.mean().item(),
            }


    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def save_replay_buffer(self, f):
        self.buffer.save(LOG_DIR_PATH / f)

    def save_model(self, path):
        torch.save(
            {
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_t": self.q1_t.state_dict(),
                "q2_t": self.q2_t.state_dict(),
                "policy": self.policy.state_dict(),
                "q_opt": self.q_optim.state_dict(),
                "p_opt": self.p_optim.state_dict(),
            },
            path,
        )

    def load_model(self, path):
        ckpt = torch.load(path, map_location=self.dev)
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_t.load_state_dict(ckpt["q1_t"])
        self.q2_t.load_state_dict(ckpt["q2_t"])
        self.policy.load_state_dict(ckpt["policy"], strict=False)
        self.q_optim.load_state_dict(ckpt["q_opt"])
        self.p_optim.load_state_dict(ckpt["p_opt"])


##########################################################################
# TensorBoard 모니터링: total_reward.txt 새 라인을 기록
##########################################################################
def monitor_total_reward(total_reward_file, tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    while not os.path.exists(total_reward_file):
        print(f"Waiting for {total_reward_file} to be created...")
        time.sleep(2)

    with open(total_reward_file, "r") as f:
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
            loss_dict = agent.update()
            if loss_dict:
                tb.add_scalar("Loss/BC", loss_dict["Ld"], step)
                tb.add_scalar("Loss/Q_guidance", loss_dict["Lq"], step)
                tb.add_scalar("Loss/Q_critic", loss_dict["q_loss"], step)
                tb.add_scalar("Debug/y_mean",   loss_dict["y_mean"],  step)
                tb.add_scalar("Debug/q1_mean",  loss_dict["q1_mean"], step)
                tb.add_scalar("Debug/q2_mean",  loss_dict["q2_mean"], step)

                tb.add_scalar("Debug/r_mean",   loss_dict["r_mean"],  step)
                tb.add_scalar("Debug/qt_mean",  loss_dict["qt_mean"], step)

                step += 1

            print(f"[Offline] Episode {ep} - Losses: {loss_dict}")
            if ep % EVAL_INTERVAL == 0 and loss_dict:
                print(f"[Eval] Episode {ep} - Evaluating agent…")
                R, e80, e100 = evaluate_agent(agent)
                tb.add_scalar("Eval/TotalReward", R, ep)
                tb.add_scalar("Eval/Evac80", e80, ep)
                tb.add_scalar("Eval/Evac100", e100, ep)

                ckpt_path = LOG_DIR_PATH / f"dql_eval_ckpt_ep_{ep}.pth"
                agent.save_model(str(ckpt_path))
                print(f"[Checkpoint] Saved model at episode {ep}: {ckpt_path}")

        tb.close()
        return

    # Online roll-out fallback … (생략) …


if __name__ == "__main__":
    main()
