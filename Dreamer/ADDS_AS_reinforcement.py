from __future__ import annotations

import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from Start_training import *

from torch.utils.tensorboard import SummaryWriter
import threading, time, subprocess, webbrowser

print(f"[CONFIG] BATCH_SIZE={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}, IMAG={IMAG_HORIZON}")

HOME_DIR  = os.path.expanduser("~")
LOG_PATH  = os.path.join(HOME_DIR, LOG_DIR)
TB_DIR    = os.path.join(LOG_PATH, "tensorboard_logs")
os.makedirs(TB_DIR,  exist_ok=True)

TOTAL_REWARD_TXT         = os.path.join(LOG_PATH, "total_reward.txt")
EVAC80_TXT, EVAC100_TXT  = (os.path.join(LOG_PATH, f"evacuation_{p}.txt")
                            for p in (80, 100))


# === Utility functions =====================================================


def symlog(x: torch.Tensor) -> torch.Tensor: #Dreamer V3에 도입됨
    """Symmetric log transform (Dreamer V3)."""
    return torch.sign(x) * torch.log1p(torch.abs(x)) / math.log(SYML_LOG_BASE)


def symexp(x: torch.Tensor) -> torch.Tensor: #Dreamer V3에 도입됨, symlog의 역함수
    """Inverse of symlog (symexp)."""
    return torch.sign(x) * (SYML_LOG_BASE ** torch.abs(x) - 1.0)


def static_scan(fn, inputs, start):
    """Per‑timestep functional scan (equivalent to tf.scan).
    Returns list of outputs with same len as inputs."""
    outputs = []
    carry = start
    for inp in inputs:
        carry = fn(carry, inp)
        outputs.append(carry)
    return outputs

def launch_tensorboard(tb_log_dir, port=6006):
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(5)
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"TensorBoard launched at {url}")
    return proc

WRITER = SummaryWriter(log_dir=TB_DIR)

def monitor_metric(metric_file, metric_name, tb_log_dir):
    writer = SummaryWriter(log_dir=tb_log_dir)
    while not os.path.exists(metric_file):
        time.sleep(2)
    with open(metric_file, "r") as f:
        episode = 0
        while True:
            line = f.readline()
            if line:
                try:
                    value = float(line.strip())
                    writer.add_scalar(metric_name, value, episode)
                    episode += 1
                except ValueError:
                    pass
            else:
                time.sleep(1)

# === Replay buffer (sequence sampler) ======================================

class ReplayBuffer:
    """Stores episodes as lists of dicts, samples (B,T) chunks."""
    def __init__(self, capacity: int = REPLAY_CAPACITY):
        self.capacity = capacity
        self._episodes: deque[List[dict]] = deque(maxlen=capacity)
        self._length = 0  # total number of transitions

    def add_episode(self, episode: List[dict]):
        self._episodes.append(episode)
        self._length += len(episode)

    def sample(self, batch_size: int = BATCH_SIZE, seq_len: int = SEQ_LEN):
        episodes  = random.choices(self._episodes, k=batch_size)
        sequences = []

        for ep in episodes:
            # ───── 랜덤 시퀀스 잘라내기 ─────
            start = 0 if len(ep) < seq_len + 1 else random.randint(0, len(ep) - seq_len - 1)
            seq   = ep[start : start + seq_len + 1]          # 길이 = T+1

            # ───── obs(T+1) vs. 그 외(T) 분리 ─────
            seq_dict = {}
            for k in seq[0]:
                arr = np.stack([t[k] for t in seq])
                seq_dict[k] = arr if k == "obs" else arr[:-1]   # obs: T+1, 나머지: T
            sequences.append(seq_dict)

        # ───── 배치 결합 ─────
        batch_np = {k: np.stack([s[k] for s in sequences]) for k in sequences[0]}

        # ───── dtype & device 통일 ─────
        batch = {}
        for k, v in batch_np.items():
            if k == "obs":          # 픽셀(0‑255) → uint8 유지
                batch[k] = torch.tensor(v, dtype=torch.uint8,  device=DEVICE)
            else:                   # act, rew, done → float32
                batch[k] = torch.tensor(v, dtype=torch.float32, device=DEVICE)

        return batch
    def __len__(self):
        return self._length

# === Encoder / Decoder =====================================================

class ConvEncoder(nn.Module):

    def __init__(self, obs_shape: Tuple[int, int, int] = (1, 50, 50), depth: int = 16):
        super().__init__()
        c, h, w = obs_shape
        self.depth = depth
        self.net = nn.Sequential(
            nn.Conv2d(c,     depth,     3, stride=2, padding=1), nn.ReLU(),  # 50 → 25
            nn.Conv2d(depth, depth*2,   3, stride=2, padding=1), nn.ReLU(),  # 25 → 13
            nn.Conv2d(depth*2, depth*4, 3, stride=2, padding=1), nn.ReLU(),  # 13 → 7
            nn.Conv2d(depth*4, depth*4, 3, stride=2, padding=1), nn.ReLU(),  # 7  → 4
            nn.Flatten(),                                                   # 4×4×(depth*4)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            self.out_dim = self.net(dummy).shape[-1]  # = depth*4*4*4

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.float() / 255.0
        return self.net(obs)


class ConvDecoder(nn.Module):

    def __init__(self, in_dim: int, obs_shape: Tuple[int, int, int] = (1, 50, 50), depth: int = 16):
        super().__init__()
        c, h, w = obs_shape
        self.fc  = nn.Linear(in_dim, depth*4 * 4 * 4)          # 4×4×(depth*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(depth*4, depth*4, 4, 2, 1), nn.ReLU(),  # 4 → 8
            nn.ConvTranspose2d(depth*4, depth*2, 4, 2, 1), nn.ReLU(),  # 8 → 16
            nn.ConvTranspose2d(depth*2, depth,   4, 2, 1), nn.ReLU(),  # 16 → 32
            nn.ConvTranspose2d(depth,   c,       4, 2, 1),             # 32 → 64
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.fc.out_features // 16, 4, 4)  # (B, depth*4, 4, 4)
        x = self.net(x)
        return x[..., :50, :50]   

# === RSSM ==================================================================

class RSSM(nn.Module):
    """Recurrent State‑Space Model as in Dreamer V3.
    s_t = (h_t, z_t) where h_t : deterministic GRU hidden, z_t : stochastic."""

    def __init__(self, deter=LATENT_DIM, stoch=STOCH_DIM, act_dim=ACTION_DIM, embed_dim: int = None):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.act_dim = act_dim
        # Prior & posterior
        self.prior_fc = nn.Linear(deter, 2 * stoch)  # mean, stdlog
        self.post_fc = nn.Linear(deter + embed_dim, 2 * stoch)
        # Recurrent cell (GRU)
        self.gru = nn.GRUCell(stoch + act_dim, deter)

    def _dist(self, stats):
        mean, std_log = torch.chunk(stats, 2, -1)
        std = F.softplus(std_log) + 0.1
        return Normal(mean, std)

    def init_state(self, batch_size):
        h = torch.zeros(batch_size, self.deter, device=DEVICE)
        z = torch.zeros(batch_size, self.stoch, device=DEVICE)
        return h, z

    def obs_step(self, prev_state, action, embed):
        # Deterministic transition
        h_prev, z_prev = prev_state
        x = torch.cat([z_prev, action], -1)
        h = self.gru(x, h_prev)
        # Prior
        stats_prior = self._dist(self.prior_fc(h))
        z_prior = stats_prior.rsample()
        # Posterior (conditioning on encoder embed)
        stats_post = self._dist(self.post_fc(torch.cat([h, embed], -1)))
        z_post = stats_post.rsample()
        return (h, z_post), (stats_prior, stats_post)

    def img_step(self, prev_state, action):
        h_prev, z_prev = prev_state
        x = torch.cat([z_prev, action], -1)
        h = self.gru(x, h_prev)
        stats_prior = self._dist(self.prior_fc(h))
        z = stats_prior.rsample()
        return (h, z), stats_prior

    def get_feat(self, state):
        h, z = state
        return torch.cat([h, z], -1)

# === Actor & Critic ========================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(400, 400), act=nn.ELU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.extend([nn.Linear(last, h), act()])
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, feat_dim, act_dim=ACTION_DIM):
        super().__init__()
        self.mlp = MLP(feat_dim, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, feat):
        mean, std_log = torch.chunk(self.mlp(feat), 2, -1)
        std = torch.sigmoid(std_log + 2.0) + 0.01  # per Dreamer V3
        dist = Normal(mean, std)
        return dist


class Critic(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.v_head = MLP(feat_dim, 1)

    def forward(self, feat):
        return self.v_head(feat).squeeze(-1)

# === Dreamer Agent =========================================================

@dataclass
class ObsBatch:
    obs: torch.Tensor  # (B,T+1,C,H,W)
    act: torch.Tensor  # (B,T,A)
    rew: torch.Tensor  # (B,T)
    done: torch.Tensor  # (B,T)

class DreamerAgent:
    def __init__(self):
        # Modules
        self.encoder = ConvEncoder().to(DEVICE)
        embed_dim_t = self.encoder.out_dim
        self.rssm = RSSM(deter = LATENT_DIM, stoch=STOCH_DIM, act_dim = ACTION_DIM,embed_dim = embed_dim_t).to(DEVICE)
        feat_dim = LATENT_DIM + STOCH_DIM
        self.decoder = ConvDecoder(in_dim = feat_dim).to(DEVICE)
        
        feat_dim = LATENT_DIM + STOCH_DIM
        self.reward_head = MLP(feat_dim, 1).to(DEVICE)
        self.discount_head = MLP(feat_dim, 1).to(DEVICE)
        self.actor = Actor(feat_dim).to(DEVICE)
        self.critic = Critic(feat_dim).to(DEVICE)
        self.critic_target = Critic(feat_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optims
        self.opt_world = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.discount_head.parameters()), lr=LR_WORLD)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Replay
        self.buffer = ReplayBuffer()
        self._steps = 0

    # ---------------------------------------------------------------------
    # World‑model & actor‑critic update
    # ---------------------------------------------------------------------

    def update(self):
        if len(self.buffer) < BATCH_SIZE * SEQ_LEN:
            return  # not enough data yet
        batch = self.buffer.sample()
        loss_wm, states = self._update_world_model(batch)
        loss_ac = self._update_actor_critic(states)
        return loss_wm, loss_ac

    def _update_world_model(self, batch):
        '''One gradient step on world‑model (encoder, rssm, decoder, reward).'''
        obs = batch['obs']  # (B,T+1,C,H,W)
        act = batch['act']  # (B,T,A)
        rew = batch['rew']  # (B,T)
        done = batch['done']

        # batch['obs'] shape: (B, T, C, H, W)
        B, Tp1, C, H, W = obs.size()
        T = Tp1 - 1  # T is the sequence length (B,T,C,H,W)
        # 1) time 축을 batch로 합치기 → (B*T, C, H, W)
        obs_flat = obs[:, :-1].reshape(B * T, C, H, W)

        # 2) conv2d 등을 포함한 encoder 통과
        embed_flat = self.encoder(obs_flat)  # (B*T, D)

        # 3) 다시 시퀀스 차원 복원 → (B, T, D)
        D = embed_flat.size(-1)
        embed = embed_flat.view(B, T, D)
        # RSSM rollout over sequence
        def rssm_step(prev, inputs):
            act_t, embed_t = inputs
            return self.rssm.obs_step(prev, act_t, embed_t)[0]  # return state

        first_state = self.rssm.init_state(B)
        states,kls = [], []
        state = first_state
        
        for t in range(T):
            state, (prior, post) = self.rssm.obs_step(
                state,
                act[:, t],
                embed[:, t],
            )  # (B, D)
            states.append(state)
            kls.append(torch.distributions.kl_divergence(post, prior))


        feats = torch.stack([self.rssm.get_feat(s) for s in states], dim=1)  # (B,T,F)
        
        # Reconstruction & reward prediction
        img_pred = self.decoder(feats.reshape(-1, feats.shape[-1]))
        img_target = obs[:, 1:].reshape_as(img_pred).float()
        rew_pred = self.reward_head(feats).squeeze(-1)
        # symlog transform rewards for stability
        rew_targets = symlog(rew) #보상 타깃을 symlog로 변환
        loss_img = F.mse_loss(img_pred, img_target)
        loss_rew = F.mse_loss(rew_pred, rew_targets)

        # KL loss between prior/posterior (free nats)

        kl_loss = torch.mean(torch.stack(kls, 1))
        kl_loss = torch.max(kl_loss, torch.tensor(FREE_NATS, device=DEVICE))

        loss_wm = loss_img + loss_rew + kl_loss
        self.opt_world.zero_grad()
        loss_wm.backward()
        nn.utils.clip_grad_norm_(self.opt_world.param_groups[0]['params'], GRAD_CLIP)
        self.opt_world.step()

        global_step = self._steps
        WRITER.add_scalar("world_model/img_loss", loss_img.item(), global_step)
        WRITER.add_scalar("world_model/rew_loss", loss_rew.item(), global_step)
        WRITER.add_scalar("world_model/kl_loss", kl_loss.item(), global_step)
        WRITER.add_scalar("world_model/total", loss_wm.item(), global_step)
        
        states = [(s[0].detach(), s[1].detach()) for s in states]
        return loss_wm.item(), states

    def _imagine(self, start_state, horizon=IMAG_HORIZON):
        '''Deterministic rollout in latent space.'''
        
        states = []
        raws = []
        feats = []
        state = start_state
        for _ in range(horizon):
            feat = self.rssm.get_feat(state)
            dist = self.actor(feat.detach())
            u = dist.rsample()
            action = torch.tanh(u)  # (B,A)
            state, _ = self.rssm.img_step(state, action)
            feat = self.rssm.get_feat(state)
            states.append((feat, action))
            raws.append(u)
        feats = torch.stack([f for f, _ in states], 1)  # (B,H,F)
        actions = torch.stack([a for _, a in states], 1)
        raws = torch.stack(raws, 1)
        return feats, raws, actions

    def _update_actor_critic(self, states):
        '''Imagined trajectories → Train actor & critic.'''
        start_states = states[-1]  # use last latent of sequence as root

        feats, u_raw, actions = self._imagine(start_states)
        feats = feats.detach() 
        rew_pred = self.reward_head(feats).squeeze(-1)
        disc_pred = torch.sigmoid(self.discount_head(feats)).squeeze(-1) * DISCOUNT
        values = self.critic_target(feats).detach()

        returns = []
        g = values[:, -1]
        for t in reversed(range(IMAG_HORIZON)):
            g = rew_pred[:, t] + disc_pred[:, t] * g
            returns.insert(0, g)
        returns = torch.stack(returns, 1)  # (B,H)

        # Critic loss (two‑hot symlog)
        val_pred = self.critic(feats)
        target = symlog(returns.detach())
        loss_critic = F.mse_loss(val_pred, target)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP)
        self.opt_critic.step()

        # Actor loss (maximize predicted return via advantage)
        dist = self.actor(feats)
        log_prob_u = dist.log_prob(u_raw).sum(-1)
        advantage = (returns.detach() - symexp(val_pred).detach()) #val_pred는 log 스케일 값이기 때문에, 이를 원래 scale로 복원해야 함
        loss_actor = -(log_prob_u * advantage).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()

        WRITER.add_scalar("actor_critic/actor_loss", loss_actor.item(), self._steps)
        WRITER.add_scalar("actor_critic/critic_loss", loss_critic.item(), self._steps)
        WRITER.add_scalar("actor_critic/returns", returns.mean().item(), self._steps)
        nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP)
        self.opt_actor.step()

        # Update target critic
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.mul_(0.99).add_(0.01 * p.data)

        return {'critic': loss_critic.item(), 'actor': loss_actor.item()}


    # ---------------------------------------------------------------------
    # Environment interaction
    # ---------------------------------------------------------------------
    
    @torch.no_grad()
    def act(self, obs, training=True):
        obs_t = torch.tensor(obs, device=DEVICE).unsqueeze(0)  # (1,C,H,W)
        embed = self.encoder(obs_t)
        if self._steps == 0:
            self._state = self.rssm.init_state(1)
        self._state, _ = self.rssm.obs_step(self._state, torch.zeros(1, ACTION_DIM, device=DEVICE), embed)
        feat = self.rssm.get_feat(self._state)
        dist = self.actor(feat)
        if training:
            action = torch.tanh(dist.rsample())
        else:
            action = torch.tanh(dist.mean)
        return action.squeeze(0).detach().cpu().numpy()

    def store(self, transition):
        # Transition: dict(obs, act, rew, done).
        if transition['t'] == 0:
            self._episode = []
        self._episode.append(transition)
        if transition['done']:
            self.buffer.add_episode(self._episode)

# === Training loop =========================================================

def train(max_episodes=1_000_000, max_steps=MAX_STEPS):
    import model  # FightingModel env provided by user
    agent = DreamerAgent()
    tb_proc     = launch_tensorboard(TB_DIR, port=PORT_NUM)
    mon_total   = threading.Thread(target=monitor_metric,
                                   args=(TOTAL_REWARD_TXT, "Total Reward", TB_DIR),
                                   daemon=True).start()
    mon_evac80  = threading.Thread(target=monitor_metric,
                                   args=(EVAC80_TXT, "Evacuation Time 80", TB_DIR),
                                   daemon=True).start()
    mon_evac100 = threading.Thread(target=monitor_metric,
                                   args=(EVAC100_TXT, "Evacuation Time 100", TB_DIR),
                                   daemon=True).start()

    for ep in range(max_episodes):
        env = model.FightingModel(random.randint(3, 8), 50, 50, model_num=0, robot='Q')
        obs = env.return_current_image()  # (H,W)
        obs = obs[np.newaxis, ...]  # (1,H,W)
        total_reward = 0.0

        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps

        for t in range(max_steps):

            if env.alived_agents() < env.total_agents*0.2 and evacuation_time_80 == max_steps:
                evacuation_time_80 = t
            if env.alived_agents() == 0 and evacuation_time_100 == max_steps:
                evacuation_time_100 = t

            act = agent.act(obs)
            action_res = env.robot.receive_action(act)
            env.step()
            next_obs = env.return_current_image()[np.newaxis, ...]
            reward = 0.0  # TODO: compute/collect from env
            if(REWARD_A):
                reward += env.reward_based_alived() * REWARD_A
            if(REWARD_B):
                reward += env.reward_based_all_agents_danger() * REWARD_B
            if(REWARD_C):
                reward += env.reward_based_penalty() * REWARD_C
            if(REWARD_D):
                reward += env.reward_penalty_collision() * REWARD_D
            if(REWARD_E):
                reward += env.reward_based_gain() * REWARD_E
            if(REWARD_F):
                reward += env.reward_based_evacuated_with_robot() * REWARD_F
            reward+=REWARD_FIXED
            done = env.robot.is_game_finished or t == max_steps - 1
            agent.store({'obs': obs, 'act': act, 'rew': reward, 'done': done, 't': t})
            obs = next_obs
            total_reward += reward
            agent._steps += 1
            if done:
                break
        with open(TOTAL_REWARD_TXT, "a") as f:   f.write(f"{total_reward}\n")
        with open(EVAC80_TXT,       "a") as f:   f.write(f"{evacuation_time_80}\n")
        with open(EVAC100_TXT,      "a") as f:   f.write(f"{evacuation_time_100}\n")
        
        if ep % 1 == 0:
            if(ep>DREAMER_START_UPDATE_EP):
                for _ in range(DREAMER_UPDATE_FREQ):
                    agent.update()
        if ep % 100 == 0:
            print(f"Episode {ep} – reward {total_reward:.1f} – replay {len(agent.buffer)}")

if __name__ == "__main__":
    train()