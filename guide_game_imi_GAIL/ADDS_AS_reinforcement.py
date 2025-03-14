import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import pickle
from collections import deque
import argparse
import model

from timer_utils import Timer
from config import ENABLE_TIMER

import threading
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser

# ==============================
# 0. 설정
# ==============================

sim_timer = Timer()
learn_timer = Timer()

# 학습 모드: 
# 1) model_load = 1 -> 새로 학습
# 2) model_load = 2 -> 특정 모델 로드
# 3) model_load = 3 -> 가장 최근 모델 로드
model_load = 1 

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay_value", type=float, default=0.99)
parser.add_argument("--buffer_size", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--start_epsilon", type=float, default=1.0)
parser.add_argument("--epsilon_min", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--use_gail", type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument("--expert_dir", type=str, default="expert_data_dir")
parser.add_argument("--expert_buffer_size", type=int, default=50000)
parser.add_argument("--log_dir", type=str, default="learning_log_guide_game_gail")
parser.add_argument("--gail_alpha", type=float, default=1.0)
parser.add_argument("--gail_scale", type=float, default=0.1)
parser.add_argument("--imitation_decay", type=float, default=0.99)
parser.add_argument("--bc_init_weight", type=float, default=1.0)
parser.add_argument("--bc_alpha", type=float, default=1.0)
parser.add_argument('--imitation_lock', type=int, default=50)
args = parser.parse_args()

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, args.log_dir)
os.makedirs(log_dir, exist_ok=True)

start_episode = 0

def evaluate_discriminator(disc, expert_buffer, policy_buffer, device="cpu", batch_size=32):
    """
    Discriminator가 얼마나 전문가 vs. 정책 샘플을 잘 구분하는지 평가.
    - expert_buffer: 전문가 데이터 (ExpertBuffer or deque)
    - policy_buffer: 정책(Agent) 데이터 (ReplayBuffer)
    - batch_size: 각에서 뽑을 샘플 수
    - return: accuracy (0.0 ~ 1.0)
    """
    disc.eval()
    with torch.no_grad():
        # 1) 전문가 샘플(batch_size개) 추출
        expert_samples = random.sample(expert_buffer, batch_size)
        s_e, a_e, _, _, _ = zip(*expert_samples)  # (s,a,r,ns,d)
        s_e_t = torch.FloatTensor(s_e).unsqueeze(1).to(device)
        a_e_t = torch.FloatTensor(a_e).to(device)
        lbl_e = torch.ones((batch_size, 1), dtype=torch.float, device=device)

        # 2) 정책(Agent) 샘플(batch_size개) 추출
        policy_samples = random.sample(policy_buffer, batch_size)
        s_p, a_p, _, _, _ = zip(*policy_samples)
        s_p_t = torch.FloatTensor(s_p).unsqueeze(1).to(device)
        a_p_t = torch.FloatTensor(a_p).to(device)
        lbl_p = torch.zeros((batch_size, 1), dtype=torch.float, device=device)

        # 3) Discriminator 출력 계산 (로짓)
        logits_e = disc(s_e_t, a_e_t)
        logits_p = disc(s_p_t, a_p_t)

        # 4) 시그모이드 확률
        prob_e = torch.sigmoid(logits_e)  # 기대값은 1
        prob_p = torch.sigmoid(logits_p)  # 기대값은 0

        # 5) 0.5 기준으로 분류
        pred_e = (prob_e >= 0.5).float()  # 전문가로 예측
        pred_p = (prob_p >= 0.5).float()  # 전문가로 예측

        # 6) 정확도 계산
        correct_e = (pred_e == lbl_e).sum().item()
        correct_p = (pred_p == lbl_p).sum().item()
        accuracy = (correct_e + correct_p) / (2.0 * batch_size)
    disc.train()
    return accuracy

def monitor_disc_score(disc_score_file, tb_log_dir, start_episode):
    writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "disc_score"))
    # 파일 생성 대기
    while not os.path.exists(disc_score_file):
        print(f"Waiting for {disc_score_file} to be created...")
        time.sleep(2)
    with open(disc_score_file, "r") as f:
        episode = 0
        print("Start monitoring disc_score.txt for new scores...")
        try:
            while True:
                line = f.readline()
                if line:
                    line=line.strip()
                    if line:
                        try:
                            disc_acc = float(line)
                            writer.add_scalar("Discriminator/Accuracy", disc_acc, start_episode + episode + 1)
                            episode += 1
                        except ValueError:
                            print("Invalid value in disc_score.txt:", line)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring disc_score interrupted by user.")
        finally:
            writer.close()

def monitor_mse_loss(mse_loss_file, tb_log_dir, start_episode):
    writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "mse_loss"))
    while not os.path.exists(mse_loss_file):
        print(f"Waiting for {mse_loss_file} to be created...")
        time.sleep(2)
    with open(mse_loss_file, "r") as f:
        episode = 0
        print("Start monitoring mse_loss.txt for new values...")
        try:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            mse_val = float(line)
                            writer.add_scalar("ImitationLoss/mse_loss", mse_val, start_episode + episode + 1)
                            episode += 1
                        except ValueError:
                            print("Invalid value in mse_loss.txt:", line)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring MSE loss interrupted by user.")
        finally:
            writer.close()



def launch_tensorboard(tb_log_dir, port=6006):
    tb_process = subprocess.Popen(["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    time.sleep(5)
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"TensorBoard launched at {url}")
    return tb_process

# ==============================
# 1. Discriminator for GAIL
# ==============================
class DiscriminatorNetwork(nn.Module):
    """
    GAIL용 판별자 네트워크:
      - 입력: (state, action)
      - 출력: D(s,a) in (0,1) => "전문가로부터 왔을 확률"
    """
    def __init__(self, input_shape=(50,50), action_dim=2):
        super(DiscriminatorNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size + action_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.logits = nn.Linear(64, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state, action):
        """
        state: (B,1,H,W)
        action: (B, action_dim)
        return: (B,1) => sigmoid 로지스틱 출력
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logit = self.logits(x)
        #out = torch.sigmoid(logit)  # (B,1) #CNN으로 이미지 특징 추출하고, 추출한 특징과 action을 바탕으로 0인지 1인지 출력하는군
        out = logit #바로 logit 출력
        return out

# ========================================
# 2. ReplayBuffer (정책/환경 데이터용)
# ========================================
class ReplayBuffer:
    def __init__(self, capacity=int(1e5)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(np.array(states)).unsqueeze(1)  # (B,1,H,W)
        actions     = torch.FloatTensor(np.array(actions))              # (B,2)
        rewards     = torch.FloatTensor(np.array(rewards))              # (B,)
        next_states = torch.FloatTensor(np.array(next_states)).unsqueeze(1)
        dones       = torch.FloatTensor(np.array(dones))                    # (B,)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)
        print(f"ReplayBuffer loaded. size={len(self.buffer)}")



# ========================================
# 3. ExpertBuffer (전문가 데이터)
# ========================================
class ExpertBuffer:
    """
    이미 수집된 (state, action) 형태 전문가 데이터
    예) 사람이 PyGame으로 조종해서 만든 pkl 파일 등
    """
    def __init__(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)  # [(s,a,r,ns,d), ...]
        self.samples = data
        self.size = len(self.samples)

    def sample(self, batch_size):
        batch = random.sample(self.samples, batch_size)
        # (s, a, r, ns, d)
        states, actions, rewards, next_states, dones = zip(*batch)
        states      = torch.FloatTensor(states).unsqueeze(1)  # shape (B,1,70,70)
        actions     = torch.FloatTensor(actions)               # shape (B,2)
        rewards     = torch.FloatTensor(rewards)              # shape (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1)
        dones       = torch.FloatTensor(dones)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

# ========================================
# 4. QNetwork (Critic) for SAC
# ========================================
class QNetwork(nn.Module):
    def __init__(self, input_shape=(50,50), action_dim=2):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_out = nn.Linear(128, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1,1,*shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.q_out(x)
        return q_val

# ========================================
# 5. PolicyNetwork (Actor) for SAC
# ========================================
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape=(50,50)):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = -10
        self.log_std_max = -0.5

        # conv backbone
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        conv_out_size = self._get_conv_out(input_shape)

        self.fc_backbone = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mean_head = nn.Linear(128, 2)
        self.log_std_head = nn.Linear(128, 2)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1,1,*shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def backbone(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc_backbone(x)

    def forward(self, state):
        feat = self.backbone(state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, state):
        # reparameterization trick
        mean, log_std = self.forward(state)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        raw_action = mean + std * eps  # Gaussian 샘플
        
        # sigmoid를 이용한 스쿼싱 및 스케일링: sigmoid 출력은 [0,1]이므로 [-2,2]로 변환
        action_sigmoid = torch.sigmoid(raw_action)
        action = 4.0 * action_sigmoid - 2.0

        # 원래 Gaussian 분포에 따른 로그 확률 계산
        log_prob = -0.5 * ((raw_action - mean)**2 / (std**2 + 1e-8) + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=1)

        # 변환에 따른 로그 자코비안 보정: y = 4 * sigmoid(raw_action) - 2 이므로,
        # derivative = 4 * sigmoid(raw_action) * (1 - sigmoid(raw_action))
        log_det_jacobian = torch.log(4.0 * action_sigmoid * (1.0 - action_sigmoid) + 1e-6)
        log_det_jacobian = log_det_jacobian.sum(dim=1)

        # 최종 로그 확률에 자코비안 보정 적용 (변환에 따른 정보 손실 보정)
        log_prob = log_prob - log_det_jacobian

        return action, log_prob
# ========================================
# 6. SACAgent (with GAIL option)
# ========================================
class SACAgent:
    def __init__(self, input_shape=(50,50), gamma=0.99, alpha=0.2, tau=0.995,
                 lr=1e-4, batch_size=64, replay_size=int(1e5), device="cpu",
                 gail_alpha=0.5, start_epsilon=args.start_epsilon, start_epsilon_long=0.05, epsilon_min=args.epsilon_min, imitation_decay = args.imitation_decay, bc_init_weight = args.bc_init_weight, bc_alpha = args.bc_alpha):
        """
        gail_alpha: 0~1 사이. 1이면 완전히 GAIL 보상만, 0이면 env 보상만
        start_epsilon: 초기 epsilon (랜덤 탐사율)
        start_epsilon_long: 장기 탐사율 초기값
        """
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.gail_alpha = gail_alpha
        self.gail_scaler = args.gail_scale
        self.device = torch.device(device)
        self.imitation_decay = imitation_decay
        self.bc_weight = bc_init_weight
        self.bc_alpha = bc_alpha
        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        self.imitation_lock = args.imitation_lock 

        # Critic networks
        self.q1 = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q2 = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q1_target = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q2_target = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = PolicyNetwork(input_shape).to(self.device)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.epsilon = start_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = 0.005

    def save_model(self, filepath):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy': self.policy.state_dict(),
            'q1_opt': self.q1_optimizer.state_dict(),
            'q2_opt': self.q2_optimizer.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        ckpt = torch.load(filepath)
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.q1_target.load_state_dict(ckpt['q1_target'])
        self.q2_target.load_state_dict(ckpt['q2_target'])
        self.policy.load_state_dict(ckpt['policy'])
        self.q1_optimizer.load_state_dict(ckpt['q1_opt'])
        self.q2_optimizer.load_state_dict(ckpt['q2_opt'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        print(f"Model loaded from {filepath}")


    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1 - self.tau) * param.data)

    def update_epsilon(self, is_down, decay_value):
        if is_down:
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_value)
        else:
            self.epsilon = min(1.0, self.epsilon / decay_value)

    def update_epsilon2(self, is_down, decay_value):
        if is_down:
            self.epsilon_long = max(self.epsilon_long_min, self.epsilon_long * decay_value)
        else:
            self.epsilon_long = min(1.0, self.epsilon_long / decay_value)


    def store_transition(self, s, a, r, ns, done):
        self.replay_buffer.push(s, a, r, ns, done)

    def select_action(self, state_np, deterministic=False):
        # epsilon-greedy (단, continuous라 random uniform도 가능)
        if np.random.rand() < self.epsilon:
            dx = np.random.uniform(-2,2)
            dy = np.random.uniform(-2,2)
            return np.array([dx, dy]), True

        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, log_std = self.policy.forward(state_t)
            std = log_std.exp()
            if deterministic:
                action_t = mean
            else:
                eps = torch.randn_like(mean)
                action_t = mean + std*eps
            action_t = 2.0 * torch.tanh(action_t)
        return action_t.cpu().numpy()[0], False

    def calc_gail_reward(self, disc, state_t, action_t, env_reward):
        """
        하이브리드 보상:
         r_total = gail_alpha * log(D(s,a)) + (1-gail_alpha)*env_reward
        """
        with torch.no_grad():
            logits = disc(state_t, action_t) # 전문가 데이터는 라벨이 1로 주어지기 때문에, 판별자가 전문가 데이터와 유사한 상태-액션 쌍에 대해 높은 확률을 출력, 그래서 전문가와 유사할수록 높은 값을 가지게 됨. 
            prob = torch.sigmoid(logits)
            r_gail = torch.log(prob + 1e-8).squeeze(-1) # log(D(s,a))
            # tensor화
            r_gail = self.gail_scaler * r_gail
            env_r = env_reward.to(self.device)
            r_total = self.gail_alpha*r_gail + (1.0 - self.gail_alpha)*env_r
        return r_total

    def update(self, disc=None, expert_states = None, expert_actions = None):
        if len(self.replay_buffer) < self.batch_size*50:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = (
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device)
        )

        # 만약 GAIL 사용 (disc != None)이면 하이브리드 보상 계산
        if disc is not None:
            # rewards shape=(B,) -> tensor
            new_rewards = self.calc_gail_reward(disc, states, actions, rewards)
            rewards = new_rewards.detach().clone()

        # --------- Q target
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample_action(next_states)
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            q_target = rewards + self.gamma * (1-dones) * (q_next - self.alpha * next_log_prob)

        # ----- Update Q1, Q2
        q1_val = self.q1(states, actions).squeeze(-1)
        q2_val = self.q2(states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()

        # ----- Update Policy
        new_action, log_prob = self.policy.sample_action(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        bc_loss = 0.0
        if expert_states is not None and expert_actions is not None:
            bc_loss = self.calc_bc_loss_in_actor_update(expert_states, expert_actions, wri=1)
        policy_loss = (1-self.bc_alpha) * policy_loss + (self.bc_alpha) * bc_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # soft update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)
        
    def save_replay_buffer(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        self.replay_buffer.save(filepath)

    def load_replay_buffer(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        self.replay_buffer.load(filepath)
        print("Replay buffer loaded.")
        print("Replay buffer size:", len(self.replay_buffer))
    

    def decay_bc_weight(self):
        """
        매 에피소드가 끝날 때마다 imitation_decay를 곱해서
        BC 손실 가중치를 점차 감소.
        """
        self.bc_weight *= self.imitation_decay
        self.bc_alpha *= self.imitation_decay

    def calc_bc_loss_in_actor_update(self, expert_states, expert_actions, wri=1):
        """
        SAC actor loss 계산 시, BC Loss도 함께 더해서 계산하는 방식.
        (예시: update() 함수 내부에서 호출 가능)
        """
        predicted_mean, _ = self.policy.forward(expert_states)
        predicted_action = 4.0 * torch.sigmoid(predicted_mean) - 2.0
        loss_bc = F.mse_loss(predicted_action, expert_actions)

        mse_loss_file = os.path.join(log_dir, "mse_loss.txt")
        if not os.path.exists(mse_loss_file):
            open(mse_loss_file, "w").close()
        if(wri ==1):
            with open(mse_loss_file, "a") as f:
                f.write(f"{loss_bc.item()}\n")
            return self.bc_weight * loss_bc

def load_expert_data_randomly(prefix, capacity):
    """
    전역 변수 log_dir에서, 주어진 접두어(prefix)로 시작하는 .pkl 파일 목록을 가져온 후,
    파일 이름만으로 랜덤하게 선택하여, 파일을 하나씩 읽고 내부 데이터를 버퍼에 채웁니다.
    버퍼가 capacity에 도달하면 중단합니다.
    
    예:
        prefix = "imitation_dataset"  # 파일명이 imitation_dataset_1.pkl, imitation_dataset_2.pkl 등
    """
    expert_files = [f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith(".pkl")]
    if not expert_files:
        print(f"[WARN] {log_dir} 내에서 '{prefix}'로 시작하는 파일을 찾지 못했습니다.")
        return []
    
    buffer = []
    # 읽지 않은 파일 목록 (중복 선택되지 않도록 함)
    remaining_files = expert_files.copy()
    
    # capacity를 채울 때까지 또는 파일 목록이 소진될 때까지 반복
    while len(buffer) < capacity and remaining_files:
        # 남은 파일 중 하나를 랜덤 선택
        fname = random.choice(remaining_files)
        remaining_files.remove(fname)
        full_path = os.path.join(log_dir, fname)
        
        try:
            with open(full_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[WARN] {fname} 로딩 실패: {e}")
            continue
        
        # 파일 내부 데이터의 순서를 랜덤하게 섞음
        random.shuffle(data)
        for sample in data:
            if len(buffer) < capacity:
                buffer.append(sample)
            else:
                break

    print(f"{log_dir} 내 {len(expert_files)}개 파일 중 {len(expert_files) - len(remaining_files)}개 파일에서 데이터를 읽어 총 {len(buffer)}개의 샘플을 로드 (capacity={capacity})")
    return buffer

def monitor_total_reward(total_reward_file, tb_log_dir, start_episode):
    # total_reward 로그는 tb_log_dir/total_reward 하위 디렉토리에 기록
    writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "total_reward"))
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
                            writer.add_scalar("Total Reward", total_reward, start_episode + episode + 1)
                            episode += 1
                        except ValueError:
                            print("Invalid value in total_reward.txt:", line)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring interrupted by user.")
        finally:
            writer.close()

def monitor_total_gail_reward(gail_reward_file, tb_log_dir, start_episode):
    # GAIL 로그는 tb_log_dir/gail_reward 하위 디렉토리에 기록
    writer = SummaryWriter(log_dir=os.path.join(tb_log_dir, "gail_reward"))
    while not os.path.exists(gail_reward_file):
        print(f"Waiting for {gail_reward_file} to be created...")
        time.sleep(2)
    with open(gail_reward_file, "r") as f:
        episode = 0
        print("Start monitoring total_reward_gail.txt for GAIL rewards...")
        try:
            while True:
                line = f.readline()
                if line:
                    line = line.strip()
                    if line:
                        try:
                            gail_r = float(line)
                            writer.add_scalar("GAIL Reward", gail_r, start_episode + episode + 1)
                            episode += 1
                        except ValueError:
                            print("Invalid value in total_reward_gail.txt:", line)
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("Monitoring GAIL reward interrupted by user.")
        finally:
            writer.close()


def update_discriminator_and_log(disc, disc_optimizer, 
                                 expert_buffer, policy_buffer, 
                                 disc_score_path, device="cpu", batch_size=32):
    """
    1) Discriminator update
    2) Evaluate & log disc accuracy
    """
    # --- 1) Update Discriminator (간단 예시) ---
    disc_optimizer.zero_grad()
    exp_samples = random.sample(expert_buffer, batch_size)
    pol_samples = random.sample(policy_buffer, batch_size)

    s_e, a_e, _, _, _ = zip(*exp_samples)
    s_e_t = torch.FloatTensor(s_e).unsqueeze(1).to(device)
    a_e_t = torch.FloatTensor(a_e).to(device)
    lbl_e = torch.ones((batch_size,1), dtype=torch.float, device=device)

    s_p, a_p, _, _, _ = zip(*pol_samples)
    s_p_t = torch.FloatTensor(s_p).unsqueeze(1).to(device)
    a_p_t = torch.FloatTensor(a_p).to(device)
    lbl_p = torch.zeros((batch_size,1),dtype=torch.float,device=device)

    all_s = torch.cat([s_e_t, s_p_t], dim=0)
    all_a = torch.cat([a_e_t, a_p_t], dim=0)
    all_l = torch.cat([lbl_e, lbl_p], dim=0)

    preds = disc(all_s, all_a)
    loss_disc = F.binary_cross_entropy_with_logits(preds, all_l)
    loss_disc.backward()
    disc_optimizer.step()

    # --- 2) Evaluate accuracy ---
    disc_accuracy = evaluate_discriminator(disc, expert_buffer, policy_buffer, device, batch_size)

    # --- 2-1) disc_score.txt 파일에 기록 ---
    with open(disc_score_path, "a") as f:
        f.write(f"{disc_accuracy}\n")



# ====================================================
# 7. 메인 학습 루프 (SAC + GAIL 하이브리드)
# ====================================================
if __name__ == "__main__":

    start_episode = 0
    # ------------------------------------------
    # TensorBoard, timer, etc. (기존 그대로)
    # ------------------------------------------
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    total_reward_gail_file = os.path.join(log_dir, "total_reward_gail.txt")
    disc_score_file = os.path.join(log_dir, "disc_score.txt")
    mse_loss_file = os.path.join(log_dir, "mse_loss.txt")

    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")

    tb_process = launch_tensorboard(tb_log_dir, port=6006)
    monitor_thread1 = threading.Thread(
        target=monitor_total_reward, 
        args=(total_reward_file, tb_log_dir, start_episode), 
        daemon=True)
    monitor_thread1.start()

    monitor_thread2 = threading.Thread(
        target=monitor_total_gail_reward,
        args=(total_reward_gail_file, tb_log_dir, start_episode),
        daemon=True)
    monitor_thread2.start()

    monitor_thread3 = threading.Thread(
        target=monitor_disc_score,
        args=(disc_score_file, tb_log_dir, start_episode),
        daemon=True
    )
    monitor_thread3.start()

    monitor_thread4 = threading.Thread(
        target=monitor_mse_loss,
        args = (mse_loss_file, tb_log_dir, start_episode),
        daemon = True
    )
    monitor_thread4.start()
    # ------------------------------------------
    # Hyperparams & GAIL 여부
    # ------------------------------------------
    max_episodes = 2000
    max_steps = 3000
    number_of_agents = 20

    epsilon_path = os.path.join(log_dir, "start_epsilon.txt")

    # [1] epsilon 등 로드
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            try:
                lines = f.readlines()
                if len(lines) >= 2:
                    start_epsilon = float(lines[0].strip())
                    start_epsilon_long = float(lines[1].strip())
                    print(f"Loaded start_epsilon: {start_epsilon}, start_epsilon_long: {start_epsilon_long}")
                else:
                    print("Not enough lines in start_epsilon.txt. Resetting values.")
                    start_epsilon = args.start_epsilon
                    start_epsilon_long = 0.05
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to defaults.")
                start_epsilon = args.start_epsilon
                start_epsilon_long = 0.05
    else:
        start_epsilon = args.start_epsilon
        start_epsilon_long = 0.05
        print("No start_epsilon.txt found. Initializing values to defaults.")

    # ------------------------------------------
    # [2] SACAgent 생성
    # ------------------------------------------
    agent = SACAgent(
        input_shape=(50,50),
        alpha=args.alpha,
        lr=args.lr,
        start_epsilon=start_epsilon,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        replay_size=args.buffer_size,
        device=args.device,
        gail_alpha=args.gail_alpha
    )
    print(f"Agent initialized, lr={args.lr}, alpha={agent.alpha}, batch_size={args.batch_size}, replay_size={args.buffer_size}")

    replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

    # ------------------------------------------
    # [3] 모델 로드(1: fresh, 2: specified, 3: latest)
    # ------------------------------------------
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "sac_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)

        if os.path.exists(model_path):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_name)
            if os.path.exists(replay_buffer_path):
                agent.load_replay_buffer("replay_buffer.pkl")
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir) if f.startswith("sac_checkpoint") and f.endswith(".pth")]
        if model_files:
            latest_model = max(model_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            latest_model_path = os.path.join(log_dir, latest_model)
            start_episode = int(latest_model.split("_")[-1].split(".")[0])
            print(f"Loading latest model: {latest_model}")
            agent.load_model(latest_model_path)
            if os.path.exists(replay_buffer_path):
                print(f"Loading replay buffer from {replay_buffer_path}")
                agent.load_replay_buffer(replay_buffer_path)
        else:
            pass

    # ------------------------------------------
    # [4] GAIL 여부, 전문가 데이터 불러오기
    # ------------------------------------------
    use_gail = args.use_gail  # 예: python xxx.py --use_gail
    if use_gail:
        print("GAIL mode ON. Loading expert dataset.")
        # 예: (s,a,r,ns,d) 형태
        expert_samples = load_expert_data_randomly(args.expert_dir, capacity=args.expert_buffer_size)

        disc = DiscriminatorNetwork().to(agent.device)
        disc_optimizer = optim.Adam(disc.parameters(), lr=1e-4)

        # 전문가 버퍼 (메모리에 올려두고 샘플링)
        from collections import deque
        expert_buffer = deque(expert_samples)  # RAM 상에 임시
        # or ExpertBuffer class
        def sample_expert(batch_size=64):
            batch = random.sample(expert_buffer, batch_size)
            s, a, r, ns, d = zip(*batch)
            s_t  = torch.FloatTensor(s).unsqueeze(1).to(agent.device)  # (B,1,70,70)
            a_t  = torch.FloatTensor(a).to(agent.device)               # (B,2)
            return s_t, a_t
    else:
        disc = None
        disc_optimizer = None
        expert_buffer = None

    abnormal_reward = 0

    # ------------------------------------------
    # [5] Main training loop
    # ------------------------------------------
    for episode in range(max_episodes):
        print(f"Episode {start_episode + episode + 1}")

        episode_gail_reward = 0

        # 매 10 에피소드마다, expert_buffer 새로 로딩 (GAIL ON일 때만)
        if use_gail and (episode % 10 == 0) and (episode != 0):
            expert_samples = load_expert_data_randomly(args.expert_dir, args.expert_buffer_size)
            expert_buffer = deque(expert_samples, maxlen=args.expert_buffer_size)

        # Create environment
        while True:
            try:
                env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
                break
            except Exception as e:
                print(e, "Retrying environment creation...")

        state = env_model.return_current_image()
        total_reward = 0
        reward = 0
        buffered_state = state
        buffered_action = None
        abnormal_reward = 0

        try:
            for step in range(max_steps):

                # 1) Select action
                if step % 3 == 0:
                    # exploration
                    if np.random.rand() < agent.epsilon_long:
                        env_model.robot.now_exploration = 0

                    action_np, _ = agent.select_action(state)
                    dx, dy = action_np[0], action_np[1]
                    real_action = env_model.robot.receive_action([dx, dy])
                    # 실제 적용된 action(벽 등으로 약간 바뀔 수 있다고 가정)
                    action_np[0] = real_action[0]
                    action_np[1] = real_action[1]

                    buffered_state = state
                    buffered_action = action_np

                # Simulation time check
                sim_timer.start()
                # 2) Step environment
                env_model.step()
                sim_timer.stop()

                # 3) Reward
                r_a = env_model.reward_based_alived()
                #r_d = env_model.reward_based_all_agents_danger()
                reward = r_a

                # 4) Next state
                next_state = env_model.return_current_image()

                # 5) Done?
                done = (step >= max_steps - 1) or (env_model.robot.is_game_finished)
                if env_model.robot.is_game_finished:
                    reward += 10

                # 6) Store transition
                if step % 3 == 2 and step > 5:
                    agent.store_transition(
                        buffered_state,
                        buffered_action,
                        reward,
                        next_state,
                        float(done)
                    )
                    total_reward += reward

                    if use_gail and disc is not None:
                        with torch.no_grad():
                            s_t = torch.FloatTensor(buffered_state).unsqueeze(0).unsqueeze(0).to(agent.device)
                            a_t = torch.FloatTensor(buffered_action).unsqueeze(0).to(agent.device)
                            rew_t = torch.tensor([reward], dtype=torch.float32).to(agent.device)
                            r_gail_t = agent.calc_gail_reward(disc, s_t, a_t, rew_t)
                            episode_gail_reward += r_gail_t.item()
                    #print("reward : ", reward)
                    reward = 0
                # 7) GAIL Discriminator Update (예: 5 스텝마다)
                if use_gail and disc is not None and disc_optimizer is not None:
                    if step % 100 == 0 and len(agent.replay_buffer) > agent.batch_size and len(expert_buffer) > 100:
                        update_discriminator_and_log(
                            disc=disc, 
                            disc_optimizer=disc_optimizer,
                            expert_buffer=expert_buffer, 
                            policy_buffer=agent.replay_buffer.buffer, 
                            disc_score_path=disc_score_file,
                            device=agent.device,
                            batch_size=32
                        )

                # 8) SAC update (with disc for GAIL reward?)
                if step % 3 == 2:
                    learn_timer.start()
                    if use_gail and disc is not None:
                        
                        be_batch_size = agent.batch_size
                        if len(expert_buffer) < be_batch_size:
                            be_batch_size = len(expert_buffer)
                        expert_states, expert_actions = sample_expert(be_batch_size)
                        agent.update(disc=disc, expert_states=expert_states, expert_actions=expert_actions)
                    else:
                        agent.update()           # no GAIL
                    learn_timer.stop()

                state = next_state
                if done:
                    break
                if(episode > agent.imitation_lock):
                    agent.decay_bc_weight()

        except Exception as e:
            print(e)
            print("error occured. retry.")
            print(f"{step} steps done.")
            env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
            abnormal_reward = 1

        # Possibly update epsilon
        if (episode + 1) % 1 == 0: # every 10 episodes
            decay_value = args.decay_value
            agent.update_epsilon(True, decay_value)
            agent.update_epsilon2(True, decay_value)

        print("Total reward :", total_reward)
        print("GAIL reward sum : ", episode_gail_reward)
        print("now_epsilon : ", agent.epsilon)
        print("now_epsilon_long : ", agent.epsilon_long)

        # Save model occasionally
        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        if not os.path.exists(reward_file_path):
            open(reward_file_path, "w").close()

        if (episode + 1) % 50 == 0:
            model_filename = os.path.join(log_dir, f"sac_checkpoint_ep_{start_episode + episode + 1}.pth")
            agent.save_model(model_filename)
            replay_buffer_filename = "replay_buffer.pkl"
            agent.save_replay_buffer(replay_buffer_filename)

        with open(reward_file_path, "a") as f:
            if(abnormal_reward != 1):
                f.write(f"{total_reward}\n")

        if use_gail:
            gail_reward_path = os.path.join(log_dir, "total_reward_gail.txt")
            if not os.path.exists(gail_reward_path):
                open(gail_reward_path, "w").close()
            with open(gail_reward_path, "a") as f:
                f.write(f"{episode_gail_reward}\n")

        # epsilon 저장
        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon)+"\n")
            f.write(str(agent.epsilon_long))

        if ENABLE_TIMER:
            print(f"episode {start_episode+episode+1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode+episode+1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
