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
from typing import Tuple
from Start_training import START_DECAY_STEP, START_EPSILON, EPSILON_MIN, SCHEDULER_TYPE, DECAY_VALUE, REWARD_A, REWARD_B, REWARD_C, REWARD_D, REWARD_E, REWARD_F, REWARD_G, REWARD_H, REWARD_I, REWARD_J, REWARD_K, CROWD_NUMBER_MIN, CROWD_NUMBER_MAX, LINEARLY_DECAY_STEP, START_UPDATE_STEP, LOG_DIR, FINISHED_BONUS, REWARD_FIXED, MAP_NUM, EXPLORATION_TYPE, \
                            LR, BUFFER_SIZE, BATCH_SIZE, LOG_STD_MAX, LOG_STD_MIN, ALPHA_START, ALPHA_END, ALPHA_DECAY_STEPS, DEVICE, SCALE_CHECK, ACTION_SCALE, START_BATCH_TIMES, MAX_STEPS, PORT_NUM, LONG_EPSILON_MIN, START_LONG_EPSILON, \
                            GAMMA_START, GAMMA_SCHEDULE_STEP, GAMMA_END, MAP_NUM_RANDOM
from heat_map import HeatMapLogger #@for heat_map
# Timer instances
sim_timer = Timer() 
learn_timer = Timer()
home_dir = os.path.expanduser("~")

model_load = 3
# start_fresh : 1
# load specified model : 2
# load latest model : 3

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, LOG_DIR)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

heat_logger = HeatMapLogger(   #@for heat_map
    save_root = os.path.join(log_dir, "heat_maps"),
    map_size = (50, 50),
    known_maps = MAP_NUM_RANDOM
)

def launch_tensorboard(tb_log_dir, port=6006):
    """
    TensorBoard를 백그라운드에서 실행하고 기본 브라우저에 해당 URL을 엽니다.
    """
    # tensorboard 실행 (포트 지정)
    tb_process = subprocess.Popen(["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    # 잠시 대기한 후, 브라우저에서 TensorBoard URL 열기
    time.sleep(5)  # TensorBoard가 시작할 시간을 줌
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"TensorBoard launched at {url}")
    return tb_process

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
    
    
def interest_direction(heat: np.ndarray, wall: np.ndarray, pos_xy: Tuple[float, float]) -> float:
    """가장 방문 빈도가 낮은 셀을 확률적으로 뽑아 그 방향(라디안)을 반환."""
    x0, y0 = map(int, np.round(pos_xy))
    H, W = heat.shape

    # (1) 방문 횟수 → 관심도(작을수록 ↑)
    interest = np.log(heat.max() + 1) - np.log(heat + 1)
    interest[wall == 1] = 0.0           # 장애물은 제외
    if interest.sum() == 0:
        return np.random.uniform(-np.pi, np.pi)

    # (2) 확률화 후 목표 셀 샘플
    prob = interest.ravel() / interest.sum()
    tgt_idx = np.random.choice(H * W, p=prob)
    ty, tx = divmod(tgt_idx, W)

    # (3) 현재 위치 → 목표 셀 벡터 각도
    return np.arctan2(ty - y0, tx - x0)


def guided_random_action(agent, env_model):
    """Heat Map + Obstacle mask 로 유도된 (dx,dy) 한 쌍 반환."""
    heat = agent.current_heat
    wall = agent.obstacle_mask
    if heat is None or wall is None:
        return np.random.uniform(-2, 2), np.random.uniform(-2, 2)

    x0, y0 = env_model.robot.xy
    step = agent.step_size

    for _ in range(30):                       # 30회까지 재시도
        theta_c = interest_direction(heat, wall, (x0, y0))
        theta = theta_c + np.random.uniform(-np.pi / 2, np.pi / 2)   # ±90°
        dx, dy = step * np.cos(theta), step * np.sin(theta)

        nx, ny = int(round(x0 + dx)), int(round(y0 + dy))
        if nx < 0 or nx >= wall.shape[1] or ny < 0 or ny >= wall.shape[0]:
            continue
        if wall[ny, nx] == 1:                 # 충돌
            continue
        return dx, dy

    # fallback : 모든 재시도 실패
    print("heat 기반 exploration 실패!")
    return np.random.uniform(-2, 2), np.random.uniform(-2, 2)



##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e4), device=None):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        # 1) 무작위로 batch_size만큼 샘플 추출
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 2) numpy array로 묶기 (zero-copy에 가깝게)
        states_np      = np.stack(states, axis=0).astype(np.float32)       # (B, H, W)
        next_states_np = np.stack(next_states, axis=0).astype(np.float32)  # (B, H, W)
        actions_np     = np.stack(actions, axis=0).astype(np.float32)      # (B, action_dim)
        rewards_np     = np.array(rewards, dtype=np.float32)               # (B,)
        dones_np       = np.array(dones,   dtype=np.float32)               # (B,)

        # 3) torch tensor로 변환 & 차원 맞추기
        states_t       = torch.from_numpy(states_np).unsqueeze(1).to(device)       # (B,1,H,W)
        next_states_t  = torch.from_numpy(next_states_np).unsqueeze(1).to(device)  # (B,1,H,W)
        actions_t      = torch.from_numpy(actions_np).to(device)                   # (B,action_dim)
        rewards_t      = torch.from_numpy(rewards_np).to(device)                   # (B,)
        dones_t        = torch.from_numpy(dones_np).to(device)                     # (B,)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)

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
        
class EpsilonScheduler:
    """
    Epsilon Scheduler for epsilon-greedy exploration.
    
    Parameters:
      - start_epsilon: 초기 ε 값.
      - epsilon_min: 최소 ε 값.
      - start_decay_step: ε 감소를 시작할 step(또는 에피소드) 번호.
      - scheduler_type: "exponential" (지수적 감소) 또는 "linear" (선형 감소).
      - decay_value: 지수적 감소 시 매 step마다 곱할 값 (예: 0.99).
      - linear_decay_steps: 선형 감소 시 start_epsilon에서 epsilon_min까지 감소시키는 총 step 수.
    """
    def __init__(self, start_epsilon, epsilon_min, start_decay_step, scheduler_type="e",
                 decay_value=0.99, linear_decay_steps=1000):
        self.start_epsilon = START_EPSILON
        self.epsilon_min = epsilon_min
        self.start_decay_step = start_decay_step
        self.scheduler_type = scheduler_type
        self.decay_value = decay_value
        self.linear_decay_steps = linear_decay_steps

    def get_epsilon(self, now_epsilon, episode):
        # 아직 감소 시작 전이면 초기값 반환
        if episode < self.start_decay_step:
            return now_epsilon

        if self.scheduler_type == "e":
            # 현재 step 이후부터 지수적으로 감소
            epsilon = now_epsilon * self.decay_value
            return max(epsilon, self.epsilon_min)
        elif self.scheduler_type == "l":
            # 선형적으로 감소: 감쇠 시작부터 linear_decay_steps 동안 선형적으로 감소
            fraction = min(1 / self.linear_decay_steps, 1.0)
            epsilon = now_epsilon-fraction
            return epsilon
        else:
            raise ValueError("scheduler_type must be either 'exponential' or 'linear'")
        
    
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



##########################################################################
# 5) SAC Agent for Action
##########################################################################
class SACAgent:
    def __init__(self, input_shape=(50,50), gamma=GAMMA_START, alpha=0.2, tau=0.995, lr=1e-4, batch_size=64, replay_size=int(1e5), device="cpu", start_epsilon = 1.0, start_epsilon_long = 0.1, long_epsilon_min=0, heat_logger=None):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = long_epsilon_min

        self.heat_logger = heat_logger
        self.current_map_id = None
        self.current_heat = None
        self.step_size = 2.0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=int(replay_size))
        

        # Critic networks
        self.q1 = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q2 = QNetwork(input_shape, action_dim=2).to(self.device) #Q값의 과대평가 문제 줄이기 위해 double Q 도입
        # self.q1, self.q2 -> 현재 상태 s와 행동 a에 대해 Q-value를 근사하는 네트워크
        # predicted Q와 target Q의 차이를 줄이자

        self.q1_target = QNetwork(input_shape, action_dim=2).to(self.device)
        self.q2_target = QNetwork(input_shape, action_dim=2).to(self.device)
        # self.q1_target, self.q2_target -> Q의 Ground Truth 근사치 제공
        # q-network 업데이트 시 사용하는 Target 값을 제공

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = PolicyNetwork(input_shape).to(self.device)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr) #parameter optimizaing
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)



# ------------------------------------------------- #
    # Soft update
    # ------------------------------------------------- #
    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(
                self.tau * target_param.data + (1 - self.tau) * param.data
            )

    # ------------------------------------------------- #
    # Store experience
    # ------------------------------------------------- #
    def store_transition(self, s, a, r, s_next, done):
        # if -20 <= a[0] <= 20 and -20 <= a[1] <= 20:
        self.replay_buffer.push(s, a, r, s_next, done)

    def heat_loaded(self, map_id: int, obstacle_mask: np.ndarray):
        self.current_map_id = map_id

        try:
            self.current_heat = self.heat_logger._load(map_id)
            print("heat_map load 성공")
        except FileNotFoundError:
            H, W = self.heat_logger.map_size
            self.current_heat = np.zeros((H, W), dtype=np.float32)

        self.obstacle_mask = obstacle_mask

    # ------------------------------------------------- #
    # Select action
    # ------------------------------------------------- #
    def select_action(self, state_np, deterministic=False, env_model=None):
        """
        state_np: shape (H, W) or (1, H, W)
        returns action_np shape (4,) = [dx, dy, mode0, mode1]
        If using epsilon > 0.0 for random exploration, 
        we can do random direction + random mode sometimes.
        """

        if(EXPLORATION_TYPE == 0):
            # Epsilon check
            if np.random.rand() < self.epsilon:
                if self.current_heat == None :
                    # random direction in [-1,1], random mode
                    dx = np.random.uniform(-2,2)
                    dy = np.random.uniform(-2,2)
                else:
                    dx, dy = guided_random_action(self, env_model)

                return np.array([dx, dy]), True

        # Otherwise use the policy
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        # state_np는 2D 배열인데, 차원을 추가하여 모델 입력에 적합한 차원으로 만들려는 것

        with torch.no_grad():
            if deterministic:
                # 결정적 행동 선택: mean에 대해 바로 sigmoid 변환.
                mean, _ = self.policy.forward(state_t)
                action_t = 4*torch.sigmoid(mean)-2
            else:
                # 비결정적 선택: sample_action에서 샘플링 (자코비안 보정 포함)
                action_t, log_prob = self.policy.sample_action(state_t)
        action_np = action_t.cpu().numpy()[0]
        print(action_np)

        return action_np, False

    def update_gamma(self, start_gamma, end_gamma, ascent_steps, now_episode):
        self.gamma = gamma_ascent_schedule(start_gamma, end_gamma, ascent_steps, now_episode)
    
    def update_alpha(self, start_alpha, end_alpha, decay_steps, now_episode):
        self.alpha = alpha_decay_schedule(start_alpha, end_alpha, decay_steps, now_episode)



    # ------------------------------------------------- #
    # Update (one gradient step)
    # ------------------------------------------------- #
    def update(self):
        if len(self.replay_buffer) < self.batch_size*START_BATCH_TIMES:
            return
        
        # sample = self.replay_buffer.sample(self.batch_size)
        #states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 1. Replay Buffer에서 샘플 가져오기
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)


        # (B,1,H,W), (B,4), (B,), (B,1,H,W), (B,)
        # Q target:
        with torch.no_grad():
            # next action, next log prob
            next_action, next_log_prob = self.policy.sample_action(next_states) #update 할때 최적 정책으로 update -> off policy !!
            # compute target Q
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)  # (B,)
            # soft state value
            q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_prob)

        # ----- Update Q1, Q2 -----
        q1_val = self.q1(states, actions).squeeze(-1)  # (B,) #q value를 scalar 값으로
        q2_val = self.q2(states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_val, q_target) # q의 실제와 예측 차이 계산
        loss_q2 = F.mse_loss(q2_val, q_target)
        max_grad_norm = 1.0

        self.q1_optimizer.zero_grad() #이전 단계의 기울기 초기화, optimizer.step()이 호출 될때 기울기가 누적되지 않도록 함 
        loss_q1.backward()
        # 손실값으로부터 모든 파라미터에 대한 기울기 계산
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_grad_norm)      
        self.q1_optimizer.step() # q1 update
        # optimizer가 저장된 기울기(.grad)를 사용하여 네트워크의 파라미터 업데이트

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_grad_norm)
        self.q2_optimizer.step() # q2 update

        # ----- Update Policy -----
        # re-sample action from current policy
        new_action, log_prob = self.policy.sample_action(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)  # (B,)

        # policy loss = alpha * log_prob - Q
        policy_loss = (self.alpha * log_prob - q_new).mean() #여기서 self.alpha * log_prob가 entropy term
        # policy_loss는 PyTorch의 스칼라 텐서로, 자동 미분 지원 
        # 계산된 기울기는 각 파라미터의 .grad 속성에 저장

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optimizer.step()
        # optimizer가 저장된 기울기(.grad)를 사용하여 네트워크의 파라미터 업데이트

        # soft updatel
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    # def heat_loaded(self, map_id : int):
    #     self.current_map_id = map_id
    #     self.current_heat = self.heat_logger._load(map_id)

    
    # ------------------------------------------------- #
    # Save / Load
    # ------------------------------------------------- #
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

    def reset(self):
        pass

    def save_replay_buffer(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        self.replay_buffer.save(filepath)

    def load_replay_buffer(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        self.replay_buffer.load(filepath)
        print("Replay buffer loaded.")
        print("Replay buffer size:", len(self.replay_buffer))

    def load_map_info(self, map_id: int, wall_mask: np.ndarray):
        self.current_heat = self.heat_logger._load(map_id)
        self.obstacle_mask = wall_mask

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

        

##########################################################################
# Example usage in your training loop
##########################################################################
if __name__ == "__main__":
        

    import time
    import model  # Your environment code (model.FightingModel)


    # TensorBoard 로그 경로 및 total_reward.txt 경로 설정
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")

    tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    # 별도 스레드에서 total_reward.txt 모니터링 시작
    monitor_thread = threading.Thread(target=monitor_total_reward, args=(total_reward_file, tb_log_dir), daemon=True)
    monitor_thread.start()
    # 추가: 새로운 지표(80%, 100% 대피시간) 모니터링 쓰레드
    monitor_thread_80 = threading.Thread(
        target=monitor_metric, 
        args=(evacuation_time_80_file, "Evacuation Time 80", tb_log_dir),
        daemon=True
    )
    monitor_thread_80.start()

    monitor_thread_100 = threading.Thread(
        target=monitor_metric, 
        args=(evacuation_time_100_file, "Evacuation Time 100", tb_log_dir),
        daemon=True
    )
    monitor_thread_100.start()



    # hyperparams
    max_episodes = 9999999
    start_episode = 0
    
    epsilon_path = os.path.join(log_dir, "start_epsilon.txt")
    start_epsilon = 0
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
                    start_epsilon = START_EPSILON
                    start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to defaults.")
                start_epsilon = START_EPSILON
                start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
    else:
        start_epsilon = START_EPSILON
        start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
        print("No start_epsilon.txt found. Initializing values to defaults.")
    
    agent = SACAgent(input_shape=(50,50), alpha=ALPHA_START, lr=float(LR), start_epsilon=start_epsilon, start_epsilon_long = float(START_LONG_EPSILON), long_epsilon_min=float(LONG_EPSILON_MIN), batch_size=int(BATCH_SIZE), replay_size=float(BUFFER_SIZE), device=DEVICE)
    print(f"Agent initialized, lr={LR}, alpha={agent.alpha}, batch_size={BATCH_SIZE}, replay_size={BUFFER_SIZE}")
    replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

        
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "sac_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)

        if(os.path.exists(model_path)):
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

    abnormal_reward = 0
    max_steps = MAX_STEPS

    epsilon_scheduler = EpsilonScheduler(start_epsilon=start_epsilon, epsilon_min = EPSILON_MIN, start_decay_step = START_DECAY_STEP, scheduler_type=SCHEDULER_TYPE, decay_value=DECAY_VALUE, linear_decay_steps = LINEARLY_DECAY_STEP)

    for episode in range(max_episodes):
        episode_num = start_episode+episode
        print(f"Episode {episode_num}")
        # Create environment
        while True:
            try:
                number_of_agents = 0
                if (CROWD_NUMBER_MIN == CROWD_NUMBER_MAX):
                    number_of_agents = CROWD_NUMBER_MIN
                else:
                    number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
                 
                env_model = model.FightingModel(number_of_agents, 50, 50, model_num = MAP_NUM, robot = 'Q')
                break
            except Exception as e:
                print(e, "Retrying environment creation...")
            
        if(episode_num>500):
            agent.heat_loaded(env_model.map_num, env_model.wall_mask)

        state = env_model.return_current_image()
        total_reward = 0
        reward = 0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps

        buffered_state = state
        buffered_action = None
        abnormal_reward = 0
        agent.update_alpha(ALPHA_START, ALPHA_END, ALPHA_DECAY_STEPS, episode_num)
        agent.update_gamma(GAMMA_START, GAMMA_END, GAMMA_SCHEDULE_STEP, episode_num)
        try:
            for step in range(max_steps):
                # 1) Select action
            
                if(step%ACTION_SCALE==0):
                    
                    if(np.random.rand() < agent.epsilon_long):
                        env_model.robot.now_exploration = 1
                    
                    action_np, _ = agent.select_action(state, env_model = env_model)
                    dx, dy = action_np[0], action_np[1]
                    real_action = env_model.robot.receive_action([dx, dy])
                    action_np[0] = real_action[0]
                    action_np[1] = real_action[1]
                    buffered_state = state
                    buffered_action = action_np
                
                
                # Simulation time check
                sim_timer.start()
                # 2) Step environment
                env_model.step()

                if step % ACTION_SCALE == 0: #@ for heatmap
                    hx, hy = map(int, np.round(env_model.robot.xy)) 
                    heat_logger.update(env_model.map_num, hx, hy) 

                sim_timer.stop()
                reward = 0
                r_k = 0
                # 4) Next state
                next_state = env_model.return_current_image()

                # 5) Done?
                done = (step >= max_steps-1) or (env_model.robot.is_game_finished)
                if(env_model.robot.is_game_finished):
                    reward += FINISHED_BONUS * (1-step/max_steps)

                if (REWARD_K):
                    r_k += env_model.reward_penalty_collision() * REWARD_K
                # 6) Store transition
                if((step%ACTION_SCALE==(ACTION_SCALE-1) and step>ACTION_SCALE) or (env_model.robot.is_game_finished and step>ACTION_SCALE)):
                    r_a = 0
                    r_b = 0
                    r_c = 0
                    r_d = 0
                    r_e = 0
                    r_f = 0              
                    r_g = 0
                    r_h = 0
                    r_i = 0
                    r_j = 0      

                    if (REWARD_A):
                        r_a = env_model.reward_based_alived() * REWARD_A
                    if (REWARD_B):
                        r_b = env_model.reward_based_all_agents_danger() * REWARD_B
                    if (REWARD_C):
                        r_c = env_model.reward_based_gain() * REWARD_C
                    if (REWARD_D):
                        r_d = env_model.reward_penalty() * REWARD_D
                    if (REWARD_E):
                        r_e = env_model.reward_based_evacuated_with_robot() * REWARD_E
                    if (REWARD_F):
                        r_f = env_model.reward_based_distance_from_near_agents() * REWARD_F
                    if (REWARD_G):
                        r_g = env_model.reward_based_distance_from_near_agent_gain() * REWARD_G
                    if (REWARD_H):
                        r_h = env_model.reward_based_gain_with_time_bonus() * REWARD_H
                    if (REWARD_I):
                        r_i = env_model.reward_based_alived_root() * REWARD_I
                    if (REWARD_J):
                        r_j = env_model.reward_based_all_agents_danger_log() * REWARD_J
                    
                    reward += (r_a + r_b + r_c + r_d + r_e + r_g + r_h + r_i+r_j+r_k+REWARD_FIXED)
                    
                    if(SCALE_CHECK):
                        print("reward_a : ", r_a)
                        print("reward_b : ", r_b)
                        #print("reward_c : ", r_c)
                        print("reward_d : ", r_d)
                        #print("reward_e : ", r_e)
                        #print("reward f : ", r_f)
                        #print("reward g : ", r_g)
                        #print("reward h : ", r_h)
                        #print("reward i : ", r_i)
                        #print("reward j : ", r_j)
                        print("reward k : ", r_k)

                    r_k = 0

                    agent.store_transition(    
                        buffered_state,
                        buffered_action,
                        reward, 
                        next_state, 
                        float(done)
                    )
                    total_reward += reward

                    reward = 0

                # 7) Update agent
                if(step%ACTION_SCALE==(ACTION_SCALE-1) and episode_num>=START_UPDATE_STEP):
                    learn_timer.start()
                    agent.update()
                    learn_timer.stop()

                state = next_state


                if (env_model.alived_agents() < env_model.total_agents * 0.2 and evacuation_time_80 == max_steps):
                    evacuation_time_80 = step
                if (env_model.alived_agents() < 1 and evacuation_time_100 == max_steps):
                    evacuation_time_100 = step
                if done:
                    break
        except Exception as e:
            print(e)
            print("error occured. retry.")
            env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
            abnormal_reward = 1

        heat_logger.flush_episode() #@for heatmap

        # Possibly update epsilon, or do other logging

        agent.epsilon = epsilon_scheduler.get_epsilon(agent.epsilon, episode_num)
        #print("writing.. ", PORT_NUM)
        print("-----------------------------------------------")
        print("Total reward:", total_reward)
        print("now_epsilon : ", agent.epsilon)
        print("now_alpha : ", agent.alpha)
        print("now_gamma : ", agent.gamma)
        print("evacuation_time_80 : ", evacuation_time_80)
        print("evacuation_time_100 : ", evacuation_time_100)
        
        print("-----------------------------------------------")
        
        # print("now_epsilon_long : ", agent.epsilon_long)
        # Save model occasionally

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        evacuation_time_80_file_path = os.path.join(log_dir, "evacuation_80.txt")
        evacuation_time_100_file_path = os.path.join(log_dir, "evacuation_100.txt")
        if not os.path.exists(reward_file_path):
            # 파일이 없으면 빈 파일 생성
            open(reward_file_path, "w").close()
        if not os.path.exists(evacuation_time_80_file_path):
            open(evacuation_time_80_file_path, "w").close()
        if not os.path.exists(evacuation_time_100_file_path):
            open(evacuation_time_100_file_path, "w").close()

        if (episode_num) % 50 == 0:
            model_filename = os.path.join(log_dir, f"sac_checkpoint_ep_{episode_num}.pth")
            agent.save_model(model_filename)
            replay_buffer_filename = "replay_buffer.pkl"
            agent.save_replay_buffer(replay_buffer_filename)

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        with open(reward_file_path, "a") as f:
            if(abnormal_reward != 1):
                f.write(f"{total_reward}\n")

        with open(evacuation_time_80_file_path, "a") as f:
            if(abnormal_reward != 1):
                f.write(f"{evacuation_time_80}\n")
        with open(evacuation_time_100_file_path, "a") as f:
                f.write(f"{evacuation_time_100}\n")

        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon)+"\n")
            f.write(str(agent.epsilon_long))


        # each episode time print
        if ENABLE_TIMER:
            print(f"episode {episode_num} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {episode_num} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
