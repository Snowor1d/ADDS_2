import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import time
import pickle
import argparse
import threading
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser

# 타이머, 설정 import (사용자 환경에 맞추어 경로/이름 수정)
from timer_utils import Timer
from config import ENABLE_TIMER
from Start_training import (
    START_DECAY_STEP, START_EPSILON, EPSILON_MIN, SCHEDULER_TYPE,
    DECAY_VALUE, REWARD_A, REWARD_B, REWARD_C, REWARD_D, REWARD_E, REWARD_F, REWARD_G, REWARD_H, REWARD_I, REWARD_J,
    REWARD_K, CROWD_NUMBER_MIN, CROWD_NUMBER_MAX, LINEARLY_DECAY_STEP, START_UPDATE_STEP, LOG_DIR, FINISHED_BONUS, REWARD_FIXED
)

sim_timer = Timer() 
learn_timer = Timer()
home_dir = os.path.expanduser("~")

model_load = 3  # 1: start fresh, 2: load specified, 3: load latest

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--buffer_size", type=int, default=1e5)  # PPO에서는 의미가 크지 않음 (잔여)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--log_std_max", type=float, default=1)
parser.add_argument("--log_std_min", type=float, default=-0.5)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--scale_check", type=int, default=0)
parser.add_argument("--action_scale", type=float, default=3)
parser.add_argument("--start_batch_times", type=float, default=50)  # SAC에서 쓰이던 변수 (잔여)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--gamma", type=float, default=0.999)
parser.add_argument("--port_num", type=int, default=6006)
parser.add_argument("--long_epsilon_min", type=float, default=0)
parser.add_argument("--start_long_epsilon", type=float, default=0)
args = parser.parse_args()

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, LOG_DIR)
os.makedirs(log_dir, exist_ok=True)

ACTION_SCALE = args.action_scale
os.makedirs(log_dir, exist_ok=True)

SCALE_CHECK = args.scale_check
START_BATCH_TIMES = args.start_batch_times

###############################################################################
# TensorBoard 런처 및 모니터링 스레드 함수들 (기존 코드 유지)
###############################################################################
def launch_tensorboard(tb_log_dir, port=6006):
    tb_process = subprocess.Popen(["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    time.sleep(5)  # TensorBoard가 시작할 시간을 줌
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"TensorBoard launched at {url}")
    return tb_process

def monitor_metric(metric_file, metric_name, tb_log_dir):
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
                            writer.add_scalar(f"{metric_name}", value, episode)
                            print(f"Episode {episode} - {metric_name} = {value}")
                            episode += 1
                        except ValueError:
                            print(f"Invalid value in {metric_file}: {line}")
                else:
                    time.sleep(1)
        except KeyboardInterrupt:
            print(f"Monitoring for {metric_file} interrupted by user.")
        finally:
            writer.close()

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


###############################################################################
# EpsilonScheduler (원 코드 유지 - PPO에서도 사용 의존)
###############################################################################
class EpsilonScheduler:
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
            epsilon = now_epsilon * self.decay_value
            return max(epsilon, self.epsilon_min)
        elif self.scheduler_type == "l":
            fraction = min(1 / self.linear_decay_steps, 1.0)
            epsilon = now_epsilon - fraction
            return epsilon
        else:
            raise ValueError("scheduler_type must be either 'exponential' or 'linear'")


###############################################################################
# PPO를 위한 Actor-Critic 네트워크
###############################################################################
class ActorCritic(nn.Module):
    """
    - 입력: (B, 1, H, W) 형태의 state
    - 출력:
      1) 정책(행동) 분포의 mean, log_std (각 2차원: dx, dy)
      2) 상태 가치함수 V(s) (1차원)
    """
    def __init__(self, input_shape=(50,50), log_std_min=-0.5, log_std_max=1.0):
        super(ActorCritic, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # CNN 파트 (원래 PolicyNetwork 구조와 유사)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        # conv_out_size 계산
        conv_out_size = self._get_conv_out(input_shape)

        # FC 백본
        self.fc_backbone = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01, inplace=True)
        )

        # Actor head: mean, log_std
        self.mean_head = nn.Linear(64, 2)      # (dx, dy)
        self.log_std_head = nn.Linear(64, 2)   # log_std

        # Critic head: value
        self.value_head = nn.Linear(64, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        x = F.leaky_relu(self.bn1(self.conv1(dummy)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        return int(np.prod(x.size()[1:]))

    def cnn_feature(self, state):
        x = F.leaky_relu(self.bn1(self.conv1(state)), negative_slope=0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        feat = self.fc_backbone(x)
        return feat

    def forward(self, state):
        """
        return:
          mean (B,2),
          log_std (B,2),
          value (B,1)
        """
        feat = self.cnn_feature(state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        value = self.value_head(feat)
        return mean, log_std, value

    def sample_action(self, state):
        """
        상태 state에 대해 행동 a ~ pi(a|s)를 샘플링하고,
        해당 log_prob(확률의 로그), 예측 value도 함께 반환.
        """
        mean, log_std, value = self.forward(state)
        std = log_std.exp()

        # reparameterization trick
        eps = torch.randn_like(mean)
        pre_tanh = mean + std * eps  # 가우시안
        sigma = torch.sigmoid(pre_tanh)
        action = 4*sigma - 2  # [-2, 2]로 매핑

        # (가우시안 부분) log_prob
        gauss_log_prob = -0.5 * ((pre_tanh - mean) / (std + 1e-8))**2 \
                         - log_std \
                         - 0.5*np.log(2*np.pi)
        gauss_log_prob = gauss_log_prob.sum(dim=1)

        # 시그모이드 변환의 Jacobian 보정
        # (action = 4*sigma - 2 => sigma = (a+2)/4 이지만 여기서는 reparameterization된 pre_tanh 이용)
        # 미분값 = 4 * sigma*(1-sigma)
        # log(4*sigma*(1-sigma))를 더 빼줘야 하므로
        jacobian = torch.log(4*sigma*(1-sigma) + 1e-8).sum(dim=1)

        log_prob = gauss_log_prob - jacobian
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(self, states, actions):
        """
        PPO 학습 시, 저장된 (state, action)에 대해
        현재 정책 기준 log_prob, entropy, value 등을 계산하기 위한 함수.
        """
        # Forward
        mean, log_std, value = self.forward(states)
        std = log_std.exp()

        # actions는 이미 [-2,2] 범위. 이를 역연산으로 pre_tanh를 얻어야 함.
        # a = 4 * sigmoid(pre_tanh) - 2
        # => sigma = (a + 2) / 4
        # => pre_tanh = log(sigma/(1-sigma))
        sigma = (actions + 2) / 4.0
        sigma = torch.clamp(sigma, 1e-7, 1 - 1e-7)
        pre_tanh = torch.log(sigma/(1-sigma))

        # (pre_tanh - mean)의 가우시안 log_prob
        gauss_log_prob = -0.5 * (((pre_tanh - mean)/(std+1e-8))**2) \
                         - log_std \
                         - 0.5*np.log(2*np.pi)
        gauss_log_prob = gauss_log_prob.sum(dim=1)

        # sigmoid Jacobian
        jacobian = torch.log(4*sigma*(1-sigma) + 1e-8).sum(dim=1)
        log_prob = gauss_log_prob - jacobian

        entropy = 0.5 * (2*np.pi*np.e*(std**2)).sum(dim=1)  # 일반 가우시안 근사 Entropy (추가 보정 가능)
        # 여기서는 sigmoid 변환으로 약간 다를 수 있으나, 간단히 std 기반의 가우시안 엔트로피로 계산

        return log_prob, entropy, value.squeeze(-1)


###############################################################################
# On-Policy Trajectory 버퍼 (PPO 용도)
###############################################################################
class PPORolloutBuffer:
    """
    한 에피소드(or 여러 step) 동안의 transition들을 저장했다가,
    episode가 끝날 때 한 번에 PPO 업데이트에 사용
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []


###############################################################################
# PPO Agent
###############################################################################
class PPOAgent:
    def __init__(self,
                 input_shape=(50,50),
                 gamma=0.99,
                 lr=1e-4,
                 clip_range=0.2,
                 n_epochs=10,
                 device="cpu",
                 start_epsilon=1.0,
                 start_epsilon_long=0.1,
                 long_epsilon_min=0):
        self.gamma = gamma
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.device = torch.device(device)

        self.epsilon = start_epsilon
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = long_epsilon_min

        self.rollout_buffer = PPORolloutBuffer()

        self.ac = ActorCritic(input_shape=input_shape,
                              log_std_min=args.log_std_min,
                              log_std_max=args.log_std_max).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def select_action(self, state_np, deterministic=False):
        """
        state_np: (H, W)
        - epsilon 탐색: 일정 확률로 임의 행동
        - 그 외에는 policy 기반 행동 샘플링
        """
        if np.random.rand() < self.epsilon:
            # random
            dx = np.random.uniform(-2, 2)
            dy = np.random.uniform(-2, 2)
            return np.array([dx, dy]), True, 0.0  # log_prob=0(임의로)
        
        # PPO 정책 사용
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, log_std, value = self.ac.forward(state_t)
                std = log_std.exp()
                # 결정적: mean만 사용
                pre_tanh = mean
                sigma = torch.sigmoid(pre_tanh)
                action_t = 4*sigma - 2
                # log_prob은 구해줘야 하지만, 여기서는 deterministic이라 0으로 둬도 됨
                log_prob = 0.0
            else:
                action_t, log_prob_t, value_t = self.ac.sample_action(state_t)

        action_np = action_t.cpu().numpy()[0]  # (2,)
        if not deterministic:
            log_prob = log_prob_t.cpu().item()
            value = value_t.cpu().item()
        else:
            value = 0.0
        
        return action_np, False, (log_prob, value)

    def store_transition(self, s, a, r, done, log_prob, value):
        """PPO는 (state, action, reward, done, log_prob, value)를 저장."""
        self.rollout_buffer.store(s, a, r, done, log_prob, value)

    def finish_trajectory(self):
        """
        한 에피소드가 끝나면(혹은 특정 스텝마다) Advantage, Return 계산을 위해
        trajectory를 정리하고 PPO 업데이트에 쓸 준비.
        """
        pass  # 간단 구현: update()에서 한 번에 계산

    def update(self):
        """
        PPO 업데이트: rollout_buffer에 쌓인 한 에피소드 데이터를 사용.
        1) Return, Advantage 계산
        2) 여러 epoch 동안 미니배치 학습 (여기서는 전체 배치 한번에 학습 예시)
        """
        # 모든 transition을 tensor로 변환
        states = torch.FloatTensor(np.array(self.rollout_buffer.states)).unsqueeze(1).to(self.device)   # (T,1,H,W)
        actions = torch.FloatTensor(np.array(self.rollout_buffer.actions)).to(self.device)              # (T,2)
        rewards = np.array(self.rollout_buffer.rewards)                                                 # (T,)
        dones = np.array(self.rollout_buffer.dones)                                                     # (T,)
        old_log_probs = torch.FloatTensor(np.array(self.rollout_buffer.log_probs)).to(self.device)       # (T,)
        values = torch.FloatTensor(np.array(self.rollout_buffer.values)).to(self.device)                 # (T,)

        # 1) Return, Advantage 계산 (간단 버전)
        # 에피소드 단위 가정 -> 마지막은 done = True
        returns = []
        gae = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            # next_value = 0 if done, else value of next state
            next_value = values[step+1] if (step+1 < len(values)) else 0
            delta = rewards[step] + self.gamma * next_value * mask - values[step]
            gae = delta + self.gamma * 0.95 * mask * gae  # lam=0.95
            returns.insert(0, gae + values[step])
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - values

        # 2) 여러 epoch 반복 학습
        for _ in range(self.n_epochs):
            # 한번에 전체 배치를 사용 (간단화)
            log_probs, entropy, new_values = self.ac.evaluate_actions(states, actions)
            ratio = torch.exp(log_probs - old_log_probs)  # pi(a|s) / pi_old(a|s)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = policy_loss + 0.5*value_loss  # value_loss 가중치 0.5 예시

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
            self.optimizer.step()

        # rollout buffer 비우기
        self.rollout_buffer.clear()

    def save_model(self, filepath):
        torch.save({
            'actor_critic': self.ac.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        filepath = os.path.join(log_dir, filepath)
        ckpt = torch.load(filepath)
        self.ac.load_state_dict(ckpt['actor_critic'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Model loaded from {filepath}")

    def reset(self):
        pass


###############################################################################
# 메인 학습 루프
###############################################################################
if __name__ == "__main__":
    import model  # 사용자가 만든 환경 (model.FightingModel)

    # TensorBoard 로그 경로
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")

    tb_process = launch_tensorboard(tb_log_dir, port=args.port_num)

    # 모니터링 스레드 시작
    monitor_thread = threading.Thread(
        target=monitor_total_reward, args=(total_reward_file, tb_log_dir), daemon=True
    )
    monitor_thread.start()

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

    max_episodes = 9999999
    start_episode = 0

    # epsilon 불러오기
    epsilon_path = os.path.join(log_dir, "start_epsilon.txt")
    start_epsilon = START_EPSILON
    start_epsilon_long = args.start_long_epsilon

    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            lines = f.readlines()
            if len(lines) >= 2:
                try:
                    start_epsilon = float(lines[0].strip())
                    start_epsilon_long = float(lines[1].strip())
                    print(f"Loaded start_epsilon: {start_epsilon}, start_epsilon_long: {start_epsilon_long}")
                except:
                    print("Invalid value in start_epsilon.txt. Using default.")
            else:
                print("Not enough lines in start_epsilon.txt. Using default.")
    else:
        print("No start_epsilon.txt found. Using defaults.")

    # PPO Agent 생성
    agent = PPOAgent(
        input_shape=(50,50),
        gamma=args.gamma,
        lr=args.lr,
        device=args.device,
        start_epsilon=start_epsilon,
        start_epsilon_long=start_epsilon_long
    )
    print(f"PPO Agent initialized. lr={args.lr}, batch_size={args.batch_size}")

    # 모델 / 버퍼 로드 (PPO에선 off-policy 버퍼가 없으므로, SAC용 replay buffer 로드는 무의미. 그대로 제거)
    if model_load == 1:
        pass  # start fresh
    elif model_load == 2:
        print("load specified model")
        model_name = "ppo_checkpoint_ep_200.pth"  # 예시
        model_path = os.path.join(log_dir, model_name)
        if os.path.exists(model_path):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_name)
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir)
                       if f.startswith("ppo_checkpoint") and f.endswith(".pth")]
        if model_files:
            latest_model = max(model_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            latest_model_path = os.path.join(log_dir, latest_model)
            start_episode = int(latest_model.split("_")[-1].split(".")[0])
            print(f"Loading latest model: {latest_model}")
            agent.load_model(latest_model_path)

    abnormal_reward = 0
    max_steps = args.max_steps

    # Epsilon 스케줄러 (기존 유지)
    epsilon_scheduler = EpsilonScheduler(
        start_epsilon=start_epsilon,
        epsilon_min=EPSILON_MIN,
        start_decay_step=START_DECAY_STEP,
        scheduler_type=SCHEDULER_TYPE,
        decay_value=DECAY_VALUE,
        linear_decay_steps=LINEARLY_DECAY_STEP
    )

    for episode in range(max_episodes):
        print(f"Episode {start_episode + episode + 1}")

        # 환경 생성
        while True:
            try:
                if (CROWD_NUMBER_MIN == CROWD_NUMBER_MAX):
                    number_of_agents = CROWD_NUMBER_MIN
                else:
                    number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
                env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
                break
            except Exception as e:
                print(e, "Retrying environment creation...")

        state = env_model.return_current_image()
        total_reward = 0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps

        buffered_state = state
        buffered_action = None
        buffered_logprob = 0.0
        buffered_value = 0.0
        abnormal_reward = 0

        #try:
        for step in range(max_steps):
            # exploration
            if np.random.rand() < agent.epsilon_long:
                env_model.robot.now_exploration = 1

            # ACTION_SCALE에 따라 행동 갱신
            if (step % ACTION_SCALE == 0):
                action_np, is_random, extra_info = agent.select_action(state)
                # extra_info = (log_prob, value)
                if not is_random and extra_info != 0.0:
                    buffered_logprob, buffered_value = extra_info
                else:
                    buffered_logprob, buffered_value = 0.0, 0.0

                dx, dy = action_np[0], action_np[1]
                real_action = env_model.robot.receive_action([dx, dy])
                action_np[0] = real_action[0]
                action_np[1] = real_action[1]
                buffered_state = state
                buffered_action = action_np

            # 환경 스텝
            sim_timer.start()
            env_model.step()
            sim_timer.stop()

            reward = 0
            next_state = env_model.return_current_image()
            done = (step >= max_steps - 1) or (env_model.robot.is_game_finished)
            if env_model.robot.is_game_finished:
                reward += FINISHED_BONUS * (1 - step / max_steps)

            # 보상 설계
            if ((step % ACTION_SCALE == (ACTION_SCALE - 1) and step > ACTION_SCALE)
                or (env_model.robot.is_game_finished and step > ACTION_SCALE)):
                r_a = env_model.reward_based_alived() * REWARD_A if REWARD_A else 0
                r_b = env_model.reward_based_all_agents_danger() * REWARD_B if REWARD_B else 0
                r_c = env_model.reward_based_gain() * REWARD_C if REWARD_C else 0
                r_d = env_model.reward_penalty() * REWARD_D if REWARD_D else 0
                r_e = env_model.reward_based_evacuated_with_robot() * REWARD_E if REWARD_E else 0
                r_f = env_model.reward_based_distance_from_near_agents() * REWARD_F if REWARD_F else 0
                r_g = env_model.reward_based_distance_from_near_agent_gain() * REWARD_G if REWARD_G else 0
                r_h = env_model.reward_based_gain_with_time_bonus() * REWARD_H if REWARD_H else 0
                r_i = env_model.reward_based_alived_root() * REWARD_I if REWARD_I else 0
                r_j = env_model.reward_based_all_agents_danger_log() * REWARD_J if REWARD_J else 0
                r_k = 0  # REWARD_K 사용 시 추가

                reward += (r_a + r_b + r_c + r_d + r_e + r_f + r_g + r_h + r_i + r_j + REWARD_FIXED)
                if SCALE_CHECK:
                    print(f"reward_a:{r_a}, reward_b:{r_b}, reward_c:{r_c}, reward_d:{r_d}, reward_e:{r_e}, reward_f:{r_f}, reward_g:{r_g}, reward_h:{r_h}, reward_i:{r_i}, reward_j:{r_j}")

                total_reward += reward

                # PPO 버퍼에 저장
                agent.store_transition(
                    buffered_state,
                    buffered_action,
                    reward,
                    float(done),
                    buffered_logprob,
                    buffered_value
                )
                reward = 0

            state = next_state

            if (env_model.alived_agents() < env_model.total_agents * 0.2 and evacuation_time_80 == max_steps):
                evacuation_time_80 = step
            if (env_model.alived_agents() < 1 and evacuation_time_100 == max_steps):
                evacuation_time_100 = step
            if done:
                break

        # except Exception as e:
        #     print(e)
        #     print("error occurred. retry.")
        #     env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
        #     abnormal_reward = 1

        # 에피소드 종료 후 PPO 업데이트
        agent.finish_trajectory()  # 여기서는 아무것도 안 하지만 구조상 포함
        agent.update()

        # epsilon decay
        agent.epsilon = epsilon_scheduler.get_epsilon(agent.epsilon, episode + start_episode)
        print("Total reward:", total_reward)
        print("now_epsilon:", agent.epsilon)
        print("evacuation_time_80:", evacuation_time_80)
        print("evacuation_time_100:", evacuation_time_100)

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        evacuation_time_80_file_path = os.path.join(log_dir, "evacuation_80.txt")
        evacuation_time_100_file_path = os.path.join(log_dir, "evacuation_100.txt")
        if not os.path.exists(reward_file_path):
            open(reward_file_path, "w").close()
        if not os.path.exists(evacuation_time_80_file_path):
            open(evacuation_time_80_file_path, "w").close()
        if not os.path.exists(evacuation_time_100_file_path):
            open(evacuation_time_100_file_path, "w").close()

        # 모델 주기적 저장
        if (episode + 1) % 50 == 0:
            model_filename = os.path.join(log_dir, f"ppo_checkpoint_ep_{start_episode + episode + 1}.pth")
            agent.save_model(model_filename)

        # 보상/대피시간 기록
        if abnormal_reward != 1:
            with open(reward_file_path, "a") as f:
                f.write(f"{total_reward}\n")
            with open(evacuation_time_80_file_path, "a") as f:
                f.write(f"{evacuation_time_80}\n")
            with open(evacuation_time_100_file_path, "a") as f:
                f.write(f"{evacuation_time_100}\n")

        # epsilon 파일에 저장
        epsilon_path = os.path.join(log_dir, "start_epsilon.txt")
        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon) + "\n")
            f.write(str(agent.epsilon_long))

        # 타이머 로그
        if ENABLE_TIMER:
            print(f"episode {start_episode + episode + 1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode + episode + 1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
