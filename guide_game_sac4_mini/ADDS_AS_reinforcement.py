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
from Start_training import log_dir

# Timer instances
sim_timer = Timer() 
learn_timer = Timer()
home_dir = os.path.expanduser("~")

model_load = 3
# start_fresh : 1
# load specified model : 2
# load latest model : 3

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=2e-4) ## 1e-4 → 2e-4로 조정
parser.add_argument("--decay_value", type=float, default=0.99)
parser.add_argument("--buffer_size", type=int, default=1e5)
parser.add_argument("--batch_size", type=float, default=64)
parser.add_argument("--start_epsilon", type=float, default=1.0)
parser.add_argument("--epsilon_min", type=float, default=0.1)
parser.add_argument("--log_dir", type=str, default=log_dir)
parser.add_argument("--log_std_max", type=float, default=1)
parser.add_argument("--log_std_min", type=float, default=-0.5)
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--reward_A", type=float, default=0)
parser.add_argument("--reward_B", type=float, default=0)
parser.add_argument("--reward_C", type=float, default=0)
parser.add_argument("--reward_D", type=float, default=0)
parser.add_argument("--reward_E", type=float, default=0)
parser.add_argument("--reward_F", type=float, default=0)
parser.add_argument("--reward_G", type=float, default=0)
parser.add_argument("--reward_H", type=float, default=0)
parser.add_argument("--reward_I", type=float, default=0)
parser.add_argument("--reward_J", type=float, default=0)
parser.add_argument("--finished_bonus", type=float, default=0)
parser.add_argument("--scale_check", type=int, default=0)
parser.add_argument("--action_scale", type=float, default=3)
parser.add_argument("--start_batch_times", type=float, default=50)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--crowd_number", type=int, default=20)
parser.add_argument("--gamma", type=float, default=0.99)  # 0.999 → 0.99로 조정
parser.add_argument("--port_num", type=int, default=6006)
parser.add_argument("--long_epsilon_min", type=float, default=0)
parser.add_argument("--start_long_epsilon", type=float, default=0)
args = parser.parse_args()

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, args.log_dir)
os.makedirs(log_dir, exist_ok=True)

ACTION_SCALE = args.action_scale
os.makedirs(log_dir, exist_ok=True)

REWARD_A = args.reward_A
REWARD_B = args.reward_B
REWARD_C = args.reward_C
REWARD_D = args.reward_D
REWARD_E = args.reward_E
REWARD_F = args.reward_F
REWARD_G = args.reward_G
REWARD_H = args.reward_H
REWARD_I = args.reward_I
REWARD_J = args.reward_J
FINISHED_BONUS = args.finished_bonus
SCALE_CHECK = args.scale_check
START_BATCH_TIMES = args.start_batch_times

def launch_tensorboard(tb_log_dir, port=6006):
    """
    TensorBoard를 백그라운드에서 실행하고 기본 브라우저에 해당 URL을 엽니다.
    """
    tb_process = subprocess.Popen(["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    time.sleep(5)  # TensorBoard가 시작할 시간을 줌
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"TensorBoard launched at {url}")
    return tb_process

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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1).to(device)  # (B,1,H,W)
        actions     = torch.FloatTensor(actions).to(device)              # (B,action_dim)
        rewards     = torch.FloatTensor(rewards).to(device)              # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
        dones       = torch.FloatTensor(dones).to(device)                # (B,)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)
    
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)

##########################################################################
# 3) Critic (Q) Network
##########################################################################
class QNetwork(nn.Module):
    def __init__(self, input_shape=(50,50), action_dim=2):
        super(QNetwork, self).__init__()
        # 3단계 CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Conv 결과 크기 파악
        conv_out_size = self._get_conv_out(input_shape)
        
        # FC: conv 특징 + action_dim → Q-value
        self.fc1 = nn.Linear(conv_out_size + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, shape[0], shape[1])
        x = self.conv1(dummy)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        return int(np.prod(x.size()))

    def forward(self, state, action):
        x = self.leaky_relu(self.conv1(state))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = torch.cat([x, action], dim=1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        q_val = self.q_out(x)
        return q_val

##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape=(50,50)):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = args.log_std_min
        self.log_std_max = args.log_std_max

        # 3단계 CNN
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        conv_out_size = self._get_conv_out(input_shape)
        
        # FC
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # mean, log_std
        self.mean_head = nn.Linear(512, 2)
        self.log_std_head = nn.Linear(512, 2)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, shape[0], shape[1])
        x = self.conv1(dummy)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        return int(np.prod(x.size()))

    def forward(self, state):
        x = self.leaky_relu(self.conv1(state))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, state, temperature=1.0):
        B = state.size(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        eps = torch.randn_like(mean) * temperature
        u = mean + std * eps
        sigma = torch.sigmoid(u)
        action = 4 * sigma - 2  # [-2, 2] 범위로 매핑
        
        # 가우시안 로그확률
        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2 * log_std + np.log(2*np.pi))
        log_prob_u = log_prob_u.sum(dim=1)
        
        # 시그모이드 야코비안 보정
        jacobian = torch.log(4 * sigma * (1 - sigma) + 1e-8).sum(dim=1)
        log_prob = log_prob_u - jacobian

        return action, log_prob

##########################################################################
# 5) SAC Agent for Action
##########################################################################
class SACAgent:
    def __init__(self, input_shape=(50,50), gamma=args.gamma, alpha=0.2, tau=0.995, lr=1e-4,
                 batch_size=64, replay_size=int(1e5), device="cpu",
                 start_epsilon=1.0, start_epsilon_long=0.1, long_epsilon_min=0):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = long_epsilon_min
        self.epsilon_min = args.epsilon_min

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=int(replay_size))

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

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(
                self.tau * target_param.data + (1 - self.tau) * param.data
            )

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

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.push(s, a, r, s_next, done)

    def select_action(self, state_np, deterministic=False):
        # Epsilon 탐험
        if np.random.rand() < self.epsilon:
            dx = np.random.uniform(-2, 2)
            dy = np.random.uniform(-2, 2)
            return np.array([dx, dy]), True
        
        # SAC Policy 사용
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy.forward(state_t)
                action_t = 4 * torch.sigmoid(mean) - 2
            else:
                action_t, log_prob = self.policy.sample_action(state_t)
        action_np = action_t.cpu().numpy()[0]
        return action_np, False

    def update(self):
        if len(self.replay_buffer) < self.batch_size * START_BATCH_TIMES:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample_action(next_states)
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_prob)

        # Update Q1, Q2
        q1_val = self.q1(states, actions).squeeze(-1)
        q2_val = self.q2(states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)
        max_grad_norm = 1.0

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_grad_norm)
        self.q2_optimizer.step()

        # Update Policy
        new_action, log_prob = self.policy.sample_action(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
        self.policy_optimizer.step()

        # Soft update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

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

##########################################################################
# TensorBoard 모니터링 함수: total_reward.txt 파일의 새 라인을 지속적으로 읽어 기록
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

##########################################################################
# Training loop
##########################################################################
if __name__ == "__main__":
    import time
    import model  # Your environment code (model.FightingModel)

    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")

    tb_process = launch_tensorboard(tb_log_dir, port=args.port_num)
    monitor_thread = threading.Thread(target=monitor_total_reward, args=(total_reward_file, tb_log_dir), daemon=True)
    monitor_thread.start()

    max_episodes = 9999999
    start_episode = 0
    epsilon_path = os.path.join(log_dir, "start_epsilon.txt")
    
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
                    start_epsilon_long = args.start_long_epsilon
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to defaults.")
                start_epsilon = args.start_epsilon
                start_epsilon_long = args.start_long_epsilon
    else:
        start_epsilon = args.start_epsilon
        start_epsilon_long = args.start_long_epsilon
        print("No start_epsilon.txt found. Initializing values to defaults.")
    
    agent = SACAgent(
        input_shape=(50,50), 
        alpha=args.alpha, 
        lr=float(args.lr), 
        start_epsilon=start_epsilon, 
        start_epsilon_long=float(args.start_long_epsilon), 
        long_epsilon_min=float(args.long_epsilon_min),
        batch_size=int(args.batch_size), 
        replay_size=float(args.buffer_size), 
        device=args.device
    )
    print(f"Agent initialized, lr={args.lr}, alpha={agent.alpha}, batch_size={args.batch_size}, replay_size={args.buffer_size}")
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
    number_of_agents = args.crowd_number
    max_steps = args.max_steps

    for episode in range(max_episodes):
        print(f"Episode {start_episode + episode + 1}")
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
            for step in range(args.max_steps):
                if(step % ACTION_SCALE == 0):
                    if(np.random.rand() < agent.epsilon_long):
                        env_model.robot.now_exploration = 1
                    
                    action_np, _ = agent.select_action(state)
                    dx, dy = action_np[0], action_np[1]
                    real_action = env_model.robot.receive_action([dx, dy])
                    action_np[0] = real_action[0]
                    action_np[1] = real_action[1]
                    buffered_state = state
                    buffered_action = action_np

                sim_timer.start()
                env_model.step()
                sim_timer.stop()
                reward = 0
                next_state = env_model.return_current_image()

                done = (step >= args.max_steps - 1) or (env_model.robot.is_game_finished)
                if(env_model.robot.is_game_finished):
                    reward += FINISHED_BONUS

                if ((step % ACTION_SCALE == (ACTION_SCALE - 1) and step > ACTION_SCALE) or (env_model.robot.is_game_finished)):
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
                        r_j = env_model.reward_based_distance_from_all_agents() * REWARD_J
                    
                    reward += (r_a + r_b + r_c + r_d + r_e + r_g + r_h + r_i + r_j)

                    if(SCALE_CHECK):
                        print("reward_a : ", r_a)
                        print("reward_b : ", r_b)
                        print("reward_c : ", r_c)
                        print("reward_d : ", r_d)
                        print("reward_e : ", r_e)
                        print("reward f : ", r_f)
                        print("reward g : ", r_g)
                        print("reward h : ", r_h)
                        print("reward i : ", r_i)
                        print("reward j : ", r_j)

                    agent.store_transition(
                        buffered_state,
                        buffered_action,
                        reward,
                        next_state,
                        float(done)
                    )
                    total_reward += reward
                    reward = 0

                if(step % ACTION_SCALE == (ACTION_SCALE - 1)):
                    learn_timer.start()
                    agent.update()
                    learn_timer.stop()

                state = next_state
                if done:
                    break
        
        except Exception as e:
            print(e)
            print("error occured. retry.")
            env_model = model.FightingModel(number_of_agents, 50, 50, 2, 'Q')
            abnormal_reward = 1

        # Epsilon 업데이트
        decay_value = args.decay_value
        if (len(agent.replay_buffer) > agent.batch_size * START_BATCH_TIMES):
            agent.update_epsilon(True, args.decay_value)
            agent.update_epsilon2(True, args.decay_value)

        print("Total reward:", total_reward)
        print("now_epsilon:", agent.epsilon)
        print("now_epsilon_long:", agent.epsilon_long)

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

        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon)+"\n")
            f.write(str(agent.epsilon_long))

        if ENABLE_TIMER:
            print(f"episode {start_episode+episode+1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode+episode+1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
