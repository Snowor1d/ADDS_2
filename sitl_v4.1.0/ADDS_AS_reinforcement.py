#########################################################
# ADDS_AS_reinforcement.py
#   - 버전2: Truncated IS + Retrace + Bias Correction
#            완전한 ACER 알고리즘 예시
#########################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import time
import pickle
import argparse

from timer_utils import Timer
from config import ENABLE_TIMER

# Timer instances
sim_timer = Timer() 
learn_timer = Timer()

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_v4.1.1")
os.makedirs(log_dir, exist_ok=True)

model_load = 3
# start_fresh : 1
# load specified model : 2
# load latest model : 3

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay_value", type=float, default=0.99)
parser.add_argument("--buffer_size", type=int, default=1e5)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

#########################################################
# 0) Discrete Action 매핑 함수
#########################################################
def int_action_to_dxdy(a):
    # 0: Up, 1: Down, 2: Left, 3: Right
    if a == 0:
        return (0, -2)
    elif a == 1:
        return (0,  2)
    elif a == 2:
        return (-2, 0)
    elif a == 3:
        return (2,  0)
    else:
        return (0,0)

#########################################################
# 1) Replay Buffer
#    - ACER에서는 behavior policy 확률(또는 logits)을 저장해야 함
#########################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e5), device=None):
        self.buffer = deque(maxlen=capacity)
        self.device = device

    def push(self, state, action, reward, next_state, done, behavior_probs):
        """
        behavior_probs: 행동했던 시점의 π_behavior(a|s)
                        shape: (4,) or logits(4,)
        """
        self.buffer.append((state, action, reward, next_state, done, behavior_probs))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, beh_probs = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1).to(device)  # (B,1,H,W)
        actions     = torch.LongTensor(actions).to(device)               # (B,)
        rewards     = torch.FloatTensor(rewards).to(device)              # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
        dones       = torch.FloatTensor(dones).to(device)                # (B,)

        # behavior_probs -> (B,4)
        beh_probs   = torch.FloatTensor(beh_probs).to(device)
        return states, actions, rewards, next_states, dones, beh_probs

    def __len__(self):
        return len(self.buffer)

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self.buffer, f)
    def load(self, filepath):
        with open(filepath, "rb") as f:
            self.buffer = pickle.load(f)


#########################################################
# 2) Q-Network (Critic)
#########################################################
class QNetworkDiscrete(nn.Module):
    def __init__(self, input_shape=(70,70), num_actions=4):
        super(QNetworkDiscrete, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_out = nn.Linear(128, num_actions)  # (B,4)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1,1,*shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.q_out(x)  # (B,4)
        return q_values


#########################################################
# 3) Policy Network (Actor)
#########################################################
from torch.distributions import Categorical

class PolicyNetworkDiscrete(nn.Module):
    def __init__(self, input_shape=(70,70), num_actions=4):
        super(PolicyNetworkDiscrete, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.logits = nn.Linear(128, num_actions)  # (B,4)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.logits(x)  # (B,4)
        return out

    def get_probs(self, state):
        """ return softmax probs (B,4) """
        logits = self.forward(state)
        return F.softmax(logits, dim=-1)


#########################################################
# 4) ACER Agent (완전 ACER)
#########################################################
class FullACERAgent:
    def __init__(self, input_shape=(70,70), gamma=0.99,
                 lr=1e-4, batch_size=64, replay_size=int(1e5), 
                 device="cpu", start_epsilon=1.0, c=10.0):
        
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_min = 0.1

        # Truncated IS clipping
        self.c = c

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size, device=self.device)

        # Networks
        self.q_network = QNetworkDiscrete(input_shape, num_actions=4).to(self.device)
        self.q_target  = QNetworkDiscrete(input_shape, num_actions=4).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        self.policy_network = PolicyNetworkDiscrete(input_shape, num_actions=4).to(self.device)

        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        # Soft update 계수
        self.tau = 0.995

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

    def store_transition(self, s, a, r, s_next, done, behavior_probs):
        """
        behavior_probs: 현재 policy가 (s)에서 액션별로 갖는 확률(softmax)
        -> 실제 행동 a에 대한 behavior policy prob = behavior_probs[a].
        """
        self.replay_buffer.push(s, a, r, s_next, done, behavior_probs)

    def select_action(self, state_np, deterministic=False):
        # e-greedy + Policy 샘플링
        if np.random.rand() < self.epsilon:
            # 완전 랜덤
            action = np.random.randint(0,4)
            # behavior_probs(4차원)를 uniform으로 가정
            behavior_probs = np.ones(4)/4.0
            return action, behavior_probs
        else:
            # policy network
            state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.policy_network.get_probs(state_t)  # (1,4)
            probs_np = probs.cpu().numpy().reshape(-1)  # (4,)

            if deterministic:
                action = np.argmax(probs_np)
            else:
                action = np.random.choice(4, p=probs_np)
            return action, probs_np

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 샘플링
        states, actions, rewards, next_states, dones, beh_probs = self.replay_buffer.sample(self.batch_size, self.device)

        # 현재 policy 확률
        current_probs = self.policy_network.get_probs(states)  # (B,4)
        pi_a = current_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # behavior policy가 선택한 a의 확률
        beh_a = beh_probs.gather(1, actions.unsqueeze(1)).squeeze(1)     # (B,)

        # importance weight
        # rho = pi_current(a|s) / pi_behavior(a|s), clipped by c
        rho = torch.min(pi_a / (beh_a + 1e-8), torch.tensor(self.c, device=self.device))

        # Critic target (Retrace 1-step)
        with torch.no_grad():
            # Q'(s') from target network
            q_next = self.q_target(next_states)  # (B,4)
            # V(s') = Σ_a pi(a|s') * Q(s',a)
            next_probs = self.policy_network.get_probs(next_states)  # (B,4)
            v_next = (next_probs * q_next).sum(dim=1)  # (B,)

            # Q(s', a') 중 실제로 한 a'가 아니라, Retrace를 위해 아래 항 구성
            # Retrace target = r + gamma * [ rho*( Q(s', a') - V(s') ) + V(s') ]
            # 여기서는 "행동 a' = actions' "을 1-step만 가정
            # 실제 ACER는 n-step, 여러 transition을 연결
            # 여기서는 단순히 1-step만 예시
            # "actions'"는 replay에서 next action... (이 예시는 단순화)

            # 여기서는 "max Q(s',a') 또는 policy-value"를 쓰는 대신, 논문 식에 따라:
            # Q^retrace(s,a) = r + gamma * [ rho*( Q(s', a') - V(s') ) + V(s') ]
            # 단, a'은 next에서 샘플한 것이 아닌, off-policy -> single-step 시 a' = actions'(?) 
            # => n-step 구현이 필요하지만, 예시로 간단화 (a'가 next step의 실제 action이라고 가정)
            # => 실제로는 batch에 next_action도 저장해서 처리.

            # 일단 간단히 DQN 스타일 + rho항을 섞어서 예시
            # => r + gamma * maxQ(s') 부분을 retrace 형태로 치환:
            # => r + gamma * [rho*(maxQ(s') - v(s')) + v(s') ]
            max_q_next, _ = torch.max(q_next, dim=1)  # (B,)
            retrace_target = rewards + self.gamma * (
                rho * (max_q_next - v_next) + v_next
            )

        # --- Critic Update ---
        q_vals = self.q_network(states)          # (B,4)
        q_acted = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        critic_loss = F.mse_loss(q_acted, retrace_target)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # --- Policy Update ---
        # Advantage = Q(s,a) - V(s)
        q_vals_detach = q_vals.detach()
        v_s = (current_probs * q_vals_detach).sum(dim=1)  # (B,)
        adv = q_vals_detach.gather(1, actions.unsqueeze(1)).squeeze(1) - v_s

        # Policy gradient에 rho 적용
        # ACER Policy Loss = - E[ rho * log(pi(a|s)) * Advantage ]
        # + Bias correction = E[ (1-rho) * Q(s,a) ]
        log_pi_a = torch.log(pi_a + 1e-8)
        policy_loss = -(rho * log_pi_a * adv).mean()
        bias_correction = ((1.0 - rho) * q_acted).mean()

        total_loss = critic_loss + policy_loss + bias_correction

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        self.policy_optimizer.step()

        # soft update
        self.soft_update(self.q_network, self.q_target)

    def save_model(self, filepath):
        torch.save({
            'q': self.q_network.state_dict(),
            'q_target': self.q_target.state_dict(),
            'policy': self.policy_network.state_dict(),
            'q_opt': self.q_optimizer.state_dict(),
            'policy_opt': self.policy_optimizer.state_dict()
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        ckpt = torch.load(filepath)
        self.q_network.load_state_dict(ckpt['q'])
        self.q_target.load_state_dict(ckpt['q_target'])
        self.policy_network.load_state_dict(ckpt['policy'])
        self.q_optimizer.load_state_dict(ckpt['q_opt'])
        self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
        print(f"Model loaded from {filepath}")

    def save_replay_buffer(self, filepath):
        self.replay_buffer.save(filepath)

    def load_replay_buffer(self, filepath):
        self.replay_buffer.load(filepath)
        print("Replay buffer loaded.")
        print("Replay buffer size:", len(self.replay_buffer))


#########################################################
# Main training loop
#########################################################
if __name__ == "__main__":
    import model  # Your environment: model.FightingModel

    max_episodes = 1500
    max_steps = 1500
    number_of_agents = 30
    start_episode = 0

    epsilon_path = os.path.join(log_dir, "start_epsilon.txt")
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            try:
                start_epsilon = float(f.read().strip())
                print(f"Loaded start_epsilon: {start_epsilon}")
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to 1.0")
                start_epsilon = 1.0
    else:
        start_epsilon = 1.0
        print("No start_epsilon.txt found. Initializing start_epsilon to 1.0")

    agent = FullACERAgent(
        input_shape=(70,70),
        gamma=0.99,
        lr=args.lr,
        batch_size=int(args.batch_size),
        replay_size=int(args.buffer_size),
        device="cpu",
        start_epsilon=start_epsilon,
        c=10.0  # Truncated IS clip
    )
    print(f"ACER Agent initialized, lr={args.lr}, batch_size={agent.batch_size}, replay_size={args.buffer_size}")

    replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

    # 모델 로드
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "acer_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)
        if(os.path.exists(model_path)):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_path)
            if os.path.exists(replay_buffer_path):
                agent.load_replay_buffer(replay_buffer_path)
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir) if f.startswith("acer_checkpoint") and f.endswith(".pth")]
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

    # 훈련 루프
    for episode in range(max_episodes):
        print(f"Episode {start_episode+episode+1}")
        while True:
            try:
                env_model = model.FightingModel(number_of_agents, 70, 70, 2, 'Q')
                break
            except Exception as e:
                print(e, "Retrying environment creation...")

        state = env_model.return_current_image()
        total_reward = 0
        reward_acc = 0
        buffered_state = state
        buffered_action = None
        buffered_probs = None  # behavior policy 확률

        for step in range(max_steps):
            if step % 3 == 0:
                action_int, beh_probs = agent.select_action(state)
                dx, dy = int_action_to_dxdy(action_int)
                env_model.robot.receive_action([dx, dy])
                buffered_state = state
                buffered_action = action_int
                buffered_probs = beh_probs

            sim_timer.start()
            env_model.step()
            sim_timer.stop()

            reward_acc += env_model.reward_based_gain()
            total_reward += reward_acc

            next_state = env_model.return_current_image()
            done = (step >= max_steps-1) or (env_model.alived_agents() <= 1)

            if step % 3 == 2:
                # Transition 저장
                agent.store_transition(
                    buffered_state,
                    buffered_action,
                    reward_acc,
                    next_state,
                    float(done),
                    buffered_probs
                )
                print("reward : ", reward_acc)
                reward_acc = 0

                # 학습
                learn_timer.start()
                agent.update()
                learn_timer.stop()

            state = next_state
            if done:
                break

        agent.update_epsilon(True, args.decay_value)
        print("Total reward:", total_reward)
        print("now_epsilon:", agent.epsilon)

        # 모델/리플레이 버퍼 저장
        if (episode+1) % 10 == 0:
            model_filename = os.path.join(log_dir, f"acer_checkpoint_ep_{start_episode + episode + 1}.pth")
            agent.save_model(model_filename)
            agent.save_replay_buffer(replay_buffer_path)

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        if not os.path.exists(reward_file_path):
            open(reward_file_path, "w").close()
        with open(reward_file_path, "a") as f:
            f.write(f"{total_reward}\n")

        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon))

        if ENABLE_TIMER:
            print(f"episode {start_episode+episode+1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode+episode+1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
