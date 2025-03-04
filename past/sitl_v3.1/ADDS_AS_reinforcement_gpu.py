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

# Timer instances
sim_timer = Timer() 
learn_timer = Timer()

##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e4)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1).to(device)  # (B,1,H,W) if grayscale
        actions     = torch.FloatTensor(actions).to(device)             # (B,4)
        rewards     = torch.FloatTensor(rewards).to(device)             # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
        dones       = torch.FloatTensor(dones).to(device)               # (B,)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

##########################################################################
# 3) Critic (Q) Network
##########################################################################
class HybridQNetwork(nn.Module):
    def __init__(self, input_shape=(70,70), action_dim=4):
        super(HybridQNetwork, self).__init__()

        # Feature extractor (conv) for state:
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_out = nn.Linear(128, 1)  # final Q-value

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)  # (B, C, H, W) = (1,1,H,W)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Concatenate state and action
        x = torch.cat([x, action], dim=1)  # (B, conv_out_size + 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.q_out(x)
        return q_val

##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class HybridPolicyNetwork(nn.Module):
    def __init__(self, input_shape=(70,70)):
        super(HybridPolicyNetwork, self).__init__()
        self.log_std_min = -20
        self.log_std_max =  2

        # Feature extractor (conv)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Outputs
        self.mean_head = nn.Linear(128, 2)
        self.log_std_head = nn.Linear(128, 2)
        self.mode_logits = nn.Linear(128, 2)

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

        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        mode_logits = self.mode_logits(x)
        return mean, log_std, mode_logits

    def sample_action(self, state, temperature=1.0):
        mean, log_std, mode_logits = self.forward(state)
        
        # Continuous action sampling
        std = log_std.exp()
        eps = torch.randn_like(mean)
        direction = mean + std * eps
        log_prob_cont = -0.5 * (((direction - mean) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi)).sum(dim=1)

        # Discrete action sampling
        mode_one_hot = F.gumbel_softmax(mode_logits, tau=temperature, hard=True)
        log_prob_mode = F.log_softmax(mode_logits, dim=-1) * mode_one_hot

        # Combine action
        action = torch.cat([direction, mode_one_hot], dim=1)
        log_prob = log_prob_cont + log_prob_mode.sum(dim=-1)
        return action, log_prob

##########################################################################
# 5) SAC Agent for Hybrid Action
##########################################################################
class HybridSACAgent:
    def __init__(self, input_shape=(70,70), gamma=0.99, alpha=0.2, tau=0.995, lr=1e-4, batch_size=64, replay_size=int(1e5), device="cpu"):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)

        # Critic networks
        self.q1 = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q2 = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q1_target = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q2_target = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = HybridPolicyNetwork(input_shape).to(self.device)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        # Target Q calculation
        with torch.no_grad():
            next_action, next_log_prob = self.policy.sample_action(next_states)
            q1_next = self.q1_target(next_states, next_action)
            q2_next = self.q2_target(next_states, next_action)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            q_target = rewards + self.gamma * (1 - dones) * (q_next - self.alpha * next_log_prob)

        # Q1, Q2 loss
        q1_val = self.q1(states, actions).squeeze(-1)
        q2_val = self.q2(states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        # Policy loss
        new_action, log_prob = self.policy.sample_action(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)
        policy_loss = (self.alpha * log_prob - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, net, net_target):
        for param, target_param in zip(net.parameters(), net_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def select_action(self, state_np, deterministic=False):
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_t, _ = self.policy.sample_action(state_t)
        return action_t.cpu().numpy()[0]
