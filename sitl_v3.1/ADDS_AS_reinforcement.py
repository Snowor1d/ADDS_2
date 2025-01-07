import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e4)):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        state: (H, W) or (C, H, W) as np array
        action: np.array of shape (4,) 
                e.g. [dx, dy, mode_onehot0, mode_onehot1]
        reward: float
        next_state: np.array
        done: float(0 or 1)
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1)  # (B,1,H,W) if grayscale
        actions     = torch.FloatTensor(actions)               # (B,4)
        rewards     = torch.FloatTensor(rewards)              # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1)
        dones       = torch.FloatTensor(dones)                # (B,)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

##########################################################################
# 2) Gumbel-Softmax Utility
##########################################################################
def gumbel_softmax_sample(logits, eps=1e-8, temperature=1.0):
    """
    Sample from Gumbel-Softmax distribution (reparameterization trick).
    logits: (B,2) for 2 discrete modes
    returns a (B,2) one-hot-like sample with gradients
    """
    # Gumbel noise
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + eps) + eps)
    # Add gumbel noise
    y = logits + g
    # Softmax
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax_log_prob(sample, logits, eps=1e-8):
    """
    Computes log pi(mode|s) for the discrete part using the Gumbel-Softmax sample.
    sample: (B,2) ~ one-hot
    logits: (B,2) raw logits
    This is approximate because we used Gumbel. We can also approximate log prob 
    by log softmax(logits).
    """
    # (B,2)
    log_probs = F.log_softmax(logits, dim=-1)
    # Sum over discrete dimension where sample = 1. 
    # Or do an elementwise product:
    # sum(sample * log_probs, dim=-1).
    return (sample * log_probs).sum(dim=-1)

##########################################################################
# 3) Critic (Q) Network
##########################################################################
class HybridQNetwork(nn.Module): # 주어진 상태 s와 행동 a에 대해 Q(s,a), 즉 이 행동이 얼마나 좋은지를 나타내는 값 예측 (Critic)
    """
    Q(s, a) where:
      - s: (C, H, W) or (1, H, W)
      - a: [dx, dy, mode0, mode1] (4-dim)
    """
    def __init__(self, input_shape=(70,70), action_dim=4):
        super(HybridQNetwork, self).__init__()

        # Feature extractor (conv) for state:
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        # We'll combine conv-out + action (4 dims) -> fully-connected
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
        """
        state: (B, 1, H, W)
        action: (B, 4)
        returns Q(s,a): (B,1)
        """
        # Convolution
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        # Concatenate action
        x = torch.cat([x, action], dim=1)  # (B, conv_out_size + 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_val = self.q_out(x)
        return q_val

##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class HybridPolicyNetwork(nn.Module):#행동을 샘플링하고 정책 학습, 주어진 상태 s에 대해 행동 a 결정, 연속적 & 이산적 행동 가능, 혼합 가능 (Actor)
    """
    Outputs distribution parameters for:
      - continuous direction: mean, log_std (2D)
      - discrete mode: logits (2D)
    We combine these into an action = [dx, dy, mode0, mode1].
    We'll do the reparam trick for direction, Gumbel-Softmax for mode.
    """
    def __init__(self, input_shape=(70,70)):
        super(HybridPolicyNetwork, self).__init__()
        self.log_std_min = -20
        self.log_std_max =  2

        # Feature extractor (conv)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        conv_out_size = self._get_conv_out(input_shape)

        # Shared fc
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 128)

        # Outputs for direction (continuous, 2 dims)
        self.mean_head = nn.Linear(128, 2)
        self.log_std_head = nn.Linear(128, 2)

        # Outputs for mode (discrete, 2 dims)
        self.mode_logits = nn.Linear(128, 2)

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, state):
        """
        Returns dict of {mean, log_std, mode_logits}
        state: (B,1,H,W)
        """
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        # clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        mode_logits = self.mode_logits(x)
        return mean, log_std, mode_logits

    def sample_action(self, state, temperature=1.0):
        """
        Returns a tuple (action, log_prob).
        action = [dx, dy, mode0, mode1], shape (B,4).
        log_prob = (B,) 
        """
        mean, log_std, mode_logits = self.forward(state)

        # 1) Continuous (direction)
        std = log_std.exp()
        # Reparameterization trick: N(0,1) -> sample
        eps = torch.randn_like(mean)
        direction = mean + std * eps  # shape (B,2)
        # log prob of the continuous part
        # log N(direction | mean, std)
        log_prob_cont = -0.5 * (((direction - mean) / (std+1e-8))**2 + 2*log_std + np.log(2*np.pi)).sum(dim=1)
        
        # We might apply a Tanh if we want to bound in [-1,1], e.g.:
        # direction = torch.tanh(direction)
        # If so, adjust the log_prob with the Tanh correction. (omitted here)

        # 2) Discrete (mode) using Gumbel-Softmax
        # sample one-hot
        mode_one_hot = gumbel_softmax_sample(mode_logits, temperature=temperature)
        # approximate log prob
        log_prob_mode = gumbel_softmax_log_prob(mode_one_hot, mode_logits)

        # Combine action
        action = torch.cat([direction, mode_one_hot], dim=1)  # (B,4)

        # Combine log-probs
        log_prob = log_prob_cont + log_prob_mode  # (B,)

        return action, log_prob

##########################################################################
# 5) SAC Agent for Hybrid Action
##########################################################################
class HybridSACAgent:
    def __init__(
        self,
        input_shape=(70,70),
        gamma=0.99,
        alpha=0.2,        # temperature (entropy weight)
        tau=0.995,        # soft-update
        lr=1e-4,
        batch_size=64,
        replay_size=int(1e5),
        start_epsilon=1.0  # if you still want some random exploring
    ):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size

        # For optional epsilon exploration (if you want it):
        self.epsilon = start_epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)

        # Critic networks (two Qs + two targets)
        self.q1 = HybridQNetwork(input_shape, action_dim=4)
        self.q2 = HybridQNetwork(input_shape, action_dim=4)  # Q값의 과대평가 문제 줄이기 위해 double Q 도입
        #self.q1, self.q2 -> 현재 상태 s와 행동 a에 대해 Q-값을 근사하는 네트워크


        self.q1_target = HybridQNetwork(input_shape, action_dim=4)
        self.q2_target = HybridQNetwork(input_shape, action_dim=4) 
        #self.q1_target, self.q2_target -> Q값의 Ground Truth 근사치를 제공?
        #q-network 업데이트 시 사용하는 Target 값을 제공 

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = HybridPolicyNetwork(input_shape)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
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
    # Epsilon update (only if you want random exploration)
    # ------------------------------------------------- #
    def update_epsilon(self, is_down, decay_value):
        if is_down:
            self.epsilon = max(self.epsilon_min, self.epsilon * decay_value)
        else:
            self.epsilon = min(1.0, self.epsilon / decay_value)

    # ------------------------------------------------- #
    # Store experience
    # ------------------------------------------------- #
    def store_transition(self, s, a, r, s_next, done):
        # if -20 <= a[0] <= 20 and -20 <= a[1] <= 20:
        self.replay_buffer.push(s, a, r, s_next, done)

    # ------------------------------------------------- #
    # Select action
    # ------------------------------------------------- #
    def select_action(self, state_np, deterministic=False):
        """
        state_np: shape (H, W) or (1, H, W)
        returns action_np shape (4,) = [dx, dy, mode0, mode1]
        If using epsilon > 0.0 for random exploration, 
        we can do random direction + random mode sometimes.
        """
        # Epsilon check
        if np.random.rand() < self.epsilon:
            # random direction in [-1,1], random mode
            dx = np.random.uniform(-20,20)
            dy = np.random.uniform(-20,20)
            m_idx = np.random.randint(0,2)
            mode = np.array([1,0]) if m_idx==0 else np.array([0,1])
            return np.concatenate([[dx, dy], mode]), True

        # Otherwise use the policy
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        with torch.no_grad():
            action_t, _ = self.policy.sample_action(state_t)
        action = action_t.cpu().numpy()[0]  # shape (4,)
        print(action)
        if deterministic:
            # You could return mean + argmax for the mode 
            # (requires rewriting sample_action). 
            pass
        return action, False



    # ------------------------------------------------- #
    # Update (one gradient step)
    # ------------------------------------------------- #
    def update(self):
        if len(self.replay_buffer) < self.batch_size*100:
            return
        
        # sample = self.replay_buffer.sample(self.batch_size)
        #states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 1. Replay Buffer에서 샘플 가져오기
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)


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
        q1_val = self.q1(states, actions).squeeze(-1)  # (B,)
        q2_val = self.q2(states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)

        self.q1_optimizer.zero_grad()
        loss_q1.backward()       
        self.q1_optimizer.step() # q1 update

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step() # q2 update

        # ----- Update Policy -----
        # re-sample action from current policy
        new_action, log_prob = self.policy.sample_action(states)
        q1_new = self.q1(states, new_action)
        q2_new = self.q2(states, new_action)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)  # (B,)

        # policy loss = alpha * log_prob - Q
        policy_loss = (self.alpha * log_prob - q_new).mean() #여기서 self.alpha * log_prob가 entropy term

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # soft update
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

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

##########################################################################
# Example usage in your training loop
##########################################################################
if __name__ == "__main__":
    import time
    import model  # Your environment code (model.FightingModel)

    # hyperparams
    max_episodes = 1000
    max_steps = 1500
    number_of_agents = 30
    start_episode = 0
    agent = HybridSACAgent(input_shape=(70,70), alpha=0.2, lr=1e-4, start_epsilon=1.0)

    # model_name = "learning_log/hybrid_sac_checkpoint_ep_250.pth"
    # if model_name.split("_")[-1].split(".")[0].isdigit():
    #     start_episode = int(model_name.split("_")[-1].split(".")[0])
    #     agent.load_model(model_name)

    for episode in range(max_episodes):
        print(f"Episode {start_episode+episode+1}")
        # Create environment
        while True:
            try:
                env_model = model.FightingModel(number_of_agents, 70, 70, 2, 'Q')
                break
            except Exception as e:
                print(e, "Retrying environment creation...")
        
        state = env_model.return_current_image()
        total_reward = 0
        for step in range(max_steps):
            # 1) Select action
            action_np, _ = agent.select_action(state)
            # action_np = [dx, dy, mode0, mode1]
            dx, dy = action_np[0], action_np[1]
            mode = np.argmax(action_np[2:])  # 0->GUIDE, 1->NOT_GUIDE

            # If you need to convert (dx, dy, mode) into actual env steps:
            # e.g. angle/distance or applying in robot.receive_action(...) 
            # This depends on how your environment expects "continuous direction."
            # Example dummy:
            real_action = env_model.robot.receive_action(([dx, dy], mode))

            # 2) Step environment
            env_model.step()

            # 3) Reward
            reward = env_model.check_reward_danger() / 30
            print("reward : ", reward)
            total_reward += reward

            # 4) Next state
            next_state = env_model.return_current_image()

            # 5) Done?
            done = (step >= max_steps-1) or (env_model.alived_agents() <= 1)

            # 6) Store transition
            agent.store_transition(
                state, 
                action_np, 
                reward, 
                next_state, 
                float(done)
            )

            # 7) Update agent
            agent.update()

            state = next_state
            if done:
                break

        # Possibly update epsilon, or do other logging
        decay_value = 0.99
        if(agent.epsilon < 0.1):
            deacy_value = 1
        agent.update_epsilon(True, decay_value)
        print("Total reward:", total_reward)
        print("now_epsilon : ", agent.epsilon)
        # Save model occasionally
        log_dir = "learning_log"
        os.makedirs(log_dir, exist_ok=True)  

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        if not os.path.exists(reward_file_path):
            # 파일이 없으면 빈 파일 생성
            open(reward_file_path, "w").close()

        if (episode+1) % 10 == 0:
            agent.save_model(f"learning_log/hybrid_sac_checkpoint_ep_{start_episode+episode+1}.pth")
        with open("learning_log/total_reward.txt", "a") as f:
            f.write(f"{total_reward}\n")
