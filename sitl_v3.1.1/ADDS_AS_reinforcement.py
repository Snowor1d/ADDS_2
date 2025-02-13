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
# Timer instances
sim_timer = Timer() 
learn_timer = Timer()
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log")
os.makedirs(log_dir, exist_ok=True)
model_load = 3

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--decay_value", type=float, default=0.99)
parser.add_argument("--buffer_size", type=int, default=1e5)
parser.add_argument("--batch_size", type=float, default=64)
args = parser.parse_args()


##########################################################################
# 1) Replay Buffer
##########################################################################
class ReplayBuffer:
    def __init__(self, capacity=int(1e4), device=None):
        self.buffer = deque(maxlen=int(capacity))
        self.device = device
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, int(batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.FloatTensor(states).unsqueeze(1).to(device)  # (B,1,H,W) if grayscale
        actions     = torch.FloatTensor(actions).to(device)             # (B,4)
        rewards     = torch.FloatTensor(rewards).to(device)             # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device)
        dones       = torch.FloatTensor(dones).to(device)               # (B,)
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
# 2) Gumbel-Softmax Utility
##########################################################################
def gumbel_softmax_sample(logits, eps=1e-8, temperature=1.0):
    # 이산적 행동을 연속적으로 표현하고, 미분 가능하도록 만들어 역전파를 가능하게 함
    """
    Sample from Gumbel-Softmax distribution (reparameterization trick).
    logits: (B,2) for 2 discrete modes
    returns a (B,2) one-hot-like sample with gradients
    """
    # Gumbel noise
    U = torch.rand_like(logits) # logit 값에 랜덤 노이즈 추가
    g = -torch.log(-torch.log(U + eps) + eps) # gumbel noise 계산
    # Add gumbel noise
    y = logits + g
    # Softmax
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax_log_prob(sample, logits, eps=1e-8):
    # 로그 확률 계산
    # Gumbel-Softmax로 생성된 샘플에 대해 로그 확률 계산
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

        self.fc_backbone = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        #state feature만 받아서 direction의 mean, log_std 추론
        self.dir_mean_head = nn.Linear(128, 2)
        self.dir_logstd_head = nn.Linear(128, 2)

        #mode는 state feature + direction feature를 받아서 mode_logits 추론
        self.mode_fc = nn.Sequential(
            nn.Linear(128+2, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        ) #mode 네트워크 fc구조가 좀 복잡한가?
        

    def _get_conv_out(self, shape):
        dummy = torch.zeros(1, 1, *shape)
        o = self.conv1(dummy)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def backbone(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)        
        feat = self.fc_backbone(x)       
        return feat

    def direction_head(self, feat):
        # feat : (B, 128), returns mean_dir, log_std_dir -> shape (B, 2)
        mean_dir = self.dir_mean_head(feat)
        log_std_dir = self.dir_logstd_head(feat)
        log_std_dir = torch.clamp(log_std_dir, self.log_std_min, self.log_std_max)
        return mean_dir, log_std_dir
    
    def mode_head(self, feat, direction):
        # feat: (B, 128), direction: (B, 2), returns mode_logits -> (B, num_modes)

        x = torch.cat([feat, direction], dim=1)
        logits = self.mode_fc(x)
        return logits

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
        returns: action=(B, 2+num_modes), log_prob=(B,)
                 => [dx, dy, mode_onehot...], log pi(a|s)
        """

        B = state.size(0)
        feat = self.backbone(state)
        mean_dir, log_std_dir = self.direction_head(feat)
        std_dir = log_std_dir.exp()

        eps = torch.randn_like(mean_dir)
        direction = mean_dir + std_dir * eps  # reparam trick

        # log prob of direction
        # sum over dim=1 (dx, dy)
        log_prob_dir = -0.5 * (((direction - mean_dir) / (std_dir+1e-8))**2 + 2*log_std_dir + np.log(2*np.pi)).sum(dim=1)

        mode_logits = self.mode_head(feat, direction)
        mode_one_hot = gumbel_softmax_sample(mode_logits, temperature=temperature)
        log_prob_mode = gumbel_softmax_log_prob(mode_one_hot, mode_logits)

        action = torch.cat([direction, mode_one_hot], dim=1)
        log_prob = log_prob_dir + log_prob_mode
        
        return action, log_prob
    
##########################################################################
# 5) SAC Agent for Hybrid Action
##########################################################################
class HybridSACAgent:
    def __init__(self, input_shape=(70,70), gamma=0.99, alpha=0.2, tau=0.995, lr=1e-4, batch_size=64, replay_size=int(1e5), device="cpu", start_epsilon = 1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_min = 0.1

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_size)
        

        # Critic networks
        self.q1 = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q2 = HybridQNetwork(input_shape, action_dim=4).to(self.device) #Q값의 과대평가 문제 줄이기 위해 double Q 도입
        # self.q1, self.q2 -> 현재 상태 s와 행동 a에 대해 Q-value를 근사하는 네트워크
        # predicted Q와 target Q의 차이를 줄이자

        self.q1_target = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        self.q2_target = HybridQNetwork(input_shape, action_dim=4).to(self.device)
        # self.q1_target, self.q2_target -> Q의 Ground Truth 근사치 제공
        # q-network 업데이트 시 사용하는 Target 값을 제공

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = HybridPolicyNetwork(input_shape).to(self.device)

        # Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr) #parameter optimizaing
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # optimizer는 loss function을 최소화 하도록 네트워크 파라미터 업데이트


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
            dx = np.random.uniform(-30,30)
            dy = np.random.uniform(-30,30)
            m_idx = np.random.randint(0,2)
            mode = np.array([1,0]) if m_idx==0 else np.array([0,1])
            return np.concatenate([[dx, dy], mode]), True

        # Otherwise use the policy
        state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
        # state_np는 2D 배열인데, 차원을 추가하여 모델 입력에 적합한 차원으로 만들려는 것

        with torch.no_grad():
            action_t, _ = self.policy.sample_action(state_t)
        action = action_t.cpu().numpy()[0]  # shape (4,)
        print("action : ", action)
        if deterministic:
            # You could return mean + argmax for the mode 
            # (requires rewriting sample_action). 
            pass
        return action, False



    # ------------------------------------------------- #
    # Update (one gradient step)
    # ------------------------------------------------- #
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
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
# Example usage in your training loop
##########################################################################
if __name__ == "__main__":
    import time
    import model  # Your environment code (model.FightingModel)

    # hyperparams
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


    agent = HybridSACAgent(input_shape=(70,70), alpha=0.2, lr=float(args.lr), start_epsilon=float(start_epsilon), batch_size=float(args.batch_size), replay_size=float(args.buffer_size))
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
        #try:
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

            # Simulation time check
            sim_timer.start()
            # 2) Step environment
            env_model.step()
            sim_timer.stop()

            # Learning time check
            learn_timer.start()
            # 3) Reward
            reward = env_model.check_reward_danger()
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
            learn_timer.stop()

            state = next_state
            if done:
                break
        # except Exception as e:
        #     print(e)
        #     print("error occured. retry.")
        #     env_model = model.FightingModel(number_of_agents, 70, 70, 2, 'Q')

        # Possibly update epsilon, or do other logging
        decay_value = 0.99
        if(agent.epsilon < 0.1):
            deacy_value = 1
        agent.update_epsilon(True, decay_value)
        print("Total reward:", total_reward)
        print("now_epsilon : ", agent.epsilon)
        # Save model occasionally

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        if not os.path.exists(reward_file_path):
            # 파일이 없으면 빈 파일 생성
            open(reward_file_path, "w").close()

        if (episode+1) % 10 == 0:
            model_filename = f"hybrid_sac_checkpoint_ep_{start_episode + episode + 1}.pth"
            agent.save_model(model_filename)
            replay_buffer_filename = "replay_buffer.pkl"
            agent.save_replay_buffer(replay_buffer_filename)

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        with open(reward_file_path, "a") as f:
            f.write(f"{total_reward}\n")

        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon))

        # each episode time print
        if ENABLE_TIMER:
            print(f"episode {start_episode+episode+1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode+episode+1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()