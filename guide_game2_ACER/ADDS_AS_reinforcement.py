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
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
import threading
import subprocess
import webbrowser

# Timer instances
sim_timer = Timer() 
learn_timer = Timer()
home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, "learning_log_guide_game2_ACER")
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


#########################################################
# 0) Discrete Action 매핑 함수
#########################################################
def int_action_to_dxdy(a):
    """
    0: Up, 1: Down, 2: Left, 3: Right
    return (dx, dy)
    """

    if a == 0:
        return (0, 2)   # Up
    elif a == 1:
        return (0,  -2)   # Down
    elif a == 2:
        return (-2, 0)   # Left
    elif a == 3:
        return (2,  0)   # Right
    else:
        return (0,0)

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
        actions     = torch.LongTensor(actions).to(device)               # (B,)  int
        rewards     = torch.FloatTensor(rewards).to(device)             # (B,)
        next_states = torch.FloatTensor(next_states).unsqueeze(1).to(device) # (B,1,H,W)
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
# 2) Q-Network (Critic)
##########################################################################
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

##########################################################################
# 3) Policy Network (Actor)
##########################################################################
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
        out = self.logits(x)  # (B,4) => logits
        return out

    def sample_action(self, state):
        """
        state: (B,1,H,W)
        return: action (B,) in {0,1,2,3}, log_prob (B,)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        return a, log_prob


#########################################################
# 4) Actor-Critic Agent (간단 ACER 유사)
#########################################################
class DiscreteACAgent:
    def __init__(self, input_shape=(70,70), gamma=0.99,
                 lr=1e-4, batch_size=64, replay_size=int(1e5), 
                 device="cpu", start_epsilon=1.0):
        
        self.num_actions = 4

        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.epsilon = start_epsilon
        self.epsilon_min = 0.1

        # Replay Buffer
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

    def store_transition(self, s, a, r, s_next, done):
        self.replay_buffer.push(s, a, r, s_next, done)

    def select_action(self, state_np, deterministic=False):
        """
        Epsilon-greedy로 행동을 선택
        - state_np: (H, W)
        - 반환: 정수 행동(0~3)
        """
        # epsilon에 따라 random action
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
            return action
        else:
            # Q값 가장 큰 행동
            state_t = torch.FloatTensor(state_np).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)
            with torch.no_grad():
                q_values = self.q_network(state_t)  # shape: (1, 4)
                action = q_values.argmax(dim=1).item()
            if action==0:
                print("UP")
            elif action==1:
                print("DOWN")
            elif action==2:
                print("LEFT")
            else :
                print("RIGHT")

            return action

    def update(self):
        if len(self.replay_buffer) < self.batch_size*10:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)
        
        # ------------------------
        # 1) Critic Update
        # ------------------------
        # Q(s,a) ~ target
        with torch.no_grad():
            q_next = self.q_target(next_states)       # (B,4)
            max_q_next, _ = torch.max(q_next, dim=1)  # (B,)
            q_target_vals = rewards + self.gamma * (1-dones) * max_q_next

        # 현재 Q
        q_vals = self.q_network(states)          # (B,4)
        q_acted = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        critic_loss = F.mse_loss(q_acted, q_target_vals)

        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        # ------------------------
        # 2) Policy(Actor) Update
        # ------------------------
        # Advantage = Q(s,a) - V(s) 
        # V(s) = Σ_a π(a|s) * Q(s,a)
        logits = self.policy_network(states)    # (B,4)
        probs = F.softmax(logits, dim=-1)       # (B,4)
        dist = Categorical(probs)

        q_vals_detached = q_vals.detach()       # (B,4)
        v_s = (probs * q_vals_detached).sum(dim=1)  # (B,)

        q_a = q_vals_detached.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)
        advantage = q_a - v_s

        log_probs = dist.log_prob(actions)  # (B,)

        policy_loss = -(log_probs * advantage).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
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

    tb_process = launch_tensorboard(tb_log_dir, port=6006)
    # 별도 스레드에서 total_reward.txt 모니터링 시작
    monitor_thread = threading.Thread(target=monitor_total_reward, args=(total_reward_file, tb_log_dir), daemon=True)
    monitor_thread.start()

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
    

    agent = DiscreteACAgent(
        input_shape   = (70,70),
        gamma         = 0.99,
        lr            = args.lr,
        batch_size    = int(args.batch_size),
        replay_size   = int(args.buffer_size),
        device        = "cpu",
        start_epsilon = start_epsilon
    )
    print(f"Agent initialized (DQN), lr={args.lr}, batch_size={args.batch_size}, replay_size={args.buffer_size}")
    replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

        
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "acer_checkpoint_ep_590.pth"
        model_path = os.path.join(log_dir, model_name)

        if(os.path.exists(model_path)):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_name)
            if os.path.exists(replay_buffer_path):
                agent.load_replay_buffer("replay_buffer.pkl")
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


    abnormal_reward = 0
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
        reward = 0
        buffered_state = state
        buffered_action = None
        abnormal_reward = 0
        try:
            for step in range(max_steps):
                # 1) Select action

                if(step%3==0):
                    action_np = agent.select_action(state, deterministic=False)
                    action = int_action_to_dxdy(action_np)
                    dx, dy = action[0], action[1]
                    real_action = env_model.robot.receive_action([dx, dy])
                    buffered_state = state
                    buffered_action = action_np
                
                # Simulation time check
                sim_timer.start()
                # 2) Step environment
                env_model.step()
                sim_timer.stop()

                # 3) Reward
                reward += env_model.reward_total()

                # 4) Next state
                next_state = env_model.return_current_image()

                # 5) Done?
                done = (step >= max_steps-1) or (env_model.robot.is_game_finished)
                if(env_model.robot.is_game_finished):
                    final_reward = 100 + (max_steps - step)*0.1 ## 탈출 성공 시 max_step 과 비교해 적은 step 내에서 탈출할 수록 크다..
                    reward += final_reward
                    print("@@@@@@@@ robot got the fianl reward @@@@@@")

                # 6) Store transition
                if(step%3==2):
                    agent.store_transition(
                        buffered_state,
                        buffered_action,
                        reward, 
                        next_state, 
                        float(done)
                    )
                    print("reward : ", reward)
                    total_reward += reward
                    reward = 0

                # 7) Update agent
                if(step%3==2):
                    learn_timer.start()
                    agent.update()
                    learn_timer.stop()

                state = next_state
                if done:
                    if max_steps <= (step + 1):
                        print("total reward = ", total_reward)
                        final_reward = abs(total_reward*0.1)
                        total_reward -= final_reward # 탈출 실패 시 total reward의 10%를 패널티로 받음
                        print(f"total reward -= {final_reward} : {total_reward}")
                    break
        except Exception as e:
            print(e)
            print("error occured. retry.")
            abnormal_reward = 1
            env_model = model.FightingModel(number_of_agents, 70, 70, 2, 'Q')

        # Possibly update epsilon, or do other logging
        decay_value = args.decay_value
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
            model_filename = os.path.join(log_dir, f"acer_checkpoint_ep_{start_episode + episode + 1}.pth")
            agent.save_model(model_filename)
            replay_buffer_filename = "replay_buffer.pkl"
            agent.save_replay_buffer(replay_buffer_filename)

        reward_file_path = os.path.join(log_dir, "total_reward.txt")
        with open(reward_file_path, "a") as f:
            if(abnormal_reward != 1):
                f.write(f"{total_reward}\n")

        with open(epsilon_path, "w") as f:
            f.write(str(agent.epsilon))


        # each episode time print
        if ENABLE_TIMER:
            print(f"episode {start_episode+episode+1} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {start_episode+episode+1} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()