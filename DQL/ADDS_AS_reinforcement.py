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
from Start_training import START_DECAY_STEP, START_EPSILON, EPSILON_MIN, SCHEDULER_TYPE, DECAY_VALUE, REWARD_A, REWARD_B, REWARD_C, REWARD_D, REWARD_E, REWARD_F, REWARD_G, REWARD_H, REWARD_I, REWARD_J, REWARD_K, CROWD_NUMBER_MIN, CROWD_NUMBER_MAX, LINEARLY_DECAY_STEP, START_UPDATE_STEP, LOG_DIR, FINISHED_BONUS, REWARD_FIXED, MAP_NUM, EXPLORATION_TYPE, \
                            LR, BUFFER_SIZE, BATCH_SIZE, LOG_STD_MAX, LOG_STD_MIN, ALPHA_START, ALPHA_END, ALPHA_DECAY_STEPS, DEVICE, SCALE_CHECK, ACTION_SCALE, START_BATCH_TIMES, MAX_STEPS, PORT_NUM, LONG_EPSILON_MIN, START_LONG_EPSILON, \
                            GAMMA_START, GAMMA_SCHEDULE_STEP, GAMMA_END, MAP_NUM_RANDOM
from einops import repeat

# Timer instances
sim_timer = Timer() 
learn_timer = Timer()
model_load = 3
# start_fresh : 1
# load specified model : 2
# load latest model : 3

home_dir = os.path.expanduser("~")
log_dir = os.path.join(home_dir, LOG_DIR)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

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



# --------------------------------------------------------------------- #
#                     DiffusionPolicy  (조건부 DDPM)                     #
# --------------------------------------------------------------------- #
class SinPosEmb(nn.Module): #DDPM에서 timestep을 임베딩하는 역할, 확산 과정에서 "현재 몇 번째 타임스텝인지"라는 정수형 시간 정보를 벡터 표현으로 변환
    def __init__(self, dim):
        super().__init__(); self.dim = dim
    def forward(self, t):                 # t: (B,1)
        freqs = torch.arange(self.dim, device=t.device) / self.dim
        emb = torch.einsum('bi,d->bid', t, freqs) * 2 * torch.pi
        return torch.cat([emb.sin(), emb.cos()], dim=-1).squeeze(1)

class EpsModel(nn.Module): # DDPM에서 εθ(a_i, s, t_idx)를 예측하는 네트워크, 주어진 노이즈가 섞인 행동을 보고 노이즈를 예측
    def __init__(self, a_dim, s_dim, hidden=256):
        super().__init__()
        self.t_emb = SinPosEmb(hidden//2)
        self.net = nn.Sequential(
            nn.Linear(a_dim + s_dim + hidden, hidden),
            nn.Mish(), 
            nn.Linear(hidden, hidden), 
            nn.Mish(),
            nn.Linear(hidden, a_dim)
        )
    def forward(self, a_i, s, t_idx):     # (B,a) (B,s) (B,1)
        emb = self.t_emb(t_idx)
        x = torch.cat([a_i, s, emb], dim=-1)
        return self.net(x)

class DiffusionPolicy(nn.Module):
    def __init__(self, s_dim, a_dim, N=5, beta_min=.1, beta_max=10.):
        super().__init__()
        self.N, self.a_dim = N, a_dim
        self.register_buffer("betas", self._make_beta(N, beta_min, beta_max))
        self.eps_model = EpsModel(a_dim, s_dim)
        self.alphas   = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)

        #self.betas -> 각 timesteps에서 추가되는 노이즈의 분산
        #self.alphas -> 각 timesteps에서 노이즈가 제거되는 비율
        #self.alphas_bar -> 누적된 노이즈 제거 비율

    # ------------ BC 손실 (Alg.1 ③, 식 2) ------------
    def bc_loss(self, s_flat, a0):
        B = s_flat.size(0)
        i = torch.randint(1, self.N+1, (B,1), device=s_flat.device)   # (B,1)
        eps = torch.randn(B, self.a_dim, device=s_flat.device)
        alpha_bar_i = self.alphas_bar[i-1]            # (B,1)
        ai = (alpha_bar_i.sqrt() * a0 +
              (1-alpha_bar_i).sqrt() * eps) # 행동 a_0에 노이즈를 넣어 a_i 생성하는 식
        eps_hat = self.eps_model(ai, s_flat, i.float())
        return F.mse_loss(eps_hat, eps)

    # ------------ 샘플링 (Alg.1 ①②) ------------
    @torch.no_grad()
    def sample(self, s_flat):
        B = s_flat.size(0)
        a = torch.randn(B, self.a_dim, device=s_flat.device)
        for i in reversed(range(1, self.N+1)):
            beta, alpha = self.betas[i-1], self.alphas[i-1]
            alpha_bar = self.alphas_bar[i-1]
            eps_hat = self.eps_model(a, s_flat,
                                     torch.full((B,1), i, device=s_flat.device))
            coef1 = 1/alpha.sqrt()
            coef2 = beta/((1-alpha_bar).sqrt())
            a = coef1 * (a - coef2 * eps_hat) # sampling, 위 식을 반복하여 노이즈 상태를 점점 원래 행동으로 되돌림
            if i > 1:
                a += beta.sqrt() * torch.randn_like(a)
        return torch.tanh(a)*2          # [-2,2]

    # ------------ forward dummy (안쓰지만 torch.compile 호환) ------------
    def forward(self, x): return x

    def _make_beta(self, N, bmin, bmax):
        idx = torch.arange(1, N+1)
        return 1 - torch.exp(-bmin/N - 0.5*(bmax-bmin)*(2*idx-1)/N**2)
    
    

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


ETA = 1.0
DIFF_STEPS = 5
BETA_MINMAX = (.1, 10.)

class DiffusionQLAgent:
    def __init__(self, device="cpu", batch_size=BATCH_SIZE):
        self.dev = torch.device(device)
        self.batch = batch_size
        self.buffer = ReplayBuffer(BUFFER_SIZE)

        self.q1 = QNetwork().to(self.dev)
        self.q2 = QNetwork().to(self.dev)
        self.q1_t = QNetwork().to(self.dev); self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t = QNetwork().to(self.dev); self.q2_t.load_state_dict(self.q2.state_dict())
        self.q_optim = optim.Adam(
            list(self.q1.parameters())+list(self.q2.parameters()), lr=LR)

        self.policy = DiffusionPolicy(
            s_dim=50*50, a_dim=2, N=DIFF_STEPS,
            beta_min=BETA_MINMAX[0], beta_max=BETA_MINMAX[1]).to(self.dev)
        self.p_optim = optim.Adam(self.policy.parameters(), lr=LR)

        self.gamma = GAMMA_START
        self.tau   = .995
        self.eps   = 0.05            # ε-greedy (아주 작게)

    # -------------- 행동 선택 --------------
    def select_action(self, state_np):
        if np.random.rand() < self.eps:
            return np.random.uniform(-2,2, size=(2,)) , True
        s_flat = torch.from_numpy(state_np).view(1,-1).float().to(self.dev)
        with torch.no_grad():
            a = self.policy.sample(s_flat)
        return a.cpu().numpy()[0] , False

    # -------------- 학습 스텝 --------------
    def update(self):
        if len(self.buffer) < self.batch*START_BATCH_TIMES: 
            return
        s,a,r,s2,d = self.buffer.sample(self.batch, self.dev)
        B = s.size(0)
        s_f  = s.view(B,-1); s2_f = s2.view(B,-1)

        # --- ① Q 손실 & 업데이트 (Alg.1 ④⑤) ---
        with torch.no_grad():
            a2 = self.policy.sample(s2_f)
            qt = torch.min(self.q1_t(s2,a2), self.q2_t(s2,a2)).squeeze(-1)
            y  = r + self.gamma*(1-d)*qt
        q1_loss = F.mse_loss(self.q1(s,a).squeeze(-1), y)
        q2_loss = F.mse_loss(self.q2(s,a).squeeze(-1), y)

        self.q_optim.zero_grad()
        (q1_loss+q2_loss).backward()
        torch.nn.utils.clip_grad_norm_( list(self.q1.parameters())+list(self.q2.parameters()), 1.0)
        self.q_optim.step()

        # --- ② 정책 손실 (Ld + η·Lq) ---
        Ld = self.policy.bc_loss(s_f, a)                 # 식 2
        a0 = self.policy.sample(s_f)
        q_min = torch.min(self.q1(s,a0), self.q2(s,a0)).squeeze(-1)
        Lq = (-q_min).mean()
        L_total = Ld + ETA * Lq

        self.p_optim.zero_grad()
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
        self.p_optim.step()

        # --- ③ 타깃 네트워크 소프트 업데이트 ---
        with torch.no_grad():
            for p,pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.data.mul_(self.tau).add_((1-self.tau)*p.data)
            for p,pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.data.mul_(self.tau).add_((1-self.tau)*p.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
    
    def save_replay_buffer(self, f):
        self.buffer.save(os.path.join(log_dir, f))
    def load_replay_buffer(self, f):
        self.buffer.load(os.path.join(log_dir, f))

    # -------------- 버퍼 저장·로드 --------------
    def save_model(self, path):
        torch.save({
            "q1":self.q1.state_dict(),"q2":self.q2.state_dict(),
            "q1_t":self.q1_t.state_dict(),"q2_t":self.q2_t.state_dict(),
            "policy":self.policy.state_dict(),
            "q_opt":self.q_optim.state_dict(),"p_opt":self.p_optim.state_dict()
        }, path)
    def load_model(self,path):
        ckpt = torch.load(path,map_location=self.dev)
        self.q1.load_state_dict(ckpt["q1"]); self.q2.load_state_dict(ckpt["q2"])
        self.q1_t.load_state_dict(ckpt["q1_t"]); self.q2_t.load_state_dict(ckpt["q2_t"])
        self.policy.load_state_dict(ckpt["policy"], strict=False)
        self.q_optim.load_state_dict(ckpt["q_opt"]); self.p_optim.load_state_dict(ckpt["p_opt"])


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
    
    agent = DiffusionQLAgent(device=DEVICE, batch_size=int(BATCH_SIZE))
    print(f"Agent initialized, lr={LR}, batch_size={BATCH_SIZE}, replay_size={BUFFER_SIZE}")
    replay_buffer_path = os.path.join(log_dir, "replay_buffer.pkl")

        
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "dql_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)

        if(os.path.exists(model_path)):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_name)
            if os.path.exists(replay_buffer_path):
                agent.load_replay_buffer("replay_buffer.pkl")
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir) if f.startswith("dql_checkpoint") and f.endswith(".pth")]
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

    #epsilon_scheduler = EpsilonScheduler(start_epsilon=start_epsilon, epsilon_min = EPSILON_MIN, start_decay_step = START_DECAY_STEP, scheduler_type=SCHEDULER_TYPE, decay_value=DECAY_VALUE, linear_decay_steps = LINEARLY_DECAY_STEP)

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
        
        state = env_model.return_current_image()
        total_reward = 0
        reward = 0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps

        buffered_state = state
        buffered_action = None
        abnormal_reward = 0
        #agent.update_alpha(ALPHA_START, ALPHA_END, ALPHA_DECAY_STEPS, episode_num)
        #agent.update_gamma(GAMMA_START, GAMMA_END, GAMMA_SCHEDULE_STEP, episode_num)
        try:
            for step in range(max_steps):
                # 1) Select action
            
                if(step%ACTION_SCALE==0):
                    
                    action_np, _ = agent.select_action(state)
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

        # Possibly update epsilon, or do other logging

        #agent.epsilon = epsilon_scheduler.get_epsilon(agent.epsilon, episode_num)
        #print("writing.. ", PORT_NUM)
        print("-----------------------------------------------")
        print("Total reward:", total_reward)
        #print("now_epsilon : ", agent.epsilon)
        #print("now_gamma : ", agent.gamma)
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
            model_filename = os.path.join(log_dir, f"dql_checkpoint_ep_{episode_num}.pth")
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

        # with open(epsilon_path, "w") as f:
        #     f.write(str(agent.epsilon)+"\n")
        #     f.write(str(agent.epsilon_long))


        # each episode time print
        if ENABLE_TIMER:
            print(f"episode {episode_num} - Total Simulation Time: {sim_timer.get_time():.6f} 초")
            print(f"episode {episode_num} - Total Learning Time: {learn_timer.get_time():.6f} 초")
            sim_timer.reset()
            learn_timer.reset()
