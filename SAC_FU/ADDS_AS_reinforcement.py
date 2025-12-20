import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import time
from utils import Timer
from config import ENABLE_TIMER
import pickle
import argparse
import threading
from torch.utils.tensorboard import SummaryWriter
import subprocess
import webbrowser
from typing import Tuple, Any, Union, Optional
from config import *

from dataclasses import dataclass
import multiprocessing as mp
import queue
from queue import Empty, Full # for Empty


@dataclass
class TransitionMsg:
    worker_id: int
    ego_state: np.ndarray          # (4, EGO, EGO)
    global_state: np.ndarray       # (4, DOWN, DOWN)
    robot_state: np.ndarray
    action: np.ndarray
    reward: float
    next_ego_state: np.ndarray
    next_global_state: np.ndarray
    next_robot_state: np.ndarray
    done: bool

@dataclass
class EpisodeStatMsg:
    worker_id: int
    episode_idx: int
    total_reward: float
    evac_time_80: int
    evac_time_100: int
    total_lifetime: float
    map_num: int
    abnormal: int


STATE_SHAPE = (4, 50, 50)
INPUT_MAP_SIZE = 50
ROBOT_STATE_EMBEDDING = True
ROBOT_STATE_DIM = 3

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
BY_MAP_DIR = os.path.join(log_dir, "by_map")
os.makedirs(BY_MAP_DIR, exist_ok = True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# heat_logger = HeatMapLogger(   #@for heat_map
#     save_root = os.path.join(log_dir, "heat_maps"),
#     map_size = (MAP_W, MAP_H),
#     known_maps = MAP_NUM_RANDOM
# )

def ego_crop_from_full_map(full_map: np.ndarray,
                           robot_xy_px: tuple[int, int],
                           ego_size: int,
                           pad_value: int = 0) -> np.ndarray:
    """
    full_map: (H, W) uint8
    robot_xy_px: (ix, iy) in pixel coords (0..W-1, 0..H-1)
    return: (ego_size, ego_size) uint8
    """
    H, W = full_map.shape
    cx, cy = robot_xy_px
    half = ego_size // 2

    # 원하는 crop 좌표(맵 좌표 기준)
    x0, x1 = cx - half, cx - half + ego_size
    y0, y1 = cy - half, cy - half + ego_size

    # 맵과 겹치는 부분
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)

    crop = np.full((ego_size, ego_size), pad_value, dtype=full_map.dtype)

    # crop 안에서 어디에 붙일지 offset
    dx0 = sx0 - x0
    dy0 = sy0 - y0

    crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = full_map[sy0:sy1, sx0:sx1]
    return crop

def downsample_full_map(full_map: np.ndarray, target: int) -> np.ndarray:
    """
    full_map: (H, W) uint8
    return: (target, target) uint8 (adaptive pool)
    """
    x = torch.from_numpy(full_map).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    y = F.adaptive_max_pool2d(x, (target, target))
    return y.squeeze(0).squeeze(0).byte().numpy()



def normalize_map_to_50(obs, target=50):
    x = torch.from_numpy(obs).float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        y = F.adaptive_max_pool2d(x, (target, target))
        return y.squeeze().numpy()
    else:  # (C,H,W)
        x = x.unsqueeze(0)  # (1,C,H,W)
        y = F.adaptive_max_pool2d(x, (target, target))
        return y.squeeze(0).numpy()
 

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
        #print(f"Waiting for {metric_file} to be created...")
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

def ensure_file(path: str):
    if not os.path.exists(path):
        open(path, "w").close()

def map_metric_path(metric_name: str, map_num: int) -> str:
    # metric_name: "reward" | "evacuation_100"
    fname = f"{metric_name}_map_{map_num}.txt"
    return os.path.join(BY_MAP_DIR, fname)

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


def worker_process(
    worker_id: int,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    epsilon_shared: mp.Value,
    param_queue: mp.Queue,
    seed: int = 0,
):
    """
    각 worker 프로세스에서 실행되는 함수.
    - FightingModel을 생성해서 에피소드 무한히 반복
    - transition은 transition_queue에 넣고
    - 에피소드가 끝날 때 에피소드 통계는 stats_queue에 넣음
    """
    import model  # 여기에 import 해야 fork/spawn 모두 안전

    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

    max_steps = MAX_STEPS

    episode_idx = 0

    device = torch.device("cpu")
    policy = PolicyNetwork(
        ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)
    policy.eval()

    def _robot_world_to_px(env_model):
        rx, ry = env_model.robot.xy  # world coords
        ix = int(np.clip(rx / env_model.width  * MAP_W, 0, MAP_W - 1))
        iy = int(np.clip(ry / env_model.height * MAP_H, 0, MAP_H - 1))
        return ix, iy

    def _build_ego_global_frames(env_model):
        full = env_model.return_current_image(MAP_H, MAP_W)  # (H,W) uint8
        ix, iy = _robot_world_to_px(env_model)

        ego = ego_crop_from_full_map(full, (ix, iy), EGO_MAP_SIZE, pad_value=0)  # (EGO,EGO) uint8
        glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)                    # (DOWN,DOWN) uint8

        ego_f = ego.astype(np.float32) / 255.0
        glob_f = glob.astype(np.float32) / 255.0
        return ego_f, glob_f



    while True:  # 무한히 에피소드 반복
        # ----- 1) 환경 생성 -----
        while True:
            try:
                if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX:
                    number_of_agents = CROWD_NUMBER_MIN
                else:
                    number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)

                env_model = model.FightingModel(
                    number_of_agents,
                    MAP_W,
                    MAP_H,
                    model_num=-1,
                    robot='Q'
                )
                break
            except Exception as e:
                print(f"[Worker {worker_id}] env create error: {e}, retrying...")

        # ----- 2) 초기 state 세팅 -----
        ego_f, glob_f = _build_ego_global_frames(env_model)
        ego_stack = FrameStack2(4)
        glob_stack = FrameStack2(4)

        ego_state = ego_stack.reset(ego_f)
        global_state = glob_stack.reset(glob_f)
        robot_state = np.array(env_model.return_current_robot_state(), dtype=np.float32)
    

        # 에피소드 통계 변수
        total_reward = 0.0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps
        agent_total_lifetime = 0.0
        abnormal_reward = 0

        # transition을 만들기 위해 buffer
        buffered_ego_state = np.copy(ego_state)
        buffered_global_state = np.copy(global_state)
        buffered_robot_state = np.copy(robot_state)
        buffered_action = np.zeros((2,), dtype=np.float32)
        eps = 0.0
        with epsilon_shared.get_lock():
            eps = float(epsilon_shared.value)

        try:

            try:
                while True:
                    new_sd = param_queue.get_nowait()
                    policy.load_state_dict(new_sd)
            except queue.Empty:
                pass

            for step in range(max_steps):
                # -----------------------------
                # 1) ACTION_SCALE 간격으로만 action 선택
                # -----------------------------
                if step % ACTION_SCALE == 0:

                    ego_f, glob_f = _build_ego_global_frames(env_model)
                    if step > 0:
                        ego_state = ego_stack.append(ego_f)
                        global_state = glob_stack.append(glob_f)
                    dx = 0
                    dy = 0
                    # --- epsilon-greedy ---
                    if np.random.rand() < eps or policy is None:
                        # pure random
                        dx = np.random.uniform(-2, 2)
                        dy = np.random.uniform(-2, 2)
                        action_np = np.array([dx, dy], dtype=np.float32)
                    else:
                        ego_t = torch.from_numpy(ego_state).unsqueeze(0).float().to(device)
                        glob_t = torch.from_numpy(global_state).unsqueeze(0).float().to(device)
                        robot_t = torch.from_numpy(robot_state).unsqueeze(0).float().to(device)

                        with torch.no_grad():
                            action_t, _ = policy.sample_action(ego_t, glob_t, robot_t, temperature=1.0)
                        action_np = action_t.cpu().numpy()[0].astype(np.float32) # env 내부 제한 반영
                        dx = action_np[0]
                        dy = action_np[1]

                    real_action = env_model.robot.receive_action([dx, dy])
                    action_np[0] = real_action[0]
                    action_np[1] = real_action[1]

                    buffered_ego_state = np.copy(ego_state)
                    buffered_global_state = np.copy(global_state)
                    buffered_robot_state = np.copy(robot_state)
                    buffered_action = np.copy(action_np)

                # -----------------------------
                # 2) env step
                # -----------------------------
                env_model.step()


                # -----------------------------
                # 3) next state 계산
                # -----------------------------
                next_robot_state = np.array(env_model.return_current_robot_state(), dtype=np.float32)
                ego_f2, glob_f2 = _build_ego_global_frames(env_model)
                next_ego_state = ego_stack.peek_with(ego_f2)
                next_global_state = glob_stack.peek_with(glob_f2)


                # -----------------------------
                # 4) done / reward 계산
                # -----------------------------
                done = (step >= max_steps - 1) or (env_model.robot.is_game_finished)
                reward = 0.0
                r_k = 0.0

                if env_model.robot.is_game_finished:
                    reward += FINISHED_BONUS * (1 - step / max_steps)

                if REWARD_K:
                    r_k += env_model.reward_penalty_collision() * REWARD_K

                # ACTION_SCALE 마지막 스텝이거나 게임 끝난 경우에만 reward shaping + transition 전송
                if ((step % ACTION_SCALE == (ACTION_SCALE - 1) and step > ACTION_SCALE) or
                    (env_model.robot.is_game_finished and step > ACTION_SCALE)):

                    r_a = r_b = r_c = r_d = r_e = 0.0
                    r_f = r_g = r_h = r_i = r_j = 0.0
                    r_l = 0

                    if REWARD_A:
                        r_a = env_model.reward_based_alived() * REWARD_A
                    if REWARD_B:
                        r_b = env_model.reward_based_all_agents_danger() * REWARD_B
                    if REWARD_C:
                        r_c = env_model.reward_based_gain() * REWARD_C
                    if REWARD_D:
                        r_d = env_model.reward_penalty() * REWARD_D
                    if REWARD_E:
                        r_e = env_model.reward_based_evacuated_with_robot() * REWARD_E
                    if REWARD_F:
                        r_f = env_model.reward_based_distance_from_near_agents() * REWARD_F
                    if REWARD_G:
                        r_g = env_model.reward_based_distance_from_near_agent_gain() * REWARD_G
                    if REWARD_H:
                        r_h = env_model.reward_based_gain_with_time_bonus() * REWARD_H
                    if REWARD_I:
                        r_i = env_model.reward_based_alived_root() * REWARD_I
                    if REWARD_J:
                        r_j = env_model.reward_based_all_agents_danger_root() * REWARD_J
                    if REWARD_L:
                        r_l = env_model.reward_based_farthest_agent_distance() * REWARD_L

                    reward += (r_a + r_b + r_c + r_d + r_e + r_g + r_h + r_i + r_j + r_k + r_l + REWARD_FIXED)
                    r_k = 0.0

                    # -----------------------------
                    # 5) transition Queue로 전송
                    # -----------------------------
                    try:
                        msg = TransitionMsg(
                            worker_id=worker_id,
                            ego_state=buffered_ego_state,
                            global_state=buffered_global_state,
                            robot_state=buffered_robot_state,
                            action=buffered_action,
                            reward=float(reward),
                            next_ego_state=next_ego_state,
                            next_global_state=next_global_state,
                            next_robot_state=next_robot_state,
                            done=bool(done),
                        )
                        transition_queue.put(msg)  # blocking
                        total_reward += reward
                    except Exception as e:
                        print(f"[Worker {worker_id}] transition_queue.put error: {e}")
                        abnormal_reward = 1

                    # 다음 action 선택을 위해 state/robot_state 갱신
                    robot_state = next_robot_state
                else:
                    # ACTION_SCALE 중간에서도 robot_state는 최신으로 유지
                    robot_state = next_robot_state

                # 80%, 100% 대피 시간
                if (env_model.alived_agents() < env_model.total_agents * 0.2 and
                        evacuation_time_80 == max_steps):
                    evacuation_time_80 = step
                if (env_model.alived_agents() < 1 and
                        evacuation_time_100 == max_steps):
                    evacuation_time_100 = step

                if done:
                    try:
                        agent_total_lifetime = env_model.calculate_all_agents_life_time()
                    except Exception:
                        agent_total_lifetime = 0.0
                    break  # 에피소드 종료

        except Exception as e:
            print(f"[Worker {worker_id}] Error in episode loop: {e}")
            import traceback
            traceback.print_exc()
            abnormal_reward = 1

        # ----- 6) 에피소드 통계 전송 -----
        stat_msg = EpisodeStatMsg(
            worker_id=worker_id,
            episode_idx=episode_idx,
            total_reward=float(total_reward),
            evac_time_80=int(evacuation_time_80),
            evac_time_100=int(evacuation_time_100),
            total_lifetime=float(agent_total_lifetime),
            map_num=int(env_model.map_num),
            abnormal=int(abnormal_reward)
        )
        try:
            stats_queue.put(stat_msg)
        except Exception as e:
            print(f"[Worker {worker_id}] stats_queue.put error: {e}")

        episode_idx += 1
        # 여기서 바로 while True 위로 올라가서 새 env 생성 → 동기화 없이 계속 돎
    
    

##########################################################################
# Replay Buffer (Ego + Global 2-branch)
##########################################################################
class ReplayBuffer:
    """
    NumPy 기반 고정 크기 링 버퍼 + dtype 축소로 메모리 사용 최소화
    - ego map / global map을 분리 저장
    - save/load 지원 (npz)

    Parameters
    ----------
    capacity : int
        최대 저장 가능한 transition 개수
    ego_state_shape : tuple of int
        ego state shape, 예: (4, EGO, EGO)
    global_state_shape : tuple of int
        global state shape, 예: (4, DOWN, DOWN)
    action_dim : int
        액션 차원
    robot_dim : int
        로봇 상태 차원 (예: 3)
    device : torch.device or str
    state_dtype : np.dtype
        이미지 저장 dtype (np.uint8 권장)
    """
    def __init__(
        self,
        capacity: int,
        ego_state_shape: Tuple[int, int, int],
        global_state_shape: Tuple[int, int, int],
        action_dim: int = 2,
        robot_dim: int = 3,
        device=None,
        state_dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.state_dtype = state_dtype
        self.robot_dim = int(robot_dim)

        self.ego_state_shape = tuple(ego_state_shape)
        self.global_state_shape = tuple(global_state_shape)

        # 고정 크기 NumPy 배열 할당
        self.ego_states = np.zeros((self.capacity, *self.ego_state_shape), dtype=self.state_dtype)
        self.ego_next_states = np.zeros((self.capacity, *self.ego_state_shape), dtype=self.state_dtype)

        self.global_states = np.zeros((self.capacity, *self.global_state_shape), dtype=self.state_dtype)
        self.global_next_states = np.zeros((self.capacity, *self.global_state_shape), dtype=self.state_dtype)

        if self.robot_dim > 0:
            self.robot_states = np.zeros((self.capacity, self.robot_dim), dtype=np.float32)
            self.next_robot_states = np.zeros((self.capacity, self.robot_dim), dtype=np.float32)
        else:
            self.robot_states = None
            self.next_robot_states = None

        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=bool)

        self.ptr = 0
        self.size = 0

    # -----------------------
    # utils
    # -----------------------
    def _to_uint8(self, x: np.ndarray) -> np.ndarray:
        """
        float [0,1] or other -> uint8
        이미 uint8이면 그대로 반환
        """
        if x.dtype == self.state_dtype:
            return x
        # float map assumed in [0,1]
        return np.clip(x * 255.0, 0, 255).astype(self.state_dtype)

    def _check_shapes(
        self,
        ego_state: np.ndarray,
        global_state: np.ndarray,
        next_ego_state: np.ndarray,
        next_global_state: np.ndarray,
    ) -> None:
        if tuple(ego_state.shape) != self.ego_state_shape:
            raise ValueError(f"ego_state shape mismatch: got {ego_state.shape}, expected {self.ego_state_shape}")
        if tuple(next_ego_state.shape) != self.ego_state_shape:
            raise ValueError(f"next_ego_state shape mismatch: got {next_ego_state.shape}, expected {self.ego_state_shape}")
        if tuple(global_state.shape) != self.global_state_shape:
            raise ValueError(f"global_state shape mismatch: got {global_state.shape}, expected {self.global_state_shape}")
        if tuple(next_global_state.shape) != self.global_state_shape:
            raise ValueError(f"next_global_state shape mismatch: got {next_global_state.shape}, expected {self.global_state_shape}")

    # -----------------------
    # main API
    # -----------------------
    def push(
        self,
        ego_state: np.ndarray,
        global_state: np.ndarray,
        robot_state: Optional[np.ndarray],
        action: np.ndarray,
        reward: float,
        next_ego_state: np.ndarray,
        next_global_state: np.ndarray,
        next_robot_state: Optional[np.ndarray],
        done: bool,
    ) -> None:
        """
        ego_state: (4,EGO,EGO) float32 [0,1] or uint8
        global_state: (4,DOWN,DOWN) float32 [0,1] or uint8
        """
        self._check_shapes(ego_state, global_state, next_ego_state, next_global_state)

        i = self.ptr

        ego_u8 = self._to_uint8(ego_state)
        ego2_u8 = self._to_uint8(next_ego_state)
        glob_u8 = self._to_uint8(global_state)
        glob2_u8 = self._to_uint8(next_global_state)

        self.ego_states[i] = ego_u8
        self.ego_next_states[i] = ego2_u8
        self.global_states[i] = glob_u8
        self.global_next_states[i] = glob2_u8

        if self.robot_dim > 0:
            if robot_state is None or next_robot_state is None:
                raise ValueError("Buffer robot_dim > 0, but robot_state or next_robot_state is None.")
            self.robot_states[i] = robot_state
            self.next_robot_states[i] = next_robot_state

        self.actions[i] = action
        self.rewards[i] = float(reward)
        self.dones[i] = bool(done)

        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: size={self.size}, batch_size={batch_size}")
        idx = np.random.choice(self.size, batch_size, replace=False)

        # NumPy -> torch.Tensor 변환 (float32)
        batch_ego = torch.from_numpy(self.ego_states[idx].astype(np.float32) / 255.0).to(self.device)
        batch_ego2 = torch.from_numpy(self.ego_next_states[idx].astype(np.float32) / 255.0).to(self.device)

        batch_global = torch.from_numpy(self.global_states[idx].astype(np.float32) / 255.0).to(self.device)
        batch_global2 = torch.from_numpy(self.global_next_states[idx].astype(np.float32) / 255.0).to(self.device)

        if self.robot_dim > 0:
            batch_robot = torch.from_numpy(self.robot_states[idx]).to(self.device)
            batch_robot2 = torch.from_numpy(self.next_robot_states[idx]).to(self.device)
        else:
            batch_robot = None
            batch_robot2 = None

        batch_actions = torch.from_numpy(self.actions[idx]).to(self.device)
        batch_rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        batch_dones = torch.from_numpy(self.dones[idx].astype(np.float32)).to(self.device)

        # 반환 순서: ego, global, robot, action, reward, next_ego, next_global, next_robot, done
        return (
            batch_ego,
            batch_global,
            batch_robot,
            batch_actions,
            batch_rewards,
            batch_ego2,
            batch_global2,
            batch_robot2,
            batch_dones,
        )

    def __len__(self) -> int:
        return self.size

    # -----------------------
    # save / load
    # -----------------------
    def save(self, filepath: Union[str, bytes, os.PathLike]) -> None:
        """
        압축 npz 파일로 저장 (현재 size 만큼만 포함)
        """
        save_dict = {
            # ego/global
            "ego_states": self.ego_states[: self.size],
            "ego_next_states": self.ego_next_states[: self.size],
            "global_states": self.global_states[: self.size],
            "global_next_states": self.global_next_states[: self.size],

            # action/reward/done
            "actions": self.actions[: self.size],
            "rewards": self.rewards[: self.size],
            "dones": self.dones[: self.size],

            # meta
            "size": self.size,
            "ptr": self.ptr,
            "capacity": self.capacity,
            "state_dtype": np.dtype(self.state_dtype).name,
            "robot_dim": self.robot_dim,

            "ego_state_shape": np.array(self.ego_state_shape, dtype=np.int32),
            "global_state_shape": np.array(self.global_state_shape, dtype=np.int32),
        }

        if self.robot_dim > 0:
            save_dict["robot_states"] = self.robot_states[: self.size]
            save_dict["next_robot_states"] = self.next_robot_states[: self.size]

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: Union[str, bytes, os.PathLike]) -> None:
        """
        저장된 npz 파일을 읽어 버퍼 상태 복원
        """
        data = np.load(filepath, allow_pickle=False)

        # 필수 키 체크 (새 포맷)
        required = ["ego_states", "ego_next_states", "global_states", "global_next_states",
                    "actions", "rewards", "dones", "size", "ptr", "capacity",
                    "state_dtype", "robot_dim", "ego_state_shape", "global_state_shape"]
        for k in required:
            if k not in data.files:
                raise ValueError(
                    f"[ReplayBuffer.load] '{k}' not found in npz. "
                    "This looks like an old buffer format. (single-state) "
                    "Need a legacy converter if you want to load it."
                )

        cap = int(data["capacity"])
        prev_robot_dim = int(data["robot_dim"])
        prev_dtype = np.dtype(str(data["state_dtype"]))

        ego_shape = tuple(data["ego_state_shape"].astype(int).tolist())
        global_shape = tuple(data["global_state_shape"].astype(int).tolist())

        # config 변화(용량/shape/robot_dim/dtype) 감지 시 재초기화
        if (
            cap != self.capacity
            or prev_robot_dim != self.robot_dim
            or prev_dtype != self.state_dtype
            or ego_shape != self.ego_state_shape
            or global_shape != self.global_state_shape
        ):
            action_dim = int(data["actions"].shape[1])
            device = self.device
            self.__init__(
                capacity=cap,
                ego_state_shape=ego_shape,
                global_state_shape=global_shape,
                action_dim=action_dim,
                robot_dim=prev_robot_dim,
                device=device,
                state_dtype=prev_dtype,
            )

        self.size = int(data["size"])
        self.ptr = int(data["ptr"])

        # 데이터 복사
        self.ego_states[: self.size] = data["ego_states"]
        self.ego_next_states[: self.size] = data["ego_next_states"]
        self.global_states[: self.size] = data["global_states"]
        self.global_next_states[: self.size] = data["global_next_states"]

        self.actions[: self.size] = data["actions"]
        self.rewards[: self.size] = data["rewards"]
        self.dones[: self.size] = data["dones"]

        if self.robot_dim > 0:
            if "robot_states" not in data.files or "next_robot_states" not in data.files:
                raise ValueError("robot_dim > 0인데 robot_states/next_robot_states가 저장 파일에 없음.")
            self.robot_states[: self.size] = data["robot_states"]
            self.next_robot_states[: self.size] = data["next_robot_states"]

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
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.silu(out)
        
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

    
        
class CNNEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=32, out_dim=256, input_hw=(64,64)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.conv2 = nn.Conv2d(base_ch, base_ch*2, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(base_ch*2)
        self.conv3 = nn.Conv2d(base_ch*2, base_ch*4, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(base_ch*4)

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, *input_hw)
            x = self._forward_conv(dummy)
            conv_out = int(np.prod(x.shape[1:]))

        self.fc = nn.Sequential(
            nn.Linear(conv_out, out_dim),
            nn.SiLU()
        )

    def _forward_conv(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.flatten(1)
        return self.fc(x)


##########################################################################
# 3) Critic (Q) Network
##########################################################################
class QNetwork(nn.Module):
    def __init__(self, ego_hw, global_hw, action_dim=2, robot_dim=3, use_robot=True):
        super().__init__()
        self.use_robot = use_robot
        self.ego_enc = CNNEncoder(in_ch=4, base_ch=32, out_dim=256, input_hw=ego_hw)
        self.glob_enc = CNNEncoder(in_ch=4, base_ch=16, out_dim=128, input_hw=global_hw)

        robot_feat = 0
        if use_robot:
            self.robot_fc = nn.Sequential(nn.Linear(robot_dim, 32), nn.SiLU())
            robot_feat = 32

        fusion_dim = 256 + 128 + robot_feat + action_dim
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, ego_state, global_state, action, robot_state=None):
        e = self.ego_enc(ego_state)
        g = self.glob_enc(global_state)
        feats = [e, g]
        if self.use_robot:
            feats.append(self.robot_fc(robot_state))
        feats.append(action)
        x = torch.cat(feats, dim=1)
        return self.mlp(x)


##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, ego_hw, global_hw, robot_dim=3, use_robot=True):
        super().__init__()
        self.use_robot = use_robot
        self.ego_enc = CNNEncoder(in_ch=4, base_ch=32, out_dim=256, input_hw=ego_hw)
        self.glob_enc = CNNEncoder(in_ch=4, base_ch=16, out_dim=128, input_hw=global_hw)  # 전역은 가볍게

        robot_feat = 0
        if use_robot:
            self.robot_fc = nn.Sequential(nn.Linear(robot_dim, 32), nn.SiLU())
            robot_feat = 32

        fusion_dim = 256 + 128 + robot_feat
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(),
        )
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

    def forward(self, ego_state, global_state, robot_state=None):
        e = self.ego_enc(ego_state)
        g = self.glob_enc(global_state)
        feats = [e, g]
        if self.use_robot:
            feats.append(self.robot_fc(robot_state))
        x = torch.cat(feats, dim=1)

        h = self.backbone(x)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, ego_state, global_state, robot_state=None, temperature=1.0):
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        std = log_std.exp()
        u = mean + std * (torch.randn_like(mean) * temperature)
        sigma = torch.sigmoid(u)
        action = 4 * sigma - 2

        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        log_prob_u = log_prob_u.sum(dim=1)
        jacobian = torch.log(4*sigma*(1 - sigma) + 1e-8).sum(dim=1)
        log_prob = log_prob_u - jacobian
        return action, log_prob
    
class FrameStack:
    """ACTION_SCALE 간격으로만 push 되는 4‑프레임 스택"""
    def __init__(self, stack_len=4):
        self.stack_len = stack_len
        self.frames = deque(maxlen=stack_len)

    def reset(self, first_frame):
        self.frames.clear()
        first_frame = first_frame.astype(np.float32)
        for _ in range(self.stack_len):
            self.frames.append(np.copy(first_frame))
        return np.stack(list(self.frames)[::-1], axis=0)   # (4,H,W)

    def append(self, frame):
        """deque에 실제 push & 최신 스택 반환"""
        frame = frame.astype(np.float32)
        self.frames.append(np.copy(frame))
        return np.stack(list(self.frames)[::-1], axis=0)

    def peek_with(self, frame):
        """frame을 push 했다고 가정한 결과 스택 반환( deque 내용은 그대로 )"""
        frame = frame.astype(np.float32)
        tmp = list(self.frames) + [frame]
        return np.stack(tmp[-self.stack_len:][::-1], axis=0)


class FrameStack2: #for ego
    def __init__(self, stack_len=4):
        self.stack_len = stack_len
        self.frames = deque(maxlen=stack_len)

    def reset(self, first_frame: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.stack_len):
            self.frames.append(np.copy(first_frame))
        return np.stack(list(self.frames)[::-1], axis=0)  # (4,H,W)

    def append(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(np.copy(frame))
        return np.stack(list(self.frames)[::-1], axis=0)

    def peek_with(self, frame: np.ndarray) -> np.ndarray:
        tmp = list(self.frames) + [frame]
        return np.stack(tmp[-self.stack_len:][::-1], axis=0)


##########################################################################
# 5) SAC Agent for Action
##########################################################################
class SACAgent:
    def __init__(self, input_shape=(50,50), gamma=GAMMA_START, alpha=0.2, tau=0.995, lr=1e-4, batch_size=64, replay_size=int(1e5), device="cpu", start_epsilon = 1.0, start_epsilon_long = 0.1, long_epsilon_min=0):
        self.gamma = gamma
        self.alpha = alpha
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = long_epsilon_min

        # Multimodal Flag
        self.use_robot_state = ROBOT_STATE_EMBEDDING
        self.robot_dim = ROBOT_STATE_DIM if self.use_robot_state else 0

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=int(replay_size),
            ego_state_shape=(4, EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_state_shape=(4, DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=self.robot_dim,
            device=self.device,
        )

        # Critic networks
        self.q1 = QNetwork(
            ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2 = QNetwork(
            ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q1_target = QNetwork(
            ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2_target = QNetwork(
            ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)


        # self.q1_target, self.q2_target -> Q의 Ground Truth 근사치 제공
        # q-network 업데이트 시 사용하는 Target 값을 제공

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = PolicyNetwork(
            ego_hw=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_hw=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING
        ).to(self.device)

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
    # def store_transition(self, s, robot_s, a, r, s_next, robot_s_next, done):
    #     # if -20 <= a[0] <= 20 and -20 <= a[1] <= 20:
    #     self.replay_buffer.push(s, robot_s, a, r, s_next, robot_s_next, done)

    # ------------------------------------------------- #
    # Select action
    # ------------------------------------------------- #
    def select_action(self, ego_state_np, global_state_np, robot_state_np=None, deterministic=False):
        """
        state_np: shape (H, W) or (1, H, W)
        returns action_np shape (4,) = [dx, dy, mode0, mode1]
        If using epsilon > 0.0 for random exploration, 
        we can do random direction + random mode sometimes.
        """

        if(EXPLORATION_TYPE == 0):
            # Epsilon check
            if np.random.rand() < self.epsilon:
                # random direction in [-1,1], random mode
                dx = np.random.uniform(-2,2)
                dy = np.random.uniform(-2,2)
                return np.array([dx, dy]), True

        # Otherwise use the policy
        ego_t = torch.FloatTensor(ego_state_np).unsqueeze(0).to(self.device)
        global_t = torch.FloatTensor(global_state_np).unsqueeze(0).to(self.device)
        robot_t = torch.FloatTensor(robot_state_np).unsqueeze(0).to(self.device)
        # state_np는 2D 배열인데, 차원을 추가하여 모델 입력에 적합한 차원으로 만들려는 것


        with torch.no_grad():
            if deterministic:
                # 결정적 행동 선택: mean에 대해 바로 sigmoid 변환.
                mean, _ = self.policy.forward(ego_t, global_t, robot_t)
                action_t = 4*torch.sigmoid(mean)-2
            else:
                # 비결정적 선택: sample_action에서 샘플링 (자코비안 보정 포함)
                action_t, log_prob = self.policy.sample_action(ego_t, global_t, robot_t)
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
        ego_s, glob_s, robot_s, a, r, ego_s2, glob_s2, robot_s2, d = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_a, next_logp = self.policy.sample_action(ego_s2, glob_s2, robot_s2)
            q1_next = self.q1_target(ego_s2, glob_s2, next_a, robot_s2)
            q2_next = self.q2_target(ego_s2, glob_s2, next_a, robot_s2)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)
            q_target = r + self.gamma * (1 - d) * (q_next - self.alpha * next_logp)

        # ----- Update Q1, Q2 -----
        q1_val = self.q1(ego_s, glob_s, a, robot_s).squeeze(-1)  # (B,) #q value를 scalar 값으로
        q2_val = self.q2(ego_s, glob_s, a, robot_s).squeeze(-1)
        
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
        new_action, log_prob = self.policy.sample_action(ego_s, glob_s, robot_s)
        q1_new = self.q1(ego_s, glob_s, new_action, robot_s)
        q2_new = self.q2(ego_s, glob_s, new_action, robot_s)
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

        if not os.path.exists(filepath):
            print(f"[Warning] Model checkpoint not found: {filepath} — skipping loading.")
            return

        try:
            ckpt = torch.load(filepath, map_location=self.device)
        except Exception as e:
            print(f"[Warning] Failed to load model from {filepath}: {e}")
            return

        try:
            self.q1.load_state_dict(ckpt['q1'])
            self.q2.load_state_dict(ckpt['q2'])
            self.q1_target.load_state_dict(ckpt['q1_target'])
            self.q2_target.load_state_dict(ckpt['q2_target'])
            self.policy.load_state_dict(ckpt['policy'])
            self.q1_optimizer.load_state_dict(ckpt['q1_opt'])
            self.q2_optimizer.load_state_dict(ckpt['q2_opt'])
            self.policy_optimizer.load_state_dict(ckpt['policy_opt'])
            print(f"[Info] Model successfully loaded from {filepath}")
        except KeyError as e:
            print(f"[Warning] Missing key in checkpoint ({e}). The model may be incompatible.")
        except Exception as e:
            print(f"[Warning] Unexpected error while loading model: {e}")

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
    # 파일 생성 대기
    while not os.path.exists(total_reward_file):
        #print(f"Waiting for {total_reward_file} to be created...")
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
                            #print(f"Episode {episode}: Total Reward = {total_reward}")
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


    mp.set_start_method("spawn", force=True)  # or "fork", 리눅스면 fork도 가능

    # ----- 1) TensorBoard/로그 파일 설정 (기존 코드 유지 가능) -----
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")
    total_lifetime_file = os.path.join(log_dir, "total_lifetime.txt")

    #reward_vs_ls_file = os.path.join(log_dir, "reward_vs_learning_step.txt")
    #evac100_vs_ls_file = os.path.join(log_dir, "evac100_vs_learning_step.txt")

    #tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    step_writer = SummaryWriter(log_dir=tb_log_dir)


    # 파일 존재 보장
    for path in [total_reward_file, evacuation_time_80_file,
                 evacuation_time_100_file, total_lifetime_file]:
        if not os.path.exists(path):
            open(path, "w").close()

    # TensorBoard 관련 (원하면 기존 monitor thread 그대로 사용 가능)
    tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    monitor_thread = threading.Thread(
        target=monitor_total_reward,
        args=(total_reward_file, tb_log_dir),
        daemon=True
    )
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


    monitor_thread_lifetime = threading.Thread(
        target=monitor_metric,
        args = (total_lifetime_file, "Total Lifetime", tb_log_dir),
        daemon=True
    )
    monitor_thread_lifetime.start()

    # monitor_thread_100_vs_ls = threading.Thread(
    #     target=monitor_metric, 
    #     args=(evac100_vs_ls_file, "Evacuation Time 100 vs learning step", tb_log_dir),
    #     daemon=True
    # )
    # monitor_thread_100_vs_ls.start()

    # monitor_thread_reward_vs_ls = threading.Thread(
    #     target=monitor_metric, 
    #     args=(reward_vs_ls_file, "Reward vs learning step", tb_log_dir),
    #     daemon=True
    # )
    # monitor_thread_reward_vs_ls.start()

    for m in MAP_NUM_RANDOM:
        # 맵별 reward 모니터
        reward_map_file = map_metric_path("reward", m)
        ensure_file(reward_map_file)
        threading.Thread(
            target=monitor_metric,
            args=(reward_map_file, f"Reward/map_{m}", tb_log_dir),
            daemon=True
        ).start()

        # 맵별 evacuation_100 모니터
        evac100_map_file = map_metric_path("evacuation_100", m)
        ensure_file(evac100_map_file)
        threading.Thread(
            target=monitor_metric,
            args=(evac100_map_file, f"Evacuation Time 100/map_{m}", tb_log_dir),
            daemon=True
        ).start()

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
    epsilon_shared = mp.Value('d', start_epsilon)
    agent = SACAgent(input_shape=(50,50), alpha=ALPHA_START, lr=float(LR), start_epsilon=start_epsilon, start_epsilon_long = float(START_LONG_EPSILON), long_epsilon_min=float(LONG_EPSILON_MIN), batch_size=int(BATCH_SIZE), replay_size=int(BUFFER_SIZE), device=DEVICE)
    print(f"Agent initialized, lr={LR}, alpha={agent.alpha}, batch_size={BATCH_SIZE}, replay_size={BUFFER_SIZE}")
    replay_buffer_path = os.path.join(log_dir, "replay_buffer.npz")

    global_episode = 0
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
                agent.load_replay_buffer("replay_buffer.npz")
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir) if f.startswith("sac_checkpoint") and f.endswith(".pth")]
        if model_files:
            latest_model = max(model_files, key=lambda f: int(f.split("_")[-1].split(".")[0]))
            latest_model_path = os.path.join(log_dir, latest_model)
            start_episode = int(latest_model.split("_")[-1].split(".")[0])
            global_episode = start_episode
            print(f"Loading latest model: {latest_model}")
            agent.load_model(latest_model_path)
            if os.path.exists(replay_buffer_path):
                print(f"Loading replay buffer from {replay_buffer_path}")
                agent.load_replay_buffer(replay_buffer_path)
        else:
            pass

    abnormal_reward = 0
    max_steps = MAX_STEPS
    if DECAY_MODE == "learning_step":
        epsilon_scheduler = EpsilonScheduler(start_epsilon=start_epsilon, epsilon_min = EPSILON_MIN, start_decay_step = START_DECAY_LEARNING_STEP, scheduler_type=SCHEDULER_TYPE, decay_value=DECAY_VALUE, linear_decay_steps = LINEARLY_DECAY_LEARNING_STEP)
    else:
        epsilon_scheduler = EpsilonScheduler(start_epsilon=start_epsilon, epsilon_min = EPSILON_MIN, start_decay_step = START_DECAY_STEP, scheduler_type=SCHEDULER_TYPE, decay_value=DECAY_VALUE, linear_decay_steps = LINEARLY_DECAY_STEP)

    # ----- 3) Queue & Worker 프로세스 시작 -----
    N_WORKERS = N_ENVS # 원하는 만큼
    transition_queue = mp.Queue(maxsize=100*N_ENVS)
    stats_queue = mp.Queue(maxsize=100*N_ENVS)
    param_queue = mp.Queue(maxsize=1)

    workers = []
    base_seed = 1234
    param_queues = []
    for wid in range(N_WORKERS):
        pq = mp.Queue(maxsize=1)
        param_queues.append(pq)
        p = mp.Process(
            target=worker_process,
            args=(wid, transition_queue, stats_queue, epsilon_shared, pq, base_seed),
            daemon=True
        )
        p.start()
        workers.append(p)
        print(f"[Main] Worker {wid} started, pid={p.pid}")


    max_episodes = 9999999

    # update 비율 설정
    UPDATES_PER_TRANSITION = 1  # 예: transition 1개당 update 1번 정도
    pending_updates = 0.0

    sim_timer.reset()
    learn_timer.reset()

    # ----- 5) 메인 학습 루프 (완전 비동기) -----
    while global_episode < max_episodes:
        # 5-1) transition 소비 (block)
        try:
            msg: TransitionMsg = transition_queue.get(timeout=1.0)
        except queue.Empty:
            # transition이 잠시 없는 경우, stats만 처리하고 계속
            pass
        else:
            # ReplayBuffer에 push
            agent.replay_buffer.push(
                msg.ego_state, msg.global_state, msg.robot_state,
                msg.action, msg.reward,
                msg.next_ego_state, msg.next_global_state, msg.next_robot_state,
                msg.done
            )

            # 업데이트 스케줄: transition 수에 비례해서 update 횟수 누적
            if global_episode >= START_UPDATE_EPISODE:
                pending_updates += UPDATES_PER_TRANSITION
                while pending_updates >= 1.0:
                    learn_timer.start()
                    agent.update()
                    learn_timer.stop()
                    pending_updates -= 1.0

        # 5-2) stats_queue non-blocking 처리 (에피소드 통계, 스케줄 업데이트, 체크포인트 등)
        while True:
            try:
                s_msg: EpisodeStatMsg = stats_queue.get_nowait()
            except queue.Empty:
                break

            global_episode += 1
            print("-----------------------------------------------")
            print(f"[Main] Episode {global_episode} (from worker {s_msg.worker_id})")
            print("Total reward:", s_msg.total_reward)
            print("evacuation_time_80 :", s_msg.evac_time_80)
            print("evacuation_time_100:", s_msg.evac_time_100)
            print("-----------------------------------------------")

            # 맵별 metric 기록
            ensure_file(map_metric_path("reward", s_msg.map_num))
            ensure_file(map_metric_path("evacuation_100", s_msg.map_num))

            if s_msg.abnormal != 1:
                with open(map_metric_path("reward", s_msg.map_num), "a") as f:
                    f.write(f"{s_msg.total_reward}\n")
                with open(map_metric_path("evacuation_100", s_msg.map_num), "a") as f:
                    f.write(f"{s_msg.evac_time_100}\n")

                with open(total_reward_file, "a") as f:
                    f.write(f"{s_msg.total_reward}\n")
                with open(evacuation_time_80_file, "a") as f:
                    f.write(f"{s_msg.evac_time_80}\n")
                with open(evacuation_time_100_file, "a") as f:
                    f.write(f"{s_msg.evac_time_100}\n")
                with open(total_lifetime_file, "a") as f:
                    f.write(f"{s_msg.total_lifetime}\n")

            # alpha/gamma/epsilon 스케줄 업데이트 (episode 기준)
            agent.update_alpha(ALPHA_START, ALPHA_END, ALPHA_DECAY_STEPS, global_episode)
            agent.update_gamma(GAMMA_START, GAMMA_END, GAMMA_SCHEDULE_STEP, global_episode)
            agent.epsilon = max(
                epsilon_scheduler.get_epsilon(agent.epsilon, global_episode),
                EPSILON_MIN
            )

            with epsilon_shared.get_lock():
                epsilon_shared.value = float(agent.epsilon)

            # epsilon 저장
            with open(epsilon_path, "w") as f:
                f.write(str(agent.epsilon) + "\n")
                f.write(str(agent.epsilon_long))

            # 체크포인트 저장
            if global_episode % 100 == 0:
                model_filename = os.path.join(log_dir, f"sac_checkpoint_ep_{global_episode}.pth")
                agent.save_model(model_filename)
                agent.save_replay_buffer("replay_buffer.npz")

            if ENABLE_TIMER:
                print(f"Episode {global_episode} - Total Learning Time: {learn_timer.get_time():.6f} 초")
                learn_timer.reset()
        
        if global_episode % POLICY_BROADCAST_INTERVAL == 0:
            sd_cpu = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}

            for pq in param_queues:
                # 큐를 완전히 비워서 최신만 남김
                try:
                    while True:
                        pq.get_nowait()
                except Empty:
                    pass

                # block 방지
                try:
                    pq.put_nowait(sd_cpu)
                except Full:
                    pass   # 이번 broadcast는 스킵
                    