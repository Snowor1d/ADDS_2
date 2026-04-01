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


from pathlib import Path
import imageio.v2 as imageio

DEBUG_SAVE = False
home_dir = os.path.expanduser("~")
DEBUG_DIR_TEMP = os.path.join(home_dir, LOG_DIR)
DEBUG_DIR = os.path.join(DEBUG_DIR_TEMP, "debug_frames")
DEBUG_EVERY_EP = 1  
DEBUG_STEPS = {100, 200, 300}  # 초반 3번 boundary만 저장



def save_debug_triplet(save_dir: str, worker_id: int, episode_idx: int, step: int,
                       full_u8: np.ndarray,
                       ego_f: np.ndarray,
                       glob_f: np.ndarray,
                       ego_state: np.ndarray = None,
                       global_state: np.ndarray = None):
    """
    full_u8: (H,W) uint8
    ego_f, glob_f: float32 [0,1] single frame
    ego_state/global_state: (4,H,W) float32 [0,1] stack (optional)
    """
    d = Path(save_dir) / f"w{worker_id:02d}"
    d.mkdir(parents=True, exist_ok=True)

    # base name
    stem = f"ep{episode_idx:06d}_st{step:06d}"

    # full map 저장 (uint8 그대로)
    imageio.imwrite(str(d / f"{stem}_full.png"), full_u8)

    # ego/global frame 저장 (float -> uint8 변환)
    ego_u8 = np.clip(ego_f * 255.0, 0, 255).astype(np.uint8)
    glob_u8 = np.clip(glob_f * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(d / f"{stem}_ego.png"), ego_u8)
    imageio.imwrite(str(d / f"{stem}_glob.png"), glob_u8)

    # stack 확인(옵션): 채널 4장을 한 이미지로 합쳐서 저장
    if ego_state is not None:
        # (4,H,W) -> 가로로 붙이기
        es = np.clip(ego_state * 255.0, 0, 255).astype(np.uint8)
        grid = np.concatenate([es[i] for i in range(es.shape[0])], axis=1)
        imageio.imwrite(str(d / f"{stem}_egoStack.png"), grid)

    if global_state is not None:
        gs = np.clip(global_state * 255.0, 0, 255).astype(np.uint8)
        grid = np.concatenate([gs[i] for i in range(gs.shape[0])], axis=1)
        imageio.imwrite(str(d / f"{stem}_globStack.png"), grid)


@dataclass
class TransitionMsg:
    worker_id: int
    ego_state: np.ndarray
    global_state: np.ndarray
    robot_state: np.ndarray
    action: np.ndarray
    log_prob: float
    value: float
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

HEARTBEAT_PATH = os.path.join(log_dir, "heartbeat.txt")

def write_heartbeat(ep: int):
    tmp = HEARTBEAT_PATH
    with open(tmp, "w") as f:
        f.write(f"{ep}\n")
        f.write(f"{time.time()}\n")
    os.replace(tmp, HEARTBEAT_PATH)  # atomic replace


def ego_crop_from_full_map(full_map: np.ndarray,
                           robot_xy_px: tuple[int, int],
                           ego_size: int,
                           pad_value: int = 50) -> np.ndarray:
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

def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = list(values) + [next_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

def worker_process(
    worker_id: int,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    param_queue: mp.Queue,
    seed: int = 0,
):
    """
    PPO용 worker 프로세스
    - actor / critic 파라미터를 param_queue로부터 받아 사용
    - ACTION_SCALE 간격으로 action, old_log_prob, value를 계산
    - rollout transition을 transition_queue에 전송
    - episode 종료 시 통계를 stats_queue에 전송
    """

    import model  # spawn/fork 안전성 때문에 내부 import

    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    max_steps = MAX_STEPS
    episode_idx = 0
    device = torch.device("cpu")

    actor = PPOActor(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)

    critic = ValueNetwork(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)

    actor.eval()
    critic.eval()

    def _robot_world_to_px(env_model):
        rx, ry = env_model.robot.xy
        ix = int(np.clip(rx / env_model.width * MAP_W, 0, MAP_W - 1))
        iy = int(np.clip(ry / env_model.height * MAP_H, 0, MAP_H - 1))
        return ix, iy

    def _build_ego_global_frames(env_model):
        full = env_model.return_current_image(MAP_H, MAP_W)  # (H, W) uint8
        ix, iy = _robot_world_to_px(env_model)

        ego = ego_crop_from_full_map(full, (ix, iy), EGO_MAP_SIZE, pad_value=50)
        glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)

        ego_f = ego.astype(np.float32) / 255.0
        glob_f = glob.astype(np.float32) / 255.0
        return full, ego_f, glob_f

    while True:
        # --------------------------------------------------
        # 1) 환경 생성
        # --------------------------------------------------
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

        # --------------------------------------------------
        # 2) 초기 state
        # --------------------------------------------------
        full_u8, ego_f, glob_f = _build_ego_global_frames(env_model)

        ego_stack = FrameStack2(4)
        glob_stack = FrameStack2(4)

        ego_state = ego_stack.reset(ego_f)
        global_state = glob_stack.reset(glob_f)
        robot_state = np.array(env_model.return_current_robot_state(), dtype=np.float32)

        total_reward = 0.0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps
        agent_total_lifetime = 0.0
        abnormal_reward = 0

        buffered_ego_state = np.copy(ego_state)
        buffered_global_state = np.copy(global_state)
        buffered_robot_state = np.copy(robot_state)
        buffered_action = np.zeros((2,), dtype=np.float32)
        buffered_log_prob = 0.0
        buffered_value = 0.0

        try:
            # --------------------------------------------------
            # 3) 최신 파라미터 반영
            # --------------------------------------------------
            received_new_param = False
            try:
                while True:
                    new_param = param_queue.get_nowait()

                    if not isinstance(new_param, tuple):
                        raise ValueError(
                            "PPO worker expected actor/critic params, but got single state_dict"
                        )

                    if len(new_param) == 3:
                        _, actor_sd, critic_sd = new_param
                    elif len(new_param) == 2:
                        actor_sd, critic_sd = new_param
                    else:
                        raise ValueError(f"Unexpected param tuple length: {len(new_param)}")

                    actor.load_state_dict(actor_sd)
                    critic.load_state_dict(critic_sd)
                    received_new_param = True

            except Empty:
                pass
            except queue.Empty:
                pass

            if received_new_param:
                actor.eval()
                critic.eval()

            # --------------------------------------------------
            # 4) episode loop
            # --------------------------------------------------
            for step in range(max_steps):
                if step % ACTION_SCALE == 0:
                    full_u8, ego_f, glob_f = _build_ego_global_frames(env_model)

                    if DEBUG_SAVE:
                        full_u8_r = np.flip(np.flip(full_u8, axis=-1), axis=-2)
                        ego_f_r = np.flip(np.flip(ego_f, axis=-1), axis=-2)
                        glob_f_r = np.flip(np.flip(glob_f, axis=-1), axis=-2)

                        save_debug_triplet(
                            save_dir=DEBUG_DIR,
                            worker_id=worker_id,
                            episode_idx=episode_idx,
                            step=step,
                            full_u8=np.flip(full_u8_r, axis=1),
                            ego_f=np.flip(ego_f_r, axis=1),
                            glob_f=np.flip(glob_f_r, axis=1),
                            ego_state=ego_state,
                            global_state=global_state
                        )

                    if step > 0:
                        ego_state = ego_stack.append(ego_f)
                        global_state = glob_stack.append(glob_f)

                    ego_t = torch.from_numpy(ego_state).unsqueeze(0).float().to(device)
                    glob_t = torch.from_numpy(global_state).unsqueeze(0).float().to(device)
                    robot_t = torch.from_numpy(robot_state).unsqueeze(0).float().to(device)

                    
                    with torch.no_grad():
                        raw_action_t, original_log_prob_t, _ = actor.sample_action(ego_t, glob_t, robot_t)
                        value_t = critic(ego_t, glob_t, robot_t)
                    raw_action_np = raw_action_t.cpu().numpy()[0].astype(np.float32)

                    env_action = 4.0 * (1.0 / (1.0 + np.exp(-raw_action_np))) - 2.0
                    env_model.robot.receive_action(env_action.tolist())

                    buffered_ego_state = np.copy(ego_state)
                    buffered_global_state = np.copy(global_state)
                    buffered_robot_state = np.copy(robot_state)
                    buffered_action = np.copy(raw_action_np)
                    buffered_log_prob = float(original_log_prob_t.item())
                    buffered_value = float(value_t.item())

                # --------------------------------------------------
                # 5) env step
                # --------------------------------------------------
                env_model.step()

                # --------------------------------------------------
                # 6) next state 계산
                # --------------------------------------------------
                next_robot_state = np.array(env_model.return_current_robot_state(), dtype=np.float32)
                _, ego_f2, glob_f2 = _build_ego_global_frames(env_model)
                next_ego_state = ego_stack.peek_with(ego_f2)
                next_global_state = glob_stack.peek_with(glob_f2)

                # --------------------------------------------------
                # 7) done / reward 계산
                # --------------------------------------------------
                done = (step >= max_steps - 1) or (env_model.robot.is_game_finished)
                reward = 0.0
                r_k = 0.0

                if env_model.robot.is_game_finished:
                    reward += FINISHED_BONUS * (1 - step / max_steps)

                if REWARD_K:
                    r_k += env_model.reward_penalty_collision() * REWARD_K

                should_emit_transition = (
                    (step % ACTION_SCALE == (ACTION_SCALE - 1) and step >= ACTION_SCALE - 1) or
                    done
                )

                if should_emit_transition:
                    r_a = r_b = r_c = r_d = r_e = 0.0
                    r_f = r_g = r_h = r_i = r_j = r_l = 0.0

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

                    reward += (
                        r_a + r_b + r_c + r_d + r_e +
                        r_f + r_g + r_h + r_i + r_j +
                        r_k + r_l + REWARD_FIXED
                    )

                    if reward < -1e3:
                        raise RuntimeError(f"Reward collapsed: {reward}")

                    msg = TransitionMsg(
                        worker_id=worker_id,
                        ego_state=np.copy(buffered_ego_state),
                        global_state=np.copy(buffered_global_state),
                        robot_state=np.copy(buffered_robot_state),
                        action=np.copy(buffered_action),
                        log_prob=float(buffered_log_prob),
                        value=float(buffered_value),
                        reward=float(reward),
                        next_ego_state=np.copy(next_ego_state),
                        next_global_state=np.copy(next_global_state),
                        next_robot_state=np.copy(next_robot_state),
                        done=bool(done),
                    )

                    try:
                        transition_queue.put(msg)
                        total_reward += reward
                    except Exception as e:
                        print(f"[Worker {worker_id}] transition_queue.put error: {e}")
                        abnormal_reward = 1

                    robot_state = next_robot_state
                else:
                    robot_state = next_robot_state

                # --------------------------------------------------
                # 8) 통계
                # --------------------------------------------------
                if (
                    env_model.alived_agents() < env_model.total_agents * 0.2
                    and evacuation_time_80 == max_steps
                ):
                    evacuation_time_80 = step

                if (
                    env_model.alived_agents() < 1
                    and evacuation_time_100 == max_steps
                ):
                    evacuation_time_100 = step

                if done:
                    try:
                        agent_total_lifetime = env_model.calculate_all_agents_life_time()
                    except Exception:
                        agent_total_lifetime = 0.0
                    break

        except Exception as e:
            print(f"[Worker {worker_id}] Error in episode loop: {e}")
            import traceback
            traceback.print_exc()
            abnormal_reward = 1

        # --------------------------------------------------
        # 9) episode 통계 전송
        # --------------------------------------------------
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
                
class PPORolloutBuffer:
    def __init__(self, max_workers=100):
        self.max_workers = max_workers
        self.clear()

    def clear(self):
        # worker_id를 키로 가지는 딕셔너리로 초기화
        self.buffers = {i: {
            "ego": [], "glob": [], "rob": [], "act": [], "logp": [],
            "rew": [], "done": [], "val": [], 
            "next_ego": [], "next_glob": [], "next_rob": []
        } for i in range(self.max_workers)}
        self.total_steps = 0

    def add(self, worker_id, ego_state, global_state, robot_state, action, log_prob, reward, done, value, next_ego_state, next_global_state, next_robot_state):
        b = self.buffers[worker_id]
        b["ego"].append(np.array(ego_state, copy=True))
        b["glob"].append(np.array(global_state, copy=True))
        b["rob"].append(np.array(robot_state, copy=True))
        b["act"].append(np.array(action, copy=True))
        b["logp"].append(float(log_prob))
        b["rew"].append(float(reward))
        b["done"].append(float(done))
        b["val"].append(float(value))
        b["next_ego"].append(np.array(next_ego_state, copy=True))
        b["next_glob"].append(np.array(next_global_state, copy=True))
        b["next_rob"].append(np.array(next_robot_state, copy=True))
        
        self.total_steps += 1

   


##########################################################################
# 1) IMPALA CNN Components
##########################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.silu(x)
        out = self.conv1(out)
        out = F.silu(out)
        out = self.conv2(out)
        return out + x  # Skip connection

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 이미지 크기를 절반으로 줄이는 Max Pooling (stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class ImpalaCNN(nn.Module):
    # compress=False 로 설정하면 256 압축을 생략합니다.
    def __init__(self, input_shape, in_channels=4, channels=[32, 64, 64], out_dim=256, compress=True):
        super(ImpalaCNN, self).__init__()
        h, w = input_shape
        self.blocks = nn.ModuleList()
        
        cur_channels = in_channels
        for c in channels:
            self.blocks.append(ImpalaBlock(cur_channels, c))
            cur_channels = c
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1

        self.flatten_dim = cur_channels * h * w
        self.compress = compress
        
        if self.compress:
            self.fc = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.flatten_dim, out_dim),
                nn.SiLU()
            )
            self.out_dim = out_dim
        else:
            # 압축하지 않을 경우 Identity(아무것도 안 함)를 통과시키고 원래 차원 반환
            self.fc = nn.Identity()
            self.out_dim = self.flatten_dim

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class CNNEncoder(nn.Module):
    def __init__(self, input_shape=(50, 50), in_channels=4):
        super().__init__()
        # 기존과 동일한 구조 유지
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
       
        # 출력 차원 계산
        self.out_dim = self._get_conv_out(input_shape, in_channels)

    def _get_conv_out(self, shape, in_channels):
        dummy = torch.zeros(1, in_channels, *shape)
        o = F.silu(self.bn1(self.conv1(dummy)))
        o = F.silu(self.bn2(self.conv2(o)))
        o = F.silu(self.bn3(self.conv3(o)))
        return int(np.prod(o.size()[1:]))

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1) # Flatten
        return x


class PPOActor(nn.Module):
    def __init__(self, ego_shape=(25,25), global_shape=(50,50), robot_dim=3, use_robot=True):
        super().__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.use_robot_state = use_robot

        self.ego_enc = CNNEncoder(input_shape=ego_shape, in_channels=4) if EGO_USE else None
        self.glob_enc = CNNEncoder(input_shape=global_shape, in_channels=4)

        self.robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_dim, 32),
                nn.SiLU()
            )
            self.robot_feat_dim = 32

        fusion_dim = self.glob_enc.out_dim + self.robot_feat_dim
        if EGO_USE:
            fusion_dim += self.ego_enc.out_dim

        self.fc_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU()
        )

        self.mean_head = nn.Linear(64, 2)
        self.action_log_std = nn.Parameter(torch.zeros(2))

    def backbone(self, ego_state, global_state, robot_state=None):
        feats = []
        if EGO_USE:
            feats.append(self.ego_enc(ego_state))
        feats.append(self.glob_enc(global_state))

        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("robot_state is required")
            feats.append(self.robot_fc(robot_state))

        return self.fc_backbone(torch.cat(feats, dim=1))

    def forward(self, ego_state, global_state, robot_state=None):
        feat = self.backbone(ego_state, global_state, robot_state)
        mean = self.mean_head(feat)
        log_std = self.action_log_std.expand_as(mean)
        return mean, log_std

    def _squash_action(self, u):
        sigma = torch.sigmoid(u)
        return 4.0 * sigma - 2.0   # [-2, 2]

    def sample_action(self, ego_state, global_state, robot_state=None):
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate_actions(self, ego_state, global_state, robot_state, action):
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)

        # [수정됨] Inverse Jacobian 제거. 저장된 raw action으로 바로 확률 계산
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy

class ValueNetwork(nn.Module):
    def __init__(self, ego_shape=(25,25), global_shape=(50,50), robot_dim=3, use_robot=True):
        super().__init__()
        self.use_robot_state = use_robot

        self.ego_enc = CNNEncoder(input_shape=ego_shape, in_channels=4) if EGO_USE else None
        self.glob_enc = CNNEncoder(input_shape=global_shape, in_channels=4)

        self.robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_dim, 32),
                nn.SiLU()
            )
            self.robot_feat_dim = 32

        fusion_dim = self.glob_enc.out_dim + self.robot_feat_dim
        if EGO_USE:
            fusion_dim += self.ego_enc.out_dim

        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, ego_state, global_state, robot_state=None):
        feats = []
        if EGO_USE:
            feats.append(self.ego_enc(ego_state))
        feats.append(self.glob_enc(global_state))

        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("robot_state is required")
            feats.append(self.robot_fc(robot_state))

        x = torch.cat(feats, dim=1)
        return self.value_head(x).squeeze(-1)


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

class PPOAgent:
    def __init__(
        self,
        device="cpu",
        gamma=GAMMA,
        gae_lambda=0.95,
        clip_eps=0.2,
        lr_actor=1e-4,
        lr_critic=1e-4,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        value_coef=PPO_VALUE_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor = PPOActor(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.critic = ValueNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=WD_PI)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=WD_Q)

    def select_action(self, ego_state_np, global_state_np, robot_state_np):
        ego_t = torch.FloatTensor(ego_state_np).unsqueeze(0).to(self.device)
        glob_t = torch.FloatTensor(global_state_np).unsqueeze(0).to(self.device)
        robot_t = torch.FloatTensor(robot_state_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, _ = self.actor.sample_action(ego_t, glob_t, robot_t)
            value = self.critic(ego_t, glob_t, robot_t)

        return (
            action.cpu().numpy()[0].astype(np.float32),
            float(log_prob.item()),
            float(value.item())
        )
    
    def get_value(self, ego_state_np, global_state_np, robot_state_np):
        ego_t = torch.FloatTensor(ego_state_np).unsqueeze(0).to(self.device)
        glob_t = torch.FloatTensor(global_state_np).unsqueeze(0).to(self.device)
        robot_t = torch.FloatTensor(robot_state_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(ego_t, glob_t, robot_t)

        return float(value.item())

    def update(self, rollout: PPORolloutBuffer):
        all_ego, all_glob, all_rob = [], [], []
        all_act, all_logp, all_ret, all_adv = [], [], [], []

        # 워커별로 순회하며 독립적으로 GAE 계산
        for wid, b in rollout.buffers.items():
            if len(b["rew"]) == 0:
                continue
                
            # 해당 워커의 마지막 next_value 계산
            if b["done"][-1]:
                next_value = 0.0
            else:
                next_value = self.get_value(
                    b["next_ego"][-1], b["next_glob"][-1], b["next_rob"][-1]
                )

            # 이 워커의 데이터만으로 GAE 계산 (순서 보장됨!)
            advs, rets = compute_gae(
                b["rew"], b["done"], b["val"], next_value, self.gamma, self.gae_lambda
            )

            # 통합 리스트에 추가
            all_ego.extend(b["ego"])
            all_glob.extend(b["glob"])
            all_rob.extend(b["rob"])
            all_act.extend(b["act"])
            all_logp.extend(b["logp"])
            all_ret.extend(rets)
            all_adv.extend(advs)

        # numpy 배열을 Tensor로 변환
        ego_states = torch.FloatTensor(np.array(all_ego)).to(self.device)
        global_states = torch.FloatTensor(np.array(all_glob)).to(self.device)
        robot_states = torch.FloatTensor(np.array(all_rob)).to(self.device)
        actions = torch.FloatTensor(np.array(all_act)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(all_logp)).to(self.device)
        returns = torch.FloatTensor(np.array(all_ret)).to(self.device)
        advantages = torch.FloatTensor(np.array(all_adv)).to(self.device)

        # 전체 배치에 대해 Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 이후 PPO 미니배치 업데이트 로직은 기존과 동일 ---
        n = ego_states.size(0)

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, self.mini_batch_size):
                mb_idx = idx[start:start + self.mini_batch_size]

                mb_ego = ego_states[mb_idx]
                mb_glob = global_states[mb_idx]
                mb_robot = robot_states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_log_probs, entropy = self.actor.evaluate_actions(
                    mb_ego, mb_glob, mb_robot, mb_actions
                )

                values = self.critic(mb_ego, mb_glob, mb_robot).squeeze(-1) 
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

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
# PPO Example usage in your training loop
##########################################################################
if __name__ == "__main__":
    import time

    def stop_all_workers(workers):
        for wid, p in enumerate(workers):
            if p is None:
                continue
            try:
                if p.is_alive():
                    p.terminate()
                p.join(timeout=2)
            except Exception as e:
                print(f"[Main] stop worker {wid} error: {e}")

    def start_workers(ctx, n_workers, transition_queue, stats_queue, base_seed):
        workers = [None] * n_workers
        param_queues = [None] * n_workers
        for wid in range(n_workers):
            p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, base_seed)
            workers[wid] = p
            param_queues[wid] = pq
        return workers, param_queues

    def start_one_worker(ctx, wid, transition_queue, stats_queue, base_seed):
        pq = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=worker_process,
            args=(wid, transition_queue, stats_queue, pq, base_seed),
            daemon=True
        )
        p.start()
        print(f"[Main] Worker {wid} started, pid={p.pid}")
        return p, pq

    def restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, base_seed):
        old_p = workers[wid]
        try:
            if old_p is not None and old_p.is_alive():
                old_p.terminate()
                old_p.join(timeout=2)
        except Exception as e:
            print(f"[Main] terminate/join error for worker {wid}: {e}")

        p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, base_seed)
        workers[wid] = p
        param_queues[wid] = pq
        return

    def supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, base_seed):
        for wid, p in enumerate(workers):
            if p is None:
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, base_seed)
                continue

            if not p.is_alive():
                ec = p.exitcode
                print(f"[Main] Worker {wid} died. exitcode={ec} -> restarting")
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, base_seed)

    def broadcast_actor_critic(agent, param_queues):
        actor_sd = {k: v.detach().cpu() for k, v in agent.actor.state_dict().items()}
        critic_sd = {k: v.detach().cpu() for k, v in agent.critic.state_dict().items()}

        for pq in param_queues:
            try:
                while True:
                    pq.get_nowait()
            except Empty:
                pass
            except queue.Empty:
                pass

            try:
                pq.put_nowait(("ppo", actor_sd, critic_sd))
            except Full:
                pass
            except queue.Full:
                pass

    mp.set_start_method("spawn", force=True)

    # ----- 1) TensorBoard/로그 파일 설정 -----
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")
    total_lifetime_file = os.path.join(log_dir, "total_lifetime.txt")

    step_writer = SummaryWriter(log_dir=tb_log_dir)

    for path in [total_reward_file, evacuation_time_80_file,
                 evacuation_time_100_file, total_lifetime_file]:
        if not os.path.exists(path):
            open(path, "w").close()

    tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)

    monitor_thread = threading.Thread(
        target=monitor_total_reward,
        args=(total_reward_file, tb_log_dir),
        daemon=True
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

    monitor_thread_lifetime = threading.Thread(
        target=monitor_metric,
        args=(total_lifetime_file, "Total Lifetime", tb_log_dir),
        daemon=True
    )
    monitor_thread_lifetime.start()

    for m in MAP_NUM_RANDOM:
        reward_map_file = map_metric_path("reward", m)
        ensure_file(reward_map_file)
        threading.Thread(
            target=monitor_metric,
            args=(reward_map_file, f"Reward/map_{m}", tb_log_dir),
            daemon=True
        ).start()

        evac100_map_file = map_metric_path("evacuation_100", m)
        ensure_file(evac100_map_file)
        threading.Thread(
            target=monitor_metric,
            args=(evac100_map_file, f"Evacuation Time 100/map_{m}", tb_log_dir),
            daemon=True
        ).start()

    max_episodes = 9999999
    start_episode = 0
    global_episode = 0

    # -------------------------------
    # PPO Agent
    # -------------------------------
    agent = PPOAgent(
        device=DEVICE,
        gamma=float(GAMMA),
        gae_lambda=0.95,
        clip_eps=0.2,
        lr_actor=float(LR),
        lr_critic=float(LR),
        ppo_epochs=int(PPO_EPOCHS) if 'PPO_EPOCHS' in globals() else 10,
        mini_batch_size=int(PPO_MINI_BATCH_SIZE) if 'PPO_MINI_BATCH_SIZE' in globals() else 256,
        value_coef=float(PPO_VALUE_COEF) if 'PPO_VALUE_COEF' in globals() else 0.5,
        entropy_coef=float(PPO_ENTROPY_COEF) if 'PPO_ENTROPY_COEF' in globals() else 0.01,
        max_grad_norm=float(PPO_MAX_GRAD_NORM) if 'PPO_MAX_GRAD_NORM' in globals() else 0.5,
    )

    print(f"[Main] PPOAgent initialized, lr={LR}")

    # checkpoint 경로
    latest_ckpt_path = os.path.join(log_dir, "ppo_latest.pth")

    # -------------------------------
    # 모델 로드
    # -------------------------------
    if model_load == 1:
        pass

    elif model_load == 2:
        print("[Main] load specified PPO model")
        model_name = "ppo_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=DEVICE)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            start_episode = int(ckpt.get("episode", 0))
            global_episode = start_episode

    elif model_load == 3:
        print("[Main] Mode 3: Loading latest PPO model")
        if os.path.exists(latest_ckpt_path):
            ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            start_episode = int(ckpt.get("episode", 0))
            global_episode = start_episode
            print(f"[Main] Loaded latest PPO checkpoint, episode={global_episode}")

    abnormal_reward = 0
    max_steps = MAX_STEPS

    # ----- 3) Queue & Worker 프로세스 시작 -----
    ctx = mp.get_context("spawn")
    N_WORKERS = N_ENVS

    N_WORKERS_WARMUP = 10
    N_WORKERS_TRAIN = int(N_ENVS)

    transition_queue = ctx.Queue(maxsize=2 * N_WORKERS_WARMUP)
    stats_queue = ctx.Queue(maxsize=2 * N_WORKERS_WARMUP)

    workers = [None] * N_WORKERS
    param_queues = [None] * N_WORKERS
    base_seed = 1234

    if global_episode >= START_UPDATE_EPISODE:
        current_n_workers = N_WORKERS_TRAIN
    else:
        current_n_workers = N_WORKERS_WARMUP

    workers, param_queues = start_workers(
        ctx, current_n_workers, transition_queue, stats_queue, base_seed
    )
    print(f"[Main] Started {current_n_workers} workers.")

    # worker 시작 직후 actor/critic 전달
    broadcast_actor_critic(agent, param_queues)

    # -------------------------------
    # PPO rollout buffer
    # -------------------------------
    rollout = PPORolloutBuffer()
    ROLLOUT_STEPS = int(PPO_ROLLOUT_STEPS) if 'PPO_ROLLOUT_STEPS' in globals() else 4096

    sim_timer.reset()
    learn_timer.reset()
    last_supervise_t = time.time()
    write_heartbeat(global_episode)

    while global_episode < max_episodes:
        if time.time() - last_supervise_t > 10.0:
            supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, base_seed)
            last_supervise_t = time.time()

        # --------------------------------------------------
        # 1) transition 수집
        # --------------------------------------------------

        try:
            msg: TransitionMsg = transition_queue.get(timeout=1.0)
        except queue.Empty:
            pass
        else:
            # 웜업 중에는 데이터 수집 안 함 (메모리 누수 완벽 방지!)
            if global_episode < START_UPDATE_EPISODE:
                continue
            
            # ★ 수정 1: worker_id 추가
            rollout.add(
                worker_id=msg.worker_id, 
                ego_state=msg.ego_state,
                global_state=msg.global_state,
                robot_state=msg.robot_state,
                action=msg.action,
                log_prob=msg.log_prob,
                reward=msg.reward,
                done=msg.done,
                value=msg.value,
                next_ego_state=msg.next_ego_state,
                next_global_state=msg.next_global_state,
                next_robot_state=msg.next_robot_state,
            )
            # rollout_steps_collected += 1  <- 이 줄은 삭제하세요 (rollout.total_steps가 대신합니다)

            # ★ 수정 2: 깔끔해진 업데이트 로직 (중복 제거)
            if rollout.total_steps >= ROLLOUT_STEPS:
                learn_timer.start()
                
                # 내부에서 워커별로 GAE를 계산하도록 수정한 새로운 update 사용
                agent.update(rollout)
                
                learn_timer.stop()
                rollout.clear()
                
                # 업데이트 후 최신 actor/critic 다시 broadcast
                broadcast_actor_critic(agent, param_queues)

        needs_switch_workers = False

        # --------------------------------------------------
        # 2) stats 처리
        # --------------------------------------------------
        while True:
            try:
                s_msg: EpisodeStatMsg = stats_queue.get_nowait()
            except queue.Empty:
                break

            write_heartbeat(global_episode)

            if (global_episode >= START_UPDATE_EPISODE) and (current_n_workers != N_WORKERS_TRAIN):
                needs_switch_workers = True

            global_episode += 1

            print("-----------------------------------------------")
            print(f"[Main] Episode {global_episode} (from worker {s_msg.worker_id})")
            print("Total reward:", s_msg.total_reward)
            print("evacuation_time_80 :", s_msg.evac_time_80)
            print("evacuation_time_100:", s_msg.evac_time_100)
            print("-----------------------------------------------")

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

            # checkpoint 저장
            if (global_episode >= START_UPDATE_EPISODE) and (global_episode % 100 == 0):
                ckpt = {
                    "episode": global_episode,
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "actor_optimizer": agent.actor_optimizer.state_dict(),
                    "critic_optimizer": agent.critic_optimizer.state_dict(),
                }

                ep_ckpt_path = os.path.join(log_dir, f"ppo_checkpoint_ep_{global_episode}.pth")
                torch.save(ckpt, ep_ckpt_path)
                torch.save(ckpt, latest_ckpt_path)
                print(f"[Main] PPO checkpoint saved: {ep_ckpt_path}")

            if ENABLE_TIMER:
                print(f"Episode {global_episode} - Total Learning Time: {learn_timer.get_time():.6f} 초")
                learn_timer.reset()

        # --------------------------------------------------
        # 3) warmup -> full training worker 전환
        # --------------------------------------------------
        if needs_switch_workers:
            print(f"[Main] Reaching START_UPDATE_EPISODE={START_UPDATE_EPISODE}. "
                  f"Switching workers {current_n_workers} -> {N_WORKERS_TRAIN}")

            print("[Main] Draining queues before stopping workers...")
            try:
                while not transition_queue.empty():
                    transition_queue.get_nowait()
                while not stats_queue.empty():
                    stats_queue.get_nowait()
            except Exception:
                pass

            stop_all_workers(workers)

            try:
                transition_queue.close()
                stats_queue.close()
                transition_queue.join_thread()
                stats_queue.join_thread()
                time.sleep(1.0)
            except Exception:
                pass

            print("[Main] Re-creating Queues...")
            transition_queue = ctx.Queue(maxsize=2 * N_WORKERS_TRAIN)
            stats_queue = ctx.Queue(maxsize=2 * N_WORKERS_TRAIN)

            current_n_workers = N_WORKERS_TRAIN
            workers, param_queues = start_workers(
                ctx, current_n_workers,
                transition_queue, stats_queue,
                base_seed
            )

            broadcast_actor_critic(agent, param_queues)

            last_supervise_t = time.time()
            continue

        # --------------------------------------------------
        # 4) 주기적 policy broadcast
        # --------------------------------------------------
        if (global_episode >= START_UPDATE_EPISODE) and (global_episode % POLICY_BROADCAST_INTERVAL == 0):
            broadcast_actor_critic(agent, param_queues)