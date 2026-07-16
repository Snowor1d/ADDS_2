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
#import cv2

from pathlib import Path
import imageio.v2 as imageio
import torch.distributions as dist
from dreamer_v3 import DreamerAgent, DreamerConfig

DEBUG_SAVE = False
home_dir = os.path.expanduser("~")
DEBUG_DIR_TEMP = os.path.join(home_dir, LOG_DIR)
DEBUG_DIR = os.path.join(DEBUG_DIR_TEMP, "debug_frames")
DEBUG_EVERY_EP = 1  
DEBUG_STEPS = {100, 200, 300}  # 초반 3번 boundary만 저장
MAX_ROBOTS = 3
ACTION_DIM = 2

def save_debug_triplet(save_dir: str, worker_id: int, episode_idx: int, step: int, agent_idx: int,
                       full_u8: np.ndarray,
                       ego_f: np.ndarray,
                       glob_f: np.ndarray,
                       ego_state: np.ndarray = None,
                       global_state: np.ndarray = None):
    """
    agent_idx를 추가로 받아 여러 로봇의 이미지를 충돌 없이 저장합니다.
    """
    import imageio
    import numpy as np
    from pathlib import Path

    d = Path(save_dir) / f"w{worker_id:02d}"
    d.mkdir(parents=True, exist_ok=True)

    # 공통 파일명 (Global, Full Map용)
    step_stem = f"ep{episode_idx:06d}_st{step:06d}"
    # 개별 로봇 파일명 (Ego Map용)
    agent_stem = f"{step_stem}_agent{agent_idx:02d}"

    # 1. 공통 이미지 저장 (동일한 스텝에서 여러 번 호출되더라도 덮어쓰기 되므로 문제없음)
    imageio.imwrite(str(d / f"{step_stem}_full.png"), full_u8)
    
    glob_u8 = np.clip(glob_f * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(d / f"{step_stem}_glob.png"), glob_u8)

    if global_state is not None:
        gs = np.clip(global_state * 255.0, 0, 255).astype(np.uint8)
        grid_glob = np.concatenate([gs[i] for i in range(gs.shape[0])], axis=1)
        imageio.imwrite(str(d / f"{step_stem}_globStack.png"), grid_glob)

    # 2. 개별 로봇 이미지 저장 (파일명에 agent_idx가 들어가서 충돌 안 함)
    ego_u8 = np.clip(ego_f * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(d / f"{agent_stem}_ego.png"), ego_u8)

    if ego_state is not None:
        es = np.clip(ego_state * 255.0, 0, 255).astype(np.uint8)
        grid_ego = np.concatenate([es[i] for i in range(es.shape[0])], axis=1)
        imageio.imwrite(str(d / f"{agent_stem}_egoStack.png"), grid_ego)

def gather_agent_tensor(x, agent_index):
    """
    x: (B, N, ...)
    agent_index: (B,)
    return: (B, ...)
    """
    B = x.shape[0]
    view_shape = [B, 1] + [1] * (x.ndim - 2)
    expand_shape = [B, 1] + list(x.shape[2:])
    idx = agent_index.view(*view_shape).expand(*expand_shape)
    return x.gather(1, idx).squeeze(1)

def random_permute_joint_batch(
    joint_ego,
    joint_robot,
    joint_action,
    joint_mask,
    next_joint_ego,
    next_joint_robot,
    next_joint_mask,
    agent_index,
):
    """
    joint_ego:       (B, N, 4, E, E)
    joint_robot:     (B, N, R)
    joint_action:    (B, N, A)
    joint_mask:      (B, N)
    next_joint_ego:  (B, N, 4, E, E)
    next_joint_robot:(B, N, R)
    next_joint_mask: (B, N)
    agent_index:     (B,)

    Returns permuted tensors + updated agent_index
    """
    device = joint_ego.device
    B, N = joint_mask.shape

    perms = torch.stack([torch.randperm(N, device=device) for _ in range(B)], dim=0)  # (B,N)

    def permute_tensor(x):
        # x shape: (B, N, ...)
        idx = perms
        while idx.ndim < x.ndim:
            idx = idx.unsqueeze(-1)
        expand_shape = list(x.shape)
        expand_shape[1] = N
        idx = idx.expand(*expand_shape)
        return x.gather(1, idx)

    joint_ego_p = permute_tensor(joint_ego)
    joint_robot_p = permute_tensor(joint_robot)
    joint_action_p = permute_tensor(joint_action)
    joint_mask_p = permute_tensor(joint_mask.unsqueeze(-1)).squeeze(-1)

    next_joint_ego_p = permute_tensor(next_joint_ego)
    next_joint_robot_p = permute_tensor(next_joint_robot)
    next_joint_mask_p = permute_tensor(next_joint_mask.unsqueeze(-1)).squeeze(-1)

    # agent_index remap:
    # old slot -> new slot 찾기
    # perms[b, new_slot] = old_slot
    # 따라서 old agent_index가 어디로 갔는지 inverse permutation 필요
    inv_perms = torch.empty_like(perms)
    arange_n = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    inv_perms.scatter_(1, perms, arange_n)

    agent_index_p = inv_perms.gather(1, agent_index.unsqueeze(1)).squeeze(1)

    return (
        joint_ego_p,
        joint_robot_p,
        joint_action_p,
        joint_mask_p,
        next_joint_ego_p,
        next_joint_robot_p,
        next_joint_mask_p,
        agent_index_p,
    )

@dataclass
class TransitionMsg:
    worker_id: int

    # centralized critic용 joint observation
    joint_ego_state: np.ndarray          # (MAX_ROBOTS, 4, EGO, EGO)
    global_state: np.ndarray             # (4, DOWN, DOWN)
    joint_robot_state: np.ndarray        # (MAX_ROBOTS, ROBOT_STATE_DIM)
    joint_action: np.ndarray             # (MAX_ROBOTS, ACTION_DIM)
    joint_mask: np.ndarray               # (MAX_ROBOTS,) float32 or bool

    next_joint_ego_state: np.ndarray     # (MAX_ROBOTS, 4, EGO, EGO)
    next_global_state: np.ndarray        # (4, DOWN, DOWN)
    next_joint_robot_state: np.ndarray   # (MAX_ROBOTS, ROBOT_STATE_DIM)
    next_joint_mask: np.ndarray          # (MAX_ROBOTS,)
    reward: float
    done: bool

    # 어떤 robot의 actor loss용 샘플인지
    agent_index: int
    delta_t: float

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

    reward_collision: float
    reward_A: float
    reward_B: float
    reward_fixed: float


STATE_SHAPE = (4, 50, 50)
INPUT_MAP_SIZE = 50
ROBOT_STATE_EMBEDDING = True
ROBOT_STATE_DIM = 6


def make_dreamer_config(device=None) -> DreamerConfig:
    return DreamerConfig(
        max_robots=MAX_ROBOTS,
        ego_shape=(4, EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(4, DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM if ROBOT_STATE_EMBEDDING else 0,
        action_dim=ACTION_DIM,
        device=str(device or DEVICE),
        replay_capacity=int(DREAMER_REPLAY_CAPACITY),
        batch_size=int(DREAMER_BATCH_SIZE),
        sequence_length=int(DREAMER_SEQUENCE_LENGTH),
        replay_context=int(DREAMER_REPLAY_CONTEXT),
        spd_min=float(SPD_MIN),
        spd_max=float(SPD_MAX),
        discount=float(GAMMA_START),
        horizon=int(DREAMER_HORIZON),
        lambda_=float(DREAMER_LAMBDA),
        deter_size=int(DREAMER_DETER_SIZE),
        stoch_size=int(DREAMER_STOCH_SIZE),
        discrete_size=int(DREAMER_DISCRETE_SIZE),
        hidden_size=int(DREAMER_HIDDEN_SIZE),
        embed_size=int(DREAMER_EMBED_SIZE),
        action_embed_size=int(DREAMER_ACTION_EMBED_SIZE),
        conv_depth=int(DREAMER_CONV_DEPTH),
        vector_encoder_layers=int(DREAMER_VECTOR_ENCODER_LAYERS),
        rssm_blocks=int(DREAMER_RSSM_BLOCKS),
        imag_last=int(DREAMER_IMAG_LAST),
        twohot_bins=int(DREAMER_TWOHOT_BINS),
        policy_layers=int(DREAMER_POLICY_LAYERS),
        value_layers=int(DREAMER_VALUE_LAYERS),
        reward_layers=int(DREAMER_REWARD_LAYERS),
        continue_layers=int(DREAMER_CONTINUE_LAYERS),
        model_lr=float(DREAMER_MODEL_LR),
        actor_lr=float(DREAMER_ACTOR_LR),
        value_lr=float(DREAMER_VALUE_LR),
        lr_warmup_steps=int(DREAMER_LR_WARMUP_STEPS),
        target_value_tau=float(DREAMER_TARGET_VALUE_TAU),
        return_norm_decay=float(DREAMER_RETURN_NORM_DECAY),
        slow_value_target=bool(DREAMER_SLOW_VALUE_TARGET),
        contdisc=bool(DREAMER_CONTDISC),
        train_ratio=float(UPDATES_PER_TRANSITION),
        use_amp=bool(DREAMER_USE_AMP),
        decoder_bspace=int(DREAMER_DECODER_BSPACE),
        vector_decoder_layers=int(DREAMER_VECTOR_DECODER_LAYERS),
        decoder_chunk_size=int(DREAMER_DECODER_CHUNK_SIZE),
    )

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
                           angle: None,
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


# def ego_crop_from_full_map(full_map: np.ndarray,
#                            robot_xy_px: tuple[int, int],
#                            ego_size: int,
#                            robot_angle: float = 0.0,
#                            pad_value: int = 50) -> np.ndarray:
#     """
#     full_map: (H, W) uint8
#     robot_xy_px: (ix, iy) in pixel coords
#     ego_size: 최종 결과 이미지의 크기
#     robot_angle: 로봇의 현재 각도 (radian)
#     return: (ego_size, ego_size) uint8, 로봇 정면이 위를 향함
#     """
#     H, W = full_map.shape
#     cx, cy = robot_xy_px
    
#     # 1. 회전 시 모서리 여유를 위해 더 큰 영역을 먼저 추출 (대각선 길이 고려)
#     # sqrt(2) * ego_size 만큼 여유를 둡니다.
#     margin_size = int(ego_size * 1.5)
#     half_m = margin_size // 2
    
#     x0, x1 = cx - half_m, cx - half_m + margin_size
#     y0, y1 = cy - half_m, cy - half_m + margin_size

#     # 맵 범위를 벗어나는 부분 처리
#     sx0, sx1 = max(0, x0), min(W, x1)
#     sy0, sy1 = max(0, y0), min(H, y1)

#     # 여유 영역만큼의 배경 생성
#     temp_crop = np.full((margin_size, margin_size), pad_value, dtype=full_map.dtype)
#     dx0, dy0 = sx0 - x0, sy0 - y0
#     temp_crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = full_map[sy0:sy1, sx0:sx1]

#     # 2. 회전 행렬 생성
#     # robot_angle이 0일 때 '위'를 보게 하려면, OpenCV 기준으로는 추가 보정이 필요할 수 있음
#     # 일반적으로 위쪽 방향은 -90도(또는 pi/2)이므로 이를 맞춰줍니다.
#     # 로봇의 정면 각도가 0(우측)이라면, -90도를 더해 위를 보게 만듭니다.
#     angle_deg = np.degrees(robot_angle)
    
#     # 정면이 위(Up)로 오게 하기 위해: 
#     # 로봇이 보는 방향(angle_deg)을 이미지의 90도(위) 방향으로 회전시킴
#     rotate_angle = -(angle_deg - 90) 

#     center = (margin_size // 2, margin_size // 2)
#     M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)

#     # 3. 이미지 회전 수행
#     rotated_crop = cv2.warpAffine(temp_crop, M, (margin_size, margin_size), 
#                                   flags=cv2.INTER_LINEAR, 
#                                   borderMode=cv2.BORDER_CONSTANT, 
#                                   borderValue=pad_value)

#     # 4. 중심에서 최종 ego_size만큼 크롭
#     start = (margin_size - ego_size) // 2
#     final_crop = rotated_crop[start:start + ego_size, start:start + ego_size]

#     return final_crop

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

ZSG_DIR = os.path.join(log_dir, "zero_shot")
os.makedirs(ZSG_DIR, exist_ok=True)

def zsg_metric_path(map_num: int, robot_num: int) -> str:
    fname = f"zsg_evacuation_100_map_{map_num}_robot_{robot_num}.txt"
    return os.path.join(ZSG_DIR, fname)

def zsg_all_maps_metric_path(robot_num: int) -> str:
    fname = f"zsg_evacuation_100_all_maps_robot_{robot_num}.txt"
    return os.path.join(ZSG_DIR, fname)

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

def pad_robots(robots, max_robots=MAX_ROBOTS):
    real = list(robots[:max_robots])
    while len(real) < max_robots:
        real.append(None)
    return real


def robot_world_to_px(env_model, robot):
    rx, ry = robot.xy
    W = int(env_model.width)
    H = int(env_model.height)
    ix = int(np.clip(rx / max(1, env_model.width) * W, 0, W - 1))
    iy = int(np.clip(ry / max(1, env_model.height) * H, 0, H - 1))
    
    return ix, iy


def get_robot_state(env_model, rb):
    if rb is None:
        return np.zeros((ROBOT_STATE_DIM,), dtype=np.float32)

    return np.array(
        env_model.return_current_robot_state(rb.robot_index),
        dtype=np.float32
    )


def build_joint_frames(env_model, robots_padded, return_full=False):
    """
    env_model과 robots_padded로부터 현재 관측 frame을 구성합니다.

    return_full=False:
        ego_frames, glob_f, joint_robot_state, joint_mask

    return_full=True:
        full_u8, ego_frames, glob_f, joint_robot_state, joint_mask
    """
    full_u8 = env_model.return_current_image(env_model.height, env_model.width)
    glob_u8 = downsample_full_map(full_u8, DOWNSAMPLE_MAP_SIZE)
    glob_f = glob_u8.astype(np.float32) / 255.0

    ego_list = []
    robot_state_list = []
    mask_list = []

    for rb in robots_padded:
        if rb is None:
            ego_f = np.zeros((EGO_MAP_SIZE, EGO_MAP_SIZE), dtype=np.float32)
            rs = np.zeros((ROBOT_STATE_DIM,), dtype=np.float32)
            m = 0.0
        else:
            ix, iy = robot_world_to_px(env_model, rb)

            ego_u8 = ego_crop_from_full_map(
                full_u8,
                (ix, iy),
                EGO_MAP_SIZE,
                rb.angle,
                pad_value=50
            )

            ego_f = ego_u8.astype(np.float32) / 255.0
            rs = get_robot_state(env_model, rb)
            m = 1.0

        ego_list.append(ego_f)
        robot_state_list.append(rs)
        mask_list.append(m)

    ego_frames = np.stack(ego_list, axis=0)
    joint_robot_state = np.stack(robot_state_list, axis=0)
    joint_mask = np.array(mask_list, dtype=np.float32)

    if return_full:
        return full_u8, ego_frames, glob_f, joint_robot_state, joint_mask

    return ego_frames, glob_f, joint_robot_state, joint_mask


class RobotTimeline:
    def __init__(self):
        self.s_ego = None
        self.s_glob = None
        self.s_robot = None
        self.joint_a = None
        self.mask = None
        self.accum_reward = 0.0
        self.delta_t = 0
        self.active = False


# --------------------------------------------------
# 헬퍼 클래스: FRAME_STEP 간격 FrameStack
# --------------------------------------------------
class FrameStackWithStep:
    def __init__(self, k, step):
        self.k = k
        self.step = step
        self.history = []
        self.max_len = (k - 1) * step + 1

    def update(self, frame):
        self.history.append(frame)
        if len(self.history) > self.max_len:
            self.history.pop(0)
        
        # 현재부터 역순으로 step 간격만큼 프레임 추출
        stacked = []
        curr_idx = len(self.history) - 1
        for i in range(self.k):
            idx = max(0, curr_idx - i * self.step)
            stacked.append(self.history[idx])
        
        return np.stack(stacked, axis=0) # (k, H, W)
    def get_stack(self):
        """히스토리에 변화를 주지 않고 현재 시점의 스택된 프레임만 반환"""
        if not self.history:
            # 히스토리가 비어있을 경우에 대한 예외 처리 (필요 시)
            return None 
            
        stacked = []
        curr_idx = len(self.history) - 1
        for i in range(self.k):
            # 히스토리가 충분하지 않을 때는 가장 오래된 프레임(0번)으로 패딩
            idx = max(0, curr_idx - i * self.step)
            stacked.append(self.history[idx])
        
        return np.stack(stacked, axis=0) # (k, H, W)

def worker_process(
    worker_id: int,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    epsilon_shared: mp.Value,
    param_queue: mp.Queue,
    seed: int = 0,
):
    import model
    import traceback

    # --------------------------------------------------
    # 1) 초기 설정 및 시드
    # --------------------------------------------------
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    device = torch.device("cpu")

    worker_agent = DreamerAgent(make_dreamer_config(device="cpu"))
    worker_agent.world_model.eval()
    worker_agent.actor.eval()

    def _all_robots_finished(robots_padded):
        real_count = 0
        finished_count = 0
        for rb in robots_padded:
            if rb is None:
                continue
            real_count += 1
            if getattr(rb, "is_game_finished", False):
                finished_count += 1
        return (real_count > 0) and (finished_count == real_count)


    # 로봇별 독립적 타임라인 및 데이터 보존용 클래스
    class RobotTimeline:
        def __init__(self):
            self.s_ego = None      # (N, 4, E, E)
            self.s_glob = None     # (4, G, G)
            self.s_robot = None    # (N, R)
            self.joint_a = None    # (N, A)
            self.mask = None       # (N,)
            self.accum_reward = 0.0
            self.delta_t = 0
            self.active = False    # 현재 진행중인 액션 유무

    # --------------------------------------------------
    # 2) 에피소드 루프
    # --------------------------------------------------
    episode_idx = 0
    while True:
        # 환경 생성 (생략된 기존 로직 사용)
        if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX:
            number_of_agents = CROWD_NUMBER_MIN
        else:
            number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
    
        env_model = model.FightingModel(
            number_of_agents,
            model_num=-1,
            robot='Q',
            robot_num=THE_NUMBER_OF_ROBOTS
        )

        robots_real = list(getattr(env_model, "robots", []))
        robots_padded = pad_robots(robots_real, max_robots=MAX_ROBOTS)
        worker_agent.reset_policy_state()

        # 상태 관리 객체들
        # FRAME_STEP 주기를 고려한 FrameStack (내부적으로 history를 길게 가져감)
        # 예: 4개 프레임을 쌓는데 간격이 5라면 총 15스텝 전까지의 기록이 필요함
        ego_stacks = [FrameStackWithStep(4, FRAME_STEP) for _ in range(MAX_ROBOTS)]
        glob_stack = FrameStackWithStep(4, FRAME_STEP)
        
        timeline = RobotTimeline()
        current_executing_actions = np.zeros((MAX_ROBOTS, ACTION_DIM), dtype=np.float32)

        total_reward = 0.0

        episode_reward_collision = 0.0
        episode_reward_A = 0.0
        episode_reward_B = 0.0
        episode_reward_fixed = 0.0


        abnormal_reward = 0
        evacuation_time_80 = MAX_STEPS
        evacuation_time_100 = MAX_STEPS
        agent_total_lifetime = 0.0      

        try:
            # 최신 파라미터 로드
            try:
                while True:
                    new_sd = param_queue.get_nowait()
                    worker_agent.load_worker_state(new_sd)
            except queue.Empty:
                pass


            for step in range(MAX_STEPS):
                # A. 현재 시점의 원본 프레임 및 상태 관측
                full_u8, ego_frames, glob_f, joint_robot_state, joint_mask = build_joint_frames(env_model, robots_padded, return_full=True)
                if DEBUG_SAVE:
                    # 1. 공통 이미지(Global, Full)는 한 번만 Flip 처리
                    full_u8_r = np.flip(np.flip(full_u8, axis=-1), axis=-2)
                    glob_f_r = np.flip(np.flip(glob_f, axis=-1), axis=-2)
                    
                    full_u8_to_save = np.flip(full_u8_r, axis=1)
                    glob_f_to_save = np.flip(glob_f_r, axis=1)

                    # 2. 살아있는 모든 로봇을 순회하며 개별 저장
                    for agent_i in range(MAX_ROBOTS):
                        if joint_mask[agent_i] < 0.5:
                            continue  # 패딩된 빈자리(None)나 죽은 로봇은 건너뜀

                        # 해당 로봇의 Ego 이미지만 Flip 처리
                        ego_f_r = np.flip(np.flip(ego_frames[agent_i], axis=-1), axis=-2)
                        ego_f_to_save = np.flip(ego_f_r, axis=1)

                        save_debug_triplet(
                            save_dir=DEBUG_DIR,
                            worker_id=worker_id,
                            episode_idx=episode_idx,
                            step=step,
                            agent_idx=agent_i,  # <--- 어떤 로봇인지 식별자 추가
                            full_u8=full_u8_to_save,
                            ego_f=ego_f_to_save,
                            glob_f=glob_f_to_save,
                            ego_state=ego_stacks[agent_i].get_stack() if step > 0 else None,
                            global_state=glob_stack.get_stack() if step >0 else None
                        )

                # B. FRAME_STEP 간격을 반영한 Stacked State 업데이트
                # 매 스텝 호출하지만 내부적으로는 간격에 맞춰 데이터를 쌓음
                curr_glob_state = glob_stack.update(glob_f)
                curr_joint_ego = np.stack([ego_stacks[i].update(ego_frames[i]) for i in range(MAX_ROBOTS)], axis=0)


                any_finished = any(getattr(rb, "is_game_finished", False) for rb in robots_padded if rb is not None)
                global_done = any_finished or (step == MAX_STEPS - 1)
                eps = 0.0
                with epsilon_shared.get_lock():
                    eps = float(epsilon_shared.value)
                action_boundary = (step % ACTION_STEP) == 0

                # C. ACTION_STEP 경계에서만 이전 joint action을 transition으로 닫고 새 velocity를 뽑는다.
                if action_boundary or global_done:
                    if timeline.active:
                        msg = TransitionMsg(
                            worker_id=worker_id,
                            joint_ego_state=np.copy(timeline.s_ego),
                            global_state=np.copy(timeline.s_glob),
                            joint_robot_state=np.copy(timeline.s_robot),
                            joint_action=np.copy(timeline.joint_a),
                            joint_mask=np.copy(timeline.mask),
                            next_joint_ego_state=np.copy(curr_joint_ego),
                            next_global_state=np.copy(curr_glob_state),
                            next_joint_robot_state=np.copy(joint_robot_state),
                            next_joint_mask=np.copy(joint_mask),
                            reward=float(timeline.accum_reward),
                            done=bool(global_done),
                            agent_index=0,
                            delta_t=float(timeline.delta_t)
                        )
                        transition_queue.put(msg)
                        timeline.active = False
                
                if global_done :
                    break

                if action_boundary:
                    has_active_robot = False
                    pending_actions = []
                    for i, rb in enumerate(robots_padded):
                        if rb is None or getattr(rb, "is_game_finished", False):
                            current_executing_actions[i] = 0.0
                            continue

                        if np.random.rand() < eps:
                            action_i = np.random.uniform(SPD_MIN, SPD_MAX, size=(ACTION_DIM,)).astype(np.float32)
                        else:
                            action_i = None

                        pending_actions.append((i, rb, action_i))
                        has_active_robot = True

                    dreamer_joint_action = None
                    needs_policy_action = any(
                        action_i is None for _, _, action_i in pending_actions
                    )
                    track_rssm = eps < 1.0 - 1e-8
                    if has_active_robot and (needs_policy_action or track_rssm):
                        # Pure random warm-up does not use the policy, so avoid
                        # its CPU encoder/RSSM inference. Once epsilon starts
                        # decaying, advance the posterior at every boundary so
                        # random actions cannot leave the recurrent state stale.
                        dreamer_joint_action = worker_agent.select_joint_action(
                            curr_joint_ego,
                            curr_glob_state,
                            joint_robot_state,
                            joint_mask,
                            deterministic=False,
                        )

                    for i, rb, action_i in pending_actions:
                        if action_i is None:
                            action_i = dreamer_joint_action[i].astype(np.float32)
                        action_i = np.clip(action_i, SPD_MIN, SPD_MAX).astype(np.float32)
                        executed_action = rb.receive_action_velocity(action_i).astype(np.float32)
                        rb.new_order_need = False
                        current_executing_actions[i] = executed_action

                    if has_active_robot:
                        timeline.s_ego = np.copy(curr_joint_ego)
                        timeline.s_glob = np.copy(curr_glob_state)
                        timeline.s_robot = np.copy(joint_robot_state)
                        timeline.joint_a = np.copy(current_executing_actions)
                        timeline.mask = np.copy(joint_mask)
                        timeline.accum_reward = 0.0
                        timeline.delta_t = 0
                        timeline.active = True
                        if dreamer_joint_action is not None:
                            worker_agent.set_policy_prev_action(current_executing_actions)

                # D. 환경 시뮬레이션
                env_model.step()

                # E. 개별 로봇 보상 누적 (매 물리 스텝마다 발생한 페널티 합산)
                step_reward = 0.0
                for i, rb in enumerate(robots_padded):
                    if rb is not None and timeline.active:
                        # env_model에서 해당 로봇의 이번 스텝 개별 보상을 계산하여 반환
                        # (Collision penalty, Distance reward, Step penalty 등)
                        r_step = 0.0

                        # 1) Collision reward / penalty
                        r_collision = env_model.reward_penalty_collision_robot_index(rb.robot_index) * REWARD_K
                        r_step += r_collision
                        episode_reward_collision += r_collision

                        # 2) Reward A
                        if REWARD_A:
                            r_A = env_model.reward_based_alived() * REWARD_A
                        else:
                            r_A = 0.0
                        r_step += r_A
                        episode_reward_A += r_A

                        # 3) Reward B
                        if REWARD_B:
                            r_B = env_model.reward_based_all_agents_danger() * REWARD_B
                        else:
                            r_B = 0.0
                        r_step += r_B
                        episode_reward_B += r_B

                        # 4) Fixed reward / penalty
                        r_fixed = REWARD_FIXED
                        r_step += r_fixed
                        episode_reward_fixed += r_fixed

                        step_reward += r_step
                        total_reward += r_step

                if timeline.active:
                    timeline.accum_reward += step_reward
                    timeline.delta_t += 1
                # ------------------------------------------
                # 80%, 100% crowd evacuation stats
                # ------------------------------------------
                alive_agents = env_model.alived_agents()

                if (alive_agents < env_model.total_agents * 0.2) and (evacuation_time_80 == MAX_STEPS):
                    evacuation_time_80 = step

                if (alive_agents < 1) and (evacuation_time_100 == MAX_STEPS):
                    evacuation_time_100 = step

            try:
                agent_total_lifetime = env_model.calculate_all_agents_life_time()
            except: 
                agent_total_lifetime = 0.0

        except Exception as e:
            print(f"[Worker {worker_id}] Error: {e}")
            traceback.print_exc()
            abnormal_reward = 1

        # --------------------------------------------------
        # 6) episode stat send
        # --------------------------------------------------
        stat_msg = EpisodeStatMsg(
            worker_id=worker_id,
            episode_idx=episode_idx,
            total_reward=float(total_reward),
            evac_time_80=int(evacuation_time_80),
            evac_time_100=int(evacuation_time_100),
            total_lifetime=float(agent_total_lifetime),
            map_num=int(env_model.map_num),
            abnormal=int(abnormal_reward),

            reward_collision = float(episode_reward_collision),
            reward_A = float(episode_reward_A),
            reward_B = float(episode_reward_B),
            reward_fixed = float(episode_reward_fixed)
        )
        try:
            stats_queue.put(stat_msg)
        except Exception as e:
            print(f"[Worker {worker_id}] stats_queue.put error: {e}")

        episode_idx += 1

   

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


def evaluate_zero_shot_once(
    agent,
    map_num: int,
    robot_num: int,
    deterministic: bool = True,
    seed: int = 0,
):
    """
    하나의 map_num, robot_num 조합에 대해 1회 zero-shot 평가를 수행하고
    evacuation_time_100을 반환합니다.
    """
    import model

    np.random.seed(seed)
    random.seed(seed)

    device = agent.device

    # 학습용 worker와 다르게, zero-shot은 지정 map과 지정 robot 수로 생성
    if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX:
        number_of_agents = CROWD_NUMBER_MIN
    else:
        number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)

    env_model = model.FightingModel(
        number_of_agents,
        model_num=map_num,
        robot='Q',
        robot_num=robot_num
    )

    robots_real = list(getattr(env_model, "robots", []))
    robots_padded = pad_robots(robots_real, max_robots=MAX_ROBOTS)
    agent.reset_policy_state()

    ego_stacks = [FrameStackWithStep(4, FRAME_STEP) for _ in range(MAX_ROBOTS)]
    glob_stack = FrameStackWithStep(4, FRAME_STEP)

    evacuation_time_100 = MAX_STEPS

    # 평가 중에는 policy eval 모드
    agent.policy.eval()

    for step in range(MAX_STEPS):
        ego_frames, glob_f, joint_robot_state, joint_mask = build_joint_frames(
            env_model,
            robots_padded
        )

        curr_glob_state = glob_stack.update(glob_f)
        curr_joint_ego = np.stack(
            [ego_stacks[i].update(ego_frames[i]) for i in range(MAX_ROBOTS)],
            axis=0
        )

        any_finished = any(
            getattr(rb, "is_game_finished", False)
            for rb in robots_padded
            if rb is not None
        )

        alive_agents = env_model.alived_agents()

        if alive_agents < 1:
            evacuation_time_100 = step
            break

        if any_finished or step == MAX_STEPS - 1:
            break

        if (step % ACTION_STEP) == 0:
            joint_action = agent.select_joint_action(
                curr_joint_ego,
                curr_glob_state,
                joint_robot_state,
                joint_mask,
                deterministic=deterministic,
            )

            for i, rb in enumerate(robots_padded):
                if rb is None or getattr(rb, "is_game_finished", False):
                    continue

                action_i = np.clip(joint_action[i].astype(np.float32), SPD_MIN, SPD_MAX)
                rb.receive_action_velocity(action_i)
                rb.new_order_need = False
            agent.set_policy_prev_action(joint_action.astype(np.float32))

        env_model.step()

    return int(evacuation_time_100)

def run_zero_shot_evaluation(
    agent,
    episode: int,
    writer: SummaryWriter,
):
    """
    ZSG_MAP, ZSG_ROBOT_NUM, ZSG_ITERATION 기준으로 zero-shot 평가 수행.

    저장 내용:
    1) 각 map_num / robot_num 조합별 evacuation_100 평균
    2) 각 robot_num별 전체 zero-shot map 평균
    """
    print(f"[ZeroShot] Start evaluation at episode {episode}")

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    results = {}

    try:
        for robot_num in ZSG_ROBOT_NUM:
            robot_map_avg_list = []

            for map_num in ZSG_MAP:
                evac100_list = []

                for it in range(ZSG_ITERATION):
                    seed = episode * 100000 + map_num * 100 + robot_num * 10 + it

                    evac100 = evaluate_zero_shot_once(
                        agent=agent,
                        map_num=map_num,
                        robot_num=robot_num,
                        deterministic=True,
                        seed=seed,
                    )

                    evac100_list.append(evac100)

                avg_evac100 = float(np.mean(evac100_list))
                robot_map_avg_list.append(avg_evac100)

                results[(map_num, robot_num)] = avg_evac100

                # --------------------------------------------------
                # 1) map별 txt 저장
                # --------------------------------------------------
                path = zsg_metric_path(map_num, robot_num)
                ensure_file(path)

                with open(path, "a") as f:
                    f.write(f"{episode}\t{avg_evac100:.6f}\n")

                # --------------------------------------------------
                # 2) map별 TensorBoard 저장
                # --------------------------------------------------
                tb_tag = f"ZeroShot/Evacuation100/map_{map_num}/robot_{robot_num}"
                writer.add_scalar(tb_tag, avg_evac100, episode)

                print(
                    f"[ZeroShot] episode={episode}, "
                    f"map={map_num}, robot={robot_num}, "
                    f"avg_evac100={avg_evac100:.2f}, "
                    f"raw={evac100_list}"
                )

            # --------------------------------------------------
            # 3) robot_num별 전체 zero-shot map 평균
            # --------------------------------------------------
            all_maps_avg = float(np.mean(robot_map_avg_list))
            results[("all_maps", robot_num)] = all_maps_avg

            # txt 저장
            all_path = zsg_all_maps_metric_path(robot_num)
            ensure_file(all_path)

            with open(all_path, "a") as f:
                f.write(f"{episode}\t{all_maps_avg:.6f}\n")

            # TensorBoard 저장
            writer.add_scalar(
                f"ZeroShot/Evacuation100/all_maps/robot_{robot_num}",
                all_maps_avg,
                episode
            )

            print(
                f"[ZeroShot] episode={episode}, "
                f"ALL_MAPS, robot={robot_num}, "
                f"avg_evac100={all_maps_avg:.2f}, "
                f"map_avgs={robot_map_avg_list}"
            )

        writer.flush()

    finally:
        agent.epsilon = old_epsilon

    return results

##########################################################################
# Example usage in your training loop
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

    def start_workers(ctx, n_workers, transition_queue, stats_queue, epsilon_shared, base_seed):
        workers = [None] * n_workers
        param_queues = [None] * n_workers
        for wid in range(n_workers):
            p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed)
            workers[wid] = p
            param_queues[wid] = pq
        return workers, param_queues

    def start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed):
        pq = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=worker_process,
            args=(wid, transition_queue, stats_queue, epsilon_shared, pq, base_seed),
            daemon=True
        )
        p.start()
        print(f"[Main] Worker {wid} started, pid={p.pid}")
        return p, pq


    def restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed):
        old_p = workers[wid]
        try:
            if old_p is not None and old_p.is_alive():
                old_p.terminate()
                old_p.join(timeout=2)
        except Exception as e:
            print(f"[Main] terminate/join error for worker {wid}: {e}")

        # 새 큐/프로세스 생성
        p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed)

        workers[wid] = p
        param_queues[wid] = pq
        return


    def supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed):
        # 주기적으로 호출해서 죽은 worker만 재시작
        for wid, p in enumerate(workers):
            if p is None:
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed)
                continue

            if not p.is_alive():
                ec = p.exitcode
                print(f"[Main] Worker {wid} died. exitcode={ec} -> restarting")
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed)

    mp.set_start_method("spawn", force=True)  # or "fork", 리눅스면 fork도 가능

    # ----- 1) TensorBoard/로그 파일 설정 (기존 코드 유지 가능) -----
    total_reward_file = os.path.join(log_dir, "total_reward.txt")

    reward_collision_file = os.path.join(log_dir, "reward_collision.txt")
    reward_A_file = os.path.join(log_dir, "reward_A.txt")
    reward_B_file = os.path.join(log_dir, "reward_B.txt")
    reward_fixed_file = os.path.join(log_dir, "reward_fixed.txt")

    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")
    total_lifetime_file = os.path.join(log_dir, "total_lifetime.txt")

    #reward_vs_ls_file = os.path.join(log_dir, "reward_vs_learning_step.txt")
    #evac100_vs_ls_file = os.path.join(log_dir, "evac100_vs_learning_step.txt")

    #tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    step_writer = SummaryWriter(log_dir=tb_log_dir)


    # 파일 존재 보장
    for path in [
        total_reward_file,
        reward_collision_file,
        reward_A_file,
        reward_B_file,
        reward_fixed_file,
        evacuation_time_80_file,
        evacuation_time_100_file,
        total_lifetime_file
    ]:
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


    monitor_thread_reward_collision = threading.Thread(
    target=monitor_metric,
    args=(reward_collision_file, "Reward/Collision", tb_log_dir),
    daemon=True
    )
    monitor_thread_reward_collision.start()

    monitor_thread_reward_A = threading.Thread(
        target=monitor_metric,
        args=(reward_A_file, "Reward/A", tb_log_dir),
        daemon=True
    )
    monitor_thread_reward_A.start()

    monitor_thread_reward_B = threading.Thread(
        target=monitor_metric,
        args=(reward_B_file, "Reward/B", tb_log_dir),
        daemon=True
    )
    monitor_thread_reward_B.start()

    monitor_thread_reward_fixed = threading.Thread(
        target=monitor_metric,
        args=(reward_fixed_file, "Reward/Fixed", tb_log_dir),
        daemon=True
    )
    monitor_thread_reward_fixed.start()

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
                if lines:
                    start_epsilon = float(lines[0].strip())
                    print(f"Loaded start_epsilon: {start_epsilon}")
                else:
                    print("Empty start_epsilon.txt. Resetting values.")
                    start_epsilon = START_EPSILON
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to defaults.")
                start_epsilon = START_EPSILON
    else:
        start_epsilon = START_EPSILON
        print("No start_epsilon.txt found. Initializing values to defaults.")
    epsilon_shared = mp.Value('d', start_epsilon)
    agent = DreamerAgent(make_dreamer_config(device=DEVICE))
    agent.epsilon = start_epsilon
    print(
        f"DreamerV3 agent initialized, "
        f"lr(model/actor/value)="
        f"{agent.cfg.model_lr}/{agent.cfg.actor_lr}/{agent.cfg.value_lr}, "
        f"batch_size={agent.cfg.batch_size}, "
        f"seq={agent.cfg.sequence_length}, context={agent.cfg.replay_context}, "
        f"horizon={agent.cfg.horizon}, replay_size={agent.cfg.replay_capacity}"
    )
    replay_buffer_path = os.path.join(log_dir, "dreamer_replay.npz")

    global_episode = 0
    if model_load == 1:
        pass
    elif model_load == 2:
        print("load specified model")
        model_name = "dreamer_checkpoint_ep_200.pth"
        model_path = os.path.join(log_dir, model_name)

        if(os.path.exists(model_path)):
            start_episode = int(model_name.split("_")[-1].split(".")[0])
            agent.load_model(model_path)
            if os.path.exists(replay_buffer_path):
                agent.load_replay_buffer(replay_buffer_path)
    elif model_load == 3:
        print("Mode 3: Loading the latest model from log_dir.")
        model_files = [f for f in os.listdir(log_dir) if f.startswith("dreamer_checkpoint") and f.endswith(".pth")]
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

    ctx = mp.get_context("spawn")
    N_WORKERS = N_ENVS # 원하는 만큼

    N_WORKERS_WARMUP = 10
    N_WORKERS_TRAIN = int(N_ENVS)

    transition_queue = ctx.Queue(maxsize=2*N_WORKERS_WARMUP*THE_NUMBER_OF_ROBOTS)
    stats_queue = ctx.Queue(maxsize=2*N_WORKERS_WARMUP*THE_NUMBER_OF_ROBOTS)
    param_queue = mp.Queue(maxsize=1)

    workers = [None] * N_WORKERS
    param_queues = [None] * N_WORKERS
    base_seed = 1234
    if global_episode >= START_UPDATE_EPISODE:
        current_n_workers = N_WORKERS_TRAIN
    else:
        current_n_workers = N_WORKERS_WARMUP
   
    workers, param_queues = start_workers(ctx, current_n_workers, transition_queue, stats_queue, epsilon_shared, base_seed)
    print(f"[Main] Started {current_n_workers} workers for warm-up.")

    initial_sd_cpu = agent.get_worker_state()
    for pq in param_queues:
        try:
            pq.put_nowait(initial_sd_cpu)
        except Exception as e:
            print(f"[Main] initial policy broadcast failed: {e}")



    max_episodes = 9999999

    # update 비율 설정
    pending_updates = 0.0
    train_transition_count = 0
    update_attempt_count = 0
    update_success_count = 0
    update_skip_count = 0
    dreamer_episode_buffers = {}
    dreamer_pending_transitions = {}

    sim_timer.reset()
    learn_timer.reset()
    last_supervise_t = time.time()
    write_heartbeat(global_episode)
    #print(global_episode)
    # ----- 5) 메인 학습 루프 (완전 비동기) -----
    while global_episode < max_episodes:
        # 5-1) transition 소비 (block)
        if time.time() - last_supervise_t > 10.0 :
            supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed)
            last_supervise_t = time.time()

        try:
            msg: TransitionMsg = transition_queue.get(timeout=1.0)
        except queue.Empty:
            # transition이 잠시 없는 경우, stats만 처리하고 계속
            pass
        else:
            seq = dreamer_episode_buffers.setdefault(msg.worker_id, [])
            pending_msg = dreamer_pending_transitions.pop(msg.worker_id, None)

            if not seq:
                first_step = agent.replay.step_from_transition_msg(
                    msg,
                    reward=0.0,
                    is_first=True,
                    is_terminal=False,
                )
                seq.append(first_step)

            if pending_msg is not None:
                aligned_step = agent.replay.step_from_transition_msg(
                    msg,
                    reward=float(pending_msg.reward),
                    delta_t=float(pending_msg.delta_t),
                    is_first=False,
                    is_terminal=False,
                )
                seq.append(aligned_step)

            if msg.done:
                terminal_step = agent.replay.step_from_transition_msg(
                    msg,
                    use_next_obs=True,
                    action=np.zeros_like(msg.joint_action, dtype=np.float32),
                    reward=float(msg.reward),
                    is_first=False,
                    is_terminal=True,
                )
                seq.append(terminal_step)
                agent.replay.add_episode(seq)
                dreamer_episode_buffers[msg.worker_id] = []
            else:
                dreamer_pending_transitions[msg.worker_id] = msg

            # 업데이트 스케줄: transition 수에 비례해서 update 횟수 누적
            train_transition_count += 1
            if global_episode >= START_UPDATE_EPISODE:
                pending_updates += UPDATES_PER_TRANSITION
                while pending_updates >= 1.0:
                    update_attempt_count += 1
                    learn_timer.start()
                    metrics = agent.update()
                    learn_timer.stop()
                    if metrics is not None:
                        update_success_count += 1
                        for k, v in metrics.items():
                            step_writer.add_scalar(f"Dreamer/{k}", v, global_episode)
                    else:
                        update_skip_count += 1
                    pending_updates -= 1.0
       
        needs_switch_workers = False

        # 5-2) stats_queue non-blocking 처리 (에피소드 통계, 스케줄 업데이트, 체크포인트 등)
        while True:
            try:
                s_msg: EpisodeStatMsg = stats_queue.get_nowait()
            except queue.Empty:
                break
           
            seq = dreamer_episode_buffers.get(s_msg.worker_id)
            if seq:
                pending_msg = dreamer_pending_transitions.pop(s_msg.worker_id, None)
                if pending_msg is not None:
                    terminal_step = agent.replay.step_from_transition_msg(
                        pending_msg,
                        use_next_obs=True,
                        action=np.zeros_like(pending_msg.joint_action, dtype=np.float32),
                        reward=float(pending_msg.reward),
                        is_first=False,
                        is_terminal=True,
                    )
                    seq.append(terminal_step)
                seq[-1].is_terminal = True
                agent.replay.add_episode(seq)
                dreamer_episode_buffers[s_msg.worker_id] = []

            write_heartbeat(global_episode)
            #print(global_episode)

                        # ---- warmup -> train 전환 ----
            if (global_episode >= START_UPDATE_EPISODE) and (current_n_workers != N_WORKERS_TRAIN):
                needs_switch_workers = True

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

                with open(reward_collision_file, "a") as f:
                    f.write(f"{s_msg.reward_collision}\n")

                with open(reward_A_file, "a") as f:
                    f.write(f"{s_msg.reward_A}\n")

                with open(reward_B_file, "a") as f:
                    f.write(f"{s_msg.reward_B}\n")

                with open(reward_fixed_file, "a") as f:
                    f.write(f"{s_msg.reward_fixed}\n")

                with open(evacuation_time_80_file, "a") as f:
                    f.write(f"{s_msg.evac_time_80}\n")

                with open(evacuation_time_100_file, "a") as f:
                    f.write(f"{s_msg.evac_time_100}\n")

                with open(total_lifetime_file, "a") as f:
                    f.write(f"{s_msg.total_lifetime}\n")

            # epsilon 스케줄 업데이트 (episode 기준)
            agent.epsilon = max(
                epsilon_scheduler.get_epsilon(agent.epsilon, global_episode),
                EPSILON_MIN
            )

            with epsilon_shared.get_lock():
                epsilon_shared.value = float(agent.epsilon)

            # epsilon 저장
            with open(epsilon_path, "w") as f:
                f.write(str(agent.epsilon) + "\n")

            # 체크포인트 저장
            if (global_episode >= START_UPDATE_EPISODE) and (global_episode % 100 == 0):
                model_filename = os.path.join(log_dir, f"dreamer_checkpoint_ep_{global_episode}.pth")
                agent.save_model(model_filename)
                agent.save_replay_buffer(replay_buffer_path)
            
            # 여기에 zero-shot 평가 호출부 추가
            if (
                global_episode > 0
                and global_episode >= START_UPDATE_EPISODE
                and global_episode % ZSG_CYCLE_EPISODE == 0
            ):
                run_zero_shot_evaluation(
                    agent=agent,
                    episode=global_episode,
                    writer=step_writer,
                )

            if ENABLE_TIMER:
                learning_time = learn_timer.get_time()
                actual_update_ratio = (
                    update_success_count / max(train_transition_count, 1)
                )
                attempted_update_ratio = (
                    update_attempt_count / max(train_transition_count, 1)
                )
                update_success_rate = (
                    update_success_count / max(update_attempt_count, 1)
                )
                step_writer.add_scalar("Timing/learning_time_sec", learning_time, global_episode)
                step_writer.add_scalar("TrainRatio/transitions", train_transition_count, global_episode)
                step_writer.add_scalar("TrainRatio/update_attempts", update_attempt_count, global_episode)
                step_writer.add_scalar("TrainRatio/update_successes", update_success_count, global_episode)
                step_writer.add_scalar("TrainRatio/update_skips", update_skip_count, global_episode)
                step_writer.add_scalar("TrainRatio/pending_updates", pending_updates, global_episode)
                step_writer.add_scalar("TrainRatio/config_updates_per_transition", UPDATES_PER_TRANSITION, global_episode)
                step_writer.add_scalar("TrainRatio/actual_updates_per_transition", actual_update_ratio, global_episode)
                step_writer.add_scalar("TrainRatio/attempted_updates_per_transition", attempted_update_ratio, global_episode)
                step_writer.add_scalar("TrainRatio/update_success_rate", update_success_rate, global_episode)
                learn_timer.reset()
                train_transition_count = 0
                update_attempt_count = 0
                update_success_count = 0
                update_skip_count = 0
       
        #5-3) Worker 전환 로직 (stats 루프 밖에서 안전하게 수행)
        if needs_switch_workers:
            print(f"[Main] Reaching START_UPDATE_EPISODE={START_UPDATE_EPISODE}. Switching workers "
                  f"{current_n_workers} -> {N_WORKERS_TRAIN}")
           
            # Deadlock 방지를 위해 큐를 강제로 비워줌 (Drain)
            # Worker들이 put()에서 블로킹되지 않도록 함
            print("[Main] Draining queues before stopping workers...")
            try:
                while not transition_queue.empty():
                    transition_queue.get_nowait()
                while not stats_queue.empty():
                    stats_queue.get_nowait()
            except:
                pass
            dreamer_episode_buffers.clear()
            dreamer_pending_transitions.clear()

            # Worker 종료
            stop_all_workers(workers)

            # 큐 닫기 (안전장치 추가)
            try:
                transition_queue.close()
                stats_queue.close()
                transition_queue.join_thread() # 필요한 경우
                stats_queue.join_thread()
                time.sleep(1.0)
            except:
                pass

            print("[Main] Re-creating Queues...")
            transition_queue = ctx.Queue(maxsize=2*N_WORKERS_TRAIN*THE_NUMBER_OF_ROBOTS)
            stats_queue = ctx.Queue(maxsize=2*N_WORKERS_TRAIN*THE_NUMBER_OF_ROBOTS)

            # 새 worker 시작
            current_n_workers = N_WORKERS_TRAIN
            workers, param_queues = start_workers(
                ctx, current_n_workers,
                transition_queue, stats_queue,
                epsilon_shared, base_seed
            )
           
            # 최근 파라미터를 새 워커들에게 즉시 전송 (선택 사항)
            sd_cpu = agent.get_worker_state()
            for pq in param_queues:
                try:
                    pq.put_nowait(sd_cpu)
                except Exception as e:
                    print(f"[Main] initial policy broadcast failed: {e}")

            last_supervise_t = time.time()
            # 전환 직후 루프 처음으로 돌아가서 안정적으로 시작
            continue
        if (global_episode >= START_UPDATE_EPISODE) and (global_episode % POLICY_BROADCAST_INTERVAL == 0):
            sd_cpu = agent.get_worker_state()

            for pq in param_queues:
                try:
                    while True:
                        pq.get_nowait()
                except Empty:
                    pass
                except Exception as e:
                    print(f"[Main] param queue drain skipped after error: {e}")

                try:
                    pq.put_nowait(sd_cpu)
                except Full:
                    pass
                except Exception as e:
                    print(f"[Main] policy broadcast failed: {e}")
