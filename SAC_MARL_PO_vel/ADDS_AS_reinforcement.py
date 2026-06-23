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
import cv2

from pathlib import Path
import imageio.v2 as imageio
import torch.distributions as dist

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

    policy = PolicyNetwork(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)
    policy.eval()

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

        # 상태 관리 객체들
        # FRAME_STEP 주기를 고려한 FrameStack (내부적으로 history를 길게 가져감)
        # 예: 4개 프레임을 쌓는데 간격이 5라면 총 15스텝 전까지의 기록이 필요함
        ego_stacks = [FrameStackWithStep(4, FRAME_STEP) for _ in range(MAX_ROBOTS)]
        glob_stack = FrameStackWithStep(4, FRAME_STEP)
        
        timelines = [RobotTimeline() for _ in range(MAX_ROBOTS)]
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
                    policy.load_state_dict(new_sd)
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
                # C. 로봇별 이벤트 체크 (Action 결정 및 Transition 생성)
                for i, rb in enumerate(robots_padded):
                    if rb is None: continue
                    
                    # 새 오더 필요성 체크 (도착/충돌/최초 시작)
                    is_finished = getattr(rb, "is_game_finished", False)
                    needs_new = getattr(rb, "new_order_need", False) or step == 0

                    if needs_new or is_finished:
                        # 1. 이전 액션 마감 및 Transition 전송
                        if timelines[i].active:
                            msg = TransitionMsg(
                                worker_id=worker_id,
                                joint_ego_state=np.copy(timelines[i].s_ego),
                                global_state=np.copy(timelines[i].s_glob),
                                joint_robot_state=np.copy(timelines[i].s_robot),
                                joint_action=np.copy(timelines[i].joint_a),
                                joint_mask=np.copy(timelines[i].mask),
                                next_joint_ego_state=np.copy(curr_joint_ego),
                                next_global_state=np.copy(curr_glob_state),
                                next_joint_robot_state=np.copy(joint_robot_state),
                                next_joint_mask=np.copy(joint_mask),
                                reward=float(timelines[i].accum_reward),
                                done=bool(global_done),
                                agent_index=i,
                                delta_t=float(timelines[i].delta_t)
                            )
                            transition_queue.put(msg)
                            timelines[i].active = False

                        if global_done: 
                            continue
                        # 2. 인라인 액션 선택 (기존 방식 유지 + 3D 확장)
                        if np.random.rand() < eps:
                            # Waypoint: [radius, theta_idx, speed]
                            action_i = np.array([
                                np.random.uniform(R_MIN, R_MAX),           # radius
                                np.random.uniform(-1, 1),
                                np.random.uniform(-1, 1),
                                np.random.uniform(SPD_MIN, SPD_MAX)            # speed
                            ], dtype=np.float32)
                            
                        else:
                            ego_t = torch.from_numpy(curr_joint_ego[i]).unsqueeze(0).float().to(device)
                            glob_t = torch.from_numpy(curr_glob_state).unsqueeze(0).float().to(device)
                            robot_t = torch.from_numpy(joint_robot_state[i]).unsqueeze(0).float().to(device)

                            with torch.no_grad():
                                action_t, _ = policy.sample_action(ego_t, glob_t, robot_t)
                            action_i = action_t.cpu().numpy()[0].astype(np.float32)

                        rb.receive_action_from_policy(action_i)
                        rb.new_order_need = False
                        current_executing_actions[i] = action_i

                        # 4. 출발 시점의 스냅샷(Joint) 저장
                        timelines[i].s_ego = np.copy(curr_joint_ego)
                        timelines[i].s_glob = np.copy(curr_glob_state)
                        timelines[i].s_robot = np.copy(joint_robot_state)
                        timelines[i].joint_a = np.copy(current_executing_actions)
                        timelines[i].mask = np.copy(joint_mask)
                        timelines[i].accum_reward = 0.0
                        timelines[i].delta_t = 0
                        timelines[i].active = True
                
                if global_done :
                    break

                # D. 환경 시뮬레이션
                env_model.step()

                # E. 개별 로봇 보상 누적 (매 물리 스텝마다 발생한 페널티 합산)
                for i, rb in enumerate(robots_padded):
                    if rb is not None and timelines[i].active:
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

                        # 기존 누적
                        timelines[i].accum_reward += r_step
                        timelines[i].delta_t += 1
                        total_reward += r_step
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

   

##########################################################################
# Replay Buffer (Centralized SAC / Multi-Robot CTDE)
##########################################################################
class ReplayBuffer:
    """
    Joint replay buffer for centralized critic + decentralized/shared actor.

    Stored transition format
    ------------------------
    joint_ego_state        : (MAX_ROBOTS, 4, EGO, EGO)
    global_state           : (4, DOWN, DOWN)
    joint_robot_state      : (MAX_ROBOTS, ROBOT_STATE_DIM)
    joint_action           : (MAX_ROBOTS, ACTION_DIM)
    joint_mask             : (MAX_ROBOTS,)               # 1 if real robot else 0

    next_joint_ego_state   : (MAX_ROBOTS, 4, EGO, EGO)
    next_global_state      : (4, DOWN, DOWN)
    next_joint_robot_state : (MAX_ROBOTS, ROBOT_STATE_DIM)
    next_joint_mask        : (MAX_ROBOTS,)

    reward                 : scalar
    done                   : scalar
    agent_index            : scalar int
        - which robot's actor update this sample corresponds to

    Notes
    -----
    - Missing robots are zero-padded and masked out.
    - Images are stored as uint8 to reduce memory usage.
    - This buffer is compatible with centralized critic training.
    """
    def __init__(
        self,
        capacity: int,
        max_robots: int,
        ego_state_shape: Tuple[int, int, int],      # (4,EGO,EGO)
        global_state_shape: Tuple[int, int, int],   # (4,DOWN,DOWN)
        action_dim: int = 3,
        robot_dim: int = 3,
        device=None,
        state_dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = int(capacity)
        self.max_robots = int(max_robots)
        self.device = device
        self.state_dtype = state_dtype
        self.robot_dim = int(robot_dim)
        self.action_dim = int(action_dim)
        self.delta_ts = np.zeros((self.capacity,), dtype=np.float32)
        self.ego_state_shape = tuple(ego_state_shape)
        self.global_state_shape = tuple(global_state_shape)

        # --------------------------------------------------
        # image states
        # --------------------------------------------------
        self.joint_ego_states = np.zeros(
            (self.capacity, self.max_robots, *self.ego_state_shape),
            dtype=self.state_dtype
        )
        self.next_joint_ego_states = np.zeros(
            (self.capacity, self.max_robots, *self.ego_state_shape),
            dtype=self.state_dtype
        )

        self.global_states = np.zeros(
            (self.capacity, *self.global_state_shape),
            dtype=self.state_dtype
        )
        self.next_global_states = np.zeros(
            (self.capacity, *self.global_state_shape),
            dtype=self.state_dtype
        )

        # --------------------------------------------------
        # robot states / actions / masks
        # --------------------------------------------------
        self.joint_robot_states = np.zeros(
            (self.capacity, self.max_robots, self.robot_dim),
            dtype=np.float32
        )
        self.next_joint_robot_states = np.zeros(
            (self.capacity, self.max_robots, self.robot_dim),
            dtype=np.float32
        )

        self.joint_actions = np.zeros(
            (self.capacity, self.max_robots, self.action_dim),
            dtype=np.float32
        )

        self.joint_masks = np.zeros(
            (self.capacity, self.max_robots),
            dtype=np.float32
        )
        self.next_joint_masks = np.zeros(
            (self.capacity, self.max_robots),
            dtype=np.float32
        )

        # --------------------------------------------------
        # scalar values
        # --------------------------------------------------
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.agent_indices = np.zeros((self.capacity,), dtype=np.int64)

        self.ptr = 0
        self.size = 0

    # ======================================================
    # utils
    # ======================================================
    def _to_uint8(self, x: np.ndarray) -> np.ndarray:
        """
        float [0,1] -> uint8
        uint8 -> 그대로 반환
        """
        if x.dtype == self.state_dtype:
            return x
        return np.clip(x * 255.0, 0, 255).astype(self.state_dtype)

    def _check_shapes(
        self,
        joint_ego_state: np.ndarray,
        global_state: np.ndarray,
        joint_robot_state: np.ndarray,
        joint_action: np.ndarray,
        joint_mask: np.ndarray,
        next_joint_ego_state: np.ndarray,
        next_global_state: np.ndarray,
        next_joint_robot_state: np.ndarray,
        next_joint_mask: np.ndarray,
    ) -> None:
        exp_joint_ego = (self.max_robots, *self.ego_state_shape)
        exp_global = self.global_state_shape
        exp_joint_robot = (self.max_robots, self.robot_dim)
        exp_joint_action = (self.max_robots, self.action_dim)
        exp_joint_mask = (self.max_robots,)

        if tuple(joint_ego_state.shape) != exp_joint_ego:
            raise ValueError(
                f"joint_ego_state shape mismatch: got {joint_ego_state.shape}, expected {exp_joint_ego}"
            )
        if tuple(next_joint_ego_state.shape) != exp_joint_ego:
            raise ValueError(
                f"next_joint_ego_state shape mismatch: got {next_joint_ego_state.shape}, expected {exp_joint_ego}"
            )
        if tuple(global_state.shape) != exp_global:
            raise ValueError(
                f"global_state shape mismatch: got {global_state.shape}, expected {exp_global}"
            )
        if tuple(next_global_state.shape) != exp_global:
            raise ValueError(
                f"next_global_state shape mismatch: got {next_global_state.shape}, expected {exp_global}"
            )
        if tuple(joint_robot_state.shape) != exp_joint_robot:
            raise ValueError(
                f"joint_robot_state shape mismatch: got {joint_robot_state.shape}, expected {exp_joint_robot}"
            )
        if tuple(next_joint_robot_state.shape) != exp_joint_robot:
            raise ValueError(
                f"next_joint_robot_state shape mismatch: got {next_joint_robot_state.shape}, expected {exp_joint_robot}"
            )
        if tuple(joint_action.shape) != exp_joint_action:
            raise ValueError(
                f"joint_action shape mismatch: got {joint_action.shape}, expected {exp_joint_action}"
            )
        if tuple(joint_mask.shape) != exp_joint_mask:
            raise ValueError(
                f"joint_mask shape mismatch: got {joint_mask.shape}, expected {exp_joint_mask}"
            )
        if tuple(next_joint_mask.shape) != exp_joint_mask:
            raise ValueError(
                f"next_joint_mask shape mismatch: got {next_joint_mask.shape}, expected {exp_joint_mask}"
            )

    # ======================================================
    # main API
    # ======================================================
    def push(
        self,
        joint_ego_state: np.ndarray,
        global_state: np.ndarray,
        joint_robot_state: np.ndarray,
        joint_action: np.ndarray,
        joint_mask: np.ndarray,
        next_joint_ego_state: np.ndarray,
        next_global_state: np.ndarray,
        next_joint_robot_state: np.ndarray,
        next_joint_mask: np.ndarray,
        reward: float,
        done: bool,
        agent_index: int,
        delta_t: float,
    ) -> None:
        """
        Parameters
        ----------
        joint_ego_state : np.ndarray
            shape (MAX_ROBOTS, 4, EGO, EGO), float [0,1] or uint8
        global_state : np.ndarray
            shape (4, DOWN, DOWN), float [0,1] or uint8
        joint_robot_state : np.ndarray
            shape (MAX_ROBOTS, robot_dim)
        joint_action : np.ndarray
            shape (MAX_ROBOTS, action_dim)
        joint_mask : np.ndarray
            shape (MAX_ROBOTS,)
        """
        self._check_shapes(
            joint_ego_state, global_state, joint_robot_state, joint_action, joint_mask,
            next_joint_ego_state, next_global_state, next_joint_robot_state, next_joint_mask
        )

        if not (0 <= int(agent_index) < self.max_robots):
            raise ValueError(f"agent_index out of range: {agent_index}, max_robots={self.max_robots}")

        i = self.ptr

        self.joint_ego_states[i] = self._to_uint8(joint_ego_state)
        self.next_joint_ego_states[i] = self._to_uint8(next_joint_ego_state)

        self.global_states[i] = self._to_uint8(global_state)
        self.next_global_states[i] = self._to_uint8(next_global_state)

        self.joint_robot_states[i] = joint_robot_state.astype(np.float32, copy=False)
        self.next_joint_robot_states[i] = next_joint_robot_state.astype(np.float32, copy=False)

        self.joint_actions[i] = joint_action.astype(np.float32, copy=False)
        self.joint_masks[i] = joint_mask.astype(np.float32, copy=False)
        self.next_joint_masks[i] = next_joint_mask.astype(np.float32, copy=False)

        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.agent_indices[i] = int(agent_index)
        self.delta_ts[i] = max(float(delta_t), 1.0)

        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: size={self.size}, batch_size={batch_size}")

        idx = np.random.choice(self.size, batch_size, replace=False)

        # images -> float32 [0,1]
        batch_joint_ego = torch.from_numpy(
            self.joint_ego_states[idx].astype(np.float32) / 255.0
        ).to(self.device)

        batch_next_joint_ego = torch.from_numpy(
            self.next_joint_ego_states[idx].astype(np.float32) / 255.0
        ).to(self.device)

        batch_global = torch.from_numpy(
            self.global_states[idx].astype(np.float32) / 255.0
        ).to(self.device)

        batch_next_global = torch.from_numpy(
            self.next_global_states[idx].astype(np.float32) / 255.0
        ).to(self.device)

        batch_joint_robot = torch.from_numpy(
            self.joint_robot_states[idx]
        ).to(self.device)

        batch_next_joint_robot = torch.from_numpy(
            self.next_joint_robot_states[idx]
        ).to(self.device)

        batch_joint_action = torch.from_numpy(
            self.joint_actions[idx]
        ).to(self.device)

        batch_joint_mask = torch.from_numpy(
            self.joint_masks[idx]
        ).to(self.device)

        batch_next_joint_mask = torch.from_numpy(
            self.next_joint_masks[idx]
        ).to(self.device)

        batch_reward = torch.from_numpy(
            self.rewards[idx]
        ).to(self.device)

        batch_done = torch.from_numpy(
            self.dones[idx]
        ).to(self.device)

        batch_agent_index = torch.from_numpy(
            self.agent_indices[idx]
        ).to(self.device)

        batch_delta_t = torch.from_numpy(
            self.delta_ts[idx]
        ).to(self.device)

        return (
            batch_joint_ego,         # (B,N,4,E,E)
            batch_global,            # (B,4,G,G)
            batch_joint_robot,       # (B,N,R)
            batch_joint_action,      # (B,N,A)
            batch_joint_mask,        # (B,N)

            batch_next_joint_ego,    # (B,N,4,E,E)
            batch_next_global,       # (B,4,G,G)
            batch_next_joint_robot,  # (B,N,R)
            batch_next_joint_mask,   # (B,N)

            batch_reward,            # (B,)
            batch_done,              # (B,)
            batch_agent_index,       # (B,)
            batch_delta_t,
        )

    def __len__(self) -> int:
        return self.size

    # ======================================================
    # save / load
    # ======================================================
    def save(self, filepath: Union[str, bytes, os.PathLike]) -> None:
        """
        Save current valid data into compressed npz.
        """
        save_dict = {
            # data
            "joint_ego_states": self.joint_ego_states[:self.size],
            "next_joint_ego_states": self.next_joint_ego_states[:self.size],
            "global_states": self.global_states[:self.size],
            "next_global_states": self.next_global_states[:self.size],

            "joint_robot_states": self.joint_robot_states[:self.size],
            "next_joint_robot_states": self.next_joint_robot_states[:self.size],
            "joint_actions": self.joint_actions[:self.size],
            "joint_masks": self.joint_masks[:self.size],
            "next_joint_masks": self.next_joint_masks[:self.size],

            "rewards": self.rewards[:self.size],
            "dones": self.dones[:self.size],
            "agent_indices": self.agent_indices[:self.size],

            # meta
            "size": self.size,
            "ptr": self.ptr,
            "capacity": self.capacity,
            "max_robots": self.max_robots,
            "robot_dim": self.robot_dim,
            "action_dim": self.action_dim,
            "state_dtype": np.dtype(self.state_dtype).name,
            "ego_state_shape": np.array(self.ego_state_shape, dtype=np.int32),
            "global_state_shape": np.array(self.global_state_shape, dtype=np.int32),
            "delta_ts": self.delta_ts[:self.size],
        }

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: Union[str, bytes, os.PathLike]) -> None:
        """
        Restore replay buffer from compressed npz.
        """
        data = np.load(filepath, allow_pickle=False)

        required = [
            "joint_ego_states", "next_joint_ego_states",
            "global_states", "next_global_states",
            "joint_robot_states", "next_joint_robot_states",
            "joint_actions", "joint_masks", "next_joint_masks",
            "rewards", "dones", "agent_indices",
            "size", "ptr", "capacity", "max_robots",
            "robot_dim", "action_dim", "state_dtype",
            "ego_state_shape", "global_state_shape", "delta_ts"
        ]
        for k in required:
            if k not in data.files:
                raise ValueError(
                    f"[ReplayBuffer.load] '{k}' not found in npz. "
                    "This file is not the new centralized multi-robot format."
                )

        cap = int(data["capacity"])
        max_robots = int(data["max_robots"])
        prev_robot_dim = int(data["robot_dim"])
        prev_action_dim = int(data["action_dim"])
        prev_dtype = np.dtype(str(data["state_dtype"]))

        ego_shape = tuple(data["ego_state_shape"].astype(int).tolist())
        global_shape = tuple(data["global_state_shape"].astype(int).tolist())

        need_reinit = (
            cap != self.capacity
            or max_robots != self.max_robots
            or prev_robot_dim != self.robot_dim
            or prev_action_dim != self.action_dim
            or prev_dtype != self.state_dtype
            or ego_shape != self.ego_state_shape
            or global_shape != self.global_state_shape
        )

        if need_reinit:
            device = self.device
            self.__init__(
                capacity=cap,
                max_robots=max_robots,
                ego_state_shape=ego_shape,
                global_state_shape=global_shape,
                action_dim=prev_action_dim,
                robot_dim=prev_robot_dim,
                device=device,
                state_dtype=prev_dtype,
            )

        self.size = int(data["size"])
        self.ptr = int(data["ptr"])

        self.joint_ego_states[:self.size] = data["joint_ego_states"]
        self.next_joint_ego_states[:self.size] = data["next_joint_ego_states"]

        self.global_states[:self.size] = data["global_states"]
        self.next_global_states[:self.size] = data["next_global_states"]

        self.joint_robot_states[:self.size] = data["joint_robot_states"]
        self.next_joint_robot_states[:self.size] = data["next_joint_robot_states"]

        self.joint_actions[:self.size] = data["joint_actions"]
        self.joint_masks[:self.size] = data["joint_masks"]
        self.next_joint_masks[:self.size] = data["next_joint_masks"]

        self.rewards[:self.size] = data["rewards"]
        self.dones[:self.size] = data["dones"]
        self.agent_indices[:self.size] = data["agent_indices"]
        self.delta_ts[:self.size] = data["delta_ts"]
       
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
    def __init__(self, input_shape, in_channels=4, channels=[32, 64, 64], out_dim=256, compress=True):
        super(ImpalaCNN, self).__init__()
        h, w = input_shape
        self.blocks = nn.ModuleList()
        
        cur_channels = in_channels
        for c in channels:
            self.blocks.append(ImpalaBlock(cur_channels, c))
            cur_channels = c
            # MaxPool2d(kernel=3, stride=2, padding=1)에 의한 크기 변화 공식 적용
            h = (h + 2 * 1 - 3) // 2 + 1
            w = (w + 2 * 1 - 3) // 2 + 1

        self.compress = compress
        
        self.out_channels_2d = cur_channels 
        self.out_h = h
        self.out_w = w
        self.flatten_dim = cur_channels * h * w
        
        if self.compress:
            self.fc = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.flatten_dim, out_dim),
                nn.SiLU()
            )
            self.out_dim = out_dim
        else:
            self.fc = nn.Identity()
            self.out_dim = self.flatten_dim

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            
        if self.compress:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

class CNNEncoder(nn.Module):
    def __init__(self, input_shape=(50, 50), in_channels=4, compress=False):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = in_channels
        self.compress = compress # 압축 여부 플래그
        
        # 기존 CNN 구조
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        
        # SpatioContextualAttention에서 사용할 채널 수 저장
        self.out_channels_2d = 128 
        
        # 최종 출력 차원 계산을 위해 dummy 통과
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            o = self._forward_conv(dummy)
            self.out_h, self.out_w = o.shape[2], o.shape[3]
            self.flatten_dim = int(np.prod(o.size()[1:]))
            
            # compress=True일 경우 (Ego용), 1D 임베딩 크기를 줄여서 반환할 수 있도록 설정
            if self.compress:
                self.compression_layer = nn.Sequential(
                    nn.Linear(self.flatten_dim, 256),
                    nn.SiLU()
                )
                self.out_dim = 256
            else:
                self.out_dim = self.flatten_dim
    def _forward_conv(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x, return_2d=False):
        x = self._forward_conv(x)
        
        # SpatioContextualAttention용 2D 맵이 필요한 경우
        if return_2d:
            return x
            
        # 기본적으로는 Flatten된 벡터 반환
        x = x.reshape(x.size(0), -1)
        if self.compress:
            x = self.compression_layer(x)
        return x
    
class SpatioContextualAttention(nn.Module):
    def __init__(self, global_channels, context_dim, embed_dim=64):
        super().__init__()
        # Key: Global 피처 공간 투영
        self.key_conv = nn.Conv2d(global_channels, embed_dim, kernel_size=1)
        # Query: Ego 컨텍스트 투영
        self.query_fc = nn.Linear(context_dim, embed_dim)
        # Mask 생성
        self.mask_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, global_2d, context_1d):
        B, C, H, W = global_2d.shape
        
        k_2d = self.key_conv(global_2d) # (B, embed_dim, H, W)
        q_1d = self.query_fc(context_1d) # (B, embed_dim)
        
        # Query를 2D 공간으로 확장하여 Key와 상호작용
        q_2d_expanded = q_1d.view(B, -1, 1, 1).expand(-1, -1, H, W)
        interact = k_2d * q_2d_expanded 
        
        spatial_mask = self.mask_conv(interact) # (B, 1, H, W)
        attended_global_2d = global_2d * spatial_mask
        
        return attended_global_2d, spatial_mask


class CentralizedAttentionQNetwork(nn.Module):
    def __init__(
        self,
        ego_shape=(25, 25),
        global_shape=(50, 50),
        action_dim=4,
        robot_dim=3,
        max_robots=3,
        use_robot: bool = True,
        embed_dim=256,  
        num_heads=4     
    ):
        super().__init__()
        self.use_robot_state = use_robot
        self.max_robots = max_robots
        
        # --- image encoding
        self.ego_enc = CNNEncoder(input_shape=ego_shape, in_channels=4, compress=True)
        self.ego_to_embed = nn.Linear(self.ego_enc.out_dim, embed_dim)
        
        self.glob_enc = CNNEncoder(input_shape=global_shape, in_channels=4, compress=False)

        self.global_channels_2d = self.glob_enc.out_channels_2d
        self.global_spatial_size = self.glob_enc.out_h * self.glob_enc.out_w
        flattened_global_dim = self.glob_enc.flatten_dim
        
        
        # Robot stat encoding for query of spatial attention
        self.robot_state_only_enc = nn.Sequential(
            nn.Linear(robot_dim, 32),
            nn.SiLU()
        ) if use_robot else None

        # encoded robot state + encoded ego image
        self.context_dim = 128
        self.ego_context_fc = nn.Sequential(
            nn.Linear(embed_dim + (32 if use_robot else 0), self.context_dim),
            nn.SiLU()
        )

        # --- 2) Spatio-Contextual Attention ---
        self.spatial_attention = SpatioContextualAttention(
            global_channels=self.global_channels_2d, 
            context_dim=self.context_dim,
            embed_dim=64
        )
        
        self.global_1d_fc = nn.Sequential(
            nn.Linear(flattened_global_dim, embed_dim),
            nn.SiLU()
        )

        # --- 3) Action-Conditioned FiLM ----
        vision_dim = embed_dim * 2 # ego_embed + glob_embed
        self.film_gen = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, vision_dim * 2)
        )
        self.film_to_embed = nn.Linear(vision_dim, embed_dim)
        nn.init.zeros_(self.film_gen[-1].weight)
        nn.init.zeros_(self.film_gen[-1].bias)

        if self.use_robot_state:
            self.robot_action_enc = nn.Sequential(
                nn.Linear(robot_dim + action_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim) 
            )
        else:
            self.action_enc = nn.Sequential(
                nn.Linear(action_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # --- 4) Attention & Heads ---
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.fc1 = nn.Linear(embed_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, joint_ego_state, global_state, joint_action, joint_robot_state, joint_mask=None):
        B, N, C, H_e, W_e = joint_ego_state.shape
        
        # --- [Step 1] Ego Context
        ego_flat = joint_ego_state.reshape(B * N, C, H_e, W_e)
        ego_feat = self.ego_enc(ego_flat) 
        ego_embed_flat = F.silu(self.ego_to_embed(ego_feat)) # (B*N, embed_dim)

        if self.use_robot_state:
            robot_flat = joint_robot_state.reshape(B * N, -1)
            robot_feat_flat = self.robot_state_only_enc(robot_flat)
            ego_context_flat = self.ego_context_fc(torch.cat([ego_embed_flat, robot_feat_flat], dim=-1)) # (B*N, context_dim)
        else:
            ego_context_flat = self.ego_context_fc(ego_embed_flat)

        # --- [Step 2] Global 2D 추출 및 N 에이전트만큼 확장 ---
        glob_2d = self.glob_enc(global_state, return_2d=True) # (B, C_g, H_g, W_g)
        _, C_g, H_g, W_g = glob_2d.shape
        
        # 1장의 Global 지도를 N명의 에이전트가 각자 다르게 봐야 하므로 확장 후 B*N으로 형태 변환
        glob_2d_expanded = glob_2d.unsqueeze(1).expand(-1, N, -1, -1, -1).reshape(B * N, C_g, H_g, W_g)

        # --- [Step 3] Spatio-Contextual Attention ---
        # 각 에이전트의 개별 상태(ego_context_flat)에 맞춰 동일한 Global 지도에서 다른 위치를 마스킹함
        attended_glob_2d, _ = self.spatial_attention(glob_2d_expanded, ego_context_flat)
        
        # 1D로 압축 후 원래 차원(B, N, ...)으로 복구
        glob_1d_flat = attended_glob_2d.reshape(B * N, -1)
        glob_embed_flat = self.global_1d_fc(glob_1d_flat) # (B*N, embed_dim)

        ego_embed = ego_embed_flat.reshape(B, N, -1)
        glob_embed = glob_embed_flat.reshape(B, N, -1)
        
        vision_feat = torch.cat([ego_embed, glob_embed], dim=-1) # (B, N, embed_dim * 2)

        # --- [Step 4] Action-Conditioned FiLM ---
        if self.use_robot_state:
            robot_action_input = torch.cat([joint_robot_state, joint_action], dim=-1) 
            robot_action_flat = robot_action_input.reshape(B * N, -1)
            agent_embed = self.robot_action_enc(robot_action_flat).reshape(B, N, -1) 
        else:
            action_flat = joint_action.reshape(B * N, -1)
            agent_embed = self.action_enc(action_flat).reshape(B, N, -1) 

        gamma_beta = self.film_gen(agent_embed)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        film_out = (1.0 + gamma) * vision_feat + beta

        vision_embed = F.silu(self.film_to_embed(film_out))
        full_agent_embed = vision_embed + agent_embed

        if joint_mask is not None:
            full_agent_embed = full_agent_embed * joint_mask.unsqueeze(-1).float()
            key_padding_mask = (joint_mask == 0).bool()
        else:
            key_padding_mask = None

        # --- [Step 5] Self Attention & Q-Head ---
        attn_output, _ = self.attention(
            query=full_agent_embed, key=full_agent_embed, value=full_agent_embed,
            key_padding_mask=key_padding_mask
        )
        
        agent_embed_updated = self.layer_norm(full_agent_embed + attn_output)

        if joint_mask is not None:
            agent_embed_updated = agent_embed_updated * joint_mask.unsqueeze(-1).float()

        # 각 에이전트가 집중한 개별 glob_embed를 다시 결합하여 최종 Q 판단
        
        z = F.silu(self.fc1(agent_embed_updated)) 
        z = F.silu(self.fc2(z)) 
        q = self.q_out(z) 
        
        if joint_mask is not None:
            q = q * joint_mask.unsqueeze(-1).float()
            
        return q

##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, ego_shape=(25,25), global_shape=(50,50), robot_dim = 3, use_robot: bool = True):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.use_robot_state = use_robot

        # --- Ego & Global Encoders ---
        self.ego_enc = CNNEncoder(input_shape=ego_shape, compress=True)
        self.glob_enc = CNNEncoder(input_shape=global_shape, compress=False)

        self.global_channels_2d = self.glob_enc.out_channels_2d
        self.global_spatial_size = self.glob_enc.out_h * self.glob_enc.out_w
        flattened_global_dim = self.global_channels_2d * self.global_spatial_size

        #robot state encoding
        self.robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_dim, 32),
                nn.SiLU()
            )
            self.robot_feat_dim = 32
        else:
            self.robot_fc = None


        self.context_dim = 128
        self.ego_context_fc = nn.Sequential(
            nn.Linear(self.ego_enc.out_dim + self.robot_feat_dim, self.context_dim),
            nn.SiLU()
        )

        self.spatial_attention = SpatioContextualAttention(
            global_channels = self.global_channels_2d,
            context_dim = self.context_dim,
            embed_dim = 64
        )

        self.global_1d_fc = nn.Sequential(
            nn.Linear(flattened_global_dim, 256),
            nn.SiLU()
        )

        
        fusion_dim = self.context_dim + 256 # Ego Context + Modulated Global
        self.fc_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU()
        )
        
        self.r_mean = nn.Linear(64, 1)
        self.r_log_std = nn.Linear(64, 1)
        
        # 2. Direction (dx, dy)
        self.dir_mean = nn.Linear(64, 2)
        self.dir_log_std = nn.Linear(64, 2)
        
        # 3. Target Speed
        self.spd_mean = nn.Linear(64, 1)
        self.spd_log_std = nn.Linear(64, 1)
    def backbone(self, ego_state, global_state, robot_state=None):
        # 1. Ego Context 생성
        e = self.ego_enc(ego_state)
        r = self.robot_fc(robot_state) if self.use_robot_state else torch.zeros(e.size(0), 0).to(e.device)
        ego_context = self.ego_context_fc(torch.cat([e, r], dim=1)) # (B, context_dim)

        # 2. Global 2D & Spatial Attention
        g_2d = self.glob_enc(global_state, return_2d = True) # (B, C, H, W)
        attended_g_2d, _ = self.spatial_attention(g_2d, ego_context)

        # 3. Flatten & Project to 1D
        g_1d_flat = attended_g_2d.reshape(attended_g_2d.size(0), -1)
        g_1d = self.global_1d_fc(g_1d_flat) # (B, 256)

        # 5. Final Concat & Feed Forward
        combined = torch.cat([ego_context, g_1d], dim=1)
        feat = self.fc_backbone(combined)
        return feat

    def forward(self, ego_state, global_state, robot_state=None):
        feat = self.backbone(ego_state, global_state, robot_state)
        
        r_m = self.r_mean(feat)
        r_s = torch.clamp(self.r_log_std(feat), LOG_STD_MIN, LOG_STD_MAX)
        
        d_m = self.dir_mean(feat) # [dx_m, dy_m]
        d_s = torch.clamp(self.dir_log_std(feat), LOG_STD_MIN, LOG_STD_MAX)
        
        spd_m = self.spd_mean(feat)
        spd_s = torch.clamp(self.spd_log_std(feat), LOG_STD_MIN, LOG_STD_MAX)
        
        return r_m, r_s, d_m, d_s, spd_m, spd_s
    
    def sample_action(self, ego_state, global_state, robot_state=None):
        r_m, r_s, d_m, d_s, spd_m, spd_s = self.forward(ego_state, global_state, robot_state)
        
        def _sample_with_tanh(m, s, low, high):
            std = s.exp()
            u = m + std * torch.randn_like(m)

            z = torch.tanh(u)  # [-1, 1]

            # [-1, 1] -> [low, high]
            act = low + 0.5 * (high - low) * (z + 1.0)

            # Gaussian log prob
            logp_u = -0.5 * (
                ((u - m) / (std + 1e-8)) ** 2
                + 2 * s
                + np.log(2 * np.pi)
            )

            # da/du = 0.5 * (high - low) * (1 - tanh(u)^2)
            jacobian = torch.log(
                0.5 * (high - low) * (1.0 - z.pow(2)) + 1e-8
            )

            logp = (logp_u - jacobian).sum(dim=-1)
            return act, logp
        # 1) r Sampling (R_MIN ~ R_MAX)
        r_act, r_logp = _sample_with_tanh(r_m, r_s, R_MIN, R_MAX)

        # 2) dx, dy Sampling (-1.0 ~ 1.0)
        # dx, dy는 단순히 방향 벡터이므로 범위를 -1 ~ 1로 설정합니다.
        dir_act, dir_logp = _sample_with_tanh(d_m, d_s, -1.0, 1.0)

        # 3) speed Sampling (SPD_MIN ~ SPD_MAX)
        spd_act, spd_logp = _sample_with_tanh(spd_m, spd_s, SPD_MIN, SPD_MAX)

        # --- Final Combine ---
        # Action: [batch, 4] -> [r, dx, dy, speed]
        if r_act.dim() == 1:
            r_act = r_act.unsqueeze(-1)
        if spd_act.dim() == 1:
            spd_act = spd_act.unsqueeze(-1)
        if dir_act.dim() == 1:
            dir_act = dir_act.unsqueeze(-1) # 이 경우는 거의 없겠지만 안전을 위해

        # 결합 후 결과는 (B, 4)가 되어야 함
        action = torch.cat([r_act, dir_act, spd_act], dim=-1)
        
        # logp들도 (B,) 형태인지 확인 (이미 sum(dim=-1)을 해서 (B,)일 것입니다)
        total_logp = r_logp + dir_logp + spd_logp
        
        return action, total_logp
   
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
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device='cpu')
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.epsilon = start_epsilon
        self.epsilon_long = start_epsilon_long
        self.epsilon_long_min = long_epsilon_min

        # Multimodal Flag
        self.use_robot_state = ROBOT_STATE_EMBEDDING
        self.robot_dim = ROBOT_STATE_DIM if self.use_robot_state else 0
        
        self.target_entropy = -4
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=int(replay_size),
            max_robots=MAX_ROBOTS,
            ego_state_shape=(4, EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_state_shape=(4, DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=4,
            robot_dim=self.robot_dim,
            device=self.device,
        )

        self.q1 = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=4,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2 = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=4,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q1_target = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=4,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2_target = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=4,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        # self.q1_target, self.q2_target -> Q의 Ground Truth 근사치 제공
        # q-network 업데이트 시 사용하는 Target 값을 제공

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network
        self.policy = PolicyNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING
        ).to(self.device)

        # Optimizers
        self.q1_optimizer = optim.AdamW(self.q1.parameters(), lr=lr, weight_decay=WD_Q)
        self.q2_optimizer = optim.AdamW(self.q2.parameters(), lr=lr, weight_decay=WD_Q)
        self.policy_optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=WD_PI)
        


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
        # 1. 탐험 (Exploration) 로직 수정
        if (EXPLORATION_TYPE == 0):
            if np.random.rand() < self.epsilon:
                # 새로운 액션 체계 [r, theta_idx, speed]에 맞게 랜덤 생성
                r = np.random.uniform(R_MIN, R_MAX)
                dx = np.random.uniform(-1, 1)
                dy = np.random.uniform(-1, 1)
                spd = np.random.uniform(SPD_MIN, SPD_MAX)
                return np.array([r, dx, dy, spd]), True

        # 2. 데이터 텐서화
        ego_t = torch.FloatTensor(ego_state_np).unsqueeze(0).to(self.device)
        global_t = torch.FloatTensor(global_state_np).unsqueeze(0).to(self.device)
        robot_t = torch.FloatTensor(robot_state_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                r_m, _, d_m, _, spd_m, _ = self.policy.forward(ego_t, global_t, robot_t)
                
                # Sigmoid 스케일링 (PolicyNetwork.sample_action 내부 로직과 일치시켜야 함)

                r = R_MIN + 0.5 * (R_MAX - R_MIN) * (torch.tanh(r_m) + 1.0)

                spd = SPD_MIN + 0.5 * (SPD_MAX - SPD_MIN) * (torch.tanh(spd_m) + 1.0)

                dir_vec = torch.tanh(d_m)  # 이미 [-1, 1]
                
                action_t = torch.cat([r, dir_vec, spd], dim=-1) 
            else:
                # 비결정적(Stochastic) 선택: sample_action 사용 (reparameterization trick)
                action_t, _ = self.policy.sample_action(ego_t, global_t, robot_t)

        action_np = action_t.cpu().numpy()[0]
        #확인용 출력: [r, theta_idx, speed] 순서임
        print(f"DEBUG: Policy Output Shape = {action_np.shape}")
        print(f"Action: r={action_np[0]:.2f}, dx={action_np[1]:.2f}, dy={action_np[2]:.2f}, spd={action_np[3]:.2f}")
        return action_np, False
    def update_gamma(self, start_gamma, end_gamma, ascent_steps, now_episode):
        self.gamma = gamma_ascent_schedule(start_gamma, end_gamma, ascent_steps, now_episode)
   
    def update_alpha(self, start_alpha, end_alpha, decay_steps, now_episode):
        self.alpha = alpha_decay_schedule(start_alpha, end_alpha, decay_steps, now_episode)



# ------------------------------------------------- #
    # Update (one gradient step) - Waypoint Optimized
    # ------------------------------------------------- #
    def update(self):
        # 1) Replay Buffer로부터 샘플링 (delta_t 포함 필수)
        if len(self.replay_buffer) < self.batch_size * START_BATCH_TIMES:
            return

        sampled = self.replay_buffer.sample(self.batch_size)
        # buffer.sample이 반환하는 값의 순서에 맞춰서 언패킹하세요.
        # delta_t는 해당 waypoint 액션 시작부터 종료(도달/충돌)까지 걸린 시간(또는 스텝 수)입니다.
        (
            joint_ego, global_state, joint_robot, joint_action, joint_mask,
            next_joint_ego, next_global_state, next_joint_robot, next_joint_mask,
            reward, done, agent_index, delta_t 
        ) = sampled

        # 다수 로봇 환경일 경우의 순열 처리 (기존 유지)
        if MAX_ROBOTS > 1:
            (
                joint_ego, joint_robot, joint_action, joint_mask,
                next_joint_ego, next_joint_robot, next_joint_mask,
                agent_index
            ) = random_permute_joint_batch(
                joint_ego, joint_robot, joint_action, joint_mask,
                next_joint_ego, next_joint_robot, next_joint_mask,
                agent_index,
            )

        self.alpha = self.log_alpha.exp().detach()

        # -----------------------
        # 1) Critic target (Waypoint 핵심: 가변 감마)
        # -----------------------
        with torch.no_grad():
            # 다음 상태에서의 행동 샘플링
            next_joint_action, next_joint_logp = self.sample_joint_actions(
                next_joint_ego, next_global_state, next_joint_robot, next_joint_mask
            )

            # 현재 학습 대상 에이전트의 log_prob 추출
            next_logp_i = gather_agent_tensor(next_joint_logp.unsqueeze(-1), agent_index).squeeze(-1)

            # Target Q 네트워크 평가
            q1_next_all = self.q1_target(
                next_joint_ego, next_global_state, next_joint_action, next_joint_robot, next_joint_mask
            )
            q2_next_all = self.q2_target(
                next_joint_ego, next_global_state, next_joint_action, next_joint_robot, next_joint_mask
            )
            
            q1_next_i = gather_agent_tensor(q1_next_all, agent_index).squeeze(-1)
            q2_next_i = gather_agent_tensor(q2_next_all, agent_index).squeeze(-1)
            q_next = torch.min(q1_next_i, q2_next_i)

            # delta_t increase -> discount more
            # adjusted_gamma = torch.pow(self.gamma, delta_t)
            q_target = reward + self.gamma * (1 - done) * (q_next - self.alpha * next_logp_i)

        # -----------------------
        # 2) Critic update
        # -----------------------
        q1_val_all = self.q1(joint_ego, global_state, joint_action, joint_robot, joint_mask)
        q2_val_all = self.q2(joint_ego, global_state, joint_action, joint_robot, joint_mask)
        q1_val = gather_agent_tensor(q1_val_all, agent_index).squeeze(-1)
        q2_val = gather_agent_tensor(q2_val_all, agent_index).squeeze(-1)

        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q2_optimizer.step()

        # -----------------------
        # 3) Actor update
        # -----------------------
        new_joint_action, joint_logp = self.sample_joint_actions(
            joint_ego, global_state, joint_robot, joint_mask
        )
        logp_i = gather_agent_tensor(joint_logp.unsqueeze(-1), agent_index).squeeze(-1)

        q1_new_all = self.q1(joint_ego, global_state, new_joint_action, joint_robot, joint_mask)
        q2_new_all = self.q2(joint_ego, global_state, new_joint_action, joint_robot, joint_mask)

        q1_new_i = gather_agent_tensor(q1_new_all, agent_index).squeeze(-1)
        q2_new_i = gather_agent_tensor(q2_new_all, agent_index).squeeze(-1)
        q_new = torch.min(q1_new_i, q2_new_i)

        # Policy loss: Entropy-regularized Q maximization
        policy_loss = (self.alpha * logp_i - q_new).mean()

        # Alpha (Temperature) update
        alpha_loss = -(self.log_alpha * (logp_i + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.policy_optimizer.step()

        # -----------------------
        # 4) Soft update
        # -----------------------
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

    def sample_joint_actions(self, joint_ego, global_state, joint_robot_state, joint_mask):
        """
        joint_ego: (B, N, 4, E, E)
        global_state: (B, 4, G, G)
        joint_robot_state: (B, N, R)
        joint_mask: (B, N)
        """
        B, N, C, H, W = joint_ego.shape
        actions = []
        log_probs = []

        for i in range(N):
            ego_i = joint_ego[:, i]               # (B,4,E,E)
            robot_i = joint_robot_state[:, i]     # (B,R)

            act_i, logp_i = self.policy.sample_action(ego_i, global_state, robot_i)
            m_i = joint_mask[:, i].float().unsqueeze(-1)

            act_i = act_i * m_i
            logp_i = logp_i * joint_mask[:, i].float()

            actions.append(act_i)
            log_probs.append(logp_i)

        joint_action = torch.stack(actions, dim=1)   # (B,N,A)
        joint_logp = torch.stack(log_probs, dim=1)   # (B,N)
        return joint_action, joint_logp

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

        # 로봇별 action 부여
        for i, rb in enumerate(robots_padded):
            if rb is None:
                continue

            needs_new = getattr(rb, "new_order_need", False) or step == 0
            is_finished = getattr(rb, "is_game_finished", False)

            if is_finished:
                continue

            if needs_new:
                ego_i = curr_joint_ego[i]
                robot_i = joint_robot_state[i]

                action_i, _ = agent.select_action(
                    ego_i,
                    curr_glob_state,
                    robot_i,
                    deterministic=deterministic
                )

                rb.receive_action_from_policy(action_i)
                rb.new_order_need = False

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
    start_alpha = ALPHA_START
    if os.path.exists(epsilon_path):
        with open(epsilon_path, "r") as f:
            try:
                lines = f.readlines()
                if len(lines) >= 2:
                    start_epsilon = float(lines[0].strip())
                    start_epsilon_long = float(lines[1].strip())
                    start_alpha = float(lines[2].strip())
                    print(f"Loaded start_epsilon: {start_epsilon}, start_epsilon_long: {start_epsilon_long}")
                else:
                    print("Not enough lines in start_epsilon.txt. Resetting values.")
                    start_epsilon = START_EPSILON
                    start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
                    start_alpha = ALPHA_START
            except ValueError:
                print("Invalid value in start_epsilon.txt. Resetting to defaults.")
                start_epsilon = START_EPSILON
                start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
                start_alpha = ALPHA_START
    else:
        start_epsilon = START_EPSILON
        start_epsilon_long = START_LONG_EPSILON  # 기본값 설정
        start_alpha = ALPHA_START
        print("No start_epsilon.txt found. Initializing values to defaults.")
    epsilon_shared = mp.Value('d', start_epsilon)
    agent = SACAgent(input_shape=(50,50), alpha=start_alpha, lr=float(LR), start_epsilon=start_epsilon, start_epsilon_long = float(START_LONG_EPSILON), long_epsilon_min=float(LONG_EPSILON_MIN), batch_size=int(BATCH_SIZE), replay_size=int(BUFFER_SIZE), device=DEVICE)
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



    max_episodes = 9999999

    # update 비율 설정
    pending_updates = 0.0

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
            # ReplayBuffer에 push
            for agent_i in range(MAX_ROBOTS):
                if msg.joint_mask[agent_i] > 0.5:
                  
                    agent.replay_buffer.push(
                        msg.joint_ego_state,
                        msg.global_state,
                        msg.joint_robot_state,
                        msg.joint_action,
                        msg.joint_mask,
                        msg.next_joint_ego_state,
                        msg.next_global_state,
                        msg.next_joint_robot_state,
                        msg.next_joint_mask,
                        msg.reward,
                        msg.done,
                        agent_i,
                        msg.delta_t
                    )

            # 업데이트 스케줄: transition 수에 비례해서 update 횟수 누적
            if global_episode >= START_UPDATE_EPISODE:
                pending_updates += UPDATES_PER_TRANSITION
                while pending_updates >= 1.0:
                    learn_timer.start()
                    agent.update()
                    learn_timer.stop()
                    pending_updates -= 1.0
       
        needs_switch_workers = False

        # 5-2) stats_queue non-blocking 처리 (에피소드 통계, 스케줄 업데이트, 체크포인트 등)
        while True:
            try:
                s_msg: EpisodeStatMsg = stats_queue.get_nowait()
            except queue.Empty:
                break
           
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

            # alpha/gamma/epsilon 스케줄 업데이트 (episode 기준)
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
                f.write(str(agent.epsilon_long) + "\n")
                f.write(str(agent.alpha.item()))

            # 체크포인트 저장
            if (global_episode >= START_UPDATE_EPISODE) and (global_episode % 100 == 0):
                model_filename = os.path.join(log_dir, f"sac_checkpoint_ep_{global_episode}.pth")
                agent.save_model(model_filename)
                agent.save_replay_buffer("replay_buffer.npz")
            
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
                print(f"Episode {global_episode} - Total Learning Time: {learn_timer.get_time():.6f} 초")
                learn_timer.reset()
       
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
            sd_cpu = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}
            for pq in param_queues:
                pq.put(sd_cpu)

            last_supervise_t = time.time()
            # 전환 직후 루프 처음으로 돌아가서 안정적으로 시작
            continue
        if (global_episode >= START_UPDATE_EPISODE) and (global_episode % POLICY_BROADCAST_INTERVAL == 0):
            sd_cpu = {k: v.detach().cpu() for k, v in agent.policy.state_dict().items()}

            for pq in param_queues:
                try:
                    while True:
                        pq.get_nowait()
                except Empty:
                    pass

                try:
                    pq.put_nowait(sd_cpu)
                except Full:
                    pass