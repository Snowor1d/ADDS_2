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


def worker_process(
    worker_id: int,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    epsilon_shared: mp.Value,
    param_queue: mp.Queue,
    seed: int = 0,
):
    """
    Multi-robot worker process for CTDE / Centralized SAC.

    - Actor: shared PolicyNetwork, each robot acts from its own (ego, global, robot_state)
    - Critic training data: joint observation/action with zero-padding up to MAX_ROBOTS
    - For each reward emission step, one joint transition is duplicated for each real robot
      with different agent_index.
    """
    import model
    import traceback


    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

    max_steps = MAX_STEPS
    episode_idx = 0

    device = torch.device("cpu")
    policy = PolicyNetwork(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)
    policy.eval()

    def _pad_robots(robots, max_robots=MAX_ROBOTS):
        real = list(robots[:max_robots])
        while len(real) < max_robots:
            real.append(None)
        return real

    def _robot_world_to_px(env_model, robot):
        rx, ry = robot.xy
        ix = int(np.clip(rx / env_model.width * MAP_W, 0, MAP_W - 1))
        iy = int(np.clip(ry / env_model.height * MAP_H, 0, MAP_H - 1))
        return ix, iy

    def _get_robot_state(env_model, rb, idx):
        """
        Robot state getter.
        Priority:
        1) robot object method: rb.return_current_robot_state()
        2) env method with idx: env_model.return_current_robot_state(idx)
        3) env method with robot object: env_model.return_current_robot_state(rb)
        """
        if rb is None:
            return np.zeros((ROBOT_STATE_DIM,), dtype=np.float32)

        return np.array(env_model.return_current_robot_state(rb.robot_index))


    def _build_joint_frames(env_model, robots_padded):
        """
        Returns
        -------
        full_u8          : (MAP_H, MAP_W) uint8
        ego_frames       : (MAX_ROBOTS, EGO_MAP_SIZE, EGO_MAP_SIZE) float32 [0,1]
        glob_frame       : (DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE) float32 [0,1]
        joint_robot_state: (MAX_ROBOTS, ROBOT_STATE_DIM) float32
        joint_mask       : (MAX_ROBOTS,) float32
        """
        full_u8 = env_model.return_current_image(MAP_H, MAP_W)  # uint8
        glob_u8 = downsample_full_map(full_u8, DOWNSAMPLE_MAP_SIZE)
        glob_f = glob_u8.astype(np.float32) / 255.0

        ego_list = []
        robot_state_list = []
        mask_list = []

        for i, rb in enumerate(robots_padded):
            rb = robots_padded[i]
            if rb is None:
                ego_f = np.zeros((EGO_MAP_SIZE, EGO_MAP_SIZE), dtype=np.float32)
                rs = np.zeros((ROBOT_STATE_DIM,), dtype=np.float32)
                m = 0.0
            else:
                ix, iy = _robot_world_to_px(env_model, rb)
                ego_u8 = ego_crop_from_full_map(full_u8, (ix, iy), EGO_MAP_SIZE, pad_value=50)
                ego_f = ego_u8.astype(np.float32) / 255.0
                rs = _get_robot_state(env_model, rb, i)
                m = 1.0

            ego_list.append(ego_f)
            robot_state_list.append(rs)
            mask_list.append(m)

        ego_frames = np.stack(ego_list, axis=0)                    # (N,E,E)
        joint_robot_state = np.stack(robot_state_list, axis=0)     # (N,R)
        joint_mask = np.array(mask_list, dtype=np.float32)         # (N,)

        return full_u8, ego_frames, glob_f, joint_robot_state, joint_mask

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

    while True:
        # --------------------------------------------------
        # 1) environment create
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
                    robot='Q',
                    robot_num=THE_NUMBER_OF_ROBOTS
                )
                break
            except Exception as e:
                print(f"[Worker {worker_id}] env create error: {e}, retrying...")

        robots_real = list(getattr(env_model, "robots", []))
        robots_padded = _pad_robots(robots_real, max_robots=MAX_ROBOTS)

        if sum(rb is not None for rb in robots_padded) == 0:
            print(f"[Worker {worker_id}] WARNING: env_model.robots is empty.")

        # --------------------------------------------------
        # 2) initial joint state
        # --------------------------------------------------
        full_u8, ego_frames, glob_frame, joint_robot_state, joint_mask = _build_joint_frames(env_model, robots_padded)

        ego_stacks = [FrameStack2(4) for _ in range(MAX_ROBOTS)]
        glob_stack = FrameStack2(4)

        joint_ego_state_list = []
        for i in range(MAX_ROBOTS):
            s_i = ego_stacks[i].reset(ego_frames[i])   # (4,E,E)
            joint_ego_state_list.append(s_i)
        joint_ego_state = np.stack(joint_ego_state_list, axis=0)   # (N,4,E,E)

        global_state = glob_stack.reset(glob_frame)                 # (4,G,G)

        # --------------------------------------------------
        # episode stats
        # --------------------------------------------------
        total_reward = 0.0
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps
        agent_total_lifetime = 0.0
        abnormal_reward = 0

        # --------------------------------------------------
        # transition buffers (joint)
        # --------------------------------------------------
        buffered_joint_ego_state = np.copy(joint_ego_state)
        buffered_global_state = np.copy(global_state)
        buffered_joint_robot_state = np.copy(joint_robot_state)
        buffered_joint_action = np.zeros((MAX_ROBOTS, ACTION_DIM), dtype=np.float32)
        buffered_joint_mask = np.copy(joint_mask)

        eps = 0.0
        with epsilon_shared.get_lock():
            eps = float(epsilon_shared.value)

        try:
            # latest actor params
            try:
                while True:
                    new_sd = param_queue.get_nowait()
                    policy.load_state_dict(new_sd)
            except queue.Empty:
                pass

            for step in range(max_steps):
                # ------------------------------------------
                # 1) action selection every ACTION_SCALE
                # ------------------------------------------
                if step % ACTION_SCALE == 0:
                    # refresh robot handles if env may replace them internally
                    robots_real = list(getattr(env_model, "robots", []))
                    robots_padded = _pad_robots(robots_real, max_robots=MAX_ROBOTS)

                    full_u8, ego_frames, glob_frame, joint_robot_state, joint_mask = _build_joint_frames(
                        env_model, robots_padded
                    )

                    if DEBUG_SAVE:
                        # 1. 공통 이미지(Global, Full)는 한 번만 Flip 처리
                        full_u8_r = np.flip(np.flip(full_u8, axis=-1), axis=-2)
                        glob_f_r = np.flip(np.flip(glob_frame, axis=-1), axis=-2)
                        
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
                                ego_state=joint_ego_state[agent_i] if step > 0 else None,
                                global_state=global_state
                            )

                    # ACTION_SCALE boundary에서만 실제 stack append
                    if step > 0:
                        updated_joint_ego = []
                        for i in range(MAX_ROBOTS):
                            updated_joint_ego.append(ego_stacks[i].append(ego_frames[i]))
                        joint_ego_state = np.stack(updated_joint_ego, axis=0)
                        global_state = glob_stack.append(glob_frame)

                    # shared actor, per robot action selection
                    joint_action = np.zeros((MAX_ROBOTS, ACTION_DIM), dtype=np.float32)

                    for i, rb in enumerate(robots_padded):
                        if rb is None or joint_mask[i] < 0.5:
                            continue

                        ego_i = joint_ego_state[i]         # (4,E,E)
                        robot_i = joint_robot_state[i]     # (R,)
                        glob_i = global_state              # (4,G,G)

                        if np.random.rand() < eps or policy is None:
                            action_i = np.array([
                                np.random.uniform(-2, 2),
                                np.random.uniform(-2, 2)
                            ], dtype=np.float32)
                        else:
                            ego_t = torch.from_numpy(ego_i).unsqueeze(0).float().to(device)
                            glob_t = torch.from_numpy(glob_i).unsqueeze(0).float().to(device)
                            robot_t = torch.from_numpy(robot_i).unsqueeze(0).float().to(device)

                            with torch.no_grad():
                                action_t, _ = policy.sample_action(
                                    ego_t, glob_t, robot_t, temperature=1.0
                                )
                            action_i = action_t.cpu().numpy()[0].astype(np.float32)

                        real_action = rb.receive_action([action_i[0], action_i[1]])
                        joint_action[i, 0] = real_action[0]
                        joint_action[i, 1] = real_action[1]

                    buffered_joint_ego_state = np.copy(joint_ego_state)
                    buffered_global_state = np.copy(global_state)
                    buffered_joint_robot_state = np.copy(joint_robot_state)
                    buffered_joint_action = np.copy(joint_action)
                    buffered_joint_mask = np.copy(joint_mask)

                # ------------------------------------------
                # 2) env step
                # ------------------------------------------
                env_model.step()

                # ------------------------------------------
                # 3) next joint state
                # ------------------------------------------
                robots_real = list(getattr(env_model, "robots", []))
                robots_padded = _pad_robots(robots_real, max_robots=MAX_ROBOTS)

                _, next_ego_frames, next_glob_frame, next_joint_robot_state, next_joint_mask = _build_joint_frames(
                    env_model, robots_padded
                )

                next_joint_ego_list = []
                for i in range(MAX_ROBOTS):
                    next_joint_ego_list.append(ego_stacks[i].peek_with(next_ego_frames[i]))
                next_joint_ego_state = np.stack(next_joint_ego_list, axis=0)   # (N,4,E,E)
                next_global_state = glob_stack.peek_with(next_glob_frame)       # (4,G,G)

                # ------------------------------------------
                # 4) done / reward
                # ------------------------------------------
                all_finished = _all_robots_finished(robots_padded)
                done = (step >= max_steps - 1) or all_finished

                reward = 0.0
                r_k = 0.0

                if all_finished:
                    reward += FINISHED_BONUS * (1 - step / max_steps)

                if REWARD_K :
                    r_k += env_model.reward_penalty_collision() * REWARD_K

                reward_emit = (
                    (step % ACTION_SCALE == (ACTION_SCALE - 1) and step > ACTION_SCALE)
                    or (all_finished and step > ACTION_SCALE)
                )

                if reward_emit:
                    r_a = r_b = r_c = r_d = r_e = 0.0
                    r_f = r_g = r_h = r_i = r_j = 0.0
                    r_l = 0.0

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
        
                    reward += (
                        r_a + r_b + r_c + r_d + r_e +
                        r_g + r_h + r_i + r_j + r_k + r_l +
                        REWARD_FIXED
                    )
                    r_k = 0.0

                    if reward < -1e3:
                        raise RuntimeError(f"Reward collapsed: {reward}")

                    # --------------------------------------
                    # 5) send one joint transition per real robot
                    # --------------------------------------
                    try:
                        total_reward += reward  # timestep reward는 1번만 누적

                        for agent_i in range(MAX_ROBOTS):
                            if buffered_joint_mask[agent_i] < 0.5:
                                continue

                            msg = TransitionMsg(
                                worker_id=worker_id,
                                joint_ego_state=np.copy(buffered_joint_ego_state),
                                global_state=np.copy(buffered_global_state),
                                joint_robot_state=np.copy(buffered_joint_robot_state),
                                joint_action=np.copy(buffered_joint_action),
                                joint_mask=np.copy(buffered_joint_mask),

                                next_joint_ego_state=np.copy(next_joint_ego_state),
                                next_global_state=np.copy(next_global_state),
                                next_joint_robot_state=np.copy(next_joint_robot_state),
                                next_joint_mask=np.copy(next_joint_mask),

                                reward=float(reward),
                                done=bool(done),
                                agent_index=int(agent_i),
                            )
                            transition_queue.put(msg)  # blocking
                    except Exception as e:
                        print(f"[Worker {worker_id}] transition_queue.put error: {e}")
                        abnormal_reward = 1

                    # reward boundary 뒤에 실제 현재 state를 next state로 넘김
                    joint_ego_state = np.copy(next_joint_ego_state)
                    global_state = np.copy(next_global_state)
                    joint_robot_state = np.copy(next_joint_robot_state)
                    joint_mask = np.copy(next_joint_mask)
                else:
                    # 중간 step에서는 최신 robot state / mask만 유지
                    joint_robot_state = next_joint_robot_state
                    joint_mask = next_joint_mask

                # ------------------------------------------
                # 80%, 100% crowd evacuation stats
                # ------------------------------------------
                alive_agents = env_model.alived_agents()

                if (alive_agents < env_model.total_agents * 0.2) and (evacuation_time_80 == max_steps):
                    evacuation_time_80 = step

                if (alive_agents < 1) and (evacuation_time_100 == max_steps):
                    evacuation_time_100 = step

                if done:
                    try:
                        agent_total_lifetime = env_model.calculate_all_agents_life_time()
                    except Exception:
                        agent_total_lifetime = 0.0
                    break

        except Exception as e:
            print(f"[Worker {worker_id}] Error in episode loop: {e}")
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
        action_dim: int = 2,
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
            "ego_state_shape", "global_state_shape",
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
    # compress=False 로 설정하면 256 압축을 생략합니다.
    def __init__(self, input_shape, in_channels=4, channels=[16, 32, 32], out_dim=256, compress=True):
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



##########################################################################
# 3) Critic (Q) Network
##########################################################################
class QNetwork(nn.Module):
    def __init__(self, ego_shape=(25, 25), global_shape=(50, 50), action_dim=2, robot_dim = 3, use_robot: bool = True):
        super(QNetwork, self).__init__()
        self.use_robot_state = use_robot
       
        # --- Ego & Global Encoders ---
        self.ego_enc = ImpalaCNN(input_shape=ego_shape, compress=True)
        self.glob_enc = ImpalaCNN(input_shape=global_shape, compress=True)
       
        # Robot State
        robot_feat_dim = 0
        if self.use_robot_state:
            robot_input_dim = 3
            robot_embed_dim = 32
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_input_dim, robot_embed_dim),
                nn.SiLU()
            )
            robot_feat_dim = robot_embed_dim
        else:
            self.robot_fc = None

        self.img_dim = self.ego_enc.out_dim + self.glob_enc.out_dim
        cond_dim = action_dim + (robot_feat_dim if self.use_robot_state else 0)
        hidden = 256
        self.film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.img_dim)
        )

        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        fusion_dim = self.img_dim + robot_feat_dim + action_dim
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, ego_state, global_state, action, robot_state=None):
        # 1. Image Features
        e = self.ego_enc(ego_state)
        g = self.glob_enc(global_state)
        img = torch.cat([e, g], dim=1)

        r = None
        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("Model requires robot_state, but input is None")
            r = self.robot_fc(robot_state)

        # 2. Robot Features
        if self.use_robot_state:
            cond = torch.cat([action, r], dim=1)
        else:
            cond = action
        
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        img_film = (1.0 + gamma) * img + beta

        feats = [img_film]
        if self.use_robot_state:
            feats.append(r)
        feats.append(action)
        combined = torch.cat(feats, dim=1)

        # if self.use_robot_state:
        #     feats.append(r)
        # feats.append(action)
        combined = torch.cat(feats, dim=1)

        out = F.silu(self.fc1(combined))
        out = F.silu(self.fc2(out))
        q_val = self.q_out(out)
        return q_val


class CentralizedQNetwork(nn.Module):
    def __init__(
        self,
        ego_shape=(25, 25),
        global_shape=(50, 50),
        action_dim=2,
        robot_dim=3,
        max_robots=3,
        use_robot: bool = True,
    ):
        super().__init__()
        self.use_robot_state = use_robot
        self.max_robots = max_robots
        self.action_dim = action_dim
        self.robot_dim = robot_dim

        # shared ego encoder
        self.ego_enc = ImpalaCNN(input_shape=ego_shape, compress=True)
        self.glob_enc = ImpalaCNN(input_shape=global_shape, compress=True)

        self.single_ego_feat_dim = self.ego_enc.out_dim
        self.global_feat_dim = self.glob_enc.out_dim

        self.all_ego_feat_dim = self.single_ego_feat_dim * max_robots
        self.img_dim = self.global_feat_dim + self.all_ego_feat_dim

        robot_embed_dim = 32
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_dim, robot_embed_dim),
                nn.SiLU()
            )
            self.all_robot_feat_dim = robot_embed_dim * max_robots
        else:
            self.robot_fc = None
            self.all_robot_feat_dim = 0

        self.all_action_dim = action_dim * max_robots

        cond_dim = self.all_action_dim + self.all_robot_feat_dim
        hidden = 256
        self.film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * self.img_dim)
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        fusion_dim = self.img_dim + self.all_action_dim + self.all_robot_feat_dim
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(
        self,
        joint_ego_state,     # (B, N, 4, E, E)
        global_state,        # (B, 4, G, G)
        joint_action,        # (B, N, A)
        joint_robot_state,   # (B, N, R)
        joint_mask=None      # (B, N)
    ):
        B, N, C, H, W = joint_ego_state.shape
        assert N == self.max_robots

        # 1) shared ego encoding
        ego_flat = joint_ego_state.reshape(B * N, C, H, W)
        ego_feat = self.ego_enc(ego_flat).reshape(B, N, -1)   # (B, N, Fego)

        # mask 적용
        if joint_mask is not None:
            ego_feat = ego_feat * joint_mask.unsqueeze(-1).float()

        ego_feat_cat = ego_feat.reshape(B, -1)  # (B, N*Fego)

        # 2) global encoding
        glob_feat = self.glob_enc(global_state)  # (B, Fglob)

        # 3) image fusion
        img_feat = torch.cat([glob_feat, ego_feat_cat], dim=1)  # (B, img_dim)

        # 4) robot embedding
        if self.use_robot_state:
            robot_flat = joint_robot_state.reshape(B * N, self.robot_dim)
            robot_feat = self.robot_fc(robot_flat).reshape(B, N, -1)  # (B, N, Frobot)
            if joint_mask is not None:
                robot_feat = robot_feat * joint_mask.unsqueeze(-1).float()
            robot_feat_cat = robot_feat.reshape(B, -1)
        else:
            robot_feat_cat = None

        # 5) joint action flatten
        if joint_mask is not None:
            joint_action = joint_action * joint_mask.unsqueeze(-1).float()
        action_cat = joint_action.reshape(B, -1)

        # 6) FiLM conditioning
        if self.use_robot_state:
            cond = torch.cat([action_cat, robot_feat_cat], dim=1)
        else:
            cond = action_cat

        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        img_film = (1.0 + gamma) * img_feat + beta

        # 7) final Q head
        feats = [img_film, action_cat]
        if self.use_robot_state:
            feats.append(robot_feat_cat)

        x = torch.cat(feats, dim=1)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        q = self.q_out(x)
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
        self.ego_enc = ImpalaCNN(input_shape=ego_shape, compress=True)
        self.glob_enc = ImpalaCNN(input_shape=global_shape, compress=True)

        # Robot State
        self.robot_feat_dim = 0
        if self.use_robot_state :
            robot_input_dim = 3
            robot_embed_dim = 32
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_input_dim, robot_embed_dim),
                nn.SiLU()
            )
            self.robot_feat_dim = robot_embed_dim
        else :
            self.robot_fc = None

        # Fusion Dimension = Ego_CNN + Global_CNN + Robot
        fusion_dim = self.ego_enc.out_dim + self.glob_enc.out_dim + self.robot_feat_dim

        self.fc_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU()
        )
       
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def backbone(self, ego_state, global_state, robot_state=None):
        e = self.ego_enc(ego_state)
        g = self.glob_enc(global_state)
       
        feature_list = [e, g]
       
        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("Policy requires robot_state, but input is None.")
            r = self.robot_fc(robot_state)
            feature_list.append(r)

        combined = torch.cat(feature_list, dim=1)
        feat = self.fc_backbone(combined)
        return feat

    def forward(self, ego_state, global_state, robot_state=None):
        feat = self.backbone(ego_state, global_state, robot_state)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_action(self, ego_state, global_state, robot_state=None, temperature=1.0):
        # 파라미터가 2개 이미지 입력을 받도록 수정됨
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        std = log_std.exp()
        eps = torch.randn_like(mean) * temperature
        u = mean + std * eps
       
        # 기존 로직 유지 (Sigmoid Scaling)
        sigma = torch.sigmoid(u)
        action = 4 * sigma - 2  # [-2,2] 범위 매핑
       
        # 가우시안 로그확률
        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        log_prob_u = log_prob_u.sum(dim=1)
       
        # 시그모이드 야코비안 보정
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
        
        self.target_entropy = -2
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=int(replay_size),
            max_robots=MAX_ROBOTS,
            ego_state_shape=(4, EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_state_shape=(4, DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=self.robot_dim,
            device=self.device,
        )

        self.q1 = CentralizedQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2 = CentralizedQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q1_target = CentralizedQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2_target = CentralizedQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=2,
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
        if len(self.replay_buffer) < self.batch_size * START_BATCH_TIMES:
            return

        (
            joint_ego, global_state, joint_robot, joint_action, joint_mask,
            next_joint_ego, next_global_state, next_joint_robot, next_joint_mask,
            reward, done, agent_index
        ) = self.replay_buffer.sample(self.batch_size)

        self.alpha = self.log_alpha.exp().detach()

        # -----------------------
        # 1) Critic target
        # -----------------------
        with torch.no_grad():
            next_joint_action, next_joint_logp = self.sample_joint_actions(
                next_joint_ego, next_global_state, next_joint_robot, next_joint_mask
            )  # (B,N,A), (B,N)

            next_logp_i = gather_agent_tensor(next_joint_logp.unsqueeze(-1), agent_index).squeeze(-1)

            q1_next = self.q1_target(
                next_joint_ego, next_global_state, next_joint_action, next_joint_robot, next_joint_mask
            )
            q2_next = self.q2_target(
                next_joint_ego, next_global_state, next_joint_action, next_joint_robot, next_joint_mask
            )
            q_next = torch.min(q1_next, q2_next).squeeze(-1)

            q_target = reward + self.gamma * (1 - done) * (q_next - self.alpha * next_logp_i)

        # -----------------------
        # 2) Critic update
        # -----------------------
        q1_val = self.q1(joint_ego, global_state, joint_action, joint_robot, joint_mask).squeeze(-1)
        q2_val = self.q2(joint_ego, global_state, joint_action, joint_robot, joint_mask).squeeze(-1)

        loss_q1 = F.mse_loss(q1_val, q_target)
        loss_q2 = F.mse_loss(q2_val, q_target)

        max_grad_norm = 1.0

        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), max_grad_norm)
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), max_grad_norm)
        self.q2_optimizer.step()

        # -----------------------
        # 3) Actor update
        # -----------------------
        new_joint_action, joint_logp = self.sample_joint_actions(
            joint_ego, global_state, joint_robot, joint_mask
        )
        logp_i = gather_agent_tensor(joint_logp.unsqueeze(-1), agent_index).squeeze(-1)

        q1_new = self.q1(joint_ego, global_state, new_joint_action, joint_robot, joint_mask)
        q2_new = self.q2(joint_ego, global_state, new_joint_action, joint_robot, joint_mask)
        q_new = torch.min(q1_new, q2_new).squeeze(-1)

        policy_loss = (self.alpha * logp_i - q_new).mean()

        alpha_loss = -(self.log_alpha * (logp_i + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
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
                msg.agent_index,
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