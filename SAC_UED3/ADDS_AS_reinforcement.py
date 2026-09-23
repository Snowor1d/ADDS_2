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
from typing import Tuple, Any, Union, Optional, Dict
from config import *
from robot_action import ACTION_DIM as ROBOT_ACTION_DIM
import robot_action

from dataclasses import dataclass
import multiprocessing as mp
import queue
from queue import Empty, Full # for Empty


from pathlib import Path
import imageio.v2 as imageio

from ued import UEDRunner, make_value_fn

DEBUG_SAVE = False
home_dir = os.path.expanduser("~")
DEBUG_DIR_TEMP = os.path.join(home_dir, LOG_DIR)
DEBUG_DIR = os.path.join(DEBUG_DIR_TEMP, "debug_frames")
DEBUG_EVERY_EP = 1  
DEBUG_STEPS = {100, 200, 300}  # 초반 3번 boundary만 저장


REWARD_COMPONENT_NAMES = (
    "reward_a",
    "reward_b",
    "reward_c",
    "reward_d",
    "reward_e",
    "reward_f",
    "reward_g",
    "reward_h",
    "reward_i",
    "reward_j",
    "reward_k",
    "reward_l",
    # The re-entry penalty. Added with the hazard task and missed here, which
    # the unit tests could not catch: they never build the per-episode
    # component dictionary. Every transition a worker tried to send raised
    # KeyError('reward_n'), so no transition ever reached the buffer and the
    # run trained on nothing while reporting no traceback.
    "reward_n",
    "reward_fixed",
    "reward_finished_bonus",
)



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
    """One instant of the whole team, not of one robot.

    The centralised critic scores an action against what the other robots were
    doing at that instant, so the transition that reaches it has to carry the
    team together. Splitting it into per-robot messages would lose which rows
    belonged to the same instant as soon as the queue interleaved them.

    Padded slots are zeros and are excluded by `joint_mask`.
    """

    worker_id: int
    joint_ego_state: np.ndarray       # (MAX_ROBOTS, 4, EGO, EGO)
    global_state: np.ndarray          # (4, DOWN, DOWN)
    joint_robot_state: np.ndarray     # (MAX_ROBOTS, ROBOT_STATE_DIM)
    joint_action: np.ndarray          # (MAX_ROBOTS, ACTION_DIM)
    joint_mask: np.ndarray            # (MAX_ROBOTS,)
    reward: float
    next_joint_ego_state: np.ndarray
    next_global_state: np.ndarray
    next_joint_robot_state: np.ndarray
    next_joint_mask: np.ndarray
    done: bool
    # UED bookkeeping. The level id lets the main process decide whether a
    # transition is stored under the replay-only rule; the episode key groups a
    # trajectory so MaxMC can be computed from the critic once it ends. Workers
    # interleave freely on the shared queue, so the key must identify the
    # worker as well as its episode.
    level_id: int = -1
    episode_key: tuple = ()
    is_replay: bool = False


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
    reward_components: Dict[str, float]
    level_id: int = -1
    is_replay: bool = False
    # The level's own free-flow evacuation estimate, which the curriculum's
    # success threshold is a multiple of. Computed by the environment, so it
    # rides back with the episode rather than being recomputed in the main
    # process.
    freeflow_steps: float = 0.0


STATE_SHAPE = (4, 50, 50)
INPUT_MAP_SIZE = 50
ROBOT_STATE_EMBEDDING = True
# (x, y, any_follower, log_width_scale, log_height_scale). The two scale terms
# were added for map-size generalisation; raising this invalidates checkpoints
# trained with the earlier 3-dim state.
# Position (2), local follower flag (1), map scale (2), and two hazard terms:
# the signed distance to the zone boundary normalised by the zone's own scale,
# and the share of the crowd still inside it.
#
# The hazard terms are what make the task legible to the policy. The signed
# distance says which side of the boundary this robot is on and by how much,
# which decides whether its job right now is pushing people out or turning
# people away; the occupancy says how much work is left, which the ego crop
# cannot show because most of the hazard is usually outside it.
#
# The last five say what this robot is signalling: a one-hot over
# ROBOT_MODES, then the signalled heading. In the joint observation that is
# how a robot learns what its teammates are doing, which is the whole point
# of a team: two robots leading the same pocket of crowd is wasted effort,
# and a signal cannot be seen from across the map.
#
# Raising this invalidates saved checkpoints.
ROBOT_STATE_DIM = 12

# The action: two numbers for movement, two for the signalled heading, and a
# one-hot over the signal modes. robot_action.py owns the layout.
ACTION_DIM = ROBOT_ACTION_DIM

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
ZSG_DIR = os.path.join(log_dir, "zero_shot")
UED_STATE_PATH = os.path.join(log_dir, "ued_curriculum.pkl")
UED_SNAPSHOT_DIR = os.path.join(log_dir, "ued_snapshots")
os.makedirs(BY_MAP_DIR, exist_ok = True)
os.makedirs(ZSG_DIR, exist_ok=True)
os.makedirs(UED_SNAPSHOT_DIR, exist_ok=True)
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
                            if np.isfinite(value):
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


def worker_process(
    worker_id: int,
    transition_queue: mp.Queue,
    stats_queue: mp.Queue,
    epsilon_shared: mp.Value,
    param_queue: mp.Queue,
    seed: int = 0,
    level_queue: mp.Queue = None,
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
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).to(device)
    policy.eval()

    def _build_ego_global_frames(env_model, robot_index: int = 0):
        # Native resolution, one pixel per metre. Asking for a fixed size here
        # would make the ego crop cover a different physical area on every map
        # and break its correspondence with ROBOT_VISION.
        full = env_model.return_current_image()  # (world_h, world_w) uint8
        rb = (env_model.robots[robot_index] if env_model.robots
              else env_model.robot)
        ix, iy = env_model.world_to_px(rb.xy[0], rb.xy[1])

        ego = ego_crop_from_full_map(full, (ix, iy), EGO_MAP_SIZE, pad_value=50)  # (EGO,EGO) uint8
        glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)                    # (DOWN,DOWN) uint8

        ego_f = ego.astype(np.float32) / 255.0
        glob_f = glob.astype(np.float32) / 255.0
        return full, ego_f, glob_f

    def _build_team_frames(env_model, ego_stacks, glob_stack):
        """One ego view per robot, padded to MAX_ROBOTS, plus the shared global.

        Padded slots hold zeros and are excluded by the mask. Filling them with
        a copy of a real robot's view instead would be worse than useless: the
        critic attends over the slots, and a duplicate reads as a second robot
        standing exactly where the first one is.
        """
        n = len(env_model.robots)
        # The stacked shape comes from a peek, not from an attribute: a frame
        # stack is a deque of frames and has no shape of its own.
        joint_ego = np.zeros(
            (MAX_ROBOTS, ego_stacks[0].stack_len, EGO_MAP_SIZE, EGO_MAP_SIZE),
            dtype=np.float32)
        joint_robot = np.zeros((MAX_ROBOTS, ROBOT_STATE_DIM), dtype=np.float32)
        mask = np.zeros((MAX_ROBOTS,), dtype=np.float32)
        glob_f = None
        for i in range(min(n, MAX_ROBOTS)):
            _, ego_f, glob_f = _build_ego_global_frames(env_model, i)
            joint_ego[i] = ego_stacks[i].peek_with(ego_f)
            joint_robot[i] = np.asarray(
                env_model.return_current_robot_state(i), dtype=np.float32)
            mask[i] = 1.0
        global_state = glob_stack.peek_with(glob_f)
        return joint_ego, global_state, joint_robot, mask
   
    # hearbeats = ctx.Array('d', N_WORKERS)
    # for i in range(N_WORKERS):
    #     heartbeats[i] = time.time()


    while True:  # 무한히 에피소드 반복
        # ----- 1) 환경 생성 -----
        # Take a curriculum level if one is waiting. The read is non-blocking
        # and falls back to the ordinary random-map path when the queue is
        # empty, so a slow level producer can never stall a worker; the
        # watchdog in Start_training.py restarts the run after 2000 s of no
        # progress, and blocking here would be the easiest way to trigger it.
        ued_level = None
        if level_queue is not None:
            try:
                ued_level = level_queue.get_nowait()
            except Exception:
                ued_level = None

        while True:
            try:
                if ued_level is not None:
                    number_of_agents = int(ued_level.crowd_size)
                elif CROWD_NUMBER_MIN == CROWD_NUMBER_MAX:
                    number_of_agents = CROWD_NUMBER_MIN
                else:
                    number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)

                # The level carries its own world size, which is a curriculum
                # design axis now; MAP_W/MAP_H are only the fallback for the
                # non-UED path.
                world_w = int(ued_level.width) if ued_level is not None else MAP_W
                world_h = int(ued_level.height) if ued_level is not None else MAP_H

                env_model = model.FightingModel(
                    number_of_agents,
                    world_w,
                    world_h,
                    model_num=-1,
                    robot='Q',
                    level=ued_level,
                )
                break
            except Exception as e:
                print(f"[Worker {worker_id}] env create error: {e}, retrying...")
                # A level that cannot be instantiated must not trap the worker
                # in this retry loop forever.
                ued_level = None

        try:
            freeflow_steps = float(env_model.free_flow_evacuation_steps())
        except Exception as e:
            print(f"[Worker {worker_id}] free-flow estimate failed: {e}")
            freeflow_steps = 0.0

        level_id = int(getattr(ued_level, "level_id", -1)) if ued_level is not None else -1
        is_replay = bool(getattr(ued_level, "is_replay", False)) if ued_level is not None else False
        episode_key = (int(worker_id), int(episode_idx))

        # ----- 2) 초기 state 세팅 -----
        #
        # One frame stack per robot. A shared stack would mix the team's views
        # into every robot's history, so a robot's own four frames would show
        # it teleporting between its teammates' positions.
        ego_stacks = [FrameStack2(4) for _ in range(MAX_ROBOTS)]
        glob_stack = FrameStack2(4)
        glob_f = None
        for i in range(len(env_model.robots[:MAX_ROBOTS])):
            _, ego_f, glob_f = _build_ego_global_frames(env_model, i)
            ego_stacks[i].reset(ego_f)
        # Padded slots still need a stack of the right shape for the joint
        # tensor; they are zeroed and masked out, never read by the critic.
        for i in range(len(env_model.robots), MAX_ROBOTS):
            ego_stacks[i].reset(np.zeros_like(ego_f))
        global_state = glob_stack.reset(glob_f)

        joint_ego, global_state, joint_robot, joint_mask = \
            _build_team_frames(env_model, ego_stacks, glob_stack)
        buffered_joint_ego = np.copy(joint_ego)
        buffered_global_state = np.copy(global_state)
        buffered_joint_robot = np.copy(joint_robot)
        buffered_joint_action = np.zeros((MAX_ROBOTS, ACTION_DIM), dtype=np.float32)
        buffered_joint_mask = np.copy(joint_mask)
   

        # 에피소드 통계 변수
        total_reward = 0.0
        episode_reward_components = {
            name: 0.0 for name in REWARD_COMPONENT_NAMES
        }
        evacuation_time_80 = max_steps
        evacuation_time_100 = max_steps
        agent_total_lifetime = 0.0
        abnormal_reward = 0

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
                # heartbeats[worker_id] = time.time()
                if step % ACTION_SCALE == 0:

                    full_u8, ego_f, glob_f = _build_ego_global_frames(env_model)

                    if (
                        DEBUG_SAVE
                       
                    ):  
                        full_u8_r = np.flip(np.flip(full_u8, axis=-1), axis=-2)
                        ego_f_r   = np.flip(np.flip(ego_f,   axis=-1), axis=-2)
                        glob_f_r  = np.flip(np.flip(glob_f,  axis=-1), axis=-2)
                        print("저장합니다")
                        save_debug_triplet(
                            save_dir=DEBUG_DIR,
                            worker_id=worker_id,
                            episode_idx=episode_idx,
                            step=step,
                            full_u8=np.flip(full_u8_r, axis=1),
                            ego_f=np.flip(ego_f_r, axis=1),
                            glob_f=np.flip(glob_f_r, axis=1),
                            ego_state=joint_ego[0],
                            global_state=global_state
                        )
                    if step > 0:
                        for _ri in range(len(env_model.robots[:MAX_ROBOTS])):
                            _, _ef, _gf = _build_ego_global_frames(env_model, _ri)
                            ego_stacks[_ri].append(_ef)
                        global_state = glob_stack.append(_gf)
                    # One action per robot, from the same shared actor run
                    # once per robot on that robot's own view. Decentralised
                    # execution: no robot sees another's observation here, and
                    # the coordination they learn has to come through the
                    # centralised critic during training rather than through a
                    # channel that would not exist on real hardware.
                    joint_ego, global_state, joint_robot, joint_mask = \
                        _build_team_frames(env_model, ego_stacks, glob_stack)
                    joint_action = np.zeros((MAX_ROBOTS, ACTION_DIM), dtype=np.float32)

                    for ri, rb in enumerate(env_model.robots[:MAX_ROBOTS]):
                        # --- epsilon-greedy, drawn per robot ---
                        #
                        # Per robot rather than once for the team: a single
                        # draw makes the whole team explore or exploit
                        # together, which is the one correlation that stops a
                        # team from discovering a division of labour.
                        if np.random.rand() < eps or policy is None:
                            # The signal mode is explored too. Exploring only
                            # the movement would leave the policy no
                            # experience of what the modes do, and the mode is
                            # the half of the action the crowd responds to.
                            action_np = robot_action.random_action()
                        else:
                            ego_t = torch.from_numpy(joint_ego[ri]).unsqueeze(0).float().to(device)
                            glob_t = torch.from_numpy(global_state).unsqueeze(0).float().to(device)
                            robot_t = torch.from_numpy(joint_robot[ri]).unsqueeze(0).float().to(device)

                            with torch.no_grad():
                                action_t, _ = policy.sample_action(ego_t, glob_t, robot_t, temperature=1.0)
                            action_np = action_t.cpu().numpy()[0].astype(np.float32)

                        # The move can be rewritten by the robot itself when
                        # it is following a waypoint, so what goes in the
                        # buffer is what the environment used, not what the
                        # policy asked for.
                        real_move = robot_action.apply_to(rb, action_np)
                        joint_action[ri] = action_np
                        joint_action[ri, 0] = real_move[0]
                        joint_action[ri, 1] = real_move[1]

                    buffered_joint_ego = np.copy(joint_ego)
                    buffered_global_state = np.copy(global_state)
                    buffered_joint_robot = np.copy(joint_robot)
                    buffered_joint_action = np.copy(joint_action)
                    buffered_joint_mask = np.copy(joint_mask)

                # -----------------------------
                # 2) env step
                # -----------------------------
                env_model.step()


                # -----------------------------
                # 3) next state 계산
                # -----------------------------
                next_joint_ego, next_global_state, next_joint_robot, next_joint_mask = \
                    _build_team_frames(env_model, ego_stacks, glob_stack)


                # -----------------------------
                # 4) done / reward 계산
                # -----------------------------
                # Any robot may end the episode; the task is the team's.
                # Any robot may end the episode; the task is the team's. The
                # hold-for-cleared rule below can also set it, but that runs
                # after this line, so it must not be read here.
                team_finished = any(rb.is_game_finished
                                    for rb in env_model.robots)
                done = (step >= max_steps - 1) or team_finished
                reward = 0.0
                r_k = 0.0
                r_finished_bonus = 0.0

                if team_finished:
                    r_finished_bonus = FINISHED_BONUS * (1 - step / max_steps)
                    reward += r_finished_bonus

                if REWARD_K:
                    r_k += env_model.reward_penalty_collision() * REWARD_K

                # ACTION_SCALE 마지막 스텝이거나 게임 끝난 경우에만 reward shaping + transition 전송
                if ((step % ACTION_SCALE == (ACTION_SCALE - 1) and step > ACTION_SCALE) or
                    (team_finished and step > ACTION_SCALE)):

                    r_a = r_b = r_c = r_d = r_e = 0.0
                    r_f = r_g = r_h = r_i = r_j = 0.0
                    r_l = 0.0
                    r_n = 0.0

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
                    if REWARD_N:
                        r_n = env_model.reward_based_reentry() * REWARD_N

                    reward += (
                        r_a + r_b + r_c + r_d + r_e + r_f + r_g +
                        r_h + r_i + r_j + r_k + r_l + r_n + REWARD_FIXED
                    )
                    if reward < -1e3:
                        raise RuntimeError(f"Reward collapsed: {reward}")

                    # -----------------------------
                    # 5) transition Queue로 전송
                    # -----------------------------
                    try:
                        msg = TransitionMsg(
                            worker_id=worker_id,
                            joint_ego_state=buffered_joint_ego,
                            global_state=buffered_global_state,
                            joint_robot_state=buffered_joint_robot,
                            joint_action=buffered_joint_action,
                            joint_mask=buffered_joint_mask,
                            reward=float(reward),
                            next_joint_ego_state=next_joint_ego,
                            next_global_state=next_global_state,
                            next_joint_robot_state=next_joint_robot,
                            next_joint_mask=next_joint_mask,
                            done=bool(done),
                            level_id=level_id,
                            episode_key=episode_key,
                            is_replay=is_replay,
                        )
                        transition_queue.put(msg)  # blocking
                        total_reward += reward
                        component_values = {
                            "reward_a": r_a,
                            "reward_b": r_b,
                            "reward_c": r_c,
                            "reward_d": r_d,
                            "reward_e": r_e,
                            "reward_f": r_f,
                            "reward_g": r_g,
                            "reward_h": r_h,
                            "reward_i": r_i,
                            "reward_j": r_j,
                            "reward_k": r_k,
                            "reward_l": r_l,
                            "reward_n": r_n,
                            "reward_fixed": REWARD_FIXED,
                            "reward_finished_bonus": r_finished_bonus,
                        }
                        for name, value in component_values.items():
                            episode_reward_components[name] += float(value)
                    except Exception as e:
                        print(f"[Worker {worker_id}] transition_queue.put error: {e}")
                        abnormal_reward = 1

                    # The joint state is rebuilt from the environment at the
                    # top of each action step, so there is nothing to carry
                    # forward here any more; it used to hold one robot's
                    # scalar vector.

                # 80%, 100% 대피 시간
                #
                # "Cleared" now means the hazard has been empty and stayed
                # empty. Nobody is removed from the simulation, so the zone can
                # empty for a single step because a straggler stepped over the
                # line and refill on the next; recording that step as the
                # clearing time would report a social-force jostle rather than
                # the guidance. The clock is therefore only stopped after the
                # zone has held empty for DANGER_CLEAR_HOLD_STEPS, and the
                # time recorded is when it first emptied, not when the hold
                # completed, so the measure stays comparable with the
                # free-flow estimate it is normalised by.
                in_danger = env_model.alived_agents()
                if (in_danger < env_model.total_agents * 0.2 and
                        evacuation_time_80 == max_steps):
                    evacuation_time_80 = step
                # The model owns the hold rule, so the robot and the trainer
                # cannot disagree about when the task is finished. The time is
                # recorded whether or not it also ends the episode, so the
                # success criterion and its free-flow normalisation mean the
                # same thing under both termination settings.
                if (env_model.is_cleared_and_held()
                        and evacuation_time_100 == max_steps):
                    evacuation_time_100 = int(env_model.cleared_at() or step)
                if env_model.should_finish():
                    done = True

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
            abnormal=int(abnormal_reward),
            reward_components=episode_reward_components,
            level_id=level_id,
            is_replay=is_replay,
            freeflow_steps=float(freeflow_steps),
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
# Robot-order augmentation, ported from SAC_MARL_G.
#
# The critic attends over robots and is equivariant to their order by
# construction, but the episodes are not: robot zero is always the one placed
# first, so without this the buffer only ever shows one assignment of roles to
# slots and the actor can learn to rely on it. Permuting each sampled batch
# makes a team of three cover all six orderings of itself.

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


# ---------------------------------------------------------------------------
# Joint replay buffer, ported from SAC_MARL_G.
#
# A transition is now the whole team's, not one robot's. It has to be: the
# centralised critic scores an action in the context of what the other robots
# were doing at that instant, and a buffer of per-robot rows cannot reconstruct
# which rows belonged to the same instant once they have been shuffled.
#
# Rows are padded to MAX_ROBOTS and carry a mask, so a two-robot episode and a
# three-robot one sit in the same buffer and train the same weights. That is
# what makes UED_ROBOT_RANGE a curriculum axis rather than a separate
# experiment per team size.
# ---------------------------------------------------------------------------

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



# The encoder gained two options when the centralised critic was ported from
# SAC_MARL_G, and both default to the previous behaviour so the actor and the
# single-robot path are unchanged.
#
# `return_2d` hands back the convolutional feature map instead of a flat
# vector, which the critic's spatial attention needs: it asks which part of
# the global view each robot should be read against, and that question has no
# answer once the map has been flattened.
#
# `compress` puts a linear layer on the ego branch so every robot contributes
# a 256-wide embedding rather than a full flattened map. Without it the joint
# ego tensor for a team of three is three times the width of the whole
# single-robot observation, and the attention over robots is swamped by it.
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


# ---------------------------------------------------------------------------
# Centralised critic, ported from SAC_MARL_G.
#
# The task moved to a team, and a team needs a critic that sees the team. With
# one critic per robot seeing only its own observation, two robots sweeping
# the same corner of the hazard each get credited for the clearing, and
# neither learns that the other made its own work redundant. Here the critic
# takes the joint observation and the joint action and returns one value, so
# crediting is a property of what the team did together.
#
# Attention over robots rather than concatenation, so the value is invariant
# to the order the robots are listed in, and so a shorter team is handled by
# masking rather than by a differently shaped network. That is what lets
# UED_ROBOT_RANGE vary the team size within one policy.
#
# The actor stays decentralised and shared: one set of weights, run once per
# robot on that robot's own observation. That is what makes a team size the
# environment chooses at episode start something the same policy can execute.
# ---------------------------------------------------------------------------

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
        action_dim=ACTION_DIM,
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

class QNetwork(nn.Module):
    def __init__(self, ego_shape=(25, 25), global_shape=(50, 50), action_dim=ACTION_DIM, robot_dim = 3, use_robot: bool = True):
        super(QNetwork, self).__init__()
        self.use_robot_state = use_robot
       
        # --- Ego & Global Encoders ---
        # EGO_USE에 따라 encoder 생성 여부 결정
        self.ego_enc = CNNEncoder(input_shape=ego_shape, in_channels=4)
        self.glob_enc = CNNEncoder(input_shape=global_shape, in_channels=4)
       
        # Robot State
        robot_feat_dim = 0
        if self.use_robot_state:
            # Honour the argument. Both networks took robot_dim and then built
            # the layer at a hardcoded 3, which went unnoticed while the robot
            # state happened to be 3-dimensional; adding the two map-scale
            # terms made the critic reject its own observations.
            robot_input_dim = int(robot_dim)
            robot_embed_dim = 32
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_input_dim, robot_embed_dim),
                nn.SiLU()
            )
            robot_feat_dim = robot_embed_dim
        else:
            self.robot_fc = None

        # Feature Dimension 계산
        self.img_dim = self.glob_enc.out_dim
        if EGO_USE:
            self.img_dim += self.ego_enc.out_dim

        # FiLM Layer: FiLM_USE가 True일 때만 레이어 구성
        cond_dim = action_dim + (robot_feat_dim if self.use_robot_state else 0)
        if FiLM_USE:
            hidden = 256
            self.film = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 2 * self.img_dim)
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        else:
            self.film = None

        fusion_dim = self.img_dim + robot_feat_dim + action_dim
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, ego_state, global_state, action, robot_state=None):
        # 1. Image Features
        g = self.glob_enc(global_state)
        if EGO_USE:
            e = self.ego_enc(ego_state)
            img = torch.cat([e, g], dim=1)
        else:
            img = g

        r = None
        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("Model requires robot_state, but input is None")
            r = self.robot_fc(robot_state)

        # 2. Conditioning (FiLM or Cat)
        if self.use_robot_state:
            cond = torch.cat([action, r], dim=1)
        else:
            cond = action
        
        # FiLM 사용 여부에 따른 분기
        if FiLM_USE:
            gamma_beta = self.film(cond)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            img_processed = (1.0 + gamma) * img + beta
        else:
            img_processed = img

        feats = [img_processed]
        if self.use_robot_state:
            feats.append(r)
        feats.append(action)
        combined = torch.cat(feats, dim=1)

        out = F.silu(self.fc1(combined))
        out = F.silu(self.fc2(out))
        q_val = self.q_out(out)
        return q_val
    
    
##########################################################################
# 4) Policy (Actor) Network
##########################################################################
class PolicyNetwork(nn.Module):
    def __init__(self, ego_shape=(25,25), global_shape=(50,50), robot_dim = 3, use_robot: bool = True):
        super(PolicyNetwork, self).__init__()
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.use_robot_state = use_robot

        # --- Encoders ---
        self.ego_enc = CNNEncoder(input_shape=ego_shape, in_channels=4) if EGO_USE else None
        self.glob_enc = CNNEncoder(input_shape=global_shape, in_channels=4)

        # Robot State
        self.robot_feat_dim = 0
        if self.use_robot_state :
            robot_input_dim = int(robot_dim)
            robot_embed_dim = 32
            self.robot_fc = nn.Sequential(
                nn.Linear(robot_input_dim, robot_embed_dim),
                nn.SiLU()
            )
            self.robot_feat_dim = robot_embed_dim

        # Fusion Dimension 계산
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
       
        # Four continuous numbers: where to move and which heading to
        # signal. See robot_action.py for the layout.
        self.mean_head = nn.Linear(64, 4)
        self.log_std_head = nn.Linear(64, 4)

        # And a categorical head over the signal modes. A discrete head
        # rather than a continuous number cut into three bands: a threshold
        # puts flat regions in the policy gradient, so the mode would stop
        # moving as soon as the underlying number drifted into the middle of
        # a band.
        self.mode_head = nn.Linear(64, len(ROBOT_MODES))

    def backbone(self, ego_state, global_state, robot_state=None):
        g = self.glob_enc(global_state)
        feature_list = []
        
        if EGO_USE:
            e = self.ego_enc(ego_state)
            feature_list.append(e)
            
        feature_list.append(g)
       
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
        mode_logits = self.mode_head(feat)
        return mean, log_std, mode_logits

    def deterministic_action(self, ego_state, global_state, robot_state=None):
        """The greedy action, for evaluation.

        The continuous half is the squashed mean and the mode is the argmax,
        so a held-out score does not move because a coin came up differently.
        """
        mean, _, mode_logits = self.forward(ego_state, global_state,
                                            robot_state)
        cont = 4 * torch.sigmoid(mean) - 2
        one_hot = F.one_hot(mode_logits.argmax(dim=-1),
                            num_classes=mode_logits.shape[-1]).float()
        return torch.cat([cont, one_hot], dim=-1)

    def sample_action(self, ego_state, global_state, robot_state=None, temperature=1.0):
        # 파라미터가 2개 이미지 입력을 받도록 수정됨
        mean, log_std, mode_logits = self.forward(ego_state, global_state,
                                                  robot_state)
        std = log_std.exp()
        eps = torch.randn_like(mean) * temperature
        u = mean + std * eps

        # 기존 로직 유지 (Sigmoid Scaling)
        sigma = torch.sigmoid(u)
        cont = 4 * sigma - 2  # [-2,2] 범위 매핑

        # 가우시안 로그확률
        log_prob_u = -0.5 * (((u - mean) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        log_prob_u = log_prob_u.sum(dim=1)

        # 시그모이드 야코비안 보정
        jacobian = torch.log(4*sigma*(1 - sigma) + 1e-8).sum(dim=1)
        log_prob = log_prob_u - jacobian

        # The signal mode, sampled from the categorical head.
        #
        # Straight-through Gumbel-softmax: the vector handed to the critic is
        # a hard one-hot, so the environment and the critic see the same
        # discrete choice, while the gradient reaches the logits through the
        # soft relaxation. Sampling and treating the one-hot as a constant
        # would leave the mode head with no gradient from the critic at all.
        log_mode = F.log_softmax(mode_logits, dim=-1)
        one_hot = F.gumbel_softmax(mode_logits, tau=1.0, hard=True, dim=-1)
        log_prob = log_prob + (one_hot * log_mode).sum(dim=-1)

        return torch.cat([cont, one_hot], dim=-1), log_prob
   
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
        self.use_ego = EGO_USE
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
            action_dim=ACTION_DIM,
            robot_dim=self.robot_dim,
            device=self.device,
        )

        # Critic networks
        self.q1 = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=ACTION_DIM,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2 = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=ACTION_DIM,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q1_target = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=ACTION_DIM,
            robot_dim=ROBOT_STATE_DIM,
            max_robots=MAX_ROBOTS,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)

        self.q2_target = CentralizedAttentionQNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            action_dim=ACTION_DIM,
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
    def select_action(self, ego_state_np, global_state_np, robot_state_np=None,
                      deterministic=False, log_action=True):
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
                # 결정적 행동 선택: squashed mean + argmax mode.
                action_t = self.policy.deterministic_action(
                    ego_t, global_t, robot_t)
            else:
                # 비결정적 선택: sample_action에서 샘플링 (자코비안 보정 포함)
                action_t, log_prob = self.policy.sample_action(ego_t, global_t, robot_t)
        action_np = action_t.cpu().numpy()[0]
        if log_action:
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
        """One SAC step: centralised critic, shared decentralised actor.

        Two things differ from the single-robot version beyond the shapes.

        The critic returns one value per robot from the joint observation, and
        the Bellman target is taken over the team. The reward is the team's, so
        the value regressed against it has to be the team's too. Masked slots
        are zeroed before the sum, or a padded robot's arbitrary value would be
        added to every target.

        The actor update uses that critic but holds the other robots' actions
        at what the buffer recorded, replacing only the acting robot's. That is
        the counterfactual a shared decentralised actor needs: what this robot
        should have done given what the others actually did. Resampling every
        robot's action would ask what the team should have done collectively,
        which no single robot can act on at execution time.
        """
        if len(self.replay_buffer) < self.batch_size * START_BATCH_TIMES:
            return

        (joint_ego, glob_s, joint_robot, joint_action, joint_mask,
         next_joint_ego, glob_s2, next_joint_robot, next_joint_mask,
         r, d, agent_index, _dt) = self.replay_buffer.sample(self.batch_size)

        # Robot-order augmentation. Episodes always place robot zero first, so
        # without this the actor can come to rely on slot identity.
        (joint_ego, joint_robot, joint_action, joint_mask,
         next_joint_ego, next_joint_robot, next_joint_mask,
         agent_index) = random_permute_joint_batch(
            joint_ego, joint_robot, joint_action, joint_mask,
            next_joint_ego, next_joint_robot, next_joint_mask, agent_index)

        B, N = joint_mask.shape
        self.alpha = self.log_alpha.exp().detach()

        def per_robot_actions(ego, robot, glob):
            """The shared actor run once per slot, flattened into one batch."""
            flat_ego = ego.reshape(B * N, *ego.shape[2:])
            flat_glob = glob.repeat_interleave(N, dim=0)
            flat_robot = robot.reshape(B * N, -1)
            a, logp = self.policy.sample_action(flat_ego, flat_glob, flat_robot)
            return a.reshape(B, N, -1), logp.reshape(B, N)

        with torch.no_grad():
            next_a, next_logp = per_robot_actions(
                next_joint_ego, next_joint_robot, glob_s2)
            q1_next = self.q1_target(next_joint_ego, glob_s2, next_a,
                                     next_joint_robot, next_joint_mask)
            q2_next = self.q2_target(next_joint_ego, glob_s2, next_a,
                                     next_joint_robot, next_joint_mask)
            q_next = torch.min(q1_next, q2_next).squeeze(-1)          # (B, N)
            soft = (q_next - self.alpha * next_logp) * next_joint_mask
            # Mean over the real robots, so a two-robot team and a three-robot
            # one produce targets on the same scale.
            denom = next_joint_mask.sum(dim=1).clamp(min=1.0)
            q_target = r + self.gamma * (1 - d) * (soft.sum(dim=1) / denom)

        q1_val = self.q1(joint_ego, glob_s, joint_action, joint_robot,
                         joint_mask).squeeze(-1) * joint_mask
        q2_val = self.q2(joint_ego, glob_s, joint_action, joint_robot,
                         joint_mask).squeeze(-1) * joint_mask
        denom_c = joint_mask.sum(dim=1).clamp(min=1.0)
        loss_q = (F.mse_loss(q1_val.sum(dim=1) / denom_c, q_target)
                  + F.mse_loss(q2_val.sum(dim=1) / denom_c, q_target))

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss_q.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Policy 업데이트: one robot per sample, the rest held at their
        # recorded actions.
        idx = agent_index.long().clamp(0, N - 1)
        rows = torch.arange(B, device=idx.device)
        new_a, log_p = self.policy.sample_action(
            joint_ego[rows, idx], glob_s, joint_robot[rows, idx])
        mixed_action = joint_action.clone()
        mixed_action[rows, idx] = new_a
        q_new = torch.min(
            self.q1(joint_ego, glob_s, mixed_action, joint_robot, joint_mask),
            self.q2(joint_ego, glob_s, mixed_action, joint_robot, joint_mask),
        ).squeeze(-1)[rows, idx]
        
        policy_loss = (self.alpha * log_p - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
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


def _build_zero_shot_frames(env_model):
    """Build the same ego/global observations used by SAC worker processes."""
    full = env_model.return_current_image()
    ix, iy = env_model.world_to_px(env_model.robot.xy[0], env_model.robot.xy[1])

    ego = ego_crop_from_full_map(
        full,
        (ix, iy),
        EGO_MAP_SIZE,
        pad_value=50,
    )
    glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)
    return ego.astype(np.float32) / 255.0, glob.astype(np.float32) / 255.0


def evaluate_zero_shot_once(
    agent,
    map_num: int,
    robot_num: int = 1,
    deterministic: bool = True,
    seed: int = 0,
    level=None,
):
    """Run one SAC zero-shot episode and return evacuation_time_100."""
    if robot_num != 1:
        raise ValueError(
            "SAC_FE_RV3 supports one robot; ZSG_ROBOT_NUM must contain only 1."
        )

    import model

    np.random.seed(seed)
    random.seed(seed)

    if level is not None:
        number_of_agents = int(level.crowd_size)
    elif CROWD_NUMBER_MIN == CROWD_NUMBER_MAX:
        number_of_agents = CROWD_NUMBER_MIN
    else:
        number_of_agents = random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)

    # Evaluate the map in its own orientation. With augmentation on, each
    # evaluation seed would also draw one of eight symmetries, so the held-out
    # score would move for reasons that have nothing to do with the policy.
    prev_augmentation = model.MAP_DATA_AUGMENTATION
    model.MAP_DATA_AUGMENTATION = False
    try:
        env_model = model.FightingModel(
            number_of_agents,
            int(level.width) if level is not None else MAP_W,
            int(level.height) if level is not None else MAP_H,
            model_num=map_num,
            robot='Q',
            level=level,
        )
    finally:
        model.MAP_DATA_AUGMENTATION = prev_augmentation

    ego_f, glob_f = _build_zero_shot_frames(env_model)
    ego_stack = FrameStack2(4)
    glob_stack = FrameStack2(4)
    ego_state = ego_stack.reset(ego_f)
    global_state = glob_stack.reset(glob_f)

    evacuation_time_100 = MAX_STEPS

    for step in range(MAX_STEPS):
        if env_model.alived_agents() < 1:
            evacuation_time_100 = step
            break

        if env_model.robot.is_game_finished or step == MAX_STEPS - 1:
            break

        if step % ACTION_SCALE == 0:
            if step > 0:
                ego_f, glob_f = _build_zero_shot_frames(env_model)
                ego_state = ego_stack.append(ego_f)
                global_state = glob_stack.append(glob_f)

            robot_state = np.asarray(
                env_model.return_current_robot_state(),
                dtype=np.float32,
            )
            action, _ = agent.select_action(
                ego_state,
                global_state,
                robot_state,
                deterministic=deterministic,
                log_action=False,
            )
            # The whole action, not just the move: an evaluation that
            # dropped the signal would score a robot that never guides
            # anybody and call it the policy's performance.
            robot_action.apply_to(env_model.robot, action)

        env_model.step()

    return int(evacuation_time_100)


def run_zero_shot_evaluation(agent, episode: int, writer: SummaryWriter):
    """Evaluate all configured unseen maps and persist txt/TensorBoard metrics."""
    print(f"[ZeroShot] Start evaluation at episode {episode}")

    if ZSG_ITERATION <= 0:
        raise ValueError("ZSG_ITERATION must be greater than zero.")

    old_epsilon = agent.epsilon
    old_epsilon_long = agent.epsilon_long
    policy_was_training = agent.policy.training
    numpy_random_state = np.random.get_state()
    python_random_state = random.getstate()
    results = {}

    try:
        agent.epsilon = 0.0
        agent.epsilon_long = 0.0
        agent.policy.eval()

        for robot_num in ZSG_ROBOT_NUM:
            map_averages = []

            for map_num in ZSG_MAP:
                evacuation_times = []

                for iteration in range(ZSG_ITERATION):
                    seed = (
                        episode * 100000
                        + map_num * 100
                        + robot_num * 10
                        + iteration
                    )
                    evacuation_times.append(
                        evaluate_zero_shot_once(
                            agent=agent,
                            map_num=map_num,
                            robot_num=robot_num,
                            deterministic=True,
                            seed=seed,
                        )
                    )

                average = float(np.mean(evacuation_times))
                map_averages.append(average)
                results[(map_num, robot_num)] = average

                path = zsg_metric_path(map_num, robot_num)
                ensure_file(path)
                with open(path, "a") as f:
                    f.write(f"{episode}\t{average:.6f}\n")

                writer.add_scalar(
                    f"ZeroShot/Evacuation100/map_{map_num}/robot_{robot_num}",
                    average,
                    episode,
                )
                print(
                    f"[ZeroShot] episode={episode}, map={map_num}, "
                    f"robot={robot_num}, avg_evac100={average:.2f}, "
                    f"raw={evacuation_times}"
                )

            all_maps_average = float(np.mean(map_averages))
            results[("all_maps", robot_num)] = all_maps_average

            all_maps_path = zsg_all_maps_metric_path(robot_num)
            ensure_file(all_maps_path)
            with open(all_maps_path, "a") as f:
                f.write(f"{episode}\t{all_maps_average:.6f}\n")

            writer.add_scalar(
                f"ZeroShot/Evacuation100/all_maps/robot_{robot_num}",
                all_maps_average,
                episode,
            )
            print(
                f"[ZeroShot] episode={episode}, ALL_MAPS, robot={robot_num}, "
                f"avg_evac100={all_maps_average:.2f}, "
                f"map_avgs={map_averages}"
            )

        # Procedural half of the held-out set. These are the levels that test
        # transfer within the generator's own distribution, so they are the
        # direct evidence for the generalisation claim; they are reported per
        # difficulty as well as in aggregate, because a curriculum can easily
        # improve the easy end while losing ground on the hard end.
        if ZSG_HOLDOUT_LEVELS:
            from ued.holdout import holdout_levels

            per_difficulty = {}
            per_size = {}
            per_band = {}
            for level in holdout_levels():
                times = []
                for iteration in range(ZSG_HOLDOUT_ITERATION):
                    seed = episode * 100000 + int(level.source_seed or 0) + iteration
                    times.append(
                        evaluate_zero_shot_once(
                            agent=agent,
                            map_num=0,
                            robot_num=1,
                            deterministic=True,
                            seed=seed,
                            level=level,
                        )
                    )
                average = float(np.mean(times))
                results[("holdout", int(level.source_seed or 0))] = average
                per_difficulty.setdefault(int(level.difficulty or 0), []).append(average)
                # Normalised by the level's own free-flow estimate, so scores
                # from different map sizes are on one axis: a 180 m map takes
                # longer to evacuate than a 50 m one no matter how good the
                # policy is, and the raw times would just report that.
                per_size.setdefault(int(level.width), []).append(average)
                per_band.setdefault(getattr(level, "size_band", "inside"), []).append(average)

            all_holdout = [v for vals in per_difficulty.values() for v in vals]
            if all_holdout:
                holdout_average = float(np.mean(all_holdout))
                results[("holdout_all", 1)] = holdout_average
                writer.add_scalar(
                    "ZeroShot/Evacuation100/holdout_all", holdout_average, episode
                )
                for difficulty, vals in sorted(per_difficulty.items()):
                    writer.add_scalar(
                        f"ZeroShot/Evacuation100/holdout_d{difficulty}",
                        float(np.mean(vals)),
                        episode,
                    )
                for size, vals in sorted(per_size.items()):
                    writer.add_scalar(
                        f"ZeroShot/Evacuation100/holdout_size{size}",
                        float(np.mean(vals)),
                        episode,
                    )
                # Inside versus outside the training size range. Averaging the
                # two together would hide whether the policy extrapolates or
                # merely interpolates, which is the whole size-generalisation
                # question.
                for band, vals in sorted(per_band.items()):
                    band_mean = float(np.mean(vals))
                    results[("holdout_band", band)] = band_mean
                    writer.add_scalar(
                        f"ZeroShot/Evacuation100/holdout_band_{band}",
                        band_mean,
                        episode,
                    )
                holdout_path = os.path.join(ZSG_DIR, "holdout_all.txt")
                ensure_file(holdout_path)
                with open(holdout_path, "a") as f:
                    f.write(f"{episode}\t{holdout_average:.6f}\n")
                by_difficulty = {
                    d: round(float(np.mean(v)), 1)
                    for d, v in sorted(per_difficulty.items())
                }
                by_band = {
                    b: round(float(np.mean(v)), 1)
                    for b, v in sorted(per_band.items())
                }
                print(
                    f"[ZeroShot] episode={episode}, HOLDOUT_LEVELS, "
                    f"avg_evac100={holdout_average:.2f}, "
                    f"by_difficulty={by_difficulty}, by_size_band={by_band}"
                )

        writer.flush()
    finally:
        agent.epsilon = old_epsilon
        agent.epsilon_long = old_epsilon_long
        agent.policy.train(policy_was_training)
        np.random.set_state(numpy_random_state)
        random.setstate(python_random_state)

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

    def start_workers(ctx, n_workers, transition_queue, stats_queue, epsilon_shared, base_seed,
                      level_queues=None):
        workers = [None] * n_workers
        param_queues = [None] * n_workers
        for wid in range(n_workers):
            p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed,
                                     level_queues=level_queues)
            workers[wid] = p
            param_queues[wid] = pq
        return workers, param_queues

    def start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed,
                         level_queues=None):
        pq = ctx.Queue(maxsize=1)
        lq = level_queues[wid] if level_queues is not None and wid < len(level_queues) else None
        p = ctx.Process(
            target=worker_process,
            args=(wid, transition_queue, stats_queue, epsilon_shared, pq, base_seed, lq),
            daemon=True
        )
        p.start()
        print(f"[Main] Worker {wid} started, pid={p.pid}")
        return p, pq


    def restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed,
                       level_queues=None):
        old_p = workers[wid]
        try:
            if old_p is not None and old_p.is_alive():
                old_p.terminate()
                old_p.join(timeout=2)
        except Exception as e:
            print(f"[Main] terminate/join error for worker {wid}: {e}")

        # 새 큐/프로세스 생성
        p, pq = start_one_worker(ctx, wid, transition_queue, stats_queue, epsilon_shared, base_seed,
                                 level_queues=level_queues)

        workers[wid] = p
        param_queues[wid] = pq
        return


    def supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed,
                          level_queues=None):
        # 주기적으로 호출해서 죽은 worker만 재시작
        for wid, p in enumerate(workers):
            if p is None:
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed,
                               level_queues=level_queues)
                continue

            if not p.is_alive():
                ec = p.exitcode
                print(f"[Main] Worker {wid} died. exitcode={ec} -> restarting")
                restart_worker(ctx, wid, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed,
                               level_queues=level_queues)

    mp.set_start_method("spawn", force=True)  # or "fork", 리눅스면 fork도 가능

    # ----- 1) TensorBoard/로그 파일 설정 (기존 코드 유지 가능) -----
    total_reward_file = os.path.join(log_dir, "total_reward.txt")
    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    evacuation_time_80_file = os.path.join(log_dir, "evacuation_80.txt")
    evacuation_time_100_file = os.path.join(log_dir, "evacuation_100.txt")
    total_lifetime_file = os.path.join(log_dir, "total_lifetime.txt")
    reward_component_files = {
        name: os.path.join(log_dir, f"{name}.txt")
        for name in REWARD_COMPONENT_NAMES
    }

    #reward_vs_ls_file = os.path.join(log_dir, "reward_vs_learning_step.txt")
    #evac100_vs_ls_file = os.path.join(log_dir, "evac100_vs_learning_step.txt")

    #tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    step_writer = SummaryWriter(log_dir=tb_log_dir)


    # 파일 존재 보장
    for path in [total_reward_file, evacuation_time_80_file,
                 evacuation_time_100_file, total_lifetime_file,
                 *reward_component_files.values()]:
        if not os.path.exists(path):
            open(path, "w").close()

    # 기존 학습 로그에 reward component 파일만 새로 추가된 경우,
    # 복원할 수 없는 과거 episode는 nan으로 채워 episode 축을 맞춘다.
    with open(total_reward_file, "r") as f:
        total_reward_line_count = sum(1 for _ in f)
    for path in reward_component_files.values():
        with open(path, "r") as f:
            component_line_count = sum(1 for _ in f)
        if component_line_count < total_reward_line_count:
            with open(path, "a") as f:
                for _ in range(total_reward_line_count - component_line_count):
                    f.write("nan\n")

    # TensorBoard 관련 (원하면 기존 monitor thread 그대로 사용 가능)
    tb_process = launch_tensorboard(tb_log_dir, port=PORT_NUM)
    monitor_thread = threading.Thread(
        target=monitor_total_reward,
        args=(total_reward_file, tb_log_dir),
        daemon=True
    )
    monitor_thread.start()

    # episode별 reward component를 TensorBoard의 하나의 그룹으로 기록
    reward_component_monitor_threads = []
    for name, path in reward_component_files.items():
        monitor_thread_component = threading.Thread(
            target=monitor_metric,
            args=(path, f"Reward Components/{name}", tb_log_dir),
            daemon=True,
        )
        monitor_thread_component.start()
        reward_component_monitor_threads.append(monitor_thread_component)

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

    # 맵별 곡선은 고정된 맵 집합을 전제한다. UED에서는 레벨이 절차적으로
    # 만들어지고 계속 변이되므로 map_num이 모두 0이 되어 곡선이 의미를 잃고,
    # 쓰이지 않는 태일링 스레드만 수백 개 남는다. 집계 지표와 UED/* 스칼라로
    # 대체한다.
    for m in ([] if UED_ENABLED else MAP_NUM_RANDOM):
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

    transition_queue = ctx.Queue(maxsize=2*N_WORKERS_WARMUP)
    stats_queue = ctx.Queue(maxsize=2*N_WORKERS_WARMUP)
    param_queue = mp.Queue(maxsize=1)

    workers = [None] * N_WORKERS
    param_queues = [None] * N_WORKERS
    base_seed = 1234
    if global_episode >= START_UPDATE_EPISODE:
        current_n_workers = N_WORKERS_TRAIN
    else:
        current_n_workers = N_WORKERS_WARMUP

    # ----- UED curriculum -----
    # One level queue per worker slot, sized for the warm-up count so the same
    # queues survive the warm-up -> train worker switch.
    level_queues = [ctx.Queue(maxsize=UED_LEVEL_QUEUE_SIZE) for _ in range(max(N_WORKERS_WARMUP, N_WORKERS_TRAIN))]
    ued_runner = UEDRunner(value_fn=make_value_fn(agent))
    # Resume before the producer thread starts, so restored levels are what the
    # curriculum steers on rather than a population rebuilt from scratch.
    if model_load in (2, 3):
        ued_runner.load(UED_STATE_PATH)
        ued_runner.population.episode = max(ued_runner.population.episode, global_episode)
    ued_runner.start()
    if ued_runner.enabled:
        print(f"[Main] UED enabled: method={UED_METHOD} score={UED_SCORE} "
              f"replay_only_updates={UED_REPLAY_ONLY_UPDATES}")

    workers, param_queues = start_workers(ctx, current_n_workers, transition_queue, stats_queue, epsilon_shared, base_seed,
                                          level_queues=level_queues)
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
            supervise_workers(ctx, workers, param_queues, transition_queue, stats_queue, epsilon_shared, base_seed,
                              level_queues=level_queues)
            last_supervise_t = time.time()

        # 워커 레벨 큐 채우기 (non-blocking)
        ued_runner.dispatch(level_queues, global_episode)

        try:
            msg: TransitionMsg = transition_queue.get(timeout=1.0)
        except queue.Empty:
            # transition이 잠시 없는 경우, stats만 처리하고 계속
            pass
        else:
            ued_runner.on_transition(msg)

            # ReplayBuffer에 push
            if ued_runner.should_store(msg):
                # One joint transition per instant, stored once. The
                # `agent_index` the buffer carries says whose actor update a
                # sample is for; it is drawn per sample rather than fixed
                # here, so a team of three contributes to all three actors
                # from the same stored instant instead of tripling the
                # storage.
                n_real = int(msg.joint_mask.sum())
                agent.replay_buffer.push(
                    msg.joint_ego_state, msg.global_state,
                    msg.joint_robot_state, msg.joint_action, msg.joint_mask,
                    msg.next_joint_ego_state, msg.next_global_state,
                    msg.next_joint_robot_state, msg.next_joint_mask,
                    msg.reward, msg.done,
                    int(np.random.randint(max(1, n_real))), 1.0,
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
            ued_runner.population.episode = global_episode
            # The curriculum needs to know how exploratory the policy that
            # produced this episode was, so it can ignore outcomes that are
            # mostly random actions.
            ued_runner.epsilon = float(agent.epsilon)
            ued_runner.on_episode(s_msg)

            print("-----------------------------------------------")
            print(f"[Main] Episode {global_episode} (from worker {s_msg.worker_id})")
            print("Total reward:", s_msg.total_reward)
            print("evacuation_time_80 :", s_msg.evac_time_80)
            print("evacuation_time_100:", s_msg.evac_time_100)
            print("-----------------------------------------------")

            # 맵별 metric 기록 (UED에서는 map_num이 항상 0이라 의미가 없다)
            if not UED_ENABLED:
                ensure_file(map_metric_path("reward", s_msg.map_num))
                ensure_file(map_metric_path("evacuation_100", s_msg.map_num))

            if s_msg.abnormal != 1:
                if not UED_ENABLED:
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
                for name in REWARD_COMPONENT_NAMES:
                    with open(reward_component_files[name], "a") as f:
                        f.write(f"{s_msg.reward_components.get(name, 0.0)}\n")

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
                # Same cadence as the policy checkpoint: the two have to come
                # back together or the curriculum and the agent end up at
                # different points in training.
                ued_runner.save(UED_STATE_PATH)

            if (
                ZSG_CYCLE_EPISODE > 0
                and global_episode > 0
                and global_episode >= START_UPDATE_EPISODE
                and global_episode % ZSG_CYCLE_EPISODE == 0
            ):
                run_zero_shot_evaluation(
                    agent=agent,
                    episode=global_episode,
                    writer=step_writer,
                )

            # UED 진단 지표. 개체군 복잡도가 실제로 자라는지, 학습 레벨의
            # 성공률이 learnability가 겨냥하는 0.5 근처에 머무는지, 그리고
            # MaxMC와 learnability가 이 도메인에서 같은 것을 가리키는지를 본다.
            if ued_runner.enabled and global_episode % 20 == 0:
                for name, value in ued_runner.stats().items():
                    try:
                        step_writer.add_scalar(f"UED/{name}", float(value), global_episode)
                    except Exception:
                        pass

            # 개체군 그림. TensorBoard의 스텝 슬라이더로 커리큘럼이 변해가는
            # 과정을 넘겨볼 수 있다. PNG도 같이 남겨서 학습이 끝난 뒤에도
            # 같은 그림을 볼 수 있게 한다.
            if (
                ued_runner.enabled
                and UED_SNAPSHOT_EPISODES > 0
                and global_episode % UED_SNAPSHOT_EPISODES == 0
            ):
                try:
                    ued_runner.snapshot(step_writer, global_episode, save_dir=UED_SNAPSHOT_DIR)
                except Exception as e:
                    print(f"[UED] snapshot skipped: {e}")

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
            transition_queue = ctx.Queue(maxsize=2*N_WORKERS_TRAIN)
            stats_queue = ctx.Queue(maxsize=2*N_WORKERS_TRAIN)

            # 레벨 큐도 새로 만든다. 종료된 워커가 이미 꺼내간 레벨은 결과가
            # 돌아오지 않으므로, 남은 항목을 그대로 넘기면 population의 점수가
            # 영영 갱신되지 않는 레벨이 쌓인다.
            try:
                for lq in level_queues:
                    while not lq.empty():
                        lq.get_nowait()
            except Exception:
                pass
            level_queues = [ctx.Queue(maxsize=UED_LEVEL_QUEUE_SIZE)
                            for _ in range(max(N_WORKERS_WARMUP, N_WORKERS_TRAIN))]

            # 새 worker 시작
            current_n_workers = N_WORKERS_TRAIN
            workers, param_queues = start_workers(
                ctx, current_n_workers,
                transition_queue, stats_queue,
                epsilon_shared, base_seed,
                level_queues=level_queues
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
