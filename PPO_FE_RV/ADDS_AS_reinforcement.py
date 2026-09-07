"""Synchronous on-policy PPO baseline for the SAC_FE_RV3 environment.

The environment, observations, reward, action repeat, actor architecture, map
augmentation and evaluation protocol are inherited from SAC_FE_RV3.  Only the
learning algorithm is changed: replay-based SAC is replaced by clipped PPO
with GAE and a state-value critic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Dict, List, Optional, Sequence, Tuple
import multiprocessing as mp
import os
import queue
import random
import subprocess
import threading
import time
import webbrowser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import *


REWARD_COMPONENT_NAMES = (
    "reward_a", "reward_b", "reward_c", "reward_d", "reward_e",
    "reward_f", "reward_g", "reward_h", "reward_i", "reward_j",
    "reward_k", "reward_l", "reward_fixed", "reward_finished_bonus",
)

PROJECT_DIR = Path(__file__).resolve().parent
log_dir = os.path.join(os.path.expanduser("~"), LOG_DIR)
BY_MAP_DIR = os.path.join(log_dir, "by_map")
ZSG_DIR = os.path.join(log_dir, "zero_shot")
HEARTBEAT_PATH = os.path.join(log_dir, "heartbeat.txt")


def validate_config() -> None:
    if N_ENVS <= 0:
        raise ValueError("N_ENVS must be positive")
    if PPO_ROLLOUT_STEPS_PER_ENV <= 0:
        raise ValueError("PPO_ROLLOUT_STEPS_PER_ENV must be positive")
    if PPO_MINIBATCH_SIZE <= 1:
        raise ValueError("PPO_MINIBATCH_SIZE must be greater than one")
    if PPO_CHECKPOINT_INTERVAL_EPISODES <= 0:
        raise ValueError("PPO_CHECKPOINT_INTERVAL_EPISODES must be positive")
    if PPO_CHECKPOINT_INTERVAL_UPDATES <= 0:
        raise ValueError("PPO_CHECKPOINT_INTERVAL_UPDATES must be positive")
    if not 0 <= PPO_LOGPROB_WARN_TOL < PPO_LOGPROB_FAIL_TOL:
        raise ValueError(
            "PPO log-prob tolerances must satisfy 0 <= warn < fail"
        )
    if FiLM_USE and not ROBOT_STATE_EMBEDDING:
        raise ValueError(
            "FiLM_USE=True requires ROBOT_STATE_EMBEDDING=True for PPO V(s); "
            "the current action cannot condition a state-value function."
        )


def ensure_runtime_dirs() -> None:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(BY_MAP_DIR, exist_ok=True)
    os.makedirs(ZSG_DIR, exist_ok=True)


def ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w").close()


def write_heartbeat(episode: int) -> None:
    ensure_runtime_dirs()
    tmp = HEARTBEAT_PATH + ".tmp"
    with open(tmp, "w") as f:
        f.write(f"{episode}\n{time.time()}\n")
    os.replace(tmp, HEARTBEAT_PATH)


def map_metric_path(metric_name: str, map_num: int) -> str:
    return os.path.join(BY_MAP_DIR, f"{metric_name}_map_{map_num}.txt")


def zsg_metric_path(map_num: int, robot_num: int) -> str:
    return os.path.join(
        ZSG_DIR, f"zsg_evacuation_100_map_{map_num}_robot_{robot_num}.txt"
    )


def zsg_all_maps_metric_path(robot_num: int) -> str:
    return os.path.join(
        ZSG_DIR, f"zsg_evacuation_100_all_maps_robot_{robot_num}.txt"
    )


def launch_tensorboard(tb_log_dir: str, port: int = 6006):
    """Launch TensorBoard like SAC_FE_RV3; failure does not stop training."""
    try:
        process = subprocess.Popen(
            ["tensorboard", "--logdir", tb_log_dir, "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://localhost:{port}"
        threading.Thread(target=lambda: (time.sleep(2), webbrowser.open(url)), daemon=True).start()
        print(f"TensorBoard launched at {url}")
        return process
    except (FileNotFoundError, OSError) as exc:
        print(f"[Warning] TensorBoard was not launched: {exc}")
        return None


def ego_crop_from_full_map(
    full_map: np.ndarray,
    robot_xy_px: Tuple[int, int],
    ego_size: int,
    pad_value: int = 50,
) -> np.ndarray:
    height, width = full_map.shape
    center_x, center_y = robot_xy_px
    half = ego_size // 2
    x0, x1 = center_x - half, center_x - half + ego_size
    y0, y1 = center_y - half, center_y - half + ego_size
    source_x0, source_x1 = max(0, x0), min(width, x1)
    source_y0, source_y1 = max(0, y0), min(height, y1)
    crop = np.full((ego_size, ego_size), pad_value, dtype=full_map.dtype)
    target_x0, target_y0 = source_x0 - x0, source_y0 - y0
    crop[
        target_y0:target_y0 + source_y1 - source_y0,
        target_x0:target_x0 + source_x1 - source_x0,
    ] = full_map[source_y0:source_y1, source_x0:source_x1]
    return crop


def downsample_full_map(full_map: np.ndarray, target: int) -> np.ndarray:
    tensor = torch.from_numpy(full_map).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(tensor, (target, target))
    return pooled.squeeze(0).squeeze(0).byte().numpy()


def build_ego_global_frames(env_model) -> Tuple[np.ndarray, np.ndarray]:
    full = env_model.return_current_image(MAP_H, MAP_W)
    robot_x, robot_y = env_model.robot.xy
    pixel_x = int(np.clip(robot_x / env_model.width * MAP_W, 0, MAP_W - 1))
    pixel_y = int(np.clip(robot_y / env_model.height * MAP_H, 0, MAP_H - 1))
    ego = ego_crop_from_full_map(
        full, (pixel_x, pixel_y), EGO_MAP_SIZE, pad_value=50
    )
    glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)
    return ego.astype(np.float32) / 255.0, glob.astype(np.float32) / 255.0


class CNNEncoder(nn.Module):
    """Exact convolutional encoder used by the SAC_FE_RV3 actor."""

    def __init__(self, input_shape=(50, 50), in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.out_dim = self._get_conv_out(input_shape, in_channels)

    def _get_conv_out(self, shape, in_channels):
        dummy = torch.zeros(1, in_channels, *shape)
        output = F.silu(self.bn1(self.conv1(dummy)))
        output = F.silu(self.bn2(self.conv2(output)))
        output = F.silu(self.bn3(self.conv3(output)))
        return int(np.prod(output.size()[1:]))

    def forward(self, state):
        state = F.silu(self.bn1(self.conv1(state)))
        state = F.silu(self.bn2(self.conv2(state)))
        state = F.silu(self.bn3(self.conv3(state)))
        return state.view(state.size(0), -1)


class PolicyNetwork(nn.Module):
    """SAC_FE_RV3 actor architecture with PPO action-evaluation helpers."""

    def __init__(
        self,
        ego_shape=(25, 25),
        global_shape=(50, 50),
        robot_dim=3,
        use_robot=True,
    ):
        super().__init__()
        self.log_std_min = PPO_LOG_STD_MIN
        self.log_std_max = PPO_LOG_STD_MAX
        self.use_robot_state = use_robot
        self.ego_enc = CNNEncoder(ego_shape, in_channels=4) if EGO_USE else None
        self.glob_enc = CNNEncoder(global_shape, in_channels=4)

        self.robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(nn.Linear(robot_dim, 32), nn.SiLU())
            self.robot_feat_dim = 32
        else:
            self.robot_fc = None

        fusion_dim = self.glob_enc.out_dim + self.robot_feat_dim
        if EGO_USE:
            fusion_dim += self.ego_enc.out_dim
        self.fc_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU(),
        )
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def backbone(self, ego_state, global_state, robot_state=None):
        features = []
        if EGO_USE:
            if ego_state is None:
                raise ValueError("ego_state is required when EGO_USE=True")
            features.append(self.ego_enc(ego_state))
        features.append(self.glob_enc(global_state))
        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("robot_state is required")
            features.append(self.robot_fc(robot_state))
        return self.fc_backbone(torch.cat(features, dim=1))

    def forward(self, ego_state, global_state, robot_state=None):
        feature = self.backbone(ego_state, global_state, robot_state)
        mean = self.mean_head(feature)
        log_std = torch.clamp(
            self.log_std_head(feature), self.log_std_min, self.log_std_max
        )
        return mean, log_std

    @staticmethod
    def squash(raw_action):
        return 4.0 * torch.sigmoid(raw_action) - 2.0

    @staticmethod
    def _squashed_log_prob(distribution, raw_action):
        sigma = torch.sigmoid(raw_action)
        base_log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        log_jacobian = torch.log(4.0 * sigma * (1.0 - sigma) + 1e-8).sum(dim=-1)
        return base_log_prob - log_jacobian

    @staticmethod
    def _base_log_prob(distribution, raw_action):
        return distribution.log_prob(raw_action).sum(dim=-1)

    def sample_action(self, ego_state, global_state, robot_state=None, temperature=1.0):
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        std = log_std.exp() * temperature
        distribution = torch.distributions.Normal(mean, std)
        raw_action = distribution.sample()
        action = self.squash(raw_action)
        # PPO only uses a likelihood ratio for the same stored raw action.
        # The sigmoid-affine Jacobian is action-only and cancels exactly
        # between old and new policies, so the base Gaussian density is both
        # mathematically equivalent and substantially more stable.
        log_prob = self._base_log_prob(distribution, raw_action)
        return action, log_prob, raw_action

    def evaluate_raw_actions(
        self, ego_state, global_state, robot_state, raw_action
    ):
        mean, log_std = self.forward(ego_state, global_state, robot_state)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        log_prob = self._base_log_prob(distribution, raw_action)
        # The transformed distribution has no convenient analytic entropy.
        # Base Gaussian entropy is the conventional stable PPO proxy.
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy

    def deterministic_action(self, ego_state, global_state, robot_state=None):
        mean, _ = self.forward(ego_state, global_state, robot_state)
        return self.squash(mean)


class ValueNetwork(nn.Module):
    """V(s) counterpart of the SAC Q-network.

    The image encoders, robot embedding, optional FiLM modulation and hidden
    widths are retained.  The current action is deliberately absent because an
    action-conditioned critic is Q(s,a), not the state-value V(s) used by PPO.
    """

    def __init__(
        self,
        ego_shape=(25, 25),
        global_shape=(50, 50),
        robot_dim=3,
        use_robot=True,
    ):
        super().__init__()
        self.use_robot_state = use_robot
        self.ego_enc = CNNEncoder(ego_shape, in_channels=4) if EGO_USE else None
        self.glob_enc = CNNEncoder(global_shape, in_channels=4)
        robot_feat_dim = 0
        if use_robot:
            self.robot_fc = nn.Sequential(nn.Linear(robot_dim, 32), nn.SiLU())
            robot_feat_dim = 32
        else:
            self.robot_fc = None

        self.img_dim = self.glob_enc.out_dim
        if EGO_USE:
            self.img_dim += self.ego_enc.out_dim

        if FiLM_USE:
            if not use_robot:
                raise ValueError("FiLM value critic requires robot state")
            self.film = nn.Sequential(
                nn.Linear(robot_feat_dim, 256), nn.SiLU(),
                nn.Linear(256, 2 * self.img_dim),
            )
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        else:
            self.film = None

        self.fc1 = nn.Linear(self.img_dim + robot_feat_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.value_out = nn.Linear(256, 1)

    def forward(self, ego_state, global_state, robot_state=None):
        image_features = []
        if EGO_USE:
            if ego_state is None:
                raise ValueError("ego_state is required when EGO_USE=True")
            image_features.append(self.ego_enc(ego_state))
        image_features.append(self.glob_enc(global_state))
        image = torch.cat(image_features, dim=1)

        robot = None
        if self.use_robot_state:
            if robot_state is None:
                raise ValueError("robot_state is required")
            robot = self.robot_fc(robot_state)

        if self.film is not None:
            gamma, beta = self.film(robot).chunk(2, dim=1)
            image = (1.0 + gamma) * image + beta

        features = [image]
        if robot is not None:
            features.append(robot)
        output = F.silu(self.fc1(torch.cat(features, dim=1)))
        output = F.silu(self.fc2(output))
        return self.value_out(output).squeeze(-1)


class FrameStack:
    def __init__(self, stack_len=4):
        self.stack_len = stack_len
        self.frames = deque(maxlen=stack_len)

    def reset(self, first_frame):
        self.frames.clear()
        for _ in range(self.stack_len):
            self.frames.append(np.copy(first_frame).astype(np.float32))
        return np.stack(list(self.frames)[::-1], axis=0)

    def append(self, frame):
        self.frames.append(np.copy(frame).astype(np.float32))
        return np.stack(list(self.frames)[::-1], axis=0)

    def peek_with(self, frame):
        temporary = list(self.frames) + [np.copy(frame).astype(np.float32)]
        return np.stack(temporary[-self.stack_len:][::-1], axis=0)


class FrameStack2(FrameStack):
    pass


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


@dataclass
class CollectCommand:
    policy_version: int
    rollout_steps: int
    actor_state: Dict[str, np.ndarray]
    value_state: Dict[str, np.ndarray]


@dataclass
class RolloutBatch:
    worker_id: int
    policy_version: int
    ego_states: Optional[np.ndarray]
    global_states: np.ndarray
    robot_states: np.ndarray
    raw_actions: np.ndarray
    old_log_probs: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_stats: List[EpisodeStatMsg]
    simulator_steps: int
    sampled_actions: int

    def __len__(self):
        return int(self.rewards.shape[0])


@dataclass
class WorkerError:
    worker_id: int
    policy_version: int
    message: str


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE with correct terminal and time-limit bootstrapping.

    A true terminal has no bootstrap.  A time-limit truncation bootstraps from
    V(final_observation), while recursion stops so it cannot enter the reset
    episode that follows.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    terminated = np.asarray(terminated, dtype=np.float32)
    truncated = np.asarray(truncated, dtype=np.float32)
    if not (
        rewards.shape == values.shape == next_values.shape
        == terminated.shape == truncated.shape
    ):
        raise ValueError("All GAE inputs must have identical shapes")

    deltas = rewards + gamma * (1.0 - terminated) * next_values - values
    episode_ends = np.maximum(terminated, truncated)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        gae = (
            deltas[index]
            + gamma * gae_lambda * (1.0 - episode_ends[index]) * gae
        )
        advantages[index] = gae
    return advantages, advantages + values


class PPOAgent:
    def __init__(
        self,
        input_shape=(50, 50),
        device=DEVICE,
        gamma=GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_eps=PPO_CLIP_EPS,
        value_clip_eps=PPO_VALUE_CLIP_EPS,
        lr_actor=LR,
        lr_critic=LR,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=PPO_MINIBATCH_SIZE,
        value_coef=PPO_VALUE_COEF,
        entropy_coef=PPO_ENTROPY_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        target_kl=PPO_TARGET_KL,
    ):
        del input_shape  # retained for FightingModel compatibility
        validate_config()
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.value_clip_eps = float(value_clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl) if target_kl is not None else None

        self.actor = PolicyNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)
        self.value = ValueNetwork(
            ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
            global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
            robot_dim=ROBOT_STATE_DIM,
            use_robot=ROBOT_STATE_EMBEDDING,
        ).to(self.device)
        # Compatibility for existing evaluation scripts which access .policy.
        self.policy = self.actor
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=lr_actor, weight_decay=WD_PI
        )
        self.value_optimizer = optim.AdamW(
            self.value.parameters(), lr=lr_critic, weight_decay=WD_Q
        )

    def select_action(
        self,
        ego_state_np,
        global_state_np,
        robot_state_np=None,
        deterministic=False,
        log_action=False,
    ):
        ego = None
        if EGO_USE:
            ego = torch.as_tensor(
                ego_state_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        glob = torch.as_tensor(
            global_state_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        robot = None
        if ROBOT_STATE_EMBEDDING:
            robot = torch.as_tensor(
                robot_state_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(ego, glob, robot)
            else:
                action, _, _ = self.actor.sample_action(ego, glob, robot)
        action_np = action.cpu().numpy()[0].astype(np.float32)
        if log_action:
            print(action_np)
        return action_np, False

    @staticmethod
    def _states_to_tensors(
        ego_states: Optional[np.ndarray],
        global_states: np.ndarray,
        robot_states: np.ndarray,
        device: torch.device,
    ):
        ego = None
        if EGO_USE:
            ego = torch.as_tensor(ego_states, dtype=torch.float32, device=device) / 255.0
        glob = torch.as_tensor(global_states, dtype=torch.float32, device=device) / 255.0
        robot = None
        if ROBOT_STATE_EMBEDDING:
            robot = torch.as_tensor(robot_states, dtype=torch.float32, device=device)
        return ego, glob, robot

    def update(self, rollouts: Sequence[RolloutBatch]) -> Dict[str, float]:
        if not rollouts:
            raise ValueError("PPO update requires at least one rollout")
        versions = {rollout.policy_version for rollout in rollouts}
        if len(versions) != 1:
            raise ValueError(f"Mixed on-policy versions: {sorted(versions)}")

        ego_parts, glob_parts, robot_parts = [], [], []
        raw_action_parts, old_log_prob_parts = [], []
        old_value_parts, return_parts, advantage_parts = [], [], []
        for rollout in rollouts:
            if EGO_USE:
                ego_parts.append(rollout.ego_states)
            glob_parts.append(rollout.global_states)
            robot_parts.append(rollout.robot_states)
            raw_action_parts.append(rollout.raw_actions)
            old_log_prob_parts.append(rollout.old_log_probs)
            old_value_parts.append(rollout.values)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.next_values,
                rollout.terminated,
                rollout.truncated,
                self.gamma,
                self.gae_lambda,
            )
            advantage_parts.append(advantages)
            return_parts.append(returns)

        ego_np = np.concatenate(ego_parts) if EGO_USE else None
        glob_np = np.concatenate(glob_parts)
        robot_np = np.concatenate(robot_parts)
        ego, glob, robot = self._states_to_tensors(
            ego_np, glob_np, robot_np, self.device
        )
        raw_actions = torch.as_tensor(
            np.concatenate(raw_action_parts), dtype=torch.float32, device=self.device
        )
        old_log_probs = torch.as_tensor(
            np.concatenate(old_log_prob_parts), dtype=torch.float32, device=self.device
        )
        old_values = torch.as_tensor(
            np.concatenate(old_value_parts), dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(
            np.concatenate(return_parts), dtype=torch.float32, device=self.device
        )
        advantages = torch.as_tensor(
            np.concatenate(advantage_parts), dtype=torch.float32, device=self.device
        )
        if PPO_ADVANTAGE_NORMALIZATION and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        # BatchNorm running statistics in the policy stay frozen.  This makes
        # rollout and update log probabilities use exactly the same statistics.
        self.actor.eval()
        with torch.no_grad():
            check_log_probs, _ = self.actor.evaluate_raw_actions(
                ego, glob, robot, raw_actions
            )
            _, rollout_log_std = self.actor.forward(ego, glob, robot)
            preupdate_logratio = check_log_probs - old_log_probs
            max_preupdate_logratio = preupdate_logratio.abs().max().item()
        if max_preupdate_logratio > PPO_LOGPROB_FAIL_TOL:
            raise RuntimeError(
                "PPO old/new log-prob mismatch before update: "
                f"max |log ratio|={max_preupdate_logratio:.6g}, "
                f"fail tolerance={PPO_LOGPROB_FAIL_TOL:.6g}"
            )
        if max_preupdate_logratio > PPO_LOGPROB_WARN_TOL:
            print(
                "[Warning] Small PPO old/new log-prob numerical mismatch: "
                f"max |log ratio|={max_preupdate_logratio:.6g}. "
                "The policy version is identical; continuing."
            )

        sample_count = int(returns.shape[0])
        metrics = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_fraction": [], "actor_grad_norm": [],
            "value_grad_norm": [],
        }
        epochs_completed = 0
        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(sample_count, device=self.device)
            epoch_kls = []
            for start in range(0, sample_count, self.mini_batch_size):
                indices = permutation[start:start + self.mini_batch_size]
                mb_ego = ego[indices] if EGO_USE else None
                mb_glob = glob[indices]
                mb_robot = robot[indices] if ROBOT_STATE_EMBEDDING else None
                mb_raw_actions = raw_actions[indices]
                mb_old_log_probs = old_log_probs[indices]
                mb_old_values = old_values[indices]
                mb_returns = returns[indices]
                mb_advantages = advantages[indices]

                new_log_probs, entropy = self.actor.evaluate_raw_actions(
                    mb_ego, mb_glob, mb_robot, mb_raw_actions
                )
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * mb_advantages
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                actor_loss = policy_loss - self.entropy_coef * entropy.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.value.train()
                predicted_values = self.value(mb_ego, mb_glob, mb_robot)
                value_unclipped_loss = (predicted_values - mb_returns).pow(2)
                clipped_values = mb_old_values + torch.clamp(
                    predicted_values - mb_old_values,
                    -self.value_clip_eps,
                    self.value_clip_eps,
                )
                value_clipped_loss = (clipped_values - mb_returns).pow(2)
                value_loss = 0.5 * torch.maximum(
                    value_unclipped_loss, value_clipped_loss
                ).mean()
                self.value_optimizer.zero_grad(set_to_none=True)
                (self.value_coef * value_loss).backward()
                value_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_eps
                    ).float().mean()
                epoch_kls.append(float(approx_kl.item()))
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy.mean().item()))
                metrics["approx_kl"].append(float(approx_kl.item()))
                metrics["clip_fraction"].append(float(clip_fraction.item()))
                metrics["actor_grad_norm"].append(float(actor_grad_norm))
                metrics["value_grad_norm"].append(float(value_grad_norm))
            epochs_completed += 1
            if (
                self.target_kl is not None
                and epoch_kls
                and np.mean(epoch_kls) > self.target_kl
            ):
                break

        returns_np = returns.detach().cpu().numpy()
        old_values_np = old_values.detach().cpu().numpy()
        return_variance = float(np.var(returns_np))
        explained_variance = (
            1.0 - float(np.var(returns_np - old_values_np)) / return_variance
            if return_variance > 1e-8 else float("nan")
        )
        result = {
            name: float(np.mean(values)) if values else float("nan")
            for name, values in metrics.items()
        }
        result.update({
            "explained_variance": explained_variance,
            "epochs_completed": float(epochs_completed),
            "max_preupdate_logratio": float(max_preupdate_logratio),
            "rollout_log_std_min": float(rollout_log_std.min().item()),
            "rollout_log_std_mean": float(rollout_log_std.mean().item()),
            "rollout_log_std_max": float(rollout_log_std.max().item()),
            "samples": float(sample_count),
        })
        return result

    def checkpoint(self, episode: int, policy_version: int, env_steps: int, policy_steps: int):
        return {
            "algorithm": "PPO",
            "episode": int(episode),
            "policy_version": int(policy_version),
            "env_steps": int(env_steps),
            "policy_steps": int(policy_steps),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def save_model(
        self,
        filepath: str,
        episode: int = 0,
        policy_version: int = 0,
        env_steps: int = 0,
        policy_steps: int = 0,
    ):
        torch.save(
            self.checkpoint(episode, policy_version, env_steps, policy_steps), filepath
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> Dict[str, int]:
        path = filepath
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(log_dir, path)
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("algorithm", "PPO") != "PPO":
            raise ValueError(f"Not a PPO checkpoint: {path}")
        self.actor.load_state_dict(checkpoint["actor"])
        self.value.load_state_dict(checkpoint["value"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "value_optimizer" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        return {
            "episode": int(checkpoint.get("episode", 0)),
            "policy_version": int(checkpoint.get("policy_version", 0)),
            "env_steps": int(checkpoint.get("env_steps", 0)),
            "policy_steps": int(checkpoint.get("policy_steps", 0)),
        }


def _float_stack_to_uint8(state: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(state, 0.0, 1.0) * 255.0).astype(np.uint8)


class PPOEnvRunner:
    """Persistent action-repeat environment owned by one rollout worker."""

    def __init__(self, worker_id: int, seed: int):
        import model

        self.model_module = model
        self.worker_id = worker_id
        self.episode_idx = 0
        self.simulator_steps_total = 0
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self._reset_episode()

    def _reset_episode(self):
        while True:
            try:
                number_agents = (
                    CROWD_NUMBER_MIN if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX
                    else random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
                )
                self.env = self.model_module.FightingModel(
                    number_agents, MAP_W, MAP_H, model_num=-1, robot="Q"
                )
                break
            except Exception as exc:
                print(f"[Worker {self.worker_id}] env create error: {exc}; retrying")
        ego_frame, global_frame = build_ego_global_frames(self.env)
        self.ego_stack = FrameStack2(4)
        self.global_stack = FrameStack2(4)
        self.ego_state = self.ego_stack.reset(ego_frame)
        self.global_state = self.global_stack.reset(global_frame)
        self.robot_state = np.asarray(
            self.env.return_current_robot_state(), dtype=np.float32
        )
        self.sim_step = 0
        self.total_reward = 0.0
        self.reward_components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        self.evacuation_time_80 = MAX_STEPS
        self.evacuation_time_100 = MAX_STEPS

    def _episode_stat(self, abnormal=0):
        try:
            lifetime = float(self.env.calculate_all_agents_life_time())
        except Exception:
            lifetime = 0.0
        stat = EpisodeStatMsg(
            worker_id=self.worker_id,
            episode_idx=self.episode_idx,
            total_reward=float(self.total_reward),
            evac_time_80=int(self.evacuation_time_80),
            evac_time_100=int(self.evacuation_time_100),
            total_lifetime=lifetime,
            map_num=int(self.env.map_num),
            abnormal=int(abnormal),
            reward_components=dict(self.reward_components),
        )
        self.episode_idx += 1
        return stat

    def _reward_at_boundary(self, step_index: int, terminated: bool):
        values = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
        if REWARD_A:
            values["reward_a"] = self.env.reward_based_alived() * REWARD_A
        if REWARD_B:
            values["reward_b"] = self.env.reward_based_all_agents_danger() * REWARD_B
        if REWARD_C:
            values["reward_c"] = self.env.reward_based_gain() * REWARD_C
        if REWARD_D:
            values["reward_d"] = self.env.reward_penalty() * REWARD_D
        if REWARD_E:
            values["reward_e"] = self.env.reward_based_evacuated_with_robot() * REWARD_E
        if REWARD_F:
            values["reward_f"] = self.env.reward_based_distance_from_near_agents() * REWARD_F
        if REWARD_G:
            values["reward_g"] = self.env.reward_based_distance_from_near_agent_gain() * REWARD_G
        if REWARD_H:
            values["reward_h"] = self.env.reward_based_gain_with_time_bonus() * REWARD_H
        if REWARD_I:
            values["reward_i"] = self.env.reward_based_alived_root() * REWARD_I
        if REWARD_J:
            values["reward_j"] = self.env.reward_based_all_agents_danger_root() * REWARD_J
        if REWARD_K:
            # SAC_FE_RV3 evaluates collision reward on the emission substep.
            values["reward_k"] = self.env.reward_penalty_collision() * REWARD_K
        if REWARD_L:
            values["reward_l"] = self.env.reward_based_farthest_agent_distance() * REWARD_L
        values["reward_fixed"] = REWARD_FIXED
        if terminated:
            values["reward_finished_bonus"] = FINISHED_BONUS * (
                1.0 - step_index / MAX_STEPS
            )
        reward = float(sum(values.values()))
        if reward < -1e3:
            raise RuntimeError(f"Reward collapsed: {reward}")
        return reward, values

    def collect_transition(self, actor, value):
        """Return one SAC-compatible ACTION_SCALE transition and any episode stats."""
        episode_stats = []
        sampled_actions = 0
        simulator_steps = 0
        while True:
            ego_tensor = None
            if EGO_USE:
                ego_tensor = torch.from_numpy(self.ego_state).unsqueeze(0).float()
            global_tensor = torch.from_numpy(self.global_state).unsqueeze(0).float()
            robot_tensor = None
            if ROBOT_STATE_EMBEDDING:
                robot_tensor = torch.from_numpy(self.robot_state).unsqueeze(0).float()
            with torch.no_grad():
                action, log_prob, raw_action = actor.sample_action(
                    ego_tensor, global_tensor, robot_tensor
                )
                state_value = value(ego_tensor, global_tensor, robot_tensor)
            sampled_actions += 1
            action_np = action.numpy()[0].astype(np.float32)
            raw_action_np = raw_action.numpy()[0].astype(np.float32)
            self.env.robot.receive_action(action_np.tolist())

            transition_ego = (
                _float_stack_to_uint8(self.ego_state) if EGO_USE else None
            )
            transition_global = _float_stack_to_uint8(self.global_state)
            transition_robot = np.copy(self.robot_state)

            emitted = False
            reward = 0.0
            terminated = truncated = False
            next_ego_state = self.ego_state
            next_global_state = self.global_state
            next_robot_state = self.robot_state
            # One sampled command is held for exactly ACTION_SCALE simulator
            # steps, matching the action-repeat behavior in SAC_FE_RV3.
            for _ in range(ACTION_SCALE):
                step_index = self.sim_step
                self.env.step()
                self.sim_step += 1
                self.simulator_steps_total += 1
                simulator_steps += 1
                terminated = bool(self.env.robot.is_game_finished)
                truncated = bool(self.sim_step >= MAX_STEPS and not terminated)

                next_robot_state = np.asarray(
                    self.env.return_current_robot_state(), dtype=np.float32
                )
                next_ego_frame, next_global_frame = build_ego_global_frames(self.env)
                next_ego_state = self.ego_stack.peek_with(next_ego_frame)
                next_global_state = self.global_stack.peek_with(next_global_frame)

                if (
                    self.env.alived_agents() < self.env.total_agents * 0.2
                    and self.evacuation_time_80 == MAX_STEPS
                ):
                    self.evacuation_time_80 = step_index
                if (
                    self.env.alived_agents() < 1
                    and self.evacuation_time_100 == MAX_STEPS
                ):
                    self.evacuation_time_100 = step_index

                # Preserve the current SAC_FE_RV3 emission cadence exactly,
                # including its skipped first action-repeat block.
                emitted = bool(
                    (step_index % ACTION_SCALE == ACTION_SCALE - 1 and step_index > ACTION_SCALE)
                    or (terminated and step_index > ACTION_SCALE)
                )
                if emitted:
                    reward, components = self._reward_at_boundary(
                        step_index, terminated
                    )
                    self.total_reward += reward
                    for name, component in components.items():
                        self.reward_components[name] += float(component)
                if emitted or terminated or truncated:
                    break

            if emitted:
                next_ego_tensor = None
                if EGO_USE:
                    next_ego_tensor = torch.from_numpy(next_ego_state).unsqueeze(0).float()
                next_global_tensor = torch.from_numpy(next_global_state).unsqueeze(0).float()
                next_robot_tensor = None
                if ROBOT_STATE_EMBEDDING:
                    next_robot_tensor = torch.from_numpy(next_robot_state).unsqueeze(0).float()
                with torch.no_grad():
                    bootstrap_value = (
                        0.0 if terminated else float(
                            value(
                                next_ego_tensor, next_global_tensor, next_robot_tensor
                            ).item()
                        )
                    )

            if terminated or truncated:
                episode_stats.append(self._episode_stat())
                self._reset_episode()
            else:
                self.ego_state = self.ego_stack.append(next_ego_frame)
                self.global_state = self.global_stack.append(next_global_frame)
                self.robot_state = next_robot_state

            if emitted:
                return {
                    "ego": transition_ego,
                    "glob": transition_global,
                    "robot": transition_robot,
                    "raw_action": raw_action_np,
                    "log_prob": float(log_prob.item()),
                    "value": float(state_value.item()),
                    "next_value": float(bootstrap_value),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "episode_stats": episode_stats,
                    "simulator_steps": simulator_steps,
                    "sampled_actions": sampled_actions,
                }


def worker_process(
    worker_id: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
):
    torch.set_num_threads(1)
    runner = PPOEnvRunner(worker_id, seed)
    actor = PolicyNetwork(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).cpu()
    value = ValueNetwork(
        ego_shape=(EGO_MAP_SIZE, EGO_MAP_SIZE),
        global_shape=(DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE),
        robot_dim=ROBOT_STATE_DIM,
        use_robot=ROBOT_STATE_EMBEDDING,
    ).cpu()

    while True:
        command = command_queue.get()
        if command is None:
            return
        try:
            actor.load_state_dict({
                name: torch.from_numpy(array)
                for name, array in command.actor_state.items()
            })
            value.load_state_dict({
                name: torch.from_numpy(array)
                for name, array in command.value_state.items()
            })
            actor.eval()
            value.eval()
            storage = {
                "ego": [], "glob": [], "robot": [], "raw_action": [],
                "log_prob": [], "value": [], "next_value": [], "reward": [],
                "terminated": [], "truncated": [],
            }
            episode_stats = []
            simulator_steps = sampled_actions = 0
            for _ in range(command.rollout_steps):
                transition = runner.collect_transition(actor, value)
                for key in storage:
                    if key == "ego" and not EGO_USE:
                        continue
                    storage[key].append(transition[key])
                episode_stats.extend(transition["episode_stats"])
                simulator_steps += transition["simulator_steps"]
                sampled_actions += transition["sampled_actions"]

            result_queue.put(RolloutBatch(
                worker_id=worker_id,
                policy_version=command.policy_version,
                ego_states=(np.asarray(storage["ego"], dtype=np.uint8) if EGO_USE else None),
                global_states=np.asarray(storage["glob"], dtype=np.uint8),
                robot_states=np.asarray(storage["robot"], dtype=np.float32),
                raw_actions=np.asarray(storage["raw_action"], dtype=np.float32),
                old_log_probs=np.asarray(storage["log_prob"], dtype=np.float32),
                values=np.asarray(storage["value"], dtype=np.float32),
                next_values=np.asarray(storage["next_value"], dtype=np.float32),
                rewards=np.asarray(storage["reward"], dtype=np.float32),
                terminated=np.asarray(storage["terminated"], dtype=np.float32),
                truncated=np.asarray(storage["truncated"], dtype=np.float32),
                episode_stats=episode_stats,
                simulator_steps=simulator_steps,
                sampled_actions=sampled_actions,
            ))
        except Exception as exc:
            import traceback
            result_queue.put(WorkerError(
                worker_id=worker_id,
                policy_version=getattr(command, "policy_version", -1),
                message=f"{exc}\n{traceback.format_exc()}",
            ))


def _numpy_state_dict(module: nn.Module):
    # Sending torch storages through multiprocessing.Queue starts PyTorch's
    # resource-sharer socket.  Plain NumPy snapshots are portable and make the
    # immutability of one policy version explicit.
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in module.state_dict().items()
    }


def start_workers(ctx, number_workers: int, result_queue):
    workers, command_queues = [], []
    for worker_id in range(number_workers):
        command_queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=worker_process,
            args=(
                worker_id,
                command_queue,
                result_queue,
                PPO_BASE_SEED + worker_id,
            ),
            daemon=True,
        )
        process.start()
        workers.append(process)
        command_queues.append(command_queue)
        print(f"[Main] PPO worker {worker_id} started, pid={process.pid}")
    return workers, command_queues


def stop_workers(workers, command_queues):
    for command_queue in command_queues:
        try:
            command_queue.put_nowait(None)
        except Exception:
            pass
    for process in workers:
        process.join(timeout=3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)


def collect_synchronous_rollouts(
    agent: PPOAgent,
    policy_version: int,
    workers,
    command_queues,
    result_queue,
):
    command = CollectCommand(
        policy_version=policy_version,
        rollout_steps=PPO_ROLLOUT_STEPS_PER_ENV,
        actor_state=_numpy_state_dict(agent.actor),
        value_state=_numpy_state_dict(agent.value),
    )
    for command_queue in command_queues:
        command_queue.put(command)

    by_worker = {}
    while len(by_worker) < len(workers):
        try:
            result = result_queue.get(timeout=60)
        except queue.Empty:
            dead = [index for index, process in enumerate(workers) if not process.is_alive()]
            if dead:
                raise RuntimeError(f"PPO rollout workers died: {dead}")
            write_heartbeat(-1)
            continue
        if isinstance(result, WorkerError):
            raise RuntimeError(
                f"Worker {result.worker_id} failed at policy version "
                f"{result.policy_version}:\n{result.message}"
            )
        if result.policy_version != policy_version:
            raise RuntimeError(
                f"Stale rollout from worker {result.worker_id}: "
                f"expected {policy_version}, got {result.policy_version}"
            )
        if result.worker_id in by_worker:
            raise RuntimeError(f"Duplicate rollout from worker {result.worker_id}")
        if len(result) != PPO_ROLLOUT_STEPS_PER_ENV:
            raise RuntimeError(
                f"Worker {result.worker_id} returned {len(result)} transitions"
            )
        by_worker[result.worker_id] = result
    return [by_worker[index] for index in sorted(by_worker)]


def evaluate_zero_shot_once(
    agent,
    map_num: int,
    robot_num: int = 1,
    deterministic: bool = True,
    seed: int = 0,
):
    if robot_num != 1:
        raise ValueError("PPO_FE_RV supports one robot; ZSG_ROBOT_NUM must contain only 1")
    import model

    np.random.seed(seed)
    random.seed(seed)
    number_agents = (
        CROWD_NUMBER_MIN if CROWD_NUMBER_MIN == CROWD_NUMBER_MAX
        else random.randint(CROWD_NUMBER_MIN, CROWD_NUMBER_MAX)
    )
    env = model.FightingModel(
        number_agents, MAP_W, MAP_H, model_num=map_num, robot="Q"
    )
    ego_frame, global_frame = build_ego_global_frames(env)
    ego_stack, global_stack = FrameStack2(4), FrameStack2(4)
    ego_state = ego_stack.reset(ego_frame)
    global_state = global_stack.reset(global_frame)
    evacuation_time_100 = MAX_STEPS
    for step in range(MAX_STEPS):
        if env.alived_agents() < 1:
            evacuation_time_100 = step
            break
        if env.robot.is_game_finished or step == MAX_STEPS - 1:
            break
        if step % ACTION_SCALE == 0:
            if step > 0:
                ego_frame, global_frame = build_ego_global_frames(env)
                ego_state = ego_stack.append(ego_frame)
                global_state = global_stack.append(global_frame)
            robot_state = np.asarray(env.return_current_robot_state(), dtype=np.float32)
            action, _ = agent.select_action(
                ego_state, global_state, robot_state,
                deterministic=deterministic, log_action=False,
            )
            env.robot.receive_action(action.tolist())
        env.step()
    return int(evacuation_time_100)


def run_zero_shot_evaluation(agent, episode: int, writer: SummaryWriter):
    print(f"[ZeroShot] Start PPO evaluation at episode {episode}")
    actor_was_training = agent.actor.training
    numpy_state, python_state = np.random.get_state(), random.getstate()
    results = {}
    try:
        agent.actor.eval()
        for robot_num in ZSG_ROBOT_NUM:
            map_averages = []
            for map_num in ZSG_MAP:
                times = []
                for iteration in range(ZSG_ITERATION):
                    seed = episode * 100000 + map_num * 100 + robot_num * 10 + iteration
                    times.append(evaluate_zero_shot_once(
                        agent, map_num, robot_num, deterministic=True, seed=seed
                    ))
                average = float(np.mean(times))
                map_averages.append(average)
                results[(map_num, robot_num)] = average
                path = zsg_metric_path(map_num, robot_num)
                ensure_file(path)
                with open(path, "a") as f:
                    f.write(f"{episode}\t{average:.6f}\n")
                writer.add_scalar(
                    f"ZeroShot/Evacuation100/map_{map_num}/robot_{robot_num}",
                    average, episode,
                )
            all_maps_average = float(np.mean(map_averages))
            results[("all_maps", robot_num)] = all_maps_average
            path = zsg_all_maps_metric_path(robot_num)
            ensure_file(path)
            with open(path, "a") as f:
                f.write(f"{episode}\t{all_maps_average:.6f}\n")
            writer.add_scalar(
                f"ZeroShot/Evacuation100/all_maps/robot_{robot_num}",
                all_maps_average, episode,
            )
        writer.flush()
    finally:
        agent.actor.train(actor_was_training)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return results


def _append_value(path: str, value) -> None:
    ensure_file(path)
    with open(path, "a") as f:
        f.write(f"{value}\n")


def log_episode_stat(stat, files, writer, global_episode):
    if stat.abnormal:
        return
    _append_value(map_metric_path("reward", stat.map_num), stat.total_reward)
    _append_value(map_metric_path("evacuation_100", stat.map_num), stat.evac_time_100)
    _append_value(files["reward"], stat.total_reward)
    _append_value(files["evac80"], stat.evac_time_80)
    _append_value(files["evac100"], stat.evac_time_100)
    _append_value(files["lifetime"], stat.total_lifetime)
    writer.add_scalar("Episode/TotalReward", stat.total_reward, global_episode)
    writer.add_scalar("Episode/Evacuation80", stat.evac_time_80, global_episode)
    writer.add_scalar("Episode/Evacuation100", stat.evac_time_100, global_episode)
    writer.add_scalar("Episode/TotalLifetime", stat.total_lifetime, global_episode)
    for name in REWARD_COMPONENT_NAMES:
        value = stat.reward_components.get(name, 0.0)
        _append_value(files[name], value)
        writer.add_scalar(f"RewardComponents/{name}", value, global_episode)


def main():
    validate_config()
    ensure_runtime_dirs()
    np.random.seed(PPO_BASE_SEED)
    random.seed(PPO_BASE_SEED)
    torch.manual_seed(PPO_BASE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PPO_BASE_SEED)

    tb_log_dir = os.path.join(log_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_log_dir)
    launch_tensorboard(tb_log_dir, PORT_NUM)
    files = {
        "reward": os.path.join(log_dir, "total_reward.txt"),
        "evac80": os.path.join(log_dir, "evacuation_80.txt"),
        "evac100": os.path.join(log_dir, "evacuation_100.txt"),
        "lifetime": os.path.join(log_dir, "total_lifetime.txt"),
    }
    files.update({
        name: os.path.join(log_dir, f"{name}.txt")
        for name in REWARD_COMPONENT_NAMES
    })
    for path in files.values():
        ensure_file(path)

    agent = PPOAgent(device=DEVICE)
    global_episode = policy_version = env_steps = policy_steps = 0
    latest_path = os.path.join(log_dir, "ppo_latest.pth")
    if PPO_MODEL_LOAD == 2:
        named_path = os.path.join(log_dir, PPO_NAMED_CHECKPOINT)
        if os.path.exists(named_path):
            counters = agent.load_model(named_path)
            global_episode = counters["episode"]
            policy_version = counters["policy_version"]
            env_steps = counters["env_steps"]
            policy_steps = counters["policy_steps"]
    elif PPO_MODEL_LOAD == 3 and os.path.exists(latest_path):
        counters = agent.load_model(latest_path)
        global_episode = counters["episode"]
        policy_version = counters["policy_version"]
        env_steps = counters["env_steps"]
        policy_steps = counters["policy_steps"]
        print(f"[Main] Loaded latest PPO checkpoint at episode {global_episode}")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=2 * N_ENVS)
    workers, command_queues = start_workers(ctx, N_ENVS, result_queue)
    next_checkpoint_episode = (
        (global_episode // PPO_CHECKPOINT_INTERVAL_EPISODES) + 1
    ) * PPO_CHECKPOINT_INTERVAL_EPISODES
    next_checkpoint_version = (
        (policy_version // PPO_CHECKPOINT_INTERVAL_UPDATES) + 1
    ) * PPO_CHECKPOINT_INTERVAL_UPDATES
    next_zsg_episode = (
        (global_episode // ZSG_CYCLE_EPISODE) + 1
    ) * ZSG_CYCLE_EPISODE if ZSG_CYCLE_EPISODE > 0 else None
    write_heartbeat(global_episode)

    try:
        while global_episode < PPO_MAX_EPISODES:
            collection_start = time.time()
            rollouts = collect_synchronous_rollouts(
                agent, policy_version, workers, command_queues, result_queue
            )
            collection_seconds = time.time() - collection_start
            env_steps += sum(batch.simulator_steps for batch in rollouts)
            policy_steps += sum(batch.sampled_actions for batch in rollouts)

            for batch in rollouts:
                for stat in batch.episode_stats:
                    global_episode += 1
                    log_episode_stat(stat, files, writer, global_episode)
                    print(
                        f"[Main] Episode {global_episode} worker={stat.worker_id} "
                        f"reward={stat.total_reward:.4f} evac100={stat.evac_time_100}"
                    )

            update_start = time.time()
            metrics = agent.update(rollouts)
            update_seconds = time.time() - update_start
            policy_version += 1
            for name, value in metrics.items():
                writer.add_scalar(f"PPO/{name}", value, env_steps)
            writer.add_scalar("Timing/CollectionSeconds", collection_seconds, env_steps)
            writer.add_scalar("Timing/UpdateSeconds", update_seconds, env_steps)
            writer.add_scalar("Counters/PolicySteps", policy_steps, env_steps)
            writer.add_scalar("Counters/PolicyVersion", policy_version, env_steps)
            writer.flush()
            print(
                f"[PPO] version={policy_version} env_steps={env_steps} "
                f"samples={int(metrics['samples'])} kl={metrics['approx_kl']:.6f} "
                f"clipfrac={metrics['clip_fraction']:.4f} "
                f"epochs={int(metrics['epochs_completed'])}"
            )
            write_heartbeat(global_episode)

            if policy_version >= next_checkpoint_version:
                agent.save_model(
                    latest_path, global_episode, policy_version,
                    env_steps, policy_steps,
                )
                while next_checkpoint_version <= policy_version:
                    next_checkpoint_version += PPO_CHECKPOINT_INTERVAL_UPDATES

            if global_episode >= next_checkpoint_episode:
                checkpoint_path = os.path.join(
                    log_dir, f"ppo_checkpoint_ep_{global_episode}.pth"
                )
                agent.save_model(
                    checkpoint_path, global_episode, policy_version,
                    env_steps, policy_steps,
                )
                # Also refresh latest when an episode-indexed archive is made.
                agent.save_model(
                    latest_path, global_episode, policy_version,
                    env_steps, policy_steps,
                )
                while next_checkpoint_episode <= global_episode:
                    next_checkpoint_episode += PPO_CHECKPOINT_INTERVAL_EPISODES

            if next_zsg_episode is not None and global_episode >= next_zsg_episode:
                run_zero_shot_evaluation(agent, global_episode, writer)
                while next_zsg_episode <= global_episode:
                    next_zsg_episode += ZSG_CYCLE_EPISODE
    finally:
        stop_workers(workers, command_queues)
        writer.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
