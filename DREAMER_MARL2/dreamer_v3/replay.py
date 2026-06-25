from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import torch


@dataclass
class DreamerStep:
    joint_ego: np.ndarray
    global_state: np.ndarray
    joint_robot: np.ndarray
    action: np.ndarray
    reward: float
    is_first: bool
    is_terminal: bool
    joint_mask: np.ndarray
    delta_t: float


class DreamerSequenceReplay:
    """Episode replay that samples contiguous sequences for RSSM training."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        device: Union[torch.device, str],
        state_dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.device = torch.device(device)
        self.state_dtype = state_dtype
        self.episodes: deque[list[DreamerStep]] = deque()
        self.num_steps = 0

    def __len__(self) -> int:
        return self.num_steps

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    def add_episode(self, steps: list[DreamerStep]) -> None:
        if not steps:
            return
        self.episodes.append(steps)
        self.num_steps += len(steps)
        while self.num_steps > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.num_steps -= len(removed)

    def add_transition_msg(self, msg: Any, force_terminal: bool = False) -> None:
        step = self.step_from_transition_msg(msg, force_terminal=force_terminal)
        self.add_episode([step])

    def step_from_transition_msg(self, msg: Any, force_terminal: bool = False) -> DreamerStep:
        return DreamerStep(
            joint_ego=self._to_uint8(msg.joint_ego_state),
            global_state=self._to_uint8(msg.global_state),
            joint_robot=np.asarray(msg.joint_robot_state, dtype=np.float32),
            action=np.asarray(msg.joint_action, dtype=np.float32),
            reward=float(msg.reward),
            is_first=False,
            is_terminal=bool(msg.done or force_terminal),
            joint_mask=np.asarray(msg.joint_mask, dtype=np.float32),
            delta_t=max(float(getattr(msg, "delta_t", 1.0)), 1.0),
        )

    def can_sample(self, batch_size: int) -> bool:
        usable = [
            ep for ep in self.episodes
            if len(ep) >= self.sequence_length
        ]
        return len(usable) > 0 and self.num_steps >= batch_size * self.sequence_length

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        valid = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        if not valid:
            raise ValueError("No replay episode is long enough for sequence sampling.")

        chunks = []
        for _ in range(batch_size):
            ep = valid[np.random.randint(0, len(valid))]
            start = np.random.randint(0, len(ep) - self.sequence_length + 1)
            chunks.append(ep[start:start + self.sequence_length])

        return self._collate(chunks)

    def save(self, filepath: str) -> None:
        arrays = {
            "episodes": np.array(list(self.episodes), dtype=object),
            "capacity": np.array(self.capacity, dtype=np.int64),
            "sequence_length": np.array(self.sequence_length, dtype=np.int64),
            "num_steps": np.array(self.num_steps, dtype=np.int64),
        }
        np.savez_compressed(filepath, **arrays)

    def load(self, filepath: str) -> None:
        data = np.load(filepath, allow_pickle=True)
        self.capacity = int(data["capacity"])
        self.sequence_length = int(data["sequence_length"])
        self.episodes = deque(data["episodes"].tolist())
        self.num_steps = int(data["num_steps"])

    def _collate(self, chunks: list[list[DreamerStep]]) -> dict[str, torch.Tensor]:
        def stack(name: str):
            return np.stack([
                np.stack([getattr(step, name) for step in seq], axis=0)
                for seq in chunks
            ], axis=0)

        joint_ego = stack("joint_ego").astype(np.float32) / 255.0
        global_state = stack("global_state").astype(np.float32) / 255.0
        joint_robot = stack("joint_robot").astype(np.float32)
        action = stack("action").astype(np.float32)
        joint_mask = stack("joint_mask").astype(np.float32)

        reward = np.asarray(
            [[step.reward for step in seq] for seq in chunks],
            dtype=np.float32,
        )
        is_first = np.asarray(
            [[step.is_first for step in seq] for seq in chunks],
            dtype=np.float32,
        )
        is_terminal = np.asarray(
            [[step.is_terminal for step in seq] for seq in chunks],
            dtype=np.float32,
        )
        delta_t = np.asarray(
            [[step.delta_t for step in seq] for seq in chunks],
            dtype=np.float32,
        )

        batch = {
            "joint_ego": joint_ego,
            "global_state": global_state,
            "joint_robot": joint_robot,
            "action": action,
            "reward": reward,
            "is_first": is_first,
            "is_terminal": is_terminal,
            "continue": 1.0 - is_terminal,
            "joint_mask": joint_mask,
            "delta_t": delta_t,
        }
        return {
            key: torch.from_numpy(value).to(self.device)
            for key, value in batch.items()
        }

    def _to_uint8(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype == self.state_dtype:
            return arr
        return np.clip(arr * 255.0, 0, 255).astype(self.state_dtype)
