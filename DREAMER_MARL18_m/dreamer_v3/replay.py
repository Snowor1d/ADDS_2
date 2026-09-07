from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import pickle
import time
from typing import Any, Union
import uuid
import zipfile

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


@dataclass(frozen=True)
class DreamerReplaySnapshot:
    """Shallow, stable view of completed episodes for asynchronous saving."""

    episodes: tuple[list[DreamerStep], ...]
    capacity: int
    sequence_length: int
    context_length: int
    num_steps: int


class DreamerSequenceReplay:
    """Episode replay that samples contiguous sequences for RSSM training."""

    def __init__(
        self,
        capacity: int,
        sequence_length: int,
        device: Union[torch.device, str],
        context_length: int = 0,
        state_dtype: np.dtype = np.uint8,
    ) -> None:
        self.capacity = int(capacity)
        self.sequence_length = int(sequence_length)
        self.context_length = int(context_length)
        self.device = torch.device(device)
        self.state_dtype = state_dtype
        self.episodes: deque[list[DreamerStep]] = deque()
        self.current_episode: list[DreamerStep] = []
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
        is_first = len(self.current_episode) == 0
        step = self.step_from_transition_msg(
            msg,
            force_terminal=force_terminal,
            is_first=is_first,
        )
        self.current_episode.append(step)
        if step.is_terminal:
            self.add_episode(self.current_episode)
            self.current_episode = []

    def step_from_transition_msg(
        self,
        msg: Any,
        force_terminal: bool = False,
        *,
        use_next_obs: bool = False,
        action: np.ndarray | None = None,
        reward: float | None = None,
        is_first: bool = False,
        is_terminal: bool | None = None,
        delta_t: float | None = None,
    ) -> DreamerStep:
        if use_next_obs:
            joint_ego = msg.next_joint_ego_state
            global_state = msg.next_global_state
            joint_robot = msg.next_joint_robot_state
            joint_mask = msg.next_joint_mask
        else:
            joint_ego = msg.joint_ego_state
            global_state = msg.global_state
            joint_robot = msg.joint_robot_state
            joint_mask = msg.joint_mask
        terminal = bool(msg.done or force_terminal) if is_terminal is None else bool(is_terminal)
        return DreamerStep(
            joint_ego=self._to_uint8(joint_ego),
            global_state=self._to_uint8(global_state),
            joint_robot=np.asarray(joint_robot, dtype=np.float32),
            action=np.asarray(msg.joint_action if action is None else action, dtype=np.float32),
            reward=float(msg.reward if reward is None else reward),
            is_first=bool(is_first),
            is_terminal=terminal,
            joint_mask=np.asarray(joint_mask, dtype=np.float32),
            delta_t=max(float(getattr(msg, "delta_t", 1.0) if delta_t is None else delta_t), 1.0),
        )

    def can_sample(self, batch_size: int) -> bool:
        usable = [
            ep for ep in self.episodes
            if len(ep) >= self.sequence_length + self.context_length
        ]
        return len(usable) > 0 and self.num_steps >= batch_size * self.sequence_length

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        cpu_batch = self.sample_cpu(
            batch_size,
            pin_memory=self.device.type == "cuda" and torch.cuda.is_available(),
        )
        return self.to_device(cpu_batch, non_blocking=self.device.type == "cuda")

    def prepare_sample(self, batch_size: int) -> list[list[DreamerStep]]:
        """Select immutable step references before an optional collate thread runs."""
        sample_length = self.sequence_length + self.context_length
        valid = [ep for ep in self.episodes if len(ep) >= sample_length]
        if not valid:
            raise ValueError("No replay episode is long enough for sequence sampling.")

        chunks = []
        for _ in range(batch_size):
            ep = valid[np.random.randint(0, len(valid))]
            start = np.random.randint(0, len(ep) - sample_length + 1)
            chunks.append(ep[start:start + sample_length])
        return chunks

    def sample_cpu(
        self,
        batch_size: int,
        *,
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.collate(self.prepare_sample(batch_size), pin_memory=pin_memory)

    def save(self, filepath: str) -> None:
        self.save_snapshot(self.snapshot(), filepath)

    def snapshot(self) -> DreamerReplaySnapshot:
        """Capture completed episode references without copying transition arrays.

        Completed episode lists and their DreamerStep objects are never mutated by
        this replay implementation. Converting the live deque to a tuple makes it
        safe for a background writer even when capacity eviction continues.
        """
        return DreamerReplaySnapshot(
            episodes=tuple(self.episodes),
            capacity=self.capacity,
            sequence_length=self.sequence_length,
            context_length=self.context_length,
            num_steps=self.num_steps,
        )

    @staticmethod
    def save_snapshot(snapshot: DreamerReplaySnapshot, filepath: str) -> None:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        arrays = {
            "episodes": np.array(snapshot.episodes, dtype=object),
            "capacity": np.array(snapshot.capacity, dtype=np.int64),
            "sequence_length": np.array(snapshot.sequence_length, dtype=np.int64),
            "context_length": np.array(snapshot.context_length, dtype=np.int64),
            "num_steps": np.array(snapshot.num_steps, dtype=np.int64),
        }
        tmp_path = f"{filepath}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **arrays)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def load(self, filepath: str) -> bool:
        try:
            data = np.load(filepath, allow_pickle=True)
            self.capacity = int(data["capacity"])
            self.sequence_length = int(data["sequence_length"])
            self.context_length = int(data["context_length"]) if "context_length" in data else self.context_length
            self.episodes = deque(data["episodes"].tolist())
            self.num_steps = int(data["num_steps"])
        except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
            bad_path = f"{filepath}.bad.{int(time.time())}"
            try:
                os.replace(filepath, bad_path)
                print(f"[DreamerReplay] invalid replay buffer moved to {bad_path}: {exc}")
            except OSError:
                print(f"[DreamerReplay] invalid replay buffer ignored: {filepath}: {exc}")
            self.episodes = deque()
            self.current_episode = []
            self.num_steps = 0
            return False
        self.current_episode = []
        return True

    def collate(
        self,
        chunks: list[list[DreamerStep]],
        *,
        pin_memory: bool = False,
    ) -> dict[str, torch.Tensor]:
        def stack(name: str):
            return np.stack([
                np.stack([getattr(step, name) for step in seq], axis=0)
                for seq in chunks
            ], axis=0)

        joint_ego = stack("joint_ego").astype(self.state_dtype, copy=False)
        global_state = stack("global_state").astype(self.state_dtype, copy=False)
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
        loss_mask = np.ones_like(reward, dtype=np.float32)
        if self.context_length > 0:
            loss_mask[:, :self.context_length] = 0.0

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
            "loss_mask": loss_mask,
        }
        tensor_batch = {key: torch.from_numpy(value) for key, value in batch.items()}
        if pin_memory:
            tensor_batch = {key: value.pin_memory() for key, value in tensor_batch.items()}
        return tensor_batch

    def to_device(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.device.type == "cpu":
            return batch
        return {
            key: value.to(self.device, non_blocking=non_blocking)
            for key, value in batch.items()
        }

    def _to_uint8(self, value: np.ndarray) -> np.ndarray:
        arr = np.asarray(value)
        if arr.dtype == self.state_dtype:
            return arr
        return np.clip(arr * 255.0, 0, 255).astype(self.state_dtype)
