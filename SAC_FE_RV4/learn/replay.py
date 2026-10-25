"""Replay storage that keeps each rendered frame once.

RV3 stored, per transition, the state stack and the next-state stack: eight
frames (4 ego + 4 global, twice), about 25 kB, 25 GB at the default one
million transitions, and saved all of it compressed every 100 episodes. But
consecutive decisions of one episode share three of their four frames, and a
transition's next state is the next transition's state. Here a frame is
written once into a ring and a transition holds the ids of its frames; the
stacks are gathered when a batch is drawn. One million transitions take
about 4 GB.

Ids are global counters; a ring slot is valid for id i while it still holds
i, so a transition whose frames were overwritten is detected and redrawn
rather than read wrong. The frame ring is sized well above the transition
ring, so that happens only at the oldest edge.

Transitions of an episode are linked (next_t), which gives n-step returns:
rewards are summed along the link until N_STEP, the episode's end, or a
successor that has not arrived yet, and the target bootstraps from the last
included transition's next state with gamma ** (number summed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

STATE_DIM = 5


@dataclass
class EpisodeCursor:
    """Where one in-flight episode is: its frame stack (ids, newest first),
    its static global layer and the last transition written."""
    episode_id: int
    stack: List[int]
    prev_t: int = -1
    t_ids: List[int] = field(default_factory=list)


class FrameReplay:
    def __init__(self, capacity: int, ego_hw: int, glob_hw: int, stack: int,
                 static_channel: bool, device="cpu"):
        self.cap = int(capacity)
        self.stack = int(stack)
        self.ego_hw = int(ego_hw)
        self.glob_hw = int(glob_hw)
        self.static = bool(static_channel)
        self.device = torch.device(device)
        self.fcap = int(self.cap * 1.1) + 65536
        self.ecap = max(8192, self.cap // 16)

        self.ego_f = np.zeros((self.fcap, ego_hw, ego_hw), np.uint8)
        self.glob_f = np.zeros((self.fcap, glob_hw, glob_hw), np.uint8)
        self.ego_static_f = np.zeros((self.fcap, ego_hw, ego_hw), np.uint8) if self.static else None
        self.f_id = np.full(self.fcap, -1, np.int64)
        self.f_next = 0

        self.glob_static_e = np.zeros((self.ecap, glob_hw, glob_hw), np.uint8) if self.static else None
        self.e_id = np.full(self.ecap, -1, np.int64)
        self.e_next = 0

        c = self.cap
        self.s_f = np.zeros((c, self.stack), np.int64)
        self.s2_f0 = np.zeros(c, np.int64)
        self.ep = np.zeros(c, np.int64)
        self.state = np.zeros((c, STATE_DIM), np.float32)
        self.state2 = np.zeros((c, STATE_DIM), np.float32)
        self.action = np.zeros((c, 2), np.float32)
        self.reward = np.zeros(c, np.float32)
        self.terminal = np.zeros(c, bool)   # the task ended (everyone out)
        self.last = np.zeros(c, bool)       # the episode ended (terminal or time limit)
        self.trainable = np.ones(c, bool)   # False: kept for scoring only (UED_REPLAY_ONLY_UPDATES)
        self.t_id = np.full(c, -1, np.int64)
        self.next_t = np.full(c, -1, np.int64)
        self.t_next = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------ writing

    def _put_frame(self, ego, glob, ego_static) -> int:
        fid = self.f_next
        slot = fid % self.fcap
        self.ego_f[slot] = ego
        self.glob_f[slot] = glob
        if self.static:
            self.ego_static_f[slot] = ego_static
        self.f_id[slot] = fid
        self.f_next += 1
        return fid

    def begin_episode(self, ego, glob, ego_static=None, glob_static=None) -> EpisodeCursor:
        eid = self.e_next
        self.e_next += 1
        es = eid % self.ecap
        self.e_id[es] = eid
        if self.static:
            self.glob_static_e[es] = glob_static
        fid = self._put_frame(ego, glob, ego_static)
        return EpisodeCursor(episode_id=eid, stack=[fid] * self.stack)

    def add(self, cur: EpisodeCursor, state, action, reward, next_ego, next_glob,
            next_ego_static, next_state, terminal: bool, last: bool,
            trainable: bool = True) -> int:
        f0 = self._put_frame(next_ego, next_glob, next_ego_static)
        tid = self.t_next
        slot = tid % self.cap
        self.s_f[slot] = cur.stack
        self.s2_f0[slot] = f0
        self.ep[slot] = cur.episode_id
        self.state[slot] = state
        self.state2[slot] = next_state
        self.action[slot] = action
        self.reward[slot] = reward
        self.terminal[slot] = bool(terminal)
        self.last[slot] = bool(last)
        self.trainable[slot] = bool(trainable)
        self.t_id[slot] = tid
        self.next_t[slot] = -1
        if cur.prev_t >= 0:
            ps = cur.prev_t % self.cap
            if self.t_id[ps] == cur.prev_t:
                self.next_t[ps] = tid
        self.t_next += 1
        self.size = min(self.size + 1, self.cap)
        cur.prev_t = tid
        cur.stack = [f0] + cur.stack[:-1]
        cur.t_ids.append(tid)
        return tid

    def advance_without_storing(self, cur: EpisodeCursor, next_ego, next_glob, next_ego_static):
        """Keep an episode's stack moving when its transitions are not kept
        (UED_REPLAY_ONLY_UPDATES): frames still go into the ring so later
        stored transitions of the same episode could reference them."""
        f0 = self._put_frame(next_ego, next_glob, next_ego_static)
        cur.stack = [f0] + cur.stack[:-1]
        cur.prev_t = -1

    # ------------------------------------------------------------ reading

    def _frames_valid(self, fids: np.ndarray) -> np.ndarray:
        return self.f_id[fids % self.fcap] == fids

    def _valid(self, slots: np.ndarray, for_training: bool = False) -> np.ndarray:
        tid = self.t_id[slots]
        ok = tid >= 0
        if for_training:
            ok &= self.trainable[slots]
        ok &= self._frames_valid(self.s_f[slots]).all(axis=1)
        ok &= self._frames_valid(self.s2_f0[slots])
        if self.static:
            e = self.ep[slots]
            ok &= self.e_id[e % self.ecap] == e
        return ok

    def sample_slots(self, batch: int, rng: np.random.Generator) -> np.ndarray:
        if self.size < batch:
            raise ValueError(f"not enough samples: {self.size} < {batch}")
        slots = rng.integers(0, self.size, size=batch)
        for _ in range(50):
            bad = ~self._valid(slots, for_training=True)
            if not bad.any():
                return slots
            slots[bad] = rng.integers(0, self.size, size=int(bad.sum()))
        return slots[self._valid(slots, for_training=True)]

    def _nstep(self, slots: np.ndarray, n: int, gamma: float, timeout_bootstrap: bool):
        """Sum rewards along the episode link. Returns (reward, discount,
        done, end_slot): the target is reward + discount * (1 - done) * V(s2
        of end_slot)."""
        B = slots.shape[0]
        R = self.reward[slots].astype(np.float64).copy()
        disc = np.full(B, gamma, np.float64)
        end = slots.copy()
        active = ~self.last[slots]
        for _ in range(1, n):
            if not active.any():
                break
            nxt = self.next_t[end]
            has = active & (nxt >= 0)
            nslot = np.where(has, nxt % self.cap, end)
            has &= self.t_id[nslot] == nxt
            has &= self._frames_valid(self.s2_f0[nslot])
            R = np.where(has, R + disc * self.reward[nslot], R)
            disc = np.where(has, disc * gamma, disc)
            end = np.where(has, nslot, end)
            active = has & ~self.last[nslot]
        done = self.terminal[end] if timeout_bootstrap else self.last[end]
        return R.astype(np.float32), disc.astype(np.float32), done, end

    def _obs(self, fids_stack: np.ndarray, ego_static_fid: np.ndarray, ep: np.ndarray):
        fs = fids_stack % self.fcap
        ego = self.ego_f[fs]                     # (B, S, E, E)
        glob = self.glob_f[fs]                   # (B, S, G, G)
        if self.static:
            ego = np.concatenate([ego, self.ego_static_f[ego_static_fid % self.fcap][:, None]], 1)
            glob = np.concatenate([glob, self.glob_static_e[ep % self.ecap][:, None]], 1)
        return ego, glob

    def batch_from_slots(self, slots: np.ndarray, n_step: int, gamma: float,
                         timeout_bootstrap: bool) -> Dict[str, torch.Tensor]:
        R, disc, done, end = self._nstep(slots, n_step, gamma, timeout_bootstrap)
        s_f = self.s_f[slots]
        ego, glob = self._obs(s_f, s_f[:, 0], self.ep[slots])
        s2 = np.concatenate([self.s2_f0[end][:, None], self.s_f[end][:, :-1]], axis=1)
        ego2, glob2 = self._obs(s2, s2[:, 0], self.ep[end])
        dev = self.device

        def img(x):
            return torch.from_numpy(x).to(dev, non_blocking=True).float().div_(255.0)

        def t(x):
            return torch.from_numpy(np.ascontiguousarray(x)).to(dev, non_blocking=True)

        return {
            "ego": img(ego), "glob": img(glob), "state": t(self.state[slots]),
            "action": t(self.action[slots]), "reward": t(R), "discount": t(disc),
            "done": t(done.astype(np.float32)),
            "ego2": img(ego2), "glob2": img(glob2), "state2": t(self.state2[end]),
        }

    def obs_from_slots(self, slots: np.ndarray, nxt: bool = False) -> Dict[str, torch.Tensor]:
        """(ego, glob, state) of the transitions' states, or of their next
        states with nxt=True, as network inputs."""
        if nxt:
            s2 = np.concatenate([self.s2_f0[slots][:, None], self.s_f[slots][:, :-1]], axis=1)
            ego, glob = self._obs(s2, s2[:, 0], self.ep[slots])
            st = self.state2[slots]
        else:
            s_f = self.s_f[slots]
            ego, glob = self._obs(s_f, s_f[:, 0], self.ep[slots])
            st = self.state[slots]
        dev = self.device
        return {"ego": torch.from_numpy(ego).to(dev).float().div_(255.0),
                "glob": torch.from_numpy(glob).to(dev).float().div_(255.0),
                "state": torch.from_numpy(np.ascontiguousarray(st)).to(dev)}

    def sample(self, batch: int, n_step: int, gamma: float, timeout_bootstrap: bool,
               rng: np.random.Generator) -> Dict[str, torch.Tensor]:
        return self.batch_from_slots(self.sample_slots(batch, rng), n_step, gamma,
                                     timeout_bootstrap)

    def episode_slots(self, t_ids: Sequence[int]) -> np.ndarray:
        """Slots of the given transition ids that are still stored intact."""
        ids = np.asarray(list(t_ids), np.int64)
        if ids.size == 0:
            return ids
        slots = ids % self.cap
        ok = (self.t_id[slots] == ids) & self._valid(slots)
        return slots[ok]

    # ---------------------------------------------------------- persistence

    _ARRAYS = ("ego_f", "glob_f", "ego_static_f", "f_id", "glob_static_e", "e_id",
               "s_f", "s2_f0", "ep", "state", "state2", "action", "reward",
               "terminal", "last", "trainable", "t_id", "next_t")

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for name in self._ARRAYS:
            arr = getattr(self, name)
            if arr is not None:
                np.save(os.path.join(directory, name + ".npy"), arr)
        np.save(os.path.join(directory, "counters.npy"),
                np.array([self.f_next, self.e_next, self.t_next, self.size, self.cap,
                          self.fcap, self.ecap, self.stack, self.ego_hw, self.glob_hw,
                          int(self.static)], np.int64))

    def load(self, directory: str) -> bool:
        p = os.path.join(directory, "counters.npy")
        if not os.path.exists(p):
            return False
        c = np.load(p)
        if tuple(c[4:]) != (self.cap, self.fcap, self.ecap, self.stack, self.ego_hw,
                            self.glob_hw, int(self.static)):
            print("[replay] stored buffer has a different shape; starting empty")
            return False
        for name in self._ARRAYS:
            if getattr(self, name) is not None:
                getattr(self, name)[...] = np.load(os.path.join(directory, name + ".npy"))
        self.f_next, self.e_next, self.t_next, self.size = (int(x) for x in c[:4])
        # In-flight episodes of the previous process are gone; their last
        # transitions keep next_t = -1, which ends their n-step sums there.
        return True
