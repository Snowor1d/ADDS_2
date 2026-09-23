"""Replay storage by source rather than by rendered image.

docs/outdoor_madrl_redesign.md section 6. The previous buffer kept every
rendered stack twice (state and next state) for every transition: 35 kB per
transition before the new branches. Here:

  * static layers are stored once per (map geometry, hazard placement) key,
    in RAM with a copy on disk, so an entry evicted from RAM can be reloaded
    for as long as the buffer still refers to it;
  * each decision instant stores the team's DecisionRecord once (the robots'
    own measurements, poses, modes, message availability, and the critic's
    true crowd summary), about 6 kB for three robots;
  * state, next state and the four-instant windows are rebuilt from indices
    with `sim.observation.build_observations`, the same function the worker
    acted on. A window never crosses an episode boundary, and a transition
    whose history has been overwritten is not sampled.

Nothing written under one set of schema versions is read under another.
"""

from __future__ import annotations

import collections
import json
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from sim.observation import (DecisionRecord, ObsRequest, StaticLayers,
                             build_observations, critic_privileged, obs_shapes)


class SchemaMismatch(RuntimeError):
    """Stored data was written under different schema versions."""


def check_schema(stored: Dict[str, str], expected: Dict[str, str],
                 what: str) -> None:
    bad = {k: (stored.get(k), v) for k, v in expected.items()
           if stored.get(k) != v}
    if bad:
        detail = ", ".join(f"{k}: stored {a!r} != current {b!r}"
                           for k, (a, b) in bad.items())
        raise SchemaMismatch(f"{what} was written under different schema "
                             f"versions ({detail}); refusing to read it")


class StaticStore:
    """Static layers by key: an LRU in RAM, every entry also on disk."""

    def __init__(self, directory: Optional[str], max_entries: int = 256):
        self.directory = directory
        self.max_entries = int(max_entries)
        self._ram: "collections.OrderedDict[str, StaticLayers]" = \
            collections.OrderedDict()
        self._lock = threading.Lock()
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"{key}.npz")

    def put(self, static: StaticLayers) -> None:
        with self._lock:
            self._ram[static.key] = static
            self._ram.move_to_end(static.key)
            while len(self._ram) > self.max_entries:
                self._ram.popitem(last=False)
        if self.directory and not os.path.exists(self._path(static.key)):
            tmp = self._path(static.key) + ".tmp.npz"
            np.savez(tmp, meta=np.array(json.dumps(static.meta())),
                     **static.arrays())
            os.replace(tmp, self._path(static.key))

    def get(self, key: str) -> StaticLayers:
        with self._lock:
            st = self._ram.get(key)
            if st is not None:
                self._ram.move_to_end(key)
                return st
        if not self.directory or not os.path.exists(self._path(key)):
            raise KeyError(f"static layers {key} are neither in RAM nor on "
                           "disk")
        with np.load(self._path(key), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            arrays = {k: data[k] for k in data.files if k != "meta"}
        st = StaticLayers.from_parts(meta, arrays)
        with self._lock:
            self._ram[key] = st
            while len(self._ram) > self.max_entries:
                self._ram.popitem(last=False)
        return st

    def __contains__(self, key: str) -> bool:
        with self._lock:
            if key in self._ram:
                return True
        return bool(self.directory) and os.path.exists(self._path(key))


def joint_observation(cfg, statics: List[StaticLayers],
                      windows: List[List[Optional[DecisionRecord]]],
                      arena: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, np.ndarray]:
    """Team inputs for a batch of instants: every real robot's actor input,
    padded to MAX_ROBOTS with a mask, plus the critic's privileged map.

    Built with `build_observations`, so the learner, the curriculum's value
    estimate and the rollout worker all see the same arrays.
    """
    R = int(cfg.MAX_ROBOTS)
    requests: List[ObsRequest] = []
    rows: List[int] = []
    B = len(windows)
    mask = np.zeros((B, R), np.float32)
    for b, (st, win) in enumerate(zip(statics, windows)):
        for r in range(int(win[-1].n_robots)):
            requests.append(ObsRequest(st, win, r))
            rows.append(b * R + r)
            mask[b, r] = 1.0
    shapes = obs_shapes(cfg)
    # Built in place: each robot's input goes straight into its joint slot.
    # Real slots are overwritten channel by channel, so only the padded ones
    # need zeroing.
    if arena and arena["ego"].shape[0] == B * R:
        # Reused batch memory: first-touch page faults on ~200 MB per batch
        # cost as much as building the batch.
        flat = arena
        flat["state"].fill(0.0)
    else:
        flat = {k: np.empty((B * R,) + shapes[k], np.float32)
                for k in ("ego", "mid", "glob")}
        flat["state"] = np.zeros((B * R,) + shapes["state"], np.float32)
        if arena is not None:
            arena.clear()
            arena.update(flat)
    pad = np.ones(B * R, bool)
    pad[rows] = False
    for k in ("ego", "mid", "glob"):
        flat[k][pad] = 0.0
    build_observations(requests, cfg, out=flat, rows=rows)
    out = {k: v.reshape((B, R) + shapes[k]) for k, v in flat.items()}
    out["mask"] = mask
    out["priv"] = critic_privileged([w[-1] for w in windows], cfg, statics)
    return out


class ReplayBuffer:
    """Decision records in a ring, linked into episodes by index.

    Protocol, per episode: `push(record, ...)` for every decision instant in
    order, the action and its outcome arriving with the record they belong
    to, and a last `push(final_record, action=None)` carrying the state after
    the final action. Records from different workers interleave freely.
    """

    def __init__(self, cfg, capacity: int, static_store: StaticStore):
        self.cfg = cfg
        self.capacity = int(capacity)
        self.statics = static_store
        R = int(cfg.MAX_ROBOTS)
        E = int(cfg.EGO_MAP_SIZE)
        G = int(cfg.OBS_GLOBAL_SIZE)
        from sim.robot_action import ACTION_DIM
        self.action_dim = ACTION_DIM
        self._mask_bytes = (E * E + 7) // 8
        C = self.capacity
        self.n_robots = np.zeros(C, np.int8)
        self.anchors = np.zeros((C, R, 2), np.int32)
        self.counts = np.zeros((C, R, E, E), np.uint8)
        self.observed = np.zeros((C, R, self._mask_bytes), np.uint8)
        self.pose = np.zeros((C, R, 2), np.float32)
        self.mode = np.zeros((C, R), np.int8)
        self.signal = np.zeros((C, R, 2), np.float32)
        self.avail = np.zeros((C, R, R), np.uint8)
        self.priv = np.zeros((C, G, G), np.uint8)
        self.action = np.zeros((C, R, ACTION_DIM), np.float32)
        self.has_action = np.zeros(C, bool)
        self.step_rewards = np.zeros((C, int(cfg.ACTION_SCALE)), np.float32)
        self.hold = np.zeros(C, np.int16)
        self.terminal = np.zeros(C, bool)
        self.episode = np.full(C, -1, np.int64)
        self.step = np.full(C, -1, np.int32)
        self.prev = np.full(C, -1, np.int64)
        self.next = np.full(C, -1, np.int64)
        self.static_id = np.full(C, -1, np.int32)
        self._static_keys: List[str] = []
        self._static_index: Dict[str, int] = {}
        self._last: Dict[int, int] = {}       # episode uid -> newest index
        self.ptr = 0
        self.size = 0
        # Held by push and sample, so a prefetch thread can rebuild a batch
        # while the main thread keeps inserting.
        self.lock = threading.RLock()
        self._arena_sets = [({}, {}), ({}, {})]
        self._arena_turn = 0

    # ---------------------------------------------------------- insertion

    def _static_slot(self, key: str) -> int:
        slot = self._static_index.get(key)
        if slot is None:
            slot = len(self._static_keys)
            self._static_keys.append(key)
            self._static_index[key] = slot
        return slot

    def push(self, episode_uid: int, step: int, record: DecisionRecord,
             static_key: str, action: Optional[np.ndarray] = None,
             step_rewards=(), terminal: bool = False) -> int:
        with self.lock:
            return self._push(episode_uid, step, record, static_key, action,
                              step_rewards, terminal)

    def _push(self, episode_uid, step, record, static_key, action,
              step_rewards, terminal) -> int:
        if static_key not in self.statics:
            raise KeyError(f"static layers {static_key} were never stored")
        i = self.ptr
        old_uid = int(self.episode[i])
        if old_uid >= 0 and self._last.get(old_uid) == i:
            del self._last[old_uid]
        self.n_robots[i] = record.n_robots
        self.anchors[i] = record.anchors
        self.counts[i] = record.counts
        self.observed[i] = np.packbits(
            record.observed.reshape(record.observed.shape[0], -1), axis=1)
        self.pose[i] = record.pose
        self.mode[i] = record.mode
        self.signal[i] = record.signal
        self.avail[i] = record.avail
        self.priv[i] = record.priv
        self.has_action[i] = action is not None
        if action is not None:
            self.action[i] = action
        sr = np.asarray(step_rewards, dtype=np.float32).reshape(-1)
        if sr.shape[0] > self.step_rewards.shape[1]:
            raise ValueError(f"an action held {sr.shape[0]} steps, more than "
                             f"ACTION_SCALE={self.step_rewards.shape[1]}")
        self.step_rewards[i] = 0.0
        self.step_rewards[i, :sr.shape[0]] = sr
        self.hold[i] = int(sr.shape[0])
        self.terminal[i] = bool(terminal)
        self.episode[i] = int(episode_uid)
        self.step[i] = int(step)
        self.static_id[i] = self._static_slot(static_key)
        self.next[i] = -1
        prev = self._last.get(int(episode_uid), -1)
        if prev >= 0 and self.episode[prev] == episode_uid \
                and self.step[prev] == step - 1:
            self.prev[i] = prev
            self.next[prev] = i
        else:
            self.prev[i] = -1
        self._last[int(episode_uid)] = i
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return i

    def end_episode(self, episode_uid: int) -> None:
        with self.lock:
            self._last.pop(int(episode_uid), None)

    # ------------------------------------------------------------ reading

    def record(self, i: int) -> DecisionRecord:
        E = int(self.cfg.EGO_MAP_SIZE)
        R = int(self.cfg.MAX_ROBOTS)
        obs = np.unpackbits(self.observed[i], axis=1)[:, :E * E]
        return DecisionRecord(
            n_robots=int(self.n_robots[i]), anchors=self.anchors[i],
            counts=self.counts[i], observed=obs.reshape(R, E, E).astype(bool),
            pose=self.pose[i], mode=self.mode[i], signal=self.signal[i],
            avail=self.avail[i], priv=self.priv[i])

    def _linked(self, a: int, b: int) -> bool:
        """b is still the record that followed a in the same episode."""
        return (b >= 0 and self.episode[b] == self.episode[a]
                and self.step[b] == self.step[a] + 1)

    def window_indices(self, i: int) -> Optional[List[Optional[int]]]:
        """Indices of the H-instant window ending at i, oldest first, None
        before the episode's start. None when the history was overwritten."""
        H = int(self.cfg.OBS_HISTORY_DECISIONS)
        out = [i]
        cur = i
        for _ in range(H - 1):
            if self.step[cur] == 0:
                break
            p = int(self.prev[cur])
            if p < 0 or not self._linked(p, cur):
                return None
            out.append(p)
            cur = p
        out.reverse()
        return [None] * (H - len(out)) + out

    def valid_transition(self, i: int) -> bool:
        if not self.has_action[i] or self.episode[i] < 0:
            return False
        n = int(self.next[i])
        if n < 0 or not self._linked(i, n):
            return False
        return (self.window_indices(i) is not None
                and self.window_indices(n) is not None)

    def __len__(self) -> int:
        return self.size

    def _joint(self, indices: List[int],
               arena: Optional[Dict[str, np.ndarray]] = None
               ) -> Dict[str, np.ndarray]:
        """Joint observation (B, N, ...) plus mask and privileged map."""
        cache: Dict[int, DecisionRecord] = {}

        def rec(j):
            if j not in cache:
                cache[j] = self.record(j)
            return cache[j]

        statics, windows = [], []
        for i in indices:
            windows.append([None if j is None else rec(j)
                            for j in self.window_indices(i)])
            statics.append(
                self.statics.get(self._static_keys[int(self.static_id[i])]))
        return joint_observation(self.cfg, statics, windows, arena)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None
               ) -> Dict[str, np.ndarray]:
        """A batch of valid transitions, rebuilt from their records.

        The observation arrays live in memory that the call after next
        reuses (two sets alternate), so one batch can be trained on while the
        following one is built."""
        with self.lock:
            return self._sample(batch_size, rng)

    def _sample(self, batch_size, rng):
        rng = rng or np.random.default_rng()
        if self.size == 0:
            raise ValueError("empty replay buffer")
        chosen: List[int] = []
        tries = 0
        while len(chosen) < batch_size:
            tries += 1
            if tries > 50:
                raise ValueError("not enough valid transitions to sample")
            cand = rng.integers(0, self.size, size=4 * batch_size)
            for i in cand:
                i = int(i)
                if self.valid_transition(i):
                    chosen.append(i)
                    if len(chosen) == batch_size:
                        break
        nxt = [int(self.next[i]) for i in chosen]
        arenas = self._arena_sets[self._arena_turn]
        self._arena_turn ^= 1
        s = self._joint(chosen, arenas[0])
        s2 = self._joint(nxt, arenas[1])
        n_real = s["mask"].sum(1).astype(np.int64)
        agent = (rng.random(len(chosen)) * n_real).astype(np.int64)
        return {
            "obs": {k: s[k] for k in ("ego", "mid", "glob", "state", "priv")},
            "mask": s["mask"],
            "action": self.action[chosen].copy(),
            "step_rewards": self.step_rewards[chosen].copy(),
            "hold": self.hold[chosen].astype(np.float32),
            "terminal": self.terminal[chosen].astype(np.float32),
            "next_obs": {k: s2[k] for k in ("ego", "mid", "glob", "state",
                                              "priv")},
            "next_mask": s2["mask"],
            "agent_index": agent,
            "indices": np.asarray(chosen),
        }

    # -------------------------------------------------------- persistence

    ARRAYS = ("n_robots", "anchors", "counts", "observed", "pose", "mode",
              "signal", "avail", "priv", "action", "has_action", "step_rewards",
              "hold", "terminal", "episode", "step", "prev", "next",
              "static_id")

    def save(self, path: str, schema: Dict[str, str]) -> None:
        n = self.size
        payload = {k: getattr(self, k)[:n] for k in self.ARRAYS}
        meta = {"schema": schema, "capacity": self.capacity, "ptr": self.ptr,
                "size": self.size, "static_keys": self._static_keys}
        tmp = path + ".tmp.npz"
        np.savez(tmp, meta=np.array(json.dumps(meta)), **payload)
        os.replace(tmp, path)

    def load(self, path: str, schema: Dict[str, str]) -> None:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            check_schema(meta.get("schema", {}), schema,
                         f"replay buffer {path}")
            n = int(meta["size"])
            if n > self.capacity:
                raise ValueError(f"stored buffer holds {n} > capacity "
                                 f"{self.capacity}")
            for k in self.ARRAYS:
                arr = data[k]
                if arr.shape[1:] != getattr(self, k).shape[1:]:
                    raise SchemaMismatch(f"replay array {k} has shape "
                                         f"{arr.shape[1:]}, expected "
                                         f"{getattr(self, k).shape[1:]}")
                getattr(self, k)[:n] = arr
        self.size = n
        self.ptr = int(meta["ptr"]) % self.capacity
        self._static_keys = list(meta["static_keys"])
        self._static_index = {k: i for i, k in enumerate(self._static_keys)}
        missing = [k for k in set(self._static_keys) if k not in self.statics]
        if missing:
            raise KeyError(f"{len(missing)} static layer entries referenced by "
                           "the buffer are missing from the static store")
        self._last = {}

    def nbytes(self) -> int:
        return int(sum(getattr(self, k).nbytes for k in self.ARRAYS))


class BatchPrefetcher:
    """Rebuilds the next batch on a background thread while the GPU trains on
    the current one. One batch ahead at most, so the two alternating arenas
    in ReplayBuffer are never both in use."""

    def __init__(self, buffer: ReplayBuffer, batch_size: int, seed: int = 0):
        import queue as _q
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self._q: "_q.Queue" = _q.Queue(maxsize=1)
        self._want = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="replay-prefetch")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            if not self._want.wait(timeout=0.5):
                continue
            self._want.clear()
            try:
                batch = self.buffer.sample(self.batch_size, self.rng)
            except ValueError:
                batch = None
            self._q.put(batch)

    def get(self, timeout: float = 30.0):
        """The prefetched batch (None if the buffer could not supply one),
        and a request for the next."""
        if not hasattr(self, "_primed"):
            self._primed = True
            self._want.set()
        batch = self._q.get(timeout=timeout)
        self._want.set()
        return batch

    def stop(self):
        self._stop.set()
