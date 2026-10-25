"""Which map each training episode runs: dataset maps, curriculum levels, or
a mix (TRAIN_MAP_SOURCE). Main process only.

    task = sampler.next_task(episode)        # sent to a worker
    sampler.on_episode(task, outcome, ...)   # when the worker reports back

Fresh levels are produced by UED_GENERATION_WORKERS background processes
(generation takes about a second at 100 m, sometimes tens of seconds), so
dispatching never waits on the generator unless nothing at all is available.

Training generator seeds never fall in UED_RESERVED_SEED_RANGES, the block
the validation, tuning and test suites are generated from.
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
import queue
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ued.level import Level, generate_level, next_level_id, seed_is_reserved
from ued.population import LevelPopulation
from ued import score as score_mod


@dataclass
class Task:
    task_id: int
    kind: str                       # "dataset" | "level"
    crowd: int
    augmentation: str = "identity"
    map_num: Optional[int] = None
    level: Optional[Level] = None
    seed: Optional[int] = None      # initial placement seed; None = the worker's own
    is_new: bool = False
    need_shuttle_ref: bool = False
    trainable: bool = True

    def label(self) -> str:
        return f"map{self.map_num}" if self.kind == "dataset" else f"level{self.level.level_id}"


def _draw_training_seed(rng: random.Random, reserved) -> int:
    while True:
        s = rng.randrange(0, 2 ** 31 - 1)
        if not seed_is_reserved(s, reserved):
            return s


def producer_main(out_q, stop_ev, seed: int):
    """Background level generator (spawned; reads the configuration from the
    inherited environment)."""
    from configs import cfg
    rng = random.Random(seed)
    sizes = list(cfg.UED_MAP_SIZES_M)
    weights = list(cfg.UED_MAP_SIZE_WEIGHTS) if cfg.UED_MAP_SIZE_WEIGHTS else None
    while not stop_ev.is_set():
        s = _draw_training_seed(rng, cfg.UED_RESERVED_SEED_RANGES)
        size = rng.choices(sizes, weights=weights, k=1)[0]
        diff = rng.randint(*cfg.UED_DIFFICULTY_RANGE)
        crowd = rng.randint(*cfg.UED_CROWD_RANGE)
        lv = generate_level(s, size, diff, crowd, timeout_s=float(cfg.UED_GEN_TIMEOUT_S),
                            max_density=float(cfg.UED_MAX_OBSTACLE_DENSITY),
                            placement_seed=rng.randrange(0, 2 ** 31 - 1))
        if lv is None:
            continue
        payload = lv.to_json()
        while not stop_ev.is_set():
            try:
                out_q.put(payload, timeout=1.0)
                break
            except queue.Full:
                continue


class TaskSampler:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.seed = int(seed)
        self.population = LevelPopulation(cfg, rng=random.Random(seed + 1))
        self.uses_ued = cfg.TRAIN_MAP_SOURCE in ("ued", "mixed")
        self._ids = itertools.count(1)
        self._procs: List = []
        self._fresh_q = None
        self._stop = None
        self.n_new = 0
        self.n_replay = 0
        self.n_dataset = 0
        self.n_unscored = 0
        self.n_waited = 0

    # ------------------------------------------------------------ producers

    def start(self, ctx=None):
        if not self.uses_ued or self._procs:
            return
        ctx = ctx or mp.get_context("spawn")
        self._fresh_q = ctx.Queue(maxsize=int(self.cfg.UED_FRESH_QUEUE))
        self._stop = ctx.Event()
        for i in range(int(self.cfg.UED_GENERATION_WORKERS)):
            p = ctx.Process(target=producer_main, args=(self._fresh_q, self._stop,
                                                        self.seed * 1000 + 17 * i + 1),
                            daemon=True, name=f"ued-producer-{i}")
            p.start()
            self._procs.append(p)

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        self._procs = []

    def _supervise(self):
        for i, p in enumerate(self._procs):
            if not p.is_alive():
                ctx = mp.get_context("spawn")
                q = ctx.Process(target=producer_main,
                                args=(self._fresh_q, self._stop,
                                      self.rng.randrange(1, 2 ** 30)), daemon=True)
                q.start()
                self._procs[i] = q

    def _take_fresh(self, block_s: float) -> Optional[Level]:
        if self._fresh_q is None:
            # No producer (tests, or start() not called): generate inline.
            for _ in range(50):
                s = _draw_training_seed(self.rng, self.cfg.UED_RESERVED_SEED_RANGES)
                lv = generate_level(s, self.rng.choice(self.cfg.UED_MAP_SIZES_M),
                                    self.rng.randint(*self.cfg.UED_DIFFICULTY_RANGE),
                                    self.rng.randint(*self.cfg.UED_CROWD_RANGE),
                                    timeout_s=float(self.cfg.UED_GEN_TIMEOUT_S),
                                    max_density=float(self.cfg.UED_MAX_OBSTACLE_DENSITY),
                                    placement_seed=self.rng.randrange(0, 2 ** 31 - 1))
                if lv is not None:
                    lv.level_id = next_level_id()
                    return lv
            return None
        try:
            d = self._fresh_q.get(timeout=block_s) if block_s > 0 else self._fresh_q.get_nowait()
        except queue.Empty:
            self._supervise()
            return None
        lv = Level.from_json(d)
        lv.level_id = next_level_id()
        return lv

    # ------------------------------------------------------------- sampling

    def p_new(self, episode: int) -> float:
        c = self.cfg
        if c.UED_METHOD == "dr" or episode < int(c.UED_WARMUP_EPISODES):
            return 1.0
        if c.UED_METHOD == "plr":
            return 1.0 - float(c.UED_P_REPLAY)
        t = min(1.0, max(0, episode - int(c.UED_WARMUP_EPISODES)) / max(1, int(c.UED_P_NEW_DECAY_EPISODES)))
        return float(c.UED_P_NEW_START) + (float(c.UED_P_NEW_END) - float(c.UED_P_NEW_START)) * t

    def _aug(self, use: bool, transforms) -> str:
        return self.rng.choice(list(transforms)) if use else "identity"

    def _dataset_task(self) -> Task:
        c = self.cfg
        maps = list(c.DATASET_MAPS)
        w = list(c.DATASET_MAP_WEIGHTS) if c.DATASET_MAP_WEIGHTS else None
        m = self.rng.choices(maps, weights=w, k=1)[0]
        self.n_dataset += 1
        return Task(task_id=next(self._ids), kind="dataset", map_num=int(m),
                    crowd=self.rng.randint(*c.DATASET_CROWD_RANGE),
                    augmentation=self._aug(c.DATASET_AUGMENTATION, c.DATASET_AUGMENTATION_TRANSFORMS))

    def _level_task(self, level: Level, is_new: bool) -> Task:
        c = self.cfg
        rec = self.population.get(level.level_id)
        need_ref = (c.UED_SCORE == "shuttle_regret" or c.UED_SUCCESS_REFERENCE == "shuttle") and \
            (rec is None or rec.ref_shuttle is None)
        return Task(task_id=next(self._ids), kind="level", level=level, crowd=int(level.crowd),
                    seed=level.placement_seed, is_new=is_new, need_shuttle_ref=need_ref,
                    trainable=not (c.UED_REPLAY_ONLY_UPDATES and is_new),
                    augmentation=self._aug(c.UED_AUGMENTATION, c.UED_AUGMENTATION_TRANSFORMS))

    def next_task(self, episode: int) -> Task:
        c = self.cfg
        if c.TRAIN_MAP_SOURCE == "dataset" or (
                c.TRAIN_MAP_SOURCE == "mixed" and self.rng.random() < float(c.MIXED_DATASET_PROB)):
            return self._dataset_task()
        want_new = self.rng.random() < self.p_new(episode) or len(self.population) == 0
        if not want_new:
            lv = self.population.sample()
            if lv is not None:
                self.n_replay += 1
                return self._level_task(lv, is_new=False)
        lv = self._take_fresh(block_s=0.0)
        if lv is None and len(self.population) > 0:
            lv = self.population.sample()
            self.n_replay += 1
            self.n_waited += 1
            return self._level_task(lv, is_new=False)
        while lv is None:
            self.n_waited += 1
            lv = self._take_fresh(block_s=5.0)
        self.n_new += 1
        return self._level_task(lv, is_new=True)

    # ------------------------------------------------------------- outcomes

    def on_episode(self, task: Task, outcome: Dict, episode: int, epsilon: float,
                   values: Optional[np.ndarray] = None, rewards: Optional[np.ndarray] = None,
                   last_value: float = 0.0, last_terminal: bool = True,
                   gamma: float = 0.99) -> None:
        """outcome: evac_time_100, freeflow_steps, shuttle_time (optional)."""
        if task.kind != "level":
            return
        c = self.cfg
        pop = self.population
        rec = pop.get(task.level.level_id)
        if rec is None:
            rec = pop.add(task.level)
        if epsilon > float(c.UED_SCORE_MAX_EPSILON):
            self.n_unscored += 1
            if outcome.get("shuttle_time") and rec.ref_shuttle is None:
                rec.ref_shuttle = float(outcome["shuttle_time"])
            if outcome.get("freeflow_steps") and rec.ref_freeflow is None:
                rec.ref_freeflow = float(outcome["freeflow_steps"])
            return
        mm = pv = ret = None
        if rewards is not None and len(rewards):
            ret = score_mod.discounted_return(rewards, gamma)
            if values is not None and len(values) == len(rewards):
                best = ret if rec.best_return is None else max(rec.best_return, ret)
                mm = score_mod.maxmc(values, best)
                pv = score_mod.pvl(rewards, values, last_value, last_terminal, gamma,
                                   float(c.UED_PVL_LAMBDA))
        rec = pop.update(task.level.level_id, episode, float(outcome["evac_time_100"]),
                         freeflow=outcome.get("freeflow_steps"),
                         shuttle=outcome.get("shuttle_time"),
                         maxmc_value=mm, pvl_value=pv, ret=ret)
        if rec is not None and not task.is_new and pop.should_breed(rec):
            pop.breed(rec)

    def needs_values(self) -> bool:
        return self.uses_ued and self.cfg.UED_SCORE in ("maxmc", "pvl", "hybrid")

    # ---------------------------------------------------------- diagnostics

    def stats(self) -> Dict[str, float]:
        out = {f"ued/{k}": v for k, v in self.population.stats().items()} if self.uses_ued else {}
        tot = max(1, self.n_new + self.n_replay)
        out.update({"ued/n_new": float(self.n_new), "ued/n_replay": float(self.n_replay),
                    "ued/frac_replay": self.n_replay / tot, "ued/n_unscored": float(self.n_unscored),
                    "ued/n_waited": float(self.n_waited), "ued/n_dataset": float(self.n_dataset)})
        if self._fresh_q is not None:
            try:
                out["ued/fresh_ready"] = float(self._fresh_q.qsize())
            except NotImplementedError:
                pass
        return out

    def state_dict(self) -> Dict:
        return {"population": self.population.state_dict(), "n_new": self.n_new,
                "n_replay": self.n_replay, "n_dataset": self.n_dataset,
                "rng": self.rng.getstate()}

    def load_state_dict(self, st: Dict) -> None:
        self.population.load_state_dict(st["population"])
        self.n_new = int(st.get("n_new", 0))
        self.n_replay = int(st.get("n_replay", 0))
        self.n_dataset = int(st.get("n_dataset", 0))
        if "rng" in st:
            try:
                self.rng.setstate(st["rng"])
            except Exception:
                pass
