"""The level population: scores, prioritised sampling, breeding, eviction.

Lives in the main process. Workers only receive levels and report outcomes.

Scores (UED_SCORE) are described in configs/training/ued.py. Raw scores of
different kinds are never compared directly: sampling and the breeding bar
use ranks, so learnability (bounded by 0.25), MaxMC (a return gap) and the
shuttle regret (a time ratio) share one axis.

Sampling follows Robust PLR: P(level) = (1 - rho) * P_score + rho * P_stale,
P_score proportional to (1 / rank) ** (1 / beta), P_stale proportional to the
episodes since the level was last scored.

Outcomes are kept as raw evacuation times with the episode they came from, so
the success threshold can change and old outcomes age out
(UED_TRIAL_MAX_AGE) instead of being frozen into success bits.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ued.level import Level, reserve_level_ids
from ued.mutate import MutationFailed, mutate_level

_BETA_PRIOR = 1.0   # Beta(1, 1) smoothing of the success rate


@dataclass
class LevelRecord:
    level: Level
    history: int = 10
    evac_times: deque = None
    evac_episodes: deque = None
    regrets: deque = None
    ref_freeflow: Optional[float] = None
    ref_shuttle: Optional[float] = None
    maxmc: Optional[float] = None
    pvl: Optional[float] = None
    best_return: Optional[float] = None
    trials: int = 0
    visits: int = 0
    children: int = 0
    last_update_episode: int = 0
    inserted_at: int = 0
    inherited_score: float = 0.0

    def __post_init__(self):
        if self.evac_times is None:
            self.evac_times = deque(maxlen=self.history)
            self.evac_episodes = deque(maxlen=self.history)
            self.regrets = deque(maxlen=self.history)


class LevelPopulation:
    def __init__(self, cfg, rng: Optional[random.Random] = None):
        self.cfg = cfg
        self.capacity = int(cfg.UED_POP_SIZE)
        self.rng = rng or random.Random()
        self._records: Dict[int, LevelRecord] = {}
        self._lock = threading.RLock()
        self.episode = 0
        self.n_bred = 0
        self.n_evicted = 0
        self.n_mutation_failures = 0
        self.n_inserted = 0

    def __len__(self):
        with self._lock:
            return len(self._records)

    def get(self, level_id: int) -> Optional[LevelRecord]:
        with self._lock:
            return self._records.get(level_id)

    # ------------------------------------------------------------- outcomes

    def _fresh_idx(self, rec: LevelRecord) -> List[int]:
        age = int(self.cfg.UED_TRIAL_MAX_AGE)
        if age <= 0:
            return list(range(len(rec.evac_times)))
        cut = self.episode - age
        return [i for i, e in enumerate(rec.evac_episodes) if e >= cut]

    def threshold(self, rec: LevelRecord) -> float:
        c = self.cfg
        cap = float(c.MAX_STEPS)
        ref = rec.ref_shuttle if c.UED_SUCCESS_REFERENCE == "shuttle" else rec.ref_freeflow
        if ref is None or ref <= 0:
            return cap
        return min(cap, float(c.UED_SUCCESS_K) * float(ref))

    def success_rate(self, rec: LevelRecord) -> float:
        idx = self._fresh_idx(rec)
        lim = self.threshold(rec)
        cap = float(self.cfg.MAX_STEPS)
        s = sum(1 for i in idx if rec.evac_times[i] < lim and rec.evac_times[i] < cap)
        return (s + _BETA_PRIOR) / (len(idx) + 2 * _BETA_PRIOR)

    def learnability(self, rec: LevelRecord) -> float:
        p = self.success_rate(rec)
        return p * (1.0 - p)

    def evac_ratio(self, rec: LevelRecord) -> Optional[float]:
        if not rec.ref_freeflow:
            return None
        done = [t for t in rec.evac_times if t < float(self.cfg.MAX_STEPS)]
        if not done:
            return None
        return float(np.mean(done)) / float(rec.ref_freeflow)

    def raw_score(self, rec: LevelRecord) -> float:
        s = self.cfg.UED_SCORE
        fresh = len(self._fresh_idx(rec))
        if s == "learnability":
            return self.learnability(rec) if fresh else rec.inherited_score
        if s == "maxmc":
            return rec.maxmc if rec.maxmc is not None else rec.inherited_score
        if s == "pvl":
            return rec.pvl if rec.pvl is not None else rec.inherited_score
        if s == "shuttle_regret":
            return float(np.mean(rec.regrets)) if rec.regrets else rec.inherited_score
        # hybrid
        if fresh >= int(self.cfg.UED_MIN_TRIALS):
            return self.learnability(rec)
        if rec.maxmc is not None:
            return rec.maxmc
        return rec.inherited_score

    def add(self, level: Level, inherited_score: float = 0.0) -> LevelRecord:
        with self._lock:
            rec = LevelRecord(level=level, history=int(self.cfg.UED_TRIAL_HISTORY),
                              last_update_episode=self.episode, inserted_at=self.episode,
                              inherited_score=float(inherited_score))
            self._records[level.level_id] = rec
            self.n_inserted += 1
            if len(self._records) > self.capacity:
                self._evict()
            return rec

    def _evict(self):
        tried = [r for r in self._records.values() if r.trials > 0]
        pool = tried or list(self._records.values())
        if tried:
            victim = min(pool, key=self.raw_score)
        else:
            victim = min(pool, key=lambda r: r.inserted_at)
        self._records.pop(victim.level.level_id, None)
        self.n_evicted += 1

    def update(self, level_id: int, episode: int, evac_time: float,
               freeflow: Optional[float] = None, shuttle: Optional[float] = None,
               maxmc_value: Optional[float] = None, pvl_value: Optional[float] = None,
               ret: Optional[float] = None) -> Optional[LevelRecord]:
        with self._lock:
            self.episode = max(self.episode, int(episode))
            rec = self._records.get(level_id)
            if rec is None:
                return None
            if freeflow and rec.ref_freeflow is None:
                rec.ref_freeflow = float(freeflow)
            if shuttle and rec.ref_shuttle is None:
                rec.ref_shuttle = float(shuttle)
            rec.evac_times.append(float(evac_time))
            rec.evac_episodes.append(int(self.episode))
            if rec.ref_shuttle:
                rec.regrets.append((float(evac_time) - rec.ref_shuttle) / rec.ref_shuttle)
            if ret is not None:
                rec.best_return = float(ret) if rec.best_return is None else max(rec.best_return, float(ret))
            # Smoothed, so one noisy trajectory does not dominate.
            if maxmc_value is not None:
                rec.maxmc = maxmc_value if rec.maxmc is None else 0.7 * rec.maxmc + 0.3 * maxmc_value
            if pvl_value is not None:
                rec.pvl = pvl_value if rec.pvl is None else 0.7 * rec.pvl + 0.3 * pvl_value
            rec.trials += 1
            rec.last_update_episode = self.episode
            return rec

    # ------------------------------------------------------------- sampling

    def _rank_weights(self, recs: Sequence[LevelRecord]) -> np.ndarray:
        n = len(recs)
        scores = np.array([self.raw_score(r) for r in recs], dtype=np.float64)
        # Rank 1 = highest score; tied levels share their average rank. With
        # few trials many levels have exactly the same learnability, and a
        # stable sort would hand the higher ranks to whichever was inserted
        # first.
        ranks = np.empty(n)
        uniq, inv = np.unique(-scores, return_inverse=True)
        counts = np.bincount(inv)
        start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        ranks = (start + (counts + 1) / 2.0)[inv]
        beta = max(1e-6, float(self.cfg.UED_SCORE_TEMPERATURE))
        w = (1.0 / ranks) ** (1.0 / beta)
        return w / w.sum()

    def _stale_weights(self, recs: Sequence[LevelRecord]) -> np.ndarray:
        s = np.array([max(1.0, float(self.episode - r.last_update_episode)) for r in recs])
        return s / s.sum()

    def sample(self) -> Optional[Level]:
        with self._lock:
            recs = list(self._records.values())
            if not recs:
                return None
            rho = float(self.cfg.UED_STALENESS_COEF)
            p = (1.0 - rho) * self._rank_weights(recs) + rho * self._stale_weights(recs)
            rec = recs[int(self.rng.choices(range(len(recs)), weights=p.tolist(), k=1)[0])]
            rec.visits += 1
            return rec.level

    # ------------------------------------------------------------- breeding

    def should_breed(self, rec: LevelRecord) -> bool:
        c = self.cfg
        if c.UED_METHOD != "accel":
            return False
        with self._lock:
            if len(self._fresh_idx(rec)) < int(c.UED_MIN_TRIALS):
                return False
            if rec.children >= int(c.UED_MAX_CHILDREN_PER_LEVEL):
                return False
            measured = [r for r in self._records.values()
                        if len(self._fresh_idx(r)) >= int(c.UED_MIN_TRIALS)]
            if len(measured) < 8:
                return False
            mine = self.raw_score(rec)
            below = sum(1 for r in measured if self.raw_score(r) <= mine)
            return below / len(measured) >= float(c.UED_MUTATE_THRESHOLD)

    def breed(self, rec: LevelRecord) -> Optional[Level]:
        try:
            child = mutate_level(rec.level, self.cfg, rng=self.rng)
        except MutationFailed:
            self.n_mutation_failures += 1
            return None
        self.add(child, inherited_score=self.raw_score(rec))
        rec.children += 1
        self.n_bred += 1
        return child

    # ---------------------------------------------------------- diagnostics

    def stats(self) -> Dict[str, float]:
        with self._lock:
            recs = list(self._records.values())
            if not recs:
                return {"pop_size": 0.0}
            tried = [r for r in recs if r.trials > 0]
            out = {
                "pop_size": float(len(recs)),
                "n_scored": float(len(tried)),
                "n_bred": float(self.n_bred),
                "n_evicted": float(self.n_evicted),
                "n_mutation_failures": float(self.n_mutation_failures),
                "mean_generation": float(np.mean([r.level.generation for r in recs])),
                "max_generation": float(max(r.level.generation for r in recs)),
                "mean_obstacle_density": float(np.mean([r.level.obstacle_density() for r in recs])),
                "mean_n_obstacles": float(np.mean([len(r.level.obstacles) for r in recs])),
                "mean_crowd": float(np.mean([r.level.crowd for r in recs])),
                "mean_map_size_m": float(np.mean([r.level.width for r in recs])),
                "mean_difficulty": float(np.mean([r.level.difficulty or 0 for r in recs])),
            }
            if tried:
                sc = [self.raw_score(r) for r in tried]
                out["score_mean"] = float(np.mean(sc))
                out["score_max"] = float(np.max(sc))
                out["success_rate_mean"] = float(np.mean([self.success_rate(r) for r in tried]))
                out["learnability_mean"] = float(np.mean([self.learnability(r) for r in tried]))
                ratios = [x for x in (self.evac_ratio(r) for r in tried) if x is not None]
                if ratios:
                    out["evac_ratio_mean"] = float(np.mean(ratios))
                    out["evac_ratio_median"] = float(np.median(ratios))
                out["staleness_mean"] = float(np.mean([self.episode - r.last_update_episode
                                                       for r in tried]))
                regs = [float(np.mean(r.regrets)) for r in tried if r.regrets]
                if regs:
                    out["shuttle_regret_mean"] = float(np.mean(regs))
                    out["frac_beating_shuttle"] = float(np.mean([g < 0 for g in regs]))
            return out

    def snapshot_levels(self, k: int = 16) -> List[Level]:
        """The k highest-scoring levels (for pictures)."""
        with self._lock:
            recs = sorted(self._records.values(), key=self.raw_score, reverse=True)
            return [r.level for r in recs[:k]]

    # ---------------------------------------------------------- persistence

    def state_dict(self) -> Dict:
        with self._lock:
            return {
                "version": 1, "episode": self.episode, "n_bred": self.n_bred,
                "n_evicted": self.n_evicted, "n_mutation_failures": self.n_mutation_failures,
                "n_inserted": self.n_inserted,
                "records": [{
                    "level": r.level.to_json(),
                    "evac_times": list(r.evac_times), "evac_episodes": list(r.evac_episodes),
                    "regrets": list(r.regrets), "ref_freeflow": r.ref_freeflow,
                    "ref_shuttle": r.ref_shuttle, "maxmc": r.maxmc, "pvl": r.pvl,
                    "best_return": r.best_return, "trials": r.trials, "visits": r.visits,
                    "children": r.children, "last_update_episode": r.last_update_episode,
                    "inserted_at": r.inserted_at, "inherited_score": r.inherited_score,
                } for r in self._records.values()],
            }

    def load_state_dict(self, st: Dict) -> None:
        with self._lock:
            self._records.clear()
            self.episode = int(st.get("episode", 0))
            self.n_bred = int(st.get("n_bred", 0))
            self.n_evicted = int(st.get("n_evicted", 0))
            self.n_mutation_failures = int(st.get("n_mutation_failures", 0))
            self.n_inserted = int(st.get("n_inserted", 0))
            top = 0
            for d in st.get("records", []):
                lv = Level.from_json(d["level"])
                rec = LevelRecord(level=lv, history=int(self.cfg.UED_TRIAL_HISTORY))
                rec.evac_times.extend(d["evac_times"])
                rec.evac_episodes.extend(d["evac_episodes"])
                rec.regrets.extend(d.get("regrets", []))
                for k in ("ref_freeflow", "ref_shuttle", "maxmc", "pvl", "best_return"):
                    setattr(rec, k, d.get(k))
                for k in ("trials", "visits", "children", "last_update_episode", "inserted_at"):
                    setattr(rec, k, int(d.get(k, 0)))
                rec.inherited_score = float(d.get("inherited_score", 0.0))
                self._records[lv.level_id] = rec
                top = max(top, lv.level_id)
            reserve_level_ids(top + 1)
