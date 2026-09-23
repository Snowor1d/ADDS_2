"""The level population: scoring, sampling, breeding and eviction.

Lives in the main process only. Workers never see it; they receive plain
`Level` objects over their queue and send episode outcomes back.

Two score functions are supported because they fail in opposite ways here.
Learnability, p*(1-p) over a level's success history, targets the levels the
agent solves inconsistently, but it needs several episodes per level to
estimate p at all, and episodes in this environment run up to MAX_STEPS steps
of social-force simulation with a full crowd. MaxMC scores a single episode
from the SAC critic, so it covers a freshly bred level's cold start, but it
inherits the criticism that regret proxies track success rate rather than
regret. The hybrid uses MaxMC until a level has enough trials, then switches.
"""

from __future__ import annotations

import math
import random
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    MAX_STEPS,
    UED_METHOD,
    UED_MAX_CHILDREN_PER_LEVEL,
    UED_MIN_TRIALS,
    UED_MUTATE_THRESHOLD,
    UED_MUTATIONS_PER_CHILD,
    UED_P_NEW_DECAY_EPISODES,
    UED_P_NEW_END,
    UED_P_NEW_START,
    UED_POP_SIZE,
    UED_SCORE,
    UED_SCORE_TEMPERATURE,
    UED_STALENESS_COEF,
    UED_SCORE_MAX_EPSILON,
    UED_SUCCESS_K,
    UED_TRIAL_HISTORY,
    UED_TRIAL_MAX_AGE,
)
from ued.level import Level, generate_random_level
from ued.mutate import MutationFailed, mutate_level

# Beta(1,1) smoothing, so a single lucky episode cannot pin a level's success
# rate at 0 or 1 and hand it a learnability score of zero.
_BETA_PRIOR = 1.0


@dataclass
class LevelRecord:
    level: Level
    # Raw evacuation times, not success bits. Storing the times means the
    # success threshold can be changed after the fact and the whole history
    # re-scored, instead of being frozen the moment each episode ended. Each
    # entry keeps the episode it came from, so stale outcomes can be dropped
    # rather than averaged in forever.
    evac_times: deque = field(default_factory=lambda: deque(maxlen=UED_TRIAL_HISTORY))
    evac_episodes: deque = field(default_factory=lambda: deque(maxlen=UED_TRIAL_HISTORY))
    freeflow_steps: Optional[float] = None
    maxmc: Optional[float] = None
    trials: int = 0
    visits: int = 0
    # How many descendants this level has produced, capped so no single
    # lineage can take over the population.
    children: int = 0
    last_update_episode: int = 0
    inserted_at: int = 0
    inherited_score: float = 0.0

    def threshold(self, k: float = UED_SUCCESS_K) -> float:
        """Steps within which this level counts as solved.

        A multiple of the level's own free-flow estimate. Falls back to the
        global cap only when no estimate has arrived yet, which happens for a
        level that has not run an episode.
        """
        if self.freeflow_steps is None or self.freeflow_steps <= 0:
            return float(MAX_STEPS)
        return min(float(MAX_STEPS), float(k) * float(self.freeflow_steps))

    def fresh_times(self, now_episode: Optional[int] = None) -> List[float]:
        """Outcomes recent enough to describe the current policy."""
        if now_episode is None or UED_TRIAL_MAX_AGE <= 0 or not self.evac_episodes:
            return list(self.evac_times)
        cutoff = int(now_episode) - int(UED_TRIAL_MAX_AGE)
        return [
            t for t, e in zip(self.evac_times, self.evac_episodes) if e >= cutoff
        ]

    def successes(self, k: float = UED_SUCCESS_K,
                  now_episode: Optional[int] = None) -> int:
        limit = self.threshold(k)
        # A run that hit the hard timeout never finished evacuating, so it is a
        # failure whatever the threshold says.
        return sum(1 for t in self.fresh_times(now_episode) if t < limit and t < MAX_STEPS)

    @property
    def evac_ratio(self) -> Optional[float]:
        """Mean actual time over free-flow time, the quantity K is chosen from."""
        if not self.evac_times or not self.freeflow_steps:
            return None
        finished = [t for t in self.evac_times if t < MAX_STEPS]
        if not finished:
            return None
        return sum(finished) / (len(finished) * float(self.freeflow_steps))

    def success_rate_at(self, now_episode: Optional[int] = None) -> float:
        fresh = self.fresh_times(now_episode)
        n = len(fresh)
        s = self.successes(now_episode=now_episode)
        return (s + _BETA_PRIOR) / (n + 2.0 * _BETA_PRIOR)

    def learnability_at(self, now_episode: Optional[int] = None) -> float:
        p = self.success_rate_at(now_episode)
        return p * (1.0 - p)

    def fresh_trials(self, now_episode: Optional[int] = None) -> int:
        """Trials that still count, which is what the breeding bar uses."""
        return len(self.fresh_times(now_episode))

    @property
    def success_rate(self) -> float:
        return self.success_rate_at(None)

    @property
    def learnability(self) -> float:
        return self.learnability_at(None)


class LevelPopulation:
    """Score-prioritised level buffer with ACCEL-style breeding.

    Thread-safe: the producer thread inserts freshly generated levels while the
    main training loop samples and updates.
    """

    def __init__(
        self,
        capacity: int = UED_POP_SIZE,
        score_fn: str = UED_SCORE,
        rho: float = UED_STALENESS_COEF,
        rng: Optional[random.Random] = None,
    ):
        self.capacity = int(capacity)
        self.score_fn = score_fn
        self.rho = float(rho)
        self.rng = rng or random.Random()
        self._records: Dict[int, LevelRecord] = {}
        self._lock = threading.RLock()
        self.episode = 0
        self.n_bred = 0
        self.n_evicted = 0
        self.n_mutation_failures = 0

    # -- basic container -------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def add(self, level: Level, inherited_score: float = 0.0) -> None:
        with self._lock:
            rec = LevelRecord(
                level=level,
                last_update_episode=self.episode,
                inserted_at=self.episode,
                inherited_score=float(inherited_score),
            )
            self._records[level.level_id] = rec
            if len(self._records) > self.capacity:
                self._evict_locked()

    def _evict_locked(self) -> None:
        # Levels that have never run have no score of their own, so evicting on
        # score would just discard whatever was inserted most recently. Only
        # levels with at least one trial are eligible; if none are, fall back to
        # the oldest insertion.
        tried = [r for r in self._records.values() if r.trials > 0]
        if tried:
            victim = min(tried, key=lambda r: self._raw_score(r))
        else:
            victim = min(self._records.values(), key=lambda r: r.inserted_at)
        self._records.pop(victim.level.level_id, None)
        self.n_evicted += 1

    # -- scoring ---------------------------------------------------------

    def _raw_score(self, rec: LevelRecord) -> float:
        fresh = rec.fresh_trials(self.episode)
        if self.score_fn == "learnability":
            return rec.learnability_at(self.episode) if fresh > 0 else rec.inherited_score
        if self.score_fn == "maxmc":
            return rec.maxmc if rec.maxmc is not None else rec.inherited_score
        if self.score_fn == "failure_rate":
            return (1.0 - rec.success_rate_at(self.episode)) if fresh > 0 else rec.inherited_score
        # hybrid
        if fresh >= UED_MIN_TRIALS:
            return rec.learnability_at(self.episode)
        if rec.maxmc is not None:
            return rec.maxmc
        return rec.inherited_score

    def _ranked_scores(self, records: Sequence[LevelRecord]) -> List[float]:
        """Rank-normalise raw scores so learnability and MaxMC share an axis.

        The two score functions have different units and the hybrid mixes them
        within one population, so absolute values are not comparable. Ranking
        also makes sampling insensitive to score outliers.
        """
        n = len(records)
        if n == 0:
            return []
        order = sorted(range(n), key=lambda i: self._raw_score(records[i]), reverse=True)
        weights = [0.0] * n
        beta = max(1e-6, float(UED_SCORE_TEMPERATURE))
        for rank, idx in enumerate(order, start=1):
            weights[idx] = 1.0 / (rank ** (1.0 / beta))
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else [1.0 / n] * n

    def _staleness_weights(self, records: Sequence[LevelRecord]) -> List[float]:
        raw = [max(1.0, float(self.episode - r.last_update_episode)) for r in records]
        total = sum(raw)
        n = len(records)
        return [x / total for x in raw] if total > 0 else [1.0 / n] * n

    # -- sampling --------------------------------------------------------

    def p_new(self) -> float:
        """Probability of running a fresh level instead of replaying one."""
        if UED_METHOD == "dr":
            return 1.0
        span = max(1, int(UED_P_NEW_DECAY_EPISODES))
        t = min(1.0, self.episode / span)
        return UED_P_NEW_START + (UED_P_NEW_END - UED_P_NEW_START) * t

    def sample(self) -> Optional[Level]:
        """Draw a level to train on, mixing score priority with staleness.

        Staleness matters more here than in the original setting: workers run
        asynchronously, so a level's score can be many episodes out of date by
        the time it is considered again.
        """
        with self._lock:
            records = list(self._records.values())
            if not records:
                return None
            if UED_METHOD == "dr":
                rec = self.rng.choice(records)
                rec.visits += 1
                return rec.level

            score_w = self._ranked_scores(records)
            stale_w = self._staleness_weights(records)
            mixed = [
                (1.0 - self.rho) * s + self.rho * c for s, c in zip(score_w, stale_w)
            ]
            rec = self.rng.choices(records, weights=mixed, k=1)[0]
            rec.visits += 1
            return rec.level

    # -- updates ---------------------------------------------------------

    def update(
        self,
        level_id: int,
        evac_time: float,
        freeflow_steps: Optional[float] = None,
        maxmc: Optional[float] = None,
        episode: Optional[int] = None,
    ) -> Optional[LevelRecord]:
        """Fold one episode outcome into a level's score.

        Takes the raw evacuation time rather than a success flag, so whether it
        counts as a success is decided by the level's own threshold at read
        time and can be revisited later.
        """
        with self._lock:
            if episode is not None:
                self.episode = max(self.episode, int(episode))
            rec = self._records.get(level_id)
            if rec is None:
                return None
            if freeflow_steps is not None and freeflow_steps > 0:
                # Level-intrinsic and constant, so the first episode's value
                # stands for the rest.
                if rec.freeflow_steps is None:
                    rec.freeflow_steps = float(freeflow_steps)
            rec.evac_times.append(float(evac_time))
            rec.evac_episodes.append(int(self.episode))
            rec.trials += 1
            if maxmc is not None:
                # Smooth so one noisy trajectory does not dominate a level that
                # has already been measured.
                rec.maxmc = float(maxmc) if rec.maxmc is None else 0.7 * rec.maxmc + 0.3 * float(maxmc)
            rec.last_update_episode = self.episode
            return rec

    def should_breed(self, rec: LevelRecord) -> bool:
        """Whether this level is worth editing for a descendant.

        Two things the naive version got wrong for this task.

        The comparison is by rank, not by raw score. In hybrid mode a level
        with few trials is scored by MaxMC and a well-measured one by
        learnability, and those are different units: learnability is bounded by
        0.25 while MaxMC is an unbounded return gap. Comparing them directly
        against a shared median put essentially every barely-tried level above
        the bar and almost no well-measured one, which is the opposite of the
        intent.

        And a level must have been measured on its own before it breeds.
        Children start on their parent's score so they get sampled soon, so
        breeding from an unmeasured level means breeding from a score that was
        inherited rather than observed; a single lucky ancestor could otherwise
        fill the population with one lineage.
        """
        if UED_METHOD != "accel":
            return False
        if getattr(rec.level, "plan", None) is None:
            # Nothing to edit. Mutation works on the street plan, and a level
            # without one, a real OSM crop or a numbered map, can only fail.
            # Saying no here rather than letting breed() raise matters because
            # a failed breed leaves the child count untouched, so the level
            # would qualify again on every future visit and burn an attempt
            # each time, for as long as it stayed in the population.
            return False
        with self._lock:
            if rec.fresh_trials(self.episode) < UED_MIN_TRIALS:
                return False
            if rec.children >= UED_MAX_CHILDREN_PER_LEVEL:
                return False

            records = [r for r in self._records.values()
                       if r.fresh_trials(self.episode) >= UED_MIN_TRIALS]
            if len(records) < 8:
                return False

            my_score = self._raw_score(rec)
            n_below = sum(1 for r in records if self._raw_score(r) <= my_score)
            percentile = n_below / len(records)
            return percentile >= UED_MUTATE_THRESHOLD

    def breed(self, rec: LevelRecord) -> Optional[Level]:
        """Produce and insert one edited descendant of a high-scoring level."""
        try:
            child = mutate_level(rec.level, rng=self.rng)
        except MutationFailed:
            self.n_mutation_failures += 1
            return None
        # The child starts on its parent's score so it is sampled soon; its own
        # first episode replaces that with a real MaxMC measurement.
        self.add(child, inherited_score=self._raw_score(rec))
        rec.children += 1
        self.n_bred += 1
        return child

    # -- diagnostics -----------------------------------------------------

    def stats(self) -> Dict[str, float]:
        with self._lock:
            records = list(self._records.values())
            if not records:
                return {"pop_size": 0}
            tried = [r for r in records if r.trials > 0]
            densities = [r.level.density() for r in records]
            gens = [r.level.generation for r in records]
            n_obs = [len(r.level.obstacles) for r in records]
            crowds = [r.level.crowd_size for r in records]
            out = {
                "pop_size": len(records),
                "n_scored": len(tried),
                "mean_density": sum(densities) / len(densities),
                "mean_generation": sum(gens) / len(gens),
                "max_generation": max(gens),
                "mean_n_obstacles": sum(n_obs) / len(n_obs),
                "mean_crowd_size": sum(crowds) / len(crowds),
                "n_bred": self.n_bred,
                "n_evicted": self.n_evicted,
                "n_mutation_failures": self.n_mutation_failures,
                "p_new": self.p_new(),
            }
            if tried:
                out["mean_success_rate"] = sum(r.success_rate for r in tried) / len(tried)
                # The quantity UED_SUCCESS_K should be chosen from: how many
                # times its free-flow estimate a level actually takes.
                ratios = [r.evac_ratio for r in tried if r.evac_ratio is not None]
                if ratios:
                    out["mean_evac_ratio"] = sum(ratios) / len(ratios)
                    out["n_finished_levels"] = len(ratios)
                ff = [r.freeflow_steps for r in tried if r.freeflow_steps]
                if ff:
                    out["mean_freeflow_steps"] = sum(ff) / len(ff)
                out["mean_learnability"] = sum(r.learnability for r in tried) / len(tried)
                out["mean_staleness"] = sum(
                    self.episode - r.last_update_episode for r in tried
                ) / len(tried)
                mm = [r.maxmc for r in tried if r.maxmc is not None]
                if mm:
                    out["mean_maxmc"] = sum(mm) / len(mm)
            return out

    # -- persistence -----------------------------------------------------

    def state_dict(self) -> Dict:
        """Everything needed to resume the curriculum where it left off.

        Without this the watchdog's restart in Start_training.py silently
        resets the curriculum while the policy resumes from its checkpoint,
        which degrades ACCEL to domain randomisation over a long run.
        """
        with self._lock:
            return {
                # 2: records hold raw evacuation times and a free-flow
                # reference instead of success bits.
                "version": 3,
                "episode": self.episode,
                "n_bred": self.n_bred,
                "n_evicted": self.n_evicted,
                "n_mutation_failures": self.n_mutation_failures,
                "records": [
                    {
                        "level": rec.level,
                        "evac_times": list(rec.evac_times),
                        "evac_episodes": list(rec.evac_episodes),
                        "freeflow_steps": rec.freeflow_steps,
                        "maxmc": rec.maxmc,
                        "trials": rec.trials,
                        "visits": rec.visits,
                        "children": rec.children,
                        "last_update_episode": rec.last_update_episode,
                        "inserted_at": rec.inserted_at,
                        "inherited_score": rec.inherited_score,
                    }
                    for rec in self._records.values()
                ],
            }

    def load_state_dict(self, state: Dict) -> None:
        from ued.level import reserve_level_ids

        version = int(state.get("version", 1))
        if version < 2:
            # v1 stored success bits against the old MAX_STEPS threshold, which
            # cannot be re-scored against a level's free-flow reference. Better
            # to start the population over than to carry scores that mean
            # something different from the ones being written now.
            raise ValueError(
                f"curriculum state version {version} predates the normalised "
                "success criterion and cannot be re-scored"
            )
        if version < 3:
            # v2 levels were pickled by the scatter and street-map generators
            # and carry no street plan, so none of them can be bred from. A
            # restored v2 population would sit there unmutatable and quietly
            # turn ACCEL back into domain randomisation, which is exactly the
            # failure the persistence was added to prevent.
            raise ValueError(
                f"curriculum state version {version} holds levels from the "
                "generators that citygen replaced; they carry no street plan "
                "and cannot be mutated"
            )

        with self._lock:
            self._records.clear()
            self.episode = int(state.get("episode", 0))
            self.n_bred = int(state.get("n_bred", 0))
            self.n_evicted = int(state.get("n_evicted", 0))
            self.n_mutation_failures = int(state.get("n_mutation_failures", 0))

            max_id = 0
            for d in state.get("records", []):
                level = d["level"]
                rec = LevelRecord(
                    level=level,
                    freeflow_steps=d.get("freeflow_steps"),
                    maxmc=d.get("maxmc"),
                    trials=int(d.get("trials", 0)),
                    visits=int(d.get("visits", 0)),
                    children=int(d.get("children", 0)),
                    last_update_episode=int(d.get("last_update_episode", 0)),
                    inserted_at=int(d.get("inserted_at", 0)),
                    inherited_score=float(d.get("inherited_score", 0.0)),
                )
                rec.evac_times.extend(d.get("evac_times", []))
                rec.evac_episodes.extend(d.get("evac_episodes", []))
                self._records[level.level_id] = rec
                max_id = max(max_id, int(level.level_id))

            # New levels must not reuse a restored id.
            reserve_level_ids(max_id + 1)

    def score_correlation(self) -> Optional[float]:
        """Spearman correlation between MaxMC and learnability.

        This is the diagnostic that decides the score-function question for this
        domain: if the regret proxy really does fail to identify the levels the
        agent solves inconsistently, it shows up here as a weak correlation.
        """
        with self._lock:
            pairs = [
                (r.maxmc, r.learnability)
                for r in self._records.values()
                if r.maxmc is not None and r.trials >= UED_MIN_TRIALS
            ]
        if len(pairs) < 10:
            return None
        return _spearman([p[0] for p in pairs], [p[1] for p in pairs])


def _rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    rx, ry = _rankdata(xs), _rankdata(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)
