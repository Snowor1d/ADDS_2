"""Glue between the training loop and the level population.

Everything the main process needs for the curriculum lives here so that
ADDS_AS_reinforcement.py only calls four methods: `dispatch` to keep the
workers fed, `on_transition` and `on_episode` to fold results back in, and
`stats` for logging.

Level production runs on its own thread. Generating a map takes 0.8-2.1 s on
average and up to 8.6 s at the top difficulty, so doing it on the path that
feeds a worker would stall rollouts.
"""

from __future__ import annotations

import os
import pickle
import random
import threading
import time
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

import numpy as np

from config import (
    MAX_STEPS,
    UED_ENABLED,
    UED_LEVEL_QUEUE_SIZE,
    UED_METHOD,
    UED_PRODUCER_TARGET,
    UED_REPLAY_ONLY_UPDATES,
    UED_SCORE,
    UED_SCORE_MAX_EPSILON,
    UED_WARMUP_EPISODES,
)
from ued.level import generate_random_level
from ued.population import LevelPopulation

# How many states of one episode are kept for the MaxMC estimate. A full
# episode is up to MAX_STEPS/ACTION_SCALE transitions and each carries two map
# stacks, so keeping all of them for every in-flight worker would cost
# hundreds of megabytes. A few dozen samples are enough for a mean.
MAXMC_SAMPLES = 64

# Guard against trajectories that never report a terminal transition, which
# happens whenever a worker is killed mid-episode.
MAX_TRACKED_EPISODES = 64

# Minimum gap between queue top-ups. Episodes here last thousands of steps, so
# there is no benefit to refilling more often than this.
DISPATCH_INTERVAL_SEC = 0.5


class _EpisodeTrace:
    __slots__ = ("ego", "glob", "robot", "n_seen", "ret", "level_id", "touched")

    def __init__(self):
        self.ego: List[np.ndarray] = []
        self.glob: List[np.ndarray] = []
        self.robot: List[np.ndarray] = []
        self.n_seen = 0
        self.ret = 0.0
        self.level_id = -1
        self.touched = time.time()


class UEDRunner:
    def __init__(
        self,
        value_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.enabled = bool(UED_ENABLED)
        self.rng = rng or random.Random()
        self.population = LevelPopulation(rng=self.rng)
        self.value_fn = value_fn

        self._fresh: List = []
        self._fresh_lock = threading.Lock()
        self._stop = threading.Event()
        self._producer: Optional[threading.Thread] = None

        self._traces: "OrderedDict[tuple, _EpisodeTrace]" = OrderedDict()
        self._level_best_return: Dict[int, float] = {}
        self._global_best_return: Optional[float] = None

        self.n_dispatched_new = 0
        self.n_dispatched_replay = 0
        self.n_skipped_transitions = 0
        self._last_dispatch = 0.0
        # Exploration rate of the policy that produced the episodes arriving
        # now. Set by the training loop; until it falls, outcomes say more
        # about the random action distribution than about the policy.
        self.epsilon = 1.0
        self.n_unscored_episodes = 0

    # -- producer --------------------------------------------------------

    def start(self):
        if not self.enabled or self._producer is not None:
            return
        self._producer = threading.Thread(target=self._produce, daemon=True, name="ued-producer")
        self._producer.start()

    def stop(self):
        self._stop.set()

    def _produce(self):
        while not self._stop.is_set():
            with self._fresh_lock:
                have = len(self._fresh)
            if have >= UED_PRODUCER_TARGET:
                time.sleep(0.5)
                continue
            try:
                level = generate_random_level(self.rng)
            except Exception as e:
                print(f"[UED] level generation failed: {e}")
                time.sleep(1.0)
                continue
            with self._fresh_lock:
                self._fresh.append(level)

    def _take_fresh(self):
        with self._fresh_lock:
            return self._fresh.pop() if self._fresh else None

    # -- dispatch --------------------------------------------------------

    def dispatch(self, level_queues, global_episode: int):
        """Top up each worker's level queue.

        Called from the main loop; never blocks. A worker whose queue stays
        empty simply falls back to the ordinary random-map path.
        """
        if not self.enabled or not level_queues:
            return
        self.population.episode = max(self.population.episode, int(global_episode))

        # The main loop calls this once per consumed transition, which is far
        # more often than a worker can start an episode. Each pass probes every
        # worker queue, so throttling keeps it off the hot path.
        now = time.time()
        if now - self._last_dispatch < DISPATCH_INTERVAL_SEC:
            return
        self._last_dispatch = now

        for lq in level_queues:
            if lq is None:
                continue
            for _ in range(UED_LEVEL_QUEUE_SIZE):
                if lq.full():
                    break
                level = self._next_level()
                if level is None:
                    break
                try:
                    lq.put_nowait(level)
                except Exception:
                    break

    def _next_level(self):
        """Choose between replaying a scored level and running a fresh one."""
        # During warm-up every level is fresh, so the population accumulates a
        # spread of scored levels before the curriculum starts steering toward
        # any of them. Steering on two or three scored levels would just lock
        # onto whichever happened to be measured first.
        if self.population.episode < UED_WARMUP_EPISODES:
            replay_ok = False
        else:
            replay_ok = UED_METHOD == "accel" and len(self.population) > 0
        if replay_ok and self.rng.random() > self.population.p_new():
            level = self.population.sample()
            if level is not None:
                level.is_replay = True
                self.n_dispatched_replay += 1
                return level

        level = self._take_fresh()
        if level is None:
            return None
        level.is_replay = False
        # A fresh level joins the population so its outcome can be scored and,
        # if it turns out to sit at the learning frontier, bred from.
        self.population.add(level)
        self.n_dispatched_new += 1
        return level

    # -- transition accounting -------------------------------------------

    def should_store(self, msg) -> bool:
        """Whether this transition goes into the SAC replay buffer.

        Robust PLR trains only on replayed levels. That rule cannot hold
        exactly against a buffer that mixes the whole training history, so it
        is an experiment axis rather than a default: see UED_REPLAY_ONLY_UPDATES.
        """
        if not self.enabled or not UED_REPLAY_ONLY_UPDATES:
            return True
        if getattr(msg, "level_id", -1) < 0:
            return True
        if msg.is_replay:
            return True
        self.n_skipped_transitions += 1
        return False

    def on_transition(self, msg):
        """Reservoir-sample states so MaxMC can be computed when the episode ends."""
        if not self.enabled or self.value_fn is None:
            return
        if UED_SCORE not in ("maxmc", "hybrid"):
            return
        key = getattr(msg, "episode_key", ())
        if not key or getattr(msg, "level_id", -1) < 0:
            return

        trace = self._traces.get(key)
        if trace is None:
            trace = _EpisodeTrace()
            trace.level_id = int(msg.level_id)
            self._traces[key] = trace
            while len(self._traces) > MAX_TRACKED_EPISODES:
                self._traces.popitem(last=False)

        trace.touched = time.time()
        trace.ret += float(msg.reward)
        trace.n_seen += 1
        if len(trace.ego) < MAXMC_SAMPLES:
            trace.ego.append(msg.ego_state)
            trace.glob.append(msg.global_state)
            trace.robot.append(msg.robot_state)
        else:
            # Reservoir replacement keeps the sample uniform over the episode
            # rather than biased to its opening steps.
            j = self.rng.randrange(trace.n_seen)
            if j < MAXMC_SAMPLES:
                trace.ego[j] = msg.ego_state
                trace.glob[j] = msg.global_state
                trace.robot[j] = msg.robot_state

    def _consume_trace(self, worker_id: int, episode_idx: int):
        return self._traces.pop((int(worker_id), int(episode_idx)), None)

    def _maxmc(self, trace: _EpisodeTrace) -> Optional[float]:
        """mean_t max(R_best - V(s_t), 0) over a sample of the trajectory.

        The reference return is the best seen anywhere so far, not just on this
        level. PLR takes it per level, which works there because every level in
        the buffer is replayed many times; here MaxMC exists specifically to
        score a level on its first episode, and per level the reference would
        then equal that same episode's own return, collapsing the score to a
        one-sided value error that is not comparable across levels. Scores are
        ranked against each other, so they have to share a reference point.
        """
        if trace is None or not trace.ego or self.value_fn is None:
            return None
        level_best = max(self._level_best_return.get(trace.level_id, trace.ret), trace.ret)
        self._level_best_return[trace.level_id] = level_best
        self._global_best_return = (
            trace.ret if self._global_best_return is None
            else max(self._global_best_return, trace.ret)
        )
        best = max(level_best, self._global_best_return)
        try:
            values = self.value_fn(
                np.asarray(trace.ego, dtype=np.float32),
                np.asarray(trace.glob, dtype=np.float32),
                np.asarray(trace.robot, dtype=np.float32),
            )
        except Exception as e:
            print(f"[UED] value estimate failed: {e}")
            return None
        gaps = np.maximum(best - np.asarray(values, dtype=np.float64), 0.0)
        return float(gaps.mean())

    # -- episode accounting ----------------------------------------------

    def on_episode(self, s_msg) -> None:
        """Fold an episode outcome into the population and maybe breed."""
        if not self.enabled:
            return
        trace = self._consume_trace(s_msg.worker_id, s_msg.episode_idx)
        if getattr(s_msg, "level_id", -1) < 0 or s_msg.abnormal == 1:
            return

        # While most actions are still random, an episode's outcome describes
        # the random action distribution rather than the policy, and a success
        # rate built from it picks breeding parents for the wrong reason. The
        # episode has already trained the agent by this point; only the
        # curriculum's view of the level is withheld.
        if self.epsilon > UED_SCORE_MAX_EPSILON:
            self.n_unscored_episodes += 1
            return

        # The raw time goes in, not a verdict. Whether it counts as a success
        # is decided against the level's own free-flow estimate, so the
        # criterion scales with the level instead of with MAX_STEPS.
        maxmc = self._maxmc(trace) if UED_SCORE in ("maxmc", "hybrid") else None

        rec = self.population.update(
            s_msg.level_id,
            evac_time=float(s_msg.evac_time_100),
            freeflow_steps=getattr(s_msg, "freeflow_steps", None),
            maxmc=maxmc,
            episode=self.population.episode,
        )
        if rec is None:
            return
        if self.population.should_breed(rec):
            self.population.breed(rec)

    # -- persistence -----------------------------------------------------

    def save(self, path: str) -> bool:
        """Write the curriculum state so a restart resumes it.

        Start_training.py kills and relaunches the trainer after 2000 s of
        stalled progress, and the policy resumes from its own checkpoint. If
        the population did not resume with it, the curriculum would restart
        from nothing while the agent kept its skill, which over a long run
        quietly turns ACCEL back into domain randomisation.

        Written to a temporary file and renamed, so a crash during the write
        leaves the previous checkpoint intact rather than a truncated one.
        """
        if not self.enabled:
            return False
        state = {
            "version": 1,
            "population": self.population.state_dict(),
            "level_best_return": dict(self._level_best_return),
            "global_best_return": self._global_best_return,
            "n_dispatched_new": self.n_dispatched_new,
            "n_dispatched_replay": self.n_dispatched_replay,
            "n_skipped_transitions": self.n_skipped_transitions,
        }
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"[UED] could not save curriculum state: {e}")
            try:
                os.remove(tmp)
            except OSError:
                pass
            return False

    def load(self, path: str) -> bool:
        if not self.enabled or not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.population.load_state_dict(state["population"])
            self._level_best_return = dict(state.get("level_best_return", {}))
            self._global_best_return = state.get("global_best_return")
            self.n_dispatched_new = int(state.get("n_dispatched_new", 0))
            self.n_dispatched_replay = int(state.get("n_dispatched_replay", 0))
            self.n_skipped_transitions = int(state.get("n_skipped_transitions", 0))
        except Exception as e:
            # A corrupt curriculum file must not stop training; starting the
            # population over is bad but recoverable, crashing on boot is not.
            print(f"[UED] could not load curriculum state ({e}); starting fresh")
            return False
        print(
            f"[UED] resumed curriculum: {len(self.population)} levels, "
            f"episode={self.population.episode}, bred={self.population.n_bred}"
        )
        return True

    # -- snapshots -------------------------------------------------------

    def snapshot(self, writer, episode: int, save_dir: Optional[str] = None) -> bool:
        """Push pictures of the population into TensorBoard.

        The scalars already say whether mean density and generation are
        rising; these say what the levels actually look like while it happens.
        TensorBoard keeps one image per step under a tag, so its step slider
        becomes a way to scrub through the curriculum's evolution.
        """
        if not self.enabled:
            return False
        try:
            return self._snapshot_inner(writer, episode, save_dir)
        except Exception as e:
            # Pictures are diagnostics. Losing them is an inconvenience;
            # losing the run because a figure failed to draw is not acceptable,
            # and an unguarded add_image already killed one run this way.
            print(f"[UED] snapshot failed at episode {episode}: {e}")
            return False

    def _snapshot_inner(self, writer, episode: int, save_dir: Optional[str]) -> bool:
        from ued import render

        wrote = False

        levels, titles = render.population_sample(self.population, n=12, top=True)
        if levels:
            img = render.render_grid(
                levels, titles=titles, cols=4,
                suptitle=f"top-scoring levels @ episode {episode}",
            )
            if img is not None:
                self._add_image(writer, "UED/population_top", img, episode)
                wrote = True
                if save_dir:
                    self._write_png(save_dir, f"population_top_ep{episode}.png", img)

        best_id = render.best_scored_level_id(self.population)
        if best_id is not None:
            chain, chain_titles = render.lineage(self.population, best_id, max_depth=8)
            # A chain of one is just the level again; only worth a figure once
            # editing has actually produced a descendant.
            if len(chain) > 1:
                img = render.render_grid(
                    chain, titles=chain_titles, cols=len(chain), tile_inches=1.4,
                    suptitle=f"lineage of level {best_id} @ episode {episode}",
                )
                if img is not None:
                    self._add_image(writer, "UED/lineage_best", img, episode)
                    wrote = True
                    if save_dir:
                        self._write_png(save_dir, f"lineage_ep{episode}.png", img)

        if wrote:
            writer.flush()
        return wrote

    @staticmethod
    def _add_image(writer, tag: str, img, step: int) -> None:
        """Log an image without going through torch's add_image.

        torch 2.5's tensorboard helper reaches for PIL.Image.Resampling, which
        only exists from Pillow 9.1; this machine has 9.0.1, so add_image
        raises. The image is already a finished RGB array that needs no
        resizing, so the PNG bytes go straight into the summary proto and the
        Pillow path is skipped entirely.
        """
        import imageio.v2 as imageio
        from tensorboard.compat.proto.summary_pb2 import Summary

        png = imageio.imwrite("<bytes>", img, format="png")
        proto = Summary.Image(
            height=int(img.shape[0]),
            width=int(img.shape[1]),
            colorspace=3,
            encoded_image_string=png,
        )
        writer._get_file_writer().add_summary(
            Summary(value=[Summary.Value(tag=tag, image=proto)]), step
        )

    @staticmethod
    def _write_png(save_dir: str, name: str, img) -> None:
        try:
            import imageio.v2 as imageio

            os.makedirs(save_dir, exist_ok=True)
            imageio.imwrite(os.path.join(save_dir, name), img)
        except Exception as e:
            print(f"[UED] could not write snapshot {name}: {e}")

    # -- logging ---------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        out = dict(self.population.stats())
        with self._fresh_lock:
            out["fresh_pool"] = len(self._fresh)
        out["dispatched_new"] = self.n_dispatched_new
        out["dispatched_replay"] = self.n_dispatched_replay
        out["skipped_transitions"] = self.n_skipped_transitions
        out["epsilon_seen"] = self.epsilon
        out["scoring_active"] = 1.0 if self.epsilon <= UED_SCORE_MAX_EPSILON else 0.0
        out["unscored_episodes"] = self.n_unscored_episodes
        # The realised fresh fraction, which can fall below p_new whenever the
        # producer cannot keep a level ready; worth watching directly rather
        # than inferring from the two counters.
        total_dispatched = self.n_dispatched_new + self.n_dispatched_replay
        if total_dispatched:
            out["fresh_fraction"] = self.n_dispatched_new / total_dispatched
        if self._global_best_return is not None:
            out["best_return_seen"] = self._global_best_return
        corr = self.population.score_correlation()
        if corr is not None:
            out["spearman_maxmc_learnability"] = corr
        return out


def make_value_fn(agent):
    """V(s) = min(Q1, Q2)(s, a~pi) - alpha * log pi(a|s), from the SAC critic.

    This is what makes a regret-style score available off-policy at all: the
    GAE-based proxy the original methods use needs on-policy rollouts, but
    MaxMC only needs a value estimate, and SAC has the pieces for one.
    """
    import torch

    def value_fn(ego: np.ndarray, glob: np.ndarray, robot: np.ndarray) -> np.ndarray:
        device = agent.device
        with torch.no_grad():
            ego_t = torch.as_tensor(ego, dtype=torch.float32, device=device)
            glob_t = torch.as_tensor(glob, dtype=torch.float32, device=device)
            robot_t = torch.as_tensor(robot, dtype=torch.float32, device=device)
            action, log_prob = agent.policy.sample_action(ego_t, glob_t, robot_t)
            q1 = agent.q1(ego_t, glob_t, action, robot_t).squeeze(-1)
            q2 = agent.q2(ego_t, glob_t, action, robot_t).squeeze(-1)
            alpha = agent.alpha.to(device) if hasattr(agent.alpha, "to") else agent.alpha
            v = torch.min(q1, q2) - alpha * log_prob.squeeze(-1)
            return v.detach().cpu().numpy()

    return value_fn
