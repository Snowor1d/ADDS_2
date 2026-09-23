"""Distributed SAC trainer for outdoor hazard-zone evacuation.

docs/outdoor_madrl_redesign.md. The entry point resolves and validates the
three configuration files once, hands the resulting ResolvedConfig to every
worker, and is the only process that writes metrics (JSONL, TXT, TensorBoard
and W&B from one event).

    worker processes   build a level, run it with learn.rollout.run_episode,
                       send the episode's static layers once and one message
                       per decision instant (the team's DecisionRecord, the
                       action and the per-step rewards), then an episode summary
    main process       stores records in the replay buffer, updates the
                       agent, logs, checkpoints, feeds the curriculum, and
                       starts periodic validation in a separate process

Run from the project root:  python3 -m learn.ADDS_AS_reinforcement
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from learn.frame_stack import FrameStack, FrameStack2          # noqa: F401
from learn.networks import CentralizedCritic, PolicyNetwork     # noqa: F401
from learn.replay import (BatchPrefetcher, ReplayBuffer, SchemaMismatch,
                          StaticStore, check_schema)
from learn.sac import SACAgent, exploration_action, make_value_fn
from sim.robot_action import ACTION_DIM                          # noqa: F401


# ------------------------------------------------------------------ messages

@dataclass
class EpisodeStartMsg:
    worker_id: int
    episode_uid: int
    static_meta: dict
    static_arrays: Dict[str, np.ndarray]


@dataclass
class TransitionMsg:
    worker_id: int
    episode_uid: int
    episode_key: tuple
    step: int
    record: object
    action: Optional[np.ndarray]
    step_rewards: List[float]
    terminal: bool
    static_key: str
    level_id: int = -1
    is_replay: bool = False

    @property
    def reward(self) -> float:
        return float(sum(self.step_rewards))


@dataclass
class EpisodeStatMsg:
    worker_id: int
    episode_idx: int
    episode_uid: int
    level_id: int
    is_replay: bool
    abnormal: int
    metrics: Dict[str, float]
    evac_time_100: int
    freeflow_steps: float = 0.0
    total_reward: float = 0.0
    level_name: str = ""
    video_path: str = ""


def episode_uid(worker_id: int, episode_idx: int, generation: int = 0) -> int:
    return (int(generation) * 1000 + int(worker_id)) * 10_000_000 + int(episode_idx)


def episode_metrics(cfg, res, level, model) -> Dict[str, float]:
    """The per-episode record every sink receives."""
    t = res.task
    m = {
        "episode/total_reward": res.total_reward,
        "episode/hazard_person_steps": t.get("hazard_person_steps"),
        "episode/reentries": t.get("reentries"),
        "episode/reentries_after_clear": t.get("reentries_after_clear"),
        "episode/held_clear_success": t.get("held_clear_success"),
        "episode/evac_time_100": (t.get("held_clear_step")
                                  if t.get("held_clear_step") is not None
                                  else res.steps),
        "episode/first_empty_step": (t.get("first_empty_step")
                                     if t.get("first_empty_step") is not None
                                     else res.steps),
        "episode/mean_occupancy": t.get("mean_occupancy"),
        "episode/outflows": t.get("outflows"),
        "episode/inflows": t.get("inflows"),
        "episode/evacuation_departures": t.get("evacuation_departures"),
        "episode/informed_trip_outflows": t.get("informed_trip_outflows"),
        "episode/background_departures": t.get("background_departures"),
        "episode/stuck_releases": t.get("stuck_releases"),
        "episode/initial_population": t.get("initial_population"),
        "episode/initial_inside": t.get("initial_inside"),
        "episode/actual_density": getattr(level, "crowd_density", None),
        "episode/map_size_m": float(model.width),
        "episode/robot_num": float(len(model.robots)),
        "episode/steps": res.steps,
        "episode/command_norm_mean": res.command_norm_mean,
        "episode/robot_speed_mean": res.robot_speed_mean,
        "episode/mode_switch_rate": res.mode_switch_rate,
        "episode/inference_ms_mean": res.inference_ms_mean,
        "episode/sim_ms_per_step": res.sim_ms_per_step,
        "episode/perceptibility": getattr(level, "perceptibility", None),
        "episode/hazard_area_fraction": (
            float(model.danger_zone.area()) / float(model.width * model.height)
            if getattr(model, "danger_zone", None) is not None else None),
    }
    # Which training crop, as an index into DATASET_SITES (metrics are
    # numbers); the name itself goes to the JSONL record by the main process.
    site = getattr(level, "site_key", None)
    if site is not None and site in tuple(cfg.DATASET_SITES):
        m["episode/site_index"] = float(list(cfg.DATASET_SITES).index(site))
    for k, v in res.components.items():
        m[f"episode/reward/{k}"] = v
    for k, v in res.mode_share.items():
        m[f"episode/mode_share/{k}"] = v
    return {k: v for k, v in m.items() if v is not None}


# ------------------------------------------------------------------- worker

def make_level_source(cfg):
    """A function rng -> level for episodes the curriculum did not supply.

    TRAIN_MAP_SOURCE decides: "dataset" draws real OSM crops, site x size
    uniformly; "ued" generates citygen levels (the curriculum supplies most
    of them through the level queue, this covers the rest)."""
    if cfg.TRAIN_MAP_SOURCE == "dataset":
        from learn.training_maps import OsmTrainingMaps
        maps = OsmTrainingMaps(cfg)
        return maps.sample
    from ued.level import generate_random_level
    return generate_random_level


def worker_process(worker_id: int, cfg, transition_queue, stats_queue,
                   epsilon_shared, param_queue, seed: int = 0,
                   level_queue=None, generation: int = 0,
                   video_request=None, log_dir: str = ""):
    """Roll out episodes forever under the resolved configuration `cfg`."""
    from configs import config_fingerprint_of_modules
    if config_fingerprint_of_modules() != cfg.base_fingerprint:
        raise RuntimeError(
            "the configuration files changed on disk after the run was "
            "resolved; restart the run so every process uses the same values")
    import sim.model as model_module
    from learn.rollout import run_episode
    from learn.zero_shot import EpisodeMetrics
    from sim.observation import build_static_layers

    torch.set_num_threads(1)
    rng = random.Random(seed * 7919 + worker_id)
    np.random.seed((seed + worker_id) % (2 ** 32))
    random.seed(seed + worker_id)
    policy = PolicyNetwork(cfg).eval()
    fresh_level = make_level_source(cfg)
    episode_idx = 0

    def act(obs, rec, eps):
        n = obs["state"].shape[0]
        out = np.zeros((n, ACTION_DIM), np.float32)
        with torch.no_grad():
            t = {k: torch.as_tensor(obs[k]) for k in ("ego", "mid", "glob",
                                                      "state")}
            a, _ = policy.sample_action(t["ego"], t["mid"], t["glob"],
                                        t["state"])
        out[:] = a.numpy()
        for i in range(n):
            # Per robot, so a team can discover a division of labour.
            if rng.random() < eps:
                out[i] = exploration_action(rng)
        return out

    while True:
        level = None
        if level_queue is not None:
            try:
                level = level_queue.get_nowait()
            except Exception:
                level = None
        for _attempt in range(5):
            try:
                if level is None:
                    level = fresh_level(rng)
                model = model_module.FightingModel(
                    int(level.crowd_size), int(level.width),
                    int(level.height), robot="Q", level=level)
                break
            except Exception as exc:
                print(f"[Worker {worker_id}] level build failed: {exc}")
                level = None
        else:
            continue
        uid = episode_uid(worker_id, episode_idx, generation)
        key = (worker_id, episode_idx)
        level_id = int(getattr(level, "level_id", -1))
        is_replay = bool(getattr(level, "is_replay", False))
        abnormal = 0
        try:
            freeflow = float(model.free_flow_evacuation_steps())
        except Exception:
            freeflow = 0.0
        try:
            static = build_static_layers(model, cfg)
            transition_queue.put(EpisodeStartMsg(
                worker_id, uid, static.meta(), static.arrays()))
            try:
                while True:
                    policy.load_state_dict(param_queue.get_nowait())
            except queue.Empty:
                pass
            with epsilon_shared.get_lock():
                eps = float(epsilon_shared.value)

            def emit(tr):
                transition_queue.put(TransitionMsg(
                    worker_id, uid, key, tr.step, tr.record, tr.action,
                    list(tr.step_rewards), bool(tr.terminal), static.key,
                    level_id, is_replay))

            # Claim a pending video request, if any: one worker records the
            # next episode it starts.
            recorder = None
            if video_request is not None:
                with video_request.get_lock():
                    wanted = int(video_request.value)
                    if wanted >= 0:
                        video_request.value = -1
                if wanted >= 0:
                    from learn.episode_video import EpisodeRecorder
                    name = (f"{getattr(level, 'site_key', 'generated')}_"
                            f"{int(level.width)}m")
                    recorder = EpisodeRecorder(
                        cfg, model,
                        os.path.join(log_dir, "videos",
                                     f"ep{wanted:07d}_{name}_w{worker_id}.mp4"),
                        title=f"ep~{wanted} {name} {len(model.robots)} robot(s)")
            tm = EpisodeMetrics().start(model)
            try:
                res = run_episode(model, cfg,
                                  lambda obs, rec: act(obs, rec, eps),
                                  gamma=float(cfg.GAMMA_START),
                                  seed=rng.randrange(1 << 30), static=static,
                                  emit=emit, task_metrics=tm,
                                  on_step=(recorder.step if recorder else None))
            finally:
                video_path = recorder.close() if recorder else None
            metrics = episode_metrics(cfg, res, level, model)
            evac = int(metrics.get("episode/evac_time_100", cfg.MAX_STEPS))
        except Exception as exc:
            print(f"[Worker {worker_id}] episode failed: {exc}")
            traceback.print_exc()
            abnormal, metrics, evac, res = 1, {}, int(cfg.MAX_STEPS), None
            video_path = None
        stats_queue.put(EpisodeStatMsg(
            worker_id, episode_idx, uid, level_id, is_replay, abnormal,
            metrics, evac, freeflow,
            float(res.total_reward) if res is not None else 0.0,
            f"{getattr(level, 'site_key', 'generated')}_{int(level.width)}m",
            video_path or ""))
        episode_idx += 1


# --------------------------------------------------------------- validation

def validation_process(cfg, checkpoint: str, episode: int, out_queue,
                       output_dir: str, off_cache_path: str):
    """Evaluate a frozen policy on the validation levels; report the summary
    to the main process, which is the only one that logs."""
    import pickle
    from learn.zero_shot import run_validation

    torch.set_num_threads(max(1, os.cpu_count() // 8))
    try:
        agent = SACAgent(cfg, device="cpu")
        agent.load(checkpoint, policy_only=True)
        agent.policy.eval()
        off_cache = {}
        if os.path.exists(off_cache_path):
            with open(off_cache_path, "rb") as fh:
                off_cache = pickle.load(fh)
        summary = run_validation(agent, cfg, episode, off_cache=off_cache,
                                 output_dir=output_dir)
        with open(off_cache_path + ".tmp", "wb") as fh:
            pickle.dump(off_cache, fh)
        os.replace(off_cache_path + ".tmp", off_cache_path)
        out_queue.put(("ok", episode, checkpoint, summary))
    except Exception as exc:
        out_queue.put(("error", episode, checkpoint,
                       f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


# ---------------------------------------------------------------- resuming

def _checkpoints(log_dir: str) -> List[Tuple[int, str]]:
    out = []
    for f in os.listdir(log_dir):
        if f.startswith("sac_checkpoint_ep_") and f.endswith(".pth"):
            try:
                out.append((int(f[len("sac_checkpoint_ep_"):-4]),
                            os.path.join(log_dir, f)))
            except ValueError:
                pass
    return sorted(out)


def resume(cfg, agent: SACAgent, buffer: ReplayBuffer, log_dir: str
           ) -> Tuple[int, int]:
    """Continue the newest compatible checkpoint, or refuse loudly.

    Returns (global_episode, global_update)."""
    if cfg.ALLOW_WEIGHT_TRANSFER_FROM:
        path = str(cfg.ALLOW_WEIGHT_TRANSFER_FROM)
        agent.load(path, allow_transfer=True)
        print(f"[Main] explicit weight transfer from {path}; training starts "
              "at episode 0 with an empty buffer")
        return 0, 0
    if cfg.RESUME_MODE == "fresh":
        return 0, 0
    found = _checkpoints(log_dir)
    if not found:
        return 0, 0
    episode, path = found[-1]
    try:
        ckpt = agent.load(path)
    except SchemaMismatch as exc:
        raise SystemExit(
            f"[Main] {exc}\nThe newest checkpoint in {log_dir} belongs to a "
            "different schema. Use a new LOG_DIR, set RESUME_MODE='fresh', or "
            "name it in ALLOW_WEIGHT_TRANSFER_FROM as an explicit transfer "
            "experiment.") from None
    buf_path = os.path.join(log_dir, "replay_buffer.npz")
    if os.path.exists(buf_path):
        try:
            buffer.load(buf_path, cfg.schema_versions())
            print(f"[Main] replay buffer restored: {len(buffer)} records")
        except SchemaMismatch as exc:
            raise SystemExit(f"[Main] {exc}") from None
        except Exception as exc:
            print(f"[Main] replay buffer not restored ({exc}); starting empty")
    print(f"[Main] resumed {path}")
    return int(episode), int(ckpt.get("updates", 0))


class EpsilonScheduler:
    """Linear decay per episode after START_DECAY_STEP, floored at
    EPSILON_MIN (the schedule the previous trainer used)."""

    def __init__(self, cfg):
        self.start = float(cfg.START_EPSILON)
        self.floor = float(cfg.EPSILON_MIN)
        self.begin = int(cfg.START_DECAY_STEP)
        self.span = max(1, int(cfg.LINEARLY_DECAY_STEP))

    def value(self, episode: int) -> float:
        if episode < self.begin:
            return self.start
        frac = min(1.0, (episode - self.begin) / self.span)
        return max(self.floor, self.start + (self.floor - self.start) * frac)


def _train_split(cfg) -> dict:
    """What the active training mode draws from, task parameters included."""
    mode = "DATASET" if cfg.TRAIN_MAP_SOURCE == "dataset" else "UED"
    task = {k: cfg[f"{mode}_{k}"] for k in (
        "DANGER_AREA_RANGE", "DANGER_SHAPES", "DANGER_INSIDE_FRACTION",
        "DANGER_PERCEPTIBILITY", "PRIOR_INFORMED_FRACTION", "ROBOT_RANGE",
        "AUGMENTATION", "AUGMENTATION_TRANSFORMS")}
    out = {"source": cfg.TRAIN_MAP_SOURCE,
           "reserved_seed_ranges": [list(r) for r in
                                    cfg.TRAIN_RESERVED_SEED_RANGES],
           "task": {k: list(v) if isinstance(v, tuple) else v
                    for k, v in task.items()}}
    if mode == "DATASET":
        out.update(sites=list(cfg.DATASET_SITES),
                   sizes_m=list(cfg.DATASET_SIZES_M),
                   sampling="uniform over site x size",
                   density_by_size={str(k): v for k, v in
                                    cfg.DATASET_DENSITY_BY_SIZE.items()})
    else:
        out.update(method=cfg.UED_METHOD, sizes_m=list(cfg.UED_MAP_SIZES_M),
                   size_weights=list(cfg.UED_MAP_SIZE_WEIGHTS),
                   density_by_size={str(k): v for k, v in
                                    cfg.UED_DENSITY_BY_SIZE.items()},
                   difficulty_range=list(cfg.UED_DIFFICULTY_RANGE))
    return out


def run_metadata(cfg, extra: Optional[dict] = None) -> dict:
    from configs import validation_seeds
    from learn.metrics_logger import code_version

    return {
        "experiment_id": cfg.EXPERIMENT_ID,
        "config": cfg.to_dict(),
        "config_fingerprint": cfg.fingerprint,
        "schema_versions": cfg.schema_versions(),
        "observation_schema": cfg.observation_schema(),
        "code_version": code_version(),
        "map_split": {
            "train": _train_split(cfg),
            "validation": {f"{s}m_d{d}_{k}": seed for (s, d, k), seed in
                           validation_seeds(cfg).items()},
            "final_zero_shot": {
                "site": cfg.FINAL_ZERO_SHOT_SITE,
                "size_m": cfg.FINAL_ZERO_SHOT_SIZE_M,
                "hazard_seeds": list(cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS),
                "crowd_seeds": list(cfg.FINAL_ZERO_SHOT_CROWD_SEEDS),
                "density": cfg.FINAL_ZERO_SHOT_DENSITY},
            "auxiliary_osm": list(cfg.ZSG_REAL_SITES),
        },
        **(extra or {}),
    }


# --------------------------------------------------------------------- main

def main(overrides: Optional[dict] = None, max_episodes: Optional[int] = None):
    from configs import resolve_config
    from learn.metrics_logger import MetricsLogger, TrainStatsAccumulator
    from sim.observation import StaticLayers
    from ued import UEDRunner

    cfg = resolve_config(overrides)
    home = os.path.expanduser("~")
    log_dir = os.path.join(home, cfg.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    val_dir = os.path.join(log_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    heartbeat = os.path.join(log_dir, "heartbeat.txt")

    device = cfg.DEVICE if (cfg.DEVICE != "cuda"
                            or torch.cuda.is_available()) else "cpu"
    store = StaticStore(os.path.join(log_dir, cfg.REPLAY_STATIC_DIR),
                        int(cfg.REPLAY_STATIC_CACHE_ENTRIES))
    buffer = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store)
    agent = SACAgent(cfg, device=device, replay=buffer)
    global_episode, global_update = resume(cfg, agent, buffer, log_dir)
    logger = MetricsLogger(
        cfg, log_dir,
        run_metadata=run_metadata(cfg, {"resumed_episode": global_episode}),
        start_episode=global_episode, start_update=global_update)
    train_acc = TrainStatsAccumulator()
    eps_sched = EpsilonScheduler(cfg)
    agent.epsilon = eps_sched.value(global_episode)

    ctx = mp.get_context("spawn")
    n_workers = int(cfg.N_ENVS)
    transition_queue = ctx.Queue(maxsize=64 * n_workers)
    stats_queue = ctx.Queue(maxsize=4 * n_workers)
    val_queue = ctx.Queue()
    epsilon_shared = ctx.Value("d", float(agent.epsilon))
    # Episode number a video is wanted for, or -1; the first worker to start
    # an episode after it is set records that episode.
    video_request = ctx.Value("i", -1)
    level_queues = [ctx.Queue(maxsize=int(cfg.UED_LEVEL_QUEUE_SIZE))
                    for _ in range(n_workers)]
    value_fn = make_value_fn(agent)
    ued = UEDRunner(value_fn=value_fn, schema=cfg.schema_versions())
    ued_path = os.path.join(log_dir, "ued_curriculum.pkl")
    if global_episode > 0:
        ued.load(ued_path)
        ued.population.episode = max(ued.population.episode, global_episode)
    ued.start()

    generation = int(time.time()) % 100_000
    workers, param_queues = [], []

    def start_worker(wid):
        pq = ctx.Queue(maxsize=1)
        p = ctx.Process(target=worker_process,
                        args=(wid, cfg, transition_queue, stats_queue,
                              epsilon_shared, pq, 1234 + generation, level_queues[wid],
                              generation, video_request, log_dir), daemon=True)
        p.start()
        return p, pq

    for wid in range(n_workers):
        p, pq = start_worker(wid)
        workers.append(p)
        param_queues.append(pq)
    print(f"[Main] {n_workers} workers, device={device}, log_dir={log_dir}, "
          f"config {cfg.fingerprint}")

    statics_by_uid: Dict[int, StaticLayers] = {}
    windows: Dict[int, List] = {}
    pending_updates = 0.0
    prefetch = None
    last_supervise = time.time()
    validation_proc = None
    best_score = None
    off_cache_path = os.path.join(val_dir, "off_control_cache.pkl")
    H = int(cfg.OBS_HISTORY_DECISIONS)
    stop_at = None if max_episodes is None else global_episode + int(max_episodes)

    def write_heartbeat():
        with open(heartbeat + ".tmp", "w") as fh:
            fh.write(f"{global_episode}\n{time.time()}\n")
        os.replace(heartbeat + ".tmp", heartbeat)

    write_heartbeat()
    try:
        while stop_at is None or global_episode < stop_at:
            if time.time() - last_supervise > 10.0:
                for wid, p in enumerate(workers):
                    if not p.is_alive():
                        print(f"[Main] worker {wid} died (exit {p.exitcode}); "
                              "restarting")
                        workers[wid], param_queues[wid] = start_worker(wid)
                last_supervise = time.time()
            ued.dispatch(level_queues, global_episode)

            try:
                msg = transition_queue.get(timeout=1.0)
            except queue.Empty:
                msg = None
            if isinstance(msg, EpisodeStartMsg):
                st = StaticLayers.from_parts(msg.static_meta, msg.static_arrays)
                store.put(st)
                statics_by_uid[msg.episode_uid] = st
                windows[msg.episode_uid] = []
            elif isinstance(msg, TransitionMsg):
                win = windows.setdefault(msg.episode_uid, [])
                win.append(msg.record)
                del win[:-H]
                st = statics_by_uid.get(msg.episode_uid)
                if st is not None and msg.action is not None:
                    padded = [None] * (H - len(win)) + list(win)
                    msg.value_sample = (st, padded)
                    ued.on_transition(msg)
                if ued.should_store(msg):
                    buffer.push(msg.episode_uid, msg.step, msg.record,
                                msg.static_key, msg.action, msg.step_rewards,
                                msg.terminal)
                if msg.action is None:
                    buffer.end_episode(msg.episode_uid)
                    statics_by_uid.pop(msg.episode_uid, None)
                    windows.pop(msg.episode_uid, None)
                if global_episode >= int(cfg.START_UPDATE_EPISODE):
                    pending_updates += float(cfg.UPDATES_PER_TRANSITION)
                    while pending_updates >= 1.0:
                        pending_updates -= 1.0
                        if prefetch is None:
                            prefetch = BatchPrefetcher(buffer, agent.batch_size)
                        batch = prefetch.get()
                        if batch is None:
                            break
                        train_acc.add(agent.update(batch))
                        global_update += 1
                        if global_update % int(cfg.LOG_TRAIN_EVERY_UPDATES) == 0:
                            logger.log_train(global_update, global_episode, {
                                **train_acc.pop_means(),
                                "train/replay_records": len(buffer),
                                "train/epsilon": agent.epsilon})

            while True:
                try:
                    s = stats_queue.get_nowait()
                except queue.Empty:
                    break
                global_episode += 1
                ued.population.episode = global_episode
                ued.epsilon = float(agent.epsilon)
                if s.abnormal != 1:
                    ued.on_episode(s)
                    logger.log_episode(global_episode, global_update,
                                       {**s.metrics,
                                        "episode/epsilon": agent.epsilon,
                                        "episode/is_replay": float(s.is_replay)})
                    logger.note("episode_map", global_episode=global_episode,
                                map=s.level_name)
                    if s.video_path:
                        logger.log_video(global_episode, global_update,
                                         s.video_path, int(cfg.VIDEO_FPS),
                                         caption=f"{s.level_name}, episode "
                                                 f"{global_episode}")
                if (int(cfg.VIDEO_EVERY_EPISODES) > 0 and global_episode
                        % int(cfg.VIDEO_EVERY_EPISODES) == 0):
                    with video_request.get_lock():
                        video_request.value = int(global_episode)
                else:
                    logger.note("episode_failed", worker=s.worker_id,
                                episode_uid=s.episode_uid)
                agent.epsilon = eps_sched.value(global_episode)
                with epsilon_shared.get_lock():
                    epsilon_shared.value = float(agent.epsilon)
                write_heartbeat()
                if (global_episode >= int(cfg.START_UPDATE_EPISODE)
                        and global_episode % int(cfg.POLICY_BROADCAST_INTERVAL) == 0):
                    sd = {k: v.detach().cpu()
                          for k, v in agent.policy.state_dict().items()}
                    for pq in param_queues:
                        try:
                            while True:
                                pq.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            pq.put_nowait(sd)
                        except queue.Full:
                            pass
                if global_episode % 100 == 0 and \
                        global_episode >= int(cfg.START_UPDATE_EPISODE):
                    path = os.path.join(log_dir,
                                        f"sac_checkpoint_ep_{global_episode}.pth")
                    agent.save(path)
                    buffer.save(os.path.join(log_dir, "replay_buffer.npz"),
                                cfg.schema_versions())
                    ued.save(ued_path)
                    logger.note("checkpoint", path=path,
                                global_episode=global_episode,
                                global_update=global_update)
                if (cfg.PERIODIC_VALIDATION
                        and int(cfg.VALIDATION_CYCLE_EPISODE) > 0
                        and global_episode >= int(cfg.START_UPDATE_EPISODE)
                        and global_episode % int(cfg.VALIDATION_CYCLE_EPISODE) == 0
                        and (validation_proc is None
                             or not validation_proc.is_alive())):
                    path = os.path.join(val_dir,
                                        f"policy_ep_{global_episode}.pth")
                    agent.save(path)
                    validation_proc = ctx.Process(
                        target=validation_process,
                        args=(cfg, path, global_episode, val_queue, val_dir,
                              off_cache_path), daemon=True)
                    validation_proc.start()
                if ued.enabled and global_episode % 20 == 0:
                    logger.log_episode(global_episode, global_update, {
                        f"episode/ued/{k}": v for k, v in ued.stats().items()})

            try:
                status, ep, path, payload = val_queue.get_nowait()
            except queue.Empty:
                status = None
            if status == "ok":
                logger.log_eval(ep, global_update, payload)
                from learn.zero_shot import selection_score
                score = selection_score(payload, "eval/validation")
                if score is not None and (best_score is None or score > best_score):
                    best_score = score
                    best = os.path.join(log_dir, "best_validation.pth")
                    import shutil
                    shutil.copyfile(path, best)
                    logger.upload_selected_model(
                        best, name=f"{cfg.EXPERIMENT_ID}-best-validation",
                        aliases=["best"],
                        metadata={"episode": ep, "score": score,
                                  **cfg.schema_versions()})
            elif status == "error":
                logger.note("validation_failed", episode=ep, error=payload)
                print(f"[Main] validation at episode {ep} failed:\n{payload}")
    finally:
        if prefetch is not None:
            prefetch.stop()
        for p in workers:
            if p.is_alive():
                p.terminate()
        ued.stop()
        logger.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
