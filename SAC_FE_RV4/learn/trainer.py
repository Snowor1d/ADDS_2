"""The training process.

    python3 Start_training.py                 # with the watchdog (restarts on stall/crash)
    python3 ADDS_AS_reinforcement.py          # without it
    ... --preset presets/rv3_parity.json --set LR=3e-4    # overrides, see configs/__init__.py

Main loop (single process, workers and curriculum producers around it):

    keep every worker's task queue filled        ued/runner.TaskSampler
    take worker messages into the replay         learn/replay.FrameReplay
    run UPDATES_PER_TRANSITION updates per stored transition (the workers
    wait on a full queue meanwhile, as in RV3)  learn/sac.SACLearner
    per finished episode: log, curriculum update, epsilon, checkpoint,
    policy broadcast, periodic validation

Resuming: the newest ckpt_ep_*.pth in ~/<LOG_DIR> is loaded when its
configuration fingerprint matches; a mismatch stops with an explanation
rather than mixing two configurations in one run (RESUME_MODE = "fresh"
ignores old checkpoints). The replay is not saved by default; after a resume
the learner waits for RESUME_REFILL_TRANSITIONS new transitions.
"""

from __future__ import annotations

import glob
import json
import multiprocessing as mp
import os
import queue
import signal
import socket
import sys
import time
from typing import Dict, List, Optional


class EpsilonSchedule:
    """RV3's per-episode epsilon schedule."""

    def __init__(self, cfg, start: float):
        self.cfg = cfg
        self.eps = float(start)

    def step(self, episode: int) -> float:
        c = self.cfg
        if episode >= int(c.START_DECAY_STEP):
            if c.SCHEDULER_TYPE == "e":
                self.eps = self.eps * float(c.DECAY_VALUE)
            else:
                self.eps = self.eps - 1.0 / max(1, int(c.LINEARLY_DECAY_STEP))
        self.eps = max(self.eps, float(c.EPSILON_MIN))
        return self.eps


class WorkerPool:
    def __init__(self, cfg, ctx, out_q, epsilon_shared, base_seed: int):
        self.cfg = cfg
        self.ctx = ctx
        self.out_q = out_q
        self.eps = epsilon_shared
        self.base_seed = int(base_seed)
        self.procs: Dict[int, mp.Process] = {}
        self.task_qs: Dict[int, mp.Queue] = {}
        self.param_qs: Dict[int, mp.Queue] = {}
        self.inflight: Dict[int, int] = {}
        self.retiring: set = set()
        self.last_params = None
        self.restarts = 0

    def _start(self, wid: int):
        from learn.worker import worker_main
        tq = self.ctx.Queue()
        pq = self.ctx.Queue(maxsize=1)
        p = self.ctx.Process(target=worker_main, args=(wid, tq, self.out_q, pq, self.eps,
                                                       self.base_seed), daemon=True,
                             name=f"rollout-{wid}")
        p.start()
        self.procs[wid], self.task_qs[wid], self.param_qs[wid] = p, tq, pq
        self.inflight[wid] = 0
        if self.last_params is not None:
            self._send_params(wid, self.last_params)

    def size(self) -> int:
        return len([w for w in self.procs if w not in self.retiring])

    def resize(self, n: int):
        active = sorted(w for w in self.procs if w not in self.retiring)
        for wid in range(n):
            if wid not in self.procs:
                self._start(wid)
            self.retiring.discard(wid)
        for wid in active:
            if wid >= n and wid not in self.retiring:
                # Finish the episode in hand, then exit.
                self.retiring.add(wid)
                self.task_qs[wid].put(None)

    def fill(self, make_task, depth: int, register):
        for wid in list(self.procs):
            if wid in self.retiring:
                continue
            while self.inflight[wid] < depth:
                t = make_task()
                register(t)
                self.task_qs[wid].put(t)
                self.inflight[wid] += 1

    def done(self, wid: int):
        if wid in self.inflight:
            self.inflight[wid] = max(0, self.inflight[wid] - 1)

    def supervise(self, on_lost):
        for wid, p in list(self.procs.items()):
            if p.is_alive():
                continue
            if wid in self.retiring:
                self.procs.pop(wid)
                self.task_qs.pop(wid)
                self.param_qs.pop(wid)
                self.inflight.pop(wid, None)
                self.retiring.discard(wid)
                continue
            print(f"[main] worker {wid} died (exit {p.exitcode}); restarting", flush=True)
            on_lost(wid)
            self.restarts += 1
            self._start(wid)

    def _send_params(self, wid, sd):
        q = self.param_qs[wid]
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(sd)
        except queue.Full:
            pass

    def broadcast(self, sd):
        self.last_params = sd
        for wid in self.procs:
            self._send_params(wid, sd)

    def stop(self):
        for wid, tq in self.task_qs.items():
            try:
                tq.put(None)
            except Exception:
                pass
        deadline = time.time() + 5
        for p in self.procs.values():
            p.join(timeout=max(0.1, deadline - time.time()))
        for p in self.procs.values():
            if p.is_alive():
                p.terminate()
        self.procs.clear()


class ValidationManager:
    def __init__(self, cfg, log_dir: str, ctx):
        self.cfg = cfg
        self.dir = os.path.join(log_dir, "validation")
        os.makedirs(self.dir, exist_ok=True)
        self.ctx = ctx
        self.proc = None
        self.out = None
        self.episode = None
        self.snapshot = None

    def busy(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, episode: int, ckpt: dict) -> bool:
        import torch
        from learn.validation import validation_main
        if self.busy():
            return False
        self.snapshot = os.path.join(self.dir, f"snapshot_ep_{episode}.pth")
        torch.save(ckpt, self.snapshot)
        self.out = os.path.join(self.dir, f"result_ep_{episode}.json")
        self.episode = episode
        # Not a daemon: it starts its own pool of evaluation processes.
        self.proc = self.ctx.Process(target=validation_main, args=(self.snapshot, self.out),
                                     daemon=False, name="validation")
        self.proc.start()
        return True

    def poll(self):
        if self.proc is None or self.proc.is_alive():
            return None
        self.proc.join(timeout=1)
        self.proc = None
        try:
            with open(self.out, encoding="utf-8") as fh:
                res = json.load(fh)
        except (OSError, ValueError):
            print(f"[main] validation at episode {self.episode} produced no result", flush=True)
            return None
        res["episode"] = self.episode
        return res

    def stop(self):
        if self.busy():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.proc.terminate()
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()


def _latest_checkpoint(log_dir: str) -> Optional[str]:
    files = glob.glob(os.path.join(log_dir, "ckpt_ep_*.pth"))
    if not files:
        return None
    return max(files, key=lambda f: int(os.path.basename(f)[8:-4]))


def _train_maps_description(cfg) -> dict:
    if cfg.TRAIN_MAP_SOURCE == "dataset":
        return {"source": "dataset", "maps": list(cfg.DATASET_MAPS)}
    d = {"source": cfg.TRAIN_MAP_SOURCE, "ued_method": cfg.UED_METHOD,
         "generator_sizes_m": list(cfg.UED_MAP_SIZES_M),
         "reserved_seed_ranges": [list(r) for r in cfg.UED_RESERVED_SEED_RANGES]}
    if cfg.TRAIN_MAP_SOURCE == "mixed":
        d["maps"] = list(cfg.DATASET_MAPS)
    return d


def main(argv: Optional[List[str]] = None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    from configs import cfg
    from learn.metrics_logger import MetricsLogger, TrainStatsAccumulator, code_version
    from learn.replay import FrameReplay
    from learn.sac import SACLearner
    from learn.validation import flatten_for_logging
    from ued.runner import TaskSampler

    log_dir = os.path.join(os.path.expanduser("~"), cfg.LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    heartbeat = os.path.join(log_dir, "heartbeat.txt")
    device = cfg.DEVICE if (cfg.DEVICE != "cuda" or torch.cuda.is_available()) else "cpu"
    fp = cfg.fingerprint()
    max_episodes = args.max_episodes if args.max_episodes is not None else int(cfg.MAX_EPISODES)

    learner = SACLearner(cfg, device)
    fpt = cfg.REPLAY_FRAMES_PER_TRANSITION
    if fpt is None:
        # a waypoint transition brings one frame per 2 s window it spans, up
        # to FRAME_STACK (cartesian: 3.4 on average with 14 s decisions;
        # polar: 1.2 with a random policy, measured)
        if cfg.ROBOT_ACTION_MODE == "velocity":
            fpt = 1.1
        elif cfg.ROBOT_WAYPOINT_PARAM == "polar":
            fpt = 1.5
        else:
            fpt = float(min(int(cfg.FRAME_STACK), cfg.decision_max_ticks()))
    replay = FrameReplay(int(cfg.BUFFER_SIZE), int(cfg.EGO_MAP_SIZE), int(cfg.DOWNSAMPLE_MAP_SIZE),
                         int(cfg.FRAME_STACK), bool(cfg.OBS_STATIC_DISTANCE_CHANNEL), device=device,
                         action_dim=cfg.action_dim(), frames_per_transition=float(fpt))
    frame_bytes = int(cfg.EGO_MAP_SIZE) ** 2 * (2 if cfg.OBS_STATIC_DISTANCE_CHANNEL else 1) \
        + int(cfg.DOWNSAMPLE_MAP_SIZE) ** 2
    print(f"[main] replay: {int(cfg.BUFFER_SIZE)} transitions, {replay.fcap} frames "
          f"(~{replay.fcap * frame_bytes / 1e9:.1f} GB), action dim {cfg.action_dim()}, "
          f"robot action mode {cfg.ROBOT_ACTION_MODE}; workers wait while more than "
          f"{cfg.MAX_PENDING_UPDATES} updates are pending", flush=True)
    sampler = TaskSampler(cfg, seed=int(time.time()) % 100000)
    start_episode = 0
    epsilon0 = float(cfg.START_EPSILON)
    best_val = None
    resumed = False

    ck_path = _latest_checkpoint(log_dir) if cfg.RESUME_MODE == "latest" else None
    if ck_path:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        if ck.get("config_fingerprint") != fp:
            changed = sorted(k for k in set(ck.get("config", {})) | set(cfg.as_dict())
                             if ck.get("config", {}).get(k) != cfg.get(k))
            raise SystemExit(
                f"[main] {ck_path} was trained with a different configuration "
                f"(fingerprint {ck.get('config_fingerprint')} vs {fp}; differing settings: "
                f"{changed[:20]}). Use another LOG_DIR, restore the settings, or "
                f"--set RESUME_MODE=fresh to ignore the old checkpoints.")
        learner.load_state_dict(ck["learner"])
        start_episode = int(ck["global_episode"])
        epsilon0 = float(ck.get("epsilon", epsilon0))
        best_val = ck.get("best_val")
        if ck.get("sampler"):
            sampler.load_state_dict(ck["sampler"])
        resumed = True
        print(f"[main] resumed from {ck_path} at episode {start_episode}", flush=True)
        if cfg.SAVE_REPLAY and replay.load(os.path.join(log_dir, "replay")):
            print(f"[main] replay restored ({len(replay)} transitions)", flush=True)

    meta = {"config_fingerprint": fp, "code_version": code_version(),
            "host": socket.gethostname(), "train_maps": _train_maps_description(cfg),
            "resumed_from": ck_path, "device": str(device)}
    logger = MetricsLogger(cfg, log_dir, run_metadata=meta, start_episode=start_episode,
                           start_update=learner.updates)

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue(maxsize=max(16, 4 * max(int(cfg.N_ENVS), int(cfg.N_ENVS_WARMUP))))
    eps_shared = ctx.Value("d", epsilon0)
    schedule = EpsilonSchedule(cfg, epsilon0)
    pool = WorkerPool(cfg, ctx, out_q, eps_shared, base_seed=1234 + start_episode)
    validation = ValidationManager(cfg, log_dir, ctx)
    sampler.start(ctx)

    episode = start_episode
    warm = episode < int(cfg.START_UPDATE_EPISODE)
    pool.resize(int(cfg.N_ENVS_WARMUP) if warm else int(cfg.N_ENVS))
    pool.broadcast(learner.policy_state_dict())
    last_broadcast = episode

    tasks: Dict[int, object] = {}
    cursors: Dict[tuple, object] = {}
    ep_task: Dict[tuple, object] = {}
    pending = 0.0
    # Workers get no new episode while the learner is this far behind
    # (MAX_PENDING_UPDATES); wait_s is the time spent that way.
    max_pending = cfg.MAX_PENDING_UPDATES
    max_pending = None if max_pending is None else float(max_pending)
    wait_s = 0.0
    t_loop = time.time()
    acc = TrainStatsAccumulator()
    n_trans = 0
    n_trans_since_resume = 0
    t_perf = time.time()
    trans_perf = 0
    upd_perf = 0
    stop = {"flag": False}

    def _sig(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        print(f"\n[main] signal {signum}: finishing up (again to force)", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def make_checkpoint() -> dict:
        return {"format": "rv4", "config": cfg.as_dict(), "config_fingerprint": fp,
                "global_episode": episode, "learner": learner.state_dict(),
                "actor": learner.actor.state_dict(), "epsilon": schedule.eps,
                "sampler": sampler.state_dict() if sampler.uses_ued else None,
                "best_val": best_val, "train_maps": _train_maps_description(cfg),
                "saved_at": time.time()}

    def save_checkpoint():
        path = os.path.join(log_dir, f"ckpt_ep_{episode}.pth")
        tmp = path + ".tmp"
        torch.save(make_checkpoint(), tmp)
        os.replace(tmp, path)
        old = sorted(glob.glob(os.path.join(log_dir, "ckpt_ep_*.pth")),
                     key=lambda f: int(os.path.basename(f)[8:-4]))
        for f in old[:-int(cfg.KEEP_LAST_CHECKPOINTS)]:
            try:
                os.remove(f)
            except OSError:
                pass
        if cfg.SAVE_REPLAY:
            replay.save(os.path.join(log_dir, "replay"))
        logger.note("checkpoint", path=path, global_episode=episode)
        return path

    def lost_worker(wid):
        for key in [k for k in cursors if k[0] == wid]:
            cursors.pop(key, None)
            t = ep_task.pop(key, None)
            if t is not None:
                tasks.pop(t.task_id, None)
        pool.inflight[wid] = 0

    def ued_values(cur):
        """V(s_t) along the episode, for MaxMC / PVL."""
        slots = replay.episode_slots(cur.t_ids)
        if len(slots) == 0:
            return None, None, 0.0, True, None
        vals = []
        was = learner.actor.training, learner.critic.training
        learner.actor.eval()
        learner.critic.eval()
        try:
            for i in range(0, len(slots), 512):
                vals.append(learner.state_values(replay.obs_from_slots(slots[i:i + 512])).cpu().numpy())
            last = slots[-1:]
            last_v = float(learner.state_values(replay.obs_from_slots(last, nxt=True)).cpu().numpy()[0])
        finally:
            learner.actor.train(was[0])
            learner.critic.train(was[1])
        return (np.concatenate(vals), replay.reward[slots].astype(np.float64), last_v,
                bool(replay.terminal[slots[-1]]), replay.dur[slots].astype(np.float64))

    def on_end(wid, ep, task_id, st):
        nonlocal episode, last_broadcast, best_val, warm
        pool.done(wid)
        key = (wid, ep)
        cur = cursors.pop(key, None)
        task = ep_task.pop(key, None) or tasks.get(task_id)
        tasks.pop(task_id, None)
        episode += 1
        with open(heartbeat, "w") as fh:
            fh.write(f"{episode}\n{time.time()}\n")
        eps_used = float(st.get("epsilon", schedule.eps))
        if st.get("abnormal"):
            logger.note("abnormal_episode", global_episode=episode, worker=wid,
                        task=getattr(task, "label", lambda: "?")())
        else:
            m = {"episode/total_reward": st["total_reward"],
                 "episode/evac_time_100": st["evac_time_100"],
                 "episode/evac_time_80": st["evac_time_80"],
                 "episode/success": st["success"],
                 "episode/total_lifetime": st["total_lifetime"],
                 "episode/epsilon": eps_used,
                 "episode/decisions": st["decisions"],
                 "episode/robot_collision_steps": st["robot_collision_steps"],
                 "episode/duration_s": st["duration_s"],
                 "episode/sim_steps_per_s": st["steps"] / max(1e-6, st["duration_s"]),
                 "episode/map_size_m": st["map_size_m"], "episode/crowd": st["crowd"],
                 "episode/is_dataset": float(task is not None and task.kind == "dataset"),
                 "episode/is_new_level": float(task is not None and task.is_new),
                 "episode/replay_size": float(len(replay)),
                 "episode/alpha": float(learner.alpha)}
            if st.get("freeflow_steps"):
                m["episode/evac_over_freeflow"] = st["evac_time_100"] / max(1.0, st["freeflow_steps"])
            if st.get("shuttle_time"):
                m["episode/shuttle_time"] = st["shuttle_time"]
                m["episode/regret_vs_shuttle"] = (st["evac_time_100"] - st["shuttle_time"]) / st["shuttle_time"]
            for k, v in st.get("reward_components", {}).items():
                m[f"episode/reward/{k}"] = v
            logger.log_episode(episode, learner.updates, m)
            if task is not None:
                logger.write_by_map(task.label() if task.kind == "dataset" else "ued",
                                    st["total_reward"], st["evac_time_100"])
            if task is not None and task.kind == "level":
                vals = rew = dur = None
                last_v, last_term = 0.0, True
                if (sampler.needs_values() and cur is not None
                        and eps_used <= float(cfg.UED_SCORE_MAX_EPSILON)):
                    vals, rew, last_v, last_term, dur = ued_values(cur)
                elif cur is not None:
                    slots = replay.episode_slots(cur.t_ids)
                    rew = replay.reward[slots].astype(np.float64) if len(slots) else None
                    dur = replay.dur[slots].astype(np.float64) if len(slots) else None
                # gamma per transition: gamma ** (ACTION_SCALE windows it spans)
                g = learner.gamma if dur is None else np.power(learner.gamma, dur)
                sampler.on_episode(task, st, episode, eps_used, values=vals, rewards=rew,
                                   last_value=last_v, last_terminal=last_term, gamma=g)
        # schedules (RV3's linear gamma ascent and alpha decay, per episode;
        # constant with the default START == END)
        frac = lambda n: min(1.0, max(0.0, episode / max(1, int(n))))
        learner.gamma = float(cfg.GAMMA_START) + (float(cfg.GAMMA_END) - float(cfg.GAMMA_START)) * frac(cfg.GAMMA_SCHEDULE_STEP)
        if not cfg.ALPHA_AUTO:
            a = float(cfg.ALPHA_START) + (float(cfg.ALPHA_END) - float(cfg.ALPHA_START)) * frac(cfg.ALPHA_DECAY_STEPS)
            with torch.no_grad():
                learner.log_alpha.fill_(float(np.log(max(a, 1e-8))))
        eps = schedule.step(episode)
        with eps_shared.get_lock():
            eps_shared.value = eps
        if warm and episode >= int(cfg.START_UPDATE_EPISODE):
            warm = False
            print(f"[main] episode {episode}: updates start; workers "
                  f"{pool.size()} -> {cfg.N_ENVS}", flush=True)
            pool.resize(int(cfg.N_ENVS))
        if not warm and episode - last_broadcast >= int(cfg.POLICY_BROADCAST_INTERVAL):
            pool.broadcast(learner.policy_state_dict())
            last_broadcast = episode
        if not warm and episode % int(cfg.CHECKPOINT_EVERY_EPISODES) == 0:
            save_checkpoint()
        if episode % 50 == 0 and sampler.uses_ued:
            logger.log_episode(episode, learner.updates, sampler.stats())
        if (sampler.uses_ued and int(cfg.UED_SNAPSHOT_EPISODES) > 0
                and episode % int(cfg.UED_SNAPSHOT_EPISODES) == 0 and len(sampler.population)):
            try:
                from ued.render import render_levels
                pop = sampler.population
                top = pop.snapshot_levels(16)
                titles = [f"g{lv.generation} d{lv.difficulty} n{lv.crowd} "
                          f"s={pop.raw_score(pop.get(lv.level_id)):.3g}" for lv in top]
                os.makedirs(os.path.join(log_dir, "ued_snapshots"), exist_ok=True)
                img = render_levels(top, os.path.join(log_dir, "ued_snapshots", f"ep_{episode}.png"),
                                    titles)
                logger.log_media(episode, learner.updates, {"ued/top_levels": ("image", img)})
            except Exception as e:
                logger.note("ued_snapshot_error", error=str(e))
        if (cfg.PERIODIC_VALIDATION and not warm
                and episode % int(cfg.VALIDATION_CYCLE_EPISODE) == 0):
            if validation.start(episode, make_checkpoint()):
                print(f"[main] validation started at episode {episode}", flush=True)
            else:
                logger.note("validation_skipped", global_episode=episode, reason="busy")
        if episode % 10 == 0:
            print(f"[main] ep {episode} | evac100 {st.get('evac_time_100', float('nan')):.0f} "
                  f"| R {st.get('total_reward', float('nan')):.1f} | eps {eps:.3f} "
                  f"| replay {len(replay)} | upd {learner.updates}", flush=True)

    def on_validation(res):
        nonlocal best_val
        flat = flatten_for_logging(res)
        logger.log_eval(res["episode"], learner.updates, flat)
        if res.get("video"):
            logger.log_media(res["episode"], learner.updates,
                             {"eval/val/video": ("video", res["video"], res.get("video_fps", 20))})
        score = res["conditions"].get(res["policy_condition"], {}).get("mean")
        if score is not None and (best_val is None or score < best_val["score"]):
            best_val = {"score": float(score), "episode": int(res["episode"])}
            dst = os.path.join(log_dir, "best_validation.pth")
            import shutil
            shutil.copyfile(res["snapshot"], dst)
            logger.upload_model(dst, name=f"{cfg.EXPERIMENT_ID}-best", aliases=["best"],
                                metadata={"episode": res["episode"], "val_evac_time_100": score})
            logger.summary({"best_val_evac_time_100": score, "best_val_episode": res["episode"]})
        print(f"[main] validation ep {res['episode']}: policy {score} "
              f"(best {best_val})", flush=True)

    refill_needed = int(cfg.RESUME_REFILL_TRANSITIONS) if (resumed and len(replay) == 0) else 0
    depth = max(1, int(cfg.UED_LEVEL_QUEUE_SIZE))
    last_supervise = time.time()
    try:
        while not stop["flag"] and (max_episodes <= 0 or episode < max_episodes):
            if time.time() - last_supervise > 10:
                pool.supervise(lost_worker)
                last_supervise = time.time()
            now = time.time()
            throttled = max_pending is not None and pending > max_pending
            if throttled:
                wait_s += now - t_loop
            t_loop = now
            if not throttled:
                pool.fill(lambda: sampler.next_task(episode), depth,
                          lambda t: tasks.__setitem__(t.task_id, t))
            msgs = []
            try:
                msgs.append(out_q.get(timeout=0.2))
                while len(msgs) < 256:
                    msgs.append(out_q.get_nowait())
            except queue.Empty:
                pass
            for msg in msgs:
                kind = msg[0]
                if kind == "start":
                    _, wid, ep, task_id, ego, glob_, es, gs = msg
                    cursors[(wid, ep)] = replay.begin_episode(ego, glob_, es, gs)
                    ep_task[(wid, ep)] = tasks.get(task_id)
                elif kind == "trans":
                    _, wid, ep, chunk = msg
                    cur = cursors.get((wid, ep))
                    if cur is None:
                        continue
                    t = ep_task.get((wid, ep))
                    trainable = True if t is None else bool(t.trainable)
                    for item in chunk:
                        if len(item) == 9:          # velocity: one window, one frame
                            s, a, r, e2, g2, es2, s2, term, last = item
                            replay.add(cur, s, a, r, e2, g2, es2, s2, term, last, trainable=trainable)
                        else:                       # waypoint: (.., frames, .., ticks)
                            s, a, r, frames, s2, term, last, ticks = item
                            replay.add_multi(cur, s, a, r, frames, s2, term, last, ticks,
                                             trainable=trainable)
                    n_trans += len(chunk)
                    trans_perf += len(chunk)
                    n_trans_since_resume += len(chunk)
                    if not warm and trainable:
                        pending += float(cfg.UPDATES_PER_TRANSITION) * len(chunk)
                elif kind == "end":
                    _, wid, ep, task_id, st = msg
                    on_end(wid, ep, task_id, st)
            # updates
            ready = len(replay) >= int(cfg.BATCH_SIZE) * int(cfg.START_BATCH_TIMES) and \
                n_trans_since_resume >= refill_needed
            budget = int(cfg.MAX_UPDATES_PER_LOOP)
            while ready and pending >= 1.0 and budget > 0:
                batch = replay.sample(int(cfg.BATCH_SIZE), int(cfg.N_STEP), learner.gamma,
                                      bool(cfg.TIMEOUT_BOOTSTRAP), learner.rng)
                acc.add(learner.update(batch))
                pending -= 1.0
                budget -= 1
                upd_perf += 1
                if learner.updates % int(cfg.LOG_TRAIN_EVERY_UPDATES) == 0:
                    dt = max(1e-6, time.time() - t_perf)
                    logger.log_train(learner.updates, episode, {
                        **acc.pop_means(), "train/replay_size": float(len(replay)),
                        "train/updates_per_s": upd_perf / dt,
                        "train/transitions_per_s": trans_perf / dt,
                        "train/pending_updates": pending,
                        "train/worker_wait_frac": min(1.0, wait_s / dt),
                        "train/workers": float(pool.size()),
                        "train/worker_restarts": float(pool.restarts)})
                    t_perf, upd_perf, trans_perf = time.time(), 0, 0
                    wait_s = 0.0
            if not ready:
                pending = min(pending, float(cfg.MAX_UPDATES_PER_LOOP))
            res = validation.poll()
            if res is not None:
                on_validation(res)
    except KeyboardInterrupt:
        print("[main] interrupted", flush=True)
    finally:
        print("[main] stopping workers ...", flush=True)
        pool.stop()
        sampler.stop()
        validation.stop()
        try:
            if episode > start_episode and episode >= int(cfg.START_UPDATE_EPISODE):
                print(f"[main] final checkpoint {save_checkpoint()}", flush=True)
        except Exception as e:
            print(f"[main] final checkpoint failed: {e}", flush=True)
        logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
