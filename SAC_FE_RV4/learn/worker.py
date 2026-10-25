"""Rollout worker process.

Runs episodes forever on the tasks the main process sends (ued/runner.Task),
with the latest policy it was sent, and reports back through one queue so a
worker's messages arrive in order:

    ("start", wid, ep, task_id, ego, glob, ego_static, glob_static)
    ("trans", wid, ep, [transition, ...])      every WORKER_SEND_CHUNK decisions
    ("end",   wid, ep, task_id, stats)

A transition carries only the frame its decision ended on; the main process
rebuilds stacks from frames it already has (learn/replay.py).

Episode semantics are RV3's: decide every ACTION_SCALE steps (uniform random
action with probability epsilon, else a policy sample), hold the action,
compute the reward at the window's last step (learn/rewards.py). Two
differences: the first window's transition is kept (RV3 dropped it), and the
time limit is reported separately from the task's end so the learner can
bootstrap through it (TIMEOUT_BOOTSTRAP).
"""

from __future__ import annotations

import os
import queue
import random
import time
import traceback

import numpy as np


def _drain_params(param_q, policy):
    latest = None
    try:
        while True:
            latest = param_q.get_nowait()
    except queue.Empty:
        pass
    if latest is not None:
        policy.load_state_dict(latest)


def worker_main(wid: int, task_q, out_q, param_q, epsilon_shared, seed: int):
    import torch
    from configs import cfg
    from evaluation.baselines import make_controller
    from learn.networks import build_policy
    from learn.obs import ObservationBuilder, StackedObservation, network_state
    from learn.rewards import COMPONENTS, decision_reward
    from learn.rollout import EnvSpec, build_env, eval_step_cap, freeflow_steps, run_episode

    torch.set_num_threads(1)
    random.seed(seed + wid)
    np.random.seed((seed + wid) % (2 ** 32))
    torch.manual_seed(seed + wid)
    policy = build_policy(cfg).eval()
    D = int(cfg.ACTION_SCALE)
    max_steps = int(cfg.MAX_STEPS)
    chunk_n = max(1, int(cfg.WORKER_SEND_CHUNK))
    use_scale = bool(cfg.MAP_SCALE_TO_ACTOR)
    ep_counter = 0

    while True:
        task = task_q.get()
        if task is None:
            return
        _drain_params(param_q, policy)
        with epsilon_shared.get_lock():
            eps = float(epsilon_shared.value)
        ep_counter += 1
        ep = ep_counter
        t_start = time.time()
        stats = {"abnormal": 0, "worker": wid, "epsilon": eps}

        spec = EnvSpec(crowd=task.crowd, map_num=task.map_num, level=task.level,
                       augmentation=task.augmentation, seed=task.seed)
        # The curriculum's reference: the tuned nearest-person shuttle on the
        # same level and placement (only for scores that need it).
        if task.need_shuttle_ref:
            try:
                ctrl, rv = make_controller("shuttle_nearest", cfg, seed=task.seed)
                benv = build_env(spec, cfg, robot=rv)
                res = run_episode(benv, ctrl, cfg, max_steps=max_steps)
                stats["shuttle_time"] = res["evac_time_100"]
                del benv
            except Exception as e:
                print(f"[worker {wid}] shuttle reference failed: {e}", flush=True)

        try:
            if task.seed is None:
                spec.seed = random.randrange(0, 2 ** 31 - 1)
            env = None
            for attempt in range(5):
                try:
                    env = build_env(spec, cfg, robot="Q")
                    break
                except Exception as e:
                    print(f"[worker {wid}] env build error ({task.label()}): {e}; retrying",
                          flush=True)
                    spec.seed = random.randrange(0, 2 ** 31 - 1)
            if env is None:
                raise RuntimeError("env build failed 5 times")
            # Each episode continues the worker's own random stream after the
            # seeded construction, so placement is reproducible and dynamics
            # are not shared between workers.
            random.seed(seed * 7919 + wid * 104729 + ep)
            np.random.seed((seed * 7919 + wid * 104729 + ep) % (2 ** 32))
            stats["freeflow_steps"] = freeflow_steps(env, cfg)
            ob = ObservationBuilder(cfg, env)
            stack = StackedObservation(int(cfg.FRAME_STACK))
            ego, glob, es = ob.frames()
            stack.reset(ego, glob, es)
            out_q.put(("start", wid, ep, task.task_id, ego, glob, es, ob.glob_static))
            state = ob.robot_state()
            comp_tot = {k: 0.0 for k in COMPONENTS}
            total_reward = 0.0
            t80 = t100 = None
            total = env.total_agents
            chunk = []
            action = np.zeros(2, np.float32)
            coll_window = 0
            collisions = 0
            decisions = 0
            for step in range(max_steps):
                if step % D == 0:
                    if np.random.rand() < eps:
                        dx = np.random.uniform(-2, 2)
                        dy = np.random.uniform(-2, 2)
                    else:
                        e_in, g_in = stack.network_input(ob.glob_static)
                        with torch.no_grad():
                            a, _ = policy.sample(torch.from_numpy(e_in)[None],
                                                 torch.from_numpy(g_in)[None],
                                                 torch.from_numpy(network_state(state, use_scale))[None])
                        dx, dy = float(a[0, 0]), float(a[0, 1])
                    real = env.robot.receive_action([dx, dy])
                    action = np.array([real[0], real[1]], dtype=np.float32)
                    coll_window = 0
                    decisions += 1
                env.step()
                hit = int(bool(env.robot.collision_check))
                coll_window += hit
                collisions += hit
                finished = bool(env.robot.is_game_finished)
                alive = env.alived_agents()
                if t80 is None and alive < total * 0.2:
                    t80 = step + 1
                if t100 is None and alive < 1:
                    t100 = step + 1
                last = finished or step == max_steps - 1
                if step % D == D - 1 or last:
                    r, comp = decision_reward(env, cfg, hit, coll_window, finished, step, max_steps)
                    if r < -1e3:
                        raise RuntimeError(f"reward collapsed: {r}")
                    total_reward += r
                    for k, v in comp.items():
                        comp_tot[k] += v
                    ego2, glob2, es2 = ob.frames()
                    state2 = ob.robot_state()
                    chunk.append((state, action, float(r), ego2, glob2, es2, state2,
                                  bool(finished), bool(last)))
                    if len(chunk) >= chunk_n or last:
                        out_q.put(("trans", wid, ep, chunk))
                        chunk = []
                    stack.push(ego2, glob2, es2)
                    state = state2
                if last:
                    break
            stats.update({
                "total_reward": total_reward,
                "evac_time_100": float(t100 if t100 is not None else max_steps),
                "evac_time_80": float(t80 if t80 is not None else max_steps),
                "success": float(t100 is not None),
                "total_lifetime": float(env.calculate_all_agents_life_time()),
                "steps": float(step + 1),
                "decisions": float(decisions),
                "robot_collision_steps": float(collisions),
                "map_size_m": float(max(env.width, env.height)),
                "crowd": float(total),
                "reward_components": comp_tot,
            })
        except Exception as e:
            print(f"[worker {wid}] episode error: {e}", flush=True)
            traceback.print_exc()
            stats["abnormal"] = 1
            if chunk:
                out_q.put(("trans", wid, ep, chunk))
        stats["duration_s"] = time.time() - t_start
        out_q.put(("end", wid, ep, task.task_id, stats))
