"""One episode, the same way everywhere.

The training worker, the validation and zero-shot evaluators and the tests
all run episodes through `run_episode`, so they measure, observe, act and
account reward identically.

Time (docs/outdoor_madrl_redesign.md section 5.4): a decision is taken every
ACTION_SCALE simulation steps and held until the next decision or the end of
the episode. The transition for that decision carries

    reward = sum_{i<k} gamma_step^i r_i      hold = k

where k is the number of steps the action actually lasted, so the first
decision (which used to be skipped) and a decision cut short by the end of the
episode are recorded like any other. `terminal` is set only when the task
ended the episode (DANGER_TERMINATION); running out of MAX_STEPS is a time
limit, and the learner still bootstraps from the final state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from sim import robot_action
from sim.observation import (DecisionRecord, ObservationHistory, StaticLayers,
                             build_static_layers)
from sim.rewards import COMPONENTS, TaskReward


@dataclass
class Transition:
    step: int                       # decision index within the episode
    record: DecisionRecord
    action: Optional[np.ndarray]    # (MAX_ROBOTS, A); None for the final state
    reward: float = 0.0             # discounted with the rollout's gamma
    hold: int = 0
    terminal: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    # Undiscounted per-step rewards, length `hold`. The learner discounts
    # these with its own gamma, so a gamma schedule cannot put the worker and
    # the learner out of step.
    step_rewards: List[float] = field(default_factory=list)


@dataclass
class EpisodeResult:
    steps: int = 0
    decisions: int = 0
    total_reward: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    terminated: bool = False
    command_norm_mean: float = 0.0
    robot_speed_mean: float = 0.0
    mode_switch_rate: float = 0.0
    mode_share: Dict[str, float] = field(default_factory=dict)
    inference_ms_mean: float = 0.0
    sim_ms_per_step: float = 0.0
    task: Dict[str, float] = field(default_factory=dict)


ActFn = Callable[[Dict[str, np.ndarray], DecisionRecord], np.ndarray]


def run_episode(model, cfg, act_fn: Optional[ActFn], *, gamma: float,
                max_steps: Optional[int] = None, seed: int = 0,
                static: Optional[StaticLayers] = None,
                emit: Optional[Callable[[Transition], None]] = None,
                task_metrics=None, needs_obs: bool = True,
                on_step: Optional[Callable] = None) -> EpisodeResult:
    """Run one episode to MAX_STEPS or to the task's own termination.

    act_fn(obs, record) returns one action row per real robot. It is None for
    the zero-command signal-off control, which holds every robot still with
    its signal off. `emit` receives each completed transition and finally the
    state after the last action (action None).
    """
    A = int(cfg.ACTION_SCALE)
    R = int(cfg.MAX_ROBOTS)
    max_steps = int(cfg.MAX_STEPS if max_steps is None else max_steps)
    g_step = float(gamma) ** (1.0 / max(1, A))
    static = static or build_static_layers(model, cfg)
    history = ObservationHistory(cfg, static, len(model.robots), seed=seed)
    reward = TaskReward(model, cfg)
    res = EpisodeResult(components={c: 0.0 for c in COMPONENTS})
    modes = tuple(cfg.ROBOT_MODES)
    mode_counts = {m: 0 for m in modes}
    prev_modes: Optional[List[str]] = None
    switches = 0
    cmd_norms: List[float] = []
    speeds: List[float] = []
    infer_ms: List[float] = []
    sim_ms = 0.0

    pending: Optional[Transition] = None
    g_acc = 1.0
    d = -1
    off = robot_action.encode((0.0, 0.0), "off")
    for step in range(max_steps):
        if step % A == 0:
            rec = history.record(model)
            d += 1
            if pending is not None and emit is not None:
                emit(pending)
            actions = np.zeros((R, robot_action.ACTION_DIM), np.float32)
            n = len(model.robots)
            if act_fn is None:
                actions[:n] = off
            else:
                obs = history.team_observations() if needs_obs else {}
                t0 = time.perf_counter()
                actions[:n] = act_fn(obs, rec)
                infer_ms.append((time.perf_counter() - t0) * 1000.0)
            now_modes = []
            for i, rb in enumerate(model.robots[:R]):
                move = robot_action.apply_to(rb, actions[i])
                # What the environment used, not what the policy asked for.
                actions[i, 0], actions[i, 1] = float(move[0]), float(move[1])
                cmd_norms.append(math.hypot(float(move[0]), float(move[1])))
                now_modes.append(rb.mode)
                mode_counts[rb.mode] += 1
            if prev_modes is not None:
                switches += sum(a != b for a, b in zip(prev_modes, now_modes))
            prev_modes = now_modes
            pending = Transition(step=d, record=rec, action=actions,
                                 components={c: 0.0 for c in COMPONENTS})
            g_acc = 1.0
        t0 = time.perf_counter()
        model.step()
        sim_ms += (time.perf_counter() - t0) * 1000.0
        comps = reward.step(model)
        r = sum(comps.values())
        pending.reward += g_acc * r
        pending.step_rewards.append(float(r))
        pending.hold += 1
        for c, v in comps.items():
            pending.components[c] += v
            res.components[c] += v
        res.total_reward += r
        g_acc *= g_step
        for rb in model.robots:
            speeds.append(math.hypot(float(rb.vel[0]), float(rb.vel[1])))
        res.steps = step + 1
        if task_metrics is not None:
            task_metrics.step(model)
        if on_step is not None:
            on_step(model, step + 1)
        if model.should_finish():
            pending.terminal = True
            res.terminated = True
            break

    if pending is not None and emit is not None:
        emit(pending)
        final = history.record(model)
        emit(Transition(step=pending.step + 1, record=final, action=None))
    res.decisions = d + 1
    res.command_norm_mean = float(np.mean(cmd_norms)) if cmd_norms else 0.0
    res.robot_speed_mean = float(np.mean(speeds)) if speeds else 0.0
    n_mode_obs = max(1, sum(mode_counts.values()))
    res.mode_share = {m: mode_counts[m] / n_mode_obs for m in modes}
    denom = max(1, (res.decisions - 1) * max(1, len(model.robots)))
    res.mode_switch_rate = switches / denom
    res.inference_ms_mean = float(np.mean(infer_ms)) if infer_ms else 0.0
    res.sim_ms_per_step = sim_ms / max(1, res.steps)
    if task_metrics is not None:
        res.task = task_metrics.summary()
    return res
