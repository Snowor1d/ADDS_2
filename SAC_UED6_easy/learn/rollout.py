"""One episode, the same way everywhere.

The training worker, the validation and zero-shot evaluators and the tests
all run episodes through `run_episode`, so they measure, observe, act and
account reward identically.

Time (docs/outdoor_madrl_redesign.md section 5.4). Every robot is measured on
a fixed clock, one frame every `cfg.frame_steps()` simulation steps
(OBS_FRAME_PERIOD_S), and the team decides only on that clock. A decision is
held until ROBOT_DECISION_MAX_S_<MODE> has passed (a whole number of frames),
or, with ROBOT_DECISION_ON_EVENTS_<MODE>, until the first frame after some
robot arrived at its waypoint or was blocked by a wall (checked after every
step), or until the episode ends. The transition for that decision carries

    reward = sum_{i<k} gamma_step^i r_i      hold = k

where k is the number of steps the action actually lasted, so the first
decision (which used to be skipped) and a decision cut short by the end of the
episode are recorded like any other. `terminal` is set only when the task
ended the episode (DANGER_TERMINATION); running out of MAX_STEPS is a time
limit, and the learner still bootstraps from the final state.

`emit` receives every frame, in time order: a decision frame as its completed
transition (once its hold has ended), then the frames taken during that hold
(`frame_only`, no action), and at the end the state after the last action
(action None, not frame_only). Transition.step is the frame's index in the
episode; the replay links frames by it to rebuild history windows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from configs import decision_max_steps, decision_on_events, frame_steps
from sim import robot_action
from sim.observation import (DecisionRecord, ObservationHistory, StaticLayers,
                             build_static_layers)
from sim.rewards import (COMPONENTS, PER_ROBOT_VERSIONS, ROBOT_TERMS,
                         SHAPING_TERMS, TaskReward)


@dataclass
class Transition:
    step: int                       # frame index within the episode
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
    # Per step, one value per robot: its own share of the robot-charged
    # terms (sim/rewards.ROBOT_TERMS), already weighted and negative. Under
    # PER_ROBOT_VERSIONS those terms are not in `step_rewards` but charged to
    # each robot from these; under rew-v2 they are and these are unused.
    robot_penalties: List[List[float]] = field(default_factory=list)
    # A frame taken while a decision was held: history only, no action.
    frame_only: bool = False
    # Learning-only shaping (sim/rewards.SHAPING_TERMS) inside step_rewards
    # and robot_penalties over the hold, undiscounted and summed over the
    # team: what to take off to get back the task reward.
    shaping: float = 0.0


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
    # Where the robots were, sampled on the observation clock (ROBOT_KEYS).
    robots: Dict[str, float] = field(default_factory=dict)


# Means over frames: robots inside the hazard (share of robot-frames), how
# deep they were (metres past the boundary, 0 outside), metres between pairs
# of robots, and persons following a robot, per robot.
ROBOT_KEYS = ("robot_inside_share", "robot_depth_m", "robot_pair_distance_m",
              "followers_per_robot")


def _robot_sample(model, acc: Dict[str, List[float]]) -> None:
    robots = list(model.robots)
    if not robots:
        return
    xy = [(float(rb.xy[0]), float(rb.xy[1])) for rb in robots]
    zone = getattr(model, "danger_zone", None)
    if zone is not None:
        sd = [float(zone.signed_distance(x, y)) for x, y in xy]
        acc["robot_inside_share"].extend(float(v < 0.0) for v in sd)
        acc["robot_depth_m"].extend(max(0.0, -v) for v in sd)
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            acc["robot_pair_distance_m"].append(
                math.hypot(xy[i][0] - xy[j][0], xy[i][1] - xy[j][1]))
    following = sum(1 for a in model.crowds
                    if not a.dead and getattr(a, "following_robot_id", None)
                    is not None)
    acc["followers_per_robot"].append(following / len(robots))


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
    K = decision_max_steps(cfg)
    F = frame_steps(cfg)
    if K % F:
        raise ValueError(f"the longest decision ({K} steps) is not a whole "
                         f"number of observation frames ({F} steps)")
    on_events = decision_on_events(cfg)
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
    where: Dict[str, List[float]] = {k: [] for k in ROBOT_KEYS}

    pending: Optional[Transition] = None
    held: List[Transition] = []      # frames taken during the pending hold
    g_acc = 1.0
    d = -1
    frame = -1
    event = False                    # some robot arrived or was blocked
    off = robot_action.encode((0.0, 0.0), "off")
    for step in range(max_steps):
        due = False
        if step % F == 0:
            rec = history.record(model)
            frame += 1
            _robot_sample(model, where)
            due = (pending is None or pending.hold >= K
                   or (on_events and event))
            if not due:
                held.append(Transition(step=frame, record=rec, action=None,
                                       frame_only=True))
        if due:
            d += 1
            event = False
            if pending is not None and emit is not None:
                emit(pending)
                for fr in held:
                    emit(fr)
            held = []
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
            pending = Transition(step=frame, record=rec, action=actions,
                                 components={c: 0.0 for c in COMPONENTS})
            g_acc = 1.0
        t0 = time.perf_counter()
        model.step()
        sim_ms += (time.perf_counter() - t0) * 1000.0
        comps = reward.step(model)
        r = sum(comps.values())                    # what the learner is given
        shaped = sum(comps[t] for t in SHAPING_TERMS)
        pending.shaping += shaped
        pending.reward += g_acc * r
        if cfg.REWARD_VERSION in PER_ROBOT_VERSIONS:
            pending.step_rewards.append(
                float(r - sum(comps[t] for t in ROBOT_TERMS)))
        else:
            pending.step_rewards.append(float(r))
        pending.robot_penalties.append(list(reward.last_robot_penalties))
        pending.hold += 1
        for c, v in comps.items():
            pending.components[c] += v
            res.components[c] += v
        res.total_reward += r - shaped            # the task reward only
        g_acc *= g_step
        if on_events and any(getattr(rb, "decision_event", None)
                             for rb in model.robots[:R]):
            # Held until the next frame: decisions are taken on the clock.
            event = True
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
            # A terminal state's potential is zero: the last shaping term is
            # -Phi(s_{T-1}), not g Phi(s_T) - Phi(s_{T-1}).
            team, own = reward.terminal_correction()
            if team or any(own):
                if cfg.REWARD_VERSION in PER_ROBOT_VERSIONS:
                    pending.step_rewards[-1] += team
                    for j, v in enumerate(own):
                        if j < len(pending.robot_penalties[-1]):
                            pending.robot_penalties[-1][j] += v
                else:
                    pending.step_rewards[-1] += team + sum(own)
                pending.reward += g_acc / g_step * (team + sum(own))
                pending.shaping += team + sum(own)
                pending.components["shape_progress"] += team
                pending.components["shape_approach"] += sum(own)
                res.components["shape_progress"] += team
                res.components["shape_approach"] += sum(own)
            break

    if pending is not None and emit is not None:
        emit(pending)
        for fr in held:
            emit(fr)
        final = history.record(model)
        emit(Transition(step=frame + 1, record=final, action=None))
    res.decisions = d + 1
    res.command_norm_mean = float(np.mean(cmd_norms)) if cmd_norms else 0.0
    res.robot_speed_mean = float(np.mean(speeds)) if speeds else 0.0
    n_mode_obs = max(1, sum(mode_counts.values()))
    res.mode_share = {m: mode_counts[m] / n_mode_obs for m in modes}
    denom = max(1, (res.decisions - 1) * max(1, len(model.robots)))
    res.mode_switch_rate = switches / denom
    res.inference_ms_mean = float(np.mean(infer_ms)) if infer_ms else 0.0
    res.sim_ms_per_step = sim_ms / max(1, res.steps)
    res.robots = {k: float(np.mean(v)) for k, v in where.items() if v}
    if task_metrics is not None:
        res.task = task_metrics.summary()
    return res
