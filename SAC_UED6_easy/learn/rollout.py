"""One episode, the same way everywhere.

The training worker, the validation and zero-shot evaluators and the tests
all run episodes through `run_episode`, so they measure, observe, act and
account reward identically.

Time (docs/outdoor_madrl_redesign.md section 5.4, docs/async_decisions.md).
Every robot is measured on a fixed clock, one frame every `cfg.frame_steps()`
simulation steps (OBS_FRAME_PERIOD_S), and robots decide only on that clock.
Each robot holds its own decision until ROBOT_DECISION_MAX_S_<MODE> has
passed (a whole number of frames), or, with ROBOT_DECISION_ON_EVENTS_<MODE>,
until the first frame after it arrived at its waypoint or was blocked by a
wall (checked after every step), or until the episode ends. Under
ROBOT_DECISION_SCOPE = "robot" only the robots whose own decision ended decide
at a frame; the others keep what they are doing. Under "team" every robot
decides whenever any robot's decision ends.

`emit` receives every frame once, in time order, after the steps that follow
it have run:

    action     every robot's current action at this frame, (MAX_ROBOTS, A),
               when some robot decided here; None otherwise. A robot that
               decided has its new action (with the move the environment
               accepted); one that kept its decision has it as it stands now,
               the remaining way to its waypoint from where it is.
    decided    which robots took a new decision at this frame
    step_rewards, robot_penalties
               the learning reward of each step from this frame to the next
               (at most frame_steps of them): the team's part, and per step
               each robot's own share of the robot-charged terms
    terminal   the task ended the episode within these steps

and at the end the state after the last step (`final`, every robot's
decision ends there). The replay joins a robot's decision to its next one
and discounts the steps between with the learner's gamma. `terminal` is set
only when the task ended the episode (DANGER_TERMINATION); running out of
MAX_STEPS is a time limit, and the learner still bootstraps from the final
state. Transition.step is the frame's index in the episode; the replay links
frames by it to rebuild history windows.
"""

from __future__ import annotations

import inspect
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
    # (MAX_ROBOTS, A) when some robot decided at this frame (and at the final
    # frame, all zeros), else None.
    action: Optional[np.ndarray] = None
    # (MAX_ROBOTS,) robots that took a new decision at this frame.
    decided: Optional[np.ndarray] = None
    # Undiscounted per-step rewards from this frame to the next. Under
    # PER_ROBOT_VERSIONS the robot-charged terms (sim/rewards.ROBOT_TERMS) are
    # not in `step_rewards` but in `robot_penalties`, per step and robot,
    # already weighted; under rew-v2 they are in both and the per-robot list
    # is unused.
    step_rewards: List[float] = field(default_factory=list)
    robot_penalties: List[List[float]] = field(default_factory=list)
    terminal: bool = False
    # The state after the episode's last step: no steps follow it.
    final: bool = False
    components: Dict[str, float] = field(default_factory=dict)
    # Learning-only shaping (sim/rewards.SHAPING_TERMS) inside step_rewards
    # and robot_penalties, undiscounted and summed over the team: what to
    # take off to get back the task reward.
    shaping: float = 0.0


@dataclass
class EpisodeResult:
    steps: int = 0
    decisions: int = 0              # frames at which some robot decided
    robot_decisions: int = 0        # decisions summed over robots
    decision_seconds_mean: float = 0.0
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


def ongoing_action(rb, action: np.ndarray, cfg) -> np.ndarray:
    """A robot's current decision as it stands now. Under waypoint actions
    the move is re-expressed as the remaining offset from where the robot is
    to the waypoint it is walking to, so a decision taken earlier reads the
    same as a new decision for that waypoint would; under velocity actions
    the held command already is what the robot does next."""
    a = np.asarray(action, np.float32).copy()
    if cfg.ROBOT_ACTION_MODE != "waypoint":
        return a
    wp = getattr(rb, "_waypoint", None)
    if wp is None:
        return a                    # not walked yet: the decision as given
    (gx, gy), _ = wp
    dx, dy = float(gx) - float(rb.xy[0]), float(gy) - float(rb.xy[1])
    rng = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    s = max(1.0, abs(dx) / rng, abs(dy) / rng)
    a[robot_action.MOVE] = (2.0 * dx / (rng * s), 2.0 * dy / (rng * s))
    return a


def decision_holds(frames: List[Transition], robot: int) -> List[int]:
    """The steps each of a robot's decisions lasted, in order, from the
    frames run_episode emitted: from a frame where it decided to its next
    decision or the end of the episode."""
    holds: List[int] = []
    for t in frames:
        if (t.action is not None and not t.final and t.decided is not None
                and t.decided[robot]):
            holds.append(0)
        if holds:
            holds[-1] += len(t.step_rewards)
    return holds


ActFn = Callable[..., np.ndarray]


def _act_caller(act_fn: Optional[ActFn]):
    """act_fn(obs, record) returns one action row per real robot; only the
    rows of the robots that decide are applied. A function with a parameter
    named `deciding` is also told which robots decide (a scripted baseline
    must not advance the state machine of a robot that keeps its
    decision)."""
    if act_fn is None:
        return None
    try:
        wants = "deciding" in inspect.signature(act_fn).parameters
    except (TypeError, ValueError):
        wants = False
    if wants:
        return lambda obs, rec, deciding: act_fn(obs, rec, deciding=deciding)
    return lambda obs, rec, deciding: act_fn(obs, rec)


def run_episode(model, cfg, act_fn: Optional[ActFn], *, gamma: float,
                max_steps: Optional[int] = None, seed: int = 0,
                static: Optional[StaticLayers] = None,
                emit: Optional[Callable[[Transition], None]] = None,
                task_metrics=None, needs_obs: bool = True,
                on_step: Optional[Callable] = None) -> EpisodeResult:
    """Run one episode to MAX_STEPS or to the task's own termination.

    act_fn(obs, record) returns one action row per real robot (see
    `_act_caller`). It is None for the zero-command signal-off control, which holds every
    robot still with its signal off. `emit` receives every frame (see the
    module docstring) and finally the state after the last step. `gamma`
    is kept for the callers' signatures; rewards are emitted undiscounted.
    """
    K = decision_max_steps(cfg)
    F = frame_steps(cfg)
    if K % F:
        raise ValueError(f"the longest decision ({K} steps) is not a whole "
                         f"number of observation frames ({F} steps)")
    on_events = decision_on_events(cfg)
    team_scope = getattr(cfg, "ROBOT_DECISION_SCOPE", "robot") == "team"
    R = int(cfg.MAX_ROBOTS)
    A = robot_action.ACTION_DIM
    max_steps = int(cfg.MAX_STEPS if max_steps is None else max_steps)
    static = static or build_static_layers(model, cfg)
    history = ObservationHistory(cfg, static, len(model.robots), seed=seed)
    reward = TaskReward(model, cfg)
    per_robot = cfg.REWARD_VERSION in PER_ROBOT_VERSIONS
    call = _act_caller(act_fn)
    res = EpisodeResult(components={c: 0.0 for c in COMPONENTS})
    robots = list(model.robots[:R])
    n = len(robots)
    modes = tuple(cfg.ROBOT_MODES)
    mode_counts = {m: 0 for m in modes}
    prev_modes: List[Optional[str]] = [None] * n
    switches = 0
    cmd_norms: List[float] = []
    speeds: List[float] = []
    infer_ms: List[float] = []
    sim_ms = 0.0
    where: Dict[str, List[float]] = {k: [] for k in ROBOT_KEYS}

    current = np.zeros((R, A), np.float32)     # each robot's held decision
    hold: List[Optional[int]] = [None] * n     # steps since its decision
    event = [False] * n                        # it arrived or was blocked
    held_steps: List[int] = []                 # length of every ended decision
    pending: Optional[Transition] = None       # the frame whose steps run
    frame = -1
    off = robot_action.encode((0.0, 0.0), "off")
    for step in range(max_steps):
        if step % F == 0:
            if pending is not None and emit is not None:
                emit(pending)
            rec = history.record(model)
            frame += 1
            _robot_sample(model, where)
            deciding = np.zeros(R, bool)
            for i in range(n):
                deciding[i] = (hold[i] is None or hold[i] >= K
                               or (on_events and event[i]))
            if team_scope and deciding[:n].any():
                deciding[:n] = True
            pending = Transition(step=frame, record=rec,
                                 components={c: 0.0 for c in COMPONENTS})
            if deciding.any():
                res.decisions += 1
                if call is None:
                    rows = np.tile(off, (n, 1))
                else:
                    obs = history.team_observations() if needs_obs else {}
                    t0 = time.perf_counter()
                    rows = np.asarray(call(obs, rec, deciding[:n].copy()),
                                      np.float32)
                    infer_ms.append((time.perf_counter() - t0) * 1000.0)
                for i, rb in enumerate(robots):
                    if not deciding[i]:
                        continue
                    if hold[i] is not None:
                        held_steps.append(hold[i])
                    move = robot_action.apply_to(rb, rows[i])
                    current[i] = rows[i]
                    # What the environment used, not what the policy asked for.
                    current[i, 0], current[i, 1] = float(move[0]), float(move[1])
                    cmd_norms.append(math.hypot(float(move[0]), float(move[1])))
                    mode_counts[rb.mode] += 1
                    if prev_modes[i] is not None and prev_modes[i] != rb.mode:
                        switches += 1
                    prev_modes[i] = rb.mode
                    hold[i] = 0
                    event[i] = False
                    res.robot_decisions += 1
                acts = np.zeros((R, A), np.float32)
                for i, rb in enumerate(robots):
                    acts[i] = (current[i] if deciding[i]
                               else ongoing_action(rb, current[i], cfg))
                pending.action = acts
                pending.decided = deciding
        t0 = time.perf_counter()
        model.step()
        sim_ms += (time.perf_counter() - t0) * 1000.0
        comps = reward.step(model)
        r = sum(comps.values())                    # what the learner is given
        shaped = sum(comps[t] for t in SHAPING_TERMS)
        pending.shaping += shaped
        if per_robot:
            pending.step_rewards.append(
                float(r - sum(comps[t] for t in ROBOT_TERMS)))
        else:
            pending.step_rewards.append(float(r))
        pending.robot_penalties.append(list(reward.last_robot_penalties))
        for c, v in comps.items():
            pending.components[c] += v
            res.components[c] += v
        res.total_reward += r - shaped            # the task reward only
        for i, rb in enumerate(robots):
            hold[i] += 1
            if on_events and getattr(rb, "decision_event", None):
                # Acted on at the next frame: decisions are taken on the clock.
                event[i] = True
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
                if per_robot:
                    pending.step_rewards[-1] += team
                    for j, v in enumerate(own):
                        if j < len(pending.robot_penalties[-1]):
                            pending.robot_penalties[-1][j] += v
                else:
                    pending.step_rewards[-1] += team + sum(own)
                pending.shaping += team + sum(own)
                pending.components["shape_progress"] += team
                pending.components["shape_approach"] += sum(own)
                res.components["shape_progress"] += team
                res.components["shape_approach"] += sum(own)
            break

    held_steps.extend(h for h in hold if h is not None)
    if pending is not None and emit is not None:
        emit(pending)
        final = history.record(model)
        decided = np.zeros(R, bool)
        decided[:n] = True
        emit(Transition(step=frame + 1, record=final,
                        action=np.zeros((R, A), np.float32), decided=decided,
                        terminal=res.terminated, final=True))
    res.robot_decisions = max(res.robot_decisions, 0)
    res.decision_seconds_mean = (
        float(np.mean(held_steps)) * float(cfg.AGENT_TIME_STEP)
        if held_steps else 0.0)
    res.command_norm_mean = float(np.mean(cmd_norms)) if cmd_norms else 0.0
    res.robot_speed_mean = float(np.mean(speeds)) if speeds else 0.0
    n_mode_obs = max(1, sum(mode_counts.values()))
    res.mode_share = {m: mode_counts[m] / n_mode_obs for m in modes}
    res.mode_switch_rate = switches / max(1, res.robot_decisions - n)
    res.inference_ms_mean = float(np.mean(infer_ms)) if infer_ms else 0.0
    res.sim_ms_per_step = sim_ms / max(1, res.steps)
    res.robots = {k: float(np.mean(v)) for k, v in where.items() if v}
    if task_metrics is not None:
        res.task = task_metrics.summary()
    return res
