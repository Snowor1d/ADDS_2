"""Building an environment from a spec and running one controlled episode.

Training workers (learn/worker.py), evaluation (evaluation/), the curriculum
and the human-play adapter all start episodes through `build_env`, so a
scenario seeded the same way starts from the same crowd and robot placement
everywhere. `run_episode` is the evaluation loop; training has its own loop
because it also forms transitions and rewards, but it uses the same
observation stack and decision timing:

    decision at steps 0, D, 2D, ... (D = ACTION_SCALE): the stack is reset
    with the step-0 frame, and each later decision pushes its own frame;
    the action is held for D steps.

The evacuation time is the number of simulation steps run until nobody is
left (1-based, as ADDS_AS_experiment.py and ADDS_AS_HumanPlay.py counted
it), or the step cap when the episode did not finish.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from learn.obs import ObservationBuilder, StackedObservation, network_state


@dataclass
class EnvSpec:
    crowd: int
    map_num: Optional[int] = None
    level: Any = None                 # ued.level.Level
    augmentation: str = "identity"
    seed: Optional[int] = None

    def label(self) -> str:
        if self.level is not None:
            return f"level{getattr(self.level, 'level_id', '?')}"
        return f"map{self.map_num}"


def build_env(spec: EnvSpec, cfg, robot: str = "Q"):
    import model
    import sim_maps
    if spec.seed is not None:
        random.seed(int(spec.seed))
        np.random.seed(int(spec.seed) % (2 ** 32))
    if spec.level is not None:
        lv = spec.level
        return model.FightingModel(int(spec.crowd), int(lv.width), int(lv.height),
                                   robot=robot, level=lv, augmentation=spec.augmentation)
    w, h = sim_maps.map_size(int(spec.map_num), cfg)
    return model.FightingModel(int(spec.crowd), w, h, model_num=int(spec.map_num),
                               robot=robot, augmentation=spec.augmentation)


def freeflow_steps(env, cfg) -> float:
    """Steps until everyone would be out if each walked straight to the
    nearest exit at their own desired speed along the navmesh: a lower
    bound used to normalise a level's evacuation time."""
    worst = 0.0
    for ag in env.crowds:
        if ag.dead:
            continue
        _, q, d = env.nearest_exit(ag.xy)
        if d == 0:
            continue
        try:
            g = ag.point_to_point_distance(ag.xy, q)
        except Exception:
            g = d
        if not math.isfinite(g) or g > 1e8:
            g = d
        worst = max(worst, g / max(1e-6, float(ag.desired_speed_a)))
    return worst / float(cfg.AGENT_TIME_STEP)


def eval_step_cap(cfg, env) -> int:
    cap = int(cfg.EVAL_MAX_STEPS)
    if cfg.EVAL_MAX_STEPS_SCALE:
        cap = int(round(cap * max(env.width, env.height) / 100.0))
    return cap


class PolicyController:
    """The learned actor as a controller. `obs_cfg` is the configuration the
    policy was trained with (its observation contract)."""

    needs_obs = True

    def __init__(self, actor, obs_cfg, deterministic: bool = True, seed: Optional[int] = None):
        self.actor = actor.eval()
        self.cfg = obs_cfg
        self.decision_interval = int(obs_cfg.ACTION_SCALE)
        self.deterministic = bool(deterministic)
        self.use_scale = bool(getattr(actor, "use_scale", False))
        self.seed = seed
        # what the action means (learn/waypoint.py); run_episode follows it
        self.action_mode = str(getattr(obs_cfg, "ROBOT_ACTION_MODE", "velocity"))

    def reset(self, env, ob):
        if self.seed is not None and not self.deterministic:
            torch.manual_seed(int(self.seed))

    @torch.no_grad()
    def act(self, env, ob: ObservationBuilder, stack: StackedObservation, state5: np.ndarray):
        ego, glob = stack.network_input(ob.glob_static)
        ego_t = torch.from_numpy(ego)[None]
        glob_t = torch.from_numpy(glob)[None]
        st = torch.from_numpy(network_state(state5, self.use_scale))[None]
        if self.deterministic:
            a = self.actor.deterministic(ego_t, glob_t, st)
        else:
            a, _ = self.actor.sample(ego_t, glob_t, st)
        return a[0].numpy().astype(np.float64)


def _run_waypoint_episode(env, controller, cfg, max_steps: int, D: int, on_step) -> Dict[str, float]:
    """run_episode for a waypoint policy (learn/waypoint.py): frames are
    stacked every ACTION_SCALE window, the policy decides only when its
    decision ends, and the follower drives the robot in between. `cfg` is the
    policy's configuration (its waypoint settings)."""
    from learn.waypoint import WaypointFollower
    ob = ObservationBuilder(cfg, env)
    stack = StackedObservation(int(cfg.FRAME_STACK))
    controller.reset(env, ob)
    follower = WaypointFollower(env, cfg)
    total = env.total_agents
    t80 = t100 = None
    collisions = 0
    path_len = 0.0
    last_xy = list(env.robot.xy)
    decisions = 0
    deciding = True
    for step in range(max_steps):
        if step % D == 0:
            ego, glob, es = ob.frames()
            if step == 0:
                stack.reset(ego, glob, es)
            else:
                stack.push(ego, glob, es)
            if deciding:
                a = controller.act(env, ob, stack, ob.robot_state())
                follower.set(a)
                deciding = False
                decisions += 1
        env.robot.receive_action(list(follower.command()))
        env.step()
        collisions += int(bool(getattr(env.robot, "collision_check", 0)))
        xy = env.robot.xy
        path_len += math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1])
        last_xy = [xy[0], xy[1]]
        if on_step is not None:
            on_step(env, step + 1)
        alive = env.alived_agents()
        if t80 is None and alive < total * 0.2:
            t80 = step + 1
        if alive < 1:
            t100 = step + 1
        done = t100 is not None or step == max_steps - 1
        if step % D == D - 1 or done:
            if follower.end_of_tick(done):
                deciding = True
        if t100 is not None:
            break
    steps = step + 1
    finished = t100 is not None
    return {
        "evac_time_100": float(t100 if finished else max_steps),
        "evac_time_80": float(t80 if t80 is not None else max_steps),
        "success": float(finished),
        "steps": float(steps),
        "alive_end": float(env.alived_agents()),
        "total_lifetime": float(env.calculate_all_agents_life_time()),
        "robot_collision_steps": float(collisions),
        "robot_path_m": float(path_len),
        "decisions": float(decisions),
        "step_cap": float(max_steps),
    }


def run_episode(env, controller, cfg, max_steps: int,
                decision_interval: Optional[int] = None,
                on_step=None) -> Dict[str, float]:
    """Run until everyone is out or `max_steps`. `controller` is None for
    the robots that drive themselves inside model.step ('N', 'T').
    `cfg` is the observation configuration (the policy's own for a policy).
    `on_step(env, step)` is called after every simulation step (rendering,
    recording)."""
    D = int(decision_interval or getattr(controller, "decision_interval", None) or cfg.ACTION_SCALE)
    has_robot = getattr(env, "robot_version", "Q") != "N"
    active = controller is not None and has_robot
    if active and getattr(controller, "action_mode", "velocity") == "waypoint":
        return _run_waypoint_episode(env, controller, cfg, max_steps, D, on_step)
    ob = ObservationBuilder(cfg, env) if (active and getattr(controller, "needs_obs", True)) else None
    stack = StackedObservation(int(cfg.FRAME_STACK)) if ob is not None else None
    if active:
        controller.reset(env, ob)
    total = env.total_agents
    t80 = t100 = None
    collisions = 0
    path_len = 0.0
    last_xy = list(env.robot.xy) if has_robot else None
    decisions = 0
    for step in range(max_steps):
        if active and step % D == 0:
            state5 = None
            if ob is not None:
                ego, glob, es = ob.frames()
                if step == 0:
                    stack.reset(ego, glob, es)
                else:
                    stack.push(ego, glob, es)
                state5 = ob.robot_state()
            a = controller.act(env, ob, stack, state5)
            if a is not None:
                env.robot.receive_action([float(a[0]), float(a[1])])
            decisions += 1
        env.step()
        if has_robot:
            collisions += int(bool(getattr(env.robot, "collision_check", 0)))
            xy = env.robot.xy
            path_len += math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1])
            last_xy = [xy[0], xy[1]]
        if on_step is not None:
            on_step(env, step + 1)
        alive = env.alived_agents()
        if t80 is None and alive < total * 0.2:
            t80 = step + 1
        if alive < 1:
            t100 = step + 1
            break
    steps = step + 1
    finished = t100 is not None
    return {
        "evac_time_100": float(t100 if finished else max_steps),
        "evac_time_80": float(t80 if t80 is not None else max_steps),
        "success": float(finished),
        "steps": float(steps),
        "alive_end": float(env.alived_agents()),
        "total_lifetime": float(env.calculate_all_agents_life_time()),
        "robot_collision_steps": float(collisions),
        "robot_path_m": float(path_len),
        "decisions": float(decisions),
        "step_cap": float(max_steps),
    }
