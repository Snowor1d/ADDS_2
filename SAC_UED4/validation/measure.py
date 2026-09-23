"""The measurements themselves.

Each returns a dict of numbers, and each is compared against a published
value in the report rather than against a threshold invented here.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np

from validation.scenarios import (bottleneck_geometry, bottleneck_level,
                                  corridor_level)

# Published references the measurements are held against. Sources are named
# in docs/crowd_validation.md; the numbers are the ones a reader of that
# literature would expect a pedestrian model to reproduce.
REFERENCE = {
    "free_speed_mean_ms": 1.34,        # Weidmann, mean free walking speed
    "specific_flow_pms": 1.2,          # persons per metre of door per second
    "jam_density_ped_m2": 5.4,
    "weidmann_gamma": 1.913,
}


def weidmann_speed(density: float, free_speed: float = 1.34) -> float:
    """Weidmann's fitted walking speed at a given density.

    v = v0 * (1 - exp(-gamma * (1/D - 1/D_jam))), with gamma 1.913 and a jam
    density of 5.4 per square metre. Written out rather than tabulated so the
    model can be held against the curve at its own free speed: the intercept
    is a configuration choice here, the shape is the claim being tested.
    """
    if density <= 0:
        return free_speed
    g = REFERENCE["weidmann_gamma"]
    jam = REFERENCE["jam_density_ped_m2"]
    if density >= jam:
        return 0.0
    return free_speed * (1.0 - math.exp(-g * (1.0 / density - 1.0 / jam)))


def _build(level, n_agents: int, seed: int = 0):
    """A model on a hand-built level, with the crowd placed by the caller.

    The model is built with a token crowd and the rest are added directly,
    because the simulator's own placement enforces a minimum spacing by
    rejection sampling. That is right for a level at the densities levels
    actually use, and useless here: these tests deliberately run at two and
    three pedestrians per square metre, where no rejection sampler will ever
    find a free slot. The harness places every pedestrian itself anyway.
    """
    import sim.model as M
    random.seed(seed)
    np.random.seed(seed)
    seed_crowd = min(int(n_agents), 4)
    level.crowd_size = seed_crowd
    env = M.FightingModel(seed_crowd, int(level.width), int(level.height),
                          robot="Q", level=level)
    _populate(env, int(n_agents))
    return env


def _populate(env, n_agents: int):
    """Add pedestrians up to n_agents, bypassing the spacing rule."""
    from sim.agent import CrowdAgent

    while len(_crowd(env)) < n_agents:
        a = CrowdAgent(env.agent_num, env, [1.0, 1.0], 1)
        env.crowds.append(a)
        env.agent_num += 1
        env.agent_id += 1
        env.schedule.add(a)
        env.space.place_agent(a, (1.0, 1.0))
        env.agents.append(a)
        env.total_agents += 1


def _crowd(env):
    return [a for a in env.crowds if a.type != 3 and not a.dead]


def _place(env, agent, x: float, y: float):
    agent.xy = [float(x), float(y)]
    agent.pos = (float(x), float(y))
    agent.vel = [0.0, 0.0]
    agent.acc = [0.0, 0.0]
    agent._progress_anchor = None
    agent._wedged_for = 0
    env.space.move(agent.unique_id, agent.xy)


def _park_robots(env):
    """Put the robots out of the way.

    Every scenario here is about the crowd. A robot standing in a corridor
    would cue everyone near it and change what is being measured.
    """
    for rb in getattr(env, "robots", []):
        _place(env, rb, 0.5, 0.5)
        rb.scripted_goal = (0.5, 0.5)


# Densities the test is run at. The ceiling is set by the body: hexagonal
# packing of discs of radius r tops out at 0.9069 / (pi r^2) per square metre,
# which is 4.6 at the calibrated radius of 0.25 m and was 1.15 at the old
# 0.5 m. The published curve runs to a jam density of about 5.4, so at the
# old radius most of it was outside the model's geometry.
FD_DENSITIES = (0.1, 0.5, 1.0, 2.0, 3.0)


def fundamental_diagram(densities=FD_DENSITIES,
                        length: float = 40.0, width: float = 4.0,
                        margin: float = 6.0, warmup: int = 30,
                        measure: int = 50, seed: int = 0,
                        verbose: bool = True) -> List[Dict]:
    """Speed against density in a periodic corridor.

    The first test any pedestrian model is asked to pass. Everyone walks the
    same way down a corridor and anyone leaving the far end re-enters at the
    near one, which is how the published measurements keep density constant
    while the flow runs.
    """
    from config import AGENT_TIME_STEP

    x0, x1 = margin, margin + length
    y0, y1 = margin, margin + width
    out = []
    for dens in densities:
        n = max(2, int(round(dens * length * width)))
        env = _build(corridor_level(length, width, margin), n, seed)
        _park_robots(env)
        crowd = _crowd(env)
        rng = random.Random(seed + 1)
        for a in crowd:
            _place(env, a,
                   rng.uniform(x0 + 0.5, x1 - 0.5),
                   rng.uniform(y0 + 0.6, y1 - 0.6))

        speeds = []
        for step in range(warmup + measure):
            for a in crowd:
                a.scripted_goal = (a.xy[0] + 20.0, a.xy[1])
            prev = {a.unique_id: (a.xy[0], a.xy[1]) for a in crowd}
            env.step()
            for a in crowd:
                if a.xy[0] > x1:
                    _place(env, a, a.xy[0] - length, a.xy[1])
                    prev[a.unique_id] = (prev[a.unique_id][0] - length,
                                         prev[a.unique_id][1])
                elif a.xy[0] < x0:
                    _place(env, a, a.xy[0] + length, a.xy[1])
                    prev[a.unique_id] = (prev[a.unique_id][0] + length,
                                         prev[a.unique_id][1])
            if step >= warmup:
                for a in crowd:
                    px, py = prev[a.unique_id]
                    speeds.append((a.xy[0] - px) / float(AGENT_TIME_STEP))
        row = {
            "density_ped_m2": float(dens),
            "n": n,
            "flow_speed_ms": float(np.mean(speeds)),
            "speed_sd": float(np.std(speeds)),
        }
        out.append(row)
        if verbose:
            print(f"    density {dens:.2f}  n={n:4d}  "
                  f"speed {row['flow_speed_ms']:.3f} m/s", flush=True)
    return out


def free_speed(length: float = 60.0, width: float = 5.0, margin: float = 6.0,
               trials: int = 8, steps: int = 120, seed: int = 0) -> Dict:
    """Walking speed with nobody in the way.

    One pedestrian per run, not a line of them. Twenty pedestrians spaced
    along a five-metre corridor are in single file, and a fast one catching a
    slow one turns the measurement into a queue: the first version of this
    test reported 1.27 m/s with a standard deviation of 0.49 for exactly that
    reason. Free speed has to be measured free.
    """
    from config import AGENT_SPEED_MEAN, AGENT_TIME_STEP

    desired = []
    achieved = []
    for t in range(trials):
        env = _build(corridor_level(length, width, margin), 1, seed + t)
        _park_robots(env)
        crowd = _crowd(env)
        if not crowd:
            continue
        a = crowd[0]
        _place(env, a, margin + 1.0, margin + width / 2.0)
        desired.append(a.desired_speed_a)
        travelled = 0.0
        for _ in range(steps):
            a.scripted_goal = (a.xy[0] + 20.0, a.xy[1])
            prev = a.xy[0]
            env.step()
            if a.xy[0] > margin + length:
                _place(env, a, a.xy[0] - length, a.xy[1])
                prev -= length
            travelled += a.xy[0] - prev
        achieved.append(travelled / (steps * float(AGENT_TIME_STEP)))
    return {
        "configured_mean_ms": float(AGENT_SPEED_MEAN),
        "drawn_mean_ms": float(np.mean(desired)),
        "achieved_mean_ms": float(np.mean(achieved)),
        "achieved_sd": float(np.std(achieved)),
        "trials": len(achieved),
    }


def bottleneck_flow(n: int = 80, door_width: float = 1.2, room: float = 20.0,
                    margin: float = 4.0, max_steps: int = 1200,
                    speed_scale: float = 1.0, seed: int = 0,
                    queue_depth: float = 5.0) -> Dict:
    """Specific flow through a door, persons per metre of width per second.

    The second standard test. The published figure for a door in a wall is
    about 1.2 to 1.3, and it is the number the free-flow evacuation estimate
    in this project already assumes.
    """
    from config import AGENT_TIME_STEP

    level = bottleneck_level(room, door_width, margin=margin)
    x_wall, y_mid, _ = bottleneck_geometry(room, door_width, margin)
    env = _build(level, n, seed)
    _park_robots(env)
    crowd = _crowd(env)
    # The crowd waits directly in front of the opening, which is how the
    # published bottleneck experiments are set up and the only way to measure
    # the door rather than the walk to it. Spread through the room instead,
    # as this did at first, the door is never saturated: measured with 80
    # pedestrians in a 20 m room only 0 to 1 of them was in the doorway at any
    # step, so the number coming out was the rate they crossed the room, not
    # the capacity of the door.
    rng = random.Random(seed + 2)
    depth = max(2.0, float(queue_depth))
    rows = max(1, int(n * 0.6 / max(1.0, room)))
    for i, a in enumerate(crowd):
        _place(env, a,
               x_wall - 0.6 - depth * ((i % max(1, rows)) + rng.random())
               / max(1, rows),
               rng.uniform(margin + 0.6, margin + room - 0.6))
        a.desired_speed_a *= float(speed_scale)

    # Route through the navmesh to a triangle beyond the door, the way a
    # pedestrian in a level reaches anywhere.
    target = crowd[0].choice_safe_mesh([x_wall + 6.0, y_mid]) if crowd else None
    crossed = {}
    line = x_wall + 1.5
    for step in range(max_steps):
        for a in crowd:
            if target is not None:
                a.scripted_mesh = target
            else:
                a.scripted_goal = (x_wall + 8.0, y_mid)
        env.step()
        for a in crowd:
            if a.unique_id not in crossed and a.xy[0] > line:
                crossed[a.unique_id] = step
        if len(crossed) >= n:
            break

    times = sorted(crossed.values())
    if len(times) < max(4, n // 10):
        # Too few to measure a rate. Reported rather than raised: a door
        # narrower than a pedestrian's own diameter passes nobody, and that
        # is itself a result.
        return {"door_width_m": float(door_width), "crossed": len(crossed),
                "n": n, "specific_flow_pms": 0.0, "flow_ps": 0.0,
                "steady_window_steps": 0, "t90_steps": None,
                "speed_scale": float(speed_scale)}
    # Steady phase only: drop the first and last tenth, where the queue is
    # still forming and then draining.
    lo = times[len(times) // 10]
    hi = times[-max(1, len(times) // 10)]
    span = max(1, hi - lo) * float(AGENT_TIME_STEP)
    passed = sum(1 for t in times if lo <= t <= hi)
    return {
        "door_width_m": float(door_width),
        "n": n,
        "crossed": len(crossed),
        "speed_scale": float(speed_scale),
        "steady_window_steps": int(hi - lo),
        "flow_ps": passed / span,
        "specific_flow_pms": passed / span / float(door_width),
        "t90_steps": times[int(0.9 * len(times)) - 1] if times else None,
    }


def faster_is_slower(scales=(1.0, 2.0, 3.0, 4.0), n: int = 80,
                     door_width: float = 3.0, seed: int = 0) -> List[Dict]:
    """Whether pushing harder empties the room more slowly.

    A signature of the contact and friction terms rather than of any rule:
    at high desired speeds the arch in front of the door becomes stable and
    the flow drops. A model without it has no crowd pressure.
    """
    out = []
    for s in scales:
        r = bottleneck_flow(n=n, door_width=door_width, speed_scale=s,
                            seed=seed)
        out.append(r)
    return out


def lane_formation(length: float = 60.0, width: float = 6.0,
                   margin: float = 6.0, density: float = 1.0,
                   warmup: int = 20, steps: int = 200, seed: int = 0,
                   lane_dy: float = 0.8, lane_dx: float = 2.0) -> Dict:
    """Whether counterflow separates into lanes.

    The order parameter is the share of a pedestrian's near neighbours
    travelling the same way, averaged over the crowd. Half is a perfectly
    mixed crowd and one is complete separation. Reported at the start and at
    the end, because the claim is that it rises.
    """
    x0, x1 = margin, margin + length
    y0, y1 = margin, margin + width
    n = max(4, int(round(density * length * width)))
    env = _build(corridor_level(length, width, margin), n, seed)
    _park_robots(env)
    crowd = _crowd(env)
    rng = random.Random(seed + 3)
    for i, a in enumerate(crowd):
        _place(env, a, rng.uniform(x0 + 0.5, x1 - 0.5),
               rng.uniform(y0 + 0.6, y1 - 0.6))
        a._lane_dir = 1.0 if i % 2 == 0 else -1.0

    def order():
        vals = []
        for a in crowd:
            same = tot = 0
            for b in crowd:
                if b is a:
                    continue
                # A lane is about a body wide, so the neighbourhood has to
                # be narrower than one. A two-metre box spans a whole
                # four-metre corridor and reports a mixed crowd whatever the
                # lanes do, which is what the first version of this measure
                # did.
                if (abs(a.xy[0] - b.xy[0]) < lane_dx
                        and abs(a.xy[1] - b.xy[1]) < lane_dy):
                    tot += 1
                    if b._lane_dir == a._lane_dir:
                        same += 1
            if tot:
                vals.append(same / tot)
        return float(np.mean(vals)) if vals else float("nan")

    start = None
    for step in range(warmup + steps):
        for a in crowd:
            a.scripted_goal = (a.xy[0] + 20.0 * a._lane_dir, a.xy[1])
        env.step()
        for a in crowd:
            if a.xy[0] > x1:
                _place(env, a, a.xy[0] - length, a.xy[1])
            elif a.xy[0] < x0:
                _place(env, a, a.xy[0] + length, a.xy[1])
        if step == warmup:
            start = order()
    return {"density_ped_m2": density, "n": n,
            "order_start": start, "order_end": order()}
