"""Exploration-structure probe: what random behaviour reaches, per action parameterisation.

Runs fixed validation levels (same seeds as periodic validation) under
  random4s        : the trainer's warm-up exploration (uniform action every 4 s)
  random_commit   : a uniform random walkable target, held until arrival or 30 s,
                    random mode per leg (temporally extended random exploration)
  beacon_edge     : each robot walks to the hazard edge and stands there guiding
Records hazard person-steps and robot coverage of the hazard interior.
"""
import os, sys, time, copy, random, json, math
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(1)
from configs import resolve_config
from learn.zero_shot import validation_levels, build_eval_model, validation_seed, EpisodeMetrics
from learn.rollout import run_episode
from learn.sac import exploration_action
from sim import robot_action as ra

scen_idx, policy, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = resolve_config(check_data=False)
name, lv = validation_levels(cfg)[scen_idx]
seed = validation_seed(name)
lv = copy.deepcopy(lv); lv.robot_num = 3
m = build_eval_model(lv, seed)
m.should_finish = lambda: False
zone = m.danger_zone
prng = random.Random(seed + 12345)
dt_dec = float(cfg.decision_max_steps()) * float(cfg.AGENT_TIME_STEP)
lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)

def enc_toward(rb, tx, ty, speed, mode):
    dx, dy = tx - float(rb.xy[0]), ty - float(rb.xy[1])
    s = max(1.0, abs(dx) / lim, abs(dy) / lim)
    return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)

walk_pts = [((t[0][0]+t[1][0]+t[2][0])/3, (t[0][1]+t[1][1]+t[2][1])/3) for t in m.pure_mesh]
near_hz = [p for p in walk_pts if zone.signed_distance(*p) < 15.0]
legs = [None] * len(m.robots)

def act(obs, rec):
    rows = []
    now = float(m.step_count) * float(cfg.AGENT_TIME_STEP)
    for i, rb in enumerate(m.robots):
        if policy == "random4s":
            rows.append(exploration_action(prng)); continue
        if policy == "random_commit":
            leg = legs[i]
            x, y = float(rb.xy[0]), float(rb.xy[1])
            if leg is None or math.hypot(leg["t"][0]-x, leg["t"][1]-y) < 2.5 or now - leg["start"] > 30.0:
                t = near_hz[prng.randrange(len(near_hz))]
                leg = legs[i] = {"t": t, "start": now, "mode": prng.choice(["off", "guide"]), "speed": prng.uniform(0.3, 1.0)}
            rows.append(enc_toward(rb, leg["t"][0], leg["t"][1], leg["speed"], leg["mode"])); continue
        if policy == "beacon_edge":
            leg = legs[i]
            if leg is None:
                # nearest walkable point just outside the edge
                cands = [p for p in walk_pts if 1.0 <= zone.signed_distance(*p) <= 4.0]
                x, y = float(rb.xy[0]), float(rb.xy[1])
                t = min(cands, key=lambda p: math.hypot(p[0]-x, p[1]-y)) if cands else (x, y)
                leg = legs[i] = {"t": t}
            rows.append(enc_toward(rb, leg["t"][0], leg["t"][1], 1.0, "guide")); continue
        raise ValueError(policy)
    return np.stack(rows)

inside_t = [0] * len(m.robots); depth_min = [0.0] * len(m.robots); steps = [0]
def on_step(model, k):
    steps[0] += 1
    for i, rb in enumerate(model.robots):
        d = zone.signed_distance(float(rb.xy[0]), float(rb.xy[1]))
        if d < 0: inside_t[i] += 1
        depth_min[i] = min(depth_min[i], d)
tm = EpisodeMetrics().start(m)
t0 = time.time()
res = run_episode(m, cfg, act, gamma=cfg.gamma(), seed=seed, task_metrics=tm, needs_obs=False, on_step=on_step)
r = zone.radius if getattr(zone, "radius", None) else None
outd = {"scenario": name, "policy": policy, "hazard_person_steps": res.task.get("hazard_person_steps"),
        "activity_dwell_hazard_person_steps": res.task.get("activity_dwell_hazard_person_steps"),
        "ever_acted": res.task.get("ever_acted"), "followed_departures": res.task.get("followed_departures"),
        "informed_departures": res.task.get("informed_departures"), "total_reward": res.total_reward,
        "robot_inside_frac": float(np.mean(inside_t)) / max(1, steps[0]), "max_depth_m": [-d for d in depth_min],
        "hazard_radius": r, "robot_speed_mean": res.robot_speed_mean, "secs": time.time() - t0}
json.dump(outd, open(out, "w")); print(json.dumps(outd))
