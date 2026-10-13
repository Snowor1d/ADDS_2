"""One-decision signal probe: base reward vs potential-based shaping.

From states reached by the learned stochastic policy, robot 0 takes one of
several actions for one decision (8 steps); the others take their policy
actions (fixed across branches). Common random numbers per replicate.
Measures, per action: the base reward over the decision, and two candidate
potentials (training-only, privileged):
  phi_prog  = -sum_{i inside} min(1, esc_i / D_ref) / N_ref
  phi_appr  = -min_i d(robot0, person i) / 50 over people inside the hazard
              who are not following a robot and not informed (capped at 50 m)
F = g^8 phi(s') - phi(s).
"""
import os, sys, time, copy, random, json, math
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(1)
from configs import resolve_config
from learn.zero_shot import validation_levels, build_eval_model, validation_seed
from learn.sac import SACAgent, to_torch, OBS_KEYS
from sim import robot_action as ra
from sim.observation import ObservationHistory, build_static_layers
from sim.rewards import TaskReward
from validation.shuttle_baseline import ShuttlePolicy, _centre

scen_idx, ckpt, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
T0S = [int(x) for x in os.environ.get("T0S", "6,20,40").split(",")]
M = int(os.environ.get("M", "3"))
cfg = resolve_config(check_data=False)
A = int(cfg.ACTION_SCALE); K = int(cfg.decision_max_steps()); g_step = cfg.gamma() ** (1.0 / A)
agent = SACAgent(cfg, device="cpu"); agent.load(ckpt); agent.policy.eval()
name, lv = validation_levels(cfg)[scen_idx]
seed = validation_seed(name)
m = build_eval_model(lv, seed); m.should_finish = lambda: False
static = build_static_layers(m, cfg)
hist = ObservationHistory(cfg, static, len(m.robots), seed=seed)
rew = TaskReward(m, cfg)
torch.manual_seed(seed)
zone = m.danger_zone

def pol(obs, det=False):
    with torch.no_grad():
        t = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
        if det:
            return agent.policy.deterministic_action(t["ego"], t["mid"], t["glob"], t["state"]).numpy()
        a, _ = agent.policy.sample_action(t["ego"], t["mid"], t["glob"], t["state"])
        return a.numpy()

def phis(model, n_ref, d_ref):
    inside = [a for a in model.crowds if a.type in (0, 1, 2) and not a.dead and zone.contains(float(a.xy[0]), float(a.xy[1]))]
    prog = -sum(min(1.0, float(model.escape_distance(a.xy)) / d_ref) for a in inside) / n_ref
    rb = model.robots[0]
    unreached = [a for a in inside if a.type != 0 and not getattr(a, "informed_by_robot", False)]
    if unreached:
        d = min(math.hypot(float(a.xy[0]) - rb.xy[0], float(a.xy[1]) - rb.xy[1]) for a in unreached)
    else:
        d = 0.0
    appr = -min(d, 50.0) / 50.0
    nfollow0 = sum(1 for a in model.crowds if not a.dead and a.type == 0 and getattr(a, "following_robot_id", None) == rb.unique_id)
    return prog, appr, nfollow0, len(inside)

records = []
done_dec = 0
for t0 in T0S:
    while done_dec < t0:
        hist.record(m)
        acts = pol(hist.team_observations())
        for i, rb in enumerate(m.robots): ra.apply_to(rb, acts[i])
        for k in range(K):
            m.step(); rew.step(m)
        done_dec += 1
    hist.record(m)
    obs = hist.team_observations()
    base = pol(obs); det0 = pol(obs, det=True)[0]
    rb0 = m.robots[0]; x, y = float(rb0.xy[0]), float(rb0.xy[1])
    lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    def toward(tx, ty, speed, mode):
        dx, dy = tx - x, ty - y
        s = max(1.0, abs(dx) / lim, abs(dy) / lim)
        return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)
    sp = ShuttlePolicy(m, len(m.robots), cfg=cfg)
    ex = sp._exit_mesh(m.find_mesh((x, y))); exc = _centre(ex) if ex is not None else (x, y)
    cands = {"policy_det": det0, "in_off": toward(zone.cx, zone.cy, 1.0, "off"),
             "in_guide": toward(zone.cx, zone.cy, 0.6, "guide"), "out_guide": toward(exc[0], exc[1], 0.6, "guide"),
             "out_off": toward(exc[0], exc[1], 1.0, "off"), "still_guide": ra.encode((0.0, 0.0), "guide", speed=0.0)}
    p0 = phis(m, rew.n_ref, rew.d_ref)
    r_state, n_state = random.getstate(), np.random.get_state()
    for rep in range(M):
        for nm, a0 in cands.items():
            mm = copy.deepcopy(m); rr = copy.copy(rew); rr._clear = set(rew._clear)
            random.seed(10_000 * rep + 7 + seed + t0); np.random.seed((10_000 * rep + 7 + seed + t0) % 2**32)
            acts = base.copy(); acts[0] = a0
            for i, rb in enumerate(mm.robots): ra.apply_to(rb, acts[i])
            R, disc = 0.0, 1.0
            for k in range(K):
                mm.step(); R += disc * sum(rr.step(mm).values()); disc *= g_step
            p1 = phis(mm, rew.n_ref, rew.d_ref)
            records.append({"scenario": name, "t0": t0, "rep": rep, "action": nm, "R8": R,
                            "F_prog": disc * p1[0] - p0[0], "F_appr": disc * p1[1] - p0[1],
                            "phi0_appr": p0[1], "follow0_before": p0[2], "follow0_after": p1[2],
                            "inside": p0[3], "robot0_inside": bool(zone.contains(x, y)), "n_ref": rew.n_ref})
    print(f"[{name}] t0={t0} done", flush=True)
json.dump(records, open(out, "w")); print("saved", out)
