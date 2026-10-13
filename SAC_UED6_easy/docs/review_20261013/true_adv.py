"""Counterfactual single-decision advantage probe (read-only on project files).

From a state reached by the learned stochastic policy, robot 0 takes one of
several actions for ONE decision (4 s); then every robot follows the learned
stochastic policy for the rest of a fixed horizon. Common random numbers:
every branch starts from a deep copy of the same model and the same RNG
states. Reports MC discounted returns per action and the critic's Q for the
same actions at the branch state.
"""
import os, sys, time, copy, random, json, math
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(1)
from configs import resolve_config
from learn.zero_shot import validation_levels, build_eval_model, validation_seed
from learn.sac import SACAgent, to_torch, OBS_KEYS
from learn.replay import joint_observation
from sim import robot_action as ra
from sim.observation import ObservationHistory, build_static_layers
from sim.rewards import TaskReward

scen_idx, t0_dec, ckpt, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
H = int(os.environ.get("H", "400")); M = int(os.environ.get("M", "3"))
cfg = resolve_config(check_data=False)
A = int(cfg.ACTION_SCALE); K = int(cfg.decision_max_steps())
g_step = cfg.gamma() ** (1.0 / A)
agent = SACAgent(cfg, device="cpu"); agent.load(ckpt); agent.policy.eval()
name, lv = validation_levels(cfg)[scen_idx]
seed = validation_seed(name)
m = build_eval_model(lv, seed)
m.should_finish = lambda: False
static = build_static_layers(m, cfg)
hist = ObservationHistory(cfg, static, len(m.robots), seed=seed)
rew = TaskReward(m, cfg)
torch.manual_seed(seed)

def pol(obs, det=False):
    with torch.no_grad():
        t = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
        if det:
            return agent.policy.deterministic_action(t["ego"], t["mid"], t["glob"], t["state"]).numpy()
        a, _ = agent.policy.sample_action(t["ego"], t["mid"], t["glob"], t["state"])
        return a.numpy()

# prefix: t0_dec decisions under the stochastic policy
pre_R = 0.0
for d in range(t0_dec):
    hist.record(m)
    acts = pol(hist.team_observations())
    for i, rb in enumerate(m.robots): ra.apply_to(rb, acts[i])
    for k in range(K):
        m.step(); pre_R += sum(rew.step(m).values())
# branch state
rec = hist.record(m)
obs = hist.team_observations()
base = pol(obs)                       # teammates' actions, fixed across branches
det0 = pol(obs, det=True)[0]
rb0 = m.robots[0]
x, y = float(rb0.xy[0]), float(rb0.xy[1])
zone = m.danger_zone
def toward(tx, ty, speed, mode):
    dx, dy = tx - x, ty - y
    lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    s = max(1.0, abs(dx) / lim, abs(dy) / lim)
    return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)
# nearest outside ground (shuttle rule)
from validation.shuttle_baseline import ShuttlePolicy, _centre
sp = ShuttlePolicy(m, len(m.robots), cfg=cfg)
ex = sp._exit_mesh(m.find_mesh((x, y)))
exc = _centre(ex) if ex is not None else (x, y)
cands = {
    "policy_det": det0,
    "in_off": toward(zone.cx, zone.cy, 1.0, "off"),
    "in_guide": toward(zone.cx, zone.cy, 0.6, "guide"),
    "out_guide": toward(exc[0], exc[1], 0.6, "guide"),
    "out_off": toward(exc[0], exc[1], 1.0, "off"),
    "still_guide": ra.encode((0.0, 0.0), "guide", speed=0.0),
}
# critic Q at the branch state
joint = joint_observation(cfg, [static], [hist.window()])
tobs = to_torch({k: joint[k] for k in OBS_KEYS + ("priv", "priv_scalars")}, "cpu")
mask = torch.as_tensor(joint["mask"])
qpred = {}
with torch.no_grad():
    for nm, a0 in cands.items():
        act = np.zeros((1, int(cfg.MAX_ROBOTS), ra.ACTION_DIM), np.float32)
        act[0, :len(m.robots)] = base
        act[0, 0] = a0
        at = torch.as_tensor(act)
        q = torch.min(agent.q1(tobs, at, mask)[0, 0], agent.q2(tobs, at, mask)[0, 0])
        qpred[nm] = float(q)
inside0 = int(m.agents_in_danger())
r_state, n_state = random.getstate(), np.random.get_state()
results = {nm: [] for nm in cands}
t_start = time.time()
for rep in range(M):
    for nm, a0 in cands.items():
        mm = copy.deepcopy(m)
        hh = copy.copy(hist); hh.records = list(hist.records); hh.comm = copy.deepcopy(hist.comm)
        rr = copy.copy(rew); rr._clear = set(rew._clear)
        random.setstate(r_state); np.random.set_state(n_state)
        random.seed(10_000 * rep + 7 + seed)       # rep-specific crowd randomness, same across actions
        np.random.seed((10_000 * rep + 7 + seed) % 2**32)
        torch.manual_seed(1000 + rep)
        acts = base.copy(); acts[0] = a0
        for i, rb in enumerate(mm.robots): ra.apply_to(rb, acts[i])
        R, disc, t = 0.0, 1.0, 0
        for k in range(K):
            mm.step(); R += disc * sum(rr.step(mm).values()); disc *= g_step; t += 1
        while t < H:
            hh.record(mm)
            acts = pol(hh.team_observations())
            for i, rb in enumerate(mm.robots): ra.apply_to(rb, acts[i])
            for k in range(K):
                mm.step(); R += disc * sum(rr.step(mm).values()); disc *= g_step; t += 1
        results[nm].append(R)
    print(f"[{name} t0={t0_dec}] rep {rep} done {time.time()-t_start:.0f}s", flush=True)
json.dump({"scenario": name, "t0_dec": t0_dec, "inside_at_branch": inside0, "n_ref": rew.n_ref,
           "robot0_xy": [x, y], "robot0_inside": bool(zone.contains(x, y)),
           "qpred": qpred, "mc": results, "H": H, "M": M}, open(out, "w"))
print("saved", out)
