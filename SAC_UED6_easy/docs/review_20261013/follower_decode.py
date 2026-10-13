"""Can the current actor/critic inputs tell how many people follow a robot?

Runs the learned stochastic policy on a validation level and records, at every
decision and for every robot: the true number of followers (and how many of
them are inside the hazard), simple observable proxies (people within 3.5 / 5 /
10 m, previous mode), the critic's action-independent features and the actor's
backbone features. A separate fit decodes followers from each feature set.
"""
import os, sys, copy, json
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"); os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(1)
from configs import resolve_config
from learn.zero_shot import validation_levels, build_eval_model, validation_seed
from learn.sac import SACAgent, to_torch, OBS_KEYS
from learn.replay import joint_observation
from sim import robot_action as ra
from sim.observation import ObservationHistory, build_static_layers
from sim.rewards import TaskReward

scen, ckpt, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
DEC = int(os.environ.get("DEC", "100"))
cfg = resolve_config(check_data=False)
K = int(cfg.decision_max_steps())
agent = SACAgent(cfg, device="cpu"); agent.load(ckpt); agent.policy.eval()
name, lv = validation_levels(cfg)[scen]
seed = validation_seed(name)
m = build_eval_model(lv, seed); m.should_finish = lambda: False
static = build_static_layers(m, cfg); hist = ObservationHistory(cfg, static, len(m.robots), seed=seed)
rew = TaskReward(m, cfg); torch.manual_seed(seed); zone = m.danger_zone
rows = {k: [] for k in ("n_follow", "n_follow_in", "n3", "n5", "n10", "mode_guide", "inside", "critic_feat", "actor_feat", "t")}
for d in range(DEC):
    hist.record(m)
    obs = hist.team_observations()
    joint = joint_observation(cfg, [static], [hist.window()])
    with torch.no_grad():
        t = to_torch({k: joint[k] for k in OBS_KEYS + ("priv", "priv_scalars")}, "cpu")
        vision, state, B, N = agent.q1.encode_observation(t)
        cf = torch.cat([vision, state], -1).numpy()           # (N, 1024+S)
        ta = to_torch({k: obs[k] for k in OBS_KEYS}, "cpu")
        af = agent.policy.features(ta["ego"], ta["mid"], ta["glob"], ta["state"]).numpy()
        acts, _ = agent.policy.sample_action(ta["ego"], ta["mid"], ta["glob"], ta["state"])
        acts = acts.numpy()
    crowd = [a for a in m.crowds if a.type in (0, 1, 2) and not a.dead]
    P = np.array([a.xy for a in crowd], dtype=float) if crowd else np.zeros((0, 2))
    for i, rb in enumerate(m.robots):
        dist = np.hypot(P[:, 0] - rb.xy[0], P[:, 1] - rb.xy[1]) if len(P) else np.zeros(0)
        fol = [a for a in crowd if a.type == 0 and getattr(a, "following_robot_id", None) == rb.unique_id]
        rows["n_follow"].append(len(fol))
        rows["n_follow_in"].append(sum(1 for a in fol if zone.contains(float(a.xy[0]), float(a.xy[1]))))
        rows["n3"].append(int((dist <= 3.5).sum())); rows["n5"].append(int((dist <= 5).sum())); rows["n10"].append(int((dist <= 10).sum()))
        rows["mode_guide"].append(int(getattr(rb, "mode", "off") == "guide"))
        rows["inside"].append(int(zone.contains(float(rb.xy[0]), float(rb.xy[1]))))
        rows["critic_feat"].append(cf[i].astype(np.float32)); rows["actor_feat"].append(af[i].astype(np.float32)); rows["t"].append(d)
    for i, rb in enumerate(m.robots): ra.apply_to(rb, acts[i])
    for k in range(K):
        m.step(); rew.step(m)
np.savez_compressed(out, scenario=name, **{k: np.array(v) for k, v in rows.items()})
print("saved", out, "samples", len(rows["n_follow"]), "mean followers", np.mean(rows["n_follow"]), "frac>0", np.mean(np.array(rows["n_follow"]) > 0))
