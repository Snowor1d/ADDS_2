"""Read-only probe: how much does the trained critic's Q depend on the action?

Loads a checkpoint and the replay buffer of the running run, samples batches,
and compares the action-dependence of Q with the TD noise and reward scale.
"""
import os, sys, time, json
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT", "4")))
from configs import resolve_config
from learn.replay import ReplayBuffer, StaticStore
from learn.sac import SACAgent, to_torch, discounted_return, OBS_KEYS
from learn.networks import team_value
from sim import robot_action as ra

ckpt = sys.argv[1]
LOG = os.path.expanduser("~/Log_SAC_UED6_easy_fullinfo")
cfg = resolve_config(check_data=False)
t0 = time.time()
store = StaticStore(os.path.join(LOG, cfg.REPLAY_STATIC_DIR), 4096)
buf = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store)
buf.load(os.path.join(LOG, "replay_buffer.npz"), cfg.schema_versions())
print("buffer", len(buf), "loaded in", round(time.time() - t0, 1), "s", flush=True)
agent = SACAgent(cfg, device="cpu")
agent.load(ckpt)
agent.policy.eval()
print("alpha", float(agent.alpha), "updates", agent.updates, flush=True)
rng = np.random.default_rng(0)
K = 32
res = {k: [] for k in ["q_sa", "y", "td", "std_rand", "rng_rand", "std_pi", "rng_pi",
                         "mode_gap", "mode_gap_abs", "grad_norm", "std_q_states", "r4s",
                         "cont_std_pi", "pi_entropy_cont", "p_guide", "mean_abs", "logstd"]}
g_step = agent.gamma ** (1.0 / cfg.ACTION_SCALE)
for it in range(int(os.environ.get("ITERS", "4"))):
    bnp = buf.sample(128, rng)
    batch = {
        "obs": to_torch(bnp["obs"], "cpu"), "next_obs": to_torch(bnp["next_obs"], "cpu"),
        "mask": torch.as_tensor(bnp["mask"]), "next_mask": torch.as_tensor(bnp["next_mask"]),
        "action": torch.as_tensor(bnp["action"]), "step_rewards": torch.as_tensor(bnp["step_rewards"]),
        "robot_penalties": torch.as_tensor(bnp["robot_penalties"], dtype=torch.float32),
        "hold": torch.as_tensor(bnp["hold"]), "terminal": torch.as_tensor(bnp["terminal"]),
        "agent_index": torch.as_tensor(bnp["agent_index"], dtype=torch.long)}
    obs, mask, act = batch["obs"], batch["mask"], batch["action"]
    B, N = mask.shape
    with torch.no_grad():
        y = agent.robot_targets(batch)                       # (B,N)
        f1 = agent.q1.encode_observation(obs); f2 = agent.q2.encode_observation(obs)
        q = torch.min(agent.q1.values_from_features(f1, act, mask), agent.q2.values_from_features(f2, act, mask))
        res["q_sa"].append(q[mask > 0].numpy()); res["y"].append(y[mask > 0].numpy())
        res["td"].append((q - y)[mask > 0].numpy())
        r4 = discounted_return(batch["step_rewards"], batch["hold"], g_step)
        res["r4s"].append(r4.numpy())
        # robot j = 0 (slots are in model order; every sample has 3 robots)
        flat = {k: obs[k].reshape(B * N, *obs[k].shape[2:]) for k in OBS_KEYS}
        mean, log_std, logits = agent.policy.forward(flat["ego"], flat["mid"], flat["glob"], flat["state"])
        res["p_guide"].append(torch.softmax(logits, -1)[:, 1].numpy())
        res["mean_abs"].append(mean.abs().numpy()); res["logstd"].append(log_std.numpy())
        for j in range(N):
            # random actions for robot j
            qs_r, qs_p = [], []
            for k in range(K):
                a = act.clone()
                rnd = np.stack([ra.random_action(None) for _ in range(B)])
                a[:, j] = torch.as_tensor(rnd)
                qs_r.append(torch.min(agent.q1.values_from_features(f1, a, mask)[:, j],
                                      agent.q2.values_from_features(f2, a, mask)[:, j]))
                pick = {kk: obs[kk][:, j] for kk in OBS_KEYS}
                ap, _ = agent.policy.sample_action(pick["ego"], pick["mid"], pick["glob"], pick["state"])
                a2 = act.clone(); a2[:, j] = ap
                qs_p.append(torch.min(agent.q1.values_from_features(f1, a2, mask)[:, j],
                                      agent.q2.values_from_features(f2, a2, mask)[:, j]))
            qs_r = torch.stack(qs_r, 1); qs_p = torch.stack(qs_p, 1)
            res["std_rand"].append(qs_r.std(1).numpy()); res["rng_rand"].append((qs_r.max(1).values - qs_r.min(1).values).numpy())
            res["std_pi"].append(qs_p.std(1).numpy()); res["rng_pi"].append((qs_p.max(1).values - qs_p.min(1).values).numpy())
            # mode gap at the stored continuous action
            ag = act.clone(); ag[:, j, ra.MODE] = torch.tensor([0., 1.])
            ao = act.clone(); ao[:, j, ra.MODE] = torch.tensor([1., 0.])
            gap = (torch.min(agent.q1.values_from_features(f1, ag, mask)[:, j], agent.q2.values_from_features(f2, ag, mask)[:, j])
                   - torch.min(agent.q1.values_from_features(f1, ao, mask)[:, j], agent.q2.values_from_features(f2, ao, mask)[:, j]))
            res["mode_gap"].append(gap.numpy()); res["mode_gap_abs"].append(gap.abs().numpy())
        res["std_q_states"].append(np.array([q[mask > 0].std().item()]))
    # gradient of Q wrt the continuous action (robot 0)
    a = act.clone().requires_grad_(True)
    qg = agent.q1.values_from_features(f1, a, mask)[:, 0].sum()
    (gr,) = torch.autograd.grad(qg, a)
    res["grad_norm"].append(gr[:, 0, :ra.CONT_DIM].norm(dim=-1).numpy())
    print("iter", it, "done", round(time.time() - t0, 1), "s", flush=True)

def s(k):
    v = np.concatenate([np.ravel(x) for x in res[k]])
    return f"mean={v.mean():9.4f} med={np.median(v):9.4f} p90={np.percentile(v,90):9.4f} std={v.std():9.4f}"
for k in res:
    if res[k]:
        print(f"{k:16s} {s(k)}")
