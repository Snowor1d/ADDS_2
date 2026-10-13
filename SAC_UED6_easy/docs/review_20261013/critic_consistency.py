"""Is the critic's action preference signal or noise?

Same replay states and the same K candidate actions for robot j, evaluated by
(a) q1 vs q2 of one checkpoint and (b) checkpoints ep2300 vs ep2400.
Correlates the action-centred values dQ_k = Q(s,a_k) - mean_k Q(s,a_k).
"""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"); os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
torch.set_num_threads(4)
from configs import resolve_config
from learn.replay import ReplayBuffer, StaticStore
from learn.sac import SACAgent, to_torch
from sim import robot_action as ra
LOG = os.path.expanduser("~/Log_SAC_UED6_easy_fullinfo")
cfg = resolve_config(check_data=False)
store = StaticStore(os.path.join(LOG, cfg.REPLAY_STATIC_DIR), 4096)
buf = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store); buf.load(os.path.join(LOG, "replay_buffer.npz"), cfg.schema_versions())
agents = {}
for ep in (2300, 2400):
    a = SACAgent(cfg, device="cpu"); a.load(os.path.join(LOG, f"sac_checkpoint_ep_{ep}.pth")); agents[ep] = a
rng = np.random.default_rng(7); prng = __import__("random").Random(7)
b = buf.sample(128, rng)
obs = to_torch(b["obs"], "cpu"); mask = torch.as_tensor(b["mask"]); act = torch.as_tensor(b["action"])
B, N = mask.shape; K = 24; j = 0
cands = torch.as_tensor(np.stack([np.stack([ra.random_action(prng) for _ in range(B)]) for _ in range(K)]))  # (K,B,A)
def dq(critic):
    with torch.no_grad():
        f = critic.encode_observation(obs)
        vals = []
        for k in range(K):
            a = act.clone(); a[:, j] = cands[k]
            vals.append(critic.values_from_features(f, a, mask)[:, j])
        v = torch.stack(vals, 1).numpy()     # (B,K)
    return v - v.mean(1, keepdims=True)
d = {}
for ep, ag in agents.items():
    d[(ep, 1)] = dq(ag.q1); d[(ep, 2)] = dq(ag.q2)
def corr(x, y):
    r = [np.corrcoef(x[i], y[i])[0, 1] for i in range(B)]
    return np.nanmean(r), np.nanmedian(r)
print("per-state correlation of action-centred Q across 24 random actions (robot 0)")
print("  q1 vs q2, ep2400:          mean %.2f median %.2f" % corr(d[(2400, 1)], d[(2400, 2)]))
print("  q1 vs q2, ep2300:          mean %.2f median %.2f" % corr(d[(2300, 1)], d[(2300, 2)]))
print("  q1 ep2300 vs q1 ep2400:    mean %.2f median %.2f" % corr(d[(2300, 1)], d[(2400, 1)]))
print("  q2 ep2300 vs q2 ep2400:    mean %.2f median %.2f" % corr(d[(2300, 2)], d[(2400, 2)]))
print("  action-centred sd: q1 %.3f q2 %.3f; diff(q1-q2) sd %.3f" % (d[(2400,1)].std(), d[(2400,2)].std(), (d[(2400,1)] - d[(2400,2)]).std()))
# which action features does the critic prefer? regress dQ on features
X = []
Y = []
for k in range(K):
    c = cands[k].numpy()
    mode_guide = c[:, ra.MODE][:, 1]
    speed = (c[:, 2] + 2) / 4
    mag = np.hypot(c[:, 0], c[:, 1]) / 2
    X.append(np.stack([np.ones(B), mode_guide, speed, mag], 1)); Y.append(0.5 * (d[(2400, 1)][:, k] + d[(2400, 2)][:, k]))
X = np.concatenate(X); Y = np.concatenate(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
pred = X @ coef
print("  linear fit of dQ on [1, guide, speed, |move|]: coef", np.round(coef, 3), " R2 %.2f" % (1 - ((Y - pred) ** 2).mean() / Y.var()))
