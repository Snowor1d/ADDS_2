import os, sys, time
sys.path.insert(0, "/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
os.chdir("/home/amrl_sunny/ADDS_2/SAC_UED6_easy")
import numpy as np, torch
torch.set_num_threads(4)
from configs import resolve_config
from learn.replay import ReplayBuffer, StaticStore
from learn.sac import SACAgent, to_torch, OBS_KEYS
from sim import robot_action as ra
from sim.observation import MID_STATIC, GLOBAL_STATIC, EGO_STATIC, own_state_layout, teammate_state_layout
LOG = os.path.expanduser("~/Log_SAC_UED6_easy_fullinfo")
cfg = resolve_config(check_data=False)
store = StaticStore(os.path.join(LOG, cfg.REPLAY_STATIC_DIR), 4096)
buf = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store)
buf.load(os.path.join(LOG, "replay_buffer.npz"), cfg.schema_versions())
agent = SACAgent(cfg, device="cpu"); agent.load(sys.argv[1]); agent.policy.eval()
rng = np.random.default_rng(1)
b = buf.sample(256, rng)
o = b["obs"]
def st(name, x):
    x = np.asarray(x, dtype=np.float64).ravel()
    nz = x[x > 0]
    print(f"{name:34s} mean={x.mean():.4f} p50={np.median(x):.4f} p99={np.percentile(x,99):.4f} max={x.max():.4f} | nonzero frac={len(nz)/len(x):.3f} nonzero mean={nz.mean() if len(nz) else 0:.4f} p50={np.median(nz) if len(nz) else 0:.4f}")
m = o["mask"] if "mask" in o else b["mask"]
real = b["mask"] > 0
ego, mid, glob = o["ego"][real], o["mid"][real], o["glob"][real]
print("shapes ego", ego.shape, "mid", mid.shape, "glob", glob.shape, "state", o["state"][real].shape)
for i, n in enumerate(EGO_STATIC): st(f"ego/{n}", ego[:, i])
st("ego/own_crowd (newest)", ego[:, len(EGO_STATIC)])
st("ego/own_observed (newest)", ego[:, len(EGO_STATIC) + 1])
for i, n in enumerate(MID_STATIC): st(f"mid/{n}", mid[:, i])
for i, n in enumerate(("team_crowd", "team_observed", "team_age")): st(f"mid/{n}", mid[:, len(MID_STATIC) + i])
for i, n in enumerate(GLOBAL_STATIC): st(f"glob/{n}", glob[:, i])
for i, n in enumerate(("team_crowd", "team_observed")): st(f"glob/{n}", glob[:, len(GLOBAL_STATIC) + i])
st("priv", o["priv"])
print("own state layout", own_state_layout(cfg)); print("mate layout", teammate_state_layout(cfg))
print("state mean", np.round(o["state"][real].mean(0), 3))
print("priv_scalars mean", np.round(o["priv_scalars"].mean(0), 3), "std", np.round(o["priv_scalars"].std(0), 3))
# deterministic policy outputs
t = to_torch({k: o[k][real] for k in OBS_KEYS}, "cpu")
with torch.no_grad():
    mean, log_std, logits = agent.policy.forward(t["ego"], t["mid"], t["glob"], t["state"])
    det = agent.policy.deterministic_action(t["ego"], t["mid"], t["glob"], t["state"]).numpy()
mv = det[:, :2]; sp = (det[:, 2] + 2) / 4
print("det waypoint offset (m) |.|: ", np.percentile(np.hypot(mv[:,0], mv[:,1]) / 2 * cfg.ROBOT_WAYPOINT_RANGE_M, [10, 50, 90]))
print("det |ax|,|ay| saturation frac (>1.9):", np.mean(np.abs(mv) > 1.9))
print("det speed fraction pct:", np.percentile(sp, [10, 25, 50, 75, 90]))
print("det guide share:", det[:, ra.MODE][:, 1].mean())
print("mean (pre-squash) per dim abs mean:", np.abs(mean.numpy()).mean(0), " log_std mean per dim:", log_std.numpy().mean(0))
# stored actions in buffer (behaviour)
A = b["action"][real]
print("stored speed fraction pct:", np.percentile((A[:, 2] + 2) / 4, [10, 50, 90]), "stored guide share", A[:, ra.MODE][:, 1].mean())
print("hold distribution", np.bincount(b["hold"].astype(int)))
