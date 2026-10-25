"""Offline critic testbed: policy evaluation of a fixed actor on a fixed replay.

Every variant trains twin critics and their targets on the same replay with
the actor and alpha frozen, so only the critic's architecture, loss and
target differ. The encoders start from the checkpoint critic; everything that
sees the action (agent MLP, FiLM, attention, head, and the dueling value
branch) starts fresh in every variant, the baseline included.

Variants (--variant):
  mse              the current critic and loss
  hlgauss          categorical head, HL-Gauss cross-entropy (Farebrother et
                   al. 2024), value = expectation over the bins
  dueling          Q = V(s) + A(s, a) - A(s, a_ref): a value branch without
                   the action, and an action branch measured against a fixed
                   reference action (all robots still, signal off)
  hlgauss_dueling  both, dueling on the logits
  --nstep N        any variant with N-decision returns along next_dec

Metrics (every --eval-every updates, JSONL):
  probe   counterfactual probe states (results/*_state.npz + .json): per
          state, sd over candidates of robot 0's min(q1, q2) against the
          noise-corrected sd of the Monte Carlo returns, and their Pearson
          correlation across candidates
  agree   replay states: correlation of q1 and q2 over random robot-0 actions
          after removing each state's mean (action signal shared by the two
          critics rather than set by their initialisation)
  td      mean squared error of Q against the (n-step) target, held out
"""
import argparse
import copy
import glob
import json
import os
import queue
import sys
import threading
import time

ROOT = "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import resolve_config
from learn.networks import CentralizedCritic
from learn.replay import ReplayBuffer, StaticStore
from learn.sac import OBS_KEYS, SACAgent, to_torch
from sim import robot_action as ra

KEYS = OBS_KEYS + ("priv", "priv_scalars")
ENCODER_PREFIXES = ("ego.", "mid.", "glob.", "priv.", "priv_scalars.",
                    "state.", "context.", "attend.")


# ------------------------------------------------------------------ critic

class VariantCritic(CentralizedCritic):
    def __init__(self, cfg, action_dim, kind, support=None, heads=4):
        super().__init__(cfg, action_dim, heads=heads)
        self.dist = "hlgauss" in kind
        self.dueling = "dueling" in kind
        embed = self.spec["embed"]
        h1, h2 = self.spec["head"]
        out = 1
        if self.dist:
            lo, hi, nb = support
            edges = torch.linspace(lo, hi, nb + 1)
            self.register_buffer("edges", edges)
            self.register_buffer("centers", 0.5 * (edges[1:] + edges[:-1]))
            self.sigma = 0.75 * float(edges[1] - edges[0])
            out = nb
            self.head[-1] = nn.Linear(h2, nb)
        if self.dueling:
            state_dim = self.agent[0].in_features - action_dim
            vision = self.film_out.in_features
            self.v_state = nn.Sequential(nn.Linear(state_dim, embed), nn.SiLU(),
                                         nn.Linear(embed, embed))
            self.v_out = nn.Linear(vision, embed)
            self.v_attn = nn.MultiheadAttention(embed, heads, batch_first=True)
            self.v_norm = nn.LayerNorm(embed)
            self.v_head = nn.Sequential(nn.Linear(embed, h1), nn.SiLU(),
                                        nn.Linear(h1, h2), nn.SiLU(),
                                        nn.Linear(h2, out))
            ref = ra.encode((0.0, 0.0), "off", speed=0.5)
            self.register_buffer("a_ref", torch.as_tensor(ref, dtype=torch.float32))

    def _a_raw(self, features, action, mask):
        vision, state, B, N = features
        agent = self.agent(torch.cat([state, action.reshape(B * N, -1)], -1))
        gamma, beta = self.film(agent).chunk(2, -1)
        tok = F.silu(self.film_out((1 + gamma) * vision + beta)) + agent
        tok = tok.reshape(B, N, -1) * mask.unsqueeze(-1)
        out, _ = self.attn(tok, tok, tok, key_padding_mask=mask <= 0)
        tok = self.norm(tok + out) * mask.unsqueeze(-1)
        return self.head(tok)                               # (B, N, out)

    def _v_raw(self, features, mask):
        vision, state, B, N = features
        tok = F.silu(self.v_out(vision)) + self.v_state(state)
        tok = tok.reshape(B, N, -1) * mask.unsqueeze(-1)
        out, _ = self.v_attn(tok, tok, tok, key_padding_mask=mask <= 0)
        tok = self.v_norm(tok + out) * mask.unsqueeze(-1)
        return self.v_head(tok)

    def raw(self, features, action, mask):
        a = self._a_raw(features, action, mask)
        if not self.dueling:
            return a
        ref = self.a_ref.expand_as(action)
        return self._v_raw(features, mask) + a - self._a_raw(features, ref, mask)

    def values_from_features(self, features, action, mask):
        r = self.raw(features, action, mask)
        if self.dist:
            v = (r.softmax(-1) * self.centers).sum(-1)
        else:
            v = r.squeeze(-1)
        return v * mask

    def loss(self, features, action, mask, y):
        r = self.raw(features, action, mask)
        m = mask.float()
        if self.dist:
            yc = y.clamp(float(self.edges[0]), float(self.edges[-1]))
            cdf = torch.special.ndtr((self.edges - yc.unsqueeze(-1)) / self.sigma)
            p = cdf[..., 1:] - cdf[..., :-1]
            p = p / p.sum(-1, keepdim=True).clamp(min=1e-8)
            per = -(p * r.log_softmax(-1)).sum(-1)
        else:
            per = (r.squeeze(-1) - y) ** 2
        return (per * m).sum() / m.sum().clamp(min=1.0)


def make_critic(cfg, kind, support, ref_state, device):
    q = VariantCritic(cfg, ra.ACTION_DIM, kind, support)
    enc = {k: v for k, v in ref_state.items() if k.startswith(ENCODER_PREFIXES)}
    missing = q.load_state_dict(enc, strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return q.to(device)


# -------------------------------------------------------------------- data

def nstep_batch(replay, B, n, g_step, rng):
    """B transitions of up to n decisions: (indices, last next index,
    team return, own returns (B, R), bootstrap factor)."""
    with replay.lock:
        dec = np.flatnonzero(replay.has_action[:replay.size])
        idx, nxt, rt, ro, boot = [], [], [], [], []
        R = replay.robot_penalties.shape[2]
        while len(idx) < B:
            i = int(dec[rng.integers(dec.size)])
            if not replay.valid_transition(i):
                continue
            cur, team, own, disc, term = i, 0.0, np.zeros(R), 1.0, 0.0
            for j in range(n):
                h = int(replay.hold[cur])
                pw = g_step ** np.arange(h)
                team += disc * float((replay.step_rewards[cur, :h] * pw).sum())
                own += disc * (replay.robot_penalties[cur, :h, :] * pw[:, None]).sum(0)
                disc *= g_step ** h
                nx = int(replay.next_dec[cur])
                if replay.terminal[cur]:
                    term = 1.0
                    break
                if (j < n - 1 and replay.has_action[nx]
                        and replay.valid_transition(nx)):
                    cur = nx
                else:
                    break
            idx.append(i)
            nxt.append(nx)
            rt.append(team)
            ro.append(own)
            boot.append(disc * (1.0 - term))
        s = replay._joint(idx)
        s2 = replay._joint(nxt)
        act = replay.action[idx].copy()
    return {"obs": {k: s[k] for k in KEYS}, "mask": s["mask"],
            "next_obs": {k: s2[k] for k in KEYS}, "next_mask": s2["mask"],
            "action": act, "r_team": np.asarray(rt, np.float32),
            "r_own": np.asarray(ro, np.float32), "boot": np.asarray(boot, np.float32),
            "indices": np.asarray(idx)}


def to_dev(b, dev):
    out = {k: to_torch(v, dev) if isinstance(v, dict) else
           torch.as_tensor(v, dtype=torch.float32, device=dev)
           for k, v in b.items() if k != "indices"}
    return out


def permute(b):
    mask = b["mask"]
    B, N = mask.shape
    perms = torch.argsort(torch.rand(B, N, device=mask.device), dim=1)

    def p(x):
        idx = perms
        while idx.ndim < x.ndim:
            idx = idx.unsqueeze(-1)
        return x.gather(1, idx.expand(*x.shape[:2], *x.shape[2:]))

    out = dict(b)
    for k in ("obs", "next_obs"):
        out[k] = {kk: (p(v) if kk in OBS_KEYS else v) for kk, v in b[k].items()}
    for k in ("mask", "next_mask", "action", "r_own"):
        out[k] = p(b[k])
    return out


class Prefetch:
    def __init__(self, fn, depth=2):
        self.q = queue.Queue(depth)
        self.fn = fn
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            self.q.put(self.fn())

    def get(self):
        return self.q.get()


# ----------------------------------------------------------------- metrics

def load_probes(rdir):
    probes = []
    for path in sorted(glob.glob(os.path.join(rdir, "s*_t*_state.npz"))):
        d = json.load(open(path.replace("_state.npz", ".json")))
        z = np.load(path)
        cands = [str(c) for c in z["cands"]]
        mc = np.asarray([d["mc"][c] for c in cands], np.float64)   # (C, M)
        C, M = mc.shape
        grand = mc.mean()
        cm, rm = mc.mean(1), mc.mean(0)
        ms_c = M * ((cm - grand) ** 2).sum() / (C - 1)
        resid = mc - cm[:, None] - rm[None, :] + grand
        ms_e = (resid ** 2).sum() / ((C - 1) * (M - 1)) if M > 1 else 0.0
        probes.append(dict(name=os.path.basename(path)[:-10], reps=M,
                           obs={k: z[k] for k in KEYS}, mask=z["mask"],
                           applied=z["applied"], mc_mean=cm,
                           true_var=max(0.0, (ms_c - ms_e) / M),
                           q1_ref=np.asarray([d["q1"][c] for c in cands])))
    return probes


@torch.no_grad()
def probe_metrics(q1, q2, probes, dev):
    if not probes:
        return {}
    cvar, tvar, corr, rows, hi = [], [], [], [], []
    for p in probes:
        obs = to_torch(p["obs"], dev)
        mask = torch.as_tensor(p["mask"], device=dev)
        f1, f2 = q1.encode_observation(obs), q2.encode_observation(obs)
        C = p["applied"].shape[0]
        fe = lambda f: (f[0].repeat(C, 1), f[1].repeat(C, 1), C, f[3])
        a = torch.as_tensor(p["applied"], device=dev)
        mk = mask.repeat(C, 1)
        q = torch.min(q1.values_from_features(fe(f1), a, mk)[:, 0],
                      q2.values_from_features(fe(f2), a, mk)[:, 0]).cpu().numpy()
        cvar.append(q.var(ddof=1))
        tvar.append(p["true_var"])
        corr.append(float(np.corrcoef(q, p["mc_mean"])[0, 1]))
        hi.append(p["reps"] >= 10)
        rows.append({"state": p["name"], "reps": p["reps"], "critic_sd": float(q.std(ddof=1)),
                     "true_sd": float(np.sqrt(p["true_var"])), "corr": corr[-1]})
    out = {"probe_ratio": float(np.sqrt(np.mean(cvar) / max(np.mean(tvar), 1e-9))),
           "probe_corr": float(np.mean(corr)), "probe_rows": rows}
    if any(hi):
        # States with enough repeats for their Monte Carlo means to be reliable.
        sel = [i for i, h in enumerate(hi) if h]
        out["probe_ratio_hi"] = float(np.sqrt(np.mean([cvar[i] for i in sel])
                                              / max(np.mean([tvar[i] for i in sel]), 1e-9)))
        out["probe_corr_hi"] = float(np.mean([corr[i] for i in sel]))
    return out


@torch.no_grad()
def agree_metrics(q1, q2, agree, dev):
    obs, mask, base, rand = agree
    B, A = rand.shape[0], rand.shape[1]
    f1, f2 = q1.encode_observation(obs), q2.encode_observation(obs)
    v1, v2 = [], []
    for k in range(A):
        act = base.clone()
        act[:, 0] = rand[:, k]
        v1.append(q1.values_from_features(f1, act, mask)[:, 0])
        v2.append(q2.values_from_features(f2, act, mask)[:, 0])
    v1, v2 = torch.stack(v1, 1), torch.stack(v2, 1)            # (B, A)
    c1, c2 = v1 - v1.mean(1, keepdim=True), v2 - v2.mean(1, keepdim=True)
    per = (c1 * c2).sum(1) / (c1.norm(dim=1) * c2.norm(dim=1)).clamp(min=1e-9)
    return {"agree_corr": float(per.mean()),
            "action_sd": float(torch.min(v1, v2).std(1).mean())}


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mse",
                    choices=["mse", "hlgauss", "dueling", "hlgauss_dueling"])
    ap.add_argument("--nstep", type=int, default=1)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--static-dir", required=True)
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=30000)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--bins", type=int, default=201)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fill-test", action="store_true",
                    help="tiny replay from short rollouts instead of --replay (smoke test)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    cfg = resolve_config(check_data=False)
    B = int(args.batch or cfg.BATCH_SIZE)
    agent = SACAgent(cfg, device=args.device)
    ck = agent.load(args.ckpt)
    agent.policy.eval()
    for p in agent.policy.parameters():
        p.requires_grad_(False)
    g_step = agent.gamma ** (1.0 / max(1, int(cfg.ACTION_SCALE)))

    store = StaticStore(args.static_dir)
    if args.fill_test:
        from tests.test_madrl import _fill
        replay = ReplayBuffer(cfg, 2000, store)
        _fill(cfg, replay, store, agent, episodes=((3, 200), (3, 200)))
    else:
        t0 = time.time()
        replay = ReplayBuffer(cfg, int(cfg.BUFFER_SIZE), store)
        replay.load(args.replay, cfg.schema_versions())
        print(f"replay {len(replay)} records in {time.time() - t0:.0f}s", flush=True)

    sample = lambda: nstep_batch(replay, B, args.nstep, g_step, rng)

    # Support of the categorical head: the reference critic's targets, with
    # room on both sides.
    ref_state = ck["q1"]
    support = None
    if "hlgauss" in args.variant:
        ys = []
        with torch.no_grad():
            for _ in range(8):
                b = to_dev(sample(), dev)
                v2 = agent._expected_next_value(b["next_obs"], b["next_mask"], per_robot=True)
                y = b["r_team"][:, None] + b["r_own"] + b["boot"][:, None] * v2
                ys.append(y[b["mask"] > 0])
        y = torch.cat(ys)
        lo, hi = float(y.min()), float(y.max())
        span = max(hi - lo, 1.0)
        support = (lo - 0.5 * span, hi + 0.5 * span, int(args.bins))
        print("support", support, flush=True)

    q1 = make_critic(cfg, args.variant, support, ck["q1"], dev)
    q2 = make_critic(cfg, args.variant, support, ck["q2"], dev)
    q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)
    agent.q1_target, agent.q2_target = q1t, q2t
    agent.q1 = agent.q2 = None          # the checkpoint critics are not used
    opt = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()),
                            lr=float(cfg.LR), weight_decay=float(cfg.WD_Q))
    tau = 0.995

    probes = load_probes(args.probes)
    held = [to_dev(nstep_batch(replay, B, args.nstep, g_step, np.random.default_rng(10_000 + k)), dev)
            for k in range(4)]
    # Agreement set: 128 replay states, 16 uniform random robot-0 actions.
    ab = to_dev(nstep_batch(replay, 128, 1, g_step, np.random.default_rng(20_000)), dev)
    import random as _random
    _r = _random.Random(30_000)
    rand = torch.as_tensor(np.stack([[ra.random_action(_r) for _ in range(16)] for _ in range(128)]),
                           dtype=torch.float32, device=dev)
    agree = (ab["obs"], ab["mask"], ab["action"], rand)

    def evaluate(u):
        q1.eval(), q2.eval()
        with torch.no_grad():
            td = []
            for b in held:
                v2 = agent._expected_next_value(b["next_obs"], b["next_mask"], per_robot=True)
                y = b["r_team"][:, None] + b["r_own"] + b["boot"][:, None] * v2
                q = torch.min(q1(b["obs"], b["action"], b["mask"]),
                              q2(b["obs"], b["action"], b["mask"]))
                m = b["mask"]
                td.append(float((((q - y) ** 2) * m).sum() / m.sum()))
        rec = {"update": u, "variant": args.variant, "nstep": args.nstep,
               "td_mse": float(np.mean(td)), **agree_metrics(q1, q2, agree, dev),
               **probe_metrics(q1, q2, probes, dev)}
        q1.train(), q2.train()
        return rec

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log = open(args.out, "a")
    # The checkpoint critic itself, for reference (only meaningful for mse).
    if args.variant == "mse" and args.nstep == 1:
        ref1 = make_critic(cfg, "mse", None, ck["q1"], dev)
        ref2 = make_critic(cfg, "mse", None, ck["q2"], dev)
        ref1.load_state_dict(ck["q1"]), ref2.load_state_dict(ck["q2"])
        r = {"update": -1, "variant": "checkpoint", "nstep": 1,
             **agree_metrics(ref1, ref2, agree, dev), **probe_metrics(ref1, ref2, probes, dev)}
        log.write(json.dumps(r) + "\n"); log.flush()
        print({k: v for k, v in r.items() if k != "probe_rows"}, flush=True)
        del ref1, ref2

    pf = Prefetch(sample)
    t_start = time.time()
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            r = evaluate(u)
            r["elapsed_s"] = time.time() - t_start
            log.write(json.dumps(r) + "\n"); log.flush()
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items() if k != "probe_rows"}, flush=True)
        if u == args.updates:
            break
        b = permute(to_dev(pf.get(), dev))
        with torch.no_grad():
            v2 = agent._expected_next_value(b["next_obs"], b["next_mask"], per_robot=True)
            y = b["r_team"][:, None] + b["r_own"] + b["boot"][:, None] * v2
        loss = (q1.loss(q1.encode_observation(b["obs"]), b["action"], b["mask"], y)
                + q2.loss(q2.encode_observation(b["obs"]), b["action"], b["mask"], y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for net, tgt in ((q1, q1t), (q2, q2t)):
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(tau).add_((1 - tau) * p)
    log.close()


if __name__ == "__main__":
    main()
