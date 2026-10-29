"""A map-aligned (spatial) critic against the current critic, offline.

Same protocol as testbed_multi.py: the actor, alpha and replay are fixed and
every critic is trained by policy evaluation on the same batches, so they
differ only in architecture. The replay is format 2 (collect_replay.py) and
the targets are learn/sac.SACAgent.robot_targets with each variant's own
target critics.

Variants:
  vector   the current critic (CentralizedCritic). Encoders from the
           checkpoint; everything that sees the action starts fresh.
  spatial  the same checkpoint encoders as a context vector, plus a U-Net on
           the robot-centred mid map (2 m cells) that outputs one value map
           per mode and speed level. A waypoint's value is read off the map
           where the waypoint is (bilinear in position, linear in speed), so
           "go there" is valued by what the map holds there. Teammates'
           positions and waypoints are drawn into the map, so the network
           does not depend on how many there are.

Every --eval-every updates (JSONL): held-out TD error, and on every probe
state the critics' values of robot 0's candidates (min(q1, q2), q1 and q2),
so any ranking statistic can be computed afterwards (analyze_spatial.py).

Usage: testbed_spatial.py --ckpt CK --replay-dir DIR --out OUT.jsonl \
           --overrides JSON [--variants vector,spatial] [--save-dir DIR]
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

ROOT = "/home/amrl_sunny/ADDS_2/SAC_UED6_easy"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import resolve_config
from learn.networks import CentralizedCritic, _gn
from learn.replay import BatchPrefetcher, ReplayBuffer, StaticStore
from learn.sac import OBS_KEYS, SACAgent, permute_robots, to_torch
from sim import observation as obsmod
from sim import robot_action as ra

KEYS = OBS_KEYS + ("priv", "priv_scalars")
ENCODER_PREFIXES = ("ego.", "mid.", "glob.", "priv.", "priv_scalars.",
                    "state.", "context.", "attend.")


# ------------------------------------------------------------------ critic

def _block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), _gn(cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU())


class FiLM(nn.Module):
    def __init__(self, ctx, ch):
        super().__init__()
        self.lin = nn.Linear(ctx, 2 * ch)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, h, ctx):
        g, b = self.lin(ctx).chunk(2, -1)
        return h * (1 + g[:, :, None, None]) + b[:, :, None, None]


class SpatialCritic(CentralizedCritic):
    """Q(s, a) for each robot, read off a value map at the robot's waypoint.

    The U-Net reads the 2 m mid map (plus each cell's offset from the robot)
    and outputs value maps on a grid of twice the cell size (32 x 32 cells of
    4 m), which is what fits a shared 8 GB GPU at batch 128 x 3 robots; the
    first stride-2 convolution keeps the 2 m detail in its channels. The
    CentralizedCritic parent supplies the checkpoint-initialised encoders
    (its action pathway is removed); being a subclass also lets
    SACAgent._expected_next_value reuse the action-independent features
    across the enumerated modes.
    """

    def __init__(self, cfg, action_dim, width=32, speeds=3):
        super().__init__(cfg, action_dim)
        for name in ("agent", "film", "film_out", "attn", "norm", "head"):
            delattr(self, name)
        s = obsmod.obs_shapes(cfg)
        embed = self.spec["embed"]
        self.M = int(cfg.OBS_MID_SIZE)
        self.res = float(cfg.OBS_MID_RES_M)
        self.Q = self.M // 2                       # value grid side, cells of 2 * res
        self.lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)
        self.ref = float(cfg.MAP_SIZE_REFERENCE)
        own = obsmod._index(obsmod.own_state_layout(cfg))
        self.ix = (own["x"], own["y"], own["log_w"], own["log_h"])
        modes = list(cfg.ROBOT_MODES)
        self.n_modes = len(modes)
        self.mode_ch = [modes.index("guide"), modes.index("off")]
        self.speeds = speeds
        w, cw = width, 256
        self.ctx_net = nn.Sequential(nn.Linear(4 * embed + 64, cw), nn.SiLU(),
                                     nn.Linear(cw, cw), nn.SiLU())
        self.e0 = nn.Sequential(nn.Conv2d(s["mid"][0] + 3, w, 3, stride=2, padding=1),
                                _gn(w), nn.SiLU())
        self.e1 = _block(w, w)                     # 32
        self.e2 = _block(w, 2 * w, 2)              # 16
        self.e3 = _block(2 * w, 3 * w, 2)          # 8
        self.d2 = _block(5 * w, 2 * w)             # 16
        self.d1 = _block(3 * w, w)                 # 32
        self.films = nn.ModuleList([FiLM(cw, c) for c in (3 * w, 2 * w, w)])
        self.render = nn.Conv2d(4, w, 3, padding=1)
        self.mix = nn.ModuleList([nn.Conv2d(w, w, 3, padding=d, dilation=d) for d in (1, 2, 4)])
        self.mix_gn = nn.ModuleList([_gn(w) for _ in self.mix])
        self.out = nn.Conv2d(w, self.n_modes * speeds, 1)

    # positions -------------------------------------------------------
    def _pose(self, state):
        """Absolute position (m) and its fraction of a mid cell, from the
        robot's own scalars (x / W, y / H, log(W / ref), log(H / ref))."""
        ix, iy, iw, ih = self.ix
        W = torch.exp(state[:, iw]) * self.ref
        H = torch.exp(state[:, ih]) * self.ref
        pos = torch.stack([state[:, ix] * W, state[:, iy] * H], -1)
        cell = pos / self.res
        return pos, cell - torch.floor(cell)

    def _coords(self, frac):
        """(BN, 3, M, M): each mid cell centre's offset from the robot over
        the waypoint range, x and y, and its distance."""
        M = self.M
        c = torch.arange(M, device=frac.device, dtype=frac.dtype) + 0.5 - M // 2
        rx = (c[None, None, :] - frac[:, 0, None, None]) * self.res / self.lim
        ry = (c[None, :, None] - frac[:, 1, None, None]) * self.res / self.lim
        rx, ry = rx.expand(-1, M, M), ry.expand(-1, M, M)
        return torch.stack([rx, ry, torch.sqrt(rx * rx + ry * ry)], 1)

    def _grid_uv(self, frac, rel):
        """Value-grid coordinates (column, row; a cell spans [i, i+1)) of a
        point `rel` metres from the robot. The robot sits at mid coordinate
        M // 2 + frac (mid cells), and a value cell is two mid cells."""
        return (self.M // 2 + frac + rel / self.res) / 2.0

    # features ----------------------------------------------------------
    def encode_observation(self, obs):
        vision, state, B, N = super().encode_observation(obs)
        ctx = self.ctx_net(torch.cat([vision, self.state(state)], -1))
        mid = obs["mid"].reshape(B * N, *obs["mid"].shape[2:])
        pos, frac = self._pose(state)
        h1 = self.e1(self.e0(torch.cat([mid, self._coords(frac)], 1)))
        h2 = self.e2(h1)
        h3 = self.films[0](self.e3(h2), ctx)
        up = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        g2 = self.films[1](self.d2(torch.cat([up(h3), h2], 1)), ctx)
        g1 = self.films[2](self.d1(torch.cat([up(g2), h1], 1)), ctx)
        return g1, pos.view(B, N, 2), frac, B, N

    def _render(self, pos, frac, off, speed, modes, mask):
        """(BN, 4, Q, Q): every teammate's position (channel 0) and its
        waypoint, weighted by guide, off and speed (1-3), on each robot's
        value grid, bilinearly splatted. Dense (no data-dependent indexing):
        pairs that are not teammates, and corners off the grid, carry weight 0."""
        B, N, _ = pos.shape
        G = self.Q
        dev = pos.device
        rel = pos[:, None, :, :] - pos[:, :, None, :]          # [b, j, n] = pos_n - pos_j
        tgt = rel + off[:, None, :, :]
        ok = ((mask[:, :, None] > 0) & (mask[:, None, :] > 0)
              & ~torch.eye(N, dtype=torch.bool, device=dev)[None]).to(pos.dtype)
        fr = frac.view(B, N, 1, 2)
        m = modes[:, None, :, :]
        sp = speed[:, None, :].expand(B, N, N)
        # (B, N, N, 4): the point each channel is drawn at and its value.
        pts = torch.stack([rel, tgt, tgt, tgt], 3)
        val = torch.stack([torch.ones_like(sp), m[..., self.mode_ch[0]].expand(B, N, N),
                           m[..., self.mode_ch[1]].expand(B, N, N), sp], 3) * ok[..., None]
        uv = self._grid_uv(fr[:, :, :, None, :], pts) - 0.5     # (B, N, N, 4, 2)
        x0, y0 = torch.floor(uv[..., 0]), torch.floor(uv[..., 1])
        tx, ty = uv[..., 0] - x0, uv[..., 1] - y0
        item = (torch.arange(B * N, device=dev).view(B, N, 1, 1) * 4
                + torch.arange(4, device=dev).view(1, 1, 1, 4))   # (B, N, 1, 4)
        canvas = torch.zeros(B * N * 4 * G * G, device=dev, dtype=pos.dtype)
        for dx, wx in ((0, 1 - tx), (1, tx)):
            for dy, wy in ((0, 1 - ty), (1, ty)):
                xi, yi = (x0 + dx).long(), (y0 + dy).long()
                inside = ((xi >= 0) & (xi < G) & (yi >= 0) & (yi < G)).to(pos.dtype)
                idx = (item * G + yi.clamp(0, G - 1)) * G + xi.clamp(0, G - 1)
                canvas.index_add_(0, idx.reshape(-1), (val * wx * wy * inside).reshape(-1))
        return canvas.view(B * N, 4, G, G)

    def value_maps(self, features, action, mask):
        """(BN, modes, speeds, Q, Q) value maps with teammates at `action`."""
        g1, pos, frac, B, N = features
        off = action[..., ra.MOVE].clamp(-2.0, 2.0) / 2.0 * self.lim
        speed = ((action[..., ra.SPEED] + 2.0) / 4.0).clamp(0.0, 1.0)
        h = g1 + self.render(self._render(pos, frac, off, speed, action[..., ra.MODE], mask))
        for conv, gn in zip(self.mix, self.mix_gn):
            h = h + F.silu(gn(conv(h)))
        return self.out(h).view(B * N, self.n_modes, self.speeds, self.Q, self.Q)

    def values_from_features(self, features, action, mask):
        g1, pos, frac, B, N = features
        K, S, G = self.n_modes, self.speeds, self.Q
        q = self.value_maps(features, action, mask)
        flat = action.reshape(B * N, -1)
        off = flat[:, ra.MOVE].clamp(-2.0, 2.0) / 2.0 * self.lim
        grid = (self._grid_uv(frac, off) / G * 2.0 - 1.0).view(B * N, 1, 1, 2)
        v = F.grid_sample(q.view(B * N, K * S, G, G), grid, mode="bilinear",
                          padding_mode="border", align_corners=False).view(B * N, K, S)
        sp = ((flat[:, ra.SPEED] + 2.0) / 4.0).clamp(0.0, 1.0) * (S - 1)
        lo = torch.floor(sp).clamp(max=S - 2)
        t = (sp - lo).unsqueeze(1)
        lo = lo.long().unsqueeze(1)
        ws = torch.zeros(B * N, S, device=v.device, dtype=v.dtype)
        ws.scatter_(1, lo, 1 - t)
        ws.scatter_add_(1, lo + 1, t)
        val = (v * flat[:, ra.MODE, None] * ws[:, None, :]).sum((1, 2))
        return val.view(B, N) * mask


    # one robot per sample ----------------------------------------------
    # Training and targets need only the sampled robot j's value, and
    # unlike the attention critic this one does not need the other robots'
    # maps for it: computing j's alone is three times less work.
    def robot_features(self, obs, robot):
        B, N = obs["state"].shape[:2]
        rows = torch.arange(B, device=robot.device)
        sel = {k: obs[k][rows, robot].unsqueeze(1) for k in OBS_KEYS}
        sel["priv"] = obs["priv"]
        if obs.get("priv_scalars") is not None:
            sel["priv_scalars"] = obs["priv_scalars"]
        g1, _, frac, _, _ = self.encode_observation(sel)
        pos, _ = self._pose(obs["state"].reshape(B * N, -1))
        return g1, pos.view(B, N, 2), frac, robot

    def robot_values(self, feats, action, mask):
        """(B,) robot j's value, teammates at their rows of `action`."""
        g1, pos, frac, robot = feats
        B, N = mask.shape
        G, K, S = self.Q, self.n_modes, self.speeds
        rows = torch.arange(B, device=mask.device)
        off = action[..., ra.MOVE].clamp(-2.0, 2.0) / 2.0 * self.lim       # (B, N, 2)
        speed = ((action[..., ra.SPEED] + 2.0) / 4.0).clamp(0.0, 1.0)
        modes = action[..., ra.MODE]
        # Teammates on robot j's grid: a one-receiver _render.
        own = F.one_hot(robot, N).to(pos.dtype)
        ok = ((mask > 0).to(pos.dtype) * (1 - own))                          # (B, N)
        rel = pos - pos[rows, robot][:, None, :]
        tgt = rel + off
        pts = torch.stack([rel, tgt, tgt, tgt], 2)                          # (B, N, 4, 2)
        val = torch.stack([torch.ones_like(speed), modes[..., self.mode_ch[0]],
                           modes[..., self.mode_ch[1]], speed], 2) * ok[..., None]
        uv = self._grid_uv(frac[:, None, None, :], pts) - 0.5
        x0, y0 = torch.floor(uv[..., 0]), torch.floor(uv[..., 1])
        tx, ty = uv[..., 0] - x0, uv[..., 1] - y0
        item = (rows.view(B, 1, 1) * 4 + torch.arange(4, device=mask.device).view(1, 1, 4))
        canvas = torch.zeros(B * 4 * G * G, device=pos.device, dtype=pos.dtype)
        for dx, wx in ((0, 1 - tx), (1, tx)):
            for dy, wy in ((0, 1 - ty), (1, ty)):
                xi, yi = (x0 + dx).long(), (y0 + dy).long()
                inside = ((xi >= 0) & (xi < G) & (yi >= 0) & (yi < G)).to(pos.dtype)
                idx = (item * G + yi.clamp(0, G - 1)) * G + xi.clamp(0, G - 1)
                canvas.index_add_(0, idx.reshape(-1), (val * wx * wy * inside).reshape(-1))
        h = g1 + self.render(canvas.view(B, 4, G, G))
        for conv, gn in zip(self.mix, self.mix_gn):
            h = h + F.silu(gn(conv(h)))
        q = self.out(h)                                                     # (B, K*S, G, G)
        a = action[rows, robot]
        grid = (self._grid_uv(frac, off[rows, robot]) / G * 2.0 - 1.0).view(B, 1, 1, 2)
        v = F.grid_sample(q, grid, mode="bilinear", padding_mode="border",
                          align_corners=False).view(B, K, S)
        sp = ((a[:, ra.SPEED] + 2.0) / 4.0).clamp(0.0, 1.0) * (S - 1)
        lo = torch.floor(sp).clamp(max=S - 2)
        t = (sp - lo).unsqueeze(1)
        lo = lo.long().unsqueeze(1)
        ws = torch.zeros(B, S, device=v.device, dtype=v.dtype)
        ws.scatter_(1, lo, 1 - t)
        ws.scatter_add_(1, lo + 1, t)
        return (v * a[:, ra.MODE, None] * ws[:, None, :]).sum((1, 2))


def make_critic(cfg, kind, ref_state, y_mean, dev):
    q = SpatialCritic(cfg, ra.ACTION_DIM) if kind == "spatial" else CentralizedCritic(cfg, ra.ACTION_DIM)
    enc = {k: v for k, v in ref_state.items() if k.startswith(ENCODER_PREFIXES)}
    res = q.load_state_dict(enc, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    with torch.no_grad():
        (q.out.bias if kind == "spatial" else q.head[-1].bias).fill_(y_mean)
    return q.to(dev)


def robot_feats(q, obs, robot):
    return q.robot_features(obs, robot) if isinstance(q, SpatialCritic) else q.encode_observation(obs)


def robot_value(q, f, action, mask, robot):
    """(B,) the critic's value for robot `robot` (per sample)."""
    if isinstance(q, SpatialCritic):
        return q.robot_values(f, action, mask)
    rows = torch.arange(action.shape[0], device=action.device)
    return q.values_from_features(f, action, mask)[rows, robot]


def expected_next_value(agent, critics, obs, mask, held, deciding, robot):
    """learn/sac.SACAgent._expected_next_value with per-robot critic calls
    (identical math; checked against it in --check)."""
    from itertools import product
    cont, logpc, probs, logpm = agent._per_robot_hybrid(obs, mask)
    B, N, K = probs.shape
    C = cont.shape[-1]
    dec = deciding.bool() & mask.bool()
    rows = torch.arange(B, device=mask.device)
    active = torch.nonzero(dec.any(dim=0), as_tuple=False).flatten().tolist()
    expected_q = cont.new_zeros(B)
    feats = [robot_feats(q, obs, robot) for q in critics]
    for modes in product(range(K), repeat=len(active)):
        indices = torch.zeros(B, N, dtype=torch.long, device=mask.device)
        for j, mode in zip(active, modes):
            indices[:, j] = mode
        one_hot = F.one_hot(indices, K).to(cont.dtype)
        move = cont.gather(2, indices[:, :, None, None].expand(B, N, 1, C)).squeeze(2)
        action = torch.where(dec.unsqueeze(-1), torch.cat((move, one_hot), dim=-1), held)
        selected = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        factors = torch.ones_like(selected)
        for j in active:
            factors[:, j] = torch.where(dec[:, j], selected[:, j],
                                        torch.full_like(selected[:, j], 1.0 / K))
        weight = factors.prod(-1)
        v1, v2 = [robot_value(q, f, action, mask, robot) for q, f in zip(critics, feats)]
        expected_q += weight * torch.min(v1, v2)
    expected_logp = (probs * (logpc + logpm)).sum(-1)[rows, robot]
    return expected_q - agent.alpha * expected_logp


def robot_targets(agent, critics, b):
    with torch.no_grad():
        v2 = expected_next_value(agent, critics, b["next_obs"], b["next_mask"], b["next_action"],
                                 b["next_decided"], b["agent_index"])
        g_step = agent.gamma ** (1.0 / max(1, int(agent.cfg.ACTION_SCALE)))
        boot = torch.pow(torch.full_like(b["hold"], g_step), b["hold"]) * (1.0 - b["terminal"])
        return b["ret"] + boot * v2


def robot_q(q, b, action, robot):
    return robot_value(q, robot_feats(q, b["obs"], robot), action, b["mask"], robot)


# -------------------------------------------------------------------- data

def to_batch(b, dev):
    f32 = lambda k: torch.as_tensor(b[k], device=dev, dtype=torch.float32)
    return {"obs": to_torch(b["obs"], dev), "next_obs": to_torch(b["next_obs"], dev),
            "mask": f32("mask"), "next_mask": f32("next_mask"),
            "action": f32("action"), "next_action": f32("next_action"),
            "decided": torch.as_tensor(b["decided"], device=dev, dtype=torch.bool),
            "next_decided": torch.as_tensor(b["next_decided"], device=dev, dtype=torch.bool),
            "ret": f32("ret"), "hold": f32("hold"), "terminal": f32("terminal"),
            "agent_index": torch.as_tensor(b["agent_index"], device=dev, dtype=torch.long)}


def load_probes(dirs):
    """Every probe state with a dumped critic input: its group (where the
    level comes from), candidates, Monte Carlo returns per repeat and the
    joint observation."""
    probes = []
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "s*_t*_state.npz"))):
            base = path[:-len("_state.npz")]
            if base.endswith(("_r3", "_more")):
                continue
            js = json.load(open(base + ".json"))
            z = np.load(path)
            cands = [str(c) for c in z["cands"]]
            size = int(js.get("map_size", 200))
            if "level_spec" in js and ":" in str(js["level_spec"]):
                site = str(js["level_spec"]).split(":")[0]
                group = f"{'train' if site in ('dotonbori', 'covent_garden', 'hongdae') else 'unseen'}_{size}"
            else:
                group = "train_200"
            probes.append(dict(name=os.path.basename(base), group=group,
                               cands=cands, mc=np.asarray([js["mc"][c] for c in cands]),
                               obs={k: z[k] for k in KEYS}, mask=z["mask"],
                               applied=z["applied"]))
    return probes


@torch.no_grad()
def probe_values(q1, q2, probes, dev):
    rows = []
    for p in probes:
        C = p["applied"].shape[0]
        obs = {k: torch.as_tensor(v, device=dev).expand(C, *v.shape[1:]).contiguous()
               for k, v in p["obs"].items()}
        mask = torch.as_tensor(p["mask"], device=dev, dtype=torch.float32).expand(C, -1)
        a = torch.as_tensor(p["applied"], device=dev, dtype=torch.float32)
        r0 = torch.zeros(C, dtype=torch.long, device=dev)
        v1, v2 = [robot_value(q, robot_feats(q, obs, r0), a, mask, r0) for q in (q1, q2)]
        rows.append({"state": p["name"], "group": p["group"],
                     "q": torch.min(v1, v2).tolist(), "q1": v1.tolist(), "q2": v2.tolist()})
    return rows


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="vector:0,spatial:0",
                    help="KIND:INIT_SEED,...; seed 0 is named KIND, others KIND_sSEED")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--replay-dir", default=None)
    ap.add_argument("--probe-dirs", default=",".join([os.path.join(HERE, "results"),
                                                      os.path.join(HERE, "results", "gen")]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--overrides", default="")
    ap.add_argument("--updates", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--gpu-frac", type=float, default=0.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fill-test", action="store_true", help="tiny replay from short rollouts")
    ap.add_argument("--check", action="store_true", help="numerical checks of the target and robot paths")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda" and args.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_frac))
    over = json.loads(args.overrides) if args.overrides else {}
    cfg = resolve_config(over or None, check_data=False)
    B = int(args.batch or cfg.BATCH_SIZE)
    agent = SACAgent(cfg, device=args.device)
    ck = agent.load(args.ckpt)
    agent.policy.eval()
    for p in agent.policy.parameters():
        p.requires_grad_(False)
    print(f"gamma {cfg.gamma()} alpha {float(agent.alpha):.4f} scope {cfg.ROBOT_DECISION_SCOPE}",
          flush=True)

    t0 = time.time()
    if args.fill_test:
        import tempfile
        from tests.test_madrl import _fill
        store = StaticStore(tempfile.mkdtemp())
        replay = ReplayBuffer(cfg, 3000, store)
        _fill(cfg, replay, store, agent, episodes=((3, 200), (3, 200)))
    else:
        store = StaticStore(os.path.join(args.replay_dir, "replay_static"))
        replay = ReplayBuffer(cfg, 400_000, store)
        replay.load(os.path.join(args.replay_dir, "replay_buffer.npz"), cfg.schema_versions())
    print(f"replay {len(replay)} frames, {replay.n_valid()} robot decisions, "
          f"{time.time() - t0:.0f}s", flush=True)

    held = [to_batch(replay.sample(B, np.random.default_rng(10_000 + k)), dev) for k in range(4)]
    # The checkpoint's own target critics set the initial output level.
    with torch.no_grad():
        y_mean = float(torch.cat([agent.robot_targets(b) for b in held]).mean())
    if args.check:
        # The per-robot target path against learn/sac on the checkpoint critics,
        # and a spatial critic's robot path against its all-robot path.
        b = held[0]
        torch.manual_seed(5)
        ref = agent.robot_targets(b)
        torch.manual_seed(5)
        mine = robot_targets(agent, (agent.q1_target, agent.q2_target), b)
        print(f"check target vs learn/sac: max |diff| {float((ref - mine).abs().max()):.2e}", flush=True)
        sc = SpatialCritic(cfg, ra.ACTION_DIM).to(dev).eval()
        with torch.no_grad():
            rows = torch.arange(b["mask"].shape[0], device=dev)
            full = sc(b["obs"], b["action"], b["mask"])[rows, b["agent_index"]]
            one = robot_q(sc, b, b["action"], b["agent_index"])
        print(f"check spatial robot path vs all robots: max |diff| {float((full - one).abs().max()):.2e}",
              flush=True)
    print(f"target mean {y_mean:.1f}", flush=True)

    V = []
    for spec in args.variants.split(","):
        kind, init = (spec.split(":") + ["0"])[:2]
        name = kind if int(init) == 0 else f"{kind}_s{init}"
        torch.manual_seed(1000 + int(init))
        q1 = make_critic(cfg, kind, ck["q1"], y_mean, dev)
        q2 = make_critic(cfg, kind, ck["q2"], y_mean, dev)
        q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
        for p in list(q1t.parameters()) + list(q2t.parameters()):
            p.requires_grad_(False)
        opt = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()),
                                lr=float(cfg.LR), weight_decay=float(cfg.WD_Q))
        n = sum(p.numel() for p in q1.parameters())
        print(f"{name}: {n / 1e6:.2f} M parameters per critic", flush=True)
        V.append(dict(name=name, kind=kind, q1=q1, q2=q2, q1t=q1t, q2t=q2t, opt=opt))

    probes = load_probes(args.probe_dirs.split(","))
    print(f"{len(probes)} probe states: "
          + ", ".join(f"{g} {sum(p['group'] == g for p in probes)}"
                      for g in sorted({p['group'] for p in probes})), flush=True)
    log = open(args.out, "a")

    def write(r):
        log.write(json.dumps(r) + "\n")
        log.flush()

    ref1 = CentralizedCritic(cfg, ra.ACTION_DIM).to(dev)
    ref2 = CentralizedCritic(cfg, ra.ACTION_DIM).to(dev)
    ref1.load_state_dict(ck["q1"]), ref2.load_state_dict(ck["q2"])
    ref1.eval(), ref2.eval()
    write({"update": -1, "variant": "checkpoint", "probe": probe_values(ref1, ref2, probes, dev)})
    del ref1, ref2

    def evaluate(u, elapsed):
        for v in V:
            v["q1"].eval(), v["q2"].eval()
            td = []
            with torch.no_grad():
                for b in held:
                    y = robot_targets(agent, (v["q1t"], v["q2t"]), b)
                    j = b["agent_index"]
                    q = torch.min(robot_q(v["q1"], b, b["action"], j),
                                  robot_q(v["q2"], b, b["action"], j))
                    td.append(float(((q - y) ** 2).mean()))
            r = {"update": u, "variant": v["name"], "td_mse": float(np.mean(td)),
                 "elapsed_s": elapsed, "probe": probe_values(v["q1"], v["q2"], probes, dev)}
            write(r)
            print(f"u {u} {v['name']}: td {r['td_mse']:.1f} ({elapsed:.0f}s)", flush=True)
            v["q1"].train(), v["q2"].train()

    def save(u):
        os.makedirs(args.save_dir, exist_ok=True)
        for v in V:
            torch.save({"kind": v["kind"], "name": v["name"], "update": u,
                        "q1": v["q1"].state_dict(), "q2": v["q2"].state_dict()},
                       os.path.join(args.save_dir, f"{v['name']}_u{u}.pt"))

    pf = BatchPrefetcher(replay, B, seed=args.seed)
    t_start = time.time()
    tau = 0.995
    for u in range(args.updates + 1):
        if u % args.eval_every == 0:
            evaluate(u, time.time() - t_start)
        if args.save_dir and u > 0 and (u % args.save_every == 0 or u == args.updates):
            save(u)
        if u == args.updates:
            break
        b = permute_robots(to_batch(pf.get(), dev))
        j = b["agent_index"]
        for v in V:
            y = robot_targets(agent, (v["q1t"], v["q2t"]), b)
            v["opt"].zero_grad()
            # One critic's graph at a time: same gradients, half the peak memory.
            for q in (v["q1"], v["q2"]):
                F.mse_loss(robot_q(q, b, b["action"], j), y).backward()
            v["opt"].step()
            with torch.no_grad():
                for net, tgt in ((v["q1"], v["q1t"]), (v["q2"], v["q2t"])):
                    for p, tp in zip(net.parameters(), tgt.parameters()):
                        tp.mul_(tau).add_((1 - tau) * p)
    pf.stop()
    log.close()


if __name__ == "__main__":
    main()
