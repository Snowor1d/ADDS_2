"""Actor and critics.

Inputs per sample: ego (C, 25, 25), glob (C, 50, 50), the robot state slice
(3 or 5 numbers, learn/obs.py) and, for the critic, the action (2).

Encoder families (NET_ENCODER) and sizes (NET_SIZE):

  legacy_rv3  SAC_FE_RV3's networks, layer for layer, so its checkpoints
              load and its results can be reproduced. BatchNorm, flatten,
              no embedding layer.
  cnn         three stride-2 convolutions (5x5, 3x3, 3x3) with a choice of
              normalisation, then flatten (position-specific) or a 4x4
              average pool, then a linear embedding.
  impala      coordinate channels, three IMPALA stacks (conv, max-pool /2,
              two residual blocks), a 4x4 average pool and a linear
              embedding (Espeholt et al. 2018; Cobbe et al. 2020).

ENCODER_SHARING = "critic" gives every critic one shared encoder trained by
the critic loss only, and the actor reads its features detached (SAC-AE,
Yarats et al. 2019; DrQ, Kostrikov et al. 2021).

The critic conditions the image features on (action, robot embedding) by
FiLM, as RV3 did, and can add LayerNorm and dropout to its MLP (RLPD, Ball et
al. 2023; DroQ, Hiraoka et al. 2022). CRITIC_ENSEMBLE critics are kept
(REDQ, Chen et al. 2021).
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# family -> size -> widths. Shared encoders mean the "critic" sharing mode
# has one encoder for the whole ensemble.
NETWORK_SIZES = {
    "cnn": {
        "s":  dict(channels=(16, 32, 64),   embed=128, mlp=(256, 128, 64),   head=(256, 128), robot=32),
        "m":  dict(channels=(32, 64, 128),  embed=256, mlp=(512, 256, 64),   head=(512, 256), robot=32),
        "l":  dict(channels=(48, 96, 192),  embed=384, mlp=(768, 384, 128),  head=(768, 384), robot=64),
        "xl": dict(channels=(64, 128, 256), embed=512, mlp=(1024, 512, 128), head=(1024, 512), robot=64),
    },
    # IMPALA widths x1, x2, x3, x4 of the Procgen network (16, 32, 32).
    "impala": {
        "s":  dict(channels=(16, 32, 32),   embed=256, mlp=(256, 128, 64),   head=(256, 128), robot=32),
        "m":  dict(channels=(32, 64, 64),   embed=256, mlp=(512, 256, 64),   head=(512, 256), robot=32),
        "l":  dict(channels=(48, 96, 96),   embed=384, mlp=(768, 384, 128),  head=(768, 384), robot=64),
        "xl": dict(channels=(64, 128, 128), embed=512, mlp=(1024, 512, 128), head=(1024, 512), robot=64),
    },
}
POOL = 4


def network_spec(cfg) -> dict:
    fam = cfg.NET_ENCODER
    if fam == "legacy_rv3":
        return dict(family=fam, size="rv3")
    return dict(NETWORK_SIZES[fam][cfg.NET_SIZE], family=fam, size=cfg.NET_SIZE)


# =============================================================================
# SAC_FE_RV3 networks, verbatim apart from the flags being arguments
# =============================================================================

class LegacyCNNEncoder(nn.Module):
    def __init__(self, input_shape=(50, 50), in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.out_dim = self._get_conv_out(input_shape, in_channels)

    def _get_conv_out(self, shape, in_channels):
        dummy = torch.zeros(1, in_channels, *shape)
        o = F.silu(self.bn1(self.conv1(dummy)))
        o = F.silu(self.bn2(self.conv2(o)))
        o = F.silu(self.bn3(self.conv3(o)))
        return int(np.prod(o.size()[1:]))

    def forward(self, x):
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = F.silu(self.bn3(self.conv3(x)))
        return x.view(x.size(0), -1)


class LegacyQNetwork(nn.Module):
    def __init__(self, ego_shape=(25, 25), global_shape=(50, 50), action_dim=2,
                 robot_dim=3, use_robot=True, ego_use=True, film_use=True):
        super().__init__()
        self.use_robot_state = use_robot
        self.ego_use = ego_use
        self.film_use = film_use
        self.ego_enc = LegacyCNNEncoder(input_shape=ego_shape, in_channels=4)
        self.glob_enc = LegacyCNNEncoder(input_shape=global_shape, in_channels=4)
        robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(nn.Linear(3, 32), nn.SiLU())
            robot_feat_dim = 32
        else:
            self.robot_fc = None
        self.img_dim = self.glob_enc.out_dim
        if ego_use:
            self.img_dim += self.ego_enc.out_dim
        cond_dim = action_dim + (robot_feat_dim if self.use_robot_state else 0)
        if film_use:
            self.film = nn.Sequential(nn.Linear(cond_dim, 256), nn.SiLU(),
                                      nn.Linear(256, 2 * self.img_dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        else:
            self.film = None
        fusion_dim = self.img_dim + robot_feat_dim + action_dim
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, ego_state, global_state, action, robot_state=None):
        g = self.glob_enc(global_state)
        if self.ego_use:
            e = self.ego_enc(ego_state)
            img = torch.cat([e, g], dim=1)
        else:
            img = g
        r = self.robot_fc(robot_state) if self.use_robot_state else None
        cond = torch.cat([action, r], dim=1) if self.use_robot_state else action
        if self.film_use:
            gamma, beta = self.film(cond).chunk(2, dim=1)
            img = (1.0 + gamma) * img + beta
        feats = [img] + ([r] if self.use_robot_state else []) + [action]
        out = F.silu(self.fc1(torch.cat(feats, dim=1)))
        out = F.silu(self.fc2(out))
        return self.q_out(out)


class LegacyPolicyNetwork(nn.Module):
    def __init__(self, ego_shape=(25, 25), global_shape=(50, 50), robot_dim=3,
                 use_robot=True, ego_use=True, log_std_min=-20, log_std_max=0.5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.use_robot_state = use_robot
        self.ego_use = ego_use
        self.ego_enc = LegacyCNNEncoder(input_shape=ego_shape, in_channels=4) if ego_use else None
        self.glob_enc = LegacyCNNEncoder(input_shape=global_shape, in_channels=4)
        self.robot_feat_dim = 0
        if self.use_robot_state:
            self.robot_fc = nn.Sequential(nn.Linear(3, 32), nn.SiLU())
            self.robot_feat_dim = 32
        fusion_dim = self.glob_enc.out_dim + self.robot_feat_dim
        if ego_use:
            fusion_dim += self.ego_enc.out_dim
        self.fc_backbone = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU())
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def forward(self, ego_state, global_state, robot_state=None):
        g = self.glob_enc(global_state)
        feats = []
        if self.ego_use:
            feats.append(self.ego_enc(ego_state))
        feats.append(g)
        if self.use_robot_state:
            feats.append(self.robot_fc(robot_state))
        feat = self.fc_backbone(torch.cat(feats, dim=1))
        mean = self.mean_head(feat)
        log_std = torch.clamp(self.log_std_head(feat), self.log_std_min, self.log_std_max)
        return mean, log_std


# =============================================================================
# RV4 encoders
# =============================================================================

def _norm(kind: str, ch: int, hw: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    if kind == "group":
        return nn.GroupNorm(min(8, ch), ch)
    if kind == "layer":
        return nn.GroupNorm(1, ch)  # LayerNorm over (C, H, W)
    return nn.Identity()


def _coords(size: int) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    return torch.stack([xx, yy])[None]


class ConvEncoder(nn.Module):
    """Family "cnn": RV3's three convolutions with a selectable norm, then
    flatten or 4x4 pool, then a linear embedding."""

    def __init__(self, in_ch: int, size: int, channels: Sequence[int], embed: int,
                 norm: str = "group", spatial: str = "flatten", coord: bool = False):
        super().__init__()
        self.coord = bool(coord)
        if self.coord:
            self.register_buffer("coords", _coords(size), persistent=False)
        c1, c2, c3 = channels
        cin = in_ch + (2 if self.coord else 0)
        s1 = (size + 2 * 2 - 5) // 2 + 1
        s2 = (s1 + 2 - 3) // 2 + 1
        s3 = (s2 + 2 - 3) // 2 + 1
        self.conv = nn.Sequential(
            nn.Conv2d(cin, c1, 5, stride=2, padding=2), _norm(norm, c1, s1), nn.SiLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), _norm(norm, c2, s2), nn.SiLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), _norm(norm, c3, s3), nn.SiLU())
        self.pool = POOL if (spatial == "pool" and s3 > POOL) else None
        side = self.pool or s3
        self.flat_dim = c3 * side * side
        self.fc = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.SiLU())
        self.out_dim = embed

    def forward(self, x):
        if self.coord:
            x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        f = self.conv(x)
        if self.pool is not None:
            f = F.adaptive_avg_pool2d(f, self.pool)
        return self.fc(f.flatten(1))


class _Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaEncoder(nn.Module):
    """IMPALA stacks. Inputs of 48 pixels or more (the 50 x 50 global map)
    first pass a stride-2 3x3 stem: residual blocks at full resolution made
    an update of the l/xl sizes 4-5 times slower than the CNN (178 / 225 ms
    at batch 128 on an RTX 4060), more than the workers can feed. The 25 x 25
    ego crop keeps its resolution."""

    def __init__(self, in_ch: int, size: int, channels: Sequence[int], embed: int):
        super().__init__()
        self.register_buffer("coords", _coords(size), persistent=False)
        layers: List[nn.Module] = []
        prev = in_ch + 2
        side = size
        if size >= 48:
            layers += [nn.Conv2d(prev, channels[0], 3, stride=2, padding=1), nn.ReLU()]
            prev = channels[0]
            side = (side + 2 - 3) // 2 + 1
        for ch in channels:
            layers += [nn.Conv2d(prev, ch, 3, padding=1),
                       nn.MaxPool2d(3, stride=2, padding=1),
                       _Residual(ch), _Residual(ch)]
            prev = ch
            side = (side + 2 - 3) // 2 + 1
        self.conv = nn.Sequential(*layers)
        self.pool = min(POOL, side)
        self.flat_dim = prev * self.pool * self.pool
        self.fc = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.SiLU())
        self.out_dim = embed

    def forward(self, x):
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], 1)
        f = F.relu(self.conv(x))
        f = F.adaptive_avg_pool2d(f, self.pool)
        return self.fc(f.flatten(1))


class DualEncoder(nn.Module):
    """Ego and global branches, concatenated."""

    def __init__(self, cfg, spec: dict, in_ch: int):
        super().__init__()
        self.ego_use = bool(cfg.EGO_USE)
        ego_hw = int(cfg.EGO_MAP_SIZE)
        glob_hw = int(cfg.DOWNSAMPLE_MAP_SIZE)

        def make(size):
            if spec["family"] == "impala":
                return ImpalaEncoder(in_ch, size, spec["channels"], spec["embed"])
            return ConvEncoder(in_ch, size, spec["channels"], spec["embed"],
                               norm=cfg.NET_NORM, spatial=cfg.NET_SPATIAL,
                               coord=cfg.NET_COORD_CHANNELS)

        self.ego = make(ego_hw) if self.ego_use else None
        self.glob = make(glob_hw)
        self.out_dim = self.glob.out_dim + (self.ego.out_dim if self.ego_use else 0)

    def forward(self, ego, glob):
        g = self.glob(glob)
        if self.ego_use:
            return torch.cat([self.ego(ego), g], dim=1)
        return g


# =============================================================================
# Heads
# =============================================================================

def squash_sample(mean, log_std, temperature: float = 1.0, stable: bool = True):
    """a = 4 sigmoid(u) - 2, u ~ N(mean, std); returns (a, log pi(a))."""
    std = log_std.exp()
    eps = torch.randn_like(mean) * temperature
    u = mean + std * eps
    sig = torch.sigmoid(u)
    action = 4 * sig - 2
    logp_u = (-0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)
    if stable:
        log_det = (math.log(4.0) + F.logsigmoid(u) + F.logsigmoid(-u)).sum(-1)
    else:  # RV3
        log_det = torch.log(4 * sig * (1 - sig) + 1e-8).sum(-1)
    return action, logp_u - log_det


class ActorHead(nn.Module):
    def __init__(self, feat_dim: int, state_dim: int, spec: dict,
                 log_std_min: float, log_std_max: float, action_dim: int = 2):
        super().__init__()
        self.robot_fc = nn.Sequential(nn.Linear(state_dim, spec["robot"]), nn.SiLU())
        h1, h2, h3 = spec["mlp"]
        self.backbone = nn.Sequential(
            nn.Linear(feat_dim + spec["robot"], h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, h3), nn.SiLU())
        self.mean_head = nn.Linear(h3, action_dim)
        self.log_std_head = nn.Linear(h3, action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

    def forward(self, feat, state):
        x = self.backbone(torch.cat([feat, self.robot_fc(state)], dim=1))
        log_std = torch.clamp(self.log_std_head(x), self.log_std_min, self.log_std_max)
        return self.mean_head(x), log_std


class CriticHead(nn.Module):
    def __init__(self, img_dim: int, state_dim: int, spec: dict, film: bool,
                 layernorm: bool, dropout: float, action_dim: int = 2):
        super().__init__()
        rdim = spec["robot"]
        self.robot_fc = nn.Sequential(nn.Linear(state_dim, rdim), nn.SiLU())
        self.film = None
        if film:
            self.film = nn.Sequential(nn.Linear(action_dim + rdim, 256), nn.SiLU(),
                                      nn.Linear(256, 2 * img_dim))
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)
        h1, h2 = spec["head"]
        layers: List[nn.Module] = []
        prev = img_dim + rdim + action_dim
        for h in (h1, h2):
            layers.append(nn.Linear(prev, h))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.q_out = nn.Linear(prev, 1)

    def forward(self, img, state, action):
        r = self.robot_fc(state)
        if self.film is not None:
            gamma, beta = self.film(torch.cat([action, r], 1)).chunk(2, dim=1)
            img = (1.0 + gamma) * img + beta
        return self.q_out(self.mlp(torch.cat([img, r, action], 1))).squeeze(-1)


# =============================================================================
# Actor and critic ensemble
# =============================================================================

class Actor(nn.Module):
    """forward(ego, glob, state) -> (mean, log_std). `state` is the actor's
    slice of the robot state. With a shared encoder, `detach_features` keeps
    the actor loss out of it."""

    def __init__(self, cfg, encoder: Optional[nn.Module] = None):
        super().__init__()
        self.legacy = cfg.NET_ENCODER == "legacy_rv3"
        self.stable = bool(cfg.SAC_STABLE_LOGPROB)
        self.use_scale = bool(cfg.MAP_SCALE_TO_ACTOR)
        if self.legacy:
            self.net = LegacyPolicyNetwork(
                ego_shape=(cfg.EGO_MAP_SIZE,) * 2, global_shape=(cfg.DOWNSAMPLE_MAP_SIZE,) * 2,
                ego_use=cfg.EGO_USE, log_std_min=cfg.LOG_STD_MIN, log_std_max=cfg.LOG_STD_MAX)
            self.encoder = None
            self.shared_encoder = False
            return
        spec = network_spec(cfg)
        self.shared_encoder = encoder is not None
        self.encoder = encoder if encoder is not None else DualEncoder(cfg, spec, cfg.obs_channels())
        self.head = ActorHead(self.encoder.out_dim, cfg.actor_state_dim(), spec,
                              cfg.LOG_STD_MIN, cfg.LOG_STD_MAX, action_dim=cfg.action_dim())
        self.detach_features = self.shared_encoder

    def forward(self, ego, glob, state):
        if self.legacy:
            return self.net(ego, glob, state)
        feat = self.encoder(ego, glob)
        if self.detach_features:
            feat = feat.detach()
        return self.head(feat, state)

    def sample(self, ego, glob, state, temperature: float = 1.0):
        mean, log_std = self.forward(ego, glob, state)
        return squash_sample(mean, log_std, temperature, self.stable)

    def deterministic(self, ego, glob, state):
        mean, _ = self.forward(ego, glob, state)
        return 4 * torch.sigmoid(mean) - 2

    def own_parameters(self):
        """Parameters the actor loss trains (not a shared encoder)."""
        if self.legacy:
            return list(self.net.parameters())
        if self.shared_encoder:
            return list(self.head.parameters())
        return list(self.parameters())


class CriticEnsemble(nn.Module):
    """N Q-functions. forward(...) -> (N, B)."""

    def __init__(self, cfg):
        super().__init__()
        self.legacy = cfg.NET_ENCODER == "legacy_rv3"
        self.n = int(cfg.CRITIC_ENSEMBLE)
        self.use_scale = bool(cfg.MAP_SCALE_TO_CRITIC)
        if self.legacy:
            self.qs = nn.ModuleList([
                LegacyQNetwork(ego_shape=(cfg.EGO_MAP_SIZE,) * 2,
                               global_shape=(cfg.DOWNSAMPLE_MAP_SIZE,) * 2,
                               ego_use=cfg.EGO_USE, film_use=cfg.FiLM_USE)
                for _ in range(self.n)])
            self.shared = False
            return
        spec = network_spec(cfg)
        self.shared = cfg.ENCODER_SHARING == "critic"
        ch = cfg.obs_channels()
        if self.shared:
            self.encoder = DualEncoder(cfg, spec, ch)
            self.encoders = None
            img = self.encoder.out_dim
        else:
            self.encoder = None
            self.encoders = nn.ModuleList([DualEncoder(cfg, spec, ch) for _ in range(self.n)])
            img = self.encoders[0].out_dim
        self.heads = nn.ModuleList([
            CriticHead(img, cfg.critic_state_dim(), spec, cfg.FiLM_USE,
                       cfg.CRITIC_LAYERNORM, float(cfg.CRITIC_DROPOUT), action_dim=cfg.action_dim())
            for _ in range(self.n)])

    def forward(self, ego, glob, state, action, members: Optional[Sequence[int]] = None):
        idx = range(self.n) if members is None else members
        if self.legacy:
            return torch.stack([self.qs[i](ego, glob, action, state).squeeze(-1) for i in idx])
        if self.shared:
            feat = self.encoder(ego, glob)
            return torch.stack([self.heads[i](feat, state, action) for i in idx])
        return torch.stack([self.heads[i](self.encoders[i](ego, glob), state, action) for i in idx])

    def head_modules(self) -> List[nn.Module]:
        if self.legacy:
            return [q for q in self.qs]
        return list(self.heads)


def build_networks(cfg) -> Tuple[Actor, CriticEnsemble, CriticEnsemble]:
    critic = CriticEnsemble(cfg)
    actor = Actor(cfg, encoder=critic.encoder if critic.shared else None)
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    return actor, critic, target


def build_policy(cfg) -> Actor:
    """The actor alone (workers, evaluation). In "critic" sharing mode it
    owns a copy of the encoder; its state_dict holds the encoder under
    encoder.* either way."""
    if cfg.NET_ENCODER != "legacy_rv3" and cfg.ENCODER_SHARING == "critic":
        spec = network_spec(cfg)
        enc = DualEncoder(cfg, spec, cfg.obs_channels())
        actor = Actor(cfg, encoder=enc)
        actor.detach_features = False
        return actor
    return Actor(cfg)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
