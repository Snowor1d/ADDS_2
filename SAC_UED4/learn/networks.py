"""Actor and centralised critic for the multi-resolution observation.

The actor is shared and decentralised: one set of weights, run once per robot
on that robot's own input (local, middle and global branches plus its scalar
state, which includes what teammates' messages delivered). The critic sees
every robot's input and action at once, plus, during training only, the true
crowd map, and returns one value per robot; the team's value is their masked
mean (docs/outdoor_madrl_redesign.md section 5.1).

Two encoder families, three sizes each (NET_ENCODER, NET_SIZE):

  cnn     three stride-2 convolutions, then the feature map flattened into a
          linear embedding. Cheap convolutions, most parameters in the
          position-specific linear layer. "m" is the original network.
  impala  three stacks of convolution, max-pool and two residual blocks
          (Espeholt et al. 2018; Cobbe et al. 2020 found it generalises to
          unseen levels far better than a shallow CNN, and more so as it
          widens). Coordinate channels are appended to the input and the
          feature map is average-pooled to a 4 x 4 grid before the embedding,
          so position survives while most parameters sit in convolutions.

The same size has about the same parameter count in both families (see
NETWORK_SIZES and tests/test_madrl.py), so a comparison between them is a
comparison of structure rather than of capacity.

The action layout (move 2, the signalled heading 2 only with USE_DIRECT, then
the mode one-hot) is owned by sim/robot_action.py; the actor's continuous
head is as wide as its CONT_DIM and its mode head as wide as ROBOT_MODES.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import observation as obsmod

# Per family and size: conv channels per stage, embedding width per branch,
# actor MLP widths, critic context width and critic head widths. Chosen so
# that the actor is about 5.6 M parameters at "m", 12 M at "l" and 21 M at
# "xl" in both families.
NETWORK_SIZES = {
    "cnn": {
        "m":  dict(channels=(32, 64, 128), embed=256, mlp=(512, 256, 64),
                   context=128, head=(256, 128)),
        "l":  dict(channels=(48, 96, 192), embed=384, mlp=(768, 384, 128),
                   context=192, head=(384, 192)),
        "xl": dict(channels=(64, 128, 256), embed=512, mlp=(1024, 512, 128),
                   context=256, head=(512, 256)),
    },
    # The first stage is kept narrow: it runs at the highest resolution, so it
    # sets the activation memory and the update time, while the parameters
    # live in the later, coarser stages.
    "impala": {
        "m":  dict(channels=(32, 80, 144), embed=256, mlp=(512, 256, 64),
                   context=128, head=(256, 128)),
        "l":  dict(channels=(48, 128, 208), embed=384, mlp=(768, 384, 128),
                   context=192, head=(384, 192)),
        "xl": dict(channels=(64, 192, 256), embed=512, mlp=(1024, 512, 128),
                   context=256, head=(512, 256)),
    },
}

# Side of the grid the IMPALA feature map is pooled to before embedding.
IMPALA_POOL = 4


def network_spec(cfg) -> dict:
    fam = str(cfg.NET_ENCODER)
    size = str(cfg.NET_SIZE)
    if fam not in NETWORK_SIZES or size not in NETWORK_SIZES[fam]:
        raise ValueError(f"no network {fam!r} size {size!r}")
    return dict(NETWORK_SIZES[fam][size], family=fam, size=size)


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ConvEncoder(nn.Module):
    """Three stride-2 convolutions, then a linear embedding (family "cnn").

    GroupNorm rather than BatchNorm: the workers run the actor one robot at
    a time in eval mode and the learner in batches in train mode, and batch
    statistics would make the two compute different functions.
    """

    def __init__(self, in_ch: int, size: int, embed: int = 256,
                 channels: Sequence[int] = (32, 64, 128)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, c1, 5, stride=2, padding=2), _gn(c1), nn.SiLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), _gn(c2), nn.SiLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), _gn(c3), nn.SiLU(),
        )
        with torch.no_grad():
            o = self.conv(torch.zeros(1, in_ch, size, size))
        self.out_ch, self.out_h, self.out_w = o.shape[1:]
        self.flat_dim = int(np.prod(o.shape[1:]))
        self.fc = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.SiLU())
        self.embed = embed

    def features(self, x):
        return self.conv(x)

    def head(self, f):
        return self.fc(f.flatten(1))

    def forward(self, x):
        return self.head(self.features(x))


class _Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaEncoder(nn.Module):
    """IMPALA stacks (conv, max-pool /2, two residual blocks) x 3, then an
    average pool to IMPALA_POOL x IMPALA_POOL and a linear embedding.

    Two coordinate channels are appended to the input, so the convolutions
    themselves can tell where on the grid a feature is.
    """

    def __init__(self, in_ch: int, size: int, embed: int = 256,
                 channels: Sequence[int] = (48, 96, 128)):
        super().__init__()
        ys = torch.linspace(-1.0, 1.0, size)
        yy, xx = torch.meshgrid(ys, ys, indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None],
                             persistent=False)
        stages = []
        prev = in_ch + 2
        # A stride-2 stem on the 64 x 64 inputs, as the "cnn" family's first
        # layer does: residual blocks at full resolution made an update four
        # to six times slower than the CNN of the same size and did not fit
        # batch 128 in 8 GB. The 25 x 25 local crop keeps its resolution.
        if size >= 48:
            stem = int(channels[0])
            stages += [nn.Conv2d(prev, stem, 3, stride=2, padding=1), nn.ReLU()]
            prev = stem
        for ch in channels:
            stages += [nn.Conv2d(prev, ch, 3, padding=1),
                       nn.MaxPool2d(3, stride=2, padding=1),
                       _Residual(ch), _Residual(ch)]
            prev = ch
        self.conv = nn.Sequential(*stages)
        with torch.no_grad():
            o = self.conv(torch.zeros(1, in_ch + 2, size, size))
        self.out_ch, self.out_h, self.out_w = o.shape[1:]
        self.pool = min(IMPALA_POOL, int(self.out_h))
        self.flat_dim = int(self.out_ch) * self.pool * self.pool
        self.fc = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.SiLU())
        self.embed = embed

    def features(self, x):
        c = self.coords.expand(x.shape[0], -1, -1, -1)
        return F.relu(self.conv(torch.cat([x, c], dim=1)))

    def head(self, f):
        f = F.adaptive_avg_pool2d(f, self.pool)
        return self.fc(f.flatten(1))

    def forward(self, x):
        return self.head(self.features(x))


def make_encoder(spec: dict, in_ch: int, size: int) -> nn.Module:
    cls = ImpalaEncoder if spec["family"] == "impala" else ConvEncoder
    return cls(in_ch, size, embed=spec["embed"], channels=spec["channels"])


def _shapes(cfg):
    return obsmod.obs_shapes(cfg)


class PolicyNetwork(nn.Module):
    """Shared actor. Input: one robot's ego, mid, glob and state."""

    def __init__(self, cfg, action_cont: int = None):
        super().__init__()
        from sim import robot_action
        # Move (2), plus the signalled heading (2) only when "direct" exists.
        action_cont = robot_action.CONT_DIM if action_cont is None else action_cont
        s = _shapes(cfg)
        spec = network_spec(cfg)
        self.spec = spec
        self.log_std_min = float(cfg.LOG_STD_MIN)
        self.log_std_max = float(cfg.LOG_STD_MAX)
        self.ego = make_encoder(spec, s["ego"][0], s["ego"][1])
        self.mid = make_encoder(spec, s["mid"][0], s["mid"][1])
        self.glob = make_encoder(spec, s["glob"][0], s["glob"][1])
        self.state = nn.Sequential(nn.Linear(s["state"][0], 64), nn.SiLU())
        h1, h2, h3 = spec["mlp"]
        self.backbone = nn.Sequential(
            nn.Linear(3 * spec["embed"] + 64, h1), nn.SiLU(),
            nn.Linear(h1, h2), nn.SiLU(),
            nn.Linear(h2, h3), nn.SiLU())
        self.mean_head = nn.Linear(h3, action_cont)
        self.log_std_head = nn.Linear(h3, action_cont)
        self.mode_head = nn.Linear(h3, len(cfg.ROBOT_MODES))

    def features(self, ego, mid, glob, state):
        return self.backbone(torch.cat([self.ego(ego), self.mid(mid),
                                        self.glob(glob), self.state(state)],
                                       dim=-1))

    def forward(self, ego, mid, glob, state):
        f = self.features(ego, mid, glob, state)
        log_std = torch.clamp(self.log_std_head(f), self.log_std_min,
                              self.log_std_max)
        return self.mean_head(f), log_std, self.mode_head(f)

    def deterministic_action(self, ego, mid, glob, state):
        mean, _, logits = self.forward(ego, mid, glob, state)
        cont = 4 * torch.sigmoid(mean) - 2
        one_hot = F.one_hot(logits.argmax(-1), logits.shape[-1]).float()
        return torch.cat([cont, one_hot], dim=-1)

    def sample_action(self, ego, mid, glob, state, temperature: float = 1.0):
        """Squashed Gaussian move and heading, straight-through Gumbel mode.

        Same distribution and log-probability as the previous actor.
        """
        mean, log_std, logits = self.forward(ego, mid, glob, state)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean) * temperature
        sig = torch.sigmoid(u)
        cont = 4 * sig - 2
        logp_u = -0.5 * (((u - mean) / (std + 1e-8)) ** 2 + 2 * log_std
                         + np.log(2 * np.pi))
        logp = logp_u.sum(-1) - torch.log(4 * sig * (1 - sig) + 1e-8).sum(-1)
        log_mode = F.log_softmax(logits, dim=-1)
        one_hot = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        logp = logp + (one_hot * log_mode).sum(-1)
        return torch.cat([cont, one_hot], dim=-1), logp


class SpatialAttention(nn.Module):
    """Which part of the global map to read, given one robot's context."""

    def __init__(self, channels: int, context: int, embed: int = 64):
        super().__init__()
        self.key = nn.Conv2d(channels, embed, 1)
        self.query = nn.Linear(context, embed)
        self.mask = nn.Sequential(nn.Conv2d(embed, 1, 1), nn.Sigmoid())

    def forward(self, g2d, ctx):
        B, C, H, W = g2d.shape
        q = self.query(ctx).view(B, -1, 1, 1)
        return g2d * self.mask(self.key(g2d) * q)


class CentralizedCritic(nn.Module):
    """Q per robot from the joint observation and joint action.

    Attention over robots, so the value does not depend on slot order and a
    shorter team is a mask rather than a different network. The privileged
    crowd map enters as one extra embedding shared by every robot's token.
    The global branch is read through a per-robot spatial attention on its
    feature map and then embedded by the same encoder's own head.
    """

    def __init__(self, cfg, action_dim: int, heads: int = 4):
        super().__init__()
        s = _shapes(cfg)
        spec = network_spec(cfg)
        self.spec = spec
        embed = spec["embed"]
        ctx_w = spec["context"]
        self.privileged = bool(cfg.CRITIC_PRIVILEGED_CROWD)
        self.ego = make_encoder(spec, s["ego"][0], s["ego"][1])
        self.mid = make_encoder(spec, s["mid"][0], s["mid"][1])
        self.glob = make_encoder(spec, s["glob"][0], s["glob"][1])
        self.priv = make_encoder(spec, 1, s["priv"][1])
        self.state = nn.Sequential(nn.Linear(s["state"][0], 64), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(2 * embed + 64, ctx_w),
                                     nn.SiLU())
        self.attend = SpatialAttention(self.glob.out_ch, ctx_w)
        vision = 4 * embed
        self.agent = nn.Sequential(
            nn.Linear(s["state"][0] + action_dim, embed), nn.SiLU(),
            nn.Linear(embed, embed))
        self.film = nn.Sequential(nn.Linear(embed, embed), nn.SiLU(),
                                  nn.Linear(embed, 2 * vision))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.film_out = nn.Linear(vision, embed)
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed)
        h1, h2 = spec["head"]
        self.head = nn.Sequential(nn.Linear(embed, h1), nn.SiLU(),
                                  nn.Linear(h1, h2), nn.SiLU(),
                                  nn.Linear(h2, 1))

    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """obs: ego/mid/glob/state (B, N, ...), priv (B, 1, G, G).
        Returns (B, N) values, zero in padded slots."""
        ego, mid, glob, state = obs["ego"], obs["mid"], obs["glob"], obs["state"]
        B, N = state.shape[:2]
        flat = lambda x: x.reshape(B * N, *x.shape[2:])
        e = self.ego(flat(ego))
        m = self.mid(flat(mid))
        st = self.state(flat(state))
        ctx = self.context(torch.cat([e, m, st], -1))
        g2d = self.attend(self.glob.features(flat(glob)), ctx)
        g = self.glob.head(g2d)
        if self.privileged:
            p = self.priv(obs["priv"])                      # (B, embed)
        else:
            p = torch.zeros(B, e.shape[-1], device=e.device)
        p = p.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        vision = torch.cat([e, m, g, p], -1)
        agent = self.agent(torch.cat([flat(state), flat(action)], -1))
        gamma, beta = self.film(agent).chunk(2, -1)
        tok = F.silu(self.film_out((1 + gamma) * vision + beta)) + agent
        tok = tok.reshape(B, N, -1) * mask.unsqueeze(-1)
        pad = mask <= 0
        out, _ = self.attn(tok, tok, tok, key_padding_mask=pad)
        tok = self.norm(tok + out) * mask.unsqueeze(-1)
        return self.head(tok).squeeze(-1) * mask


def team_value(per_robot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean over real robots: the team's value, (B,)."""
    return (per_robot * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
