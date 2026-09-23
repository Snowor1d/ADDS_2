"""Actor and centralised critic for the multi-resolution observation.

The actor is shared and decentralised: one set of weights, run once per robot
on that robot's own input (local, middle and global branches plus its scalar
state, which includes what teammates' messages delivered). The critic sees
every robot's input and action at once, plus, during training only, the true
crowd map, and returns one value per robot; the team's value is their masked
mean (docs/outdoor_madrl_redesign.md section 5.1).

The action layout (move 2, signalled heading 2, mode one-hot) and its
squashing are unchanged from the previous networks, so sim/robot_action.py
still owns what an action means.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import observation as obsmod


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)


class ConvEncoder(nn.Module):
    """Three stride-2 convolutions, then a linear embedding.

    GroupNorm rather than BatchNorm: the workers run the actor one robot at
    a time in eval mode and the learner in batches in train mode, and batch
    statistics would make the two compute different functions.
    """

    def __init__(self, in_ch: int, size: int, embed: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), _gn(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), _gn(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), _gn(128), nn.SiLU(),
        )
        with torch.no_grad():
            o = self.conv(torch.zeros(1, in_ch, size, size))
        self.out_ch, self.out_h, self.out_w = o.shape[1:]
        self.flat_dim = int(np.prod(o.shape[1:]))
        self.fc = nn.Sequential(nn.Linear(self.flat_dim, embed), nn.SiLU())
        self.embed = embed

    def forward(self, x, return_2d: bool = False):
        f = self.conv(x)
        if return_2d:
            return f
        return self.fc(f.flatten(1))


def _shapes(cfg):
    return obsmod.obs_shapes(cfg)


class PolicyNetwork(nn.Module):
    """Shared actor. Input: one robot's ego, mid, glob and state."""

    def __init__(self, cfg, action_cont: int = 4):
        super().__init__()
        s = _shapes(cfg)
        self.log_std_min = float(cfg.LOG_STD_MIN)
        self.log_std_max = float(cfg.LOG_STD_MAX)
        self.ego = ConvEncoder(s["ego"][0], s["ego"][1])
        self.mid = ConvEncoder(s["mid"][0], s["mid"][1])
        self.glob = ConvEncoder(s["glob"][0], s["glob"][1])
        self.state = nn.Sequential(nn.Linear(s["state"][0], 64), nn.SiLU())
        self.backbone = nn.Sequential(
            nn.Linear(3 * 256 + 64, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, 64), nn.SiLU())
        self.mean_head = nn.Linear(64, action_cont)
        self.log_std_head = nn.Linear(64, action_cont)
        self.mode_head = nn.Linear(64, len(cfg.ROBOT_MODES))

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
    """

    def __init__(self, cfg, action_dim: int, embed: int = 256, heads: int = 4):
        super().__init__()
        s = _shapes(cfg)
        self.privileged = bool(cfg.CRITIC_PRIVILEGED_CROWD)
        self.ego = ConvEncoder(s["ego"][0], s["ego"][1], embed)
        self.mid = ConvEncoder(s["mid"][0], s["mid"][1], embed)
        self.glob = ConvEncoder(s["glob"][0], s["glob"][1], embed)
        self.priv = ConvEncoder(1, s["priv"][1], embed)
        self.state = nn.Sequential(nn.Linear(s["state"][0], 64), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(2 * embed + 64, 128), nn.SiLU())
        self.attend = SpatialAttention(self.glob.out_ch, 128)
        self.glob_fc = nn.Sequential(nn.Linear(self.glob.flat_dim, embed),
                                     nn.SiLU())
        vision = 4 * embed
        self.agent = nn.Sequential(
            nn.Linear(s["state"][0] + action_dim, embed), nn.SiLU(),
            nn.Linear(embed, embed))
        self.film = nn.Sequential(nn.Linear(embed, 256), nn.SiLU(),
                                  nn.Linear(256, 2 * vision))
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.film_out = nn.Linear(vision, embed)
        self.attn = nn.MultiheadAttention(embed, heads, batch_first=True)
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Sequential(nn.Linear(embed, 256), nn.SiLU(),
                                  nn.Linear(256, 128), nn.SiLU(),
                                  nn.Linear(128, 1))

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
        g2d = self.attend(self.glob(flat(glob), return_2d=True), ctx)
        g = self.glob_fc(g2d.flatten(1))
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
