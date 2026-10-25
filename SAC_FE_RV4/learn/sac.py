"""Soft Actor-Critic update with the generalisation options of
configs/training/common.py.

One `update(batch)`:

    augmentation   D4 symmetry (same k for state, action and next state) and
                   random shift (independent for each view, DrQ); K views of
                   the next state are averaged in the target, M views of the
                   state in the critic loss
    critic         target = R_n + gamma^m (1 - done) (min over a random subset
                   of the target ensemble - alpha log pi), MSE for every
                   member
    actor          every POLICY_DELAY critic updates; min of two critics, or
                   the ensemble mean when there are more than two (REDQ)
    alpha          learned towards ALPHA_TARGET_ENTROPY when ALPHA_AUTO
    targets        polyak, target <- TAU target + (1 - TAU) online
    resets         RESET_EVERY_UPDATES: re-initialise heads or everything, or
                   shrink and perturb

With NET_ENCODER = "legacy_rv3" and the RV3 settings this is RV3's update:
twin critics, fixed alpha, min-Q actor loss, no augmentation.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learn import obs as obsmod
from learn.networks import build_networks

WALL = obsmod.WALL / 255.0


def random_shift(x: torch.Tensor, pad: int, fill: float = WALL) -> torch.Tensor:
    """Pad by `pad` pixels of wall and crop back at a random offset, per
    sample (RAD / DrQ)."""
    if pad <= 0:
        return x
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad), value=fill)
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = (oy[:, None] + torch.arange(H, device=x.device)[None])          # (B, H)
    cols = (ox[:, None] + torch.arange(W, device=x.device)[None])          # (B, W)
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]    # (B, H, W, C)
    return out.permute(0, 3, 1, 2).contiguous()


def d4_batch(batch: Dict[str, torch.Tensor], k: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Apply symmetry k[b] to sample b of every map, position and action."""
    out = {key: v.clone() for key, v in batch.items()}
    for kk in range(1, 8):
        m = (k == kk)
        if not bool(m.any()):
            continue
        for key in ("ego", "glob", "ego2", "glob2"):
            out[key][m] = obsmod.d4_image(batch[key][m], kk)
        for key in ("state", "state2"):
            out[key][m] = obsmod.d4_state(batch[key][m], kk)
        out["action"][m] = obsmod.d4_action(batch["action"][m], kk)
    return out


class SACLearner:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        if self.device.type == "cuda":
            # Fixed input shapes: let cuDNN pick the fastest convolution once.
            torch.backends.cudnn.benchmark = True
        self.actor, self.critic, self.target = build_networks(cfg)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target.to(self.device)
        self.legacy = cfg.NET_ENCODER == "legacy_rv3"
        self.n = int(cfg.CRITIC_ENSEMBLE)
        self.m_target = int(cfg.CRITIC_TARGET_SUBSET)
        self.tau = float(cfg.TAU)
        self.gamma = float(cfg.GAMMA_START)
        self.actor_scale = bool(cfg.MAP_SCALE_TO_ACTOR)
        self.critic_scale = bool(cfg.MAP_SCALE_TO_CRITIC)
        self.target_entropy = float(cfg.ALPHA_TARGET_ENTROPY)
        self.log_alpha = torch.tensor(math.log(float(cfg.ALPHA_START)), device=self.device,
                                      requires_grad=True)
        self._make_optimizers()
        self.updates = 0
        self.critic_updates = 0
        self.rng = np.random.default_rng()

    def _make_optimizers(self):
        c = self.cfg
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=c.lr_critic(),
                                            weight_decay=float(c.WD_Q))
        self.actor_opt = torch.optim.AdamW(self.actor.own_parameters(), lr=c.lr_actor(),
                                           weight_decay=float(c.WD_PI))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=float(c.ALPHA_LR))

    # ------------------------------------------------------------ helpers

    @property
    def alpha(self) -> torch.Tensor:
        a = self.log_alpha.exp().detach()
        if self.cfg.ALPHA_AUTO and self.cfg.ALPHA_MIN > 0:
            a = a.clamp(min=float(self.cfg.ALPHA_MIN))
        return a

    def _sa(self, s):
        return obsmod.network_state(s, self.actor_scale)

    def _sc(self, s):
        return obsmod.network_state(s, self.critic_scale)

    def _shift(self, ego, glob):
        c = self.cfg
        if not c.AUG_RANDOM_SHIFT:
            return ego, glob
        return random_shift(ego, int(c.AUG_SHIFT_PAD_EGO)), random_shift(glob, int(c.AUG_SHIFT_PAD_GLOBAL))

    def _views(self, ego, glob, n: int):
        """n augmented copies stacked along the batch."""
        if n == 1:
            return self._shift(ego, glob)
        es, gs = zip(*[self._shift(ego, glob) for _ in range(n)])
        return torch.cat(es, 0), torch.cat(gs, 0)

    # ------------------------------------------------------------- update

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        c = self.cfg
        if c.AUG_REPLAY_D4:
            k = torch.randint(0, 8, (batch["ego"].shape[0],), device=self.device)
            batch = d4_batch(batch, k)
        B = batch["ego"].shape[0]
        K, M = int(c.AUG_DRQ_K), int(c.AUG_DRQ_M)
        alpha = self.alpha

        with torch.no_grad():
            ego2, glob2 = self._views(batch["ego2"], batch["glob2"], K)
            s2 = batch["state2"].repeat(K, 1)
            a2, logp2 = self.actor.sample(ego2, glob2, self._sa(s2))
            members = (None if self.m_target >= self.n
                       else sorted(self.rng.choice(self.n, self.m_target, replace=False).tolist()))
            q2 = self.target(ego2, glob2, self._sc(s2), a2, members).min(0).values
            v2 = (q2 - alpha * logp2).view(K, B).mean(0)
            y = batch["reward"] + batch["discount"] * (1.0 - batch["done"]) * v2

        ego, glob = self._views(batch["ego"], batch["glob"], M)
        s = batch["state"].repeat(M, 1)
        a = batch["action"].repeat(M, 1)
        q = self.critic(ego, glob, self._sc(s), a)                 # (N, M*B)
        q_loss = F.mse_loss(q, y.repeat(M)[None].expand_as(q), reduction="none").mean(1).sum()
        self.critic_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        if c.GRAD_CLIP_NORM:
            nn.utils.clip_grad_norm_(self.critic.parameters(), float(c.GRAD_CLIP_NORM))
        self.critic_opt.step()
        self.critic_updates += 1

        out = {"train/q_loss": float(q_loss.detach()) / self.n,
               "train/q_mean": float(q.detach().mean()),
               "train/target_mean": float(y.mean()),
               "train/alpha": float(alpha)}

        if self.critic_updates % max(1, int(c.POLICY_DELAY)) == 0:
            ego_a, glob_a, s_a = ego[:B], glob[:B], batch["state"]
            for p in self.critic.parameters():
                p.requires_grad_(False)
            new_a, logp = self.actor.sample(ego_a, glob_a, self._sa(s_a))
            qa = self.critic(ego_a, glob_a, self._sc(s_a), new_a)
            q_pi = qa.min(0).values if self.n <= 2 else qa.mean(0)
            actor_loss = (alpha * logp - q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if c.GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(self.actor.own_parameters(), float(c.GRAD_CLIP_NORM))
            self.actor_opt.step()
            for p in self.critic.parameters():
                p.requires_grad_(True)
            out["train/actor_loss"] = float(actor_loss.detach())
            out["train/entropy"] = float(-logp.detach().mean())
            if c.ALPHA_AUTO:
                a_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                a_loss.backward()
                self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.target.parameters()):
                tp.mul_(self.tau).add_(p.detach(), alpha=1.0 - self.tau)
            if self.legacy:
                # RV3 never copied BatchNorm running statistics into the
                # targets; they track their own batches in train mode.
                pass
            else:
                for b, tb in zip(self.critic.buffers(), self.target.buffers()):
                    tb.copy_(b)
        self.updates += 1
        if c.RESET_EVERY_UPDATES and self.updates % int(c.RESET_EVERY_UPDATES) == 0:
            self.reset_networks()
            out["train/reset"] = 1.0
        return out

    # ------------------------------------------------------------- resets

    def reset_networks(self) -> None:
        """Primacy-bias reset (Nikishin et al. 2022) or shrink & perturb
        (Ash & Adams 2020), keeping the replay."""
        c = self.cfg
        fresh_actor, fresh_critic, _ = build_networks(c)
        fresh_actor.to(self.device)
        fresh_critic.to(self.device)
        alpha_sp = float(c.SHRINK_PERTURB_ALPHA)

        def blend(dst: nn.Module, src: nn.Module):
            with torch.no_grad():
                for p, q in zip(dst.parameters(), src.parameters()):
                    if alpha_sp > 0:
                        p.mul_(alpha_sp).add_(q, alpha=1.0 - alpha_sp)
                    else:
                        p.copy_(q)

        if c.RESET_SCOPE == "all":
            blend(self.critic, fresh_critic)
            if self.actor.shared_encoder:
                blend(self.actor.head, fresh_actor.head)
            else:
                blend(self.actor, fresh_actor)
        else:
            for h, fh in zip(self.critic.head_modules(), fresh_critic.head_modules()):
                if self.legacy:
                    for name in ("fc1", "fc2", "q_out", "film", "robot_fc"):
                        if getattr(h, name, None) is not None:
                            blend(getattr(h, name), getattr(fh, name))
                else:
                    blend(h, fh)
            if self.legacy:
                for name in ("fc_backbone", "mean_head", "log_std_head", "robot_fc"):
                    blend(getattr(self.actor.net, name), getattr(fresh_actor.net, name))
            else:
                blend(self.actor.head, fresh_actor.head)
        self.target.load_state_dict(self.critic.state_dict())
        self._make_optimizers()

    # --------------------------------------------------------- state I/O

    def policy_state_dict(self) -> Dict[str, torch.Tensor]:
        """What the workers load: the actor (with its encoder)."""
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target": self.target.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": float(self.log_alpha.detach()),
            "updates": self.updates,
            "critic_updates": self.critic_updates,
            "gamma": self.gamma,
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.critic.load_state_dict(sd["critic"])
        self.actor.load_state_dict(sd["actor"])
        self.target.load_state_dict(sd["target"])
        try:
            self.critic_opt.load_state_dict(sd["critic_opt"])
            self.actor_opt.load_state_dict(sd["actor_opt"])
            self.alpha_opt.load_state_dict(sd["alpha_opt"])
        except Exception as e:  # an optimiser layout change is not fatal
            print(f"[sac] optimiser state not restored: {e}")
        with torch.no_grad():
            self.log_alpha.fill_(float(sd["log_alpha"]))
        self.updates = int(sd.get("updates", 0))
        self.critic_updates = int(sd.get("critic_updates", 0))
        self.gamma = float(sd.get("gamma", self.gamma))

    def load_legacy_rv3(self, ckpt: Dict) -> None:
        """An RV3 checkpoint into NET_ENCODER = "legacy_rv3" networks."""
        assert self.legacy, "an RV3 checkpoint needs NET_ENCODER='legacy_rv3'"
        self.actor.net.load_state_dict(ckpt["policy"])
        self.critic.qs[0].load_state_dict(ckpt["q1"])
        self.critic.qs[1].load_state_dict(ckpt["q2"])
        self.target.qs[0].load_state_dict(ckpt["q1_target"])
        self.target.qs[1].load_state_dict(ckpt["q2_target"])

    # ----------------------------------------------------------- values

    @torch.no_grad()
    def state_values(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """V(s) = E_a[min Q(s, a) - alpha log pi(a|s)] estimated with one
        sample, for curriculum scores (ued/score.py)."""
        a, logp = self.actor.sample(batch["ego"], batch["glob"], self._sa(batch["state"]))
        q = self.critic(batch["ego"], batch["glob"], self._sc(batch["state"]), a).min(0).values
        return q - self.alpha * logp
