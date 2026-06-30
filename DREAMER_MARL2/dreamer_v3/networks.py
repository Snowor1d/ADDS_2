from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import DreamerConfig


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def action_to_env(raw_action: torch.Tensor, cfg: DreamerConfig) -> torch.Tensor:
    env_action = cfg.spd_min + 0.5 * (cfg.spd_max - cfg.spd_min) * (raw_action + 1.0)
    if cfg.action_dim == 2:
        max_speed = max(abs(float(cfg.spd_min)), abs(float(cfg.spd_max)))
        norm = torch.linalg.norm(env_action, dim=-1, keepdim=True)
        scale = torch.clamp(max_speed / norm.clamp_min(1e-6), max=1.0)
        env_action = env_action * scale
    return env_action


def env_to_raw_action(action: torch.Tensor, cfg: DreamerConfig) -> torch.Tensor:
    scale = max(cfg.spd_max - cfg.spd_min, 1e-6)
    return (2.0 * (action - cfg.spd_min) / scale - 1.0).clamp(-1.0, 1.0)


def normalize_robot_state(robot_state: torch.Tensor, cfg: DreamerConfig) -> torch.Tensor:
    robot_state = torch.nan_to_num(robot_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
    robot_state = torch.clamp(robot_state, -cfg.robot_state_clip, cfg.robot_state_clip)
    if cfg.robot_dim >= 6:
        normalized = robot_state.clone()
        normalized[..., 0:2] = normalized[..., 0:2].clamp(0.0, 1.0)
        normalized[..., 2:4] = normalized[..., 2:4].clamp(-1.0, 1.0)
        normalized[..., 4:6] = (normalized[..., 4:6] / cfg.robot_map_scale_norm).clamp(0.0, 1.0)
        return normalized
    return robot_state


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, out_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_size),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.out_shape = tuple(int(x) for x in out_shape)
        out_channels, height, width = self.out_shape
        self.base_channels = int(hidden_size)
        self.start_h = max(3, int(math.ceil(height / 8)))
        self.start_w = max(3, int(math.ceil(width / 8)))
        mid_channels = max(self.base_channels // 2, 32)
        low_channels = max(self.base_channels // 4, 32)

        self.fc = nn.Sequential(
            nn.Linear(in_size, self.base_channels * self.start_h * self.start_w),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.base_channels, mid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(mid_channels, low_channels, 3, padding=1),
            nn.SiLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(low_channels, low_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(low_channels, out_channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        leading = feat.shape[:-1]
        x = feat.reshape(-1, feat.shape[-1])
        x = self.fc(x).reshape(-1, self.base_channels, self.start_h, self.start_w)
        x = self.net(x)
        _, height, width = self.out_shape
        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.reshape(*leading, *self.out_shape)


class JointObsEncoder(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        c_ego, _, _ = cfg.ego_shape
        c_global, _, _ = cfg.global_shape
        self.cfg = cfg
        self.ego_encoder = ConvEncoder(c_ego, cfg.hidden_size)
        self.global_encoder = ConvEncoder(c_global, cfg.hidden_size)
        self.robot_encoder = nn.Sequential(
            nn.Linear(cfg.robot_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2 + 64, cfg.embed_size),
            nn.SiLU(),
            nn.Linear(cfg.embed_size, cfg.embed_size),
            nn.SiLU(),
        )

    def forward(
        self,
        joint_ego: torch.Tensor,
        global_state: torch.Tensor,
        joint_robot: torch.Tensor,
        joint_mask: torch.Tensor,
    ) -> torch.Tensor:
        leading = joint_ego.shape[:-4]
        n = joint_ego.shape[-4]
        ego = joint_ego.reshape(-1, *joint_ego.shape[-3:])
        ego_feat = self.ego_encoder(ego).reshape(*leading, n, -1)
        robot_feat = self.robot_encoder(normalize_robot_state(joint_robot, self.cfg))
        mask = joint_mask.unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        ego_pooled = (ego_feat * mask).sum(dim=-2) / denom
        robot_pooled = (robot_feat * mask).sum(dim=-2) / denom

        global_feat = self.global_encoder(global_state.reshape(-1, *global_state.shape[-3:]))
        global_feat = global_feat.reshape(*leading, -1)
        return self.out(torch.cat([ego_pooled, global_feat, robot_pooled], dim=-1))

class TwoHotSymlogHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, bins: int, low: float, high: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, bins),
        )
        self.bins = int(bins)
        self.low = float(low)
        self.high = float(high)
        self.register_buffer("support", torch.linspace(self.low, self.high, self.bins))

    def logits(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.logits(feat)
        symlog_value = torch.softmax(logits, dim=-1).mul(self.support).sum(dim=-1)
        return symexp(symlog_value)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.logits(feat)
        target_dist = self.twohot(target).to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1).mean()

    def twohot(self, target: torch.Tensor) -> torch.Tensor:
        y = symlog(target).clamp(self.low, self.high)
        pos = (y - self.low) / (self.high - self.low) * (self.bins - 1)
        lower = torch.floor(pos).long()
        upper = torch.clamp(lower + 1, max=self.bins - 1)
        lower = torch.clamp(lower, min=0, max=self.bins - 1)
        upper_weight = (pos - lower.float()).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight
        dist = torch.zeros(*target.shape, self.bins, device=target.device, dtype=torch.float32)
        dist.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        dist.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return dist


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


class RSSM(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stoch_flat = cfg.stoch_size * cfg.discrete_size
        joint_action_dim = cfg.max_robots * cfg.action_dim
        joint_robot_dim = cfg.max_robots * cfg.robot_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(joint_action_dim + joint_robot_dim + cfg.max_robots, cfg.action_embed_size),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(self.stoch_flat + cfg.action_embed_size, cfg.deter_size)
        self.prior = nn.Sequential(
            nn.Linear(cfg.deter_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, self.stoch_flat),
        )
        self.posterior = nn.Sequential(
            nn.Linear(cfg.deter_size + cfg.embed_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, self.stoch_flat),
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.cfg.deter_size, device=device)
        logits = torch.zeros(batch_size, self.cfg.stoch_size, self.cfg.discrete_size, device=device)
        stoch = F.one_hot(
            torch.zeros(batch_size, self.cfg.stoch_size, dtype=torch.long, device=device),
            self.cfg.discrete_size,
        ).float()
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def obs_step(
        self,
        prev: RSSMState,
        prev_action: torch.Tensor,
        prev_robot: torch.Tensor,
        prev_mask: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        prior = self.img_step(prev, prev_action, prev_robot, prev_mask)
        logits = self.posterior(torch.cat([prior.deter, embed], dim=-1))
        logits = logits.reshape(-1, self.cfg.stoch_size, self.cfg.discrete_size)
        stoch = self._sample(logits)
        post = RSSMState(deter=prior.deter, stoch=stoch, logits=logits)
        return post, prior

    def img_step(
        self,
        prev: RSSMState,
        action: torch.Tensor,
        robot: torch.Tensor,
        mask: torch.Tensor,
    ) -> RSSMState:
        action_flat = action.reshape(action.shape[0], -1)
        robot_flat = normalize_robot_state(robot, self.cfg).reshape(robot.shape[0], -1)
        action_embed = self.action_encoder(torch.cat([action_flat, robot_flat, mask], dim=-1))
        x = torch.cat([prev.stoch.reshape(prev.stoch.shape[0], -1), action_embed], dim=-1)
        deter = self.gru(x, prev.deter)
        logits = self.prior(deter).reshape(-1, self.cfg.stoch_size, self.cfg.discrete_size)
        stoch = self._sample(logits)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch.reshape(state.stoch.shape[0], -1)], dim=-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        probs = self.dist_probs(logits)
        index = torch.distributions.Categorical(probs=probs).sample()
        sample = F.one_hot(index, self.cfg.discrete_size).float()
        return sample + probs - probs.detach()

    def dist_probs(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        unimix = float(self.cfg.unimix_ratio)
        if unimix <= 0.0:
            return probs
        uniform = torch.full_like(probs, 1.0 / self.cfg.discrete_size)
        return (1.0 - unimix) * probs + unimix * uniform

    def dist(self, logits: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(probs=self.dist_probs(logits))


class WorldModel(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = JointObsEncoder(cfg)
        self.rssm = RSSM(cfg)
        feat_size = cfg.deter_size + cfg.stoch_size * cfg.discrete_size
        self.reward_head = TwoHotSymlogHead(
            feat_size,
            cfg.hidden_size,
            cfg.twohot_bins,
            cfg.twohot_min,
            cfg.twohot_max,
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feat_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, 1),
        )
        self.ego_decoder = ConvDecoder(
            feat_size,
            cfg.hidden_size,
            (cfg.max_robots * cfg.ego_shape[0], cfg.ego_shape[1], cfg.ego_shape[2]),
        )
        self.global_decoder = ConvDecoder(
            feat_size,
            cfg.hidden_size,
            cfg.global_shape,
        )

    def observe(self, batch: dict[str, torch.Tensor]) -> tuple[list[RSSMState], list[RSSMState]]:
        joint_ego = batch["joint_ego"]
        global_state = batch["global_state"]
        joint_robot = batch["joint_robot"]
        joint_mask = batch["joint_mask"]
        action = batch["action"]
        b, t = joint_ego.shape[:2]
        embed = self.encoder(joint_ego, global_state, joint_robot, joint_mask)
        prev = self.rssm.initial(b, joint_ego.device)
        posts = []
        priors = []
        zero_action = torch.zeros_like(action[:, 0])
        for i in range(t):
            prev_action = zero_action if i == 0 else action[:, i - 1]
            prev_robot = joint_robot[:, i]
            prev_mask = joint_mask[:, i]
            post, prior = self.rssm.obs_step(prev, prev_action, prev_robot, prev_mask, embed[:, i])
            reset = batch["is_first"][:, i].reshape(b, 1)
            if reset.any():
                init = self.rssm.initial(b, joint_ego.device)
                post = RSSMState(
                    deter=post.deter * (1.0 - reset) + init.deter * reset,
                    stoch=post.stoch * (1.0 - reset.reshape(b, 1, 1)) + init.stoch * reset.reshape(b, 1, 1),
                    logits=post.logits * (1.0 - reset.reshape(b, 1, 1)) + init.logits * reset.reshape(b, 1, 1),
                )
            posts.append(post)
            priors.append(prior)
            prev = post
        return posts, priors

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], list[RSSMState]]:
        posts, priors = self.observe(batch)
        feats = torch.stack([self.rssm.get_feat(state) for state in posts], dim=1)
        b, t = feats.shape[:2]
        flat = feats.reshape(b * t, -1)

        reward_pred = self.reward_head(flat).reshape(b, t)
        continue_logit = self.continue_head(flat).reshape(b, t)
        reward_loss = self.reward_head.loss(flat, batch["reward"].reshape(b * t))
        continue_loss = F.binary_cross_entropy_with_logits(continue_logit, batch["continue"])

        ego_pred = self.ego_decoder(flat).reshape(b, t, self.cfg.max_robots, *self.cfg.ego_shape)
        global_pred = self.global_decoder(flat).reshape(b, t, *self.cfg.global_shape)
        ego_loss = F.mse_loss(torch.sigmoid(ego_pred), batch["joint_ego"])
        global_loss = F.mse_loss(torch.sigmoid(global_pred), batch["global_state"])

        post_logits = torch.stack([state.logits for state in posts], dim=1)
        prior_logits = torch.stack([state.logits for state in priors], dim=1)
        post_dist = self.rssm.dist(post_logits)
        prior_dist = self.rssm.dist(prior_logits)
        dyn_kl = torch.distributions.kl_divergence(
            self.rssm.dist(post_logits.detach()),
            prior_dist,
        ).sum(dim=-1).mean()
        rep_kl = torch.distributions.kl_divergence(
            post_dist,
            self.rssm.dist(prior_logits.detach()),
        ).sum(dim=-1).mean()
        dyn_kl = torch.clamp(dyn_kl, min=self.cfg.free_nats)
        rep_kl = torch.clamp(rep_kl, min=self.cfg.free_nats)
        prediction_loss = (
            self.cfg.reward_loss_scale * reward_loss
            + self.cfg.continue_loss_scale * continue_loss
            + self.cfg.recon_loss_scale * (ego_loss + global_loss)
        )
        dynamics_loss = dyn_kl
        representation_loss = rep_kl
        total = (
            self.cfg.prediction_loss_scale * prediction_loss
            + self.cfg.dynamics_loss_scale * dynamics_loss
            + self.cfg.representation_loss_scale * representation_loss
        )
        metrics = {
            "model_loss": float(total.detach().cpu()),
            "prediction_loss": float(prediction_loss.detach().cpu()),
            "dynamics_loss": float(dynamics_loss.detach().cpu()),
            "representation_loss": float(representation_loss.detach().cpu()),
            "reward_loss": float(reward_loss.detach().cpu()),
            "continue_loss": float(continue_loss.detach().cpu()),
            "recon_loss": float((ego_loss + global_loss).detach().cpu()),
            "dyn_kl": float(dyn_kl.detach().cpu()),
            "rep_kl": float(rep_kl.detach().cpu()),
        }
        return total, metrics, posts


class Actor(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        feat_size = cfg.deter_size + cfg.stoch_size * cfg.discrete_size
        self.net = nn.Sequential(
            nn.Linear(feat_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
        )
        self.mean = nn.Linear(cfg.hidden_size, cfg.max_robots * cfg.action_dim)
        self.std = nn.Linear(cfg.hidden_size, cfg.max_robots * cfg.action_dim)

    def forward(self, feat: torch.Tensor) -> Normal:
        h = self.net(feat)
        mean = self.mean(h).reshape(*feat.shape[:-1], self.cfg.max_robots, self.cfg.action_dim)
        std = F.softplus(self.std(h)).reshape_as(mean)
        std = torch.clamp(std, self.cfg.actor_std_min, self.cfg.actor_std_max)
        return Normal(mean, std)

    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(feat)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        raw = torch.tanh(pre_tanh)
        env_action = action_to_env(raw, self.cfg)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - raw.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=(-1, -2))
        return env_action, log_prob


class Value(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        feat_size = cfg.deter_size + cfg.stoch_size * cfg.discrete_size
        self.head = TwoHotSymlogHead(
            feat_size,
            cfg.hidden_size,
            cfg.twohot_bins,
            cfg.twohot_min,
            cfg.twohot_max,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head.loss(feat, target)
