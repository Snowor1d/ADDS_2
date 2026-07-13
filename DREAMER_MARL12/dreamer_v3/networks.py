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


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return x * scale * self.weight


class GroupedLinear(nn.Module):
    def __init__(self, groups: int, in_per_group: int, out_per_group: int) -> None:
        super().__init__()
        self.groups = int(groups)
        self.in_per_group = int(in_per_group)
        self.out_per_group = int(out_per_group)
        scale = 1.0 / math.sqrt(max(1, self.in_per_group))
        self.weight = nn.Parameter(torch.empty(self.groups, self.in_per_group, self.out_per_group))
        nn.init.trunc_normal_(self.weight, std=scale, a=-2.0 * scale, b=2.0 * scale)
        self.bias = nn.Parameter(torch.zeros(self.groups, self.out_per_group))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bgi,gio->bgo", x, self.weight) + self.bias


def linear_block(in_dim: int, out_dim: int) -> list[nn.Module]:
    linear = nn.Linear(in_dim, out_dim)
    scale = 1.0 / math.sqrt(max(1, in_dim))
    nn.init.trunc_normal_(linear.weight, std=scale, a=-2.0 * scale, b=2.0 * scale)
    nn.init.zeros_(linear.bias)
    return [linear, RMSNorm(out_dim), nn.SiLU()]


def mlp_blocks(in_dim: int, hidden_dim: int, layers: int) -> list[nn.Module]:
    layers = max(1, int(layers))
    modules: list[nn.Module] = []
    for i in range(layers):
        modules.extend(linear_block(in_dim if i == 0 else hidden_dim, hidden_dim))
    return modules


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        ChannelRMSNorm(out_channels),
        nn.SiLU(),
    ]


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


def symlog_vector_obs(vector_obs: torch.Tensor) -> torch.Tensor:
    return symlog(torch.nan_to_num(vector_obs.float(), nan=0.0, posinf=0.0, neginf=0.0))


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_size: int,
        input_shape: tuple[int, int],
        depth: int,
    ) -> None:
        super().__init__()
        depth = max(1, int(depth))
        self.conv = nn.Sequential(
            *conv_block(in_channels, depth, 4, stride=2, padding=1),
            *conv_block(depth, depth * 2, 4, stride=2, padding=1),
            *conv_block(depth * 2, depth * 4, 4, stride=2, padding=1),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_shape)
            conv_out = self.conv(dummy)
            flat_dim = int(conv_out.numel())
        self.proj = nn.Sequential(
            nn.Flatten(),
            *linear_block(flat_dim, out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class ConvDecoder(nn.Module):
    """DreamerV3-style image decoder with block-spatial projection."""

    def __init__(self, cfg: DreamerConfig, out_shape: tuple[int, int, int]) -> None:
        super().__init__()
        self.out_shape = tuple(int(x) for x in out_shape)
        out_channels, height, width = self.out_shape
        depth = max(1, int(cfg.conv_depth))
        self.deter_size = int(cfg.deter_size)
        self.stoch_size = int(cfg.stoch_size * cfg.discrete_size)
        self.bspace = int(cfg.decoder_bspace)
        self.depths = (depth * 2, depth * 3, depth * 4, depth * 4)
        self.start_h = max(1, int(math.ceil(height / 16)))
        self.start_w = max(1, int(math.ceil(width / 16)))
        spatial_size = self.start_h * self.start_w * self.depths[-1]
        if self.bspace <= 0:
            raise ValueError("decoder_bspace must be positive")
        if self.deter_size % self.bspace != 0 or spatial_size % self.bspace != 0:
            raise ValueError("deter_size and decoder spatial size must be divisible by decoder_bspace")
        self.deter_to_space = GroupedLinear(self.bspace, self.deter_size // self.bspace, spatial_size // self.bspace)
        self.stoch_hidden = nn.Sequential(*linear_block(self.stoch_size, cfg.hidden_size * 2))
        self.stoch_to_space = nn.Linear(cfg.hidden_size * 2, spatial_size)
        scale = 1.0 / math.sqrt(max(1, cfg.hidden_size * 2))
        nn.init.trunc_normal_(self.stoch_to_space.weight, std=scale, a=-2.0 * scale, b=2.0 * scale)
        nn.init.zeros_(self.stoch_to_space.bias)
        self.space_norm = ChannelRMSNorm(self.depths[-1])
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            *conv_block(self.depths[3], self.depths[2], 5, padding=2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            *conv_block(self.depths[2], self.depths[1], 5, padding=2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            *conv_block(self.depths[1], self.depths[0], 5, padding=2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(self.depths[0], out_channels, 5, padding=2),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        leading = feat.shape[:-1]
        flat = feat.reshape(-1, feat.shape[-1])
        deter = flat[:, :self.deter_size].reshape(-1, self.bspace, self.deter_size // self.bspace)
        stoch = flat[:, self.deter_size:self.deter_size + self.stoch_size]
        deter_space = self.deter_to_space(deter).reshape(-1, self.depths[-1], self.start_h, self.start_w)
        stoch_space = self.stoch_to_space(self.stoch_hidden(stoch)).reshape(-1, self.depths[-1], self.start_h, self.start_w)
        x = F.silu(self.space_norm(deter_space + stoch_space))
        x = self.net(x)
        _, height, width = self.out_shape
        if x.shape[-2:] != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x.reshape(*leading, *self.out_shape)


class VectorDecoder(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, out_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *linear_block(in_size, hidden_size),
            *linear_block(hidden_size, hidden_size),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


class JointObsEncoder(nn.Module):
    def __init__(self, cfg: DreamerConfig) -> None:
        super().__init__()
        c_ego, ego_h, ego_w = cfg.ego_shape
        c_global, global_h, global_w = cfg.global_shape
        self.cfg = cfg
        self.ego_encoder = ConvEncoder(c_ego, cfg.hidden_size, (ego_h, ego_w), cfg.conv_depth)
        self.global_encoder = ConvEncoder(c_global, cfg.hidden_size, (global_h, global_w), cfg.conv_depth)
        self.robot_encoder = nn.Sequential(
            *linear_block(cfg.robot_dim, 64),
            *linear_block(64, 64),
        )
        self.out = nn.Sequential(
            *linear_block(cfg.hidden_size * 2 + 64, cfg.embed_size),
            *linear_block(cfg.embed_size, cfg.embed_size),
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
        robot_feat = self.robot_encoder(symlog_vector_obs(joint_robot))
        mask = joint_mask.unsqueeze(-1)
        denom = mask.sum(dim=-2).clamp_min(1.0)
        ego_pooled = (ego_feat * mask).sum(dim=-2) / denom
        robot_pooled = (robot_feat * mask).sum(dim=-2) / denom

        global_feat = self.global_encoder(global_state.reshape(-1, *global_state.shape[-3:]))
        global_feat = global_feat.reshape(*leading, -1)
        return self.out(torch.cat([ego_pooled, global_feat, robot_pooled], dim=-1))

class TwoHotSymlogHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bins: int,
        low: float,
        high: float,
        layers: int = 1,
    ) -> None:
        super().__init__()
        out = nn.Linear(hidden_dim, bins)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(
            *mlp_blocks(in_dim, hidden_dim, layers),
            out,
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

    def loss(self, feat: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        loss = self.loss_per(feat, target)
        if reduction == "none":
            return loss
        if reduction != "mean":
            raise ValueError(f"Unsupported reduction: {reduction}")
        return loss.mean()

    def loss_per(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.logits(feat)
        target_dist = self.twohot(target).to(logits.device)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1)

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
        self.blocks = int(cfg.rssm_blocks)
        if cfg.deter_size % self.blocks != 0:
            raise ValueError("deter_size must be divisible by rssm_blocks")
        self.deter_per_block = cfg.deter_size // self.blocks
        joint_action_dim = cfg.max_robots * cfg.action_dim
        joint_robot_dim = cfg.max_robots * cfg.robot_dim
        action_input_dim = joint_action_dim + joint_robot_dim + cfg.max_robots
        self.deter_encoder = nn.Sequential(*linear_block(cfg.deter_size, cfg.hidden_size))
        self.stoch_encoder = nn.Sequential(*linear_block(self.stoch_flat, cfg.hidden_size))
        self.action_encoder = nn.Sequential(
            *linear_block(action_input_dim, cfg.hidden_size),
        )
        self.core_hidden = GroupedLinear(
            self.blocks,
            self.deter_per_block + cfg.hidden_size * 3,
            self.deter_per_block,
        )
        self.core_hidden_norm = RMSNorm(cfg.deter_size)
        self.core_gate = GroupedLinear(
            self.blocks,
            self.deter_per_block,
            self.deter_per_block * 3,
        )
        self.prior = nn.Sequential(
            *linear_block(cfg.deter_size, cfg.hidden_size),
            *linear_block(cfg.hidden_size, cfg.hidden_size),
            nn.Linear(cfg.hidden_size, self.stoch_flat),
        )
        self.posterior = nn.Sequential(
            *linear_block(cfg.deter_size + cfg.embed_size, cfg.hidden_size),
            nn.Linear(cfg.hidden_size, self.stoch_flat),
        )

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.cfg.deter_size, device=device)
        logits = torch.zeros(batch_size, self.cfg.stoch_size, self.cfg.discrete_size, device=device)
        stoch = torch.zeros(batch_size, self.cfg.stoch_size, self.cfg.discrete_size, device=device)
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
        deter = self._core(prev, action, robot, mask)
        logits = self.prior(deter).reshape(-1, self.cfg.stoch_size, self.cfg.discrete_size)
        stoch = self._sample(logits)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def _core(
        self,
        prev: RSSMState,
        action: torch.Tensor,
        robot: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = prev.deter.shape[0]
        action_flat = action.reshape(batch_size, -1)
        action_flat = action_flat / action_flat.detach().abs().clamp_min(1.0)
        robot_flat = symlog_vector_obs(robot).reshape(batch_size, -1)
        action_input = torch.cat([action_flat, robot_flat, mask], dim=-1)
        deter_h = self.deter_encoder(prev.deter)
        stoch_h = self.stoch_encoder(prev.stoch.reshape(batch_size, -1))
        action_h = self.action_encoder(action_input)
        dyn_h = torch.cat([deter_h, stoch_h, action_h], dim=-1)

        deter_grouped = prev.deter.reshape(batch_size, self.blocks, self.deter_per_block)
        dyn_grouped = dyn_h.unsqueeze(1).expand(-1, self.blocks, -1)
        x = torch.cat([deter_grouped, dyn_grouped], dim=-1)
        x = self.core_hidden(x).reshape(batch_size, self.cfg.deter_size)
        x = F.silu(self.core_hidden_norm(x))
        gates = self.core_gate(x.reshape(batch_size, self.blocks, self.deter_per_block))
        reset, cand, update = gates.reshape(batch_size, self.cfg.deter_size * 3).chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * prev.deter

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
            cfg.reward_layers,
        )
        continue_out = nn.Linear(cfg.hidden_size, 1)
        scale = 1.0 / math.sqrt(max(1, cfg.hidden_size))
        nn.init.trunc_normal_(continue_out.weight, std=scale, a=-2.0 * scale, b=2.0 * scale)
        nn.init.zeros_(continue_out.bias)
        self.continue_head = nn.Sequential(
            *mlp_blocks(feat_size, cfg.hidden_size, cfg.continue_layers),
            continue_out,
        )
        self.ego_decoder = ConvDecoder(
            cfg,
            (cfg.max_robots * cfg.ego_shape[0], cfg.ego_shape[1], cfg.ego_shape[2]),
        )
        self.global_decoder = ConvDecoder(
            cfg,
            cfg.global_shape,
        )
        self.robot_decoder = VectorDecoder(
            feat_size,
            cfg.hidden_size,
            cfg.max_robots * cfg.robot_dim,
        )

    def observe(self, batch: dict[str, torch.Tensor]) -> tuple[list[RSSMState], list[RSSMState]]:
        
        # 샘플된 실제 replay sequence를 바탕으로 RSSM/world model을 학습함

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
        zero_robot = torch.zeros_like(joint_robot[:, 0])
        zero_mask = torch.zeros_like(joint_mask[:, 0])
        for i in range(t): # t : 샘플된 실제 관측 sequence 길이, sequence legnth + replay context
            prev_action = zero_action if i == 0 else action[:, i - 1]
            prev_robot = zero_robot if i == 0 else joint_robot[:, i - 1]
            prev_mask = zero_mask if i == 0 else joint_mask[:, i - 1]
            reset = batch["is_first"][:, i].reshape(b, 1)
            if reset.any():
                init = self.rssm.initial(b, joint_ego.device)
                reset_stoch = reset.reshape(b, 1, 1)
                reset_robot = reset.reshape(b, 1, 1)
                prev = RSSMState(
                    deter=prev.deter * (1.0 - reset) + init.deter * reset,
                    stoch=prev.stoch * (1.0 - reset_stoch) + init.stoch * reset_stoch,
                    logits=prev.logits * (1.0 - reset_stoch) + init.logits * reset_stoch,
                )
                prev_action = prev_action * (1.0 - reset.reshape(b, 1, 1))
                prev_robot = prev_robot * (1.0 - reset_robot)
                prev_mask = prev_mask * (1.0 - reset)
            post, prior = self.rssm.obs_step(prev, prev_action, prev_robot, prev_mask, embed[:, i])
            posts.append(post) # posterior는 (prev latent + action{i-1} + robot{i-1} + mask{i-1} + embed{i})를 바탕으로 만들어짐
            priors.append(prior)
            prev = post
        return posts, priors

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], list[RSSMState]]:
        posts, priors = self.observe(batch)
        feats = torch.stack([self.rssm.get_feat(state) for state in posts], dim=1)
        b, t = feats.shape[:2]
        flat = feats.reshape(b * t, -1)
        loss_mask = batch.get("loss_mask", torch.ones(b, t, device=feats.device))
        loss_denom = loss_mask.sum().clamp_min(1.0)
        masked_mean = lambda x: (x * loss_mask).sum() / loss_denom
        flat_mask = loss_mask.reshape(b * t)
        chunk_size = max(1, int(getattr(self.cfg, "decoder_chunk_size", 0) or (b * t)))

        reward_target = batch["reward"].reshape(b * t)
        continue_target = batch["continue"]
        if self.cfg.contdisc:
            continue_target = continue_target * self.cfg.discount
        continue_target = continue_target.reshape(b * t)
        ego_target = batch["joint_ego"].reshape(b * t, self.cfg.max_robots, *self.cfg.ego_shape)
        global_target = batch["global_state"].reshape(b * t, *self.cfg.global_shape)
        robot_target = symlog_vector_obs(batch["joint_robot"]).reshape(
            b * t,
            self.cfg.max_robots,
            self.cfg.robot_dim,
        )
        robot_mask = batch["joint_mask"].reshape(b * t, self.cfg.max_robots).unsqueeze(-1)

        reward_loss_sum = flat.new_zeros(())
        continue_loss_sum = flat.new_zeros(())
        ego_loss_sum = flat.new_zeros(())
        global_loss_sum = flat.new_zeros(())
        robot_loss_sum = flat.new_zeros(())

        for start in range(0, b * t, chunk_size):
            end = min(start + chunk_size, b * t)
            feat_chunk = flat[start:end]
            mask_chunk = flat_mask[start:end]

            reward_loss_per = self.reward_head.loss(
                feat_chunk,
                reward_target[start:end],
                reduction="none",
            )
            reward_loss_sum = reward_loss_sum + (reward_loss_per * mask_chunk).sum()

            continue_logit = self.continue_head(feat_chunk).squeeze(-1)
            continue_loss_per = F.binary_cross_entropy_with_logits(
                continue_logit,
                continue_target[start:end],
                reduction="none",
            )
            continue_loss_sum = continue_loss_sum + (continue_loss_per * mask_chunk).sum()

            ego_pred = self.ego_decoder(feat_chunk).reshape(
                end - start,
                self.cfg.max_robots,
                *self.cfg.ego_shape,
            )
            ego_loss_per_robot = (
                torch.sigmoid(ego_pred) - ego_target[start:end]
            ).pow(2).sum(dim=(2, 3, 4))
            ego_mask_chunk = robot_mask[start:end].squeeze(-1)
            ego_loss_per = (
                ego_loss_per_robot * ego_mask_chunk
            ).sum(dim=1) / ego_mask_chunk.sum(dim=1).clamp_min(1.0)
            ego_loss_sum = ego_loss_sum + (ego_loss_per * mask_chunk).sum()

            global_pred = self.global_decoder(feat_chunk)
            global_loss_per = (
                torch.sigmoid(global_pred) - global_target[start:end]
            ).pow(2).sum(dim=(1, 2, 3))
            global_loss_sum = global_loss_sum + (global_loss_per * mask_chunk).sum()

            robot_pred = self.robot_decoder(feat_chunk).reshape(
                end - start,
                self.cfg.max_robots,
                self.cfg.robot_dim,
            )
            robot_mask_chunk = robot_mask[start:end]
            robot_loss_per = (
                (robot_pred - robot_target[start:end]).pow(2) * robot_mask_chunk
            ).sum(dim=(1, 2))
            robot_denom = (robot_mask_chunk.sum(dim=(1, 2)) * self.cfg.robot_dim).clamp_min(1.0)
            robot_loss_per = robot_loss_per / robot_denom
            robot_loss_sum = robot_loss_sum + (robot_loss_per * mask_chunk).sum()

        reward_loss = reward_loss_sum / loss_denom
        continue_loss = continue_loss_sum / loss_denom
        ego_loss = ego_loss_sum / loss_denom
        global_loss = global_loss_sum / loss_denom
        robot_loss = robot_loss_sum / loss_denom

        post_logits = torch.stack([state.logits for state in posts], dim=1)
        prior_logits = torch.stack([state.logits for state in priors], dim=1)
        post_dist = self.rssm.dist(post_logits)
        prior_dist = self.rssm.dist(prior_logits)
        dyn_kl = torch.distributions.kl_divergence(
            self.rssm.dist(post_logits.detach()),
            prior_dist,
        ).sum(dim=-1)
        rep_kl = torch.distributions.kl_divergence(
            post_dist,
            self.rssm.dist(prior_logits.detach()),
        ).sum(dim=-1)
        dyn_kl = masked_mean(torch.clamp(dyn_kl, min=self.cfg.free_nats))
        rep_kl = masked_mean(torch.clamp(rep_kl, min=self.cfg.free_nats))
        prediction_loss = (
            self.cfg.reward_loss_scale * reward_loss
            + self.cfg.continue_loss_scale * continue_loss
            + self.cfg.recon_loss_scale * (ego_loss + global_loss + robot_loss)
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
            "recon_loss": float((ego_loss + global_loss + robot_loss).detach().cpu()),
            "image_recon_loss": float((ego_loss + global_loss).detach().cpu()),
            "robot_recon_loss": float(robot_loss.detach().cpu()),
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
            *mlp_blocks(feat_size, cfg.hidden_size, cfg.policy_layers),
        )
        self.mean = nn.Linear(cfg.hidden_size, cfg.max_robots * cfg.action_dim)
        self.std = nn.Linear(cfg.hidden_size, cfg.max_robots * cfg.action_dim)
        nn.init.trunc_normal_(self.mean.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.zeros_(self.mean.bias)
        nn.init.trunc_normal_(self.std.weight, std=0.01, a=-0.02, b=0.02)
        nn.init.zeros_(self.std.bias)

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
            cfg.value_layers,
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.head.loss(feat, target)

    def weighted_loss(
        self,
        feat: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.head.loss(feat, target, reduction="none")
        if weights is None:
            return loss.mean()
        weights = weights.to(loss.device)
        return (loss * weights).sum() / weights.sum().clamp_min(1.0)
