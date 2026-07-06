from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import os
from typing import Any, Optional

import numpy as np
import torch

from .config import DreamerConfig
from .networks import Actor, RSSMState, Value, WorldModel, action_to_env
from .optim import LaProp
from .replay import DreamerSequenceReplay


class DreamerAgent:
    """DreamerV3-style agent for the existing ADDS multi-robot interface."""

    def __init__(self, cfg: DreamerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.world_model = WorldModel(cfg).to(self.device)
        self.actor = Actor(cfg).to(self.device)
        self.value = Value(cfg).to(self.device)
        self.target_value = Value(cfg).to(self.device)
        self.use_amp = bool(cfg.use_amp and self.device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.target_value.load_state_dict(self.value.state_dict())
        self._set_requires_grad(self.target_value, False)
        self.return_low = torch.tensor(0.0, device=self.device)
        self.return_high = torch.tensor(1.0, device=self.device)
        self.return_norm_initialized = False
        self.model_opt = LaProp(
            self.world_model.parameters(),
            lr=cfg.model_lr,
            eps=1e-20,
            warmup_steps=cfg.lr_warmup_steps,
        )
        self.actor_opt = LaProp(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            eps=1e-20,
            warmup_steps=cfg.lr_warmup_steps,
        )
        self.value_opt = LaProp(
            self.value.parameters(),
            lr=cfg.value_lr,
            eps=1e-20,
            warmup_steps=cfg.lr_warmup_steps,
        )
        self.replay = DreamerSequenceReplay(
            capacity=cfg.replay_capacity,
            sequence_length=cfg.sequence_length,
            device=self.device,
            context_length=cfg.replay_context,
        )
        self.metrics: dict[str, float] = {}
        self.epsilon = 0.0
        self._policy_state: RSSMState | None = None #RSSM latent state
        self._policy_prev_action: torch.Tensor | None = None
        self._policy_prev_robot: torch.Tensor | None = None
        self._policy_prev_mask: torch.Tensor | None = None

    @property
    def policy(self):
        return self.actor

    def select_action(
        self,
        ego_state_np: np.ndarray,
        global_state_np: np.ndarray,
        robot_state_np: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ):
        joint_ego = np.zeros((self.cfg.max_robots, *self.cfg.ego_shape), dtype=np.float32)
        joint_robot = np.zeros((self.cfg.max_robots, self.cfg.robot_dim), dtype=np.float32)
        joint_mask = np.zeros((self.cfg.max_robots,), dtype=np.float32)
        joint_ego[0] = ego_state_np.astype(np.float32)
        if robot_state_np is not None:
            joint_robot[0] = robot_state_np.astype(np.float32)
        joint_mask[0] = 1.0
        action = self.select_joint_action(
            joint_ego,
            global_state_np.astype(np.float32),
            joint_robot,
            joint_mask,
            deterministic=deterministic,
        )
        return action[0].astype(np.float32), False

    def select_joint_action(
        self,
        joint_ego_np: np.ndarray,
        global_state_np: np.ndarray,
        joint_robot_np: np.ndarray,
        joint_mask_np: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        self.world_model.eval()
        self.actor.eval() # Inference mode -> gradient 계산 없이 action만 뽑기
        with torch.no_grad():
            batch = {
                "joint_ego": torch.from_numpy(joint_ego_np[None].astype(np.float32)).to(self.device),
                "global_state": torch.from_numpy(global_state_np[None].astype(np.float32)).to(self.device),
                "joint_robot": torch.from_numpy(joint_robot_np[None].astype(np.float32)).to(self.device),
                "joint_mask": torch.from_numpy(joint_mask_np[None].astype(np.float32)).to(self.device),
            }
            embed = self.world_model.encoder(
                batch["joint_ego"],
                batch["global_state"],
                batch["joint_robot"],
                batch["joint_mask"],
            ) #관측을 encoder에 통과시켜 embedding vector 만듦
            if self._policy_state is None:
                self._policy_state = self.world_model.rssm.initial(1, self.device)
            if self._policy_prev_action is None:
                self._policy_prev_action = torch.zeros(
                    1,
                    self.cfg.max_robots,
                    self.cfg.action_dim,
                    device=self.device,
                )
            if self._policy_prev_robot is None:
                self._policy_prev_robot = torch.zeros_like(batch["joint_robot"])
            if self._policy_prev_mask is None:
                self._policy_prev_mask = torch.zeros_like(batch["joint_mask"])
            post, _ = self.world_model.rssm.obs_step(
                self._policy_state, # 이전 latent state
                self._policy_prev_action, # 이전 action
                self._policy_prev_robot, # 이전 robot state
                self._policy_prev_mask,
                embed, #현재 관측 embed와, 이전 latent, action / robot / mask를 이용해 posterior latent, 'post' 만듦
            )
            self._policy_state = RSSMState(
                deter=post.deter.detach(),
                stoch=post.stoch.detach(),
                logits=post.logits.detach(),
            )
            feat = self.world_model.rssm.get_feat(post) # embedding
            action, _ = self.actor.sample(feat, deterministic=deterministic) # posterior latent에서 action 샘플
            mask = batch["joint_mask"].unsqueeze(-1)
            action = action * mask
            self._policy_prev_action = action.detach()
            self._policy_prev_robot = batch["joint_robot"].detach()
            self._policy_prev_mask = batch["joint_mask"].detach()
        self.actor.train()
        self.world_model.train() # train 모드로 변환
        return action[0].detach().cpu().numpy().astype(np.float32)

    def reset_policy_state(self) -> None:
        self._policy_state = None
        self._policy_prev_action = None
        self._policy_prev_robot = None
        self._policy_prev_mask = None

    def set_policy_prev_action(self, joint_action_np: np.ndarray) -> None:
        action = torch.from_numpy(joint_action_np[None].astype(np.float32)).to(self.device)
        self._policy_prev_action = action.detach()

    def add_transition(self, msg: Any) -> None:
        self.replay.add_transition_msg(msg)

    def update(self):
        if not self.replay.can_sample(self.cfg.batch_size):
            return None
        batch = self.replay.sample(self.cfg.batch_size)

        with self._amp_context():
            model_loss, model_metrics, posts = self.world_model.loss(batch)
        self.model_opt.zero_grad(set_to_none=True)
        self.scaler.scale(model_loss).backward()
        self.scaler.unscale_(self.model_opt)
        model_grad_norm = self._adaptive_grad_clip(self.world_model.parameters())
        self.scaler.step(self.model_opt)
        self.scaler.update()

        with torch.no_grad():
            start, start_index, start_batch_index = self._sample_start_state(posts, batch)
            replay_posts = self._detach_posts(posts)
        del posts, model_loss
        self._set_requires_grad(self.world_model, False)
        try:
            with self._amp_context():
                actor_loss, actor_metrics, imagined_feats, returns, imag_weights = self._actor_loss(
                    start,
                    batch,
                    start_index,
                    start_batch_index,
                ) # 선택된 posterior state들에서 상상 rollout하고 actor 업데이트
            self.actor_opt.zero_grad(set_to_none=True)
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            actor_grad_norm = self._adaptive_grad_clip(self.actor.parameters())
            self.scaler.step(self.actor_opt)
            self.scaler.update()

            value_feats = imagined_feats.detach()
            value_returns = returns.detach()
            value_weights = imag_weights.detach()
            bootstrap = returns[0].detach()
            del imagined_feats, returns, imag_weights, actor_loss

            with self._amp_context():
                imag_value_loss = self._value_loss(
                    value_feats,
                    value_returns,
                    value_weights,
                )
                replay_value_loss, replay_value_metrics = self._replay_value_loss(
                    replay_posts,
                    batch,
                    start_index,
                    start_batch_index,
                    bootstrap,
                )
                value_loss = (
                    self.cfg.value_loss_scale * imag_value_loss
                    + self.cfg.replay_value_loss_scale * replay_value_loss
                )
            self.value_opt.zero_grad(set_to_none=True)
            self.scaler.scale(value_loss).backward()
            self.scaler.unscale_(self.value_opt)
            value_grad_norm = self._adaptive_grad_clip(self.value.parameters())
            self.scaler.step(self.value_opt) # value를 imagined return과 replay posterior value target 쪽으로 업데이트
            self.scaler.update()
            self._update_target_value()
        finally:
            self._set_requires_grad(self.world_model, True)

        self.metrics = {
            **model_metrics,
            **actor_metrics,
            "model_grad_norm": float(model_grad_norm.detach().cpu()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "imag_value_loss": float(imag_value_loss.detach().cpu()),
            "replay_value_loss": float(replay_value_loss.detach().cpu()),
            "value_grad_norm": float(value_grad_norm.detach().cpu()),
            "replay_steps": float(len(self.replay)),
            "replay_episodes": float(self.replay.num_episodes),
            **replay_value_metrics,
        }
        return self.metrics

    def _amp_context(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)

    def save(self, filepath: str) -> None:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "cfg": asdict(self.cfg),
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
                "target_value": self.target_value.state_dict(),
                "return_low": self.return_low.detach().cpu(),
                "return_high": self.return_high.detach().cpu(),
                "return_norm_initialized": self.return_norm_initialized,
                "model_opt": self.model_opt.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "value_opt": self.value_opt.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            print(f"[DreamerAgent] checkpoint not found: {filepath}")
            return
        try:
            ckpt = torch.load(filepath, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(filepath, map_location=self.device)
        try:
            self.world_model.load_state_dict(ckpt["world_model"])
            self.actor.load_state_dict(ckpt["actor"])
            self.value.load_state_dict(ckpt["value"])
            self.target_value.load_state_dict(ckpt.get("target_value", ckpt["value"]))
        except RuntimeError as exc:
            print(f"[DreamerAgent] incompatible checkpoint architecture, starting fresh: {exc}")
            return
        self.return_low = ckpt.get("return_low", self.return_low.detach().cpu()).to(self.device)
        self.return_high = ckpt.get("return_high", self.return_high.detach().cpu()).to(self.device)
        self.return_norm_initialized = bool(ckpt.get("return_norm_initialized", True))
        if "model_opt" in ckpt:
            self.model_opt.load_state_dict(ckpt["model_opt"])
            self.actor_opt.load_state_dict(ckpt["actor_opt"])
            self.value_opt.load_state_dict(ckpt["value_opt"])
        self.reset_policy_state()
        print(f"[DreamerAgent] loaded checkpoint: {filepath}")

    def save_model(self, filepath: str) -> None:
        self.save(filepath)

    def load_model(self, filepath: str) -> None:
        self.load(filepath)

    def get_worker_state(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            "world_model": self._state_dict_to_numpy(self.world_model.state_dict()),
            "actor": self._state_dict_to_numpy(self.actor.state_dict()),
        }

    def load_worker_state(self, state: dict[str, dict[str, np.ndarray]]) -> None:
        self.world_model.load_state_dict(self._numpy_to_state_dict(state["world_model"]))
        self.actor.load_state_dict(self._numpy_to_state_dict(state["actor"]))
        self.reset_policy_state()

    def _state_dict_to_numpy(self, state_dict):
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in state_dict.items()
        }

    def _numpy_to_state_dict(self, payload):
        return {
            key: torch.from_numpy(value).to(self.device)
            for key, value in payload.items()
        }

    def save_replay_buffer(self, filepath: str) -> None:
        self.replay.save(filepath)

    def load_replay_buffer(self, filepath: str) -> None:
        if os.path.exists(filepath):
            loaded = self.replay.load(filepath)
            if loaded:
                print(
                    f"[DreamerAgent] loaded replay buffer: {filepath} "
                    f"episodes={self.replay.num_episodes}, steps={len(self.replay)}"
                )
            else:
                print("[DreamerAgent] starting with an empty replay buffer.")

    def _sample_start_state(
        self,
        posts: list[RSSMState],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        stacked_deter = torch.stack([state.deter for state in posts], dim=1)
        stacked_stoch = torch.stack([state.stoch for state in posts], dim=1)
        stacked_logits = torch.stack([state.logits for state in posts], dim=1)
        b, t = stacked_deter.shape[:2]
        imag_last = int(getattr(self.cfg, "imag_last", -1))
        if imag_last < 0 and batch is not None and "loss_mask" in batch:
            weights = batch["loss_mask"].float()
            row_sum = weights.sum(dim=1, keepdim=True)
            weights = torch.where(row_sum > 0, weights, torch.ones_like(weights))
            index = torch.multinomial(weights, 1).squeeze(1)
            batch_index = torch.arange(b, device=stacked_deter.device)
        elif imag_last < 0:
            index = torch.randint(0, t, (b,), device=stacked_deter.device)
            batch_index = torch.arange(b, device=stacked_deter.device)
        else:
            k = min(imag_last or t, t)
            index_grid = torch.arange(t - k, t, device=stacked_deter.device)
            index_grid = index_grid.unsqueeze(0).expand(b, k)
            batch_grid = torch.arange(b, device=stacked_deter.device).unsqueeze(1).expand(b, k)
            index = index_grid.reshape(-1)
            batch_index = batch_grid.reshape(-1)
            if batch is not None and "loss_mask" in batch:
                valid = batch["loss_mask"][batch_index, index] > 0
                if valid.any():
                    index = index[valid]
                    batch_index = batch_index[valid]
        state = RSSMState(
            deter=stacked_deter[batch_index, index],
            stoch=stacked_stoch[batch_index, index],
            logits=stacked_logits[batch_index, index],
        )
        return state, index, batch_index

    def _detach_posts(self, posts: list[RSSMState]) -> list[RSSMState]:
        return [
            RSSMState(
                deter=state.deter.detach(),
                stoch=state.stoch.detach(),
                logits=state.logits.detach(),
            )
            for state in posts
        ]

    def _set_requires_grad(self, module: torch.nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _update_target_value(self) -> None:
        tau = self.cfg.target_value_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def _adaptive_grad_clip(self, parameters) -> torch.Tensor:
        params = [param for param in parameters if param.grad is not None]
        if not params:
            return torch.tensor(0.0, device=self.device)
        grad_norms = [param.grad.detach().norm(2) for param in params]
        global_norm = torch.norm(torch.stack(grad_norms), 2)
        clip = float(self.cfg.agc_clip)
        eps = float(self.cfg.agc_eps)
        with torch.no_grad():
            for param in params:
                if param.ndim < 2:
                    continue
                param_norm = param.detach().norm(2).clamp_min(eps)
                grad_norm = param.grad.detach().norm(2)
                max_norm = clip * param_norm
                if grad_norm > max_norm:
                    param.grad.mul_(max_norm / grad_norm.clamp_min(1e-6))
        return global_norm

    def _normalize_returns(self, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            flat = returns.detach().reshape(-1)
            batch_low = torch.quantile(flat, self.cfg.return_norm_low)
            batch_high = torch.quantile(flat, self.cfg.return_norm_high)
            if not self.return_norm_initialized:
                self.return_low.copy_(batch_low)
                self.return_high.copy_(batch_high)
                self.return_norm_initialized = True
            else:
                decay = self.cfg.return_norm_decay
                self.return_low.mul_(decay).add_(batch_low, alpha=1.0 - decay)
                self.return_high.mul_(decay).add_(batch_high, alpha=1.0 - decay)

            center = self.return_low
            scale = (self.return_high - self.return_low).clamp_min(self.cfg.return_norm_min_scale)

        normalized = (returns - center) / scale
        metrics = {
            "return_norm_low": float(self.return_low.detach().cpu()),
            "return_norm_high": float(self.return_high.detach().cpu()),
            "return_norm_scale": float(scale.detach().cpu()),
            "return_batch_low": float(batch_low.detach().cpu()),
            "return_batch_high": float(batch_high.detach().cpu()),
            "normed_return_mean": float(normalized.mean().detach().cpu()),
            "normed_return_abs_max": float(normalized.abs().max().detach().cpu()),
        }
        return normalized, scale, metrics

    def _actor_loss(
        self,
        start: RSSMState,
        batch: dict[str, torch.Tensor],
        start_index: torch.Tensor,
        start_batch_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
        state = start
        batch_index = start_batch_index.to(device=start_index.device, dtype=torch.long)
        robot = batch["joint_robot"][batch_index, start_index]
        mask = batch["joint_mask"][batch_index, start_index]
        feats = [self.world_model.rssm.get_feat(state)]
        rewards = []
        continues = []
        entropies = []
        log_probs = []

        for _ in range(self.cfg.horizon): # img step 
            feat = feats[-1]
            action, log_prob = self._sample_reinforce_action(feat.detach())
            action = action * mask.unsqueeze(-1)
            state = self.world_model.rssm.img_step(state, action, robot, mask)
            next_feat = self.world_model.rssm.get_feat(state)
            rewards.append(self.world_model.reward_head(next_feat).squeeze(-1))
            continues.append(torch.sigmoid(self.world_model.continue_head(next_feat).squeeze(-1)))
            entropies.append((-log_prob).clamp(-self.cfg.actor_entropy_clip, self.cfg.actor_entropy_clip))
            log_probs.append(log_prob)
            feats.append(next_feat)

        feat_seq = torch.stack(feats, dim=0)
        actor_feat_seq = feat_seq[:-1]
        reward_seq = torch.stack(rewards, dim=0)
        continue_seq = torch.stack(continues, dim=0)
        entropy_seq = torch.stack(entropies, dim=0)
        log_prob_seq = torch.stack(log_probs, dim=0)
        discount_seq = continue_seq if self.cfg.contdisc else continue_seq * self.cfg.discount
        with torch.no_grad():
            target_value_model = self.target_value if self.cfg.slow_value_target else self.value
            value_seq = target_value_model(feat_seq)
        returns = self._lambda_returns(reward_seq, value_seq, discount_seq)
        normed_returns, return_scale, norm_metrics = self._normalize_returns(returns)
        with torch.no_grad():
            baseline_seq = self.value(actor_feat_seq)
            normed_advantage = (returns - baseline_seq) / return_scale
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount_seq[:1]), discount_seq[:-1]], dim=0),
                dim=0,
            )
            weights = weights * (mask.sum(dim=-1) > 0).float().reshape(1, -1)
        denom = weights.sum().clamp_min(1.0)
        reinforce_loss = -((weights * log_prob_seq * normed_advantage.detach()).sum() / denom)
        entropy_loss = -self.cfg.entropy_scale * ((weights * entropy_seq).sum() / denom)
        actor_loss = reinforce_loss + entropy_loss
        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_return_loss": float(reinforce_loss.detach().cpu()),
            "actor_reinforce_loss": float(reinforce_loss.detach().cpu()),
            "actor_entropy_loss": float(entropy_loss.detach().cpu()),
            "actor_entropy": float(entropy_seq.mean().detach().cpu()),
            "actor_entropy_abs_max": float(entropy_seq.abs().max().detach().cpu()),
            "actor_log_prob": float(log_prob_seq.mean().detach().cpu()),
            "actor_advantage_mean": float((returns - baseline_seq).mean().detach().cpu()),
            "actor_normed_advantage_mean": float(normed_advantage.mean().detach().cpu()),
            "actor_normed_advantage_abs_max": float(normed_advantage.abs().max().detach().cpu()),
            "imag_reward": float(reward_seq.mean().detach().cpu()),
            "imag_continue": float(continue_seq.mean().detach().cpu()),
            "imag_discount": float(discount_seq.mean().detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
            "return_abs_max": float(returns.abs().max().detach().cpu()),
            "target_value_mean": float(value_seq.mean().detach().cpu()),
            "target_value_abs_max": float(value_seq.abs().max().detach().cpu()),
            "imag_start_count": float(start_index.numel()),
            "imag_starts_per_sequence": float(start_index.numel() / max(1, batch["reward"].shape[0])),
            "imag_start_index_mean": float(start_index.float().mean().detach().cpu()),
            **norm_metrics,
        }
        return actor_loss, metrics, actor_feat_seq, returns, weights

    def _sample_reinforce_action(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor.forward(feat)
        pre_tanh = dist.rsample()
        raw = torch.tanh(pre_tanh)
        env_action = action_to_env(raw, self.cfg)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - raw.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=(-1, -2))
        return env_action, log_prob

    def _value_loss(
        self,
        feat_seq: torch.Tensor,
        returns: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        value_loss = self.value.weighted_loss(feat_seq, returns, weights)
        if self.cfg.value_slowreg_scale <= 0.0:
            return value_loss
        with torch.no_grad():
            slow_target = self.target_value(feat_seq)
        slowreg_loss = self.value.weighted_loss(feat_seq, slow_target.detach(), weights)
        return value_loss + self.cfg.value_slowreg_scale * slowreg_loss

    def _replay_value_loss(
        self,
        posts: list[RSSMState],
        batch: dict[str, torch.Tensor],
        start_index: torch.Tensor,
        start_batch_index: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            feat_seq = torch.stack([self.world_model.rssm.get_feat(state) for state in posts], dim=0)
            batch_index = start_batch_index.to(device=start_index.device, dtype=torch.long)
            replay_feats_list = []
            replay_returns_list = []
            replay_weights_list = []
            disc = self.cfg.discount
            unique_batches = torch.unique(batch_index, sorted=True)
            for batch_id in unique_batches:
                flat_pos = torch.nonzero(batch_index == batch_id, as_tuple=False).squeeze(-1)
                if flat_pos.numel() < 2:
                    continue
                order = torch.argsort(start_index[flat_pos])
                flat_pos = flat_pos[order]
                seq_index = start_index[flat_pos]
                feats = feat_seq[seq_index, batch_id]
                rewards = batch["reward"][batch_id, seq_index]
                terminals = 1.0 - batch["continue"][batch_id, seq_index]
                boots = bootstrap[flat_pos]
                rets = self._replay_lambda_returns(
                    rewards,
                    boots,
                    terminals,
                    disc,
                )
                if rets.numel() == 0:
                    continue
                replay_feats_list.append(feats[:-1])
                replay_returns_list.append(rets)
                if "loss_mask" in batch:
                    replay_weights_list.append(batch["loss_mask"][batch_id, seq_index[:-1]].float())
                else:
                    replay_weights_list.append(torch.ones_like(rets))
            if replay_feats_list:
                replay_feats = torch.cat(replay_feats_list, dim=0)
                replay_returns = torch.cat(replay_returns_list, dim=0)
                replay_weights = torch.cat(replay_weights_list, dim=0)
            else:
                replay_feats = feat_seq[start_index, batch_index]
                replay_returns = bootstrap
                if "loss_mask" in batch:
                    replay_weights = batch["loss_mask"][batch_index, start_index].float()
                else:
                    replay_weights = torch.ones_like(replay_returns)
            metrics = {
                "replay_return_mean": float(replay_returns.mean().detach().cpu()),
                "replay_return_abs_max": float(replay_returns.abs().max().detach().cpu()),
                "replay_bootstrap_mean": float(bootstrap.mean().detach().cpu()),
                "replay_start_count": float(start_index.numel()),
                "replay_start_index_mean": float(start_index.float().mean().detach().cpu()),
            }
        replay_value_loss = self._value_loss(
            replay_feats.detach(),
            replay_returns.detach(),
            replay_weights.detach(),
        )
        return replay_value_loss, metrics

    def _lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        discounts: torch.Tensor,
        bootstrap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        next_value = values[-1] if bootstrap is None else bootstrap
        outputs = []
        for t in reversed(range(rewards.shape[0])):
            bootstrap_value = values[t + 1] if t + 1 < values.shape[0] else next_value
            next_value = rewards[t] + discounts[t] * (
                (1.0 - self.cfg.lambda_) * bootstrap_value + self.cfg.lambda_ * next_value
            )
            outputs.append(next_value)
        return torch.stack(list(reversed(outputs)), dim=0)

    def _replay_lambda_returns(
        self,
        rewards: torch.Tensor,
        bootstraps: torch.Tensor,
        terminals: torch.Tensor,
        discount: float,
    ) -> torch.Tensor:
        if rewards.shape[0] < 2:
            return rewards.new_zeros((0,))
        next_return = bootstraps[-1]
        outputs = []
        live = (1.0 - terminals[1:]) * discount
        cont = torch.full_like(live, self.cfg.lambda_)
        interm = rewards[1:] + (1.0 - cont) * live * bootstraps[1:]
        for t in reversed(range(live.shape[0])):
            next_return = interm[t] + live[t] * cont[t] * next_return
            outputs.append(next_return)
        return torch.stack(list(reversed(outputs)), dim=0)
