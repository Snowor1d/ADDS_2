from __future__ import annotations

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
        )
        self.metrics: dict[str, float] = {}
        self.epsilon = 0.0

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
        self.world_model.eval()
        self.actor.eval()
        with torch.no_grad():
            joint_ego = np.zeros((1, self.cfg.max_robots, *self.cfg.ego_shape), dtype=np.float32)
            joint_robot = np.zeros((1, self.cfg.max_robots, self.cfg.robot_dim), dtype=np.float32)
            joint_mask = np.zeros((1, self.cfg.max_robots), dtype=np.float32)
            joint_ego[0, 0] = ego_state_np.astype(np.float32)
            if robot_state_np is not None:
                joint_robot[0, 0] = robot_state_np.astype(np.float32)
            joint_mask[0, 0] = 1.0
            global_state = global_state_np.astype(np.float32)[None]
            batch = {
                "joint_ego": torch.from_numpy(joint_ego).to(self.device),
                "global_state": torch.from_numpy(global_state).to(self.device),
                "joint_robot": torch.from_numpy(joint_robot).to(self.device),
                "joint_mask": torch.from_numpy(joint_mask).to(self.device),
            }
            embed = self.world_model.encoder(
                batch["joint_ego"],
                batch["global_state"],
                batch["joint_robot"],
                batch["joint_mask"],
            )
            init = self.world_model.rssm.initial(1, self.device)
            zero_action = torch.zeros(1, self.cfg.max_robots, self.cfg.action_dim, device=self.device)
            zero_robot = torch.zeros_like(batch["joint_robot"])
            zero_mask = torch.zeros_like(batch["joint_mask"])
            post, _ = self.world_model.rssm.obs_step(
                init,
                zero_action,
                zero_robot,
                zero_mask,
                embed,
            )
            feat = self.world_model.rssm.get_feat(post)
            action, _ = self.actor.sample(feat, deterministic=deterministic)
        self.actor.train()
        self.world_model.train()
        return action[0, 0].detach().cpu().numpy().astype(np.float32), False

    def select_joint_action(
        self,
        joint_ego_np: np.ndarray,
        global_state_np: np.ndarray,
        joint_robot_np: np.ndarray,
        joint_mask_np: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        self.world_model.eval()
        self.actor.eval()
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
            )
            init = self.world_model.rssm.initial(1, self.device)
            zero_action = torch.zeros(1, self.cfg.max_robots, self.cfg.action_dim, device=self.device)
            zero_robot = torch.zeros_like(batch["joint_robot"])
            zero_mask = torch.zeros_like(batch["joint_mask"])
            post, _ = self.world_model.rssm.obs_step(
                init,
                zero_action,
                zero_robot,
                zero_mask,
                embed,
            )
            feat = self.world_model.rssm.get_feat(post)
            action, _ = self.actor.sample(feat, deterministic=deterministic)
            mask = batch["joint_mask"].unsqueeze(-1)
            action = action * mask
        self.actor.train()
        self.world_model.train()
        return action[0].detach().cpu().numpy().astype(np.float32)

    def add_transition(self, msg: Any) -> None:
        self.replay.add_transition_msg(msg)

    def update(self):
        if not self.replay.can_sample(self.cfg.batch_size):
            return None
        batch = self.replay.sample(self.cfg.batch_size)

        model_loss, model_metrics, posts = self.world_model.loss(batch)
        self.model_opt.zero_grad(set_to_none=True)
        model_loss.backward()
        model_grad_norm = self._adaptive_grad_clip(self.world_model.parameters())
        self.model_opt.step()

        with torch.no_grad():
            start, start_index = self._sample_start_state(posts)
        self._set_requires_grad(self.world_model, False)
        try:
            actor_loss, actor_metrics, imagined_feats, returns = self._actor_loss(
                start,
                batch,
                start_index,
            )
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = self._adaptive_grad_clip(self.actor.parameters())
            self.actor_opt.step()

            imag_value_loss = self._value_loss(imagined_feats.detach(), returns.detach())
            replay_value_loss, replay_value_metrics = self._replay_value_loss(
                posts,
                batch,
                start_index,
                returns[0].detach(),
            )
            value_loss = (
                self.cfg.value_loss_scale * imag_value_loss
                + self.cfg.replay_value_loss_scale * replay_value_loss
            )
            self.value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad_norm = self._adaptive_grad_clip(self.value.parameters())
            self.value_opt.step()
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

    def _sample_start_state(self, posts: list[RSSMState]) -> tuple[RSSMState, torch.Tensor]:
        stacked_deter = torch.stack([state.deter for state in posts], dim=1)
        stacked_stoch = torch.stack([state.stoch for state in posts], dim=1)
        stacked_logits = torch.stack([state.logits for state in posts], dim=1)
        b, t = stacked_deter.shape[:2]
        index = torch.randint(0, t, (b,), device=stacked_deter.device)
        batch_index = torch.arange(b, device=stacked_deter.device)
        state = RSSMState(
            deter=stacked_deter[batch_index, index],
            stoch=stacked_stoch[batch_index, index],
            logits=stacked_logits[batch_index, index],
        )
        return state, index

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
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        state = start
        batch_index = torch.arange(start_index.shape[0], device=start_index.device)
        robot = batch["joint_robot"][batch_index, start_index]
        mask = batch["joint_mask"][batch_index, start_index]
        feats = [self.world_model.rssm.get_feat(state)]
        rewards = []
        continues = []
        entropies = []
        log_probs = []

        for _ in range(self.cfg.horizon):
            feat = feats[-1]
            action, log_prob = self._sample_reinforce_action(feat)
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
        continue_seq = torch.stack(continues, dim=0) * self.cfg.discount
        entropy_seq = torch.stack(entropies, dim=0)
        log_prob_seq = torch.stack(log_probs, dim=0)
        value_seq = self.target_value(feat_seq)
        returns = self._lambda_returns(reward_seq, value_seq, continue_seq)
        normed_returns, return_scale, norm_metrics = self._normalize_returns(returns)
        with torch.no_grad():
            baseline_seq = self.value(actor_feat_seq)
            normed_advantage = (returns - baseline_seq) / return_scale
        return_loss = -normed_returns.mean()
        entropy_loss = self.cfg.entropy_scale * log_prob_seq.mean()
        actor_loss = return_loss + entropy_loss
        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_return_loss": float(return_loss.detach().cpu()),
            "actor_reinforce_loss": float(return_loss.detach().cpu()),
            "actor_entropy_loss": float(entropy_loss.detach().cpu()),
            "actor_entropy": float(entropy_seq.mean().detach().cpu()),
            "actor_entropy_abs_max": float(entropy_seq.abs().max().detach().cpu()),
            "actor_log_prob": float(log_prob_seq.mean().detach().cpu()),
            "actor_advantage_mean": float((returns - baseline_seq).mean().detach().cpu()),
            "actor_normed_advantage_mean": float(normed_advantage.mean().detach().cpu()),
            "actor_normed_advantage_abs_max": float(normed_advantage.abs().max().detach().cpu()),
            "imag_reward": float(reward_seq.mean().detach().cpu()),
            "imag_continue": float(continue_seq.mean().detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
            "return_abs_max": float(returns.abs().max().detach().cpu()),
            "target_value_mean": float(value_seq.mean().detach().cpu()),
            "target_value_abs_max": float(value_seq.abs().max().detach().cpu()),
            **norm_metrics,
        }
        return actor_loss, metrics, actor_feat_seq, returns

    def _sample_reinforce_action(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.actor.forward(feat)
        pre_tanh = dist.rsample()
        raw = torch.tanh(pre_tanh)
        env_action = action_to_env(raw, self.cfg)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - raw.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=(-1, -2))
        return env_action, log_prob

    def _value_loss(self, feat_seq: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        value_loss = self.value.loss(feat_seq, returns)
        if self.cfg.value_slowreg_scale <= 0.0:
            return value_loss
        with torch.no_grad():
            slow_target = self.target_value(feat_seq)
        slowreg_loss = self.value.loss(feat_seq, slow_target.detach())
        return value_loss + self.cfg.value_slowreg_scale * slowreg_loss

    def _replay_value_loss(
        self,
        posts: list[RSSMState],
        batch: dict[str, torch.Tensor],
        start_index: torch.Tensor,
        bootstrap: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        with torch.no_grad():
            feat_seq = torch.stack([self.world_model.rssm.get_feat(state) for state in posts], dim=0)
            reward_bt = batch["reward"]
            continue_bt = batch["continue"] * self.cfg.discount
            returns = []
            feats = []
            lengths = []
            for batch_idx in range(reward_bt.shape[0]):
                end = int(start_index[batch_idx].item()) + 1
                prefix_feat = feat_seq[:end, batch_idx]
                prefix_reward = reward_bt[batch_idx, :end]
                prefix_continue = continue_bt[batch_idx, :end]
                prefix_value = self.target_value(prefix_feat)
                prefix_return = self._lambda_returns(
                    prefix_reward,
                    prefix_value,
                    prefix_continue,
                    bootstrap=bootstrap[batch_idx],
                )
                feats.append(prefix_feat)
                returns.append(prefix_return)
                lengths.append(end)
            replay_feats = torch.cat(feats, dim=0)
            replay_returns = torch.cat(returns, dim=0)
            metrics = {
                "replay_return_mean": float(replay_returns.mean().detach().cpu()),
                "replay_return_abs_max": float(replay_returns.abs().max().detach().cpu()),
                "replay_bootstrap_mean": float(bootstrap.mean().detach().cpu()),
                "replay_prefix_length_mean": float(torch.tensor(lengths, dtype=torch.float32).mean()),
            }
        replay_value_loss = self._value_loss(replay_feats.detach(), replay_returns.detach())
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
