from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch

from .config import DreamerConfig
from .networks import Actor, RSSMState, Value, WorldModel
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
        self.model_opt = torch.optim.AdamW(self.world_model.parameters(), lr=cfg.model_lr)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr)
        self.value_opt = torch.optim.AdamW(self.value.parameters(), lr=cfg.value_lr)
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
            post, _ = self.world_model.rssm.obs_step(
                init,
                zero_action,
                batch["joint_robot"],
                batch["joint_mask"],
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
            post, _ = self.world_model.rssm.obs_step(
                init,
                zero_action,
                batch["joint_robot"],
                batch["joint_mask"],
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
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.grad_clip)
        self.model_opt.step()

        with torch.no_grad():
            start = self._sample_start_state(posts)
        self._set_requires_grad(self.world_model, False)
        try:
            actor_loss, actor_metrics, imagined_feats, returns = self._actor_loss(start, batch)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.actor_grad_clip)
            self.actor_opt.step()

            value_loss = self._value_loss(imagined_feats.detach(), returns.detach())
            self.value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            value_grad_norm = torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.value_grad_clip)
            self.value_opt.step()
            self._update_target_value()
        finally:
            self._set_requires_grad(self.world_model, True)

        self.metrics = {
            **model_metrics,
            **actor_metrics,
            "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "value_grad_norm": float(value_grad_norm.detach().cpu()),
            "replay_steps": float(len(self.replay)),
            "replay_episodes": float(self.replay.num_episodes),
        }
        return self.metrics

    def save(self, filepath: str) -> None:
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg,
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

    def get_worker_state(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "world_model": {
                k: v.detach().cpu()
                for k, v in self.world_model.state_dict().items()
            },
            "actor": {
                k: v.detach().cpu()
                for k, v in self.actor.state_dict().items()
            },
        }

    def load_worker_state(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.world_model.load_state_dict(state["world_model"])
        self.actor.load_state_dict(state["actor"])

    def save_replay_buffer(self, filepath: str) -> None:
        self.replay.save(filepath)

    def load_replay_buffer(self, filepath: str) -> None:
        if os.path.exists(filepath):
            self.replay.load(filepath)

    def _sample_start_state(self, posts: list[RSSMState]) -> RSSMState:
        stacked_deter = torch.stack([state.deter for state in posts], dim=1)
        stacked_stoch = torch.stack([state.stoch for state in posts], dim=1)
        stacked_logits = torch.stack([state.logits for state in posts], dim=1)
        b, t = stacked_deter.shape[:2]
        index = torch.randint(0, t, (b,), device=stacked_deter.device)
        batch_index = torch.arange(b, device=stacked_deter.device)
        return RSSMState(
            deter=stacked_deter[batch_index, index],
            stoch=stacked_stoch[batch_index, index],
            logits=stacked_logits[batch_index, index],
        )

    def _set_requires_grad(self, module: torch.nn.Module, enabled: bool) -> None:
        for param in module.parameters():
            param.requires_grad_(enabled)

    def _update_target_value(self) -> None:
        tau = self.cfg.target_value_tau
        with torch.no_grad():
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def _normalize_returns(self, returns: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
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
        return normalized, metrics

    def _actor_loss(
        self,
        start: RSSMState,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        state = start
        robot = batch["joint_robot"][:, -1]
        mask = batch["joint_mask"][:, -1]
        feats = []
        rewards = []
        continues = []
        entropies = []

        for _ in range(self.cfg.horizon):
            feat = self.world_model.rssm.get_feat(state)
            action, log_prob = self.actor.sample(feat, deterministic=False)
            state = self.world_model.rssm.img_step(state, action, robot, mask)
            next_feat = self.world_model.rssm.get_feat(state)
            rewards.append(self.world_model.reward_head(next_feat).squeeze(-1))
            continues.append(torch.sigmoid(self.world_model.continue_head(next_feat).squeeze(-1)))
            entropies.append((-log_prob).clamp(-self.cfg.actor_entropy_clip, self.cfg.actor_entropy_clip))
            feats.append(next_feat)

        feat_seq = torch.stack(feats, dim=0)
        reward_seq = torch.stack(rewards, dim=0)
        continue_seq = torch.stack(continues, dim=0) * self.cfg.discount
        entropy_seq = torch.stack(entropies, dim=0)
        value_seq = self.target_value(feat_seq)
        returns = self._lambda_returns(reward_seq, value_seq, continue_seq)
        normed_returns, norm_metrics = self._normalize_returns(returns)
        actor_loss = -(normed_returns + self.cfg.entropy_scale * entropy_seq).mean()
        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "actor_entropy": float(entropy_seq.mean().detach().cpu()),
            "actor_entropy_abs_max": float(entropy_seq.abs().max().detach().cpu()),
            "imag_reward": float(reward_seq.mean().detach().cpu()),
            "imag_continue": float(continue_seq.mean().detach().cpu()),
            "return_mean": float(returns.mean().detach().cpu()),
            "return_abs_max": float(returns.abs().max().detach().cpu()),
            "target_value_mean": float(value_seq.mean().detach().cpu()),
            "target_value_abs_max": float(value_seq.abs().max().detach().cpu()),
            **norm_metrics,
        }
        return actor_loss, metrics, feat_seq, returns

    def _value_loss(self, feat_seq: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return self.value.loss(feat_seq, returns)

    def _lambda_returns(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        next_value = values[-1]
        outputs = []
        for t in reversed(range(rewards.shape[0])):
            bootstrap = next_value if t == rewards.shape[0] - 1 else values[t + 1]
            next_value = rewards[t] + discounts[t] * (
                (1.0 - self.cfg.lambda_) * bootstrap + self.cfg.lambda_ * next_value
            )
            outputs.append(next_value)
        return torch.stack(list(reversed(outputs)), dim=0)
