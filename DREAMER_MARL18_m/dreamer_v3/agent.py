from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict
import os
from typing import Any, Optional
import uuid

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
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
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
        self._policy_prev_mask: torch.Tensor | None = None
        self._replay_prefetch_executor: ThreadPoolExecutor | None = None
        self._replay_prefetch_future: Future[dict[str, torch.Tensor]] | None = None
        if bool(getattr(cfg, "replay_prefetch", True)) and self.device.type == "cuda":
            self._replay_prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="dreamer-replay",
            )

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
            if self._policy_prev_mask is None:
                self._policy_prev_mask = torch.zeros_like(batch["joint_mask"])
            post, _ = self.world_model.rssm.obs_step(
                self._policy_state, # 이전 latent state
                self._policy_prev_action, # 이전 action
                self._policy_prev_mask,
                embed, #현재 관측 embed와 이전 latent/action/mask로 posterior latent를 만듦
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
            self._policy_prev_mask = batch["joint_mask"].detach()
        self.actor.train()
        self.world_model.train() # train 모드로 변환
        return action[0].detach().cpu().numpy().astype(np.float32)

    def reset_policy_state(self) -> None:
        self._policy_state = None
        self._policy_prev_action = None
        self._policy_prev_mask = None

    def set_policy_prev_action(self, joint_action_np: np.ndarray) -> None:
        action = torch.from_numpy(joint_action_np[None].astype(np.float32)).to(self.device)
        self._policy_prev_action = action.detach()

    def add_transition(self, msg: Any) -> None:
        self.replay.add_transition_msg(msg)

    def update(self):
        if not self.replay.can_sample(self.cfg.batch_size):
            return None
        batch = self._next_replay_batch()

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

        metric_tensors = {
            **model_metrics,
            **actor_metrics,
            "model_grad_norm": model_grad_norm.detach(),
            "actor_grad_norm": actor_grad_norm.detach(),
            "value_loss": value_loss.detach(),
            "imag_value_loss": imag_value_loss.detach(),
            "replay_value_loss": replay_value_loss.detach(),
            "value_grad_norm": value_grad_norm.detach(),
            **replay_value_metrics,
        }
        self.metrics = self._metrics_to_host(metric_tensors)
        self.metrics.update(
            replay_steps=float(len(self.replay)),
            replay_episodes=float(self.replay.num_episodes),
        )
        return self.metrics

    def _next_replay_batch(self) -> dict[str, torch.Tensor]:
        """Consume one CPU batch and overlap collation of the following batch."""
        executor = self._replay_prefetch_executor
        if executor is None:
            return self.replay.sample(self.cfg.batch_size)

        if self._replay_prefetch_future is None:
            cpu_batch = self.replay.sample_cpu(self.cfg.batch_size, pin_memory=True)
        else:
            cpu_batch = self._replay_prefetch_future.result()

        # Select step references on the learner thread. The background worker
        # only stacks that stable slice, so concurrent replay eviction cannot
        # invalidate the pending batch.
        chunks = self.replay.prepare_sample(self.cfg.batch_size)
        self._replay_prefetch_future = executor.submit(
            self.replay.collate,
            chunks,
            pin_memory=True,
        )
        return self.replay.to_device(cpu_batch, non_blocking=True)

    def _metrics_to_host(
        self,
        metrics: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Convert all scalar CUDA metrics with one batched device transfer."""
        names = list(metrics)
        if not names:
            return {}
        scalars = []
        for name in names:
            value = metrics[name]
            if not torch.is_tensor(value) or value.numel() != 1:
                raise ValueError(f"Metric {name!r} must be a scalar tensor")
            scalars.append(value.detach().reshape(()).to(self.device, dtype=torch.float32))
        host_values = torch.stack(scalars).cpu().tolist()
        return dict(zip(names, host_values))

    def _amp_context(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype)
        return torch.cuda.amp.autocast(enabled=True, dtype=self.amp_dtype)

    def save(self, filepath: str) -> None:
        self.save_checkpoint_payload(self.checkpoint_payload_cpu(), filepath)

    def checkpoint_payload_cpu(self) -> dict[str, Any]:
        """Clone a complete, immutable checkpoint snapshot onto CPU.

        The snapshot has the same schema as the historical synchronous save
        payload, so disk serialization may safely run on a background thread.
        """
        payload = {
            "cfg": asdict(self.cfg),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "target_value": self.target_value.state_dict(),
            "return_low": self.return_low,
            "return_high": self.return_high,
            "return_norm_initialized": self.return_norm_initialized,
            "model_opt": self.model_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
        }
        return self._clone_payload_to_cpu(payload)

    @staticmethod
    def save_checkpoint_payload(payload: dict[str, Any], filepath: str) -> None:
        """Atomically write a checkpoint payload prepared by the learner."""
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        tmp_path = f"{filepath}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            with open(tmp_path, "wb") as file_obj:
                torch.save(payload, file_obj)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(tmp_path, filepath)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @classmethod
    def _clone_payload_to_cpu(cls, value):
        if torch.is_tensor(value):
            return value.detach().to(device="cpu").clone()
        if isinstance(value, np.ndarray):
            return value.copy()
        if isinstance(value, dict):
            return {key: cls._clone_payload_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._clone_payload_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._clone_payload_to_cpu(item) for item in value)
        return value

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
            "encoder": self._state_dict_to_numpy(self.world_model.encoder.state_dict()),
            "rssm": self._state_dict_to_numpy(self.world_model.rssm.state_dict()),
            "actor": self._state_dict_to_numpy(self.actor.state_dict()),
        }

    def load_worker_state(self, state: dict[str, dict[str, np.ndarray]]) -> None:
        if "encoder" in state and "rssm" in state:
            self.world_model.encoder.load_state_dict(
                self._numpy_to_state_dict(state["encoder"])
            )
            self.world_model.rssm.load_state_dict(
                self._numpy_to_state_dict(state["rssm"])
            )
        elif "world_model" in state:
            # Backward compatibility with MARL17/full worker broadcasts.
            self.world_model.load_state_dict(
                self._numpy_to_state_dict(state["world_model"])
            )
        else:
            raise KeyError("Worker state must contain encoder/rssm or legacy world_model")
        self.actor.load_state_dict(self._numpy_to_state_dict(state["actor"]))
        self.reset_policy_state()

    def _state_dict_to_numpy(self, state_dict):
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in state_dict.items()
        }

    def _numpy_to_state_dict(self, payload):
        return {
            key: (
                torch.from_numpy(value).to(self.device)
                if isinstance(value, np.ndarray)
                else value.to(self.device)
            )
            for key, value in payload.items()
        }

    def save_replay_buffer(self, filepath: str) -> None:
        self.replay.save(filepath)

    def load_replay_buffer(self, filepath: str) -> None:
        if os.path.exists(filepath):
            self._clear_replay_prefetch()
            loaded = self.replay.load(filepath)
            if loaded:
                print(
                    f"[DreamerAgent] loaded replay buffer: {filepath} "
                    f"episodes={self.replay.num_episodes}, steps={len(self.replay)}"
                )
            else:
                print("[DreamerAgent] starting with an empty replay buffer.")

    def _clear_replay_prefetch(self) -> None:
        future = self._replay_prefetch_future
        self._replay_prefetch_future = None
        if future is not None:
            future.cancel()

    def close(self) -> None:
        self._clear_replay_prefetch()
        executor = self._replay_prefetch_executor
        self._replay_prefetch_executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _sample_start_state(
        self,
        posts: list[RSSMState],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[RSSMState, torch.Tensor, torch.Tensor]:
        stacked_deter = torch.stack([state.deter for state in posts], dim=1)
        stacked_stoch = torch.stack([state.stoch for state in posts], dim=1)
        stacked_logits = torch.stack([state.logits for state in posts], dim=1)
        b, t = stacked_deter.shape[:2]
        requested_k = int(getattr(self.cfg, "imag_starts_per_sequence", 16))
        if requested_k <= 0:
            raise ValueError("imag_starts_per_sequence must be positive")

        # Normal replay batches always have one fixed context prefix per row.
        # Use a fixed-shape top-k sample on that path so selecting starts does
        # not introduce a data-dependent CUDA-to-host synchronization.
        if batch is not None and hasattr(self.cfg, "replay_context"):
            context = min(max(int(self.cfg.replay_context), 0), t)
            candidate_count = t - context
            if candidate_count <= 0:
                raise ValueError("No valid posterior state is available for imagination")
            k = min(requested_k, candidate_count)
            scores = torch.rand((b, candidate_count), device=stacked_deter.device)
            index_grid = scores.topk(k, dim=1, largest=False).indices + context
            batch_grid = torch.arange(
                b,
                device=stacked_deter.device,
            ).unsqueeze(1).expand(b, k)
            index = index_grid.reshape(-1)
            batch_index = batch_grid.reshape(-1)
            state = RSSMState(
                deter=stacked_deter[batch_index, index],
                stoch=stacked_stoch[batch_index, index],
                logits=stacked_logits[batch_index, index],
            )
            return state, index, batch_index

        if batch is not None and "loss_mask" in batch:
            valid = batch["loss_mask"] > 0
            if valid.shape != (b, t):
                raise ValueError(
                    f"loss_mask shape {tuple(valid.shape)} does not match posterior shape {(b, t)}"
                )
        else:
            valid = torch.ones((b, t), dtype=torch.bool, device=stacked_deter.device)

        # Sorting independent uniform scores produces a uniformly random subset
        # without replacement. Invalid/context positions sort last and are
        # filtered, naturally yielding fewer than K starts for short rows.
        k = min(requested_k, t)
        scores = torch.rand((b, t), device=stacked_deter.device)
        ranked = scores.masked_fill(~valid, torch.inf).argsort(dim=1)[:, :k]
        ranked_valid = valid.gather(1, ranked)
        batch_grid = torch.arange(b, device=stacked_deter.device).unsqueeze(1).expand(b, k)
        index = ranked[ranked_valid]
        batch_index = batch_grid[ranked_valid]
        if index.numel() == 0:
            raise ValueError("No valid posterior state is available for imagination")
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
                param_norm = param.detach().norm(2).clamp_min(eps)
                grad_norm = param.grad.detach().norm(2)
                max_norm = clip * param_norm
                if grad_norm > max_norm:
                    param.grad.mul_(max_norm / grad_norm.clamp_min(1e-6))
        return global_norm

    def _normalize_returns(
        self,
        returns: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
            "return_norm_low": self.return_low.detach(),
            "return_norm_high": self.return_high.detach(),
            "return_norm_scale": scale.detach(),
            "return_batch_low": batch_low.detach(),
            "return_batch_high": batch_high.detach(),
            "normed_return_mean": normalized.mean().detach(),
            "normed_return_abs_max": normalized.abs().max().detach(),
        }
        return normalized, scale, metrics

    def _actor_loss(
        self,
        start: RSSMState,
        batch: dict[str, torch.Tensor],
        start_index: torch.Tensor,
        start_batch_index: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state = start
        batch_index = start_batch_index.to(device=start_index.device, dtype=torch.long)
        mask = batch["joint_mask"][batch_index, start_index]
        feats = [self.world_model.rssm.get_feat(state)]
        rewards = []
        continues = []
        entropies = []
        log_probs = []

        for _ in range(self.cfg.horizon): # img step 
            feat = feats[-1]
            action, log_prob, entropy = self._sample_reinforce_action(feat.detach(), mask)
            action = action * mask.unsqueeze(-1)
            state = self.world_model.rssm.img_step(state, action.detach(), mask)
            next_feat = self.world_model.rssm.get_feat(state)
            rewards.append(self.world_model.reward_head(next_feat).squeeze(-1))
            continues.append(torch.sigmoid(self.world_model.continue_head(next_feat).squeeze(-1)))
            entropies.append(entropy)
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
            "actor_loss": actor_loss.detach(),
            "actor_return_loss": reinforce_loss.detach(),
            "actor_reinforce_loss": reinforce_loss.detach(),
            "actor_entropy_loss": entropy_loss.detach(),
            "actor_entropy": entropy_seq.mean().detach(),
            "actor_entropy_abs_max": entropy_seq.abs().max().detach(),
            "actor_log_prob": log_prob_seq.mean().detach(),
            "actor_advantage_mean": (returns - baseline_seq).mean().detach(),
            "actor_normed_advantage_mean": normed_advantage.mean().detach(),
            "actor_normed_advantage_abs_max": normed_advantage.abs().max().detach(),
            "imag_reward": reward_seq.mean().detach(),
            "imag_continue": continue_seq.mean().detach(),
            "imag_discount": discount_seq.mean().detach(),
            "return_mean": returns.mean().detach(),
            "return_abs_max": returns.abs().max().detach(),
            "target_value_mean": value_seq.mean().detach(),
            "target_value_abs_max": value_seq.abs().max().detach(),
            "imag_start_count": actor_loss.new_tensor(start_index.numel()),
            "imag_starts_per_sequence": actor_loss.new_tensor(
                start_index.numel() / max(1, batch["reward"].shape[0])
            ),
            "imag_start_index_mean": start_index.float().mean().detach(),
            **norm_metrics,
        }
        return actor_loss, metrics, actor_feat_seq, returns, weights

    def _sample_reinforce_action(
        self,
        feat: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.actor.forward(feat)
        pre_tanh = dist.rsample()
        raw = torch.tanh(pre_tanh)
        env_action = action_to_env(raw, self.cfg)
        pre_tanh_logprob = pre_tanh.detach() # detached sample을 score-function 대상으로 사용
        raw_logprob = raw.detach()
        log_prob_per_action = dist.log_prob(pre_tanh_logprob) - torch.log(
            1.0 - raw_logprob.pow(2) + 1e-6
        )
        active_mask = mask.to(dtype=log_prob_per_action.dtype)
        log_prob_per_robot = log_prob_per_action.sum(dim=-1)
        log_prob = (log_prob_per_robot * active_mask).sum(dim=-1)
        entropy_per_robot = dist.entropy().sum(dim=-1)
        entropy = (entropy_per_robot * active_mask).sum(dim=-1)
        return env_action, log_prob, entropy

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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
                "replay_return_mean": replay_returns.mean().detach(),
                "replay_return_abs_max": replay_returns.abs().max().detach(),
                "replay_bootstrap_mean": bootstrap.mean().detach(),
                "replay_start_count": replay_returns.new_tensor(start_index.numel()),
                "replay_start_index_mean": start_index.float().mean().detach(),
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
