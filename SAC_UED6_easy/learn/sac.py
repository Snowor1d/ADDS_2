"""SAC with a centralised team critic and a shared decentralised actor.

docs/outdoor_madrl_redesign.md section 5.1 and 5.4:

  * rew-v2: the critic's per-robot outputs are evaluated as the masked team
    mean. The critic loss, the Bellman target and the actor loss all use that
    one definition, and in all three the minimum over the two critics is taken
    after averaging over the team.
  * rew-v3 (own collision): each robot's reward is the team task reward plus
    its own collisions. Each per-robot output is fitted to that robot's own
    target, and each robot's actor term reads its own output, so a robot
    that hits a wall is the one charged for it. The team mean is still what
    is logged.
  * The actor update keeps the other robots at their stored actions and
    replaces only the sampled robot's: what this robot should have done given
    what its teammates actually did.
  * A transition covers the k simulation steps an action was actually held.
    Its reward is the per-step discounted sum over those steps and the
    bootstrap is discounted by gamma_step ** k, so a shortened first or last
    interval is neither dropped nor over-discounted.
  * Exploration draws the full 7-number action, move, heading and mode.
"""

from __future__ import annotations

import os
import random
from itertools import product
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from learn.networks import CentralizedCritic, PolicyNetwork, team_value
from learn.replay import SchemaMismatch, check_schema
from sim import robot_action

OBS_KEYS = ("ego", "mid", "glob", "state")


def exploration_action(rng=None) -> np.ndarray:
    """The one exploratory action generator: move, heading and mode."""
    return robot_action.random_action(rng)


def to_torch(batch_obs: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in batch_obs.items()}


class SACAgent:
    def __init__(self, cfg, device: str = "cpu", replay=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.action_dim = robot_action.ACTION_DIM
        self.gamma = cfg.gamma()
        self.tau = 0.995
        self.batch_size = int(cfg.BATCH_SIZE)
        # Entropy temperature, learned when ALPHA_AUTO. Optimised in log space
        # so it stays positive; `self.alpha` is always the current value as a
        # constant, which is what the targets and the actor loss use.
        self.alpha_auto = bool(cfg.ALPHA_AUTO)
        self.target_entropy = (float(cfg.ALPHA_TARGET_ENTROPY)
                               if cfg.ALPHA_TARGET_ENTROPY is not None
                               else -float(robot_action.CONT_DIM))
        self.log_alpha = torch.tensor(float(np.log(float(cfg.ALPHA_START))),
                                      device=self.device,
                                      requires_grad=self.alpha_auto)
        self.alpha_opt = (torch.optim.Adam([self.log_alpha],
                                           lr=float(cfg.ALPHA_LR))
                          if self.alpha_auto else None)
        self.alpha = self.log_alpha.detach().exp()
        # A floor on the temperature. Without it alpha fell from 0.04 to
        # 0.001 within 16k updates and the squashed actor settled on the edge
        # of its range, where its gradient vanishes (2026-10-05).
        self.log_alpha_min = float(np.log(float(cfg.ALPHA_MIN)))
        from sim.rewards import PER_ROBOT_VERSIONS
        # Robot-charged terms (collision; under rew-v4 also guide use and
        # bystanders) go to the robot that incurred them.
        self.own_collision = cfg.REWARD_VERSION in PER_ROBOT_VERSIONS
        self.epsilon = float(cfg.START_EPSILON)
        self.replay = replay
        mk_q = lambda: CentralizedCritic(cfg, self.action_dim).to(self.device)
        self.q1, self.q2 = mk_q(), mk_q()
        self.q1_target, self.q2_target = mk_q(), mk_q()
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_target.parameters()) + \
                list(self.q2_target.parameters()):
            p.requires_grad_(False)
        self.policy = PolicyNetwork(cfg).to(self.device)
        lr = float(cfg.LR)
        self.q_opt = torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr, weight_decay=float(cfg.WD_Q))
        self.pi_opt = torch.optim.AdamW(self.policy.parameters(), lr=lr,
                                        weight_decay=float(cfg.WD_PI))
        self.updates = 0

    # --------------------------------------------------------------- acting

    def act(self, obs: Dict[str, np.ndarray], deterministic: bool = False,
            epsilon: float = 0.0, rng=None) -> np.ndarray:
        """Actions for every robot in `obs` (leading dim = robots)."""
        rng = rng or random
        n = obs["state"].shape[0]
        out = np.zeros((n, self.action_dim), np.float32)
        explore = [rng.random() < epsilon for _ in range(n)]
        if not all(explore):
            t = to_torch({k: obs[k] for k in OBS_KEYS}, self.device)
            with torch.no_grad():
                if deterministic:
                    a = self.policy.deterministic_action(
                        t["ego"], t["mid"], t["glob"], t["state"])
                else:
                    a, _ = self.policy.sample_action(
                        t["ego"], t["mid"], t["glob"], t["state"])
            out[:] = a.cpu().numpy()
        for i, e in enumerate(explore):
            if e:
                out[i] = exploration_action(rng)
        return out

    # -------------------------------------------------------------- learning

    def _per_robot_actions(self, obs, mask):
        B, N = mask.shape
        flat = {k: obs[k].reshape(B * N, *obs[k].shape[2:]) for k in OBS_KEYS}
        a, logp = self.policy.sample_action(flat["ego"], flat["mid"],
                                            flat["glob"], flat["state"])
        return a.reshape(B, N, -1), logp.reshape(B, N)

    def _collision_return(self, batch, g_step) -> torch.Tensor:
        """(B, N) discounted robot-charged terms of each robot (collision,
        and under rew-v4 guide use and bystanders), already weighted."""
        rp = batch["robot_penalties"]                   # (B, K, N)
        B, K, N = rp.shape
        flat = rp.permute(0, 2, 1).reshape(B * N, K)
        hold = batch["hold"].repeat_interleave(N)
        return discounted_return(flat, hold, g_step).reshape(B, N)

    def _per_robot_hybrid(self, obs, mask):
        B, N = mask.shape
        flat = {k: obs[k].reshape(B * N, *obs[k].shape[2:]) for k in OBS_KEYS}
        values = self.policy.sample_hybrid(*(flat[k] for k in OBS_KEYS))
        return tuple(v.reshape(B, N, *v.shape[1:]) for v in values)

    def _expected_next_value(self, obs, mask, per_robot):
        """Exact joint categorical expectation, conditional on sampled moves.

        Take the clipped double-Q minimum inside the expectation. In team
        reward mode, reduce over robots before taking that minimum.
        """
        cont, logpc, probs, logpm = self._per_robot_hybrid(obs, mask)
        B, N, K = probs.shape
        active = torch.nonzero(mask.any(dim=0), as_tuple=False).flatten().tolist()
        shape = (B, N) if per_robot else (B,)
        expected_q = cont.new_zeros(shape)
        critics = (self.q1_target, self.q2_target)
        features = [q.encode_observation(obs) if isinstance(q, CentralizedCritic)
                    else None for q in critics]
        for modes in product(range(K), repeat=len(active)):
            indices = torch.zeros(B, N, dtype=torch.long, device=cont.device)
            for j, mode in zip(active, modes):
                indices[:, j] = mode
            one_hot = F.one_hot(indices, K).to(cont.dtype)
            action = torch.cat((cont, one_hot), dim=-1)
            selected = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            factors = torch.where(mask.bool(), selected, torch.ones_like(selected))
            for j in active:
                # A padded slot in a mixed-size batch is still enumerated;
                # divide out its K identical copies rather than overcounting.
                factors[:, j] = torch.where(mask[:, j].bool(), selected[:, j],
                                            torch.full_like(selected[:, j], 1.0 / K))
            weight = factors.prod(-1)
            q1, q2 = [q.values_from_features(f, action, mask) if f is not None
                      else q(obs, action, mask) for q, f in zip(critics, features)]
            if not per_robot:
                q1, q2 = team_value(q1, mask), team_value(q2, mask)
            q = torch.min(q1, q2)
            expected_q += q * (weight[:, None] if per_robot else weight)
        expected_logp = logpc + (probs * logpm).sum(-1)
        if not per_robot:
            expected_logp = team_value(expected_logp, mask)
        return expected_q - self.alpha * expected_logp

    def robot_targets(self, batch) -> torch.Tensor:
        """(B, N) y_i = R_task + R_collision,i + gamma_step^k (1 - terminal)
        [min_k Q_k,i(s', a') - alpha * log pi(a'_i|s'_i)]."""
        cfg = self.cfg
        obs2 = batch["next_obs"]
        m2 = batch["next_mask"]
        with torch.no_grad():
            if cfg.SAC_MODE_ESTIMATOR == "expectation":
                v2 = self._expected_next_value(obs2, m2, per_robot=True)
            else:
                a2, logp2 = self._per_robot_actions(obs2, m2)
                q1 = self.q1_target(obs2, a2, m2)
                q2 = self.q2_target(obs2, a2, m2)
                v2 = torch.min(q1, q2) - self.alpha * logp2
            g_step = self.gamma ** (1.0 / max(1, int(cfg.ACTION_SCALE)))
            r_task = discounted_return(batch["step_rewards"], batch["hold"],
                                       g_step)
            boot = (torch.pow(torch.full_like(batch["hold"], g_step),
                              batch["hold"]) * (1.0 - batch["terminal"]))
            return (r_task[:, None] + self._collision_return(batch, g_step)
                    + boot[:, None] * v2)

    def targets(self, batch) -> torch.Tensor:
        """y = R + gamma_step^k (1 - terminal) [min_k Q_k,team(s', a')
        - alpha * mean_team log pi(a'|s')]."""
        cfg = self.cfg
        obs2 = batch["next_obs"]
        m2 = batch["next_mask"]
        with torch.no_grad():
            if cfg.SAC_MODE_ESTIMATOR == "expectation":
                v2 = self._expected_next_value(obs2, m2, per_robot=False)
            else:
                a2, logp2 = self._per_robot_actions(obs2, m2)
                q1 = team_value(self.q1_target(obs2, a2, m2), m2)
                q2 = team_value(self.q2_target(obs2, a2, m2), m2)
                ent = team_value(logp2, m2)
                v2 = torch.min(q1, q2) - self.alpha * ent
            g_step = self.gamma ** (1.0 / max(1, int(cfg.ACTION_SCALE)))
            return (discounted_return(batch["step_rewards"], batch["hold"],
                                      g_step)
                    + torch.pow(torch.full_like(batch["hold"], g_step),
                                batch["hold"])
                    * (1.0 - batch["terminal"]) * v2)

    def update(self, batch_np) -> Dict[str, float]:
        """One gradient step on a sampled batch (ReplayBuffer.sample)."""
        dev = self.device
        batch = {
            "obs": to_torch(batch_np["obs"], dev),
            "next_obs": to_torch(batch_np["next_obs"], dev),
            "mask": torch.as_tensor(batch_np["mask"], device=dev),
            "next_mask": torch.as_tensor(batch_np["next_mask"], device=dev),
            "action": torch.as_tensor(batch_np["action"], device=dev),
            "step_rewards": torch.as_tensor(batch_np["step_rewards"],
                                            device=dev),
            "robot_penalties": torch.as_tensor(
                batch_np["robot_penalties"], device=dev,
                dtype=torch.float32)
            if "robot_penalties" in batch_np else None,
            "hold": torch.as_tensor(batch_np["hold"], device=dev),
            "terminal": torch.as_tensor(batch_np["terminal"], device=dev),
            "agent_index": torch.as_tensor(batch_np["agent_index"],
                                           device=dev, dtype=torch.long),
        }
        # Robot-order augmentation: slot identity must not carry meaning.
        batch = permute_robots(batch)
        obs, mask, act = batch["obs"], batch["mask"], batch["action"]
        if self.own_collision:
            y_r = self.robot_targets(batch)
            q1_r = self.q1(obs, act, mask)
            q2_r = self.q2(obs, act, mask)
            m = mask.float()
            denom = m.sum().clamp(min=1.0)
            loss_q = ((((q1_r - y_r) ** 2) * m).sum()
                      + (((q2_r - y_r) ** 2) * m).sum()) / denom
            q1 = team_value(q1_r, mask)
            y = team_value(y_r, mask)
        else:
            y = self.targets(batch)
            q1 = team_value(self.q1(obs, act, mask), mask)
            q2 = team_value(self.q2(obs, act, mask), mask)
            loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # Actor. ACTOR_UPDATE_ROBOTS picks whose action is re-drawn from the
        # current policy and differentiated: one sampled robot per sample, or
        # every real robot. ACTOR_TEAMMATE_ACTIONS picks what the others do
        # meanwhile: what they actually did (stored), or what the current
        # policy would do now (current, no gradient through them).
        loss_pi, logp = self._actor_loss(obs, act, mask, batch["agent_index"])
        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()

        # Temperature: raise alpha while the policy's entropy is below the
        # target, lower it while above. Uses this batch's fresh actions.
        loss_alpha = None
        if self.alpha_auto:
            loss_alpha = -(self.log_alpha
                           * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            loss_alpha.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(min=self.log_alpha_min)
            self.alpha = self.log_alpha.detach().exp()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self.updates += 1
        return {"train/loss_q": float(loss_q.item()),
                "train/loss_pi": float(loss_pi.item()),
                "train/q_team": float(q1.mean().item()),
                "train/target": float(y.mean().item()),
                "train/entropy": float(-logp.mean().item()),
                "train/alpha": float(self.alpha.item()),
                "train/target_entropy": self.target_entropy,
                **({"train/loss_alpha": float(loss_alpha.item())}
                   if loss_alpha is not None else {})}

    def _team_q_min(self, obs, actions, mask):
        return torch.min(team_value(self._actor_q(self.q1, obs, actions, mask), mask),
                         team_value(self._actor_q(self.q2, obs, actions, mask), mask))

    def _robot_q_min(self, obs, actions, mask, j):
        """(B,) robot j's own value, the smaller of the two critics."""
        return torch.min(self._actor_q(self.q1, obs, actions, mask)[:, j],
                         self._actor_q(self.q2, obs, actions, mask)[:, j])

    def _actor_q(self, critic, obs, actions, mask):
        features = getattr(self, "_actor_critic_features", {}).get(id(critic))
        if features is not None:
            return critic.values_from_features(features, actions, mask)
        return critic(obs, actions, mask)

    def _actor_loss(self, obs, act, mask, agent_index):
        """(loss, log-probabilities of the differentiated actions, flat over
        real robots). The critics are frozen for the duration."""
        cfg = self.cfg
        B, N = mask.shape
        dev = mask.device
        rows = torch.arange(B, device=dev)
        critics = list(self.q1.parameters()) + list(self.q2.parameters())
        for p in critics:
            p.requires_grad_(False)
        try:
            if cfg.SAC_MODE_ESTIMATOR == "expectation":
                self._actor_critic_features = {
                    id(q): q.encode_observation(obs) for q in (self.q1, self.q2)
                    if isinstance(q, CentralizedCritic)}
                return self._expected_actor_loss(obs, act, mask, agent_index)
            if cfg.ACTOR_TEAMMATE_ACTIONS == "current":
                with torch.no_grad():
                    base, _ = self._per_robot_actions(obs, mask)
            else:
                base = act
            if cfg.ACTOR_UPDATE_ROBOTS == "one":
                idx = agent_index.clamp(0, N - 1)
                pick = {k: obs[k][rows, idx] for k in OBS_KEYS}
                new_a, logp = self.policy.sample_action(
                    pick["ego"], pick["mid"], pick["glob"], pick["state"])
                mixed = base.clone()
                mixed[rows, idx] = new_a
                if self.own_collision:
                    q1 = self.q1(obs, mixed, mask)[rows, idx]
                    q2 = self.q2(obs, mixed, mask)[rows, idx]
                    qn = torch.min(q1, q2)
                else:
                    qn = self._team_q_min(obs, mixed, mask)
                return (self.alpha * logp - qn).mean(), logp
            # Every real robot.
            new_all, logp_all = self._per_robot_actions(obs, mask)
            real = mask > 0
            if cfg.ACTOR_TEAMMATE_ACTIONS == "current" and not self.own_collision:
                # All re-drawn together: one critic pass differentiates the
                # team value with respect to every robot's action.
                # (rew-v3 goes through the per-slot loop below instead, so
                # each robot's action is judged by its own value only.)
                mixed = torch.where(real.unsqueeze(-1), new_all, base)
                qn = self._team_q_min(obs, mixed, mask)
                ent = team_value(logp_all, mask.float())
                return (self.alpha * ent - qn).mean(), logp_all[real]
            # Each robot against its teammates' stored actions: one critic
            # pass per slot, a robot's term counted only where it exists.
            total = torch.zeros((), device=dev)
            count = real.sum().clamp(min=1)
            for j in range(N):
                valid = real[:, j]
                if not bool(valid.any()):
                    continue
                mixed = base.clone()
                mixed[:, j] = new_all[:, j]
                qn = (self._robot_q_min(obs, mixed, mask, j)
                      if self.own_collision
                      else self._team_q_min(obs, mixed, mask))
                term = self.alpha * logp_all[:, j] - qn
                total = total + (term * valid.float()).sum()
            return total / count, logp_all[real]
        finally:
            self._actor_critic_features = {}
            for p in critics:
                p.requires_grad_(True)

    def _expected_actor_loss(self, obs, act, mask, agent_index):
        """Enumerate this robot's modes against fixed teammate actions."""
        cfg = self.cfg
        if cfg.ACTOR_TEAMMATE_ACTIONS == "current":
            with torch.no_grad():
                base, _ = self._per_robot_actions(obs, mask)
        else:
            base = act
        cont, logpc, probs, logpm = self._per_robot_hybrid(obs, mask)
        B, N, K = probs.shape
        real = mask > 0
        chosen = real.clone()
        if cfg.ACTOR_UPDATE_ROBOTS == "one":
            chosen.zero_()
            chosen.scatter_(1, agent_index.clamp(0, N - 1)[:, None], True)
            chosen &= real
        total = cont.new_zeros(())
        for j in range(N):
            valid = chosen[:, j]
            if not bool(valid.any()):
                continue
            expected = cont.new_zeros(B)
            for mode in range(K):
                one_hot = cont.new_zeros(B, K)
                one_hot[:, mode] = 1.0
                mixed = base.clone()
                mixed[:, j] = torch.cat((cont[:, j], one_hot), dim=-1)
                q = (self._robot_q_min(obs, mixed, mask, j) if self.own_collision
                     else self._team_q_min(obs, mixed, mask))
                # Preserve the original joint-current team objective's Q
                # gradient scale when averaging all robots' entropy terms.
                if (not self.own_collision and cfg.ACTOR_UPDATE_ROBOTS == "all"
                        and cfg.ACTOR_TEAMMATE_ACTIONS == "current"):
                    q = q * mask.sum(-1)
                expected += probs[:, j, mode] * (
                    self.alpha * (logpc[:, j] + logpm[:, j, mode]) - q)
            total += (expected * valid).sum()
        expected_logp = logpc + (probs * logpm).sum(-1)
        return total / chosen.sum().clamp(min=1), expected_logp[chosen]

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(self.tau).add_((1 - self.tau) * p)

    # ---------------------------------------------------------- persistence

    def schema(self) -> Dict[str, str]:
        return self.cfg.schema_versions()

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "schema": self.schema(),
            "observation_schema": self.cfg.observation_schema(),
            "experiment_id": self.cfg.EXPERIMENT_ID,
            "config_fingerprint": self.cfg.fingerprint,
            "critic_privileged": bool(self.cfg.CRITIC_PRIVILEGED_CROWD),
            "network": {"encoder": self.cfg.NET_ENCODER,
                        "size": self.cfg.NET_SIZE},
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "q_opt": self.q_opt.state_dict(), "pi_opt": self.pi_opt.state_dict(),
            "updates": self.updates, "epsilon": self.epsilon,
            "sac_mode_estimator": self.cfg.SAC_MODE_ESTIMATOR,
            "log_alpha": float(self.log_alpha.item()),
            "alpha_opt": (self.alpha_opt.state_dict()
                          if self.alpha_opt is not None else None),
            **(extra or {}),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @staticmethod
    def read_schema(path: str) -> Dict[str, str]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return dict(ckpt.get("schema", {}))

    def load(self, path: str, policy_only: bool = False,
             allow_transfer: bool = False) -> dict:
        """Load a checkpoint written under the same schemas.

        Anything else is refused unless `allow_transfer`, which is the explicit
        weight-transfer experiment and still requires matching tensor shapes.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if not allow_transfer:
            check_schema(ckpt.get("schema", {}), self.schema(),
                         f"checkpoint {path}")
            stored = ckpt.get("observation_schema")
            if stored is not None and stored != self.cfg.observation_schema():
                diff = sorted(k for k in set(stored) | set(
                    self.cfg.observation_schema())
                    if stored.get(k) != self.cfg.observation_schema().get(k))
                raise SchemaMismatch(f"checkpoint {path} observation settings "
                                     f"differ: {diff}")
        stored_net = ckpt.get("network")
        here = {"encoder": self.cfg.NET_ENCODER, "size": self.cfg.NET_SIZE}
        if stored_net is not None and stored_net != here:
            raise SchemaMismatch(
                f"checkpoint {path} holds a {stored_net['encoder']}/"
                f"{stored_net['size']} network; this run builds "
                f"{here['encoder']}/{here['size']}. Set NET_ENCODER and "
                "NET_SIZE to match it.")
        self.policy.load_state_dict(ckpt["policy"])
        if not policy_only:
            self.q1.load_state_dict(ckpt["q1"])
            self.q2.load_state_dict(ckpt["q2"])
            self.q1_target.load_state_dict(ckpt["q1_target"])
            self.q2_target.load_state_dict(ckpt["q2_target"])
            if not allow_transfer:
                self.q_opt.load_state_dict(ckpt["q_opt"])
                self.pi_opt.load_state_dict(ckpt["pi_opt"])
                self.updates = int(ckpt.get("updates", 0))
                self.epsilon = float(ckpt.get("epsilon", self.epsilon))
                # Resuming continues the learned temperature. A checkpoint
                # from before ALPHA_AUTO has none and keeps ALPHA_START; a
                # fixed-alpha run always uses ALPHA_START.
                if self.alpha_auto and "log_alpha" in ckpt:
                    with torch.no_grad():
                        self.log_alpha.fill_(float(ckpt["log_alpha"]))
                    self.alpha = self.log_alpha.detach().exp()
                    if ckpt.get("alpha_opt") is not None:
                        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        return ckpt


def discounted_return(step_rewards: torch.Tensor, hold: torch.Tensor,
                      g_step: float) -> torch.Tensor:
    """sum_{i<k} g^i r_i for each row, k = hold; entries past k are zero."""
    A = step_rewards.shape[1]
    powers = torch.pow(torch.full((A,), float(g_step),
                                  device=step_rewards.device),
                       torch.arange(A, device=step_rewards.device,
                                    dtype=step_rewards.dtype))
    valid = (torch.arange(A, device=step_rewards.device)[None, :]
             < hold[:, None]).to(step_rewards.dtype)
    return (step_rewards * powers[None, :] * valid).sum(1)


def permute_robots(batch):
    """Shuffle robot slots per sample, consistently across every joint
    tensor, and remap the actor's agent index."""
    mask = batch["mask"]
    B, N = mask.shape
    dev = mask.device
    perms = torch.argsort(torch.rand(B, N, device=dev), dim=1)

    def perm(x):
        idx = perms
        while idx.ndim < x.ndim:
            idx = idx.unsqueeze(-1)
        return x.gather(1, idx.expand(*x.shape[:2], *x.shape[2:]))

    out = dict(batch)
    out["obs"] = {k: (perm(v) if k in OBS_KEYS else v)
                  for k, v in batch["obs"].items()}
    out["next_obs"] = {k: (perm(v) if k in OBS_KEYS else v)
                       for k, v in batch["next_obs"].items()}
    out["mask"] = perm(mask)
    out["next_mask"] = perm(batch["next_mask"])
    out["action"] = perm(batch["action"])
    rc = batch.get("robot_penalties")
    if rc is not None:
        # (B, K, N): robots on the last axis.
        out["robot_penalties"] = rc.gather(
            2, perms.unsqueeze(1).expand(B, rc.shape[1], N))
    inv = torch.argsort(perms, dim=1)
    out["agent_index"] = inv.gather(1, batch["agent_index"].unsqueeze(1)
                                    ).squeeze(1)
    return out


def make_value_fn(agent: SACAgent):
    """V(s) for the curriculum's MaxMC score, with the same team definition:
    min over critics of the team-mean Q at a ~ pi, minus alpha times the
    team-mean log-probability.

    Takes a list of (static layers, record window) samples, the form the
    trainer hands the curriculum.
    """
    from learn.replay import joint_observation

    def value_fn(samples) -> np.ndarray:
        statics = [s for s, _ in samples]
        windows = [w for _, w in samples]
        joint = joint_observation(agent.cfg, statics, windows)
        return value_of(agent, {k: joint[k] for k in OBS_KEYS
                                + ("priv", "priv_scalars")},
                        joint["mask"])

    return value_fn


def value_of(agent: SACAgent, obs: Dict[str, np.ndarray],
             mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = to_torch(obs, agent.device)
        m = torch.as_tensor(mask, dtype=torch.float32, device=agent.device)
        a, logp = agent._per_robot_actions(t, m)
        v = torch.min(team_value(agent.q1(t, a, m), m),
                      team_value(agent.q2(t, a, m), m)) \
            - agent.alpha * team_value(logp, m)
        return v.cpu().numpy()
