"""SAC with a centralised team critic and a shared decentralised actor.

docs/outdoor_madrl_redesign.md section 5.1 and 5.4:

  * The critic's per-robot outputs are evaluated as the masked team mean.
    The critic loss, the Bellman target and the actor loss all use that one
    definition, and in all three the minimum over the two critics is taken
    after averaging over the team.
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
        self.gamma = float(cfg.GAMMA_START)
        self.tau = 0.995
        self.batch_size = int(cfg.BATCH_SIZE)
        self.alpha = torch.tensor(float(cfg.ALPHA_START), device=self.device)
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

    def targets(self, batch) -> torch.Tensor:
        """y = R + gamma_step^k (1 - terminal) [min_k Q_k,team(s', a')
        - alpha * mean_team log pi(a'|s')]."""
        cfg = self.cfg
        obs2 = batch["next_obs"]
        m2 = batch["next_mask"]
        with torch.no_grad():
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
            "hold": torch.as_tensor(batch_np["hold"], device=dev),
            "terminal": torch.as_tensor(batch_np["terminal"], device=dev),
            "agent_index": torch.as_tensor(batch_np["agent_index"],
                                           device=dev, dtype=torch.long),
        }
        # Robot-order augmentation: slot identity must not carry meaning.
        batch = permute_robots(batch)
        obs, mask, act = batch["obs"], batch["mask"], batch["action"]
        y = self.targets(batch)
        q1 = team_value(self.q1(obs, act, mask), mask)
        q2 = team_value(self.q2(obs, act, mask), mask)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # Actor: replace only the sampled robot's action.
        B, N = mask.shape
        idx = batch["agent_index"].clamp(0, N - 1)
        rows = torch.arange(B, device=dev)
        pick = {k: obs[k][rows, idx] for k in OBS_KEYS}
        new_a, logp = self.policy.sample_action(pick["ego"], pick["mid"],
                                                pick["glob"], pick["state"])
        mixed = act.clone()
        mixed[rows, idx] = new_a
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad_(False)
        qn = torch.min(team_value(self.q1(obs, mixed, mask), mask),
                       team_value(self.q2(obs, mixed, mask), mask))
        for p in list(self.q1.parameters()) + list(self.q2.parameters()):
            p.requires_grad_(True)
        loss_pi = (self.alpha * logp - qn).mean()
        self.pi_opt.zero_grad()
        loss_pi.backward()
        self.pi_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self.updates += 1
        return {"train/loss_q": float(loss_q.item()),
                "train/loss_pi": float(loss_pi.item()),
                "train/q_team": float(q1.mean().item()),
                "train/target": float(y.mean().item()),
                "train/entropy": float(-logp.mean().item()),
                "train/alpha": float(self.alpha.item())}

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
        return value_of(agent, {k: joint[k] for k in OBS_KEYS + ("priv",)},
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
