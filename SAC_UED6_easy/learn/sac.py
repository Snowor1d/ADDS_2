"""SAC with a centralised critic and a shared decentralised actor, learning
from each robot's own decisions (docs/async_decisions.md).

  * A transition is one robot's decision: from the frame at which robot j
    decided to the frame of its next decision (or the end of the episode).
    Its reward is j's learning reward over the steps between (the team's
    task terms, plus under PER_ROBOT_VERSIONS j's own robot-charged terms),
    discounted per step, and its bootstrap is discounted by gamma_step ** k
    for the k steps it lasted.
  * The critic sees every robot's current action: j's new decision and what
    its teammates are doing then. Its per-robot output Q_j is fitted to j's
    own target.
  * The target at j's next decision frame re-draws, from the current policy,
    the actions of the robots that decide there (j among them) and keeps the
    others' current actions: they are still carrying out decisions taken
    earlier. The entropy bonus is j's own.
  * The actor is trained only for robots that decided at the sampled frame,
    each against its own value, with its teammates at the actions they had
    there (ACTOR_TEAMMATE_ACTIONS="current" re-draws only the teammates that
    decided at that frame too).
  * Exploration draws the full action, move, speed and mode.
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

    def _per_robot_hybrid(self, obs, mask):
        B, N = mask.shape
        flat = {k: obs[k].reshape(B * N, *obs[k].shape[2:]) for k in OBS_KEYS}
        values = self.policy.sample_hybrid(*(flat[k] for k in OBS_KEYS))
        return tuple(v.reshape(B, N, *v.shape[1:]) for v in values)

    def _expected_next_value(self, obs, mask, held, deciding, robot):
        """(B,) soft value of robot `robot` at its next decision frame.

        The robots in `deciding` take new actions from the current policy:
        their modes are enumerated exactly (a joint categorical expectation)
        and their continuous parts sampled per mode; the others keep `held`,
        the actions they are carrying out. The clipped double-Q minimum is
        taken inside the expectation, and the entropy bonus is the robot's
        own.
        """
        cont, logpc, probs, logpm = self._per_robot_hybrid(obs, mask)
        B, N, K = probs.shape
        C = cont.shape[-1]
        dec = deciding.bool() & mask.bool()
        rows = torch.arange(B, device=mask.device)
        active = torch.nonzero(dec.any(dim=0), as_tuple=False).flatten().tolist()
        expected_q = cont.new_zeros(B)
        critics = (self.q1_target, self.q2_target)
        features = [q.encode_observation(obs) if isinstance(q, CentralizedCritic)
                    else None for q in critics]
        for modes in product(range(K), repeat=len(active)):
            indices = torch.zeros(B, N, dtype=torch.long, device=mask.device)
            for j, mode in zip(active, modes):
                indices[:, j] = mode
            one_hot = F.one_hot(indices, K).to(cont.dtype)
            # Each deciding robot's move drawn for the mode it is enumerated in.
            move = cont.gather(2, indices[:, :, None, None].expand(B, N, 1, C)
                               ).squeeze(2)
            new = torch.cat((move, one_hot), dim=-1)
            action = torch.where(dec.unsqueeze(-1), new, held)
            selected = probs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            factors = torch.ones_like(selected)
            for j in active:
                # A slot that keeps its action (or is padding) in some sample
                # is still enumerated there: divide out its K identical
                # copies rather than overcounting.
                factors[:, j] = torch.where(dec[:, j], selected[:, j],
                                            torch.full_like(selected[:, j], 1.0 / K))
            weight = factors.prod(-1)
            q1, q2 = [q.values_from_features(f, action, mask) if f is not None
                      else q(obs, action, mask) for q, f in zip(critics, features)]
            expected_q += weight * torch.min(q1[rows, robot], q2[rows, robot])
        expected_logp = (probs * (logpc + logpm)).sum(-1)[rows, robot]
        return expected_q - self.alpha * expected_logp

    def _sampled_next_value(self, obs, mask, held, deciding, robot):
        """The legacy estimator's target: one sampled action per deciding
        robot instead of the mode expectation."""
        B = mask.shape[0]
        rows = torch.arange(B, device=mask.device)
        a2, logp2 = self._per_robot_actions(obs, mask)
        dec = (deciding.bool() & mask.bool()).unsqueeze(-1)
        action = torch.where(dec, a2, held)
        q1 = self.q1_target(obs, action, mask)[rows, robot]
        q2 = self.q2_target(obs, action, mask)[rows, robot]
        return torch.min(q1, q2) - self.alpha * logp2[rows, robot]

    def robot_targets(self, batch) -> torch.Tensor:
        """(B,) y = R_j + gamma_step^k (1 - terminal) V_j(s'), R_j robot j's
        discounted learning reward over its decision (from the replay) and
        V_j its soft value at its next decision frame."""
        with torch.no_grad():
            args = (batch["next_obs"], batch["next_mask"], batch["next_action"],
                    batch["next_decided"], batch["agent_index"])
            if self.cfg.SAC_MODE_ESTIMATOR == "expectation":
                v2 = self._expected_next_value(*args)
            else:
                v2 = self._sampled_next_value(*args)
            g_step = self.gamma ** (1.0 / max(1, int(self.cfg.ACTION_SCALE)))
            boot = (torch.pow(torch.full_like(batch["hold"], g_step),
                              batch["hold"]) * (1.0 - batch["terminal"]))
            return batch["ret"] + boot * v2

    def update(self, batch_np) -> Dict[str, float]:
        """One gradient step on a sampled batch (ReplayBuffer.sample)."""
        dev = self.device
        f32 = lambda k: torch.as_tensor(batch_np[k], device=dev, dtype=torch.float32)
        batch = {
            "obs": to_torch(batch_np["obs"], dev),
            "next_obs": to_torch(batch_np["next_obs"], dev),
            "mask": f32("mask"), "next_mask": f32("next_mask"),
            "action": f32("action"), "next_action": f32("next_action"),
            "decided": torch.as_tensor(batch_np["decided"], device=dev, dtype=torch.bool),
            "next_decided": torch.as_tensor(batch_np["next_decided"], device=dev,
                                            dtype=torch.bool),
            "ret": f32("ret"), "hold": f32("hold"), "terminal": f32("terminal"),
            "agent_index": torch.as_tensor(batch_np["agent_index"], device=dev,
                                           dtype=torch.long),
        }
        # Robot-order augmentation: slot identity must not carry meaning.
        batch = permute_robots(batch)
        obs, mask, act = batch["obs"], batch["mask"], batch["action"]
        rows = torch.arange(mask.shape[0], device=dev)
        j = batch["agent_index"]
        y = self.robot_targets(batch)
        q1 = self.q1(obs, act, mask)[rows, j]
        q2 = self.q2(obs, act, mask)[rows, j]
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad()
        loss_q.backward()
        self.q_opt.step()

        # Actor: the robots that decided at the sampled frame (all of them, or
        # only the sampled one, ACTOR_UPDATE_ROBOTS), each against its own
        # value, with its teammates at their actions there.
        loss_pi, logp = self._actor_loss(obs, act, mask, batch["decided"], j)
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

    def _robot_q_min(self, obs, actions, mask, j):
        """(B,) robot j's own value, the smaller of the two critics."""
        return torch.min(self._actor_q(self.q1, obs, actions, mask)[:, j],
                         self._actor_q(self.q2, obs, actions, mask)[:, j])

    def _actor_q(self, critic, obs, actions, mask):
        features = getattr(self, "_actor_critic_features", {}).get(id(critic))
        if features is not None:
            return critic.values_from_features(features, actions, mask)
        return critic(obs, actions, mask)

    def _actor_chosen(self, mask, decided, agent_index):
        """(B, N) robots whose actions the actor loss re-draws."""
        B, N = mask.shape
        real = mask > 0
        if self.cfg.ACTOR_UPDATE_ROBOTS == "one":
            chosen = torch.zeros(B, N, dtype=torch.bool, device=mask.device)
            chosen.scatter_(1, agent_index.clamp(0, N - 1)[:, None], True)
            return chosen & real
        return decided.bool() & real

    def _actor_base(self, obs, act, mask, decided):
        """Teammates' actions during the actor update: as stored, or for the
        teammates that decided at this frame too, re-drawn from the current
        policy (ACTOR_TEAMMATE_ACTIONS)."""
        if self.cfg.ACTOR_TEAMMATE_ACTIONS != "current":
            return act
        with torch.no_grad():
            cur, _ = self._per_robot_actions(obs, mask)
        return torch.where(decided.bool().unsqueeze(-1), cur, act)

    def _actor_loss(self, obs, act, mask, decided, agent_index):
        """(loss, log-probabilities of the differentiated actions, flat over
        the chosen robots). The critics are frozen for the duration."""
        critics = list(self.q1.parameters()) + list(self.q2.parameters())
        for p in critics:
            p.requires_grad_(False)
        try:
            self._actor_critic_features = {
                id(q): q.encode_observation(obs) for q in (self.q1, self.q2)
                if isinstance(q, CentralizedCritic)}
            chosen = self._actor_chosen(mask, decided, agent_index)
            base = self._actor_base(obs, act, mask, decided)
            if self.cfg.SAC_MODE_ESTIMATOR == "expectation":
                return self._expected_actor_loss(obs, base, mask, chosen)
            return self._sampled_actor_loss(obs, base, mask, chosen)
        finally:
            self._actor_critic_features = {}
            for p in critics:
                p.requires_grad_(True)

    def _expected_actor_loss(self, obs, base, mask, chosen):
        """Enumerate each chosen robot's modes against fixed teammates."""
        cont, logpc, probs, logpm = self._per_robot_hybrid(obs, mask)
        B, N, K = probs.shape
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
                mixed[:, j] = torch.cat((cont[:, j, mode], one_hot), dim=-1)
                q = self._robot_q_min(obs, mixed, mask, j)
                expected += probs[:, j, mode] * (
                    self.alpha * (logpc[:, j, mode] + logpm[:, j, mode]) - q)
            total += (expected * valid).sum()
        expected_logp = (probs * (logpc + logpm)).sum(-1)
        return total / chosen.sum().clamp(min=1), expected_logp[chosen]

    def _sampled_actor_loss(self, obs, base, mask, chosen):
        """The legacy estimator: one sampled action per chosen robot."""
        new_all, logp_all = self._per_robot_actions(obs, mask)
        B, N = mask.shape
        total = new_all.new_zeros(())
        for j in range(N):
            valid = chosen[:, j]
            if not bool(valid.any()):
                continue
            mixed = base.clone()
            mixed[:, j] = new_all[:, j]
            term = self.alpha * logp_all[:, j] - self._robot_q_min(obs, mixed, mask, j)
            total = total + (term * valid.float()).sum()
        return total / chosen.sum().clamp(min=1), logp_all[chosen]

    def _soft_update(self, net, target):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(self.tau).add_((1 - self.tau) * p)

    # ---------------------------------------------------------- persistence

    def schema(self) -> Dict[str, str]:
        return self.cfg.schema_versions()

    def _network(self) -> Dict[str, str]:
        """What a checkpoint's weights only load into."""
        return {"encoder": self.cfg.NET_ENCODER, "size": self.cfg.NET_SIZE,
                "actor_cont_heads": self.cfg.ACTOR_CONT_HEADS}

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        payload = {
            "schema": self.schema(),
            "observation_schema": self.cfg.observation_schema(),
            "experiment_id": self.cfg.EXPERIMENT_ID,
            "config_fingerprint": self.cfg.fingerprint,
            "critic_privileged": bool(self.cfg.CRITIC_PRIVILEGED_CROWD),
            "network": self._network(),
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
        here = self._network()
        if stored_net is not None:
            # Checkpoints from before ACTOR_CONT_HEADS have shared heads.
            stored_net = {"actor_cont_heads": "shared", **stored_net}
        if stored_net is not None and stored_net != here:
            raise SchemaMismatch(
                f"checkpoint {path} holds a {stored_net['encoder']}/"
                f"{stored_net['size']} network with "
                f"{stored_net['actor_cont_heads']} actor heads; this run "
                f"builds {here['encoder']}/{here['size']} with "
                f"{here['actor_cont_heads']}. Set NET_ENCODER, NET_SIZE and "
                "ACTOR_CONT_HEADS to match it.")
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
    tensor, and remap the sampled robot's index."""
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
    for key in ("obs", "next_obs"):
        out[key] = {k: (perm(v) if k in OBS_KEYS else v)
                    for k, v in batch[key].items()}
    for key in ("mask", "next_mask", "action", "next_action", "decided",
                "next_decided"):
        if key in batch:
            out[key] = perm(batch[key])
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
