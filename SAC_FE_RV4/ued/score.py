"""Per-episode regret proxies from the SAC critic.

    MaxMC  mean_t (R_max - V(s_t)), R_max the best discounted return seen on
           the level (Jiang et al. 2021b, "Replay-Guided Adversarial
           Environment Design").
    PVL    mean_t max(0, A_t), A_t the GAE(gamma, lambda) advantage built from
           one-step TD errors (Jiang et al. 2021a, "Prioritized Level
           Replay").

V(s) comes from learn/sac.SACLearner.state_values on the episode's own
transitions, read back from the replay. Both are computed per decision
(gamma per decision, as the critic is trained).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _discounts(gamma, T: int) -> np.ndarray:
    """gamma per transition: a number, or one per transition (waypoint
    decisions of different lengths are discounted by gamma ** windows)."""
    return np.broadcast_to(np.asarray(gamma, dtype=np.float64), (T,))


def discounted_return(rewards: np.ndarray, gamma) -> float:
    g = 0.0
    d = _discounts(gamma, len(rewards))
    for t in range(len(rewards) - 1, -1, -1):
        g = float(rewards[t]) + float(d[t]) * g
    return g


def maxmc(values: np.ndarray, best_return: float) -> float:
    return float(np.mean(best_return - values))


def pvl(rewards: np.ndarray, values: np.ndarray, next_value_last: float, last_terminal: bool,
        gamma: float, lam: float) -> float:
    """values[t] = V(s_t) for the episode's consecutive decisions;
    next_value_last = V(s_T) after the last one (0 if it ended the task)."""
    T = len(rewards)
    if T == 0:
        return 0.0
    v_next = np.empty(T)
    v_next[:-1] = values[1:]
    v_next[-1] = 0.0 if last_terminal else next_value_last
    d = _discounts(gamma, T)
    delta = rewards + d * v_next - values
    adv = np.empty(T)
    run = 0.0
    for t in range(T - 1, -1, -1):
        run = delta[t] + d[t] * lam * run
        adv[t] = run
    return float(np.mean(np.maximum(adv, 0.0)))
