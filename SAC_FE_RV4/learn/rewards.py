"""The per-decision reward, as SAC_FE_RV3 computed it.

Evaluated once per decision, at the last step of the ACTION_SCALE window (or
when the task ends inside it), from the model's reward functions and the
REWARD_* weights. Two options (configs/training/common.py):

    REWARD_COLLISION_MODE  "last_step" (RV3): the wall-collision penalty
                           looks at the window's last step only; "sum": at
                           every step of the window.
    REWARD_SCALE_INVARIANT the distance-sum term divided by side / 100 m.
"""

from __future__ import annotations

from typing import Dict, Tuple

COMPONENTS = ("reward_a", "reward_b", "reward_c", "reward_d", "reward_e", "reward_f",
              "reward_g", "reward_h", "reward_i", "reward_j", "reward_k", "reward_l",
              "reward_fixed", "reward_finished_bonus")


def decision_reward(env, cfg, collision_last: int, collisions_in_window: int,
                    finished: bool, step: int, max_steps: int) -> Tuple[float, Dict[str, float]]:
    c = cfg
    comp = {k: 0.0 for k in COMPONENTS}
    if finished:
        comp["reward_finished_bonus"] = float(c.FINISHED_BONUS) * (1 - step / max_steps)
    if c.REWARD_K:
        hits = collision_last if c.REWARD_COLLISION_MODE == "last_step" else collisions_in_window
        comp["reward_k"] = -1.0 * float(hits) * float(c.REWARD_K)
    if c.REWARD_A:
        comp["reward_a"] = env.reward_based_alived() * c.REWARD_A
    if c.REWARD_B:
        rb = env.reward_based_all_agents_danger() * c.REWARD_B
        if c.REWARD_SCALE_INVARIANT:
            rb /= max(env.width, env.height) / 100.0
        comp["reward_b"] = rb
    if c.REWARD_C:
        comp["reward_c"] = env.reward_based_gain() * c.REWARD_C
    if c.REWARD_D:
        comp["reward_d"] = env.reward_penalty() * c.REWARD_D
    if c.REWARD_E:
        comp["reward_e"] = env.reward_based_evacuated_with_robot() * c.REWARD_E
    if c.REWARD_F:
        comp["reward_f"] = env.reward_based_distance_from_near_agents() * c.REWARD_F
    if c.REWARD_G:
        comp["reward_g"] = env.reward_based_distance_from_near_agent_gain() * c.REWARD_G
    if c.REWARD_H:
        comp["reward_h"] = env.reward_based_gain_with_time_bonus() * c.REWARD_H
    if c.REWARD_I:
        comp["reward_i"] = env.reward_based_alived_root() * c.REWARD_I
    if c.REWARD_J:
        comp["reward_j"] = env.reward_based_all_agents_danger_root() * c.REWARD_J
    if c.REWARD_L:
        comp["reward_l"] = env.reward_based_farthest_agent_distance() * c.REWARD_L
    comp["reward_fixed"] = float(c.REWARD_FIXED)
    total = float(sum(comp.values()))
    return total, comp
