"""The per-decision reward, as SAC_FE_RV3 computed it.

Evaluated once per decision, at the last step of the ACTION_SCALE window (or
when the task ends inside it), from the model's reward functions and the
REWARD_* weights. Two options (configs/training/common.py):

    REWARD_COLLISION_MODE  "last_step" (RV3): the wall-collision penalty
                           looks at the window's last step only; "sum": at
                           every step of the window.
    REWARD_SCALE_INVARIANT the distance-sum term divided by side / 100 m.

Two exit-distance terms, independent of each other, both over agent.danger
(whose metric is the environment setting DANGER_DISTANCE: straight line as in
RV3, or walking distance; model.danger_distance):

    reward_b_1  REWARD_B_1, RV3's penalty: -REWARD_B_1 * (sum of the exit
                distances of the people still inside), every decision.
    reward_b_2  REWARD_B_2, potential-based shaping (Ng, Harada and Russell
                1999): Phi(s) = -K * (the same sum, measured now) and
                reward_b_2 = gamma * Phi(s') - Phi(s), with gamma the
                learner's per-decision discount and Phi = 0 once everyone is
                out. Over an episode the terms telescope to
                gamma^T Phi(s_T) - Phi(s_0), so the shaping cannot change which
                policy is optimal; it pays K per person-metre of progress
                toward an exit as it happens. It needs the potential of the
                state the episode starts in: call reset_reward_state(env, cfg)
                once per episode, before the first decision.
"""

from __future__ import annotations

from typing import Dict, Tuple

COMPONENTS = ("reward_a", "reward_b_1", "reward_b_2", "reward_c", "reward_d", "reward_e", "reward_f",
              "reward_g", "reward_h", "reward_i", "reward_j", "reward_k", "reward_l",
              "reward_fixed", "reward_finished_bonus")


def potential_k(env, cfg) -> float:
    """K of reward_b_2, per person-metre. REWARD_B_2 = "auto": REWARD_A spread
    over the crowd, per metre walked in one decision at the mean speed, so
    that Phi is the reward_a cost still to pay if everyone walked straight
    out. A number is K itself."""
    if cfg.REWARD_B_2 != "auto":
        return float(cfg.REWARD_B_2)
    per_decision_m = float(cfg.AGENT_SPEED_MEAN) * int(cfg.ACTION_SCALE) * float(cfg.AGENT_TIME_STEP)
    return float(cfg.REWARD_A) / (max(1, int(env.total_agents)) * per_decision_m)


def potential(env, cfg) -> float:
    """Phi(s) = -K * sum over the people still inside (types 0, 1, 2) of
    their distance to the nearest exit (model.danger_distance, so the
    DANGER_DISTANCE metric), measured now."""
    total = 0.0
    for ag in env.crowds:
        if ag.type in (0, 1, 2) and not ag.dead:
            total += float(env.danger_distance(ag.xy))
    if total != total or total == float("inf"):
        raise RuntimeError(f"potential is not finite ({total}): a map without exits?")
    return -potential_k(env, cfg) * total


def reset_reward_state(env, cfg) -> None:
    """Start of an episode: remember Phi(s_0) for the first decision's
    shaping term. No effect while REWARD_B_2 is 0."""
    if cfg.REWARD_B_2:
        env._rv4_phi = potential(env, cfg)


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
    if c.REWARD_B_1:
        rb = env.reward_based_all_agents_danger() * c.REWARD_B_1
        if c.REWARD_SCALE_INVARIANT:
            rb /= max(env.width, env.height) / 100.0
        comp["reward_b_1"] = rb
    if c.REWARD_B_2:
        prev = getattr(env, "_rv4_phi", None)
        if prev is None:
            raise RuntimeError("REWARD_B_2 (potential shaping): call reset_reward_state(env, cfg) "
                               "when the episode starts")
        phi = 0.0 if finished else potential(env, cfg)
        comp["reward_b_2"] = float(c.GAMMA_START) * phi - prev
        env._rv4_phi = phi
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
