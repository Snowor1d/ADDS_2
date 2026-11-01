"""reward_b_1 (RV3 penalty) and reward_b_2 (potential-based shaping)."""

import json
import math
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from configs import ConfigError, resolve_config  # noqa: E402
from learn import rewards as R  # noqa: E402


def _cfg(**over):
    from tests.reference import reference_config
    return reference_config(over)


def _env(seed=5, map_num=1504):
    from learn.rollout import EnvSpec, build_env
    return build_env(EnvSpec(map_num=map_num, crowd=30, augmentation="identity", seed=seed),
                     _cfg(), robot="Q")


def _run(env, cfg, steps, rng):
    """Random robot, rewards every decision as the worker computes them.
    Returns the per-decision reward_b terms and whether the episode ended."""
    D = int(cfg.ACTION_SCALE)
    R.reset_reward_state(env, cfg)
    phi0 = getattr(env, "_rv4_phi", None)
    terms, finished = [], False
    for step in range(steps):
        if step % D == 0:
            env.robot.receive_action([rng.uniform(-2, 2), rng.uniform(-2, 2)])
        env.step()
        finished = env.alived_agents() < 1
        if step % D == D - 1 or finished:
            _, comp = R.decision_reward(env, cfg, int(bool(env.robot.collision_check)), 0,
                                        finished, step, steps)
            terms.append(comp["reward_b_2"])
        if finished:
            break
    return phi0, terms, finished


class PotentialRewardTest(unittest.TestCase):
    def test_auto_k(self):
        cfg = _cfg(REWARD_B_2="auto")
        env = _env()
        self.assertAlmostEqual(R.potential_k(env, cfg), 2.0 / (30 * 1.5 * 2.0), places=12)
        self.assertEqual(R.potential_k(env, _cfg(REWARD_B_2=0.003)), 0.003)

    def test_shaping_telescopes(self):
        cfg = _cfg(REWARD_B_1=0, REWARD_B_2="auto")
        env = _env()
        phi0, terms, finished = _run(env, cfg, 600, random.Random(1))
        g = float(cfg.GAMMA_START)
        disc = sum(g ** k * f for k, f in enumerate(terms))
        phiT = 0.0 if finished else env._rv4_phi
        self.assertAlmostEqual(disc, g ** len(terms) * phiT - phi0, places=6)

    def test_progress_toward_the_exit_pays(self):
        cfg = _cfg(REWARD_B_2="auto", GAMMA_START=1.0, GAMMA_END=1.0)
        env = _env()
        env.step()
        R.reset_reward_state(env, cfg)
        a = next(x for x in env.crowds if not x.dead)
        before = env.danger_distance(a.xy)
        # one person moved 5 m along the straight line to the nearest exit
        _, q, _ = env.nearest_exit(a.xy)
        v = np.array(q) - np.array(a.xy[:2])
        a.xy = list(np.array(a.xy[:2]) + 5.0 * v / np.linalg.norm(v))
        _, comp = R.decision_reward(env, cfg, 0, 0, False, 1, 100)
        gained = before - env.danger_distance(a.xy)
        self.assertGreater(gained, 0)
        self.assertAlmostEqual(comp["reward_b_2"], R.potential_k(env, cfg) * gained, places=9)

    def test_defaults_are_rv3(self):
        cfg = _cfg()
        self.assertEqual((cfg.REWARD_B_1, cfg.REWARD_B_2), (0.003, 0))
        env = _env()
        R.reset_reward_state(env, cfg)
        env.step()
        _, comp = R.decision_reward(env, cfg, 0, 0, False, 1, 100)
        self.assertEqual(comp["reward_b_1"], env.reward_based_all_agents_danger() * cfg.REWARD_B_1)
        self.assertEqual(comp["reward_b_2"], 0.0)

    def test_both_terms_together(self):
        cfg = _cfg(REWARD_B_2="auto")
        env = _env()
        R.reset_reward_state(env, cfg)
        for _ in range(4):
            env.step()
        _, comp = R.decision_reward(env, cfg, 0, 0, False, 3, 100)
        self.assertLess(comp["reward_b_1"], 0.0)
        self.assertNotEqual(comp["reward_b_2"], 0.0)

    def test_old_name_still_overrides(self):
        self.assertEqual(_cfg(REWARD_B=0.005).REWARD_B_1, 0.005)

    def test_bad_values_are_rejected(self):
        for bad in (-1, "half", True):
            with self.assertRaises(ConfigError):
                _cfg(REWARD_B_2=bad)

    def test_missing_reset_is_an_error(self):
        cfg = _cfg(REWARD_B_2="auto")
        env = _env()
        env.step()
        with self.assertRaises(RuntimeError):
            R.decision_reward(env, cfg, 0, 0, False, 1, 100)

    def test_gamma_schedule_is_rejected(self):
        with self.assertRaises(ConfigError):
            _cfg(REWARD_B_2="auto", GAMMA_START=0.95, GAMMA_END=0.99)


if __name__ == "__main__":
    unittest.main()
