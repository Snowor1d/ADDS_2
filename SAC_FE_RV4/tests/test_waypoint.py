"""ROBOT_ACTION_MODE = "waypoint": action decoding, the follower on real maps,
variable-length transitions in the replay, the networks and config plumbing."""

import json
import math
import os
import random
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from configs import resolve_config  # noqa: E402


def _cfg(**over):
    return resolve_config(overrides=over, use_env=False)


WP = dict(ROBOT_ACTION_MODE="waypoint")


def _env(cfg, map_num=1504, crowd=5, seed=3):
    from learn.rollout import EnvSpec, build_env
    return build_env(EnvSpec(map_num=map_num, crowd=crowd, augmentation="identity", seed=seed), cfg, robot="Q")


class DecodeTest(unittest.TestCase):
    def test_decode(self):
        from learn.waypoint import decode
        cfg = _cfg(**WP)
        r, vmax = cfg.ROBOT_WAYPOINT_RANGE_M, cfg.ROBOT_WAYPOINT_SPEED_MAX
        self.assertEqual(decode([2, -2, -2], cfg), (r, -r, 0.0))
        self.assertEqual(decode([0, 0, 2], cfg), (0.0, 0.0, vmax))
        self.assertEqual(decode([1, 0.5, 0], cfg), (r / 2, r / 4, vmax / 2))


class FollowerTest(unittest.TestCase):
    def test_reaches_targets_on_test_maps(self):
        """Random targets from wherever the robot is, on the handmade test
        maps, small crowd: a decision should end by arrival far more often
        than by the time limit or a block."""
        from learn.waypoint import WaypointFollower
        cfg = _cfg(**WP)
        D = int(cfg.ACTION_SCALE)
        rng = np.random.default_rng(0)
        ends = {"arrived": 0, "blocked_or_timeout": 0}
        for m in (1501, 1503, 1504, 1507, 1508):
            env = _env(cfg, map_num=m)
            fol = WaypointFollower(env, cfg)
            for _ in range(6):
                fol.set(np.r_[rng.uniform(-1.5, 1.5, 2), 2.0])   # full speed
                for step in range(D * fol.max_ticks):
                    env.robot.receive_action(list(fol.command()))
                    env.step()
                    if step % D == D - 1 and fol.end_of_tick(False):
                        break
                ends["arrived" if fol.arrived() else "blocked_or_timeout"] += 1
        rate = ends["arrived"] / sum(ends.values())
        self.assertGreater(rate, 0.8, ends)

    def test_projection_moves_targets_out_of_obstacles(self):
        from learn.waypoint import WaypointFollower
        from shapely.geometry import Point
        cfg = _cfg(**WP)
        env = _env(cfg)
        fol = WaypointFollower(env, cfg)
        poly = env._obstacle_polys[0]
        c = poly.representative_point()
        t = fol._project(c.x, c.y)
        self.assertFalse(poly.contains(Point(*t)))
        t2 = fol._project(-10.0, -10.0)
        self.assertTrue(0 < t2[0] < env.width and 0 < t2[1] < env.height)


class RunEpisodeTest(unittest.TestCase):
    def test_waypoint_episode_runs(self):
        from learn.rollout import eval_step_cap, run_episode
        cfg = _cfg(**WP)

        class RandomWaypoint:
            needs_obs = True
            action_mode = "waypoint"
            decision_interval = int(cfg.ACTION_SCALE)

            def reset(self, env, ob):
                self.rng = np.random.default_rng(1)

            def act(self, env, ob, stack, state5):
                return self.rng.uniform(-2, 2, 3)

        env = _env(cfg, crowd=30)
        res = run_episode(env, RandomWaypoint(), cfg, max_steps=600)
        self.assertGreater(res["decisions"], 0)
        # decisions hold for one or more 2 s windows, never less
        self.assertLessEqual(res["decisions"], res["steps"] / cfg.ACTION_SCALE + 1)
        self.assertGreater(res["robot_path_m"], 1.0)


class ReplayTest(unittest.TestCase):
    def _replay(self, action_dim=3):
        from learn.replay import FrameReplay
        return FrameReplay(64, 5, 7, 4, False, action_dim=action_dim, frames_per_transition=8.0)

    def _frame(self, v):
        return (np.full((5, 5), v, np.uint8), np.full((7, 7), v, np.uint8), None)

    def test_variable_length_discounts_and_stacks(self):
        rp = self._replay()
        cur = rp.begin_episode(*self._frame(1)[:2])
        s = np.zeros(5, np.float32)
        # decision 1: 3 windows (frames 2, 3, 4); decision 2: 6 windows (only
        # the last 4 frames, 7..10, are sent); decision 3: 1 window (frame 11)
        rp.add_multi(cur, s, np.ones(3), 1.0, [self._frame(v) for v in (2, 3, 4)], s, False, False, 3)
        rp.add_multi(cur, s, np.ones(3), 2.0, [self._frame(v) for v in (7, 8, 9, 10)], s, False, False, 6)
        rp.add_multi(cur, s, np.ones(3), 4.0, [self._frame(11)], s, True, True, 1)
        g = 0.9
        slots = np.array([0, 1, 2])
        b1 = rp.batch_from_slots(slots, 1, g, True)
        np.testing.assert_allclose(b1["discount"].numpy(), [g ** 3, g ** 6, g ** 1], rtol=1e-6)
        b3 = rp.batch_from_slots(np.array([0]), 3, g, True)
        self.assertAlmostEqual(float(b3["reward"][0]), 1.0 + g ** 3 * 2.0 + g ** 9 * 4.0, places=5)
        self.assertAlmostEqual(float(b3["discount"][0]), g ** 10, places=6)
        # stacks, newest first (pixel value = frame label)
        st = lambda t, i: [int(t[i, c, 0, 0] * 255 + 0.5) for c in range(4)]
        self.assertEqual(st(b1["ego"], 0), [1, 1, 1, 1])
        self.assertEqual(st(b1["ego2"], 0), [4, 3, 2, 1])
        self.assertEqual(st(b1["ego"], 1), [4, 3, 2, 1])
        self.assertEqual(st(b1["ego2"], 1), [10, 9, 8, 7])
        self.assertEqual(st(b1["ego"], 2), [10, 9, 8, 7])
        self.assertEqual(st(b1["ego2"], 2), [11, 10, 9, 8])

    def test_velocity_add_is_unchanged(self):
        rp = self._replay(action_dim=2)
        cur = rp.begin_episode(*self._frame(1)[:2])
        s = np.zeros(5, np.float32)
        for v in range(2, 8):
            rp.add(cur, s, np.ones(2), float(v), *self._frame(v), s, False, v == 7)
        b = rp.batch_from_slots(np.arange(6), 1, 0.99, True)
        # the next state's stack is the new frame then the state's three newest
        expect = np.concatenate([rp.s2_f0[:6, None], rp.s_f[:6, :-1]], axis=1)
        np.testing.assert_array_equal(rp.s2_f[:6], expect)
        np.testing.assert_array_equal(b["discount"].numpy(), np.full(6, np.float32(0.99)))
        self.assertEqual(b["action"].shape[1], 2)


class NetworkTest(unittest.TestCase):
    def test_waypoint_networks_and_update(self):
        from learn.obs import d4_action
        from learn.sac import SACLearner
        cfg = _cfg(NET_SIZE="s", **WP)
        learner = SACLearner(cfg, "cpu")
        B, C = 8, cfg.obs_channels()
        e = torch.rand(B, C, cfg.EGO_MAP_SIZE, cfg.EGO_MAP_SIZE)
        g = torch.rand(B, C, cfg.DOWNSAMPLE_MAP_SIZE, cfg.DOWNSAMPLE_MAP_SIZE)
        st = torch.rand(B, 5)
        a, logp = learner.actor.sample(e, g, learner._sa(st))
        self.assertEqual(tuple(a.shape), (B, 3))
        self.assertEqual(tuple(learner.critic(e, g, learner._sc(st), a).shape), (cfg.CRITIC_ENSEMBLE, B))
        batch = {"ego": e, "glob": g, "state": st, "action": a.detach(), "reward": torch.rand(B),
                 "discount": torch.full((B,), 0.9), "done": torch.zeros(B),
                 "ego2": e, "glob2": g, "state2": st}
        out = learner.update(batch)
        self.assertTrue(math.isfinite(out["train/q_loss"]))
        # D4: the offset turns, the speed does not
        x = torch.tensor([[1.0, 0.5, -1.2]])
        for k in range(8):
            y = d4_action(x, k)
            self.assertAlmostEqual(float(y[0, 2]), -1.2)
            self.assertAlmostEqual(float(torch.linalg.norm(y[0, :2])), float(torch.linalg.norm(x[0, :2])), places=6)


class UedScoreTest(unittest.TestCase):
    def test_per_transition_discounts(self):
        from ued import score
        r = np.array([1.0, 2.0, 4.0])
        d = np.array([0.5, 0.25, 0.9])            # gamma ** windows of each transition
        self.assertAlmostEqual(score.discounted_return(r, d), 1.0 + 0.5 * (2.0 + 0.25 * 4.0))
        self.assertAlmostEqual(score.discounted_return(r, 0.5), score.discounted_return(r, np.full(3, 0.5)))
        v = np.array([0.0, 1.0, 2.0])
        a = score.pvl(r, v, 0.0, True, d, 1.0)
        # lambda = 1: advantages are the discounted returns minus the values
        g = [score.discounted_return(r[t:], d[t:]) for t in range(3)]
        self.assertAlmostEqual(a, float(np.mean(np.maximum(np.array(g) - v, 0.0))))


class ConfigTest(unittest.TestCase):
    def test_velocity_fingerprint_ignores_waypoint_settings(self):
        base = _cfg().fingerprint()
        self.assertEqual(_cfg(ROBOT_WAYPOINT_RANGE_M=12.0).fingerprint(), base)
        self.assertNotEqual(_cfg(**WP).fingerprint(), base)
        self.assertNotEqual(_cfg(ROBOT_WAYPOINT_RANGE_M=12.0, **WP).fingerprint(), _cfg(**WP).fingerprint())

    def test_old_checkpoints_load_as_velocity(self):
        from learn.policy_io import policy_settings
        self.assertEqual(policy_settings({"format": "rv4", "config": {"NET_SIZE": "l"}})["ROBOT_ACTION_MODE"], "velocity")
        self.assertEqual(policy_settings({"format": "rv4", "config": {"ROBOT_ACTION_MODE": "waypoint"}})["ROBOT_ACTION_MODE"], "waypoint")

    def test_decision_length_must_fit_the_clock(self):
        from configs import ConfigError
        with self.assertRaises(ConfigError):
            _cfg(ROBOT_DECISION_MAX_S_WAYPOINT=5.0, **WP)
        self.assertEqual(_cfg(**WP).decision_max_ticks(), 10)
        with self.assertRaises(ConfigError):
            _cfg(GAMMA_START=0.95, GAMMA_END=0.99, **WP)
        self.assertEqual(_cfg().decision_max_ticks(), 1)


if __name__ == "__main__":
    unittest.main()
