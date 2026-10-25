"""Observation: RV3 equality and the D4 transforms used for augmentation."""

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
from learn import obs as O  # noqa: E402
from map_augmentation import transform_point  # noqa: E402


def _env(aug="identity", m=1500, seed=3):
    import model
    random.seed(seed)
    np.random.seed(seed)
    return model.FightingModel(30, 100, 100, model_num=m, robot="Q", augmentation=aug)


class ObservationTest(unittest.TestCase):
    def test_frames_equal_rv3(self):
        import torch.nn.functional as F
        cfg = resolve_config()
        env = _env()
        ob = O.ObservationBuilder(cfg, env)
        full = env.return_current_image(100, 100)
        np.testing.assert_array_equal(ob.full_image(), full)
        ego, glob, es = ob.frames()
        x = torch.from_numpy(full).float()[None, None]
        g = F.adaptive_max_pool2d(x, (50, 50))[0, 0].byte().numpy()
        np.testing.assert_array_equal(glob, g)
        rx, ry = env.robot.xy
        c = (int(np.clip(rx / 100 * 100, 0, 99)), int(np.clip(ry / 100 * 100, 0, 99)))
        np.testing.assert_array_equal(ego, O.ego_crop(full, c, 25))
        self.assertIsNone(es)

    def test_metric_ego_equals_normalized_at_100m(self):
        env = _env()
        a = O.ObservationBuilder(resolve_config(), env).frames()
        b = O.ObservationBuilder(resolve_config(overrides={"OBS_EGO_MODE": "metric"}), env).frames()
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])

    def test_static_channel(self):
        cfg = resolve_config(overrides={"OBS_STATIC_DISTANCE_CHANNEL": True})
        env = _env()
        ob = O.ObservationBuilder(cfg, env)
        d = ob.dist_full
        self.assertTrue((d[ob.static_full == O.WALL] == 0).all())
        self.assertEqual(int(d[ob.static_full == O.EXIT].min()), 255)
        st = O.StackedObservation(4)
        e, g, es = ob.frames()
        st.reset(e, g, es)
        ei, gi = st.network_input(ob.glob_static)
        self.assertEqual(ei.shape, (5, 25, 25))
        self.assertEqual(gi.shape, (5, 50, 50))

    def test_d4_image_matches_world_transform(self):
        """Rendering a transformed map equals transforming the rendering."""
        cfg = resolve_config()
        base = O.ObservationBuilder(cfg, _env("identity")).static_full
        for k, name in enumerate(O.D4_NAMES):
            env = _env(name)
            img = O.ObservationBuilder(cfg, env).static_full
            t = O.d4_image(torch.from_numpy(base.copy())[None, None], k)[0, 0].numpy()
            np.testing.assert_array_equal(t, img, err_msg=name)

    def test_d4_state_and_action_match_points(self):
        W = H = 100.0
        rng = np.random.default_rng(0)
        for k, name in enumerate(O.D4_NAMES):
            for _ in range(20):
                x, y = rng.uniform(1, 99, 2)
                dx, dy = rng.uniform(-2, 2, 2)
                px, py = transform_point((x, y), W, H, name)
                qx, qy = transform_point((x + dx, y + dy), W, H, name)
                s = torch.tensor([[x / W, y / H, 1.0, 2.0, 2.0]])
                st = O.d4_state(s, k)[0].numpy()
                np.testing.assert_allclose(st[:2], [px / W, py / H], atol=1e-6, err_msg=name)
                a = O.d4_action(torch.tensor([[dx, dy]]), k)[0].numpy()
                np.testing.assert_allclose(a, [qx - px, qy - py], atol=1e-6, err_msg=name)

    def test_d4_scale_swaps_on_quarter_turns(self):
        s = torch.tensor([[0.2, 0.3, 0.0, 1.0, 3.0]])
        self.assertEqual(O.d4_state(s, 1)[0, 3:].tolist(), [3.0, 1.0])
        self.assertEqual(O.d4_state(s, 2)[0, 3:].tolist(), [1.0, 3.0])

    def test_scale_feature(self):
        cfg = resolve_config(overrides={"MAP_SCALE_N": 1.5})
        env = _env()
        st = O.ObservationBuilder(cfg, env).robot_state()
        self.assertAlmostEqual(float(st[3]), 1.5 * 100 / 50)


if __name__ == "__main__":
    unittest.main()
