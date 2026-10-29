"""Walking distance to exits (learn/exit_distance.py, DANGER_DISTANCE)."""

import json
import math
import os
import random
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from learn.exit_distance import ExitDistanceField  # noqa: E402

EXIT_RIGHT = [[[96.0, 45.0], [100.0, 45.0], [100.0, 55.0], [96.0, 55.0]]]


def _straight(x, y, exits=EXIT_RIGHT):
    from shapely.geometry import Point, Polygon
    return min(Polygon(e).distance(Point(x, y)) for e in exits)


class FieldTest(unittest.TestCase):
    def test_open_space_matches_straight_line(self):
        f = ExitDistanceField([], EXIT_RIGHT, 100, 100, 0.5)
        rng = np.random.default_rng(0)
        # grid metric: within 3 % in any direction, plus about half a cell
        # because exit cells are counted from their centres
        for x, y in rng.uniform(5, 90, size=(300, 2)):
            s = _straight(x, y)
            self.assertLessEqual(abs(f(x, y) - s), 0.03 * s + 0.5, (x, y))

    def test_detour_around_wall_end(self):
        # wall x 50-51, y 0-80 between the point and the exit: the shortest
        # path goes over the wall end at (50, 80) and (51, 80)
        wall = [[[50.0, 0.0], [51.0, 0.0], [51.0, 80.0], [50.0, 80.0]]]
        f = ExitDistanceField(wall, EXIT_RIGHT, 100, 100, 0.5)
        p = (40.0, 50.0)
        exact = math.hypot(50 - 40, 80 - 50) + 1.0 + math.hypot(96 - 51, 80 - 55)
        self.assertGreater(exact, 1.4 * _straight(*p))
        self.assertLessEqual(abs(f(*p) - exact), 0.03 * exact + 0.5)

    def test_shut_in_area_gets_the_finite_cap(self):
        # a one-cell-thick diagonal wall closes the lower-left corner; moves
        # may not cut its corners, so the corner has no route
        wall = [[[0.0, 60.0], [60.0, 0.0], [60.6, 0.6], [0.6, 60.6]]]
        f = ExitDistanceField(wall, EXIT_RIGHT, 100, 100, 0.5)
        self.assertGreater(f.unreachable_free_cells, 0)
        reach_max = max(f(x, y) for x in range(35, 95, 5) for y in range(35, 95, 5))
        d = f(10.0, 10.0)
        self.assertEqual(d, f.unreachable)
        self.assertGreater(d, reach_max)          # farther than any reachable place
        self.assertLess(d, 1e5)                   # not removed by the simulator
        self.assertLess(d, 4 * math.hypot(100, 100))

    def test_no_exit_is_infinite(self):
        f = ExitDistanceField([], [], 100, 100, 0.5)
        self.assertEqual(f(50.0, 50.0), math.inf)

    def test_exit_smaller_than_a_cell_still_counts(self):
        tiny = [[[99.8, 49.9], [100.0, 49.9], [100.0, 50.1], [99.8, 50.1]]]
        f = ExitDistanceField([], tiny, 100, 100, 0.5)
        self.assertEqual(f.unreachable_free_cells, 0)
        self.assertLessEqual(abs(f(50.0, 50.0) - 49.8), 0.03 * 49.8 + 0.5)

    def test_build_time(self):
        t0 = time.time()
        ExitDistanceField([], EXIT_RIGHT, 200, 200, 0.5)
        self.assertLess(time.time() - t0, 5.0)


class DangerTest(unittest.TestCase):
    def _env(self, overrides, map_num=1504, seed=3):
        os.environ["RV4_OVERRIDES"] = json.dumps(overrides)
        from configs import refresh
        refresh()
        import importlib
        import agent
        import model
        importlib.reload(agent)
        importlib.reload(model)
        random.seed(seed)
        np.random.seed(seed)
        return model.FightingModel(30, 100, 100, model_num=map_num, robot="Q")

    def tearDown(self):
        os.environ.pop("RV4_OVERRIDES", None)
        from configs import refresh
        refresh()
        import importlib
        import agent
        import model
        importlib.reload(agent)
        importlib.reload(model)

    def test_euclidean_default_is_rv3(self):
        env = self._env({})
        for _ in range(3):
            env.step()
        for a in env.crowds:
            if not a.dead:
                self.assertAlmostEqual(env.danger_distance(a.xy), env.distance_to_exit(a.xy), places=9)

    def test_geodesic_in_the_map_1504_dead_end(self):
        env = self._env({"DANGER_DISTANCE": "geodesic"})
        env.step()
        a = next(a for a in env.crowds if not a.dead)
        a.xy = [40.5, 97.1]
        a.vel = [0.0, 0.0]
        env.step()
        straight = env.distance_to_exit(a.xy)
        self.assertAlmostEqual(straight, 21.5, delta=0.6)
        self.assertGreater(a.danger, 50.0)        # around the wall: about 56 m
        self.assertLess(a.danger, 62.0)
        # danger is taken before the agent moves in the step (< 1 m)
        self.assertLess(abs(a.danger - env.danger_distance(a.xy)), 1.0)
        # reward_b sums the walking distances
        tot = -sum(x.danger for x in env.crowds if x.type in (0, 1, 2) and not x.dead)
        self.assertAlmostEqual(env.reward_based_all_agents_danger(), tot, places=9)


class FingerprintTest(unittest.TestCase):
    def test_default_leaves_fingerprint_unchanged(self):
        from configs import _FINGERPRINT_DEFAULTS, resolve_config
        os.environ.pop("RV4_OVERRIDES", None)
        cfg = resolve_config()
        for k, v in _FINGERPRINT_DEFAULTS.items():
            self.assertEqual(cfg.get(k), v)
        base = cfg.fingerprint()
        os.environ["RV4_OVERRIDES"] = json.dumps({"DANGER_DISTANCE": "geodesic"})
        try:
            self.assertNotEqual(resolve_config().fingerprint(), base)
        finally:
            os.environ.pop("RV4_OVERRIDES", None)


if __name__ == "__main__":
    unittest.main()
