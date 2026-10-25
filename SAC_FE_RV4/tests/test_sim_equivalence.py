"""The vectorised map build (SIM_FAST_BUILD) must reproduce RV3 exactly.

Each map is built twice from the same seeds, once with the original Python
loops and once with sim_accel, and every structure the simulation reads is
compared for equality; then both copies are stepped with the same random
robot commands and every agent position must agree bit for bit.

The 200 m map costs about a minute on the slow path, so it runs only with
RV4_SLOW_TESTS=1.
"""

import math
import os
import random
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import model  # noqa: E402

MAPS = [1001, 1500, 128, 105, 130, 0]  # json, json, json, coded, coded real building, generated
if os.environ.get("RV4_SLOW_TESTS"):
    MAPS += [2000, 1506]


def build(map_num, fast, seed=123, augmentation="identity"):
    model.SIM_FAST_BUILD = fast
    random.seed(seed)
    np.random.seed(seed)
    size = 100
    if map_num in (2000, 1506):
        import json
        size = json.load(open(f"map_infos/map_{map_num}.json"))["width"]
    return model.FightingModel(30, size, size, model_num=map_num, robot="Q",
                               augmentation=augmentation)


def dist_equal(a, b):
    return a == b or (math.isinf(a) and math.isinf(b))


class SimEquivalenceTest(unittest.TestCase):
    def _compare(self, env_a, env_b):
        self.assertEqual(env_a.mesh_list, env_b.mesh_list)
        self.assertEqual(env_a.mesh, env_b.mesh)
        self.assertEqual(env_a.match_grid_to_mesh, env_b.match_grid_to_mesh)
        self.assertEqual(env_a.obstacle_mesh, env_b.obstacle_mesh)
        self.assertEqual(env_a.pure_mesh, env_b.pure_mesh)
        self.assertEqual(env_a.adjacent_mesh, env_b.adjacent_mesh)
        self.assertEqual(env_a.match_mesh_to_grid, env_b.match_mesh_to_grid)
        self.assertEqual(env_a.valid_space, env_b.valid_space)
        for m1 in env_a.mesh_list:
            ra, rb = env_a.distance[m1], env_b.distance[m1]
            na, nb = env_a.next_vertex_matrix[m1], env_b.next_vertex_matrix[m1]
            for m2 in env_a.mesh_list:
                self.assertTrue(dist_equal(ra[m2], rb[m2]), (m1, m2, ra[m2], rb[m2]))
                self.assertEqual(na[m2], nb[m2])
        np.testing.assert_array_equal(env_a.static_grid, env_b.static_grid)
        self.assertEqual(env_a.mesh_danger, env_b.mesh_danger)
        ca, cb = env_a.vision_atlas._cache, env_b.vision_atlas._cache
        self.assertEqual(set(ca), set(cb))
        for k in ca:
            np.testing.assert_array_equal(np.asarray(ca[k].exterior.coords),
                                          np.asarray(cb[k].exterior.coords))

    def _run_same(self, env_a, env_b, steps=60, seed=7):
        for env in (env_a, env_b):
            rng = np.random.RandomState(seed)
            random.seed(seed)
            np.random.seed(seed)
            for t in range(steps):
                if t % 4 == 0:
                    env.robot.receive_action([float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))])
                env.step()
        for a, b in zip(env_a.crowds, env_b.crowds):
            self.assertEqual(list(a.xy), list(b.xy))
            self.assertEqual(a.dead, b.dead)
            self.assertEqual(a.type, b.type)
        self.assertEqual(list(env_a.robot.xy), list(env_b.robot.xy))

    def test_maps(self):
        for m in MAPS:
            with self.subTest(map=m):
                slow = build(m, fast=False)
                fast = build(m, fast=True)
                self._compare(slow, fast)
                self._run_same(slow, fast)

    def test_augmented(self):
        slow = build(1500, fast=False, augmentation="reflect_rotate_90")
        fast = build(1500, fast=True, augmentation="reflect_rotate_90")
        self._compare(slow, fast)

    def tearDown(self):
        model.SIM_FAST_BUILD = True


if __name__ == "__main__":
    unittest.main()
