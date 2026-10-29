"""model.escape_distance where the navmesh has no route to safety.

Escape distance is walking distance. A triangle cut off from every safe
triangle reports `danger_unreachable` (farthest routed distance plus the crop
diagonal), not the straight line to the zone boundary it used to fall back to.
"""

import math
import random
import unittest

import numpy as np


def _model(seed=3, size=80):
    import sim.model as M
    from ued.level import generate_random_level
    lv = generate_random_level(random.Random(seed), difficulty=3, width=size, height=size)
    lv.robot_num = 1
    lv.augmentation = "identity"
    random.seed(0)
    np.random.seed(0)
    return M.FightingModel(int(lv.crowd_size), lv.width, lv.height, robot="Q", level=lv)


def _inside_triangle(m):
    """A routed triangle inside the hazard and a point of it that the
    grid lookup maps back to it."""
    for (gx, gy), mesh in m.match_grid_to_mesh.items():
        d = m.mesh_danger.get(mesh)
        xy = (gx + 0.5, gy + 0.5)
        if d is not None and 0 < d < 1e5 and not m.is_safe(xy) \
                and m.match_grid_to_mesh.get((int(xy[0]), int(xy[1]))) is mesh:
            return mesh, xy
    return None, None


class EscapeDistanceTest(unittest.TestCase):
    def setUp(self):
        self.m = _model()
        if getattr(self.m, "danger_zone", None) is None:
            self.skipTest("level without a hazard")

    def test_cap_is_beyond_every_routed_distance(self):
        m = self.m
        routed = [d for d in m.mesh_danger.values() if d < 1e5]
        self.assertTrue(routed)
        diag = math.hypot(float(m.width), float(m.height))
        self.assertAlmostEqual(m.danger_unreachable, max(routed) + diag, places=6)
        self.assertLess(m.danger_unreachable, 1e5)

    def test_routed_triangle_is_unchanged(self):
        m = self.m
        mesh, xy = _inside_triangle(m)
        self.assertIsNotNone(mesh)
        self.assertEqual(m.escape_distance(xy), float(m.mesh_danger[mesh]))

    def test_no_route_reports_the_cap_not_the_straight_line(self):
        m = self.m
        mesh, xy = _inside_triangle(m)
        self.assertIsNotNone(mesh)
        straight = float(m.danger_zone.escape_distance(*xy))
        m.mesh_danger[mesh] = 1e18          # cut this triangle off
        d = m.escape_distance(xy)
        self.assertEqual(d, m.danger_unreachable)
        self.assertGreater(d, straight)

    def test_off_mesh_point_still_uses_the_straight_line(self):
        m = self.m
        # a point no navmesh grid cell maps to, inside the hazard
        zone = m.danger_zone
        for gx in range(int(m.width)):
            for gy in range(int(m.height)):
                xy = (gx + 0.5, gy + 0.5)
                if (gx, gy) not in m.match_grid_to_mesh and not m.is_safe(xy):
                    self.assertEqual(m.escape_distance(xy), float(zone.escape_distance(*xy)))
                    return
        self.skipTest("every hazard cell is on the navmesh")


if __name__ == "__main__":
    unittest.main()
