"""Activity stays: draws, initial dwellers and the evacuation hand-over
(docs/crowd_activity_design.md)."""

import math
import random
import statistics
import unittest
from types import SimpleNamespace

from sim import dwell


def _cfg(**kw):
    base = dict(CROWD_DWELL_RANGE_S=(30.0, 600.0),
                CROWD_DWELL_DISTRIBUTION="uniform",
                CROWD_DWELL_EXP_SCALE_S=None,
                CROWD_DWELL_LOGNORMAL_MEDIAN_S=None,
                CROWD_DWELL_LOGNORMAL_SIGMA=None,
                CROWD_INITIAL_DWELL_FRACTION=0.3,
                CROWD_DWELL_WANDER_RADIUS_M=4.0,
                CROWD_DWELL_WANDER_SPEED_FACTOR=0.35,
                CROWD_DWELL_WANDER_PAUSE_S=(2.0, 10.0))
    base.update(kw)
    return _Cfg(base)


class _Cfg(dict):
    __getattr__ = dict.__getitem__


class DrawTest(unittest.TestCase):
    def test_each_distribution_stays_in_range_without_piling_on_bounds(self):
        for cfg in (_cfg(),
                    _cfg(CROWD_DWELL_DISTRIBUTION="exponential",
                         CROWD_DWELL_EXP_SCALE_S=120.0),
                    _cfg(CROWD_DWELL_DISTRIBUTION="lognormal",
                         CROWD_DWELL_LOGNORMAL_MEDIAN_S=90.0,
                         CROWD_DWELL_LOGNORMAL_SIGMA=1.0),
                    # A median below the range: the upper-tail branch.
                    _cfg(CROWD_DWELL_DISTRIBUTION="lognormal",
                         CROWD_DWELL_LOGNORMAL_MEDIAN_S=5.0,
                         CROWD_DWELL_LOGNORMAL_SIGMA=0.6)):
            self.assertEqual(dwell.dwell_config_problems(cfg), [])
            rng = random.Random(1)
            xs = [dwell.sample_s(rng, cfg) for _ in range(4000)]
            self.assertTrue(all(30.0 <= x <= 600.0 for x in xs))
            # Truncated, not clipped: no lump of mass at either bound.
            self.assertLess(sum(x in (30.0, 600.0) for x in xs), 5)

    def test_exponential_truncation_matches_its_conditional_mean(self):
        cfg = _cfg(CROWD_DWELL_DISTRIBUTION="exponential",
                   CROWD_DWELL_EXP_SCALE_S=120.0)
        rng = random.Random(2)
        xs = [dwell.sample_s(rng, cfg) for _ in range(40000)]
        # Mean of Exp(120) conditioned on [30, 600], from the closed form.
        import math
        s, a, b = 120.0, 30.0, 600.0
        want = s + (a * math.exp(-a / s) - b * math.exp(-b / s)) / (
            math.exp(-a / s) - math.exp(-b / s))
        self.assertAlmostEqual(statistics.mean(xs), want, delta=2.5)

    def test_residual_is_length_biased(self):
        """Uniform [30, 600]: the time left in a stay seen at a random moment
        has mean E[D^2] / (2 E[D]), about 200 s, not 315 s or 157.5 s."""
        rng = random.Random(3)
        xs = [dwell.sample_residual_s(rng, _cfg()) for _ in range(40000)]
        self.assertTrue(all(0.0 <= x <= 600.0 for x in xs))
        self.assertAlmostEqual(statistics.mean(xs), 200.48, delta=3.0)

    def test_steps_round_up(self):
        self.assertEqual(dwell.to_steps(30.0, 0.5), 60)
        self.assertEqual(dwell.to_steps(30.1, 0.5), 61)
        self.assertEqual(dwell.to_steps(0.2, 0.5), 1)

    def test_streams_are_per_stay_and_shared_across_conditions(self):
        a = dwell.stream(7, 12, 1).random()
        self.assertEqual(a, dwell.stream(7, 12, 1).random())
        self.assertNotEqual(a, dwell.stream(7, 12, 2).random())
        self.assertNotEqual(a, dwell.stream(7, 13, 1).random())
        self.assertNotEqual(a, dwell.stream(8, 12, 1).random())

    def test_bad_settings_are_refused(self):
        for kw in ({"CROWD_DWELL_RANGE_S": (600.0, 30.0)},
                   {"CROWD_DWELL_RANGE_S": 30.0},
                   {"CROWD_DWELL_DISTRIBUTION": "gamma"},
                   {"CROWD_DWELL_DISTRIBUTION": "exponential"},
                   {"CROWD_DWELL_DISTRIBUTION": "lognormal",
                    "CROWD_DWELL_LOGNORMAL_MEDIAN_S": 60.0},
                   {"CROWD_DWELL_DISTRIBUTION": "exponential",
                    "CROWD_DWELL_EXP_SCALE_S": 0.01},
                   {"CROWD_INITIAL_DWELL_FRACTION": 1.5},
                   {"CROWD_DWELL_WANDER_RADIUS_M": -1.0},
                   {"CROWD_DWELL_WANDER_SPEED_FACTOR": 0.0},
                   {"CROWD_DWELL_WANDER_PAUSE_S": (5.0, 2.0)}):
            self.assertTrue(dwell.dwell_config_problems(_cfg(**kw)), kw)


class AgentTest(unittest.TestCase):
    def _model(self, seed=4):
        import numpy as np
        import sim.model as M
        from ued.level import generate_city_level
        random.seed(seed)
        np.random.seed(seed)
        lv = generate_city_level(random.Random(seed), difficulty=5,
                                 crowd_size=60, width=140, height=140,
                                 morphology="grid", robot_num=1,
                                 perceptibility=0.0)
        return M.FightingModel(number_agents=60, width=140, height=140,
                               robot='N', level=lv)

    @staticmethod
    def _here(a):
        """A destination triangle centred where `a` stands: arrived."""
        x, y = float(a.xy[0]), float(a.xy[1])
        return ((x - 0.1, y - 0.1), (x + 0.2, y - 0.1), (x - 0.1, y + 0.2))

    def _dweller(self, fm):
        return next(a for a in fm.crowds
                    if not a.dead and a._dwell_reason == "activity")

    def test_initial_dwellers_start_mid_stay_where_placed(self):
        from config import CROWD_INITIAL_DWELL_FRACTION
        fm = self._model()
        d = [a for a in fm.crowds if a._dwell_reason == "activity"]
        self.assertEqual(len(d), fm.activity_counts["started_initial"])
        self.assertGreater(len(d), 0)
        self.assertLess(abs(len(d) / len(fm.crowds)
                            - float(CROWD_INITIAL_DWELL_FRACTION)), 0.2)
        a = d[0]
        before = list(a.xy)
        until = a._activity_until
        self.assertGreater(until, 0)
        a.which_goal_agent_want(a._neighbors(a.vision_radius))
        self.assertEqual(a._activity_until, until)   # not redrawn
        self.assertEqual(a.now_goal, before)

    def test_same_seed_gives_same_initial_dwellers(self):
        a = {x.unique_id: x._activity_until for x in self._model(5).crowds}
        b = {x.unique_id: x._activity_until for x in self._model(5).crowds}
        self.assertEqual(a, b)

    def test_acting_without_hazard_memory_ends_the_stay(self):
        """Told by a neighbour or a robot, not knowing where the hazard is:
        starting to act still ends the stay, and it does not drift back into
        it on a later step (design 4.2)."""
        fm = self._model()
        a = self._dweller(fm)
        a.awareness = "milling"
        a.act_after = 0.5
        self.assertFalse(a.hazard_memory)
        a.update_awareness([], robot_near=False)
        self.assertEqual(a.awareness, "acting")
        self.assertEqual(a._activity_until, 0)
        self.assertFalse(a._dwelling)
        self.assertEqual(fm.activity_counts.get("cancelled_acting"), 1)
        for _ in range(3):
            a.which_goal_agent_want([])
            self.assertNotEqual(a._dwell_reason, "activity")

    def test_following_ends_the_stay_and_release_does_not_resume_it(self):
        fm = self._model()
        a = self._dweller(fm)
        rb = SimpleNamespace(unique_id=999, mode="guide", xy=[0.0, 0.0])
        a._start_following(rb)
        self.assertEqual(a._activity_until, 0)
        self.assertFalse(a._dwelling)
        self.assertEqual(fm.activity_counts.get("cancelled_robot_follow"), 1)
        a._stop_following()
        self.assertEqual(a._activity_until, 0)

    def test_milling_keeps_the_stay(self):
        fm = self._model()
        a = self._dweller(fm)
        until = a._activity_until
        a.awareness = "milling"
        a.act_after = 50.0
        a.update_awareness([], robot_near=False)
        self.assertEqual(a._activity_until, until)

    def test_arrival_draws_once_and_the_next_stay_anew(self):
        fm = self._model()
        a = next(x for x in fm.crowds if x._dwell_reason is None)
        a.now_pointing_mesh = self._here(a)
        a._trip_goal()
        self.assertEqual(a._dwell_reason, "activity")
        first = a._activity_until - fm.step_count
        a._trip_goal()
        self.assertEqual(a._activity_until - fm.step_count, first)
        fm.step_count = a._activity_until
        a._trip_goal()
        self.assertEqual(fm.activity_counts.get("completed"), 1)
        self.assertNotEqual(a._dwell_reason, "activity")
        a.now_pointing_mesh = self._here(a)
        a._trip_goal()
        self.assertNotEqual(a._activity_until - fm.step_count, first)

    def test_after_evacuating_it_may_stay_again(self):
        """Out of believed danger and continuing its errands, a pedestrian
        dwells at its next destination like anyone else (design 1.5)."""
        fm = self._model()
        a = next(x for x in fm.crowds if x._dwell_reason is None)
        a.awareness = "acting"
        a.ever_acted = True
        a.post_safe_intent = "continue"
        far = max(fm.pure_mesh, key=lambda t: (t[0][0] - a.xy[0]) ** 2
                  + (t[0][1] - a.xy[1]) ** 2)
        a._remember_hazard(((far[0][0] + far[1][0] + far[2][0]) / 3.0,
                            (far[0][1] + far[1][1] + far[2][1]) / 3.0), 0.0)
        a._plan_version = a._memory_version
        if a._believes_dangerous(*a.xy):
            self.skipTest("the sighting covers where it stands")
        a.now_pointing_mesh = self._here(a)
        a._belief_route_goal()
        self.assertEqual(a._dwell_reason, "activity")

    def test_learning_more_shortens_a_shelter_wait_only(self):
        fm = self._model()
        a = next(x for x in fm.crowds if x._dwell_reason is None)
        a.awareness = "acting"
        a.post_safe_intent = "continue"
        a._shelter_until = fm.step_count + 50
        a._activity_until = fm.step_count + 80
        a._plan_version = a._memory_version - 1
        a._belief_route_goal()
        self.assertLessEqual(a._shelter_until, fm.step_count)
        self.assertEqual(a._activity_until, fm.step_count + 80)

    def test_staying_people_stroll_about_their_spot(self):
        """Not frozen: over a minute most initial dwellers move, and none
        strays far beyond CROWD_DWELL_WANDER_RADIUS_M of where it began."""
        from config import CROWD_DWELL_WANDER_RADIUS_M
        fm = self._model()
        start = {a.unique_id: tuple(a.xy) for a in fm.crowds
                 if a._dwell_reason == "activity"}
        moved, far = set(), 0.0
        for _ in range(120):
            fm.step()
            for a in fm.crowds:
                if (a.unique_id in start and not a.dead
                        and a._dwell_reason == "activity"):
                    if math.dist(a.xy, start[a.unique_id]) > 0.5:
                        moved.add(a.unique_id)
                    # From the spot of its current stay, which is a new one
                    # if the first ended and it walked on.
                    far = max(far, math.dist(a.xy, a._wander_anchor))
        self.assertGreater(len(moved), len(start) // 2)
        # Body contact can push a little past the radius.
        self.assertLess(far, float(CROWD_DWELL_WANDER_RADIUS_M) + 2.0)

    def test_zero_radius_stands_still(self):
        import sim.agent as A
        old = A.CROWD_DWELL_WANDER_RADIUS_M
        try:
            A.CROWD_DWELL_WANDER_RADIUS_M = 0.0
            fm = self._model()
            a = self._dweller(fm)
            for step in range(40):
                fm.step_count = step
                self.assertEqual(a._wander_goal(step), list(map(float, a.xy)))
        finally:
            A.CROWD_DWELL_WANDER_RADIUS_M = old

    def test_metrics_count_stays_and_their_end(self):
        from learn.zero_shot import EpisodeMetrics
        fm = self._model()
        tm = EpisodeMetrics().start(fm)
        for _ in range(5):
            fm.step()
            tm.step(fm)
        out = tm.summary()
        self.assertEqual(out["activity_started_initial"],
                         fm.activity_counts["started_initial"])
        self.assertGreater(out["activity_dwell_person_steps"], 0)
        self.assertGreater(out["activity_unfinished"], 0)


if __name__ == "__main__":
    unittest.main()
