"""Stage 4 of docs/outdoor_madrl_redesign.md: partial observation, team
communication and the multi-resolution input.

Pass conditions from section 9:
  * moving a person nobody can see leaves every actor input unchanged and
    changes only the critic's privileged input;
  * a teammate's new observation changes the shared input of the others;
  * overlapping observations are not counted twice;
  * the same generator serves training, viewer and evaluation.
"""

import random
import unittest

import numpy as np

from configs import resolve_config


def _cfg(**over):
    return resolve_config(over or None, check_data=False)


def _model(cfg, obstacles=(), n_people=6, robots=2, size=100, zone=None):
    from sim.danger import DangerZone
    from ued.level import Level
    import sim.model as M

    random.seed(0)
    np.random.seed(0)
    lv = Level(obstacles=[list(map(list, o)) for o in obstacles], exits=[],
               crowd_size=n_people, width=size, height=size)
    lv.danger = zone or DangerZone("circle", 70.0, 70.0, radius=12.0)
    lv.robot_num = robots
    lv.augmentation = "identity"
    m = M.FightingModel(n_people, size, size, robot="Q", level=lv)
    return m


def _place(model, agent, x, y):
    agent.xy = [float(x), float(y)]
    agent.pos = (float(x), float(y))
    model.space.move(agent.unique_id, agent.xy)


def _observe(model, cfg, seed=0):
    from sim.observation import ObservationHistory, build_static_layers
    st = build_static_layers(model, cfg)
    h = ObservationHistory(cfg, st, len(model.robots), seed=seed)
    h.record(model)
    return h, h.team_observations()


class PartialObservationTest(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg()

    def _two_robots_apart(self, model):
        _place(model, model.robots[0], 20.0, 20.0)
        _place(model, model.robots[1], 80.0, 20.0)
        for k, a in enumerate(model.crowds):
            _place(model, a, 50.0 + k, 85.0)      # far from both robots

    def test_unseen_person_changes_only_the_critic_input(self):
        m = _model(self.cfg)
        self._two_robots_apart(m)
        h1, obs1 = _observe(m, self.cfg)
        priv1 = h1.records[-1].priv.copy()
        _place(m, m.crowds[0], 10.0, 90.0)        # still unseen by anyone
        h2, obs2 = _observe(m, self.cfg)
        for key in ("ego", "mid", "glob", "state"):
            np.testing.assert_array_equal(obs1[key], obs2[key], key)
        self.assertFalse(np.array_equal(priv1, h2.records[-1].priv))

    def test_seen_person_changes_the_actor_input(self):
        m = _model(self.cfg)
        self._two_robots_apart(m)
        _, obs1 = _observe(m, self.cfg)
        _place(m, m.crowds[0], 23.0, 21.0)        # 3 m from robot 0
        _, obs2 = _observe(m, self.cfg)
        self.assertGreater(obs2["ego"][0, 3].sum(), obs1["ego"][0, 3].sum())

    def test_teammate_observation_reaches_the_other_robot(self):
        m = _model(self.cfg)
        self._two_robots_apart(m)
        _, obs1 = _observe(m, self.cfg)
        _place(m, m.crowds[0], 82.0, 22.0)        # seen by robot 1 only
        _, obs2 = _observe(m, self.cfg)
        # Robot 0's own sensor is unchanged, its team map is not.
        np.testing.assert_array_equal(obs1["ego"][0], obs2["ego"][0])
        self.assertGreater(obs2["glob"][0, 3].sum(), obs1["glob"][0, 3].sum())

    def test_without_sharing_a_teammate_observation_stays_private(self):
        cfg = _cfg(TEAM_SHARE_OBSERVATIONS=False)
        m = _model(cfg)
        self._two_robots_apart(m)
        _, obs1 = _observe(m, cfg)
        _place(m, m.crowds[0], 82.0, 22.0)
        _, obs2 = _observe(m, cfg)
        np.testing.assert_array_equal(obs1["glob"][0], obs2["glob"][0])

    def test_overlapping_observations_are_not_counted_twice(self):
        m = _model(self.cfg)
        _place(m, m.robots[0], 30.0, 30.0)
        _place(m, m.robots[1], 31.0, 30.0)        # same view, nearly
        for k, a in enumerate(m.crowds):
            _place(m, a, 50.0 + k, 85.0)
        for k in range(3):
            _place(m, m.crowds[k], 33.0 + k, 32.0)
        h, obs = _observe(m, self.cfg)
        rec = h.records[-1]
        G = self.cfg.OBS_GLOBAL_SIZE
        cw = m.width / G
        density_to_count = cw * cw * self.cfg.OBS_DENSITY_SATURATION
        team_count = obs["glob"][0, 3].sum() * density_to_count
        self.assertAlmostEqual(team_count, 3.0, places=4)
        self.assertEqual(int(rec.counts[0].sum()), 3)
        self.assertEqual(int(rec.counts[1].sum()), 3)

    def test_person_behind_a_building_is_not_measured(self):
        wall = [(24.0, 10.0), (26.0, 10.0), (26.0, 40.0), (24.0, 40.0)]
        m = _model(self.cfg, obstacles=[wall])
        _place(m, m.robots[0], 20.0, 25.0)
        _place(m, m.robots[1], 80.0, 80.0)
        for k, a in enumerate(m.crowds):
            _place(m, a, 50.0 + k, 60.0)
        _place(m, m.crowds[0], 28.0, 25.0)        # 8 m away, behind the wall
        h, obs = _observe(m, self.cfg)
        self.assertEqual(int(h.records[-1].counts[0].sum()), 0)
        _place(m, m.crowds[0], 20.0, 30.0)        # 5 m away, in the open
        h, obs = _observe(m, self.cfg)
        self.assertEqual(int(h.records[-1].counts[0].sum()), 1)

    def test_unobserved_is_not_zero(self):
        m = _model(self.cfg)
        self._two_robots_apart(m)
        _, obs = _observe(m, self.cfg)
        # The global team-observed channel is zero where no robot looked, so
        # a zero crowd density there carries no claim about emptiness.
        observed = obs["glob"][0, 4]
        self.assertGreater((observed == 0).mean(), 0.5)
        self.assertGreater(observed.max(), 0.0)

    def test_communication_delay_hides_the_newest_message(self):
        cfg = _cfg(COMM_DELAY_DECISIONS=1)
        m = _model(cfg)
        self._two_robots_apart(m)
        from sim.observation import ObservationHistory, build_static_layers
        st = build_static_layers(m, cfg)
        h = ObservationHistory(cfg, st, len(m.robots))
        h.record(m)
        _place(m, m.crowds[0], 82.0, 22.0)
        h.record(m)
        rec = h.records[-1]
        # Robot 0 holds robot 1's previous record, not the one just taken.
        self.assertEqual(int(rec.avail[0, 1]) & 1, 0)
        self.assertEqual(int(rec.avail[0, 1]) & 2, 2)
        obs = h.team_observations()
        self.assertEqual(obs["glob"][0, 3].sum(), 0.0)

    def test_old_observations_expire(self):
        cfg = _cfg()
        m = _model(cfg)
        self._two_robots_apart(m)
        _place(m, m.crowds[0], 23.0, 21.0)
        from sim.observation import ObservationHistory, build_static_layers
        st = build_static_layers(m, cfg)
        h = ObservationHistory(cfg, st, len(m.robots))
        h.record(m)
        _place(m, m.robots[0], 20.0, 60.0)        # walk away
        for _ in range(cfg.OBS_HISTORY_DECISIONS):
            h.record(m)
        obs = h.team_observations()
        self.assertEqual(obs["glob"][1, 3].sum(), 0.0)

    def test_full_information_actor_sees_everyone(self):
        cfg = _cfg(ACTOR_GLOBAL_CROWD_TRUTH=True,
                   EXPERIMENT_ID="x-fullinfo", LOG_DIR="Log_fullinfo")
        m = _model(cfg)
        self._two_robots_apart(m)
        _, obs = _observe(m, cfg)
        self.assertGreater(obs["glob"][0, 3].sum(), 0.0)


class StaticLayerTest(unittest.TestCase):
    def test_area_resample_is_exact_for_non_integer_ratios(self):
        from sim.observation import area_resample
        rng = np.random.default_rng(0)
        a = rng.random((100, 100)).astype(np.float32)
        out = area_resample(a, 64, 64)
        self.assertAlmostEqual(float(out.mean()), float(a.mean()), places=5)
        ones = area_resample(np.ones((100, 100)), 64, 64)
        np.testing.assert_allclose(ones, 1.0, rtol=1e-6)

    def test_path_field_walks_around_buildings(self):
        from sim.danger import DangerZone
        from sim.observation import build_static_layers
        cfg = _cfg()
        wall = [(40.0, 0.0), (44.0, 0.0), (44.0, 80.0), (40.0, 80.0)]
        m = _model(cfg, obstacles=[wall],
                   zone=DangerZone("circle", 60.0, 40.0, radius=8.0))
        st = build_static_layers(m, cfg)
        # Just left of the wall the boundary is ~12 m straight through the
        # wall, but the walk goes round its far end at y = 80.
        d = float(st.path[40, 38])
        self.assertGreater(d, 60.0)
        self.assertTrue(np.isfinite(st.path[40, 55]))

    def test_static_key_depends_on_the_hazard(self):
        from sim.danger import DangerZone
        from sim.observation import static_key
        cfg = _cfg()
        a = _model(cfg, zone=DangerZone("circle", 70.0, 70.0, radius=12.0))
        b = _model(cfg, zone=DangerZone("circle", 30.0, 70.0, radius=12.0))
        self.assertNotEqual(static_key(a), static_key(b))


if __name__ == "__main__":
    unittest.main()
