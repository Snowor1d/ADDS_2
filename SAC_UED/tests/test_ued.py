import os
import random
import time
import unittest

from random_map import RandomMapSpec, generate_map
from ued.level import generate_random_level, level_from_map_data
from ued.mutate import MutationFailed, is_playable, mutate_level
from ued.population import LevelPopulation


class RngIsolationTest(unittest.TestCase):
    """The separation between designer-controlled and aleatoric parameters.

    generate_map seeds the global RNG so a seed reproduces a map. If that
    reseeding leaks, everything the caller draws afterwards - crowd spawn
    positions, per-pedestrian mass and speed, the augmentation choice - becomes
    a deterministic function of the map seed, and the curriculum ends up
    selecting the crowd as well as the map.
    """

    def test_map_is_reproducible_from_its_seed(self):
        a = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=4242))
        b = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=4242))
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.exits, b.exits)

    def test_generation_does_not_disturb_the_caller_stream(self):
        random.seed(777)
        reference = [random.random() for _ in range(5)]

        random.seed(777)
        generate_map(RandomMapSpec(100, 100, difficulty=2, seed=4242))
        after = [random.random() for _ in range(5)]

        self.assertEqual(reference, after)

    def test_different_map_seeds_leave_the_same_caller_stream(self):
        random.seed(777)
        generate_map(RandomMapSpec(100, 100, difficulty=2, seed=1))
        first = [random.random() for _ in range(5)]

        random.seed(777)
        generate_map(RandomMapSpec(100, 100, difficulty=2, seed=999999))
        second = [random.random() for _ in range(5)]

        self.assertEqual(first, second)


class MutationTest(unittest.TestCase):
    """Edits must stay playable and stay cheap."""

    @classmethod
    def setUpClass(cls):
        cls.rng = random.Random(20260911)
        cls.parent = generate_random_level(cls.rng)

    def test_generated_level_passes_its_own_validator(self):
        self.assertTrue(is_playable(self.parent))

    def test_children_are_always_playable(self):
        rng = random.Random(5)
        parent = self.parent
        produced = 0
        for _ in range(30):
            try:
                child = mutate_level(parent, rng=rng)
            except MutationFailed:
                continue
            produced += 1
            # Revalidating independently catches an operator that edits state
            # the validator happened not to look at.
            self.assertTrue(is_playable(child), child.summary())
            self.assertGreaterEqual(len(child.exits), 1)
            self.assertGreaterEqual(len(child.obstacles), 1)
            parent = child
        self.assertGreater(produced, 20, "mutation success rate collapsed")

    def test_child_cost_stays_far_below_generation_cost(self):
        # Breeding runs on the producer thread alongside generation, so it only
        # has to stay well under the costs that already dominate an episode:
        # generating a map averages 0.8-2.1 s and building the environment
        # around it takes several seconds more. The ceiling is deliberately
        # loose because this timing moves with machine load; it is here to
        # catch an operator that becomes pathologically slow, not to track
        # small regressions.
        rng = random.Random(6)
        n = 20
        start = time.time()
        for _ in range(n):
            try:
                mutate_level(self.parent, rng=rng)
            except MutationFailed:
                pass
        per_child = (time.time() - start) / n
        self.assertLess(per_child, 0.5, f"{per_child:.3f}s per child")

    def test_lineage_is_recorded(self):
        rng = random.Random(7)
        child = mutate_level(self.parent, rng=rng)
        self.assertEqual(child.parent_id, self.parent.level_id)
        self.assertEqual(child.generation, self.parent.generation + 1)
        self.assertTrue(child.mutation_ops)

    def test_editing_does_not_drive_the_population_to_empty_maps(self):
        # Deleting and shrinking always validate while adding and growing often
        # do not, so an unbalanced operator set collapses every lineage to a
        # bare map regardless of what the score function prefers.
        rng = random.Random(8)
        cur = self.parent
        counts = [len(cur.obstacles)]
        for _ in range(40):
            try:
                cur = mutate_level(cur, rng=rng)
            except MutationFailed:
                continue
            counts.append(len(cur.obstacles))
        self.assertGreaterEqual(max(counts[-10:]), 3, f"obstacle counts: {counts}")


class LevelInjectionTest(unittest.TestCase):
    """A level must reach the simulator as exactly the geometry it stores."""

    def test_model_uses_the_level_geometry_verbatim(self):
        import model

        rng = random.Random(31)
        level = generate_random_level(rng)
        level.augmentation = "identity"

        env = model.FightingModel(level.crowd_size, 100, 100, robot="Q", level=level)
        self.assertEqual(env.map_num, 0)
        self.assertEqual(env.map_augmentation, "identity")

        # Exits survive verbatim. Obstacles do not: mesh_map() unions touching
        # obstacles and rewrites the list for every map, UED or not, so the
        # invariant that matters is that the blocked area is the level's.
        self.assertEqual(
            [[tuple(p) for p in poly] for poly in env.exit_list],
            [[tuple(p) for p in poly] for poly in level.exits],
        )

        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        want = unary_union([Polygon(poly) for poly in level.obstacles])
        got = unary_union([Polygon(poly) for poly in env.obstacles])
        self.assertAlmostEqual(want.area, got.area, delta=max(1.0, 0.01 * want.area))
        self.assertLess(want.symmetric_difference(got).area, 0.01 * want.area + 1.0)

    def test_two_builds_of_one_level_are_identical(self):
        # The population replays a level many times, so its geometry must not
        # drift between episodes.
        import model

        rng = random.Random(33)
        level = generate_random_level(rng)
        a = model.FightingModel(level.crowd_size, 100, 100, robot="Q", level=level)
        b = model.FightingModel(level.crowd_size, 100, 100, robot="Q", level=level)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.exit_list, b.exit_list)

    def test_augmentation_is_pinned_per_level(self):
        import model

        rng = random.Random(32)
        level = generate_random_level(rng)
        level.augmentation = "rotate_90"
        for _ in range(3):
            env = model.FightingModel(level.crowd_size, 100, 100, robot="Q", level=level)
            self.assertEqual(env.map_augmentation, "rotate_90")


class PopulationTest(unittest.TestCase):
    def _level(self, seed):
        data = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=seed))
        return level_from_map_data(data, crowd_size=30, difficulty=2)

    def test_learnability_peaks_at_a_balanced_success_rate(self):
        pop = LevelPopulation(capacity=10, score_fn="learnability")
        always = self._level(101)
        never = self._level(102)
        mixed = self._level(103)
        for lv in (always, never, mixed):
            pop.add(lv)

        # 100 steps of free-flow, so with K=3 the threshold is 300 steps.
        for i in range(8):
            pop.update(always.level_id, evac_time=100.0, freeflow_steps=100.0)
            pop.update(never.level_id, evac_time=5000.0, freeflow_steps=100.0)
            pop.update(mixed.level_id, evac_time=(100.0 if i % 2 == 0 else 5000.0),
                       freeflow_steps=100.0)

        scores = {lv.level_id: pop._raw_score(pop._records[lv.level_id])
                  for lv in (always, never, mixed)}
        self.assertGreater(scores[mixed.level_id], scores[always.level_id])
        self.assertGreater(scores[mixed.level_id], scores[never.level_id])

    def test_eviction_protects_levels_that_have_never_run(self):
        pop = LevelPopulation(capacity=2, score_fn="learnability")
        scored = self._level(201)
        pop.add(scored)
        pop.update(scored.level_id, evac_time=100.0, freeflow_steps=100.0)  # learnability near zero

        fresh_a = self._level(202)
        fresh_b = self._level(203)
        pop.add(fresh_a)
        pop.add(fresh_b)

        self.assertEqual(len(pop), 2)
        self.assertNotIn(scored.level_id, pop._records)

    def test_sampling_returns_a_member(self):
        pop = LevelPopulation(capacity=5)
        levels = [self._level(300 + i) for i in range(3)]
        for lv in levels:
            pop.add(lv)
        ids = {lv.level_id for lv in levels}
        for _ in range(20):
            self.assertIn(pop.sample().level_id, ids)


if __name__ == "__main__":
    unittest.main()


class _FakeMsg:
    def __init__(self, worker_id, episode_idx, level_id, reward, done=False):
        import numpy as np

        self.worker_id = worker_id
        self.episode_key = (worker_id, episode_idx)
        self.level_id = level_id
        self.reward = reward
        self.done = done
        self.is_replay = False
        self.ego_state = np.full((4, 4, 4), float(worker_id), dtype=np.float32)
        self.global_state = np.full((4, 4, 4), float(worker_id), dtype=np.float32)
        self.robot_state = np.full((3,), float(worker_id), dtype=np.float32)


class _FakeStat:
    def __init__(self, worker_id, episode_idx, level_id, evac_time_100):
        self.worker_id = worker_id
        self.episode_idx = episode_idx
        self.level_id = level_id
        self.evac_time_100 = evac_time_100
        self.abnormal = 0
        self.is_replay = False


class MaxMCTest(unittest.TestCase):
    """MaxMC is what makes a regret-style score available off-policy at all.

    The GAE-based proxy the original methods use needs on-policy rollouts;
    MaxMC only needs a value estimate, which SAC's critic supplies.
    """

    def _runner(self, value=0.0):
        from ued.runner import UEDRunner
        import numpy as np

        def value_fn(ego, glob, robot):
            return np.full((len(ego),), value, dtype=np.float32)

        return UEDRunner(value_fn=value_fn, rng=random.Random(1))

    def test_interleaved_workers_do_not_share_a_trajectory(self):
        # Workers put transitions on one shared queue, so an episode key that
        # ignored the worker id would merge two different levels' trajectories.
        runner = self._runner()
        for i in range(5):
            runner.on_transition(_FakeMsg(worker_id=0, episode_idx=3, level_id=11, reward=1.0))
            runner.on_transition(_FakeMsg(worker_id=1, episode_idx=3, level_id=22, reward=2.0))

        self.assertEqual(len(runner._traces), 2)
        self.assertEqual(runner._traces[(0, 3)].level_id, 11)
        self.assertEqual(runner._traces[(1, 3)].level_id, 22)
        self.assertEqual(runner._traces[(0, 3)].ret, 5.0)
        self.assertEqual(runner._traces[(1, 3)].ret, 10.0)

    def test_maxmc_is_the_mean_positive_gap_to_the_best_return(self):
        runner = self._runner(value=2.0)
        for _ in range(4):
            runner.on_transition(_FakeMsg(0, 0, level_id=7, reward=3.0))
        trace = runner._consume_trace(0, 0)
        # Return is 12, the only episode on this level so it is also the best;
        # value is 2 everywhere, so every gap is 10.
        self.assertAlmostEqual(runner._maxmc(trace), 10.0, places=5)

    def test_maxmc_is_zero_when_value_already_exceeds_the_best_return(self):
        runner = self._runner(value=100.0)
        for _ in range(3):
            runner.on_transition(_FakeMsg(0, 0, level_id=8, reward=1.0))
        trace = runner._consume_trace(0, 0)
        self.assertAlmostEqual(runner._maxmc(trace), 0.0, places=5)

    def test_maxmc_reference_is_shared_across_levels(self):
        # A level scored on its first episode must be measured against the best
        # return seen anywhere, otherwise its reference is its own return and
        # the score is not comparable with any other level's.
        runner = self._runner(value=0.0)

        for _ in range(3):
            runner.on_transition(_FakeMsg(0, 0, level_id=1, reward=10.0))
        good = runner._consume_trace(0, 0)
        self.assertAlmostEqual(runner._maxmc(good), 30.0, places=5)

        for _ in range(3):
            runner.on_transition(_FakeMsg(0, 1, level_id=2, reward=1.0))
        poor = runner._consume_trace(0, 1)
        # Return 3 against a best of 30: the gap survives instead of collapsing
        # to zero the way a per-level reference would make it.
        self.assertAlmostEqual(runner._maxmc(poor), 30.0, places=5)
        self.assertAlmostEqual(runner._global_best_return, 30.0, places=5)

    def test_trace_is_released_when_the_episode_reports(self):
        runner = self._runner()
        lv_id = 99
        for _ in range(3):
            runner.on_transition(_FakeMsg(0, 1, level_id=lv_id, reward=1.0))
        self.assertIn((0, 1), runner._traces)
        runner.on_episode(_FakeStat(0, 1, lv_id, evac_time_100=10))
        self.assertNotIn((0, 1), runner._traces)

    def test_abandoned_traces_are_bounded(self):
        # A worker killed mid-episode never reports, so traces must not grow
        # without limit.
        from ued.runner import MAX_TRACKED_EPISODES

        runner = self._runner()
        for ep in range(MAX_TRACKED_EPISODES * 3):
            runner.on_transition(_FakeMsg(0, ep, level_id=ep, reward=1.0))
        self.assertLessEqual(len(runner._traces), MAX_TRACKED_EPISODES)


class ReplayOnlyRuleTest(unittest.TestCase):
    def test_flag_controls_whether_fresh_level_transitions_are_stored(self):
        import config
        from ued.runner import UEDRunner

        runner = UEDRunner(value_fn=None, rng=random.Random(2))
        fresh = _FakeMsg(0, 0, level_id=5, reward=1.0)
        replayed = _FakeMsg(0, 1, level_id=6, reward=1.0)
        replayed.is_replay = True

        original = config.UED_REPLAY_ONLY_UPDATES
        try:
            import ued.runner as runner_mod

            runner_mod.UED_REPLAY_ONLY_UPDATES = False
            self.assertTrue(runner.should_store(fresh))
            self.assertTrue(runner.should_store(replayed))

            runner_mod.UED_REPLAY_ONLY_UPDATES = True
            self.assertFalse(runner.should_store(fresh))
            self.assertTrue(runner.should_store(replayed))
        finally:
            import ued.runner as runner_mod

            runner_mod.UED_REPLAY_ONLY_UPDATES = original



class DifficultyZeroTest(unittest.TestCase):
    """The tier ACCEL is meant to start from: an empty room with two exits."""

    def test_difficulty_zero_has_no_obstacles(self):
        for seed in (1, 7, 99, 12345):
            data = generate_map(RandomMapSpec(100, 100, difficulty=0, seed=seed))
            self.assertEqual(len(data.obstacles), 0, f"seed {seed}")
            self.assertEqual(len(data.exits), 2, f"seed {seed}")

    def test_an_obstacle_free_level_is_playable(self):
        # An open room is trivially reachable everywhere. The validator used to
        # reject it, which would have made difficulty 0 unmutatable.
        lv = generate_random_level(random.Random(3), difficulty=0)
        self.assertEqual(len(lv.obstacles), 0)
        self.assertTrue(is_playable(lv))

    def test_editing_adds_complexity_to_an_empty_room(self):
        # This is the whole point of starting at zero: complexity has to be
        # earned by editing rather than handed over by the generator.
        rng = random.Random(77)
        cur = generate_random_level(rng, difficulty=0)
        self.assertEqual(len(cur.obstacles), 0)
        for _ in range(12):
            try:
                cur = mutate_level(cur, rng=rng)
            except MutationFailed:
                continue
        self.assertGreater(len(cur.obstacles), 0)
        self.assertTrue(is_playable(cur))

    def test_higher_difficulty_is_denser(self):
        rng = random.Random(4)
        means = []
        for difficulty in (0, 1, 3, 6):
            densities = [
                generate_random_level(rng, difficulty=difficulty).density()
                for _ in range(3)
            ]
            means.append(sum(densities) / len(densities))
        self.assertEqual(means, sorted(means), f"densities not monotonic: {means}")


class PersistenceTest(unittest.TestCase):
    """The curriculum has to survive the watchdog restarting the trainer."""

    def _level(self, seed):
        data = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=seed))
        return level_from_map_data(data, crowd_size=30, difficulty=2)

    def test_round_trip_preserves_scores_and_counters(self):
        import tempfile

        from ued.runner import UEDRunner

        saved = UEDRunner(value_fn=None, rng=random.Random(1))
        if not saved.enabled:
            self.skipTest("UED_ENABLED is False")

        levels = [self._level(400 + i) for i in range(4)]
        for lv in levels:
            saved.population.add(lv)
        saved.population.episode = 1234
        for i, lv in enumerate(levels):
            saved.population.update(
                lv.level_id,
                evac_time=(100.0 if i % 2 == 0 else 5000.0),
                freeflow_steps=100.0,
                maxmc=float(i),
            )

        path = os.path.join(tempfile.mkdtemp(), "curriculum.pkl")
        self.assertTrue(saved.save(path))

        loaded = UEDRunner(value_fn=None, rng=random.Random(2))
        self.assertTrue(loaded.load(path))
        self.assertEqual(len(loaded.population), len(levels))
        self.assertEqual(loaded.population.episode, 1234)
        for lv in levels:
            before = saved.population._records[lv.level_id]
            after = loaded.population._records[lv.level_id]
            self.assertEqual(list(before.evac_times), list(after.evac_times))
            self.assertAlmostEqual(before.freeflow_steps, after.freeflow_steps)
            self.assertEqual(before.trials, after.trials)
            self.assertAlmostEqual(before.maxmc, after.maxmc)
            self.assertEqual(len(before.level.obstacles), len(after.level.obstacles))

    def test_new_ids_do_not_collide_with_restored_ones(self):
        # Ids travel to the workers and back inside every transition, so a
        # collision would credit one level's episodes to another.
        import tempfile

        from ued.level import Level
        from ued.runner import UEDRunner

        saved = UEDRunner(value_fn=None, rng=random.Random(1))
        if not saved.enabled:
            self.skipTest("UED_ENABLED is False")
        levels = [self._level(500 + i) for i in range(3)]
        for lv in levels:
            saved.population.add(lv)
        restored_ids = {lv.level_id for lv in levels}

        path = os.path.join(tempfile.mkdtemp(), "curriculum.pkl")
        saved.save(path)

        loaded = UEDRunner(value_fn=None, rng=random.Random(2))
        loaded.load(path)

        fresh_ids = {Level(obstacles=[], exits=[], crowd_size=30).level_id for _ in range(5)}
        self.assertFalse(restored_ids & fresh_ids, "new level ids collided with restored ids")

    def test_a_corrupt_state_file_does_not_stop_training(self):
        import tempfile

        from ued.runner import UEDRunner

        path = os.path.join(tempfile.mkdtemp(), "curriculum.pkl")
        with open(path, "wb") as f:
            f.write(b"not a pickle")

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        if not runner.enabled:
            self.skipTest("UED_ENABLED is False")
        self.assertFalse(runner.load(path))
        self.assertEqual(len(runner.population), 0)

    def test_missing_state_file_is_not_an_error(self):
        from ued.runner import UEDRunner

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        self.assertFalse(runner.load("/nonexistent/path/curriculum.pkl"))


class RenderTest(unittest.TestCase):
    def test_grid_renders_levels_including_an_empty_one(self):
        from ued import render

        rng = random.Random(9)
        levels = [
            generate_random_level(rng, difficulty=0),
            generate_random_level(rng, difficulty=3),
        ]
        img = render.render_grid(levels, cols=2)
        self.assertIsNotNone(img)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[2], 3)
        self.assertEqual(img.dtype.name, "uint8")

    def test_empty_input_returns_none(self):
        from ued import render

        self.assertIsNone(render.render_grid([]))

    def test_lineage_walks_surviving_ancestors(self):
        from ued import render
        from ued.population import LevelPopulation

        rng = random.Random(10)
        pop = LevelPopulation(capacity=50)
        parent = generate_random_level(rng, difficulty=1)
        pop.add(parent)
        child = mutate_level(parent, rng=rng)
        pop.add(child)
        grandchild = mutate_level(child, rng=rng)
        pop.add(grandchild)

        chain, titles = render.lineage(pop, grandchild.level_id)
        self.assertEqual([lv.level_id for lv in chain],
                         [parent.level_id, child.level_id, grandchild.level_id])
        self.assertEqual(titles[0].split()[-1], "[origin]")


class SnapshotTest(unittest.TestCase):
    """Snapshots are diagnostics and must never be able to stop a run."""

    def _runner_with_lineage(self):
        from ued.runner import UEDRunner

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        if not runner.enabled:
            self.skipTest("UED_ENABLED is False")
        rng = random.Random(1)
        base = generate_random_level(rng, difficulty=1)
        runner.population.add(base)
        runner.population.update(base.level_id, evac_time=100.0, freeflow_steps=100.0, maxmc=1.0)
        child = mutate_level(base, rng=rng)
        runner.population.add(child)
        runner.population.update(child.level_id, evac_time=9000.0, freeflow_steps=100.0, maxmc=2.0)
        return runner

    def test_snapshot_writes_images_and_pngs(self):
        import glob
        import tempfile

        from tensorboard.backend.event_processing import event_accumulator
        from torch.utils.tensorboard import SummaryWriter

        runner = self._runner_with_lineage()
        tb_dir, png_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
        writer = SummaryWriter(log_dir=tb_dir)
        self.assertTrue(runner.snapshot(writer, 20, save_dir=png_dir))
        writer.flush()

        self.assertTrue(sorted(os.listdir(png_dir)))
        event_file = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))[0]
        ea = event_accumulator.EventAccumulator(event_file, size_guidance={"images": 0})
        ea.Reload()
        self.assertIn("UED/population_top", ea.Tags().get("images", []))

    def test_a_broken_writer_does_not_raise(self):
        # torch's add_image needs Pillow >= 9.1 for a resize this image never
        # needs, and on an older Pillow it raises. A snapshot that cannot be
        # written has to be reported and swallowed, not propagated into the
        # training loop.
        class ExplodingWriter:
            def _get_file_writer(self):
                raise RuntimeError("boom")

            def add_image(self, *a, **k):
                raise RuntimeError("boom")

            def flush(self):
                pass

        runner = self._runner_with_lineage()
        self.assertFalse(runner.snapshot(ExplodingWriter(), 20))

    def test_snapshot_on_an_unscored_population_is_a_noop(self):
        from ued.runner import UEDRunner

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        if not runner.enabled:
            self.skipTest("UED_ENABLED is False")
        runner.population.add(generate_random_level(random.Random(2), difficulty=1))

        class CountingWriter:
            def __init__(self):
                self.calls = 0

            def _get_file_writer(self):
                self.calls += 1
                raise AssertionError("should not write without scored levels")

            def flush(self):
                pass

        writer = CountingWriter()
        self.assertFalse(runner.snapshot(writer, 20))
        self.assertEqual(writer.calls, 0)


class NormalisedSuccessTest(unittest.TestCase):
    """Success is measured against the level, not against MAX_STEPS.

    A hand-set global cap makes the success rate saturate at 0 or 1 depending
    on how generously it was chosen, and learnability goes to zero with it.
    The reference here is the level's own free-flow evacuation estimate.
    """

    def _level(self, seed=601):
        data = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=seed))
        return level_from_map_data(data, crowd_size=30, difficulty=2)

    def test_free_flow_estimate_is_positive_and_below_the_cap(self):
        import model
        from config import MAX_STEPS

        rng = random.Random(12)
        for difficulty in (0, 3, 6):
            level = generate_random_level(rng, difficulty=difficulty)
            env = model.FightingModel(level.crowd_size, 100, 100, robot="Q", level=level)
            estimate = env.free_flow_evacuation_steps()
            self.assertGreater(estimate, 0.0, f"difficulty {difficulty}")
            self.assertLess(estimate, MAX_STEPS, f"difficulty {difficulty}")

    def test_free_flow_grows_with_crowd_size(self):
        # The queueing term is what makes crowd size a real difficulty axis
        # rather than something the reference ignores.
        import model

        level = generate_random_level(random.Random(13), difficulty=2)
        estimates = []
        for crowd in (20, 40):
            level.crowd_size = crowd
            env = model.FightingModel(crowd, 100, 100, robot="Q", level=level)
            estimates.append(env.free_flow_evacuation_steps())
        self.assertLess(estimates[0], estimates[1])

    def test_threshold_is_a_multiple_of_free_flow(self):
        from config import UED_SUCCESS_K
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=5)
        level = self._level()
        pop.add(level)
        pop.update(level.level_id, evac_time=100.0, freeflow_steps=200.0)
        rec = pop._records[level.level_id]
        self.assertAlmostEqual(rec.threshold(), UED_SUCCESS_K * 200.0)

    def test_the_same_time_can_pass_on_one_level_and_fail_on_another(self):
        # The whole point: an absolute threshold cannot distinguish these.
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=5)
        roomy, tight = self._level(602), self._level(603)
        pop.add(roomy)
        pop.add(tight)
        pop.update(roomy.level_id, evac_time=500.0, freeflow_steps=400.0)
        pop.update(tight.level_id, evac_time=500.0, freeflow_steps=50.0)

        self.assertEqual(pop._records[roomy.level_id].successes(), 1)
        self.assertEqual(pop._records[tight.level_id].successes(), 0)

    def test_a_timeout_fails_however_loose_the_threshold(self):
        from config import MAX_STEPS
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=5)
        level = self._level(604)
        pop.add(level)
        # Free-flow so large that K times it would exceed the cap.
        pop.update(level.level_id, evac_time=float(MAX_STEPS), freeflow_steps=float(MAX_STEPS))
        self.assertEqual(pop._records[level.level_id].successes(), 0)

    def test_threshold_can_be_changed_after_the_fact(self):
        # Raw times are stored, so K is a read-time decision and the history
        # does not have to be thrown away to revisit it.
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=5)
        level = self._level(605)
        pop.add(level)
        for t in (150.0, 250.0, 350.0):
            pop.update(level.level_id, evac_time=t, freeflow_steps=100.0)

        rec = pop._records[level.level_id]
        self.assertEqual(rec.successes(k=2.0), 1)
        self.assertEqual(rec.successes(k=3.0), 2)
        self.assertEqual(rec.successes(k=4.0), 3)

    def test_evac_ratio_is_reported_for_calibrating_k(self):
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=5)
        level = self._level(606)
        pop.add(level)
        pop.update(level.level_id, evac_time=250.0, freeflow_steps=100.0)
        self.assertAlmostEqual(pop._records[level.level_id].evac_ratio, 2.5)
        self.assertIn("mean_evac_ratio", pop.stats())

    def test_learnability_still_peaks_at_balanced_outcomes(self):
        from ued.population import LevelPopulation

        pop = LevelPopulation(capacity=10, score_fn="learnability")
        always, never, mixed = self._level(607), self._level(608), self._level(609)
        for lv in (always, never, mixed):
            pop.add(lv)
        for i in range(8):
            pop.update(always.level_id, evac_time=120.0, freeflow_steps=100.0)
            pop.update(never.level_id, evac_time=5000.0, freeflow_steps=100.0)
            pop.update(mixed.level_id,
                       evac_time=(120.0 if i % 2 == 0 else 5000.0),
                       freeflow_steps=100.0)

        score = {lv.level_id: pop._raw_score(pop._records[lv.level_id])
                 for lv in (always, never, mixed)}
        self.assertGreater(score[mixed.level_id], score[always.level_id])
        self.assertGreater(score[mixed.level_id], score[never.level_id])

    def test_old_curriculum_state_is_rejected_rather_than_misread(self):
        import pickle
        import tempfile

        from ued.runner import UEDRunner

        path = os.path.join(tempfile.mkdtemp(), "old.pkl")
        with open(path, "wb") as f:
            pickle.dump({"version": 1, "population": {"version": 1, "records": []}}, f)

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        if not runner.enabled:
            self.skipTest("UED_ENABLED is False")
        self.assertFalse(runner.load(path))
