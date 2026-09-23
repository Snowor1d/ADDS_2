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
            # No lower bound on obstacles: difficulty 0 is an empty room, and a
            # lineage that starts there legitimately has none until editing
            # adds some.
            self.assertGreaterEqual(len(child.obstacles), 0)
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

        env = model.FightingModel(level.crowd_size, level.width, level.height, robot="Q", level=level)
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
        a = model.FightingModel(level.crowd_size, level.width, level.height, robot="Q", level=level)
        b = model.FightingModel(level.crowd_size, level.width, level.height, robot="Q", level=level)
        self.assertEqual(a.obstacles, b.obstacles)
        self.assertEqual(a.exit_list, b.exit_list)

    def test_augmentation_is_pinned_per_level(self):
        import model

        rng = random.Random(32)
        level = generate_random_level(rng)
        level.augmentation = "rotate_90"
        for _ in range(3):
            env = model.FightingModel(level.crowd_size, level.width, level.height, robot="Q", level=level)
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
        # Per family, not across the mixture. Two generator families now share
        # the difficulty scale and street_map is denser than random_map at the
        # same tier, so a handful of samples drawn across both is not monotonic
        # in difficulty even when each family is. The property being asserted
        # is the family's own difficulty ordering.
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        from random_map import RandomMapSpec, generate_map
        from street_map import generate_street_map, spec_from_difficulty

        def coverage(rings, size):
            polys = [Polygon(r) for r in rings if len(r) >= 3]
            if not polys:
                return 0.0
            return unary_union(polys).area / float(size * size)

        rng = random.Random(4)
        for family in ("random_map", "street_map"):
            means = []
            for difficulty in (0, 1, 3, 6):
                shots = []
                for k in range(4):
                    seed = rng.randrange(1 << 30)
                    if family == "random_map":
                        data = generate_map(RandomMapSpec(200, 200,
                                                          difficulty=difficulty,
                                                          seed=seed))
                    else:
                        spec = spec_from_difficulty(200, 200, difficulty, rng)
                        spec.seed = seed
                        data = generate_street_map(spec)
                    shots.append(coverage(data.obstacles, 200))
                means.append(sum(shots) / len(shots))
            self.assertEqual(means, sorted(means),
                             f"{family} densities not monotonic: {means}")


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
            env = model.FightingModel(level.crowd_size, level.width, level.height, robot="Q", level=level)
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
            env = model.FightingModel(crowd, level.width, level.height, robot="Q", level=level)
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


class SizeGeneralisationTest(unittest.TestCase):
    """Everything map size touches: raster, observation, generation, editing."""

    def test_raster_is_one_pixel_per_metre_at_any_size(self):
        import model

        rng = random.Random(41)
        for size in (60, 100, 170):
            level = generate_random_level(rng, difficulty=2, width=size, height=size)
            env = model.FightingModel(level.crowd_size, size, size, robot="Q", level=level)
            img = env.return_current_image()
            self.assertEqual(img.shape, (size, size), f"size {size}")

    def test_small_map_does_not_raise(self):
        # return_current_image used to rasterise against the MAP_W/MAP_H
        # constants while the grid was world-sized, so a 60-wide map wrote
        # index 82 into a 60-long axis and raised IndexError.
        import model

        level = generate_random_level(random.Random(42), difficulty=2, width=60, height=60)
        env = model.FightingModel(level.crowd_size, 60, 60, robot="Q", level=level)
        img = env.return_current_image()
        self.assertEqual(img.shape, (60, 60))
        self.assertGreater(int(img.max()), 0)

    def test_legacy_fixed_size_callers_still_get_that_size(self):
        import model

        level = generate_random_level(random.Random(43), difficulty=2, width=140, height=140)
        env = model.FightingModel(level.crowd_size, 140, 140, robot="Q", level=level)
        self.assertEqual(env.return_current_image(100, 100).shape, (100, 100))

    def test_robot_state_carries_a_symmetric_scale(self):
        import math

        import model
        from config import MAP_SIZE_REFERENCE

        rng = random.Random(44)
        for size in (60, 100, 160):
            level = generate_random_level(rng, difficulty=2, width=size, height=size)
            env = model.FightingModel(level.crowd_size, size, size, robot="Q", level=level)
            state = env.return_current_robot_state()
            self.assertEqual(len(state), 5, f"size {size}")
            self.assertAlmostEqual(state[3], math.log(size / MAP_SIZE_REFERENCE), places=6)
            self.assertAlmostEqual(state[4], math.log(size / MAP_SIZE_REFERENCE), places=6)
            # Positions stay normalised; the old constant divisor exceeded 1.
            self.assertGreaterEqual(state[0], 0.0)
            self.assertLessEqual(state[0], 1.0)
            self.assertLessEqual(state[1], 1.0)

    def test_observation_tensor_shape_is_size_independent(self):
        # The replay buffer stores fixed-shape tensors, so the ego crop and the
        # pooled global view must not change shape with the world.
        import model
        from ADDS_AS_reinforcement import (
            DOWNSAMPLE_MAP_SIZE as _d,
            ego_crop_from_full_map,
            downsample_full_map,
        )
        from config import DOWNSAMPLE_MAP_SIZE, EGO_MAP_SIZE

        rng = random.Random(45)
        for size in (60, 100, 170):
            level = generate_random_level(rng, difficulty=2, width=size, height=size)
            env = model.FightingModel(level.crowd_size, size, size, robot="Q", level=level)
            full = env.return_current_image()
            ix, iy = env.world_to_px(env.robot.xy[0], env.robot.xy[1])
            ego = ego_crop_from_full_map(full, (ix, iy), EGO_MAP_SIZE, pad_value=50)
            glob = downsample_full_map(full, DOWNSAMPLE_MAP_SIZE)
            self.assertEqual(ego.shape, (EGO_MAP_SIZE, EGO_MAP_SIZE), f"size {size}")
            self.assertEqual(glob.shape, (DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE), f"size {size}")

    def test_generator_spacing_scales_with_the_world(self):
        from random_map import _params_from_difficulty

        gaps = [_params_from_difficulty(s, s, 3)["min_obstacle_gap"] for s in (70, 100, 140, 200)]
        self.assertEqual(gaps, sorted(gaps))
        # Never below what the robot needs to pass, whatever the map size.
        self.assertGreaterEqual(_params_from_difficulty(40, 40, 3)["min_obstacle_gap"], 4.0)

    def test_generation_works_across_the_size_range(self):
        from random_map import RandomMapSpec, generate_map

        for size in (60, 100, 140, 200):
            data = generate_map(RandomMapSpec(size, size, difficulty=3, seed=31))
            self.assertEqual(data.width, size)
            self.assertGreater(len(data.obstacles), 0, f"size {size}")

    def test_sampled_levels_stay_inside_the_configured_range(self):
        from config import UED_MAP_SIZE_RANGE

        lo, hi = UED_MAP_SIZE_RANGE
        rng = random.Random(46)
        for _ in range(12):
            level = generate_random_level(rng)
            self.assertGreaterEqual(level.width, lo)
            self.assertLessEqual(level.width, hi)
            self.assertEqual(level.width, level.height)

    def test_canvas_operator_moves_a_wall_and_keeps_obstacles(self):
        """Resizing moves the crop edge; it does not scale the city.

        Scaling the geometry is the obvious way to resize a level and it is
        wrong here. Street widths and block sizes are absolute metres tied to
        the robot's body and to what real blocks measure, so scaling a level
        down gives corridors the robot cannot enter and scaling one up gives
        blocks no city has. Moving the crop edge instead reveals or hides
        fabric exactly as moving a window over a real city would.
        """
        from citygen.mutate import op_resize_canvas
        from ued.level import generate_city_level

        level = generate_city_level(random.Random(60), difficulty=4,
                                    crowd_size=30, width=100, height=100,
                                    morphology="grid")
        plan = level.plan
        widths_before = sorted(round(st.width_m, 4) for st in plan.streets)

        changed = 0
        for i in range(40):
            work = plan.copy()
            if op_resize_canvas(work, random.Random(i)) is None:
                continue
            changed += 1
            self.assertNotEqual((work.width, work.height),
                                (plan.width, plan.height))
            # The fabric itself is untouched: same streets, same widths.
            self.assertEqual(sorted(round(st.width_m, 4) for st in work.streets),
                             widths_before)
            # And it stays inside the sizes the curriculum trains on.
            from config import UED_MAP_SIZE_RANGE
            lo, hi = UED_MAP_SIZE_RANGE
            self.assertGreaterEqual(work.width, lo)
            self.assertLessEqual(work.width, hi)
        self.assertGreater(changed, 0, "resize never fired")

    def test_free_flow_estimate_grows_with_the_world(self):
        # The family is pinned and several layouts are averaged per size. Free
        # flow is set by the worst-case distance to an exit, which depends on
        # the layout at least as much as on the world: a medina at 120 m can
        # have a longer worst path than a superblock at 180 m, so a single
        # sample per size crossing morphologies is not monotonic even though
        # the criterion does scale with the world. Pin the morphology and take
        # a median.
        import statistics

        import model

        from ued.level import generate_city_level

        rng = random.Random(48)
        estimates = []
        for size in (70, 120, 180):
            shots = []
            for _ in range(3):
                level = generate_city_level(rng, difficulty=3,
                                            crowd_size=30, width=size,
                                            height=size, morphology="grid")
                level.crowd_size = 30
                env = model.FightingModel(30, size, size, robot="Q", level=level)
                shots.append(env.free_flow_evacuation_steps())
            estimates.append(statistics.median(shots))
        self.assertEqual(estimates, sorted(estimates), f"{estimates}")

    def test_holdout_spans_sizes_inside_and_outside_training(self):
        from config import UED_MAP_SIZE_RANGE
        from ued.holdout import holdout_levels

        lo, hi = UED_MAP_SIZE_RANGE
        bands = {getattr(lv, "size_band", "inside") for lv in holdout_levels()}
        self.assertIn("inside", bands)
        self.assertTrue({"below", "above"} & bands,
                        "holdout has no sizes outside the training range")
        sizes = {lv.width for lv in holdout_levels()}
        self.assertTrue(any(s < lo for s in sizes) or any(s > hi for s in sizes))


class AccelTuningTest(unittest.TestCase):
    """The breeding rule, which had to be rethought for this task."""

    def _pop(self):
        from ued.population import LevelPopulation

        return LevelPopulation(capacity=200)

    def _add_scored(self, pop, seed, evac_time, trials, freeflow=100.0):
        # City levels, because breeding needs a street plan to edit. A level
        # built through level_from_map_data has none, which is what
        # test_a_level_without_a_plan_is_never_bred checks for separately.
        from ued.level import generate_city_level

        level = generate_city_level(random.Random(seed), difficulty=4,
                                    crowd_size=30, width=100, height=100,
                                    morphology="grid")
        pop.add(level)
        for _ in range(trials):
            pop.update(level.level_id, evac_time=evac_time, freeflow_steps=freeflow)
        return pop._records[level.level_id]

    def test_an_unmeasured_level_does_not_breed(self):
        # Children inherit their parent's score, so breeding from a level whose
        # score was itself inherited compounds a guess.
        from config import UED_MIN_TRIALS

        pop = self._pop()
        for i in range(12):
            self._add_scored(pop, 700 + i, 150.0 + 40 * (i % 4), UED_MIN_TRIALS)
        fresh = self._add_scored(pop, 799, 200.0, 1)
        self.assertFalse(pop.should_breed(fresh))

    def test_maxmc_units_do_not_outrank_learnability(self):
        # should_breed used to compare raw scores: learnability is bounded by
        # 0.25 while MaxMC is an unbounded return gap, so every barely-tried
        # level beat every measured one.
        from config import UED_MIN_TRIALS

        pop = self._pop()
        frontier = None
        for i in range(12):
            # Alternating outcomes give a success rate near a half, the highest
            # learnability there is.
            from ued.level import generate_city_level

            level = generate_city_level(random.Random(800 + i), difficulty=4,
                                        crowd_size=30, width=100, height=100,
                                        morphology="grid")
            pop.add(level)
            for k in range(UED_MIN_TRIALS + 1):
                pop.update(level.level_id,
                           evac_time=(150.0 if k % 2 == 0 else 5000.0),
                           freeflow_steps=100.0)
            if i == 0:
                frontier = pop._records[level.level_id]

        # A level with a huge MaxMC but too few trials must not displace it.
        loud = self._add_scored(pop, 850, 200.0, 1)
        loud.maxmc = 10_000.0
        self.assertTrue(pop.should_breed(frontier))
        self.assertFalse(pop.should_breed(loud))

    def test_children_per_level_is_capped(self):
        from config import UED_MAX_CHILDREN_PER_LEVEL, UED_MIN_TRIALS

        pop = self._pop()
        for i in range(12):
            self._add_scored(pop, 900 + i, 5000.0, UED_MIN_TRIALS)
        from ued.level import generate_city_level

        star = None
        level = generate_city_level(random.Random(950), difficulty=4,
                                    crowd_size=30, width=100, height=100,
                                    morphology="grid")
        pop.add(level)
        for k in range(UED_MIN_TRIALS + 1):
            pop.update(level.level_id,
                       evac_time=(150.0 if k % 2 == 0 else 5000.0),
                       freeflow_steps=100.0)
        star = pop._records[level.level_id]

        bred = 0
        for _ in range(UED_MAX_CHILDREN_PER_LEVEL + 4):
            if not pop.should_breed(star):
                break
            if pop.breed(star) is not None:
                bred += 1
        self.assertLessEqual(bred, UED_MAX_CHILDREN_PER_LEVEL)
        self.assertEqual(bred, UED_MAX_CHILDREN_PER_LEVEL)
        self.assertFalse(pop.should_breed(star))

    def test_a_level_without_a_plan_is_never_bred(self):
        """Real-map crops and numbered maps have nothing to edit.

        Letting breed() raise instead would be worse than it looks: a failed
        breed leaves the child count untouched, so the level qualifies again
        on every future visit and burns an attempt each time for as long as it
        stays in the population.
        """
        from config import UED_MIN_TRIALS

        pop = self._pop()
        for i in range(12):
            self._add_scored(pop, 700 + i, 5000.0, UED_MIN_TRIALS)
        data = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=951))
        level = level_from_map_data(data, crowd_size=30, difficulty=2)
        self.assertIsNone(level.plan)
        pop.add(level)
        for k in range(UED_MIN_TRIALS + 1):
            pop.update(level.level_id,
                       evac_time=(150.0 if k % 2 == 0 else 5000.0),
                       freeflow_steps=100.0)
        self.assertFalse(pop.should_breed(pop._records[level.level_id]))


class MutationReproducibilityTest(unittest.TestCase):
    def test_a_child_depends_only_on_the_generator_passed_in(self):
        # The shape samplers reused from random_map draw from the global
        # `random` module, so a mutation used to depend on wherever that stream
        # happened to be, and the same seed gave different children in
        # different contexts.
        level = generate_random_level(random.Random(5), difficulty=2, width=100, height=100)

        random.seed(1)
        first = mutate_level(level, rng=random.Random(77))
        random.seed(999999)
        second = mutate_level(level, rng=random.Random(77))

        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.exits, second.exits)
        self.assertEqual((first.width, first.height), (second.width, second.height))
        self.assertEqual(first.crowd_size, second.crowd_size)

    def test_mutating_leaves_the_caller_stream_alone(self):
        level = generate_random_level(random.Random(6), difficulty=2, width=100, height=100)

        random.seed(4242)
        reference = [random.random() for _ in range(5)]
        random.seed(4242)
        mutate_level(level, rng=random.Random(77))
        after = [random.random() for _ in range(5)]

        self.assertEqual(reference, after)


class ConstructionCostTest(unittest.TestCase):
    """The build has to stay cheap enough for size to be a design axis.

    A 200x200 world took 39 s to construct before the navmesh containment
    test, the visibility atlas and the all-pairs distances were vectorised,
    which made large maps unusable as training levels regardless of anything
    else. These bounds are loose because the numbers move with machine load;
    they exist to catch a return to per-grid-point Python loops.
    """

    def _build_seconds(self, size):
        import time

        import model
        from random_map import RandomMapSpec, generate_map

        data = generate_map(RandomMapSpec(size, size, difficulty=3, seed=7))
        level = level_from_map_data(data, crowd_size=30, difficulty=3)
        start = time.time()
        model.FightingModel(30, size, size, robot="Q", level=level)
        return time.time() - start

    def test_a_large_world_builds_quickly(self):
        self.assertLess(self._build_seconds(200), 5.0)

    def test_cost_does_not_explode_with_area(self):
        small = self._build_seconds(100)
        large = self._build_seconds(200)
        # Four times the area. The original scaled far worse than that.
        self.assertLess(large, max(0.5, small * 12.0), f"{small:.2f}s -> {large:.2f}s")


class NetworkRobotDimTest(unittest.TestCase):
    """The networks must actually use the robot-state width they are given.

    Both QNetwork and PolicyNetwork accepted a robot_dim argument and then
    built the embedding at a hardcoded 3. That was invisible while the robot
    state was three numbers wide; adding the two map-scale terms for size
    generalisation made the critic reject its own observations at runtime.
    """

    def test_networks_match_the_configured_robot_state_width(self):
        import ADDS_AS_reinforcement as trainer

        agent = trainer.SACAgent(input_shape=(50, 50))
        for name, net in (("q1", agent.q1), ("q2", agent.q2), ("policy", agent.policy)):
            widths = [
                p.shape[1] for n, p in net.named_parameters()
                if n.endswith("robot_fc.0.weight")
            ]
            self.assertTrue(widths, f"{name} has no robot branch")
            self.assertEqual(widths[0], trainer.ROBOT_STATE_DIM, name)

    def test_value_fn_runs_on_a_real_observation_width(self):
        import numpy as np

        import ADDS_AS_reinforcement as trainer
        from config import DOWNSAMPLE_MAP_SIZE, EGO_MAP_SIZE

        agent = trainer.SACAgent(input_shape=(50, 50))
        value_fn = trainer.make_value_fn(agent)
        batch = 4
        values = value_fn(
            np.zeros((batch, 4, EGO_MAP_SIZE, EGO_MAP_SIZE), dtype=np.float32),
            np.zeros((batch, 4, DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE), dtype=np.float32),
            np.zeros((batch, trainer.ROBOT_STATE_DIM), dtype=np.float32),
        )
        self.assertEqual(values.shape, (batch,))

    def test_environment_state_width_matches_the_networks(self):
        import model

        import ADDS_AS_reinforcement as trainer

        level = generate_random_level(random.Random(70), difficulty=2, width=90, height=90)
        env = model.FightingModel(level.crowd_size, 90, 90, robot="Q", level=level)
        self.assertEqual(len(env.return_current_robot_state()), trainer.ROBOT_STATE_DIM)


class ExplorationGateTest(unittest.TestCase):
    """Scores must describe the policy, not the exploration noise.

    START_EPSILON is 1.0 and decays to 0 over thousands of episodes, so for a
    long stretch nearly every action is random. A success rate measured then
    says which levels a random walker happens to solve, which has nothing to
    do with where the policy's ability ends, and breeding from it picks parents
    for the wrong reason.
    """

    def _runner(self):
        from ued.runner import UEDRunner

        runner = UEDRunner(value_fn=None, rng=random.Random(1))
        if not runner.enabled:
            self.skipTest("UED_ENABLED is False")
        return runner

    def _level(self, seed=1300):
        data = generate_map(RandomMapSpec(100, 100, difficulty=2, seed=seed))
        return level_from_map_data(data, crowd_size=30, difficulty=2)

    def test_exploratory_episodes_do_not_score_levels(self):
        from config import UED_SCORE_MAX_EPSILON

        runner = self._runner()
        level = self._level()
        runner.population.add(level)

        runner.epsilon = min(1.0, UED_SCORE_MAX_EPSILON + 0.5)
        runner.on_episode(_FakeStat(0, 0, level.level_id, evac_time_100=150))
        self.assertEqual(runner.population._records[level.level_id].trials, 0)
        self.assertEqual(runner.n_unscored_episodes, 1)

    def test_scores_are_recorded_once_exploration_has_decayed(self):
        from config import UED_SCORE_MAX_EPSILON

        runner = self._runner()
        level = self._level(1301)
        runner.population.add(level)

        runner.epsilon = UED_SCORE_MAX_EPSILON
        runner.on_episode(_FakeStat(0, 0, level.level_id, evac_time_100=150))
        self.assertEqual(runner.population._records[level.level_id].trials, 1)


class TrialAgeWindowTest(unittest.TestCase):
    """A level's history must stay attached to a recognisable policy.

    The window used to be a fixed number of trials. At a few hundred live
    levels those trials span more episodes than the entire epsilon decay, so a
    success rate averaged over policies that no longer existed.
    """

    def _pop(self):
        from ued.population import LevelPopulation

        return LevelPopulation(capacity=50)

    def _level(self, seed=1400):
        # A city level, because breeding needs a street plan to edit and one
        # of these tests asks whether a level is breedable.
        from ued.level import generate_city_level

        return generate_city_level(random.Random(seed), difficulty=4,
                                   crowd_size=30, width=100, height=100,
                                   morphology="grid")

    def test_outcomes_older_than_the_window_stop_counting(self):
        from config import UED_TRIAL_MAX_AGE

        pop = self._pop()
        level = self._level()
        pop.add(level)

        # Two failures long ago, then two successes recently.
        pop.episode = 100
        for _ in range(2):
            pop.update(level.level_id, evac_time=5000.0, freeflow_steps=100.0, episode=100)
        pop.episode = 100 + UED_TRIAL_MAX_AGE + 500
        for _ in range(2):
            pop.update(level.level_id, evac_time=150.0, freeflow_steps=100.0,
                       episode=pop.episode)

        rec = pop._records[level.level_id]
        self.assertEqual(rec.fresh_trials(pop.episode), 2)
        # Only the recent successes count, so the rate is high.
        self.assertGreater(rec.success_rate_at(pop.episode), 0.5)
        # Ignoring age would mix in the old failures and halve it.
        self.assertLess(rec.success_rate_at(None), rec.success_rate_at(pop.episode))

    def test_breeding_needs_recent_trials_not_just_old_ones(self):
        from config import UED_MIN_TRIALS, UED_TRIAL_MAX_AGE

        pop = self._pop()
        levels = []
        for i in range(12):
            lv = self._level(1410 + i)
            pop.add(lv)
            levels.append(lv)
            for k in range(UED_MIN_TRIALS + 1):
                pop.update(lv.level_id,
                           evac_time=(150.0 if k % 2 == 0 else 5000.0),
                           freeflow_steps=100.0, episode=50)
        pop.episode = 50
        rec = pop._records[levels[0].level_id]
        self.assertTrue(pop.should_breed(rec))

        # Move far past the window: the same history is now too old to breed on.
        pop.episode = 50 + UED_TRIAL_MAX_AGE + 1000
        self.assertEqual(rec.fresh_trials(pop.episode), 0)
        self.assertFalse(pop.should_breed(rec))

    def test_age_window_survives_a_checkpoint(self):
        import tempfile

        from ued.runner import UEDRunner

        saved = UEDRunner(value_fn=None, rng=random.Random(1))
        if not saved.enabled:
            self.skipTest("UED_ENABLED is False")
        level = self._level(1450)
        saved.population.add(level)
        saved.population.episode = 900
        saved.population.update(level.level_id, evac_time=150.0,
                                freeflow_steps=100.0, episode=900)

        path = os.path.join(tempfile.mkdtemp(), "curriculum.pkl")
        self.assertTrue(saved.save(path))

        loaded = UEDRunner(value_fn=None, rng=random.Random(2))
        self.assertTrue(loaded.load(path))
        rec = loaded.population._records[level.level_id]
        self.assertEqual(list(rec.evac_episodes), [900])
        self.assertEqual(rec.fresh_trials(900), 1)


class CurriculumTimingTest(unittest.TestCase):
    """The schedules have to be consistent with each other.

    Three independent knobs decide whether the curriculum ever gets a usable
    score: how long exploration dominates, how many live levels the replay
    budget is spread across, and how long an outcome stays valid. Set them
    without reference to each other and the curriculum quietly never engages,
    which no single-value assertion would catch.
    """

    @staticmethod
    def _epsilon(episode):
        from config import (EPSILON_MIN, LINEARLY_DECAY_STEP, START_DECAY_STEP,
                            START_EPSILON)

        if episode < START_DECAY_STEP:
            return START_EPSILON
        decayed = START_EPSILON - (episode - START_DECAY_STEP) / LINEARLY_DECAY_STEP
        return max(EPSILON_MIN, decayed)

    def test_a_level_can_reach_the_breeding_bar_before_its_history_expires(self):
        # Replays are spread over the live population, so a level collects
        # roughly (window * replay_rate / population) trials before the oldest
        # ones stop counting. If that is below UED_MIN_TRIALS the window
        # expires faster than a level can earn a score and nothing ever breeds.
        from config import (UED_MIN_TRIALS, UED_POP_SIZE, UED_P_NEW_END,
                            UED_TRIAL_MAX_AGE)

        if UED_TRIAL_MAX_AGE <= 0:
            self.skipTest("age window disabled")
        trials_per_window = UED_TRIAL_MAX_AGE * (1.0 - UED_P_NEW_END) / UED_POP_SIZE
        self.assertGreater(
            trials_per_window, 1.5 * UED_MIN_TRIALS,
            f"{trials_per_window:.1f} trials per window vs a bar of {UED_MIN_TRIALS}",
        )

    def test_the_count_window_is_the_binding_one_for_active_levels(self):
        # The deque cap should bite before the age filter for a level that is
        # sampled at the average rate, leaving the age filter as the safety net
        # for rarely sampled levels whose scores would otherwise never refresh.
        from config import (UED_POP_SIZE, UED_P_NEW_END, UED_TRIAL_HISTORY,
                            UED_TRIAL_MAX_AGE)

        if UED_TRIAL_MAX_AGE <= 0:
            self.skipTest("age window disabled")
        rate = (1.0 - UED_P_NEW_END) / UED_POP_SIZE     # trials per episode
        episodes_for_full_history = UED_TRIAL_HISTORY / rate
        self.assertLess(episodes_for_full_history, UED_TRIAL_MAX_AGE)

    def test_scoring_waits_for_exploration_to_decay(self):
        # Scoring must not begin while most actions are still random.
        from config import UED_SCORE_MAX_EPSILON, UED_WARMUP_EPISODES

        self.assertLessEqual(UED_SCORE_MAX_EPSILON, 0.5)
        start = next(ep for ep in range(200000)
                     if self._epsilon(ep) <= UED_SCORE_MAX_EPSILON)
        # The gate, not UED_WARMUP_EPISODES, is what actually holds scoring
        # back; warm-up alone ends far too early to be the protection.
        self.assertGreater(start, UED_WARMUP_EPISODES)

    def test_breeding_becomes_reachable_within_a_sane_horizon(self):
        from config import (UED_MIN_TRIALS, UED_POP_SIZE, UED_P_NEW_END,
                            UED_SCORE_MAX_EPSILON)

        start = next(ep for ep in range(200000)
                     if self._epsilon(ep) <= UED_SCORE_MAX_EPSILON)
        rate = (1.0 - UED_P_NEW_END)
        # Eight levels must each reach the trial bar before breeding unlocks.
        episodes_needed = UED_MIN_TRIALS * 8 / rate
        self.assertLess(start + episodes_needed, 20000)


class CityGeneratorTest(unittest.TestCase):
    """The generator produces street networks at real densities.

    Measured against road-derived real crops, which is the only comparison
    that means anything now that both sides define traversable space the same
    way: buffer the street centrelines, and the obstacles are the complement.
    """

    def _stats(self, polys, size):
        from osm_corpus.stats import layout_stats

        return layout_stats(polys, float(size))

    def test_every_morphology_hits_its_measured_street_share(self):
        """The one number the whole generator is pinned to.

        Street share decides how much room the crowd has, so a generator that
        misses it produces maps that look like a city and do not behave like
        one. An early version picked widths from street-class bands and
        derived the spacing from them; for a colonial grid that demanded a
        spacing wider than the crop, the calibration pushed it until no street
        survived, and the pattern rendered as one solid block covering the
        whole map. This is the test that would have caught it.
        """
        import random
        import statistics

        from citygen.generate import _street_share, calibrated_network
        from citygen.morphology import MORPHOLOGIES, STREET_SHARE
        from citygen.plan import CityPlan

        for morph in sorted(MORPHOLOGIES):
            shares = []
            for seed in range(6):
                streets = calibrated_network(morph, 200, 200,
                                             random.Random(seed), 1.0)
                self.assertTrue(streets, f"{morph} produced no streets")
                plan = CityPlan(width=200, height=200, morphology=morph,
                                development=1.0, streets=streets)
                shares.append(_street_share(plan))
            got = statistics.median(shares)
            self.assertAlmostEqual(got, STREET_SHARE[morph], delta=0.04,
                                   msg=f"{morph}: {got:.3f}")

    def test_difficulty_zero_is_an_empty_field(self):
        """Not a special case, but built fraction zero."""
        import random

        from citygen.generate import generate_city_plan
        from citygen.morphology import MORPHOLOGIES

        for morph in sorted(MORPHOLOGIES):
            plan = generate_city_plan(random.Random(4), morph, difficulty=0,
                                      width=140, height=140)
            rings, trav = plan.render()
            self.assertEqual(rings, [], f"{morph} at difficulty 0")
            self.assertIsNotNone(trav)
            self.assertAlmostEqual(trav.area, 140 * 140, delta=1.0)

    def test_coverage_rises_with_difficulty(self):
        """The axis the curriculum walks, from open field to downtown."""
        import random
        import statistics

        from citygen.generate import generate_city_plan
        from citygen.morphology import MORPHOLOGIES

        morphs = sorted(MORPHOLOGIES)
        by_difficulty = []
        for d in range(7):
            covs = []
            for seed in range(14):
                rng = random.Random(1000 + seed)
                morph = morphs[seed % len(morphs)]
                plan = generate_city_plan(rng, morph, difficulty=d,
                                          width=140, height=140)
                covs.append(plan.summary()["coverage"])
            by_difficulty.append(statistics.median(covs))
        self.assertEqual(by_difficulty, sorted(by_difficulty), by_difficulty)
        self.assertEqual(by_difficulty[0], 0.0)
        # The top of the axis has to reach real density or the curriculum
        # cannot present the thing it is meant to transfer to. Road-derived
        # real crops run from about 0.56 to 0.85 built.
        self.assertGreater(by_difficulty[-1], 0.40)

    def test_blocks_are_solid(self):
        """No free space inside a block, which is the extraction's own rule.

        A road-derived real block is one polygon however many buildings stand
        on it, because nothing inside it is mapped carriageway. The earlier
        street generator split each block into separate footprints with
        setbacks between them, and those gaps were free space: it reproduced
        inside blocks exactly the overstatement of navigable space that
        road-based extraction was introduced to remove from between them.
        """
        import random

        from shapely.geometry import Polygon

        from citygen.generate import generate_city_plan
        from citygen.morphology import MORPHOLOGIES

        for morph in sorted(MORPHOLOGIES):
            plan = generate_city_plan(random.Random(9), morph, difficulty=6,
                                      width=200, height=200)
            rings, _ = plan.render()
            for ring in rings:
                poly = Polygon(ring)
                # An exterior ring only: a hole would be ground enclosed by
                # the block, which nothing can reach from the street.
                self.assertEqual(len(poly.interiors), 0, morph)

    def test_corridors_admit_the_robot(self):
        """Generated streets are passable by the robot's body, not just a point."""
        import random

        from config import ROBOT_BODY_RADIUS
        from citygen.generate import _place_exits, generate_city_plan
        from citygen.morphology import MIN_CORRIDOR_M, MORPHOLOGIES
        from citygen.validate import check

        self.assertGreater(MIN_CORRIDOR_M, 2.0 * ROBOT_BODY_RADIUS)
        failures = []
        for morph in sorted(MORPHOLOGIES):
            for seed in range(4):
                rng = random.Random(300 + seed)
                plan = generate_city_plan(rng, morph, difficulty=6,
                                          width=160, height=160)
                exits = _place_exits(plan, 2, rng)
                ok, why = check(plan, exits)
                if not ok:
                    failures.append(f"{morph}/{seed}: {why}")
        # Generation retries, so the entry point is allowed the occasional
        # unplayable draw; a pattern that fails most of the time is not.
        self.assertLess(len(failures), 4, failures)

    def test_developed_layouts_match_the_corpus_density(self):
        """Difficulty 6 against the real crops the tables were fitted to.

        Compared at difficulty 6 only. The generator spans an empty field to a
        full downtown while the corpus holds developed fabric exclusively, so
        pooling difficulties compares two different populations: at difficulty
        4 the median free-space width is 40 m, which is not a defect, it is an
        open field.

        Built coverage matches almost exactly, 0.670 generated against 0.672
        real. Two gaps remain and are asserted loosely on purpose rather than
        tuned away. The generator subdivides more finely, about 27 blocks in a
        200 m crop against 12 real, because each morphology adds crossings on
        top of its grid (alleys, medina dead ends, boulevard radials) and the
        calibration then narrows every street to hold the street share. So its
        corridors are narrower too, 5.7 m against 8.0 m. Finer grain at the
        same density is a harder level, not a wrong one.
        """
        import statistics

        from osm_corpus.collect import load_corpus
        from osm_corpus.report import corpus_rows, generator_sample

        corpus = load_corpus()
        if corpus.get("traversability") != "roads":
            self.skipTest("corpus is not road-derived")
        real = corpus_rows(corpus, 200)
        if len(real) < 10:
            self.skipTest("not enough real crops at 200 m")

        gen = generator_sample(n=21, size_m=200, difficulties=(6,))
        self.assertTrue(gen)

        real_cov = statistics.median(r["coverage"] for r in real)
        gen_cov = statistics.median(r["coverage"] for r in gen)
        self.assertAlmostEqual(gen_cov, real_cov, delta=0.10,
                               msg=f"coverage {gen_cov:.3f} vs {real_cov:.3f}")

        real_w = statistics.median(r["width_p50"] for r in real)
        gen_w = statistics.median(r["width_p50"] for r in gen)
        self.assertLess(gen_w, real_w * 2.0, (gen_w, real_w))
        self.assertGreater(gen_w, real_w * 0.5, (gen_w, real_w))


class MorphologyLineageTest(unittest.TestCase):
    """A child of a city is a city.

    This is the property the whole representation exists for. ACCEL builds
    complexity entirely through offspring, so whatever mutation preserves is
    what a lineage converges on, and the previous polygon operators preserved
    nothing structural.
    """

    def _lineage(self, morph, difficulty, generations, seed):
        from ued.level import generate_city_level
        from ued.mutate import MutationFailed, mutate_level

        rng = random.Random(seed)
        level = generate_city_level(rng, difficulty=difficulty, crowd_size=30,
                                    width=140, height=140, morphology=morph)
        out, failures = [level], 0
        for g in range(generations):
            try:
                level = mutate_level(level, rng=random.Random(seed * 97 + g))
            except MutationFailed:
                failures += 1
                continue
            out.append(level)
        return out, failures

    def test_lineage_stays_a_street_network(self):
        from citygen.morphology import MORPHOLOGIES

        for morph in sorted(MORPHOLOGIES):
            lineage, failures = self._lineage(morph, 4, 12, seed=11)
            self.assertLess(failures, 6, f"{morph}: {failures} failed breeds")
            last = lineage[-1]
            self.assertIsNotNone(last.plan)
            self.assertEqual(last.plan.morphology, morph)
            self.assertTrue(last.plan.streets,
                            f"{morph} lost its street network")
            self.assertGreater(last.generation, 0)

    def test_mutation_is_reproducible_from_its_generator_alone(self):
        """No hidden dependency on the global random stream.

        The previous operators borrowed shape samplers from `random_map` that
        drew from the `random` module directly, so the same seed gave
        different children depending on where the global stream happened to
        be, and the fix was to seed and restore it around every call. The plan
        operators take their generator as an argument, so this now follows
        from the signature rather than from a guard.
        """
        from ued.level import generate_city_level
        from ued.mutate import mutate_level

        level = generate_city_level(random.Random(5), difficulty=4,
                                    crowd_size=30, width=140, height=140,
                                    morphology="grid")
        random.seed(1234)
        first = mutate_level(level, rng=random.Random(77))
        random.seed(999)
        [random.random() for _ in range(50)]
        second = mutate_level(level, rng=random.Random(77))
        self.assertEqual(first.obstacles, second.obstacles)
        self.assertEqual(first.mutation_ops, second.mutation_ops)

    def test_operators_come_in_inverse_pairs(self):
        """Every add has a remove, or the walk drifts without selection.

        Mutation on the scatter generator collapsed complexity because the
        removing operators always validated and the adding ones often failed.
        The pairing is the structural answer, and it breaks loudly here if an
        operator is added without its inverse.
        """
        from citygen.mutate import INVERSE_PAIRS, OPERATORS

        paired = {name for pair in INVERSE_PAIRS for name in pair}
        unpaired = set(OPERATORS) - paired
        # Two operators are their own inverse, because each draws its
        # displacement symmetrically about zero: resize_canvas steps the crop
        # edge either way, and bend_street offsets a street either way. They
        # need no partner to keep the walk unbiased.
        self.assertEqual(unpaired, {"resize_canvas", "bend_street"}, unpaired)
        for a, b in INVERSE_PAIRS:
            self.assertIn(a, OPERATORS)
            self.assertIn(b, OPERATORS)

    def test_a_developed_parent_never_breeds_an_empty_child(self):
        """The complexity collapse the pairing exists to prevent.

        An empty field passes validation, because an empty field is playable.
        It is still not an acceptable child of a developed parent: the
        curriculum would watch a difficulty-4 lineage quietly become a blank
        map while every check reported success.
        """
        from ued.level import generate_city_level
        from ued.mutate import MutationFailed, mutate_level

        level = generate_city_level(random.Random(31), difficulty=4,
                                    crowd_size=30, width=120, height=120,
                                    morphology="organic")
        for g in range(25):
            try:
                level = mutate_level(level, rng=random.Random(g))
            except MutationFailed:
                continue
            self.assertTrue(level.obstacles,
                            f"generation {level.generation} came out empty")


class RoadTraversabilityTest(unittest.TestCase):
    """The road-network reading of where a robot may drive.

    Deriving free space from buildings answers "is anything built here", which
    is not the question. Between two buildings in a real downtown there is
    usually a private plot, a walled yard, a car park, planting, water or rail,
    and a guidance robot cannot cross any of it. These tests pin the inversion:
    free space is the mapped carriageway and footway, and everything else is
    obstacle until proven otherwise.
    """

    def test_width_table_covers_the_classes_we_keep(self):
        from osm_corpus.roads import (EXCLUDED_CLASSES, ROAD_WIDTHS,
                                      DEFAULT_WIDTH_M, width_for)
        # Every kept class has an explicit width; the default exists for
        # classes OSM adds later, not as the common case.
        for cls in ROAD_WIDTHS:
            self.assertNotIn(cls, EXCLUDED_CLASSES)
            self.assertGreater(width_for({"highway": cls}), 0.0)
        self.assertEqual(width_for({"highway": "no_such_class"}),
                         DEFAULT_WIDTH_M)
        # An excluded class is not traversable at any width.
        for cls in EXCLUDED_CLASSES:
            self.assertIsNone(width_for({"highway": cls}))

    def test_lane_count_beats_the_class_default(self):
        from osm_corpus.roads import METRES_PER_LANE, width_for
        tagged = width_for({"highway": "residential", "lanes": "4"})
        self.assertGreaterEqual(tagged, 4 * METRES_PER_LANE)
        self.assertGreater(tagged, width_for({"highway": "residential"}))

    def test_explicit_width_tag_wins(self):
        from osm_corpus.roads import width_for
        self.assertAlmostEqual(width_for({"highway": "residential",
                                          "width": "22"}), 22.0, places=3)

    def test_traversable_is_far_smaller_than_the_unbuilt_area(self):
        """The whole point, stated as a number.

        Measured at six downtowns, the gaps between buildings claim two to six
        times the space the street network actually offers. Palermo Soho is the
        extreme: 92 per cent of the crop is unbuilt and 15 per cent is street.
        """
        from shapely.geometry import LineString
        from osm_corpus.roads import RoadNetwork, traversable_polygon

        size = 200.0
        # A single cross of residential streets: two 9 m corridors.
        net = RoadNetwork(
            lines=[(LineString([(0, 100), (200, 100)]), 9.0),
                   (LineString([(100, 0), (100, 200)]), 9.0)],
            areas=[], source="synthetic", n_excluded=0)
        trav = traversable_polygon(net, size)
        # Two 200x9 strips less the shared square, as a fraction of the crop.
        expected = (2 * 200 * 9 - 81) / (size ** 2)
        self.assertAlmostEqual(trav.area / size ** 2, expected, places=2)
        # A building-complement reading of the same place would call almost
        # all of it free.
        self.assertLess(trav.area / size ** 2, 0.2)

    def test_obstacles_are_the_complement(self):
        from shapely.geometry import LineString, Polygon
        from osm_corpus.roads import (RoadNetwork, obstacles_from_traversable,
                                      traversable_polygon)

        size = 100.0
        net = RoadNetwork(lines=[(LineString([(0, 50), (100, 50)]), 10.0)],
                          areas=[], source="synthetic", n_excluded=0)
        trav = traversable_polygon(net, size)
        rings = obstacles_from_traversable(trav, size, simplify_m=0.0)
        blocked = sum(Polygon(r).area for r in rings)
        self.assertAlmostEqual(blocked + trav.area, size ** 2, delta=size)
        # One strip across the middle leaves a block above and below it.
        self.assertEqual(len(rings), 2)

    def test_simplification_keeps_corridors_passable(self):
        """Tolerance is bounded by what a robot needs, not by looks.

        Simplification moves walls, and a wall moved far enough closes a
        corridor that was passable, which changes what the level means without
        changing how it looks. The criterion is the robot's own criterion:
        shrink the free space by the robot's radius and the network must still
        cross the crop in both directions.

        Note what is NOT used here. The tenth percentile of twice the distance
        to the nearest obstacle saturates at one raster cell, so on a one-metre
        grid it reads 2.00 at every site and every tolerance, and a test
        resting on it would pass whatever simplification did.
        """
        import numpy as np
        from scipy.ndimage import binary_erosion
        from shapely.geometry import LineString, Polygon

        from config import OSM_SIMPLIFY_M, ROBOT_BODY_RADIUS
        from osm_corpus.roads import RoadNetwork, road_layout
        from osm_corpus.stats import rasterise

        self.assertLessEqual(OSM_SIMPLIFY_M, 2.0 * ROBOT_BODY_RADIUS,
                             "tolerance may not exceed the robot's width")

        size = 200.0
        step = 0.5
        # A ladder of narrow cross-streets: the structure simplification is
        # most likely to erase.
        lines = [(LineString([(0, y), (200, y)]), 6.0) for y in (40, 100, 160)]
        lines += [(LineString([(x, 0), (x, 200)]), 6.0) for x in (40, 100, 160)]
        net = RoadNetwork(lines=lines, areas=[], source="synthetic",
                          n_excluded=0)

        def passable(tol):
            rings, trav = road_layout(net, size, simplify_m=tol)
            polys = [Polygon(r) for r in rings if len(r) >= 3]
            free = ~rasterise(polys, size, step=step)
            # What the robot's body can occupy, rather than what a point can.
            r_cells = int(round(ROBOT_BODY_RADIUS / step))
            n = 2 * r_cells + 1
            body = np.ones((n, n), dtype=bool)
            fits = binary_erosion(free, structure=body)
            # The robot's centre cannot sit within its own radius of the crop
            # wall, so "reaches the edge" means reaching that band, not the
            # outermost cell.
            edge = r_cells + 1
            spans_x = bool(fits[:, :edge].any() and fits[:, -edge:].any())
            spans_y = bool(fits[:edge, :].any() and fits[-edge:, :].any())
            return fits, spans_x, spans_y, trav.area / size ** 2

        exact_fits, ex, ey, exact_f = passable(0.0)
        simple_fits, sx, sy, simple_f = passable(OSM_SIMPLIFY_M)

        self.assertTrue(ex and ey, "the unsimplified ladder must be crossable")
        self.assertTrue(sx and sy,
                        "simplification closed a corridor the robot needs")
        # And the room to manoeuvre must survive, not merely a thread of it.
        self.assertGreater(simple_fits.sum(), 0.9 * exact_fits.sum())
        self.assertAlmostEqual(simple_f, exact_f, delta=0.03)

    def test_road_crop_is_cheaper_to_simulate(self):
        """Vertex count is the simulation-cost statistic.

        The navmesh triangulation, the visibility atlas and the all-pairs path
        table all scale with obstacle vertices. Real building footprints put
        over a thousand of them in a 400 m crop; blocks bounded by road edges
        put about a hundred, because a block is one polygon however many
        buildings stand on it.
        """
        from shapely.geometry import LineString
        from osm_corpus.roads import RoadNetwork, road_layout
        from osm_corpus.stats import layout_stats

        size = 200.0
        lines = [(LineString([(0, y), (200, y)]), 9.0) for y in (50, 100, 150)]
        lines += [(LineString([(x, 0), (x, 200)]), 9.0) for x in (50, 100, 150)]
        net = RoadNetwork(lines=lines, areas=[], source="synthetic",
                          n_excluded=0)
        rings, _ = road_layout(net, size)
        from shapely.geometry import Polygon
        st = layout_stats([Polygon(r) for r in rings], size)
        self.assertIsNotNone(st)
        # Sixteen rectangular blocks; a per-block outline is a handful of
        # vertices, not a traced facade.
        self.assertLess(st["n_vertices"] / max(1.0, st["n_obstacles"]), 12.0)

    def test_config_switch_is_honoured(self):
        import config
        self.assertIn(config.OSM_TRAVERSABILITY, ("roads", "buildings"))

    def test_collect_rejects_an_unknown_mode(self):
        from osm_corpus.collect import collect
        with self.assertRaises(ValueError):
            collect(keys=["covent_garden"], crop_sizes=[100], mode="magic")
