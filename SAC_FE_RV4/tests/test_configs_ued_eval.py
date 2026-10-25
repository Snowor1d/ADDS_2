"""Configuration handling, the curriculum, and evaluation statistics."""

import glob
import json
import os
import random
import subprocess
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from configs import ConfigError, resolve_config  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_unknown_and_invalid_rejected(self):
        with self.assertRaises(ConfigError):
            resolve_config(overrides={"LRR": 1})
        with self.assertRaises(ConfigError):
            resolve_config(overrides={"UED_METHOD": "paired"})
        with self.assertRaises(ConfigError):
            resolve_config(overrides={"NET_ENCODER": "legacy_rv3", "MAP_SCALE_TO_ACTOR": True})
        with self.assertRaises(ConfigError):
            resolve_config(overrides={"UED_MAP_SIZES_M": 100})

    def test_json_lists_become_tuples(self):
        c = resolve_config(overrides={"UED_MAP_SIZES_M": [70, 100]})
        self.assertEqual(c.UED_MAP_SIZES_M, (70, 100))

    def test_fingerprint(self):
        a = resolve_config()
        self.assertEqual(a.fingerprint(), resolve_config().fingerprint())
        self.assertNotEqual(a.fingerprint(), resolve_config(overrides={"LR": 3e-4}).fingerprint())
        self.assertEqual(a.fingerprint(), resolve_config(overrides={"WANDB_RUN_NAME": "x"}).fingerprint())

    def test_presets_valid(self):
        for p in glob.glob(os.path.join(ROOT, "presets", "*.json")):
            with self.subTest(preset=os.path.basename(p)):
                d = json.load(open(p))
                d.pop("_comment", None)
                resolve_config(overrides=d)

    def test_cli_overrides_reach_children(self):
        code = ("from configs import apply_cli_overrides; apply_cli_overrides(['--set', 'LR=0.25']);"
                "import subprocess, sys;"
                "print(subprocess.run([sys.executable, '-c', 'from config import LR; print(LR)'],"
                " capture_output=True, text=True).stdout.strip())")
        env = dict(os.environ)
        env.pop("RV4_OVERRIDES", None)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                             text=True, env=env).stdout.strip()
        self.assertEqual(out, "0.25")


class UEDTest(unittest.TestCase):
    def setUp(self):
        self.cfg = resolve_config(overrides={"UED_MIN_TRIALS": 2})

    def test_generation_and_check(self):
        from ued.level import Level, check_level, generate_level
        lv = generate_level(12345, 100, 3, 30, timeout_s=30)
        self.assertIsNotNone(lv)
        self.assertTrue(check_level(lv)[0])
        # a wall cutting the map in two strands the far side
        cut = Level(0, 100, 100, [[[48, 0], [52, 0], [52, 100], [48, 100]]],
                    [[(0, 40), (5, 40), (5, 50), (0, 50)]], 30)
        ok, why = check_level(cut)
        self.assertFalse(ok)
        self.assertIn("reach", why)

    def test_mutation_keeps_levels_valid(self):
        from ued.level import check_level, generate_level, next_level_id
        from ued.mutate import mutate_level
        lv = generate_level(777, 100, 4, 30, timeout_s=30)
        lv.level_id = next_level_id()
        rng = random.Random(0)
        for _ in range(15):
            child = mutate_level(lv, self.cfg, rng)
            self.assertTrue(check_level(child, max_density=self.cfg.UED_MAX_OBSTACLE_DENSITY)[0])
            self.assertEqual(child.parent_id, lv.level_id)
            self.assertEqual(child.generation, lv.generation + 1)
            lv = child

    def test_population(self):
        from ued.level import generate_level, next_level_id
        from ued.population import LevelPopulation
        cfg = resolve_config(overrides={"UED_MIN_TRIALS": 2, "UED_POP_SIZE": 12})
        pop = LevelPopulation(cfg, rng=random.Random(0))
        levels = []
        for s in range(14):
            lv = generate_level(1000 + s, 100, 1 + s % 3, 30, timeout_s=30)
            lv.level_id = next_level_id()
            levels.append(lv)
            pop.add(lv)
        self.assertEqual(len(pop), 12)        # evicted to capacity
        ep = 0
        for rep in range(4):
            for lv in levels:
                if pop.get(lv.level_id) is None:
                    continue
                ep += 1
                # level i solved with probability depending on i
                t = 300.0 if (lv.level_id + rep) % 3 else 6000.0
                pop.update(lv.level_id, ep, t, freeflow=100.0)
        st = pop.stats()
        self.assertEqual(st["pop_size"], 12)
        self.assertGreater(st["learnability_mean"], 0)
        drawn = {pop.sample().level_id for _ in range(200)}
        self.assertGreater(len(drawn), 3)
        sd = pop.state_dict()
        pop2 = LevelPopulation(cfg)
        pop2.load_state_dict(json.loads(json.dumps(sd)))
        self.assertEqual(len(pop2), 12)
        self.assertAlmostEqual(pop2.stats()["learnability_mean"], st["learnability_mean"])

    def test_sampler_modes_and_reserved_seeds(self):
        from ued.runner import TaskSampler
        for over in ({"TRAIN_MAP_SOURCE": "dataset"}, {"UED_METHOD": "dr"},
                     {"UED_METHOD": "plr", "UED_WARMUP_EPISODES": 0},
                     {"TRAIN_MAP_SOURCE": "mixed", "MIXED_DATASET_PROB": 0.5}):
            cfg = resolve_config(overrides=over)
            s = TaskSampler(cfg, seed=1)
            for ep in range(6):
                t = s.next_task(ep)
                if t.kind == "level":
                    self.assertFalse(any(lo <= t.level.seed < hi
                                         for lo, hi in cfg.UED_RESERVED_SEED_RANGES))
                    s.on_episode(t, {"evac_time_100": 500.0, "freeflow_steps": 80.0}, ep, 0.0,
                                 rewards=np.array([-1.0, -1.0]))
                else:
                    self.assertIn(t.map_num, cfg.DATASET_MAPS)

    def test_scores(self):
        from ued import score
        r = np.array([-1.0, -1.0, -1.0])
        v = np.array([-3.0, -2.0, -1.0])
        self.assertAlmostEqual(score.maxmc(v, -2.0), 0.0)
        self.assertGreaterEqual(score.pvl(r, v, 0.0, True, 0.99, 0.95), 0.0)


class EvalTest(unittest.TestCase):
    def test_stats(self):
        from evaluation import stats
        res = []
        for m in range(10):
            for s in range(3):
                res.append({"id": f"{m}/{s}", "map_key": f"m{m}", "group": "g", "condition": "a",
                            "evac_time_100": 100.0 + m, "success": 1.0})
                res.append({"id": f"{m}/{s}", "map_key": f"m{m}", "group": "g", "condition": "b",
                            "evac_time_100": 110.0 + m, "success": 1.0})
        sa = stats.summarize(res, "a", B=500)
        self.assertAlmostEqual(sa["mean"], 104.5)
        self.assertLessEqual(sa["ci_low"], 104.5)
        p = stats.paired(res, "a", "b", B=500)
        self.assertAlmostEqual(p["mean_diff"], -10.0)
        self.assertEqual(p["win_rate"], 1.0)
        self.assertEqual(stats.iqm([1, 2, 3, 4, 100, 5, 6, 7]), np.mean([3, 4, 5, 6]))

    def test_suites_intact_and_disjoint(self):
        from configs import cfg
        from evaluation.suites import load_suite
        seen = {}
        for name in ("val", "tune", "test", "test_scale"):
            path = os.path.join(ROOT, "data", "suites", f"{name}.json")
            if not os.path.exists(path):
                self.skipTest("suites not built")
            s = load_suite(name, cfg)
            keys = {sc["map_key"] for sc in s["scenarios"]}
            for other, ok in seen.items():
                if name.startswith("test") and other.startswith("test"):
                    continue
                self.assertFalse(keys & ok, (name, other))
            seen[name] = keys

    def test_leakage_detects_known_copy(self):
        from configs import cfg
        import sim_maps
        from evaluation.suites import LeakageIndex, raster
        idx = LeakageIndex([1235, 1500], cfg)
        g = sim_maps.map_geometry(151, cfg)
        d, who = idx.nearest(raster(g["obstacles"], g["exits"], g["width"], g["height"]))
        self.assertEqual(who, 1235)
        self.assertLess(d, cfg.LEAKAGE_PIXEL_THRESHOLD)

    def test_scenario_reproducible(self):
        from configs import cfg
        from evaluation.suites import load_suite, scenario_spec
        from learn.rollout import build_env
        path = os.path.join(ROOT, "data", "suites", "val.json")
        if not os.path.exists(path):
            self.skipTest("suites not built")
        sc = load_suite("val", cfg)["scenarios"][0]
        a = build_env(scenario_spec(sc), cfg)
        b = build_env(scenario_spec(sc), cfg)
        self.assertEqual([list(c.xy) for c in a.crowds], [list(c.xy) for c in b.crowds])
        self.assertEqual(list(a.robot.xy), list(b.robot.xy))
        n = build_env(scenario_spec(sc), cfg, robot="N")
        self.assertEqual([list(c.xy) for c in a.crowds], [list(c.xy) for c in n.crowds])

    def test_baselines_run(self):
        from configs import cfg
        from evaluation.baselines import make_controller
        from learn.rollout import EnvSpec, build_env, run_episode
        for cond in ("no_robot", "exit_robot", "shuttle_random", "shuttle_nearest"):
            ctrl, rv = make_controller(cond, cfg, seed=1)
            env = build_env(EnvSpec(crowd=10, map_num=1500, seed=1), cfg, robot=rv)
            r = run_episode(env, ctrl, cfg, max_steps=60)
            self.assertEqual(r["steps"], 60.0)


if __name__ == "__main__":
    unittest.main()
