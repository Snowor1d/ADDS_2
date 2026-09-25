"""Stage 1 of docs/outdoor_madrl_redesign.md: three configuration files, one
resolved configuration validated at start-up, and one metric event that
reaches TXT, JSONL, TensorBoard and W&B with the same values on the same axes.
"""

import json
import os
import pickle
import tempfile
import unittest

import configs
from configs import ConfigError, collect, resolve_config


class ConfigSplitTest(unittest.TestCase):
    def test_every_setting_has_exactly_one_owner(self):
        values, owner = collect()
        self.assertEqual(set(values), set(owner))
        # config.py re-exports the union and defines nothing of its own.
        import config
        exported = {k for k in dir(config) if k[:1].isupper()}
        self.assertEqual(exported - set(values), set())

    def test_run_choices_and_training_choices_are_separate(self):
        cfg = resolve_config()
        self.assertEqual(cfg.owner("SIM_REAL_SITE"), "simulation_run")
        self.assertEqual(cfg.owner("FINAL_ZERO_SHOT_SITE"), "training/common")
        self.assertEqual(cfg.owner("TRAIN_MAP_SOURCE"), "training/common")
        self.assertEqual(cfg.owner("DANGER_SAFE_MARGIN_M"), "environment")
        self.assertEqual(cfg.owner("UED_DANGER_AREA_RANGE"), "training/ued")
        self.assertEqual(cfg.owner("UED_METHOD"), "training/ued")
        self.assertEqual(cfg.owner("DATASET_SITES"), "training/dataset")
        self.assertEqual(cfg.owner("DATASET_DANGER_PERCEPTIBILITY"),
                         "training/dataset")
        self.assertEqual(cfg.owner("WANDB_MODE"), "training/common")
        self.assertEqual(cfg.owner("EGO_MAP_SIZE"), "environment")

    def test_resolved_config_is_immutable_and_picklable(self):
        cfg = resolve_config()
        with self.assertRaises(AttributeError):
            cfg.MAX_ROBOTS = 7
        clone = pickle.loads(pickle.dumps(cfg))
        self.assertEqual(clone.fingerprint, cfg.fingerprint)
        self.assertEqual(clone.MAX_ROBOTS, cfg.MAX_ROBOTS)

    def test_import_has_no_side_effects(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            code = ("import os, sys; sys.path.insert(0, %r); os.chdir(%r); "
                    "import configs, config; configs.resolve_config(); "
                    "print(len(os.listdir('.')))" % (root, tmp))
            out = subprocess.run([sys.executable, "-c", code],
                                 capture_output=True, text=True,
                                 cwd=os.path.dirname(os.path.dirname(
                                     os.path.abspath(__file__))))
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), "0")

    def test_default_configuration_validates(self):
        resolve_config()

    def test_unsupported_crop_size_is_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            resolve_config({"SIM_REAL_SIZE": 150}, check_data=False)
        self.assertIn("SIM_REAL_SIZE=150", str(ctx.exception))

    def test_robot_count_outside_the_tensor_width_is_refused(self):
        with self.assertRaises(ConfigError):
            resolve_config({"UED_ROBOT_RANGE": (1, 4)}, check_data=False)
        with self.assertRaises(ConfigError):
            resolve_config({"FINAL_ZERO_SHOT_ROBOT_COUNTS": (0, 1)},
                           check_data=False)

    def test_final_site_cannot_be_an_auxiliary_site(self):
        with self.assertRaises(ConfigError) as ctx:
            resolve_config({"ZSG_REAL_SITES": ("myeongdong",)},
                           check_data=False)
        self.assertIn("final zero-shot site", str(ctx.exception))

    def test_spatially_overlapping_crop_is_refused(self):
        # Hongdae is a separate site; the final site's own 100 m crop shares
        # its centre and must be caught as overlap even under another name.
        from configs import _crop_bounds, _boxes_overlap
        a = _crop_bounds(37.5608977, 126.9863762, 400)
        b = _crop_bounds(37.5608977, 126.9863762, 100)
        c = _crop_bounds(37.5548801, 126.921889, 400)
        self.assertTrue(_boxes_overlap(a, b))
        self.assertFalse(_boxes_overlap(a, c))

    def test_validation_seeds_are_reserved_from_training(self):
        with self.assertRaises(ConfigError):
            resolve_config({"VALIDATION_SEED_BASE": 10}, check_data=False)
        from configs import validation_seeds
        taken = next(iter(validation_seeds(resolve_config(check_data=False)).values()))
        with self.assertRaises(ConfigError):
            resolve_config({"FINAL_ZERO_SHOT_HAZARD_SEEDS": (taken,)},
                           check_data=False)

    def test_street_hazard_on_osm_is_refused(self):
        with self.assertRaises(ConfigError):
            resolve_config({"FINAL_ZERO_SHOT_HAZARD_SHAPES": ("street",)},
                           check_data=False)

    def test_secret_in_configuration_is_refused(self):
        with self.assertRaises(ConfigError):
            resolve_config({"WANDB_ENTITY": "0123456789abcdef0123456789abcdef01234567"},
                           check_data=False)

    def test_full_information_actor_needs_its_own_run(self):
        with self.assertRaises(ConfigError):
            resolve_config({"ACTOR_GLOBAL_CROWD_TRUTH": True},
                           check_data=False)
        resolve_config({"ACTOR_GLOBAL_CROWD_TRUTH": True,
                        "EXPERIMENT_ID": "outdoor-madrl-v2-fullinfo",
                        "LOG_DIR": "Log_SAC_UED4_fullinfo"}, check_data=False)

    def test_osm_training_sites_are_checked(self):
        with self.assertRaises(ConfigError):     # the final site
            resolve_config({"DATASET_SITES": ("myeongdong",)})
        with self.assertRaises(ConfigError):     # no 400 m crop
            resolve_config({"DATASET_SITES": ("surry_hills",),
                            "DATASET_SIZES_M": (100, 200, 400),
                            "DATASET_DENSITY_BY_SIZE": {100: None, 200: None,
                                                        400: None}})
        with self.assertRaises(ConfigError):     # also an auxiliary site
            resolve_config({"DATASET_SITES": ("shibuya",),
                            "ZSG_REAL_SITES": ("shibuya",)})
        with self.assertRaises(ConfigError):     # unknown training mode
            resolve_config({"TRAIN_MAP_SOURCE": "osm"})
        with self.assertRaises(ConfigError):     # a street hazard on OSM crops
            resolve_config({"DATASET_DANGER_SHAPES": ("street",)})
        resolve_config({"TRAIN_MAP_SOURCE": "ued"})

    def test_one_element_tuples_need_their_comma(self):
        for bad in ({"ZSG_ROBOT_NUM": 3}, {"UED_DANGER_SHAPES": "circle"},
                    {"DATASET_SIZES_M": 100},
                    {"DATASET_DANGER_SHAPES": "rect"}):
            with self.assertRaises(ConfigError):
                resolve_config(bad, check_data=False)

    def test_override_must_name_an_existing_setting(self):
        with self.assertRaises(ConfigError):
            resolve_config({"MAX_ROBOT": 3}, check_data=False)

    def test_log_dir_is_not_shared_with_sac_ued3(self):
        self.assertNotEqual(resolve_config().LOG_DIR, "Log_SAC_UED3")


class _FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = "fake-run-1"
        self.logged = []
        self.metrics = []
        self.artifacts = []
        self.finished = False

    def define_metric(self, name, step_metric=None):
        self.metrics.append((name, step_metric))

    def log(self, data):
        self.logged.append(dict(data))

    def log_artifact(self, art, aliases=None):
        self.artifacts.append(art)

    def finish(self):
        self.finished = True


class _FakeArtifact:
    def __init__(self, name, type, metadata=None):
        self.name, self.type, self.metadata = name, type, metadata or {}
        self.files = {}

    def add_file(self, path, name=None):
        with open(path, encoding="utf-8") as fh:
            self.files[name or os.path.basename(path)] = fh.read()


class _FakeVideo:
    def __init__(self, path, fps=None, format=None, caption=None):
        self.path, self.fps, self.format, self.caption = path, fps, format, caption


class _FakeWandb:
    Artifact = _FakeArtifact
    Video = _FakeVideo

    def __init__(self, fail_on_log=False):
        self.runs = []
        self.fail_on_log = fail_on_log

    def init(self, **kwargs):
        run = _FakeRun(**kwargs)
        if self.fail_on_log:
            def boom(data):
                raise RuntimeError("network down")
            run.log = boom
        self.runs.append(run)
        return run


def _metadata(cfg):
    return {"config": cfg.to_dict(), "config_fingerprint": cfg.fingerprint,
            "schema_versions": cfg.schema_versions(),
            "experiment_id": cfg.EXPERIMENT_ID}


class LoggerTest(unittest.TestCase):
    def _read_tb(self, log_dir):
        from tensorboard.backend.event_processing.event_accumulator import \
            EventAccumulator
        acc = EventAccumulator(os.path.join(log_dir, "tensorboard_logs"))
        acc.Reload()
        return {tag: [(e.step, e.value) for e in acc.Scalars(tag)]
                for tag in acc.Tags()["scalars"]}

    def test_one_event_reaches_every_sink_with_the_same_values_and_axes(self):
        from learn.metrics_logger import MetricsLogger
        cfg = resolve_config({"WANDB_MODE": "online"}, check_data=False)
        fake = _FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            log = MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                                wandb_module=fake)
            episodes = [(1, 0, {"episode/total_reward": -3.5,
                                "episode/reward/person_time": -2.0}),
                        (2, 40, {"episode/total_reward": -1.25,
                                 "episode/reward/person_time": -0.5})]
            for ep, up, m in episodes:
                log.log_episode(ep, up, m)
            log.log_train(200, 2, {"train/loss_q": 0.75})
            log.close()

            # JSONL
            events = [json.loads(l) for l in open(os.path.join(tmp, "events.jsonl"))]
            eps = [e for e in events if e["kind"] == "episode"]
            self.assertEqual([e["global_episode"] for e in eps], [1, 2])
            self.assertEqual(eps[1]["metrics"]["episode/total_reward"], -1.25)
            train = [e for e in events if e["kind"] == "train"][0]
            self.assertEqual(train["global_update"], 200)

            # TXT
            lines = open(os.path.join(tmp, "total_reward.txt")).read().split()
            self.assertEqual([float(x) for x in lines], [-3.5, -1.25])
            comp = open(os.path.join(tmp, "reward_person_time.txt")).read().split()
            self.assertEqual([float(x) for x in comp], [-2.0, -0.5])

            # TensorBoard: episode metrics on the episode axis, learner
            # metrics on the update axis.
            tb = self._read_tb(tmp)
            self.assertEqual(tb["episode/total_reward"], [(1, -3.5), (2, -1.25)])
            self.assertEqual(tb["train/loss_q"], [(200, 0.75)])

            # W&B: same values, each carrying both axes, and the axes bound
            # to the right metric families.
            run = fake.runs[0]
            ep_logs = [d for d in run.logged if "episode/total_reward" in d]
            self.assertEqual([(d["global_episode"], d["episode/total_reward"])
                              for d in ep_logs], [(1, -3.5), (2, -1.25)])
            tr = [d for d in run.logged if "train/loss_q" in d][0]
            self.assertEqual((tr["global_update"], tr["train/loss_q"]), (200, 0.75))
            self.assertIn(("episode/*", "global_episode"), run.metrics)
            self.assertIn(("train/*", "global_update"), run.metrics)
            self.assertTrue(run.finished)
            self.assertNotIn("sync_tensorboard", run.kwargs)

    def test_config_reaches_wandb_as_values_and_as_files(self):
        from learn.metrics_logger import MetricsLogger
        cfg = resolve_config({"WANDB_MODE": "online"}, check_data=False)
        fake = _FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                          wandb_module=fake).close()
        run = fake.runs[0]
        conf = run.kwargs["config"]
        # Every setting is a top-level key with its resolved value.
        self.assertEqual(conf["LR"], cfg.LR)
        self.assertEqual(conf["DATASET_SITES"], list(cfg.DATASET_SITES))
        self.assertEqual(conf["setting_file"]["UED_METHOD"], "training/ued")
        self.assertEqual(conf["run/config_fingerprint"], cfg.fingerprint)
        # The files themselves, comments and all, plus the merged values.
        art = run.artifacts[0]
        self.assertEqual(art.type, "config")
        self.assertIn(cfg.fingerprint, art.name)
        for name in ("configs/environment.py", "configs/simulation_run.py",
                     "configs/training/common.py", "configs/training/ued.py",
                     "configs/training/dataset.py", "config_resolved.json"):
            self.assertIn(name, art.files)
        self.assertIn("TRAIN_MAP_SOURCE", art.files["configs/training/common.py"])
        resolved = json.loads(art.files["config_resolved.json"])
        self.assertEqual(resolved["config_fingerprint"], cfg.fingerprint)

    def test_episode_video_goes_to_wandb_on_the_episode_axis(self):
        from learn.metrics_logger import MetricsLogger
        cfg = resolve_config({"WANDB_MODE": "online"}, check_data=False)
        fake = _FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            log = MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                                wandb_module=fake)
            log.log_video(200, 50, "/tmp/x.mp4", 20, caption="gastown_100m")
            log.close()
            events = [json.loads(l) for l in open(os.path.join(tmp, "events.jsonl"))]
        v = [d for d in fake.runs[0].logged if "episode/video" in d][0]
        self.assertEqual(v["global_episode"], 200)
        self.assertEqual((v["episode/video"].path, v["episode/video"].fps),
                         ("/tmp/x.mp4", 20))
        self.assertTrue(any(e["kind"] == "video" and e["path"] == "/tmp/x.mp4"
                            for e in events))

    def test_video_speed_that_needs_sub_step_frames_is_refused(self):
        with self.assertRaises(ConfigError):
            resolve_config({"VIDEO_SPEEDUP": 5, "VIDEO_FPS": 20},
                           check_data=False)

    def test_run_name(self):
        from learn.metrics_logger import MetricsLogger, run_name
        cfg = resolve_config({"WANDB_MODE": "online", "WANDB_RUN_NAME": None,
                              "DATASET_SIZES_M": (100,),
                              "DATASET_DENSITY_BY_SIZE": {100: None}},
                             check_data=False)
        self.assertTrue(run_name(cfg, now=0).startswith(
            f"{cfg.EXPERIMENT_ID}-dataset-100m-"))
        self.assertTrue(run_name(cfg, 500).endswith("-resume500"))
        named = resolve_config({"WANDB_MODE": "online",
                                "WANDB_RUN_NAME": "A-baseline"},
                               check_data=False)
        self.assertEqual(run_name(named), "A-baseline")
        fake = _FakeWandb()
        with tempfile.TemporaryDirectory() as tmp:
            MetricsLogger(named, tmp, run_metadata=_metadata(named),
                          wandb_module=fake).close()
        self.assertEqual(fake.runs[0].kwargs["name"], "A-baseline")
        with self.assertRaises(ConfigError):
            resolve_config({"WANDB_RUN_NAME": "  "}, check_data=False)

    def test_wandb_failure_keeps_local_logs(self):
        from learn.metrics_logger import MetricsLogger
        cfg = resolve_config({"WANDB_MODE": "online"}, check_data=False)
        with tempfile.TemporaryDirectory() as tmp:
            log = MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                                wandb_module=_FakeWandb(fail_on_log=True))
            log.log_episode(1, 0, {"episode/total_reward": 1.0})
            log.log_episode(2, 0, {"episode/total_reward": 2.0})
            log.close()
            kinds = [json.loads(l)["kind"]
                     for l in open(os.path.join(tmp, "events.jsonl"))]
            self.assertIn("wandb_error", kinds)
            self.assertEqual(kinds.count("episode"), 2)
            self.assertEqual(open(os.path.join(tmp, "total_reward.txt")).read().split(),
                             ["1.0", "2.0"])

    def test_resume_only_when_counters_continue(self):
        from learn.metrics_logger import MetricsLogger
        cfg = resolve_config({"WANDB_MODE": "online"}, check_data=False)
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeWandb()
            log = MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                                wandb_module=fake, start_episode=0)
            log.log_episode(100, 10, {"episode/total_reward": 0.0})
            log.close()
            # Restart from an older checkpoint: counters roll back -> new run.
            fake2 = _FakeWandb()
            MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                          wandb_module=fake2, start_episode=50,
                          start_update=5).close()
            self.assertNotIn("id", fake2.runs[0].kwargs)
            self.assertEqual(fake2.runs[0].kwargs["group"], cfg.EXPERIMENT_ID)
            # Restart that carries on -> same run id.
            fake3 = _FakeWandb()
            with open(os.path.join(tmp, "wandb_run_state.json"), "w") as fh:
                json.dump({"run_id": "fake-run-1",
                           "experiment_id": cfg.EXPERIMENT_ID,
                           "config_fingerprint": cfg.fingerprint,
                           "global_episode": 100, "global_update": 10}, fh)
            MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg),
                          wandb_module=fake3, start_episode=100,
                          start_update=10).close()
            self.assertEqual(fake3.runs[0].kwargs.get("id"), "fake-run-1")
            self.assertNotIn("name", fake3.runs[0].kwargs)
            self.assertTrue(fake2.runs[0].kwargs["name"].endswith("-resume50"))

    def test_real_wandb_disabled_and_offline(self):
        from learn.metrics_logger import MetricsLogger
        for mode in ("disabled", "offline"):
            cfg = resolve_config({"WANDB_MODE": mode}, check_data=False)
            with tempfile.TemporaryDirectory() as tmp:
                old = os.environ.get("WANDB_SILENT")
                os.environ["WANDB_SILENT"] = "true"
                try:
                    log = MetricsLogger(cfg, tmp, run_metadata=_metadata(cfg))
                    log.log_episode(1, 0, {"episode/total_reward": 1.5})
                    log.log_train(10, 1, {"train/loss_q": 0.1})
                    log.close()
                finally:
                    if old is None:
                        os.environ.pop("WANDB_SILENT", None)
                    else:
                        os.environ["WANDB_SILENT"] = old
                kinds = [json.loads(l)["kind"]
                         for l in open(os.path.join(tmp, "events.jsonl"))]
                self.assertNotIn("wandb_error", kinds, mode)
                if mode == "offline":
                    self.assertTrue(os.path.isdir(os.path.join(tmp, "wandb")))


if __name__ == "__main__":
    unittest.main()
