"""Stages 2, 3 and 5 of docs/outdoor_madrl_redesign.md.

  2  team-Q objective, 7-number exploration, every robot driven by the policy
  3  rew-v2 terms and their time aggregation over a held action
  5  replay rebuilt from records: same arrays the worker acted on, windows
     that never cross an episode, overwritten history never sampled, and
     nothing read across a schema change
"""

import os
import random
import tempfile
import unittest

import numpy as np
import torch

from configs import resolve_config


def _cfg(**over):
    return resolve_config(over or None, check_data=False)


def _level(size=80, robots=2, seed=3, difficulty=3):
    from ued.level import generate_random_level
    lv = generate_random_level(random.Random(seed), difficulty=difficulty,
                               width=size, height=size)
    lv.robot_num = robots
    lv.augmentation = "identity"
    return lv


def _model(level, seed=0):
    import sim.model as M
    random.seed(seed)
    np.random.seed(seed)
    return M.FightingModel(int(level.crowd_size), level.width, level.height,
                           robot="Q", level=level)


def _fill(cfg, buffer, store, agent, episodes=((1, 40), (2, 40), (3, 40)),
          capture=None):
    """Roll short episodes into the buffer; optionally keep the observations
    the actor saw, by (episode uid, decision index)."""
    from learn.rollout import run_episode
    from sim.observation import build_static_layers

    for uid, (robots, steps) in enumerate(episodes):
        m = _model(_level(robots=robots, seed=uid + 5), seed=uid)
        st = build_static_layers(m, cfg)
        store.put(st)
        seen = {"d": -1}

        def act(obs, rec, uid=uid):
            seen["d"] += 1
            if capture is not None:
                capture[(uid, seen["d"])] = {k: v.copy() for k, v in obs.items()}
            return agent.act(obs, epsilon=0.3)

        def emit(tr, uid=uid, st=st):
            buffer.push(uid, tr.step, tr.record, st.key, tr.action,
                        tr.step_rewards, tr.terminal)

        run_episode(m, cfg, act, gamma=0.99, max_steps=steps, seed=uid,
                    static=st, emit=emit)
        buffer.end_episode(uid)


class TeamQTest(unittest.TestCase):
    def test_target_takes_the_minimum_after_the_team_mean(self):
        from learn.sac import SACAgent, discounted_return
        cfg = _cfg()
        agent = SACAgent(cfg)
        B, N = 2, cfg.MAX_ROBOTS
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        # Per robot: critic 1 is lower for robot 0, critic 2 for robot 1, so
        # min-then-mean and mean-then-min differ.
        q1 = torch.tensor([[0.0, 10.0, 99.0], [4.0, 99.0, 99.0]])
        q2 = torch.tensor([[10.0, 0.0, 99.0], [6.0, 99.0, 99.0]])
        agent.q1_target = lambda o, a, m: q1 * m
        agent.q2_target = lambda o, a, m: q2 * m
        logp = torch.tensor([[1.0, 3.0, 50.0], [2.0, 50.0, 50.0]])
        agent._per_robot_actions = lambda o, m: (None, logp)
        agent.alpha = torch.tensor(0.5)
        batch = {"next_obs": {}, "next_mask": mask,
                 "step_rewards": torch.tensor([[1.0, 1.0, 1.0, 1.0],
                                               [2.0, 0.0, 0.0, 0.0]]),
                 "hold": torch.tensor([4.0, 1.0]),
                 "terminal": torch.tensor([0.0, 1.0])}
        y = agent.targets(batch)
        g = agent.gamma ** (1 / cfg.ACTION_SCALE)
        # Team means: q1 5 and 4, q2 5 and 6; entropy means 2 and 2.
        v0 = min(5.0, 5.0) - 0.5 * 2.0
        r0 = 1 + g + g ** 2 + g ** 3
        self.assertAlmostEqual(float(y[0]), r0 + g ** 4 * v0, places=5)
        # Terminal: no bootstrap, one step of reward.
        self.assertAlmostEqual(float(y[1]), 2.0, places=5)
        dr = discounted_return(batch["step_rewards"], batch["hold"], g)
        self.assertAlmostEqual(float(dr[1]), 2.0, places=6)

    def test_critic_is_order_equivariant_and_ignores_padding(self):
        from learn.networks import CentralizedCritic
        from sim.observation import obs_shapes
        from sim.robot_action import ACTION_DIM
        cfg = _cfg()
        torch.manual_seed(0)
        q = CentralizedCritic(cfg, ACTION_DIM).eval()
        B, N = 2, cfg.MAX_ROBOTS
        sh = obs_shapes(cfg)
        obs = {k: torch.randn(B, N, *sh[k]) for k in ("ego", "mid", "glob", "state")}
        obs["priv"] = torch.randn(B, *sh["priv"])
        act = torch.randn(B, N, ACTION_DIM)
        mask = torch.zeros(B, N)
        mask[:, :2] = 1.0
        perm = torch.tensor([1, 0, 2])
        with torch.no_grad():
            out = q(obs, act, mask)
            pobs = {k: (v[:, perm] if k != "priv" else v) for k, v in obs.items()}
            out_p = q(pobs, act[:, perm], mask[:, perm])
            noisy = dict(obs)
            noisy["state"] = obs["state"].clone()
            noisy["state"][:, 2] = 1e3
            out_n = q(noisy, act, mask)
        self.assertLess((out[:, 0] - out_p[:, 1]).abs().max().item(), 1e-4)
        self.assertLess((out[:, :2] - out_n[:, :2]).abs().max().item(), 1e-4)
        self.assertEqual(float(out[:, 2].abs().max()), 0.0)

    def test_exploration_draws_all_seven_numbers(self):
        from learn.sac import exploration_action
        from sim.robot_action import ACTION_DIM
        rng = random.Random(0)
        acts = np.stack([exploration_action(rng) for _ in range(300)])
        from sim import robot_action as ra
        self.assertEqual(acts.shape[1], ACTION_DIM)
        modes = acts[:, ra.MODE].argmax(1)
        self.assertEqual(set(modes.tolist()), set(range(len(ra.ROBOT_MODES))))
        self.assertGreater(np.abs(acts[:, ra.MOVE]).max(), 0.5)
        if ra.SIGNAL is not None:
            self.assertGreater(np.abs(acts[:, ra.SIGNAL]).max(), 0.5)

    def test_select_action_explores_with_the_full_action(self):
        from learn.sac import SACAgent
        from sim.observation import obs_shapes
        cfg = _cfg()
        agent = SACAgent(cfg)
        sh = obs_shapes(cfg)
        obs = {k: np.zeros((2,) + sh[k], np.float32)
               for k in ("ego", "mid", "glob", "state")}
        out = agent.act(obs, epsilon=1.0, rng=random.Random(1))
        from sim.robot_action import ACTION_DIM
        self.assertEqual(out.shape, (2, ACTION_DIM))

    def test_loaded_policy_drives_every_robot(self):
        cfg = _cfg()
        m = _model(_level(robots=3))
        m.use_model(None, cfg=cfg, deterministic=False)
        m.step()
        modes = [rb.mode for rb in m.robots]
        # Every robot received an action (the default before acting is off
        # with a zero heading; a sampled policy changes at least the heading).
        self.assertEqual(len(modes), 3)
        self.assertTrue(all(tuple(rb.action[:2]) != (0, 0) for rb in m.robots))


class UpdateTest(unittest.TestCase):
    def test_update_runs_for_teams_of_one_two_and_three(self):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        cfg = _cfg(BATCH_SIZE=8)
        with tempfile.TemporaryDirectory() as tmp:
            store = StaticStore(tmp)
            buf = ReplayBuffer(cfg, 500, store)
            agent = SACAgent(cfg)
            _fill(cfg, buf, store, agent)
            teams = set(int(n) for n in buf.n_robots[:buf.size])
            self.assertEqual(teams, {1, 2, 3})
            before = [p.detach().clone() for p in agent.policy.parameters()]
            for _ in range(2):
                info = agent.update(buf.sample(8, np.random.default_rng(0)))
            changed = sum(1 for b, a in zip(before, agent.policy.parameters())
                          if not torch.equal(b, a.detach()))
            self.assertEqual(changed, len(before))
            self.assertTrue(np.isfinite(info["train/loss_q"]))


class RewardTimeTest(unittest.TestCase):
    def test_first_and_cut_short_decisions_are_recorded(self):
        from learn.rollout import run_episode
        cfg = _cfg()
        m = _model(_level(robots=1))
        got = []
        from sim.robot_action import encode
        guide = encode((0.0, 0.0), "guide")[None]
        run_episode(m, cfg, lambda o, r: guide,
                    gamma=0.99, max_steps=10, emit=got.append)
        acted = [t for t in got if t.action is not None]
        self.assertEqual([t.step for t in acted], [0, 1, 2])
        self.assertEqual([t.hold for t in acted], [4, 4, 2])
        self.assertEqual(sum(len(t.step_rewards) for t in acted), 10)
        self.assertIsNone(got[-1].action)
        self.assertFalse(any(t.terminal for t in acted))

    def test_task_termination_marks_the_last_transition_terminal(self):
        from learn.rollout import run_episode
        cfg = _cfg()
        m = _model(_level(robots=1))
        # The robots also ask should_finish, so decide by the step count.
        m.should_finish = lambda: m.step_count >= 6
        got = []
        from sim.robot_action import encode
        guide = encode((0.0, 0.0), "guide")[None]
        run_episode(m, cfg, lambda o, r: guide,
                    gamma=0.99, max_steps=100, emit=got.append)
        acted = [t for t in got if t.action is not None]
        self.assertTrue(acted[-1].terminal)
        self.assertEqual(acted[-1].hold, 2)

    def test_reentry_counts_only_after_clearing_and_never_on_departure(self):
        from sim.rewards import TaskReward
        cfg = _cfg(REWARD_MIN_REFERENCE_POPULATION=1)
        m = _model(_level(robots=1))
        zone = m.danger_zone
        inside = (zone.cx, zone.cy)
        far = next((x, y) for x in range(2, m.width - 2, 3)
                   for y in range(2, m.height - 2, 3)
                   if zone.signed_distance(x, y) > 10)
        people = [a for a in m.crowds if not a.dead][:3]
        for a in m.crowds:
            a.xy = list(far)
        tr = TaskReward(m, cfg)
        # Person 0 walks in from clear ground: one entry.
        people[0].xy = list(inside)
        c = tr.step(m)
        self.assertEqual(tr.last_raw["entries"], 1)
        # Jitter on the boundary without clearing the margin: no new entry.
        edge_out = zone.nearest_safe_point(inside[0], inside[1], margin=0.1)
        people[0].xy = list(edge_out)
        tr.step(m)
        people[0].xy = list(inside)
        tr.step(m)
        self.assertEqual(tr.last_raw["entries"], 0)
        # Person 1 leaves the crop: not an entry, no reward change from it.
        people[1].dead = True
        tr.step(m)
        self.assertEqual(tr.last_raw["entries"], 0)
        self.assertLess(c["reentry"], 0.0)

    def test_person_time_scale_is_comparable_across_map_sizes(self):
        from sim.rewards import TaskReward
        cfg = _cfg()
        per_size = {}
        for size in (100, 200):
            m = _model(_level(size=size, robots=1, seed=11))
            tr = TaskReward(m, cfg)
            c = tr.step(m)
            per_size[size] = c["person_time"] / cfg.REWARD_W_PERSON_TIME
        # Normalised by the people inside at the start, a full hazard costs
        # about AGENT_TIME_STEP per step whatever the map size.
        for v in per_size.values():
            self.assertLess(abs(v + cfg.AGENT_TIME_STEP), 0.2)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        self.cfg = _cfg(BATCH_SIZE=8)
        self.tmp = tempfile.TemporaryDirectory()
        self.store = StaticStore(self.tmp.name)
        self.buf = ReplayBuffer(self.cfg, 400, self.store)
        self.agent = SACAgent(self.cfg)
        self.seen = {}
        _fill(self.cfg, self.buf, self.store, self.agent, capture=self.seen)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rebuilt_observation_is_what_the_actor_saw(self):
        checked = 0
        for i in range(self.buf.size):
            if not self.buf.has_action[i]:
                continue
            key = (int(self.buf.episode[i]), int(self.buf.step[i]))
            joint = self.buf._joint([i])
            n = int(self.buf.n_robots[i])
            for k in ("ego", "mid", "glob", "state"):
                np.testing.assert_array_equal(joint[k][0, :n], self.seen[key][k],
                                              f"{k} at {key}")
            checked += 1
        self.assertGreater(checked, 20)

    def test_windows_never_cross_episodes(self):
        for i in range(self.buf.size):
            win = self.buf.window_indices(i)
            self.assertIsNotNone(win)
            real = [j for j in win if j is not None]
            self.assertEqual({int(self.buf.episode[j]) for j in real},
                             {int(self.buf.episode[i])})
            steps = [int(self.buf.step[j]) for j in real]
            self.assertEqual(steps, sorted(steps))
            if int(self.buf.step[i]) == 0:
                self.assertEqual(len(real), 1)

    def test_overwritten_history_is_not_sampled(self):
        from learn.replay import ReplayBuffer
        small = ReplayBuffer(self.cfg, 25, self.store)
        _fill(self.cfg, small, self.store, self.agent,
              episodes=((2, 60),))
        valid = [i for i in range(small.size) if small.valid_transition(i)]
        for i in valid:
            self.assertIsNotNone(small.window_indices(i))
            self.assertIsNotNone(small.window_indices(int(small.next[i])))
        # The oldest surviving records lost their predecessors.
        oldest = int(np.argmin(np.where(small.step[:small.size] >= 0,
                                        small.step[:small.size], 1 << 30)))
        if small.step[oldest] >= self.cfg.OBS_HISTORY_DECISIONS:
            self.assertFalse(small.valid_transition(oldest))

    def test_static_layers_reload_from_disk_after_eviction(self):
        from learn.replay import StaticStore
        cold = StaticStore(self.tmp.name, max_entries=1)
        key = self.buf._static_keys[0]
        st = cold.get(key)
        self.assertEqual(st.key, key)

    def test_schema_mismatch_is_refused(self):
        from learn.replay import SchemaMismatch
        path = os.path.join(self.tmp.name, "buf.npz")
        self.buf.save(path, self.cfg.schema_versions())
        from learn.replay import ReplayBuffer
        other = ReplayBuffer(self.cfg, 400, self.store)
        other.load(path, self.cfg.schema_versions())
        self.assertEqual(other.size, self.buf.size)
        wrong = dict(self.cfg.schema_versions(), reward_version="rew-v1")
        with self.assertRaises(SchemaMismatch):
            ReplayBuffer(self.cfg, 400, self.store).load(path, wrong)

    def test_checkpoint_schema_mismatch_is_refused(self):
        from learn.replay import SchemaMismatch
        from learn.sac import SACAgent
        path = os.path.join(self.tmp.name, "ck.pth")
        self.agent.save(path)
        SACAgent(self.cfg).load(path)
        other = _cfg(REWARD_VERSION="rew-v9")
        with self.assertRaises(SchemaMismatch):
            SACAgent(other).load(path)

    def test_storage_per_instant_matches_the_budget(self):
        per = self.buf.nbytes() / self.buf.capacity
        # Section 6: about 6.2 kB of dynamic data per instant for three robots.
        self.assertLess(per, 7_000)


if __name__ == "__main__":
    unittest.main()


class OsmTrainingMapsTest(unittest.TestCase):
    """TRAIN_MAP_SOURCE = "dataset": one (site, size) pair per episode,
    uniformly, with the DATASET_* task parameters."""

    def test_samples_every_site_and_size_with_a_fresh_task(self):
        from learn.training_maps import OsmTrainingMaps
        cfg = _cfg(DATASET_SITES=("gastown", "mitte"),
                   DATASET_SIZES_M=(100, 200),
                   DATASET_DENSITY_BY_SIZE={100: None, 200: None},
                   DATASET_DANGER_SHAPES=("circle", "rect"),
                   DATASET_DANGER_PERCEPTIBILITY=(0.3, 0.4))
        maps = OsmTrainingMaps(cfg)
        self.assertEqual(maps.pairs, [("gastown", 100), ("gastown", 200),
                                      ("mitte", 100), ("mitte", 200)])
        rng = random.Random(0)
        seen = set()
        zones = set()
        for _ in range(40):
            lv = maps.sample(rng)
            seen.add((lv.site_key, int(lv.width)))
            zones.add((round(lv.danger.cx, 3), round(lv.danger.cy, 3)))
            lo, hi = cfg.CROWD_DENSITY_RANGE
            self.assertGreaterEqual(lv.crowd_density, lo - 0.005)
            self.assertLessEqual(lv.crowd_density, hi + 0.005)
            self.assertIn(lv.danger.shape, cfg.DATASET_DANGER_SHAPES)
            # The dataset file's own task parameters, not the curriculum's.
            self.assertTrue(0.3 <= lv.perceptibility <= 0.4)
            self.assertTrue(1 <= lv.robot_num <= cfg.MAX_ROBOTS)
        self.assertEqual(seen, set(maps.pairs))
        self.assertGreater(len(zones), 30)      # a new hazard each episode

    def test_augmentation_can_be_switched_off(self):
        from learn.training_maps import OsmTrainingMaps
        base = dict(DATASET_SITES=("gastown",), DATASET_SIZES_M=(100,),
                    DATASET_DENSITY_BY_SIZE={100: None})
        rng = random.Random(3)
        on = OsmTrainingMaps(_cfg(**base, DATASET_AUGMENTATION=True))
        off = OsmTrainingMaps(_cfg(**base, DATASET_AUGMENTATION=False))
        self.assertGreater(len({on.sample(rng).augmentation
                                for _ in range(40)}), 1)
        self.assertEqual({off.sample(rng).augmentation for _ in range(20)},
                         {"identity"})

    def test_ued_augmentation_can_be_switched_off(self):
        import ued.level as L
        import config
        saved = config.UED_AUGMENTATION
        try:
            config.UED_AUGMENTATION = False
            self.assertEqual({L.sample_augmentation(random.Random(k))
                              for k in range(20)}, {"identity"})
        finally:
            config.UED_AUGMENTATION = saved

    def test_a_sampled_crop_runs(self):
        from learn.training_maps import OsmTrainingMaps
        import sim.model as M
        cfg = _cfg(DATASET_SITES=("gastown",), DATASET_SIZES_M=(100,),
                   DATASET_DENSITY_BY_SIZE={100: None})
        lv = OsmTrainingMaps(cfg).sample(random.Random(1))
        m = M.FightingModel(int(lv.crowd_size), lv.width, lv.height,
                            robot="Q", level=lv)
        for _ in range(3):
            m.step()
        self.assertEqual(len(m.robots), lv.robot_num)


class NetworkSizeTest(unittest.TestCase):
    """NET_ENCODER x NET_SIZE: every combination builds, runs, and matches
    the other family's parameter count at the same size."""

    def test_sizes_are_matched_across_families_and_grow(self):
        from learn.networks import (CentralizedCritic, PolicyNetwork,
                                    parameter_count)
        from sim.robot_action import ACTION_DIM
        counts = {}
        for fam in ("cnn", "impala"):
            for size in ("m", "l", "xl"):
                cfg = _cfg(NET_ENCODER=fam, NET_SIZE=size)
                counts[fam, size] = (
                    parameter_count(PolicyNetwork(cfg)),
                    parameter_count(CentralizedCritic(cfg, ACTION_DIM)))
        for size in ("m", "l", "xl"):
            a_cnn, c_cnn = counts["cnn", size]
            a_imp, c_imp = counts["impala", size]
            self.assertLess(abs(a_imp / a_cnn - 1.0), 0.10, size)
            self.assertLess(abs(c_imp / c_cnn - 1.0), 0.12, size)
        for fam in ("cnn", "impala"):
            actor = [counts[fam, s][0] for s in ("m", "l", "xl")]
            self.assertEqual(actor, sorted(actor))
            self.assertGreater(actor[2], 3 * actor[0])

    def test_impala_runs_and_keeps_most_parameters_in_convolutions(self):
        from learn.networks import PolicyNetwork, parameter_count
        from sim.observation import obs_shapes
        cfg = _cfg(NET_ENCODER="impala", NET_SIZE="m")
        pol = PolicyNetwork(cfg)
        conv = sum(parameter_count(getattr(pol, k).conv)
                   for k in ("ego", "mid", "glob"))
        fc = sum(parameter_count(getattr(pol, k).fc)
                 for k in ("ego", "mid", "glob"))
        self.assertGreater(conv, fc)
        sh = obs_shapes(cfg)
        x = {k: torch.randn(2, *sh[k]) for k in ("ego", "mid", "glob", "state")}
        a, logp = pol.sample_action(x["ego"], x["mid"], x["glob"], x["state"])
        from sim.robot_action import ACTION_DIM
        self.assertEqual(tuple(a.shape), (2, ACTION_DIM))

    def test_checkpoint_of_another_network_is_refused(self):
        from learn.replay import SchemaMismatch
        from learn.sac import SACAgent
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.pth")
            SACAgent(_cfg(NET_ENCODER="cnn", NET_SIZE="m")).save(path)
            with self.assertRaises(SchemaMismatch):
                SACAgent(_cfg(NET_ENCODER="impala", NET_SIZE="m")).load(path)

    def test_unknown_network_is_refused_at_start(self):
        from configs import ConfigError
        with self.assertRaises(ConfigError):
            _cfg(NET_SIZE="s")
        with self.assertRaises(ConfigError):
            _cfg(NET_ENCODER="resnet")


class DirectModeSwitchTest(unittest.TestCase):
    """USE_DIRECT: with it off (the default) "direct" and the signalled
    heading are gone from the action and from the observed state."""

    def test_default_has_no_direct_and_no_heading(self):
        from sim import robot_action as ra
        from sim.observation import own_state_layout, teammate_state_layout
        cfg = _cfg()
        self.assertFalse(cfg.USE_DIRECT)
        self.assertEqual(tuple(cfg.ROBOT_MODES), ("off", "guide"))
        self.assertEqual((ra.CONT_DIM, ra.ACTION_DIM), (2, 4))
        self.assertIsNone(ra.SIGNAL)
        self.assertNotIn("signal_x", own_state_layout(cfg))
        self.assertNotIn("mode_direct", teammate_state_layout(cfg))
        with self.assertRaises(ValueError):
            ra.encode((0, 0), "direct")
        move, mode, heading = ra.decode(ra.encode((1.0, -1.0), "guide"))
        self.assertEqual((move, mode, heading), ((1.0, -1.0), "guide", (0.0, 0.0)))

    def test_policy_emits_the_short_action(self):
        from learn.networks import PolicyNetwork
        from sim.observation import obs_shapes
        from sim import robot_action as ra
        cfg = _cfg()
        pol = PolicyNetwork(cfg)
        sh = obs_shapes(cfg)
        x = {k: torch.randn(3, *sh[k]) for k in ("ego", "mid", "glob", "state")}
        a = pol.deterministic_action(x["ego"], x["mid"], x["glob"], x["state"])
        self.assertEqual(tuple(a.shape), (3, ra.ACTION_DIM))
        self.assertTrue(torch.all(a[:, ra.MODE].sum(1) == 1))

    def test_override_cannot_change_the_action_layout(self):
        from configs import ConfigError
        with self.assertRaises(ConfigError):
            _cfg(USE_DIRECT=True)

    def test_layouts_with_direct(self):
        from sim.observation import own_state_layout, teammate_state_layout

        class C:
            ROBOT_MODES = ("off", "guide", "direct")
        own = own_state_layout(C)
        self.assertIn("mode_direct", own)
        self.assertIn("signal_x", own)
        self.assertEqual(len(teammate_state_layout(C)), 9)
