"""Stages 2, 3 and 5 of docs/outdoor_madrl_redesign.md.

  2  team-Q objective, 7-number exploration, every robot driven by the policy
  3  rew-v2 terms and their time aggregation over a held action
  5  replay rebuilt from records: same arrays the worker acted on, windows
     that never cross an episode, overwritten history never sampled, and
     nothing read across a schema change
"""

import os
import math
import random
import tempfile
import unittest

import numpy as np
import torch

from configs import resolve_config


def _cfg(**over):
    return resolve_config(over or None, check_data=False)


def _dmax(seconds):
    """The longest decision, for whichever action mode the config sets."""
    return {"ROBOT_DECISION_MAX_S_VELOCITY": seconds,
            "ROBOT_DECISION_MAX_S_WAYPOINT": seconds}


def _devents(on):
    return {"ROBOT_DECISION_ON_EVENTS_VELOCITY": on,
            "ROBOT_DECISION_ON_EVENTS_WAYPOINT": on}


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
                        tr.step_rewards, tr.terminal,
                     tr.robot_penalties)

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


class AlphaAutoTest(unittest.TestCase):
    """The entropy temperature is learned toward a target entropy."""

    def _agent_and_buffer(self, tmp, **over):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        cfg = _cfg(BATCH_SIZE=8, **over)
        store = StaticStore(tmp)
        buf = ReplayBuffer(cfg, 500, store)
        agent = SACAgent(cfg)
        _fill(cfg, buf, store, agent)
        return cfg, agent, buf

    def test_default_target_is_minus_the_continuous_dimensions(self):
        from learn.sac import SACAgent
        from sim import robot_action
        agent = SACAgent(_cfg())
        self.assertTrue(agent.alpha_auto)
        self.assertEqual(agent.target_entropy, -float(robot_action.CONT_DIM))
        self.assertAlmostEqual(float(agent.alpha), 0.2, places=6)

    def test_alpha_moves_toward_the_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A target far above any reachable entropy: alpha has to rise.
            _, up, buf = self._agent_and_buffer(tmp, ALPHA_TARGET_ENTROPY=50.0)
            for _ in range(3):
                info = up.update(buf.sample(8, np.random.default_rng(0)))
            self.assertGreater(float(up.alpha), 0.2)
            self.assertIn("train/loss_alpha", info)
        with tempfile.TemporaryDirectory() as tmp:
            # Far below: alpha has to fall.
            _, down, buf = self._agent_and_buffer(tmp,
                                                  ALPHA_TARGET_ENTROPY=-50.0)
            for _ in range(3):
                down.update(buf.sample(8, np.random.default_rng(0)))
            self.assertLess(float(down.alpha), 0.2)

    def test_fixed_alpha_stays_put(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, agent, buf = self._agent_and_buffer(tmp, ALPHA_AUTO=False)
            for _ in range(3):
                info = agent.update(buf.sample(8, np.random.default_rng(0)))
            self.assertAlmostEqual(float(agent.alpha), 0.2, places=6)
            self.assertNotIn("train/loss_alpha", info)

    def test_resume_keeps_the_learned_alpha(self):
        from learn.sac import SACAgent
        with tempfile.TemporaryDirectory() as tmp:
            cfg, agent, buf = self._agent_and_buffer(tmp,
                                                     ALPHA_TARGET_ENTROPY=50.0)
            for _ in range(3):
                agent.update(buf.sample(8, np.random.default_rng(0)))
            path = os.path.join(tmp, "ckpt.pt")
            agent.save(path)
            again = SACAgent(cfg)
            again.load(path)
            self.assertAlmostEqual(float(again.alpha), float(agent.alpha),
                                   places=6)

    def test_own_collision_is_charged_to_the_robot_that_hit(self):
        """rew-v3: a collision lowers only the colliding robot's target."""
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            agent, buf = ActorUpdateModeTest._run(self, tmp)
            if not agent.own_collision:
                self.skipTest("REWARD_VERSION is not rew-v3")
            b = buf.sample(4, np.random.default_rng(0))
            dev = agent.device
            batch = {
                "next_obs": {k: torch.as_tensor(v, device=dev)
                             for k, v in b["next_obs"].items()},
                "next_mask": torch.as_tensor(b["next_mask"], device=dev),
                "step_rewards": torch.as_tensor(b["step_rewards"], device=dev),
                "hold": torch.as_tensor(b["hold"], device=dev),
                "terminal": torch.ones(4, device=dev),   # no bootstrap noise
            }
            rc = np.zeros_like(b["robot_penalties"])
            batch["robot_penalties"] = torch.as_tensor(rc, device=dev)
            y0 = agent.robot_targets(batch)
            w = float(agent.cfg.REWARD_W_COLLISION)
            rc[:, 0, 0] = -w                            # robot 0, first step
            batch["robot_penalties"] = torch.as_tensor(rc, device=dev)
            y1 = agent.robot_targets(batch)
            d = (y1 - y0).cpu().numpy()
            np.testing.assert_allclose(d[:, 0], -w, atol=1e-5)
            np.testing.assert_allclose(d[:, 1:], 0.0, atol=1e-6)

    def test_alpha_never_falls_below_the_floor(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            agent, buf = ActorUpdateModeTest._run(self, tmp)
            with torch.no_grad():
                agent.log_alpha.fill_(-20.0)
            agent.update(buf.sample(8, np.random.default_rng(1)))
            self.assertGreaterEqual(float(agent.alpha),
                                    float(agent.cfg.ALPHA_MIN) * (1 - 1e-6))

    def test_invalid_settings_are_refused(self):
        from configs import ConfigError
        for bad in ({"ALPHA_START": 0.0}, {"ALPHA_LR": -1.0},
                    {"ALPHA_AUTO": "yes"}, {"ALPHA_TARGET_ENTROPY": "auto"}):
            with self.assertRaises(ConfigError):
                _cfg(**bad)


class ActorUpdateModeTest(unittest.TestCase):
    """ACTOR_UPDATE_ROBOTS x ACTOR_TEAMMATE_ACTIONS."""

    def _run(self, tmp, **over):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        cfg = _cfg(BATCH_SIZE=8, **over)
        store = StaticStore(tmp)
        buf = ReplayBuffer(cfg, 500, store)
        agent = SACAgent(cfg)
        _fill(cfg, buf, store, agent)
        return agent, buf

    def test_every_combination_trains(self):
        for robots in ("all", "one"):
            for mates in ("stored", "current"):
                with self.subTest(robots=robots, mates=mates), \
                        tempfile.TemporaryDirectory() as tmp:
                    agent, buf = self._run(tmp, ACTOR_UPDATE_ROBOTS=robots,
                                           ACTOR_TEAMMATE_ACTIONS=mates)
                    before = [p.detach().clone()
                              for p in agent.policy.parameters()]
                    info = agent.update(buf.sample(8,
                                                   np.random.default_rng(0)))
                    self.assertTrue(np.isfinite(info["train/loss_pi"]))
                    changed = sum(1 for b, a in zip(
                        before, agent.policy.parameters())
                        if not torch.equal(b, a.detach()))
                    self.assertEqual(changed, len(before))
                    # The critics are trainable again afterwards.
                    self.assertTrue(all(p.requires_grad
                                        for p in agent.q1.parameters()))

    def test_all_against_stored_teammates_scores_each_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, buf = self._run(tmp, ACTOR_UPDATE_ROBOTS="all",
                                   ACTOR_TEAMMATE_ACTIONS="stored")
            batch = buf.sample(8, np.random.default_rng(0))
            calls = {"n": 0}
            # rew-v3 scores each slot by its own value, rew-v2 by the team's.
            name = "_robot_q_min" if agent.own_collision else "_team_q_min"
            orig = getattr(agent, name)

            def counted(*a, **k):
                calls["n"] += 1
                return orig(*a, **k)
            setattr(agent, name, counted)
            agent.update(batch)
            N = int(np.asarray(batch["mask"]).shape[1])
            real_slots = int((np.asarray(batch["mask"]) > 0).any(0).sum())
            self.assertEqual(calls["n"], real_slots)
            self.assertLessEqual(real_slots, N)

    def test_invalid_settings_are_refused(self):
        from configs import ConfigError
        for bad in ({"ACTOR_UPDATE_ROBOTS": "every"},
                    {"ACTOR_TEAMMATE_ACTIONS": "latest"}):
            with self.assertRaises(ConfigError):
                _cfg(**bad)


class TeamIntentTest(unittest.TestCase):
    """TEAM_SHARE_INTENT: where each teammate is heading."""

    def test_layout_grows_only_when_on(self):
        from sim.observation import (own_state_layout, state_dim,
                                     teammate_state_layout)
        off, on = _cfg(), _cfg(TEAM_SHARE_INTENT=True)
        self.assertNotIn("intent_dx", own_state_layout(off))
        self.assertIn("intent_dx", own_state_layout(on))
        self.assertIn("intent_dy", teammate_state_layout(on))
        self.assertEqual(state_dim(on) - state_dim(off),
                         2 + 2 * (int(on.MAX_ROBOTS) - 1))
        self.assertNotEqual(off.observation_schema(),
                            on.observation_schema())

    def test_a_teammate_slot_shows_where_it_is_heading(self):
        from sim.observation import (CommChannel, build_static_layers,
                                     build_observations, record_decision,
                                     teammate_state_layout, own_state_layout,
                                     ObsRequest)
        cfg = _cfg(TEAM_SHARE_INTENT=True)
        m = _model(_level(robots=2, seed=5))
        a, b = m.robots[0], m.robots[1]
        a.action[0], a.action[1] = 0.0, 0.0
        b.action[0], b.action[1] = 1.0, 0.0      # due +x
        import config
        if config.ROBOT_ACTION_MODE == "waypoint":
            # Heading for its waypoint: where the action put it.
            b._waypoint = m.nearest_main_ground(
                float(b.xy[0]) + 0.5 * config.ROBOT_WAYPOINT_RANGE_M,
                float(b.xy[1]), b.body_radius)
            want_x = float(b._waypoint[0][0])
        else:
            from configs import decision_max_steps
            reach = (config.ROBOT_SPEED_MAX * config.ROBOT_TIME_STEP
                     * decision_max_steps(config))
            want_x = float(b.xy[0]) + reach
        rec = record_decision(m, cfg, 0, CommChannel(cfg, 2, seed=0))
        self.assertAlmostEqual(float(rec.intent[1, 0]), want_x, places=4)
        self.assertAlmostEqual(float(rec.intent[0, 0]), float(a.xy[0]),
                               places=4)
        st = build_static_layers(m, cfg)
        H = int(cfg.OBS_HISTORY_DECISIONS)
        window = [None] * (H - 1) + [rec]
        obs = build_observations([ObsRequest(static=st, window=window,
                                             receiver=0)], cfg)
        own = {n: i for i, n in enumerate(own_state_layout(cfg))}
        mate = {n: i for i, n in enumerate(teammate_state_layout(cfg))}
        scale = float(cfg.OBS_TEAM_DISTANCE_SCALE_M)
        o = len(own)
        s0 = obs["state"][0]
        want = (float(rec.intent[1, 0]) - float(a.xy[0])) / scale
        self.assertAlmostEqual(float(s0[o + mate["intent_dx"]]), want,
                               places=4)
        self.assertAlmostEqual(float(s0[own["intent_dx"]]), 0.0, places=4)

    def test_a_replay_saved_before_intents_still_loads(self):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmp:
            store = StaticStore(tmp)
            buf = ReplayBuffer(cfg, 200, store)
            _fill(cfg, buf, store, SACAgent(cfg), episodes=((2, 20),))
            path = os.path.join(tmp, "buf.npz")
            buf.save(path, cfg.schema_versions())
            with np.load(path, allow_pickle=False) as data:
                kept = {k: data[k] for k in data.files if k != "intent"}
            np.savez(path, **kept)
            again = ReplayBuffer(cfg, 200, store)
            again.load(path, cfg.schema_versions())
            n = again.size
            np.testing.assert_array_equal(again.intent[:n], again.pose[:n])


class RewardTimeTest(unittest.TestCase):
    def test_first_and_cut_short_decisions_are_recorded(self):
        from learn.rollout import run_episode
        # Fixed-length decisions: under waypoint mode a robot standing on
        # its own waypoint "arrives" every step, which is an event, not the
        # timing this test is about.
        cfg = _cfg(**_dmax(2.0), **_devents(False))
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
        cfg = _cfg(**_dmax(2.0), **_devents(False))
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
        # Pinned so the short episodes hold enough decisions to check.
        self.cfg = _cfg(BATCH_SIZE=8, **_dmax(2.0))
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
        other = _cfg(REWARD_VERSION=("rew-v2-person-time"
                                     if self.cfg.REWARD_VERSION != "rew-v2-person-time"
                                     else "rew-v4-robot-costs"))
        with self.assertRaises(SchemaMismatch):
            SACAgent(other).load(path)

    def test_fixed_reference_divides_by_the_constants(self):
        """REWARD_REFERENCE = "fixed": N_ref and D_ref are the configured
        constants whatever the episode, and the switch is part of the reward
        fingerprint."""
        from sim.rewards import TaskReward
        m = _model(_level(robots=1))
        fixed = _cfg(REWARD_REFERENCE="fixed", REWARD_FIXED_N_REF=123.0,
                     REWARD_FIXED_D_REF=45.0)
        task = TaskReward(m, fixed)
        self.assertEqual((task.n_ref, task.d_ref), (123.0, 45.0))
        self.assertNotEqual(fixed.schema_versions()["reward_fingerprint"],
                            _cfg().schema_versions()["reward_fingerprint"])

    def test_a_changed_reward_weight_refuses_the_old_checkpoint(self):
        """Weights are not versioned by hand; the reward fingerprint makes a
        resume under different ones refuse what was written before."""
        from learn.replay import SchemaMismatch
        from learn.sac import SACAgent
        path = os.path.join(self.tmp.name, "ck_w.pth")
        self.agent.save(path)
        other = _cfg(REWARD_W_COLLISION=float(self.cfg.REWARD_W_COLLISION) * 2)
        self.assertNotEqual(other.schema_versions()["reward_fingerprint"],
                            self.cfg.schema_versions()["reward_fingerprint"])
        with self.assertRaises(SchemaMismatch):
            SACAgent(other).load(path)

    def test_the_critic_is_told_the_episode_scales(self):
        """N_ref, D_ref and perceptibility reach the critic, through the
        record and the replay, and change its output; the actor never sees
        them."""
        import torch
        from learn.networks import CentralizedCritic
        from sim.robot_action import ACTION_DIM
        b = self.buf.sample(4, np.random.default_rng(0))
        sc = np.asarray(b["obs"]["priv_scalars"])
        self.assertEqual(sc.shape, (4, 3))
        self.assertTrue(np.all(sc[:, 1] > 0.0))           # D_ref was set
        critic = CentralizedCritic(self.cfg, ACTION_DIM)
        obs = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32)
               for k, v in b["obs"].items()}
        act = torch.as_tensor(b["action"])
        mask = torch.as_tensor(b["mask"])
        with torch.no_grad():
            q0 = critic(obs, act, mask)
            obs["priv_scalars"] = obs["priv_scalars"] + 1.0
            q1 = critic(obs, act, mask)
        self.assertFalse(torch.allclose(q0, q1))
        from sim.observation import own_state_layout
        self.assertNotIn("perceptibility", own_state_layout(self.cfg))

    def test_storage_per_instant_matches_the_budget(self):
        per = self.buf.nbytes() / self.buf.capacity
        # About 6.4 kB of dynamic data per instant for three robots with the
        # 25 x 25 ego crop, 10 kB with 41 x 41 (docs/replay_memory_budget.md).
        self.assertLess(per, 7_000 if self.cfg.EGO_MAP_SIZE <= 25 else 10_500)


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
        # Move (2), plus a speed share under waypoint mode, then the modes.
        want = (3, 5) if cfg.ROBOT_ACTION_MODE == "waypoint" else (2, 4)
        self.assertEqual((ra.CONT_DIM, ra.ACTION_DIM), want)
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


class DecisionTimingTest(unittest.TestCase):
    """A team decision is held up to ROBOT_DECISION_MAX_S_<MODE>, and ends early
    when a robot arrives at its waypoint or is blocked by a wall."""

    def setUp(self):
        import sim.agent as A
        self._A = A
        self._saved = A.ROBOT_ACTION_MODE

    def tearDown(self):
        self._A.ROBOT_ACTION_MODE = self._saved

    def _model_with_wall(self):
        from sim.danger import DangerZone
        from ued.level import Level
        import sim.model as M
        random.seed(0)
        np.random.seed(0)
        wall = [[0.0, 28.0], [60.0, 28.0], [60.0, 32.0], [0.0, 32.0]]
        lv = Level(obstacles=[wall], exits=[], crowd_size=3, width=60,
                   height=60)
        lv.danger = DangerZone("circle", 50.0, 50.0, radius=5.0)
        lv.robot_num = 1
        lv.augmentation = "identity"
        return M.FightingModel(3, 60, 60, robot="Q", level=lv)

    def _holds(self, m, cfg, move, steps):
        from learn.rollout import run_episode
        from sim.robot_action import encode
        got = []
        act = encode(move, "guide")[None]
        run_episode(m, cfg, lambda o, r: act, gamma=0.99, max_steps=steps,
                    emit=got.append)
        return [t.hold for t in got if t.action is not None]

    def test_the_longest_decision_is_the_configured_time(self):
        cfg = _cfg(**_dmax(4.0), **_devents(False))
        self.assertEqual(cfg.decision_max_steps(), 8)
        m = _model(_level(robots=1))
        self.assertEqual(self._holds(m, cfg, (0.0, 0.0), 20), [8, 8, 4])

    def test_decision_timing_follows_the_action_mode(self):
        """A waypoint setting must not leak into velocity training."""
        from configs import decision_max_steps, decision_on_events
        cfg = _cfg(ROBOT_DECISION_MAX_S_VELOCITY=2.0,
                   ROBOT_DECISION_ON_EVENTS_VELOCITY=False,
                   ROBOT_DECISION_MAX_S_WAYPOINT=30.0,
                   ROBOT_DECISION_ON_EVENTS_WAYPOINT=True)
        from types import SimpleNamespace
        base = dict(cfg._values)
        vel = SimpleNamespace(**{**base, "ROBOT_ACTION_MODE": "velocity"})
        wp = SimpleNamespace(**{**base, "ROBOT_ACTION_MODE": "waypoint"})
        self.assertEqual((decision_max_steps(vel), decision_on_events(vel)),
                         (4, False))
        self.assertEqual((decision_max_steps(wp), decision_on_events(wp)),
                         (60, True))
        want = (60, True) if cfg.ROBOT_ACTION_MODE == "waypoint" else (4, False)
        self.assertEqual((cfg.decision_max_steps(), cfg.decision_on_events()),
                         want)

    def test_the_longest_decision_is_whole_steps(self):
        from configs import ConfigError
        with self.assertRaises(ConfigError):
            _cfg(**_dmax(1.2))

    def test_arrival_ends_the_decision(self):
        from config import ROBOT_WAYPOINT_RANGE_M
        move = (2.0 * 2.0 / float(ROBOT_WAYPOINT_RANGE_M), 0.0)  # 2 m right
        for on_events, want in ((True, 3), (False, 8)):
            # Resolved before the simulator is switched, which the config
            # check would otherwise refuse.
            cfg = _cfg(**_dmax(4.0), **_devents(on_events))
            self._A.ROBOT_ACTION_MODE = "waypoint"
            m = self._model_with_wall()
            m.robots[0].xy = [10.0, 10.0]
            # One step to initialise the robot, two to walk 2 m.
            self.assertEqual(self._holds(m, cfg, move, 8)[0], want)
            self._A.ROBOT_ACTION_MODE = self._saved

    def test_a_wall_ends_the_decision(self):
        m = self._model_with_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 27.0]
        cfg = _cfg(**_dmax(4.0), **_devents(True))
        holds = self._holds(m, cfg, (0.0, 1.0), 8)
        self.assertLess(holds[0], 8)
        # Velocity: driving into the wall is "blocked". Waypoint: a target in
        # the wall is moved to the nearest ground, reached, and "arrived".
        import config
        self.assertEqual(rb.decision_event,
                         "arrived" if config.ROBOT_ACTION_MODE == "waypoint"
                         else "blocked")

    def test_a_replay_saved_with_shorter_decisions_still_loads(self):
        from learn.replay import ReplayBuffer, StaticStore
        from learn.sac import SACAgent
        cfg = _cfg(**_dmax(2.0))
        longer = _cfg(**_dmax(4.0))
        with tempfile.TemporaryDirectory() as tmp:
            store = StaticStore(tmp)
            buf = ReplayBuffer(cfg, 200, store)
            _fill(cfg, buf, store, SACAgent(cfg), episodes=((2, 20),))
            path = os.path.join(tmp, "buf.npz")
            buf.save(path, cfg.schema_versions())
            again = ReplayBuffer(longer, 200, store)
            again.load(path, longer.schema_versions())
            n = again.size
            self.assertEqual(again.step_rewards.shape[1], 8)
            np.testing.assert_array_equal(again.step_rewards[:n, :4],
                                          buf.step_rewards[:n])
            self.assertFalse(again.step_rewards[:n, 4:].any())


class WaypointActionTest(unittest.TestCase):
    """ROBOT_ACTION_MODE = "waypoint": the move is a target point, walked
    to along the navmesh."""

    def setUp(self):
        import sim.agent as A
        self._A = A
        self._saved = A.ROBOT_ACTION_MODE

    def tearDown(self):
        self._A.ROBOT_ACTION_MODE = self._saved

    def _model_with_slotted_wall(self):
        """A wall across the map with one 2 m gap far to the side."""
        from sim.danger import DangerZone
        from ued.level import Level
        import sim.model as M
        random.seed(0)
        np.random.seed(0)
        left = [[0.0, 28.0], [40.0, 28.0], [40.0, 32.0], [0.0, 32.0]]
        right = [[42.0, 28.0], [60.0, 28.0], [60.0, 32.0], [42.0, 32.0]]
        lv = Level(obstacles=[left, right], exits=[], crowd_size=3, width=60,
                   height=60)
        lv.danger = DangerZone("circle", 50.0, 50.0, radius=5.0)
        lv.robot_num = 1
        lv.augmentation = "identity"
        return M.FightingModel(3, 60, 60, robot="Q", level=lv)

    def _drive(self, m, rb, steps):
        from config import ROBOT_TIME_STEP
        for _ in range(steps):
            vx, vy = rb._waypoint_velocity(ROBOT_TIME_STEP)
            rb.xy = rb._move_robot_with_walls(vx, vy, ROBOT_TIME_STEP)

    def test_the_robot_walks_round_a_wall_through_a_narrow_gap(self):
        from config import ROBOT_WAYPOINT_RANGE_M
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 20.0]
        # (10, 40): straight ahead, behind the wall; the gap is 30 m aside.
        rb.receive_action([0.0, 2.0 * 20.0 / float(ROBOT_WAYPOINT_RANGE_M)])
        self._drive(m, rb, 150)
        self.assertLess(math.dist(rb.xy, (10.0, 40.0)), 0.5)

    def test_a_held_velocity_toward_the_same_point_stops_at_the_wall(self):
        from config import ROBOT_SPEED_MAX, ROBOT_TIME_STEP
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 20.0]
        for _ in range(150):
            rb.xy = rb._move_robot_with_walls(0.0, ROBOT_SPEED_MAX,
                                              ROBOT_TIME_STEP)
        self.assertLess(rb.xy[1], 28.0)

    def test_it_stops_on_arrival(self):
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 10.0]
        rb.receive_action([0.2, 0.0])     # a tenth of the range to the right
        self._drive(m, rb, 20)
        from config import ROBOT_TIME_STEP
        self.assertEqual(rb._waypoint_velocity(ROBOT_TIME_STEP), (0.0, 0.0))

    def test_heading_target_is_the_waypoint(self):
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 10.0]
        rb.receive_action([0.2, 0.0])
        self._drive(m, rb, 1)
        from config import ROBOT_WAYPOINT_RANGE_M
        wx, wy = rb.heading_target()
        self.assertAlmostEqual(wx, 10.0 + 0.1 * float(ROBOT_WAYPOINT_RANGE_M),
                               places=4)
        self.assertAlmostEqual(wy, 10.0, places=4)

    def test_the_action_speed_scales_the_walk(self):
        from config import ROBOT_SPEED_MAX, ROBOT_TIME_STEP
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 10.0]
        rb.receive_action([1.0, 0.0], speed_fraction=0.5)   # 10 m right
        vx, vy = rb._waypoint_velocity(ROBOT_TIME_STEP)
        self.assertAlmostEqual(math.hypot(vx, vy), 0.5 * ROBOT_SPEED_MAX)
        rb.receive_action([1.0, 0.0])
        vx, vy = rb._waypoint_velocity(ROBOT_TIME_STEP)
        self.assertAlmostEqual(math.hypot(vx, vy), float(ROBOT_SPEED_MAX))

    def test_speed_fraction_reads_the_action(self):
        from sim import robot_action as ra
        v = np.zeros(ra.ACTION_DIM + 1, np.float32)
        saved = (ra.SPEED, ra.ACTION_DIM)
        try:
            ra.SPEED, ra.ACTION_DIM = 2, len(v)
            for raw, want in ((-2.0, 0.0), (0.0, 0.5), (2.0, 1.0), (9.0, 1.0)):
                v[2] = raw
                self.assertAlmostEqual(ra.speed_fraction(v), want)
        finally:
            ra.SPEED, ra.ACTION_DIM = saved
        if ra.SPEED is None:
            self.assertEqual(ra.speed_fraction(v), 1.0)

    def test_the_planned_path_goes_through_the_gap(self):
        from config import ROBOT_WAYPOINT_RANGE_M
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 20.0]
        self.assertEqual(rb.planned_path(), [])
        rb.receive_action([0.0, 2.0 * 20.0 / float(ROBOT_WAYPOINT_RANGE_M)])
        self._drive(m, rb, 1)
        pts = rb.planned_path()
        self.assertEqual(pts[0], (float(rb.xy[0]), float(rb.xy[1])))
        self.assertLess(math.dist(pts[-1], (10.0, 40.0)), 1e-6)
        # Some point on the way crosses the wall line inside the gap.
        self.assertTrue(any(39.5 < x < 42.5 and 27.0 < y < 33.0
                            for x, y in pts[1:-1]))
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            self.assertTrue(m.is_free_segment(x0, y0, x1, y1, padding=0.0))

    def test_the_renderer_draws_the_plan(self):
        from config import ROBOT_WAYPOINT_RANGE_M
        from viz.continuous_renderer import ContinuousRenderer
        self._A.ROBOT_ACTION_MODE = "waypoint"
        m = self._model_with_slotted_wall()
        rb = m.robots[0]
        rb.xy = [10.0, 20.0]
        rb.receive_action([0.0, 2.0 * 20.0 / float(ROBOT_WAYPOINT_RANGE_M)])
        self._drive(m, rb, 1)
        r = ContinuousRenderer(world_size=(60.0, 60.0), robot_style="circle")
        r.draw(m)
        dashed = [ln for ln in r.ax.lines if ln.get_linestyle() == "--"]
        self.assertEqual(len(dashed), 1)
        self.assertEqual(tuple(dashed[0].get_xydata()[-1]), (10.0, 40.0))

    def test_a_target_inside_a_building_moves_to_the_network(self):
        m = self._model_with_slotted_wall()
        (x, y), tri = m.nearest_main_ground(20.0, 30.0, 0.5)
        self.assertTrue(m.is_free_point(x, y, padding=0.5)
                        or tri in m.main_walkable_component())
        self.assertFalse(28.0 < y < 32.0 and x < 40.0)

    def test_a_target_in_a_building_is_charged_once_to_its_robot(self):
        """rew-v5: the distance a requested waypoint was moved to reachable
        ground, over ROBOT_WAYPOINT_RANGE_M, once per decision, to that robot
        alone; a target on open ground costs nothing."""
        import math
        import config
        from sim.rewards import TaskReward
        if config.ROBOT_ACTION_MODE != "waypoint":
            self.skipTest("ROBOT_ACTION_MODE is not waypoint")
        cfg = _cfg(REWARD_VERSION="rew-v5-projection", REWARD_W_PROJECTION=0.2)
        m = _model(_level(robots=2, seed=4))
        task = TaskReward(m, cfg)
        a, b = m.robots[0], m.robots[1]
        a.receive_action([0.0, 0.0])            # its own spot: reachable
        b.receive_action([2.0, 2.0])            # 20 m diagonally: may be moved
        total_a = total_b = total_comp = 0.0
        for _ in range(6):                      # the robots move from step 2
            m.step()
            comps = task.step(m)
            total_a += task.last_robot_penalties[0]
            total_b += task.last_robot_penalties[1]
            total_comp += comps["projection"]
        self.assertAlmostEqual(total_a, 0.0, places=6)
        self.assertLessEqual(total_b, 0.0)
        self.assertGreaterEqual(total_b, -0.2 - 1e-9)   # one charge, capped
        self.assertAlmostEqual(total_comp, total_b, places=6)

    def test_the_signal_holds_for_the_whole_decision(self):
        """Whatever mode a decision sets is the mode every step of it runs
        with, however long the decision lasts (waypoint decisions end on
        arrival, a block or the time limit)."""
        from learn.rollout import run_episode
        from sim import robot_action as ra
        cfg = _cfg()
        m = _model(_level(robots=2, seed=4))
        rng = np.random.default_rng(0)
        said = []

        def act(o, r):
            modes = ["guide" if rng.random() < 0.5 else "off"
                     for _ in m.robots]
            said.append(modes)
            return np.stack([ra.encode((1.5, -0.5), md) for md in modes])
        seen = []
        run_episode(m, cfg, act, gamma=0.99, max_steps=120, needs_obs=False,
                    on_step=lambda model, k: seen.append(
                        (len(said) - 1, [rb.mode for rb in model.robots])))
        self.assertGreater(len(said), 1)
        for d, modes in seen:
            self.assertEqual(modes, said[d][:len(modes)])

    def test_the_action_schema_follows_the_mode(self):
        from configs import ConfigError
        import config
        other = ("velocity" if config.ROBOT_ACTION_MODE == "waypoint"
                 else "waypoint")
        other_schema = ("act-v2-move2-mode2-waypoint-speed"
                        if other == "waypoint" else "act-v2-move2-mode2")
        with self.assertRaises(ConfigError):
            _cfg(ACTION_SCHEMA_VERSION=other_schema)
        with self.assertRaises(ConfigError):
            _cfg(ROBOT_ACTION_MODE=other)        # the simulator disagrees


class RobotStuckTest(unittest.TestCase):
    """Robots used to spawn with their body inside a wall (13% of robots on
    OSM crops, mostly in inner building corners) and could then never move."""

    def _model_with_block(self):
        from sim.danger import DangerZone
        from ued.level import Level
        import sim.model as M
        random.seed(0)
        np.random.seed(0)
        block = [[40.0, 40.0], [60.0, 40.0], [60.0, 60.0], [40.0, 60.0]]
        lv = Level(obstacles=[block], exits=[], crowd_size=5, width=100,
                   height=100)
        lv.danger = DangerZone("circle", 20.0, 80.0, radius=8.0)
        lv.robot_num = 3
        lv.augmentation = "identity"
        return M.FightingModel(5, 100, 100, robot="Q", level=lv)

    def test_robots_spawn_clear_of_walls_on_the_main_network(self):
        from learn.training_maps import OsmTrainingMaps
        import sim.model as M
        cfg = _cfg(DATASET_SITES=("gastown", "mitte", "gracia"),
                   DATASET_SIZES_M=(100,), DATASET_DENSITY_BY_SIZE={100: None})
        maps = OsmTrainingMaps(cfg)
        rng = random.Random(4)
        for k in range(6):
            lv = maps.sample(rng)
            lv.robot_num = 3
            random.seed(k)
            m = M.FightingModel(int(lv.crowd_size), lv.width, lv.height,
                                robot="Q", level=lv)
            main = m.main_walkable_component()
            for rb in m.robots:
                self.assertTrue(m.is_free_point(rb.xy[0], rb.xy[1],
                                                padding=rb.body_radius))
                self.assertIn(m.find_mesh(rb.xy), main)

    def test_the_crowd_spawns_on_the_main_network(self):
        """Sealed courtyards on OSM crops are walkable but unreachable.
        Pedestrians placed there pressed into their corners for the rest of
        the episode (la_latina and khao_san: 10 of 14 wedged pedestrians)."""
        from learn.training_maps import OsmTrainingMaps
        import sim.model as M
        cfg = _cfg(DATASET_SITES=("la_latina",), DATASET_SIZES_M=(100,),
                   DATASET_DENSITY_BY_SIZE={100: None})
        maps = OsmTrainingMaps(cfg)
        rng = random.Random(0)
        lv = maps.sample(rng)
        random.seed(0)
        np.random.seed(0)
        m = M.FightingModel(int(lv.crowd_size), lv.width, lv.height,
                            robot="Q", level=lv)
        main = m.main_walkable_component()
        self.assertLess(len(main), len(m.pure_mesh),
                        "this crop is expected to contain a sealed pocket")
        for a in m.crowds:
            self.assertIn(m.find_mesh(a.xy), main)
        for mesh in m._edge_meshes():
            self.assertIn(mesh, main)

    def test_a_pedestrian_in_a_sealed_pocket_heads_for_the_network(self):
        from learn.training_maps import OsmTrainingMaps
        import sim.model as M
        cfg = _cfg(DATASET_SITES=("la_latina",), DATASET_SIZES_M=(100,),
                   DATASET_DENSITY_BY_SIZE={100: None})
        lv = OsmTrainingMaps(cfg).sample(random.Random(0))
        random.seed(0)
        np.random.seed(0)
        m = M.FightingModel(int(lv.crowd_size), lv.width, lv.height,
                            robot="Q", level=lv)
        a = m.crowds[0]
        a.xy = [34.75, 84.25]           # inner corner of a sealed courtyard
        here = m.find_mesh(a.xy)
        self.assertNotIn(here, m.main_walkable_component())
        goal = a._explore_randomly(here)
        self.assertGreater(math.dist(a.xy, goal), 1.0)

    def test_a_robot_inside_a_wall_backs_out_but_never_goes_in(self):
        m = self._model_with_block()
        rb = m.robots[0]
        rb.xy = [39.8, 50.0]            # body 0.5 m, 0.2 m from the block face
        # Pushing further in is refused.
        end = rb._move_robot_with_walls(2.0, 0.0, 0.5)
        self.assertLessEqual(end[0], 39.8 + 1e-9)
        # Moving away is allowed until the body is clear.
        rb.xy = [39.8, 50.0]
        end = rb._move_robot_with_walls(-2.0, 0.0, 0.5)
        self.assertLess(end[0], 39.8 - 0.5)
        self.assertTrue(m.is_free_point(end[0], end[1], padding=rb.body_radius))
