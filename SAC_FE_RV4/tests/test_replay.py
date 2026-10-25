"""Frame-deduplicating replay: stacks, n-step returns, wraparound."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from learn.replay import FrameReplay  # noqa: E402


def frame(ep, t, hw):
    # A frame whose pixels encode (episode, decision).
    return np.full((hw, hw), (ep * 37 + t * 3) % 251, np.uint8)


class ReplayTest(unittest.TestCase):
    def _fill(self, rb, episodes, length, interleave=True, static=False):
        """Several episodes written interleaved, as concurrent workers do.
        Returns the ground truth per transition id."""
        truth = {}
        curs = {}
        stacks = {}
        for ep in range(episodes):
            e0, g0 = frame(ep, 0, 5), frame(ep, 0, 7)
            es = frame(ep, 0, 5) if static else None
            gs = np.full((7, 7), ep, np.uint8) if static else None
            curs[ep] = rb.begin_episode(e0, g0, es, gs)
            stacks[ep] = [0, 0, 0, 0]
        order = [(ep, t) for t in range(1, length + 1) for ep in range(episodes)] if interleave \
            else [(ep, t) for ep in range(episodes) for t in range(1, length + 1)]
        for ep, t in order:
            last = t == length
            s = np.array([ep, t - 1, 0, 0, 0], np.float32)
            s2 = np.array([ep, t, 0, 0, 0], np.float32)
            tid = rb.add(curs[ep], s, np.array([t, -t], np.float32), float(t),
                         frame(ep, t, 5), frame(ep, t, 7), frame(ep, t, 5) if static else None,
                         s2, terminal=last, last=last)
            truth[tid] = (ep, t, list(stacks[ep]))
            stacks[ep] = [t] + stacks[ep][:3]
        return truth

    def test_stacks_reconstruct(self):
        rb = FrameReplay(1000, 5, 7, 4, static_channel=False)
        truth = self._fill(rb, episodes=3, length=9)
        slots = np.arange(len(rb))
        b = rb.batch_from_slots(slots, 1, 0.99, True)
        for slot in slots:
            ep, t, st = truth[int(rb.t_id[slot])]
            want = np.stack([frame(ep, k, 5) for k in st]).astype(np.float32) / 255
            np.testing.assert_allclose(b["ego"][slot].numpy(), want)
            want2 = np.stack([frame(ep, k, 7) for k in [t] + st[:3]]).astype(np.float32) / 255
            np.testing.assert_allclose(b["glob2"][slot].numpy(), want2)
            self.assertEqual(float(b["state"][slot, 1]), t - 1)

    def test_nstep(self):
        rb = FrameReplay(1000, 5, 7, 4, static_channel=False)
        truth = self._fill(rb, episodes=2, length=6)
        g = 0.9
        slots = np.arange(len(rb))
        b = rb.batch_from_slots(slots, 3, g, True)
        for i, slot in enumerate(slots):
            ep, t, _ = truth[int(rb.t_id[slot])]
            m = min(3, 6 - t + 1)
            R = sum(g ** k * (t + k) for k in range(m))
            self.assertAlmostEqual(float(b["reward"][i]), R, places=4)
            self.assertAlmostEqual(float(b["discount"][i]), g ** m, places=5)
            self.assertEqual(float(b["done"][i]), 1.0 if t + m - 1 == 6 else 0.0)
            self.assertEqual(float(b["state2"][i, 1]), t + m - 1)

    def test_timeout_bootstrap(self):
        rb = FrameReplay(100, 5, 7, 4, static_channel=False)
        cur = rb.begin_episode(frame(0, 0, 5), frame(0, 0, 7))
        z = np.zeros(5, np.float32)
        rb.add(cur, z, np.zeros(2, np.float32), 1.0, frame(0, 1, 5), frame(0, 1, 7), None, z,
               terminal=False, last=True)
        self.assertEqual(float(rb.batch_from_slots(np.array([0]), 1, 0.99, True)["done"][0]), 0.0)
        self.assertEqual(float(rb.batch_from_slots(np.array([0]), 1, 0.99, False)["done"][0]), 1.0)

    def test_wraparound_never_returns_stale_frames(self):
        rb = FrameReplay(50, 5, 7, 4, static_channel=False)
        rb.fcap = 60  # force frame overwrites
        rb.ego_f = np.zeros((60, 5, 5), np.uint8)
        rb.glob_f = np.zeros((60, 7, 7), np.uint8)
        rb.f_id = np.full(60, -1, np.int64)
        truth = self._fill(rb, episodes=4, length=40)
        rng = np.random.default_rng(0)
        for _ in range(20):
            slots = rb.sample_slots(16, rng)
            b = rb.batch_from_slots(slots, 1, 0.99, True)
            for i, slot in enumerate(slots):
                ep, t, st = truth[int(rb.t_id[slot])]
                want = np.stack([frame(ep, k, 5) for k in st]).astype(np.float32) / 255
                np.testing.assert_allclose(b["ego"][i].numpy(), want)

    def test_static_channel_and_trainable(self):
        rb = FrameReplay(100, 5, 7, 4, static_channel=True)
        self._fill(rb, episodes=2, length=5, static=True)
        b = rb.batch_from_slots(np.arange(len(rb)), 1, 0.99, True)
        self.assertEqual(tuple(b["ego"].shape[1:]), (5, 5, 5))
        self.assertEqual(tuple(b["glob"].shape[1:]), (5, 7, 7))
        rb.trainable[:5] = False
        rng = np.random.default_rng(1)
        s = rb.sample_slots(8, rng)
        self.assertTrue(rb.trainable[s].all())


if __name__ == "__main__":
    unittest.main()
