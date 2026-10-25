"""Networks, the SAC update under each option, checkpoints."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from configs import resolve_config  # noqa: E402
from learn.networks import NETWORK_SIZES, build_networks, build_policy  # noqa: E402
from learn.sac import SACLearner, d4_batch, random_shift  # noqa: E402

RV3_DIR = os.path.join(os.path.dirname(ROOT), "SAC_FE_RV3")
RV3_CKPT = os.path.expanduser("~/SOTA_MODELS/3maps.pth")


def batch(cfg, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = cfg.obs_channels()
    E, G = cfg.EGO_MAP_SIZE, cfg.DOWNSAMPLE_MAP_SIZE
    r = lambda *s: torch.rand(*s, generator=g)
    return {"ego": r(B, C, E, E), "glob": r(B, C, G, G), "state": r(B, 5),
            "action": r(B, 2) * 4 - 2, "reward": -r(B), "discount": torch.full((B,), 0.99),
            "done": torch.zeros(B), "ego2": r(B, C, E, E), "glob2": r(B, C, G, G), "state2": r(B, 5)}


class NetworkTest(unittest.TestCase):
    def test_all_families_and_sizes(self):
        for fam in ("cnn", "impala"):
            for size in NETWORK_SIZES[fam]:
                for share in ("separate", "critic"):
                    cfg = resolve_config(overrides={"NET_ENCODER": fam, "NET_SIZE": size,
                                                    "ENCODER_SHARING": share})
                    actor, critic, target = build_networks(cfg)
                    b = batch(cfg, B=3)
                    a, lp = actor.sample(b["ego"], b["glob"], b["state"][:, :3])
                    self.assertEqual(tuple(a.shape), (3, 2))
                    self.assertTrue(((a >= -2) & (a <= 2)).all())
                    q = critic(b["ego"], b["glob"], b["state"][:, :3], a)
                    self.assertEqual(tuple(q.shape), (2, 3))
                    pol = build_policy(cfg)
                    pol.load_state_dict(actor.state_dict())

    def test_options(self):
        for over in ({"NET_SPATIAL": "pool", "NET_COORD_CHANNELS": True},
                     {"NET_NORM": "batch"}, {"NET_NORM": "layer"}, {"NET_NORM": "none"},
                     {"OBS_STATIC_DISTANCE_CHANNEL": True},
                     {"MAP_SCALE_TO_ACTOR": True, "MAP_SCALE_TO_CRITIC": True},
                     {"CRITIC_LAYERNORM": True, "CRITIC_DROPOUT": 0.1, "CRITIC_ENSEMBLE": 4},
                     {"EGO_USE": False}, {"FiLM_USE": False}):
            cfg = resolve_config(overrides=over)
            actor, critic, _ = build_networks(cfg)
            b = batch(cfg, B=2)
            sa = b["state"] if cfg.MAP_SCALE_TO_ACTOR else b["state"][:, :3]
            sc = b["state"] if cfg.MAP_SCALE_TO_CRITIC else b["state"][:, :3]
            a, _ = actor.sample(b["ego"], b["glob"], sa)
            self.assertEqual(critic(b["ego"], b["glob"], sc, a).shape[0], cfg.CRITIC_ENSEMBLE)

    def test_stable_logprob_matches_rv3_away_from_saturation(self):
        from learn.networks import squash_sample
        torch.manual_seed(0)
        mean = torch.zeros(1000, 2)
        log_std = torch.full((1000, 2), -1.0)
        torch.manual_seed(1)
        _, a = squash_sample(mean, log_std, stable=True)
        torch.manual_seed(1)
        _, b = squash_sample(mean, log_std, stable=False)
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-5)

    @unittest.skipUnless(os.path.isdir(RV3_DIR) and os.path.exists(RV3_CKPT), "needs SAC_FE_RV3 and 3maps.pth")
    def test_legacy_actor_equals_rv3_class(self):
        """Run RV3's own PolicyNetwork in a subprocess (its module writes log
        folders under HOME on import, so HOME is a temporary directory) and
        compare outputs on the same input."""
        with tempfile.TemporaryDirectory() as home:
            g = torch.Generator().manual_seed(0)
            ego = torch.rand(4, 4, 25, 25, generator=g)
            glob = torch.rand(4, 4, 50, 50, generator=g)
            st = torch.rand(4, 3, generator=g)
            inp = os.path.join(home, "inp.pt")
            torch.save({"ego": ego, "glob": glob, "st": st}, inp)
            code = (
                "import sys, torch, json; sys.path.insert(0, %r);"
                "import ADDS_AS_reinforcement as R;"
                "p = R.PolicyNetwork(); ck = torch.load(%r, map_location='cpu', weights_only=False);"
                "p.load_state_dict(ck['policy']); p.eval(); d = torch.load(%r);"
                "m, s = p(d['ego'], d['glob'], d['st']); torch.save({'m': m, 's': s}, %r)"
            ) % (RV3_DIR, RV3_CKPT, inp, os.path.join(home, "out.pt"))
            env = dict(os.environ, HOME=home,
                       PYTHONUSERBASE=os.path.expanduser("~/.local"))
            env.pop("RV4_OVERRIDES", None)
            env.pop("RV4_PRESET", None)
            subprocess.run([sys.executable, "-c", code], cwd=RV3_DIR, env=env, check=True,
                           capture_output=True)
            ref = torch.load(os.path.join(home, "out.pt"))
        from learn.policy_io import load_policy
        actor, pcfg, meta = load_policy(RV3_CKPT)
        with torch.no_grad():
            m, s = actor(ego, glob, st)
        torch.testing.assert_close(m, ref["m"])
        torch.testing.assert_close(s, ref["s"])


class SACTest(unittest.TestCase):
    def _run(self, over, n=3):
        cfg = resolve_config(overrides=over)
        L = SACLearner(cfg, "cpu")
        for i in range(n):
            out = L.update(batch(cfg, B=8, seed=i))
            for k, v in out.items():
                self.assertTrue(np.isfinite(v), (over, k, v))
        pol = build_policy(cfg)
        pol.load_state_dict(L.policy_state_dict())
        return L

    def test_variants(self):
        for over in ({}, {"NET_ENCODER": "legacy_rv3"}, {"NET_ENCODER": "impala", "NET_SIZE": "s"},
                     {"ENCODER_SHARING": "critic"}, {"CRITIC_ENSEMBLE": 5, "CRITIC_TARGET_SUBSET": 2},
                     {"AUG_RANDOM_SHIFT": True, "AUG_DRQ_K": 2, "AUG_DRQ_M": 2},
                     {"AUG_REPLAY_D4": True}, {"ALPHA_AUTO": True, "ALPHA_MIN": 0.01},
                     {"POLICY_DELAY": 2}, {"GRAD_CLIP_NORM": 1.0},
                     {"RESET_EVERY_UPDATES": 2, "RESET_SCOPE": "heads"},
                     {"RESET_EVERY_UPDATES": 2, "RESET_SCOPE": "all", "SHRINK_PERTURB_ALPHA": 0.5},
                     {"NET_ENCODER": "legacy_rv3", "RESET_EVERY_UPDATES": 2}):
            with self.subTest(over=over):
                self._run(over)

    def test_shared_encoder_not_trained_by_actor(self):
        cfg = resolve_config(overrides={"ENCODER_SHARING": "critic", "POLICY_DELAY": 1})
        L = SACLearner(cfg, "cpu")
        ids = {id(p) for p in L.actor_opt.param_groups[0]["params"]}
        self.assertFalse(any(id(p) in ids for p in L.critic.encoder.parameters()))

    def test_alpha_learns_when_auto(self):
        cfg = resolve_config(overrides={"ALPHA_AUTO": True, "ALPHA_LR": 0.1})
        L = SACLearner(cfg, "cpu")
        a0 = float(L.alpha)
        for i in range(5):
            L.update(batch(cfg, seed=i))
        self.assertNotAlmostEqual(a0, float(L.alpha))

    def test_state_dict_round_trip(self):
        cfg = resolve_config()
        L = self._run({})
        L2 = SACLearner(cfg, "cpu")
        L2.load_state_dict(L.state_dict())
        b = batch(cfg, seed=9)
        with torch.no_grad():
            torch.testing.assert_close(L.actor.deterministic(b["ego"], b["glob"], b["state"][:, :3]),
                                       L2.actor.deterministic(b["ego"], b["glob"], b["state"][:, :3]))

    def test_random_shift_and_d4_batch(self):
        x = torch.arange(2 * 1 * 5 * 5, dtype=torch.float32).view(2, 1, 5, 5)
        y = random_shift(x, 1)
        self.assertEqual(y.shape, x.shape)
        cfg = resolve_config()
        b = batch(cfg, B=4)
        k = torch.tensor([0, 1, 4, 7])
        out = d4_batch(b, k)
        torch.testing.assert_close(out["ego"][0], b["ego"][0])
        torch.testing.assert_close(out["ego"][1], torch.rot90(b["ego"][1], 1, dims=(-2, -1)))


if __name__ == "__main__":
    unittest.main()
