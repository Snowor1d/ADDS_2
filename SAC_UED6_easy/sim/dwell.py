"""Activity dwell times (docs/crowd_activity_design.md).

How long a pedestrian stays at a destination, in seconds, drawn from a
distribution truncated to CROWD_DWELL_RANGE_S. Truncated means conditioned on
the range: a draw outside it is not clipped onto the bound, which would pile
probability mass at 30 s and 600 s.

Every stay is drawn from its own random stream, keyed by the episode's dwell
seed, the pedestrian and which stay of theirs it is. A pedestrian's second
stay is a new draw, but the same pedestrian's second stay gets the same draw
under every condition run from one seed (robots off, shuttle, policy), so a
paired comparison is not also comparing luck in the dwell draws. The global
`random` would not do: robot behaviour changes how many draws come before.

Constants only from config at call time; importing this module has no side
effects and imports nothing from the simulator.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist
from typing import List

DISTRIBUTIONS = ("uniform", "exponential", "lognormal")
# Below this much probability inside the range the truncated distribution is
# numerically meaningless; the configuration is refused.
MIN_MASS = 1e-9


def stream(dwell_seed: int, person_id: int, k: int) -> random.Random:
    """The random stream of `person_id`'s k-th stay (k = 0 is the stay an
    initial dweller is already in when the episode starts)."""
    # A str seed is hashed with SHA-512, the same in every process and
    # independent of PYTHONHASHSEED.
    return random.Random(f"dwell:{int(dwell_seed)}:{int(person_id)}:{int(k)}")


def _params(cfg=None):
    if cfg is None:
        import config as cfg
    lo, hi = (float(v) for v in cfg.CROWD_DWELL_RANGE_S)
    return (lo, hi, str(cfg.CROWD_DWELL_DISTRIBUTION),
            cfg.CROWD_DWELL_EXP_SCALE_S, cfg.CROWD_DWELL_LOGNORMAL_MEDIAN_S,
            cfg.CROWD_DWELL_LOGNORMAL_SIGMA)


def sample_s(rng: random.Random, cfg=None) -> float:
    """One stay in seconds, inside CROWD_DWELL_RANGE_S, by inverse CDF."""
    lo, hi, dist, scale, median, sigma = _params(cfg)
    u = rng.random()
    if dist == "uniform" or hi == lo:
        return lo + u * (hi - lo)
    if dist == "exponential":
        # F(x) = 1 - exp(-x/s); the truncated inverse, written with expm1 and
        # log1p so a narrow range far in the tail keeps its precision.
        s = float(scale)
        a = -math.expm1(-lo / s)
        b = -math.expm1(-hi / s)
        return min(hi, max(lo, -s * math.log1p(-(a + u * (b - a)))))
    if dist == "lognormal":
        mu, sg = math.log(float(median)), float(sigma)
        nd = NormalDist()
        za, zb = (math.log(lo) - mu) / sg, (math.log(hi) - mu) / sg
        if za > 0:
            # Upper tail: work with survival probabilities, which keep their
            # precision where the CDF has rounded to 1.
            sa, sb = nd.cdf(-za), nd.cdf(-zb)
            z = -nd.inv_cdf(sa - u * (sa - sb))
        else:
            a, b = nd.cdf(za), nd.cdf(zb)
            z = nd.inv_cdf(a + u * (b - a))
        return min(hi, max(lo, math.exp(mu + sg * z)))
    raise ValueError(f"unknown CROWD_DWELL_DISTRIBUTION {dist!r}")


def sample_residual_s(rng: random.Random, cfg=None) -> float:
    """Time left in a stay observed at a random moment: the stationary
    residual. A stay in progress is length-biased (density d f(d) / E[D]),
    drawn here by accepting a stay d with probability d / hi, and the time
    left is a uniform fraction of it. Drawing from f itself would overstate
    the time left; starting from zero would set every initial dweller off at
    once."""
    lo, hi = _params(cfg)[:2]
    if hi <= 0.0:
        return 0.0
    for _ in range(100000):
        d = sample_s(rng, cfg)
        if rng.random() * hi <= d:
            return rng.random() * d
    raise RuntimeError("length-biased dwell draw did not converge")


def to_steps(seconds: float, dt: float) -> int:
    """Whole simulation steps that cover `seconds`: rounded up, so a stay
    never ends before the time drawn."""
    return int(math.ceil(float(seconds) / float(dt) - 1e-9))


def dwell_config_problems(cfg) -> List[str]:
    """Configuration errors in the dwell settings, for configs.validate."""
    p: List[str] = []
    rng = cfg.CROWD_DWELL_RANGE_S
    if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
        return [f"CROWD_DWELL_RANGE_S={rng!r} must be (low, high) in seconds"]
    lo, hi = (float(v) for v in rng)
    if not 0.0 <= lo <= hi:
        p.append(f"CROWD_DWELL_RANGE_S={rng!r} must have 0 <= low <= high")
    dist = cfg.CROWD_DWELL_DISTRIBUTION
    if dist not in DISTRIBUTIONS:
        p.append(f"CROWD_DWELL_DISTRIBUTION={dist!r}; expected one of "
                 f"{DISTRIBUTIONS}")
    frac = cfg.CROWD_INITIAL_DWELL_FRACTION
    if not (isinstance(frac, (int, float)) and 0.0 <= float(frac) <= 1.0):
        p.append(f"CROWD_INITIAL_DWELL_FRACTION={frac!r} must be in [0, 1]")
    if p:
        return p

    def positive(name):
        v = cfg[name]
        if not (isinstance(v, (int, float)) and float(v) > 0.0):
            p.append(f"{name}={v!r} must be a positive number when "
                     f"CROWD_DWELL_DISTRIBUTION={dist!r}")
            return False
        return True

    if dist == "exponential" and positive("CROWD_DWELL_EXP_SCALE_S"):
        s = float(cfg.CROWD_DWELL_EXP_SCALE_S)
        if hi > lo and math.exp(-lo / s) - math.exp(-hi / s) < MIN_MASS:
            p.append("the exponential dwell distribution has almost no "
                     "probability inside CROWD_DWELL_RANGE_S")
    if dist == "lognormal":
        ok = positive("CROWD_DWELL_LOGNORMAL_MEDIAN_S")
        ok = positive("CROWD_DWELL_LOGNORMAL_SIGMA") and ok
        if ok and lo <= 0.0:
            p.append("a lognormal dwell needs CROWD_DWELL_RANGE_S low > 0")
        elif ok and hi > lo:
            mu = math.log(float(cfg.CROWD_DWELL_LOGNORMAL_MEDIAN_S))
            sg = float(cfg.CROWD_DWELL_LOGNORMAL_SIGMA)
            nd = NormalDist()
            za, zb = (math.log(lo) - mu) / sg, (math.log(hi) - mu) / sg
            mass = (nd.cdf(-za) - nd.cdf(-zb)) if za > 0 else (nd.cdf(zb) - nd.cdf(za))
            if mass < MIN_MASS:
                p.append("the lognormal dwell distribution has almost no "
                         "probability inside CROWD_DWELL_RANGE_S")
    return p
