"""The seven street patterns the corpus actually contains.

Each function draws a street network for one morphology tag from
`osm_corpus/sites.py`. They exist separately rather than as one parameterised
family because the differences are structural, not a matter of degree: a
medina is not a grid with more noise, it is a network of lanes at walking
scale with dead ends that a grid never has, and a superblock is not a coarse
grid, it is wide arterials around an interior that is barely served at all.

Keeping them separate also buys the reporting that motivated the choice. A
generated `medina` level can be held against Chandni Chowk, Khan el-Khalili
and Marrakesh Medina specifically, instead of against a corpus average that
no single real place resembles.

Two numbers describe a pattern, and both are measured rather than chosen:

    BLOCK_SPACING_M   how far apart the streets run, in metres
    STREET_SHARE      what fraction of the crop is street

Everything else follows. For two street families spaced `b` apart with
carriageway width `w`, the blocks are (b - w) square, so the street share is
1 - ((b - w) / b)^2 and therefore

    w = b * (1 - sqrt(1 - street_share))

The width is the derived quantity, which is the right way round: the corpus
counts blocks in a crop and measures how much of it is street, and neither of
those is a width. An earlier version of this file inverted the dependency,
picking widths from street-class bands and deriving the spacing. For a
colonial grid that demanded a spacing wider than the crop, the calibration
loop pushed the spacing up until no street survived, and the pattern rendered
as a single solid block covering the entire map.

Spacing is absolute, not a fraction of the crop, because a real block is
about sixty metres across whether you look at two hundred metres of the city
or four hundred. A bigger crop holds more blocks, not bigger ones.

Evidence behind each pattern, counted from the corpus:

    grid            9 sites    colonial_grid   2 sites
    lowrise_dense   8 sites    boulevard       1 site
    organic         5 sites    superblock      1 site
    medina          3 sites

`boulevard` and `superblock` rest on one crop each. Read those two as
plausible rather than as fitted; the corpus does not constrain them.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import ROBOT_BODY_RADIUS

from citygen.plan import CityPlan, Street

# No corridor may be narrower than this, whatever a morphology asks for. A
# lane the robot cannot enter is not a difficulty, it is a wall that looks
# like a lane: the crowd queues at it for a reason the policy can neither see
# nor fix, and the episode then measures the geometry rather than the policy.
MIN_CORRIDOR_M = 2.0 * ROBOT_BODY_RADIUS + 1.0

# Street spacing in metres, and the share of the crop that is street.
#
# Fitted to the road-derived corpus by `python3 -m citygen.fit`, pooling all
# crop sizes. Pooling is legitimate because spacing is an absolute length: a
# real block is about sixty metres across whether the crop is 100 m or 400 m,
# so a larger crop holds more blocks rather than bigger ones. Pooling triples
# the crops behind each figure, which matters most for the morphologies that
# have one or two sites.
#
# Spacing is recovered as size / sqrt(block count). That is exact for a square
# grid and approximate for everything else, which is the right trade: the
# corpus counts blocks, and a block count is far more stable to measure than
# individual streets in a rendered layout.
#
# Crops behind each figure:
#     grid 27, lowrise_dense 23, organic 15, medina 9,
#     colonial_grid 6, boulevard 3, superblock 3
BLOCK_SPACING_M: Dict[str, float] = {
    "boulevard": 47.1,
    "colonial_grid": 63.2,
    "grid": 58.3,
    "lowrise_dense": 50.0,
    "medina": 60.3,
    "organic": 48.5,
    "superblock": 63.2,
}

STREET_SHARE: Dict[str, float] = {
    "boulevard": 0.375,
    "colonial_grid": 0.236,
    "grid": 0.334,
    "lowrise_dense": 0.328,
    "medina": 0.315,
    "organic": 0.310,
    "superblock": 0.266,
}

# Two caveats on the table above, both visible in `citygen.fit` output.
#
# The medina figures spread more than any other morphology: at 200 m the three
# sites report street shares of 0.233, 0.441 and 0.556. A dense irregular alley
# network is exactly the case the square-grid spacing formula fits worst, and a
# 60 m spacing does not describe Marrakesh. The median is used because it is
# what the corpus supports, not because the model is a good one there.
#
# An earlier hand-set medina used a 32 m spacing on the reasoning that medina
# lanes are at walking scale. The corpus disagrees, and the corpus wins: what
# few enclosed faces OSM's road data cuts a medina into are large, whatever
# the lanes between them look like.

# Width of each street class relative to the network's mean carriageway.
# Relative rather than absolute so the mean stays where the corpus puts it: an
# absolute band would override the measured street share whenever a pattern
# happened to draw more arterials than usual.
CLASS_WIDTH: Dict[str, float] = {
    "arterial": 1.75,
    "street": 1.0,
    "lane": 0.62,
    "alley": 0.38,
    "stub": 0.38,
}


def base_width(morphology: str, spacing: Optional[float] = None,
               share: Optional[float] = None) -> float:
    """Mean carriageway width implied by a pattern's spacing and share."""
    b = BLOCK_SPACING_M[morphology] if spacing is None else float(spacing)
    s = STREET_SHARE[morphology] if share is None else float(share)
    s = max(0.02, min(0.95, s))
    return max(MIN_CORRIDOR_M, b * (1.0 - math.sqrt(1.0 - s)))


def _w(rng: random.Random, base: float, kind: str) -> float:
    rel = CLASS_WIDTH.get(kind, 1.0)
    return max(MIN_CORRIDOR_M, base * rel * rng.uniform(0.85, 1.15))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _jitter_line(p0, p1, rng: random.Random, amp: float,
                 segments: int = 1) -> List[Tuple[float, float]]:
    """A centreline from p0 to p1, bent by up to `amp` metres.

    Straight when amp is zero, which is what a planned grid wants. The bend is
    perpendicular to the line, so the street still goes where it was meant to.
    """
    if segments < 1:
        segments = 1
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return [p0, p1]
    nx, ny = -dy / length, dx / length
    pts = [p0]
    for i in range(1, segments):
        f = i / segments
        off = rng.uniform(-amp, amp) if amp > 0 else 0.0
        pts.append((p0[0] + dx * f + nx * off, p0[1] + dy * f + ny * off))
    pts.append(p1)
    return pts


def _cut_positions(extent: float, target: float, rng: random.Random,
                   irregularity: float) -> List[float]:
    """Street positions across `extent`, about `target` apart.

    Positions rather than a count, so an irregular fabric has genuinely uneven
    blocks. Drawing a count and dividing evenly gives uniform blocks with a
    jittered boundary, which reads as a grid with noise and not as an
    irregular plan.
    """
    if target <= 0:
        return []
    pos, x = [], 0.0
    while True:
        step = target * (1.0 + rng.uniform(-irregularity, irregularity))
        x += max(target * 0.35, step)
        if x >= extent - target * 0.2:
            break
        pos.append(x)
    return pos


def _rotated_grid(W: int, H: int, rng: random.Random, spacing: float,
                  base: float, irregularity: float, bend: float,
                  minor_kind: str = "street",
                  arterial_every: int = 0,
                  cross_spacing: Optional[float] = None) -> List[Street]:
    """Two families of streets at right angles, rotated as a whole.

    Rotated as a whole, not per street: per-street angles make the families
    cross at varying angles, which produces acute slivers no real plan
    contains and pinch points the navmesh then has to resolve.
    """
    theta = rng.uniform(0.0, math.pi / 2.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    span = math.hypot(W, H)
    cx, cy = W / 2.0, H / 2.0

    def to_world(u, v):
        return (cx + u * cos_t - v * sin_t, cy + u * sin_t + v * cos_t)

    streets: List[Street] = []
    cross_spacing = cross_spacing or spacing

    for axis, target, kind in ((0, spacing, "street"),
                               (1, cross_spacing, minor_kind)):
        offsets = [o - span / 2.0
                   for o in _cut_positions(span, target, rng, irregularity)]
        for k, off in enumerate(offsets):
            if axis == 0:
                p0, p1 = to_world(-span / 2.0, off), to_world(span / 2.0, off)
            else:
                p0, p1 = to_world(off, -span / 2.0), to_world(off, span / 2.0)
            wide = arterial_every and (k % arterial_every == 0)
            this = "arterial" if wide else kind
            streets.append(Street(
                _jitter_line(p0, p1, rng, bend, segments=3 if bend > 0 else 1),
                _w(rng, base, this), this))
    return streets


def _crossing(W: int, H: int, rng: random.Random, base: float, kind: str,
              bend: float, segments: int) -> Street:
    """One street cutting right across the crop in a random direction."""
    a = rng.uniform(0, 2 * math.pi)
    cx, cy = rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.15 * H, 0.85 * H)
    r = math.hypot(W, H)
    p0 = (cx - math.cos(a) * r, cy - math.sin(a) * r)
    p1 = (cx + math.cos(a) * r, cy + math.sin(a) * r)
    return Street(_jitter_line(p0, p1, rng, bend, segments), _w(rng, base, kind),
                  kind)


# ---------------------------------------------------------------------------
# the patterns. `t` is development: 0 is bare, 1 is the full pattern.
# ---------------------------------------------------------------------------

def grid(W, H, rng, t, scale=1.0):
    """Nine sites, Eixample to SoHo. Regular blocks, streets at one scale."""
    sp = BLOCK_SPACING_M["grid"] * scale
    return _rotated_grid(W, H, rng, sp, base_width("grid"),
                         irregularity=0.10, bend=0.0, arterial_every=3)


def colonial_grid(W, H, rng, t, scale=1.0):
    """French Quarter, Centro Historico. Squarer and stricter than a grid."""
    sp = BLOCK_SPACING_M["colonial_grid"] * scale
    return _rotated_grid(W, H, rng, sp, base_width("colonial_grid"),
                         irregularity=0.04, bend=0.0, minor_kind="lane",
                         arterial_every=4)


def lowrise_dense(W, H, rng, t, scale=1.0):
    """Eight sites, Shibuya to Surry Hills. A fine mesh, permeable below it."""
    sp = BLOCK_SPACING_M["lowrise_dense"] * scale
    base = base_width("lowrise_dense")
    streets = _rotated_grid(W, H, rng, sp, base, irregularity=0.30, bend=1.5,
                            minor_kind="lane", arterial_every=4)
    # Mid-block alleys. This is what separates the pattern from a plain grid:
    # the fabric is permeable at a scale below the street network itself.
    for _ in range(int(round(_lerp(0, 5, t)))):
        streets.append(_crossing(W, H, rng, base, "alley", 2.5, 4))
    return streets


def organic(W, H, rng, t, scale=1.0):
    """Five sites, Covent Garden to Trastevere. Bent streets, uneven blocks."""
    sp = BLOCK_SPACING_M["organic"] * scale
    base = base_width("organic")
    streets = _rotated_grid(W, H, rng, sp, base, irregularity=0.45,
                            bend=sp * 0.16, minor_kind="lane",
                            arterial_every=5)
    # Streets answering to no family at all, which is what makes a medieval
    # core unlike a jittered grid.
    for _ in range(int(round(_lerp(1, 3, t)))):
        streets.append(_crossing(W, H, rng, base, "lane", sp * 0.22, 4))
    return streets


def medina(W, H, rng, t, scale=1.0):
    """Three sites. Lanes at walking scale, and dead ends."""
    sp = BLOCK_SPACING_M["medina"] * scale
    base = base_width("medina")
    streets = _rotated_grid(W, H, rng, sp, base, irregularity=0.55,
                            bend=sp * 0.20, minor_kind="alley",
                            arterial_every=6)
    # Dead ends, the structurally defining feature: a crowd that turns into
    # one has to come back out, and no grid ever asks that of it.
    for _ in range(int(round(_lerp(1, 7, t)))):
        x0, y0 = rng.uniform(0.1 * W, 0.9 * W), rng.uniform(0.1 * H, 0.9 * H)
        a = rng.uniform(0, 2 * math.pi)
        length = sp * rng.uniform(0.4, 0.9)
        p1 = (x0 + math.cos(a) * length, y0 + math.sin(a) * length)
        streets.append(Street(_jitter_line((x0, y0), p1, rng, 2.0, 2),
                              _w(rng, base, "stub"), "stub"))
    return streets


def boulevard(W, H, rng, t, scale=1.0):
    """Pigalle. Wide radials meeting at a place, with fabric between them.

    One corpus site, so the shape is set by hand. What makes Pigalle Pigalle
    is that several wide streets converge: that produces acute-angled blocks a
    grid cannot make, and one congestion point the whole crowd must pass.
    """
    base = base_width("boulevard")
    cx, cy = W * rng.uniform(0.35, 0.65), H * rng.uniform(0.35, 0.65)
    r = math.hypot(W, H)
    streets: List[Street] = []
    first = rng.uniform(0, math.pi)
    n_radials = int(round(_lerp(3, 5, t)))
    for i in range(n_radials):
        a = first + math.pi * i / n_radials + rng.uniform(-0.12, 0.12)
        streets.append(Street(
            [(cx - math.cos(a) * r, cy - math.sin(a) * r),
             (cx + math.cos(a) * r, cy + math.sin(a) * r)],
            _w(rng, base, "arterial"), "arterial"))
    # Background fabric filling the wedges. Coarser than the radials, because
    # the radials already supply most of the street area.
    streets.extend(_rotated_grid(
        W, H, rng, BLOCK_SPACING_M["boulevard"] * 1.6 * scale, base,
        irregularity=0.35, bend=1.0, minor_kind="alley"))
    return streets


def superblock(W, H, rng, t, scale=1.0):
    """Gangnam. Very wide arterials, very large blocks, thin interior.

    One corpus site, also set by hand. The structure that matters for
    evacuation is the asymmetry: crossing the crop is quick along an arterial
    and slow anywhere else, so where the crowd starts decides the problem.
    """
    sp = BLOCK_SPACING_M["superblock"] * scale
    base = base_width("superblock")
    streets = _rotated_grid(W, H, rng, sp, base, irregularity=0.08, bend=0.0,
                            arterial_every=1)
    # Interior service lanes, few and narrow, appearing only as the fabric
    # develops. They are what makes the interior reachable at all.
    for _ in range(int(round(_lerp(0, 4, t)))):
        streets.append(_crossing(W, H, rng, base, "lane", 0.0, 1))
    return streets


MORPHOLOGIES: Dict[str, Callable[..., List[Street]]] = {
    "grid": grid,
    "colonial_grid": colonial_grid,
    "lowrise_dense": lowrise_dense,
    "organic": organic,
    "medina": medina,
    "boulevard": boulevard,
    "superblock": superblock,
}

# How many corpus sites back each pattern. Reported beside results so a
# per-morphology number is read with the right amount of confidence.
CORPUS_SUPPORT: Dict[str, int] = {
    "grid": 9, "lowrise_dense": 8, "organic": 5, "medina": 3,
    "colonial_grid": 2, "boulevard": 1, "superblock": 1,
}


def street_network(morphology: str, W: int, H: int, rng: random.Random,
                   development: float, scale: float = 1.0) -> List[Street]:
    try:
        fn = MORPHOLOGIES[morphology]
    except KeyError:
        raise KeyError(f"unknown morphology {morphology!r}; "
                       f"known: {sorted(MORPHOLOGIES)}")
    return fn(int(W), int(H), rng, float(development), scale)
