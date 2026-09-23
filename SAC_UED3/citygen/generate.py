"""Drawing a level: pick a morphology, develop it, calibrate, place exits.

The development axis is the one the curriculum moves along. At development 0
no block is built, so the crop is an open field and the robot's only problem
is reaching the exit. As development rises, blocks build up, and the fabric
that appears is whichever morphology was drawn rather than a generic one.

Density is calibrated rather than predicted. `_block_spacing` gets a plain
grid to its target street share exactly, but every morphology adds something
the formula does not see: wider arterials every few streets, mid-block alleys,
medina dead ends, boulevard radials, and the small closing buffer the shared
geometry kernel applies to weld hairline gaps. Measured against the eight
crops available while this was written, those extras cost between 0.01 and
0.38 of street share depending on the pattern, which is far too much to leave
uncorrected and too pattern-specific to model. So the spacing is scaled until
the rendered street share matches, which is robust to every one of them.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from citygen.morphology import (CORPUS_SUPPORT, MIN_CORRIDOR_M, MORPHOLOGIES,
                                STREET_SHARE, street_network)
from citygen.plan import BlockMark, CityPlan, Street

# How close the rendered street share has to come to the morphology's target
# before calibration stops. Tighter than this is chasing render noise: the
# closing buffer alone moves the figure by a few thousandths.
CALIBRATION_TOLERANCE = 0.02
CALIBRATION_PASSES = 4

# Built fraction across the development axis. Zero at the bottom, because that
# is the empty field. One at the top, which needs justifying, because a real
# downtown crop obviously does hold squares, yards and car parks.
#
# Under road-derived traversability those are already counted. A square is
# mapped as a pedestrian area and a yard is not mapped as road at all, so the
# first is street and the second is block before any of this runs. The corpus
# shows it directly: real 200 m crops measure 0.672 built against a 0.328
# street share, and those sum to one, so nothing is left over to be an
# "unbuilt block".
#
# Holding this at 0.92 was double-counting. It cost about 0.06 of coverage at
# the top of the axis, which is the difference between a generated downtown
# and a generated downtown with a hole in it.
BUILT_AT_ZERO = 0.0
BUILT_AT_FULL = 1.0

# Exit openings, in metres. Absolute rather than scaled: a door is a physical
# width, and the free-flow evacuation estimate divides the crowd by exit width,
# so scaling it with the map would change the success criterion's meaning.
EXIT_WIDTH_M = (8.0, 14.0)
EXIT_DEPTH_M = 5.0
EXIT_INSET_M = 1.0


def development_for(difficulty: int, span: Tuple[int, int] = (0, 6)) -> float:
    """Difficulty tier to development level.

    Difficulty 0 maps to exactly 0, which is the empty field, and it has to be
    exact rather than merely small: a nearly empty field still has obstacles,
    and the point of the bottom tier is that the curriculum starts from a map
    where nothing has to be avoided at all.
    """
    lo, hi = span
    if difficulty <= lo:
        return 0.0
    return (float(difficulty) - lo) / max(1.0, float(hi - lo))


def _street_share(plan: CityPlan) -> float:
    rings, trav = plan.render()
    area = float(plan.width * plan.height)
    if trav is None:
        return 0.0
    return float(trav.area / area)


def _scaled(streets: List[Street], k: float) -> List[Street]:
    return [Street(list(st.points), max(MIN_CORRIDOR_M, st.width_m * k), st.kind)
            for st in streets]


def calibrated_network(morphology: str, W: int, H: int, rng: random.Random,
                       development: float,
                       target: Optional[float] = None) -> List[Street]:
    """A street network whose rendered street share hits the target.

    Calibration scales the carriageway widths, holding the street spacing at
    the value the corpus measured. That is the right way round for the same
    reason the width is derived in the first place: the spacing is what was
    counted in real crops, and the width is already whatever the spacing and
    the share imply, so nudging it is nudging the derived number rather than
    the measured one.

    Scaling the spacing instead does not even work for every pattern. A
    boulevard's street area is dominated by a fixed number of radials and a
    superblock's by arterials that are wide whatever their spacing, so
    widening the spacing removes streets without removing much street: those
    two stayed 0.09 and 0.05 above target however many passes ran. A width
    scale moves every feature at once, including the radials.

    Spacing is still the fallback. If the widths hit the floor set by the
    robot's own size and the crop is still too open, the only way left to
    remove street area is to have fewer streets.
    """
    target = STREET_SHARE.get(morphology, 0.30) if target is None else target

    def share_of(streets):
        # Measured with every block built, because that is the quantity the
        # network controls. How much is then left open is the development
        # axis, and a separate decision.
        probe = CityPlan(width=W, height=H, morphology=morphology,
                         development=development, streets=streets)
        return _street_share(probe)

    spacing_scale = 1.0
    for _ in range(3):
        drawn = street_network(morphology, W, H, rng, development,
                               spacing_scale)
        if not drawn:
            return drawn
        k = 1.0
        streets = drawn
        for _ in range(CALIBRATION_PASSES):
            share = share_of(streets)
            if share <= 1e-6 or abs(share - target) <= CALIBRATION_TOLERANCE:
                return streets
            # Street area is close to linear in width at these widths, so the
            # ratio of achieved to wanted share is a direct correction.
            k *= max(0.35, min(2.5, target / share))
            streets = _scaled(drawn, k)
        # Widths are on the floor and the crop is still too open: fewer
        # streets is the only remaining move.
        if share_of(streets) > target and all(
                st.width_m <= MIN_CORRIDOR_M + 1e-9 for st in streets):
            spacing_scale *= 1.35
            continue
        return streets
    return streets


def _place_exits(plan: CityPlan, n_exits: int,
                 rng: random.Random) -> List[List[Tuple[int, int]]]:
    """Exits on the wall, where the main street network reaches it.

    An exit is the researcher's choice and not a feature of the map, so it is
    not read off the fabric. But it does have to open onto ground the crowd can
    actually stand on and walk away from, and on a road-derived layout most of
    the boundary is the side of a block.

    The candidates are therefore boundary cells belonging to the largest
    connected component of space the robot's body fits in. Choosing on raw
    traversable space instead is not enough: a medina lane can touch the wall
    and be pinched below the robot's width just inside, and an exit placed
    there opened onto a pocket holding two per cent of the walkable ground
    while the rest of the network had no exit at all.
    """
    import numpy as np
    from scipy.ndimage import label
    from shapely.geometry import box

    from citygen.validate import RASTER_STEP_M, passable_mask

    W, H = float(plan.width), float(plan.height)
    rings, trav = plan.render()
    if not rings:
        # An empty field: every stretch of wall is as good as any other.
        candidates = [(side, u)
                      for side in ("bottom", "right", "top", "left")
                      for u in np.linspace(0.1, 0.9, 9) *
                      (W if side in ("bottom", "top") else H)]
    else:
        fits, _, _ = passable_mask(plan, [])
        if not fits.any():
            return []
        labels, _ = label(fits)
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        main = int(counts.argmax())
        mask = labels == main

        step = RASTER_STEP_M
        ny, nx = mask.shape
        # A cell within the robot's radius of the wall cannot hold its centre,
        # so "reaches the wall" means reaching that band.
        from config import ROBOT_BODY_RADIUS
        band = max(1, int(round((ROBOT_BODY_RADIUS + 1.0) / step)))
        candidates = []
        for j in np.flatnonzero(mask[:band, :].any(axis=0)):
            candidates.append(("bottom", (j + 0.5) * step))
        for j in np.flatnonzero(mask[-band:, :].any(axis=0)):
            candidates.append(("top", (j + 0.5) * step))
        for i in np.flatnonzero(mask[:, :band].any(axis=1)):
            candidates.append(("left", (i + 0.5) * step))
        for i in np.flatnonzero(mask[:, -band:].any(axis=1)):
            candidates.append(("right", (i + 0.5) * step))

    if not candidates:
        return []

    def perimeter_pos(side, u):
        return {"bottom": u, "right": W + u,
                "top": W + H + (W - u), "left": 2 * W + H + (H - u)}[side]

    perimeter = 2 * (W + H)
    candidates = list(candidates)
    rng.shuffle(candidates)
    chosen: List[Tuple[str, float]] = []
    for min_sep in (0.30 * perimeter, 0.12 * perimeter, 0.0):
        for side, u in candidates:
            if len(chosen) >= n_exits:
                break
            p = perimeter_pos(side, u)
            if any(min(abs(p - perimeter_pos(s2, u2)),
                       perimeter - abs(p - perimeter_pos(s2, u2))) < min_sep
                   for s2, u2 in chosen):
                continue
            chosen.append((side, u))
        if len(chosen) >= n_exits:
            break

    out = []
    for side, u in chosen[:n_exits]:
        half = rng.uniform(*EXIT_WIDTH_M) / 2.0
        d, ins = EXIT_DEPTH_M, EXIT_INSET_M
        if side in ("bottom", "top"):
            lo, hi = max(ins, u - half), min(W - ins, u + half)
            if hi - lo < MIN_CORRIDOR_M:
                continue
            poly = box(lo, ins, hi, ins + d) if side == "bottom" \
                else box(lo, H - ins - d, hi, H - ins)
        else:
            lo, hi = max(ins, u - half), min(H - ins, u + half)
            if hi - lo < MIN_CORRIDOR_M:
                continue
            poly = box(ins, lo, ins + d, hi) if side == "left" \
                else box(W - ins - d, lo, W - ins, hi)
        coords = list(poly.exterior.coords)[:-1]
        out.append([(int(round(x)), int(round(y))) for x, y in coords])
    return out


def generate_city_plan(rng: random.Random, morphology: Optional[str] = None,
                       difficulty: Optional[int] = None,
                       width: int = 100, height: int = 100,
                       development: Optional[float] = None) -> CityPlan:
    """One street plan, developed to the requested level."""
    if morphology is None:
        morphology = rng.choice(sorted(MORPHOLOGIES))
    if development is None:
        development = development_for(0 if difficulty is None else difficulty)

    if development <= 0.0:
        # The empty field. No network at all, rather than a network with every
        # block left open: an unbuilt block still has street around it, and at
        # the bottom of the axis there should be nothing to walk around.
        plan = CityPlan(width=int(width), height=int(height),
                        morphology=morphology, development=0.0, streets=[])
        # Mark the one face explicitly open. Without this the plan carries no
        # decisions at all, and the moment mutation cuts the first street into
        # it both halves default to built, so the entire crop turns to
        # obstacle and every child of an empty field is rejected. The
        # curriculum could then never leave difficulty 0 by editing, which is
        # precisely the progression the bottom tier exists to start.
        plan.marks = [BlockMark(width / 2.0, height / 2.0, built=False)]
        return plan

    streets = calibrated_network(morphology, int(width), int(height), rng,
                                 development)
    plan = CityPlan(width=int(width), height=int(height),
                    morphology=morphology, development=float(development),
                    streets=streets)
    built = BUILT_AT_ZERO + (BUILT_AT_FULL - BUILT_AT_ZERO) * development
    plan.place_marks(built, rng)
    return plan
