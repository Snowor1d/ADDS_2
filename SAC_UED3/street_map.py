#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A second map generator: streets first, buildings second.

`random_map.py` scatters obstacles across open ground and lets the free space
be whatever is left over. Measured against real downtown crops that produces
the wrong thing in one specific, decisive way: a real downtown's free space is
a network of streets between building masses, and the generator's is one large
connected clearing with lumps in it.

    statistic                      real 200 m crops      random_map
    free width p50                 about 17 m            about 58 m
    fraction >40 m from a building 0.20                  0.63
    free-space components          13 to 17              always 1

So this generator inverts the order. It lays out a street network first, fills
the resulting blocks with footprints, and leaves the streets as the free space.
The free space is then channelised by construction rather than by luck, which
is the property the guidance task actually turns on: in a street network the
robot's decision is which route to commit the crowd to, and in a clearing
almost any direction works.

It is a second family rather than a replacement. Parks, plazas and campus
grounds really are open ground with obstacles on them, and `random_map` makes
those well. Sampling between the two is what covers the range.

The parameters are the axes urban morphology actually varies along, so the
range each one is given is what decides whether the training distribution
contains the real one:

  regularity      perfect grid through to organic, by jittering the cut lines
  orientation     one dominant street axis, two crossing, or none
  block_scale     characteristic block size relative to the map
  coverage        built fraction within blocks
  setback         gap between a footprint and its block edge
  street_hierarchy how much wider arterials are than side streets
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from shapely.affinity import rotate as _rotate
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

from config import ROBOT_BODY_RADIUS

Ring = List[List[int]]

# Two adjacent blocks end up exactly one street width apart, so the street
# width is what decides whether the crowd can get between them. The coverage
# feedback narrows streets to hit its target and will happily go below this,
# which produces layouts that are dense on paper and impassable in fact.
MIN_STREET_WIDTH_M = 2.0 * ROBOT_BODY_RADIUS + 1.0


@dataclass
class StreetMapSpec:
    width: int
    height: int
    seed: Optional[int] = None

    # 0 = perfect grid, 1 = fully organic. Interpolates the jitter applied to
    # every cut line and to every block edge.
    regularity: float = 0.5

    # Characteristic block side as a fraction of the smaller map dimension.
    block_scale: float = 0.28

    # Built fraction of the whole map, which is the statistic the corpus
    # reports as "coverage".
    coverage: float = 0.35

    # Street widths in metres. Arterials are drawn from the wide end.
    street_width_min: float = 8.0
    street_width_max: float = 16.0
    arterial_fraction: float = 0.25
    arterial_multiplier: float = 1.8

    # Gap between a footprint and the edge of its block.
    # Gap between a footprint and the edge of its block. Two neighbours are
    # therefore 2 x this apart, which is what keeps a level passable.
    setback_m: float = 2.0

    # Number of dominant street directions: 1 gives parallel strips, 2 a grid,
    # 0 lets every cut pick its own angle.
    n_orientations: int = 2

    # Fraction of blocks left unbuilt. A real crop holds squares, yards and
    # car parks, and without them the generator reports almost no ground more
    # than 40 m from a building where real crops report a fifth of the area.
    open_block_fraction: float = 0.2

    # Footprints per block, and how uneven they are.
    splits_per_block: Tuple[int, int] = (1, 4)

    attempts: int = 60


@dataclass
class StreetMapData:
    width: int
    height: int
    obstacles: List[Ring]
    streets: List[Polygon] = field(default_factory=list)
    seed_used: int = 0
    coverage: float = 0.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _subdivide(rng: random.Random, rect: Tuple[float, float, float, float],
               target: float, regularity: float,
               depth: int = 0, max_depth: int = 6) -> List[Tuple[float, float, float, float]]:
    """Recursively cut a rectangle until its parts are near the block size.

    Cutting first and building second is the whole point: the cut lines become
    the streets, so the free space is a network rather than a remainder.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if depth >= max_depth or (w <= target and h <= target):
        return [rect]

    # Jitter the cut position; a grid cuts in the middle, organic fabric does
    # not.
    spread = _lerp(0.04, 0.34, 1.0 - regularity)
    frac = 0.5 + rng.uniform(-spread, spread)

    if w >= h:
        cut = x0 + w * frac
        return (_subdivide(rng, (x0, y0, cut, y1), target, regularity, depth + 1, max_depth) +
                _subdivide(rng, (cut, y0, x1, y1), target, regularity, depth + 1, max_depth))
    cut = y0 + h * frac
    return (_subdivide(rng, (x0, y0, x1, cut), target, regularity, depth + 1, max_depth) +
            _subdivide(rng, (x0, cut, x1, y1), target, regularity, depth + 1, max_depth))


def _block_polygon(rng: random.Random, rect, spec: StreetMapSpec,
                   angle: float, street_gain: float = 1.0) -> Optional[Polygon]:
    """A block shrunk by half its bounding street widths, then rotated."""
    x0, y0, x1, y1 = rect
    wide = rng.random() < spec.arterial_fraction
    base = rng.uniform(spec.street_width_min, spec.street_width_max) * street_gain
    street = max(MIN_STREET_WIDTH_M,
                 base * (spec.arterial_multiplier if wide else 1.0))

    inset = street / 2.0
    bx0, by0 = x0 + inset, y0 + inset
    bx1, by1 = x1 - inset, y1 - inset
    if bx1 - bx0 < 6.0 or by1 - by0 < 6.0:
        return None

    poly = box(bx0, by0, bx1, by1)
    if abs(angle) > 1e-9:
        poly = _rotate(poly, angle, origin="center")
    return poly


def _split_block(rng: random.Random, block: Polygon, spec: StreetMapSpec) -> List[Polygon]:
    """Divide a block into footprints, leaving a setback around each.

    Real blocks are subdivided into parcels, which is why a dense downtown crop
    holds tens of footprints rather than a handful; random_map produces 4 to 11
    where Paris and Barcelona produce 39 to 46.
    """
    lo, hi = spec.splits_per_block
    n = rng.randint(lo, hi)
    parts = [block]
    for _ in range(n - 1):
        parts.sort(key=lambda p: -p.area)
        target = parts.pop(0)
        minx, miny, maxx, maxy = target.bounds
        w, h = maxx - minx, maxy - miny
        if max(w, h) < 14.0:
            parts.append(target)
            continue
        spread = _lerp(0.05, 0.3, 1.0 - spec.regularity)
        frac = 0.5 + rng.uniform(-spread, spread)
        if w >= h:
            cut = minx + w * frac
            a = target.intersection(box(minx, miny, cut, maxy))
            b = target.intersection(box(cut, miny, maxx, maxy))
        else:
            cut = miny + h * frac
            a = target.intersection(box(minx, miny, maxx, cut))
            b = target.intersection(box(minx, cut, maxx, maxy))
        for piece in (a, b):
            if piece.geom_type == "Polygon" and piece.area > 20:
                parts.append(piece)

    out = []
    # Two footprints in one block are 2 x setback apart, so the setback floor
    # is half the passability gap.
    setback = max(MIN_STREET_WIDTH_M / 2.0, spec.setback_m)
    for piece in parts:
        shrunk = piece.buffer(-setback, join_style=2)
        if shrunk.is_empty:
            continue
        for part in (shrunk.geoms if shrunk.geom_type == "MultiPolygon" else [shrunk]):
            if part.geom_type == "Polygon" and part.area >= 25.0:
                out.append(part)
    return out


def _ring_int(poly: Polygon) -> Ring:
    coords = list(poly.exterior.coords)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    return [[int(round(x)), int(round(y))] for x, y in coords]


def generate_street_map(spec: StreetMapSpec) -> StreetMapData:
    """Lay out streets, fill the blocks, return the footprints.

    The caller's global RNG stream is left untouched, matching
    random_map.generate_map: a map seed must not become a hidden input to the
    crowd spawn or the per-pedestrian parameters drawn afterwards.
    """
    seed_used = spec.seed if spec.seed is not None else random.randrange(0, 2**31 - 1)
    saved = random.getstate()
    try:
        return _generate_seeded(spec, seed_used)
    finally:
        random.setstate(saved)


def _generate_seeded(spec: StreetMapSpec, seed_used: int) -> StreetMapData:
    rng = random.Random(seed_used)
    W, H = int(spec.width), int(spec.height)

    if spec.coverage <= 1e-9:
        # The empty-room tier: no footprints at all, matching random_map's
        # difficulty 0 so the two families agree at the easy end.
        return StreetMapData(width=W, height=H, obstacles=[],
                             seed_used=seed_used, coverage=0.0)
    target = max(20.0, spec.block_scale * min(W, H))

    # Dominant street directions. Two gives a grid; one gives parallel strips;
    # zero lets each block choose, which is what organic fabric looks like.
    if spec.n_orientations <= 0:
        angles = None
    else:
        base = rng.uniform(0.0, 90.0)
        angles = [base + i * (90.0 / max(1, spec.n_orientations))
                  for i in range(spec.n_orientations)]

    # `coverage` used to only pick the closest of several independent attempts,
    # which left it almost inert: every tier came out near 0.22 while real
    # crops reach 0.49, because what actually sets coverage is the ratio of
    # block area to cell area and that is fixed by the street width and the
    # setbacks. So each attempt now feeds the last one's error back into those
    # two, which is the shortest path from the knob to the quantity it names.
    street_gain = 1.0
    block_gain = 1.0

    best: Optional[StreetMapData] = None
    for attempt in range(spec.attempts):
        rects = _subdivide(rng, (0.0, 0.0, float(W), float(H)),
                           target * block_gain, spec.regularity)

        # One angle for the whole layout, not one per block. Rotating each
        # block about its own centre pushes its corners past the street inset,
        # so two neighbours could end up touching however wide the street was
        # meant to be; measured gaps came out under a metre against a three
        # metre floor. Real street grids are rotated as a whole, and the
        # irregularity of organic fabric comes from the jittered subdivision
        # rather than from blocks pointing in different directions.
        layout_angle = (rng.uniform(0.0, 90.0) if angles is None
                        else rng.choice(angles) % 90.0)

        footprints: List[Polygon] = []
        for rect in rects:
            if rng.random() < spec.open_block_fraction:
                continue                      # left as open ground
            block = _block_polygon(rng, rect, spec, 0.0, street_gain)
            if block is None:
                continue
            footprints.extend(_split_block(rng, block, spec))

        if not footprints:
            continue

        world = box(0, 0, W, H)
        if abs(layout_angle) > 1e-9:
            centre = (W / 2.0, H / 2.0)
            footprints = [_rotate(p, layout_angle, origin=centre) for p in footprints]
        clipped_all = []
        for piece in footprints:
            c = piece.intersection(world)
            if c.is_empty:
                continue
            for part in (c.geoms if c.geom_type == "MultiPolygon" else [c]):
                if part.geom_type == "Polygon" and part.area >= 25.0:
                    clipped_all.append(part)
        footprints = clipped_all
        if not footprints:
            continue

        merged = unary_union(footprints)
        parts = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
        parts = [p for p in parts if p.geom_type == "Polygon" and p.area >= 25.0]
        if not parts:
            continue

        # Clipping and integer rounding can still bring two masses closer than
        # the robot can pass, so the invariant is enforced here rather than
        # merely intended upstream. Largest first, so what survives is the
        # structure that matters.
        kept: List[Polygon] = []
        for poly in sorted(parts, key=lambda p: -p.area):
            if any(poly.distance(k) < MIN_STREET_WIDTH_M for k in kept):
                continue
            kept.append(poly)
        parts = kept
        if not parts:
            continue

        cov = sum(p.area for p in parts) / float(W * H)
        data = StreetMapData(width=W, height=H,
                             obstacles=[_ring_int(p) for p in parts],
                             seed_used=seed_used, coverage=cov)
        if best is None or abs(cov - spec.coverage) < abs(best.coverage - spec.coverage):
            best = data
        if abs(cov - spec.coverage) <= 0.02:
            return data

        # Narrow the streets and enlarge the blocks when short of the target,
        # and the reverse when over it. Clamped so a target the layout cannot
        # reach degrades instead of collapsing the street network.
        if cov > 1e-6:
            ratio = spec.coverage / cov
            street_gain = max(0.45, min(1.8, street_gain / (ratio ** 0.35)))
            block_gain = max(0.7, min(2.0, block_gain * (ratio ** 0.25)))

    if best is None:
        raise RuntimeError("street map generation produced no footprints")
    return best


def clear_exit_approach(obstacles: Sequence[Ring], exits: Sequence[Polygon],
                        W: int, H: int, depth: float = 12.0) -> List[Ring]:
    """Remove footprints from the mouth of each exit.

    The street layout and the exits are generated independently, so a footprint
    can land squarely in an opening. Clearing the approach is the same
    requirement the real-map export applies: an exit the crowd cannot walk up
    to is not an exit.
    """
    if not exits:
        return list(obstacles)

    aprons = []
    for e in exits:
        x0, y0, x1, y1 = e.bounds
        eps = 1e-6
        if x0 <= eps:
            aprons.append(box(0, y0, min(W, x1 + depth), y1))
        elif x1 >= W - eps:
            aprons.append(box(max(0.0, x0 - depth), y0, W, y1))
        elif y0 <= eps:
            aprons.append(box(x0, 0, x1, min(H, y1 + depth)))
        else:
            aprons.append(box(x0, max(0.0, y0 - depth), x1, H))
    apron = unary_union(aprons)

    out: List[Ring] = []
    for ring in obstacles:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        cut = poly.difference(apron)
        if cut.is_empty:
            continue
        for part in (cut.geoms if cut.geom_type == "MultiPolygon" else [cut]):
            if part.geom_type == "Polygon" and part.area >= 25.0:
                out.append(_ring_int(part))
    return out


def finalise_for_simulation(obstacles: Sequence[Ring], exits: Sequence[Polygon],
                            W: int, H: int) -> List[Ring]:
    """Make a street layout satisfy the invariants the simulator needs.

    Raising coverage toward what real downtowns reach pushes the layout into a
    regime where three separate things break, and all three are about the free
    space rather than about the buildings:

      * a footprint beside an exit leaves no room to stand at the mouth, which
        the approach corridor alone does not fix because it only clears what is
        in front;
      * clipping a rotated layout leaves slivers a metre or two off the
        boundary, so the strip between building and wall is too narrow to walk;
      * at high coverage the street network can pinch closed and strand part of
        the free space away from any exit.

    Each is repaired here rather than avoided by keeping coverage low, because
    the coverage is the point.
    """
    world = box(0, 0, W, H)
    gap = MIN_STREET_WIDTH_M

    polys: List[Polygon] = []
    for ring in obstacles:
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == "Polygon" and not poly.is_empty:
            polys.append(poly)

    # 1) A footprint inside the boundary band either reaches the wall or gets
    # out of the band. Extending is preferred: it keeps the built area, and a
    # building flush to the edge of a crop is what real crops look like.
    walls = (
        ("x0", box(-gap, -gap, 0, H + gap)),
        ("x1", box(W, -gap, W + gap, H + gap)),
        ("y0", box(-gap, -gap, W + gap, 0)),
        ("y1", box(-gap, H, W + gap, H + gap)),
    )
    fixed: List[Polygon] = []
    for poly in polys:
        for _, strip in walls:
            d = poly.distance(strip)
            if 1e-9 < d < gap:
                # Bridge the gap to that wall so nothing is left floating in
                # the band with an unwalkable strip behind it.
                bridged = unary_union([poly, poly.buffer(d + 0.5, join_style=2)
                                       .intersection(strip.buffer(gap))])
                merged = unary_union([poly, bridged]).intersection(world)
                if merged.geom_type == "Polygon" and not merged.is_empty:
                    poly = merged
        if poly.geom_type == "Polygon" and poly.area >= 25.0:
            fixed.append(poly)
    polys = fixed

    # 2) Keep clear of the exits, all the way round rather than only in front.
    # This has to come after the wall bridging, not before: exits sit flush
    # against a wall, so extending a footprint to reach the wall walks it
    # straight back into the exit mouth and undoes the clearance.
    if exits:
        keepout = unary_union([e.buffer(gap, join_style=2) for e in exits])
        cut = []
        for poly in polys:
            rest = poly.difference(keepout)
            if rest.is_empty:
                continue
            for part in (rest.geoms if rest.geom_type == "MultiPolygon" else [rest]):
                if part.geom_type == "Polygon" and part.area >= 25.0:
                    cut.append(part)
        polys = cut

    # Re-merge and re-apply the spacing floor after those edits.
    if polys:
        merged = unary_union(polys)
        parts = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
        kept: List[Polygon] = []
        for poly in sorted([p for p in parts if p.geom_type == "Polygon"
                            and p.area >= 25.0], key=lambda p: -p.area):
            if any(poly.distance(k) < gap for k in kept):
                continue
            kept.append(poly)
        polys = kept

    # 3) Drop the smallest masses until nothing is stranded. Smallest first, so
    # the street structure that carries the layout survives.
    from random_map import (_has_enclosed_free_pocket_grid,
                            _has_exit_disconnected_free_region_grid)

    for _ in range(12):
        if not polys or not exits:
            break
        blocked = unary_union(polys)
        stranded = _has_exit_disconnected_free_region_grid(
            W, H, blocked, list(exits), grid_step=2, pinch_cells=1)
        pocket = _has_enclosed_free_pocket_grid(
            W, H, blocked, grid_step=2, pinch_cells=1)
        if not (stranded or pocket):
            break
        polys.sort(key=lambda p: p.area)
        polys.pop(0)

    return [_ring_int(p) for p in polys]


def spec_from_difficulty(W: int, H: int, difficulty: int,
                         rng: Optional[random.Random] = None) -> StreetMapSpec:
    """Map the existing 0-6 difficulty scale onto street-network parameters.

    Difficulty here means the same thing it means for random_map: how hard the
    free space is to guide a crowd through. More coverage, smaller blocks and
    less regularity all make the route choice harder.
    """
    rng = rng or random
    d = max(0, min(6, int(difficulty)))
    t = d / 6.0

    if d == 0:
        # Tier 0 is the empty room, the same as random_map's. Coaxing the block
        # machinery toward it does not work: a single block the size of the map
        # still leaves a fifth of it built. Asking for zero coverage is honest
        # and generate_street_map short-circuits on it.
        return StreetMapSpec(width=W, height=H, coverage=0.0,
                             splits_per_block=(1, 1))

    # Tuned against the corpus rather than guessed. The first pass undershot
    # every channelisation target: streets came out at a 12 m median against a
    # real 17 m, open ground at 0.03 against a real 0.20, and 53 footprints
    # against a real 15 to 45. So streets are wider, blocks are split less,
    # coverage reaches further, and some blocks are left unbuilt, because a
    # real downtown crop contains squares and yards and not only frontage.
    return StreetMapSpec(
        width=W, height=H,
        regularity=_lerp(0.9, 0.15, t) + rng.uniform(-0.08, 0.08),
        block_scale=_lerp(0.42, 0.24, t) * rng.uniform(0.85, 1.15),
        coverage=_lerp(0.22, 0.50, t),
        street_width_min=_lerp(14.0, 9.0, t),
        street_width_max=_lerp(28.0, 16.0, t),
        # Two footprints in a block end up 2 x setback apart, so the floor
        # here is what guarantees the passability gap with margin.
        setback_m=_lerp(4.0, 2.0, t),
        n_orientations=rng.choice([1, 2, 2, 2, 0]),
        splits_per_block=(1, int(round(_lerp(1, 3, t)))),
        open_block_fraction=_lerp(0.30, 0.12, t),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--difficulty", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    from osm_corpus.stats import LABELS, layout_stats

    rng = random.Random(0)
    print(f"{'seed':>10s} {'obstacles':>9s} {'coverage':>9s} "
          f"{'width_p50':>10s} {'open_frac':>10s} {'components':>11s}")
    for i in range(args.n):
        spec = spec_from_difficulty(args.size, args.size, args.difficulty, rng)
        spec.seed = 1000 + i
        data = generate_street_map(spec)
        polys = [Polygon(r) for r in data.obstacles]
        st = layout_stats(polys, float(args.size))
        print(f"{spec.seed:10d} {st['n_obstacles']:9.0f} {st['coverage']:9.3f} "
              f"{st['width_p50']:10.1f} {st['open_fraction']:10.2f} "
              f"{st['free_components']:11.0f}")
