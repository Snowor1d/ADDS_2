"""Crowd size as a density, not a headcount.

Four places used to decide how many pedestrians a level holds, and they
disagreed: the curriculum drew 10 to 100 regardless of map size, real crops
scaled with crop area under a cap of 400, the held-out set was pinned at 30,
and the numbered maps used a constant. Expressed as density those are four
different crowds, so training and evaluation happened at different levels of
congestion and neither matched a real downtown.

Everything now derives from one number: persons per square metre of *walkable*
ground. Walkable rather than crop area because that is what the pedestrian
level-of-service literature is defined over, and because built coverage varies
by more than a factor of two across the corpus. Dividing by crop area would
make the same figure mean a quiet street in one city and a crush in another.

The density is drawn per level from a band rather than fixed, for the same
reason every other design variable is a band: a policy that only ever saw one
congestion level has no reason to generalise across congestion.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence


def walkable_area(width: float, height: float,
                  obstacles: Sequence[Sequence[Sequence[float]]]) -> float:
    """Area of the crop a pedestrian could stand on, in square metres.

    The obstacle union is clipped to the crop, so a footprint that overhangs
    the boundary is not subtracted twice or counted outside the map.
    """
    w = float(width)
    h = float(height)
    if not obstacles:
        return w * h

    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    polys = []
    for ring in obstacles:
        if len(ring) < 3:
            continue
        poly = Polygon([(float(x), float(y)) for x, y in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty:
            polys.append(poly)
    if not polys:
        return w * h

    free = box(0.0, 0.0, w, h).difference(unary_union(polys))
    return max(0.0, float(free.area))


def sample_density(rng: Optional[random.Random] = None) -> float:
    """Draw a target density for one level, persons per walkable square metre."""
    from config import CROWD_DENSITY_RANGE

    rng = rng or random
    lo, hi = CROWD_DENSITY_RANGE
    return rng.uniform(float(lo), float(hi))


def crowd_size_for(walkable_m2: float,
                   rng: Optional[random.Random] = None,
                   density: Optional[float] = None) -> int:
    """How many pedestrians a level with this much walkable ground holds.

    The clamp is a simulation-cost limit, not a modelling claim. Step cost
    grows faster than linearly in the crowd because a denser crowd also means
    more neighbours inside each pedestrian's interaction radius, so a 400 m
    crop at the target density would cost half an hour per episode. Where the
    clamp binds the level runs below the target density, which is why
    `realised_density` exists and why levels record what they actually got
    rather than what was asked for.
    """
    from config import CROWD_SIZE_LIMIT

    if density is None:
        density = sample_density(rng)
    n = int(round(float(density) * max(0.0, float(walkable_m2))))
    lo, hi = CROWD_SIZE_LIMIT
    return max(int(lo), min(int(hi), n))


def realised_density(crowd_size: int, walkable_m2: float) -> float:
    """The density a level actually runs at, after the cost clamp."""
    if walkable_m2 <= 0:
        return 0.0
    return float(crowd_size) / float(walkable_m2)


def crowd_size_for_level(width: float, height: float,
                         obstacles: Sequence[Sequence[Sequence[float]]],
                         rng: Optional[random.Random] = None,
                         density: Optional[float] = None):
    """Convenience for callers holding geometry: returns (n, realised density)."""
    area = walkable_area(width, height, obstacles)
    n = crowd_size_for(area, rng=rng, density=density)
    return n, realised_density(n, area)
