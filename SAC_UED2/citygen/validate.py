"""Is a plan playable, judged the way the robot experiences it.

The old scatter generator validated with gap rules between polygons: keep
obstacles a minimum distance apart, forbid wall pockets, check no free region
is cut off from an exit. Those rules exist because obstacles were placed
independently and could trap each other.

A street plan cannot go wrong that way. What it can do is produce a corridor
too narrow for the robot's body, or leave part of the network unreachable from
any exit, and neither is visible in a polygon-gap check. So validation here is
the robot's own question, asked on the rendered geometry: shrink the free
space by the robot's radius, and check that what remains is connected and
reaches every exit.

That formulation catches the failure that matters and ignores the ones that do
not. A corridor pinched to 2.5 m fails even though every polygon is far from
every other. A blind alley is fine, because a crowd can walk in and back out.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

# Free space is rasterised at this resolution for the reachability test. Half a
# metre resolves a corridor at the robot's scale without making the grid large
# enough to matter: a 400 m crop is 800 x 800 cells, which is milliseconds.
RASTER_STEP_M = 0.5

# A plan has to leave the crowd somewhere to be. Below this the crop is a wall
# with slots in it rather than a place.
MIN_FREE_FRACTION = 0.04

# Share of the robot-passable free space that must be reachable from an exit.
# Not all of it: a plan can legitimately contain a pocket the size of a single
# cell behind a rounded corner, and rejecting the whole level for that would
# throw away most mutations for no gain. But a plan where a tenth of the
# walkable ground cannot reach an exit is a plan that will spawn pedestrians
# who never evacuate, and the episode then runs to the step limit for a reason
# that has nothing to do with the policy.
MIN_REACHABLE_SHARE = 0.90


def passable_mask(plan, exits: Sequence[Sequence[Tuple[int, int]]],
                  step: float = RASTER_STEP_M):
    """Cells whose centre the robot's body can occupy, and the exit seeds.

    Exits are unioned into the free space before eroding. An exit is a hole in
    the boundary, so the ground inside it is walkable by definition even where
    a block was drawn across it, and the rendered obstacles do not know that.
    """
    import numpy as np
    from scipy.ndimage import binary_erosion
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    from config import ROBOT_BODY_RADIUS
    from osm_corpus.stats import rasterise

    rings, _ = plan.render()
    W, H = float(plan.width), float(plan.height)

    polys = [Polygon(r) for r in rings if len(r) >= 3]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if p.geom_type == "Polygon" and not p.is_empty]

    exit_polys = [Polygon(e) for e in exits if len(e) >= 3]
    if exit_polys:
        cut = unary_union(exit_polys)
        kept = []
        for p in polys:
            d = p.difference(cut)
            if d.is_empty:
                continue
            for part in (d.geoms if d.geom_type == "MultiPolygon" else [d]):
                if part.geom_type == "Polygon" and not part.is_empty:
                    kept.append(part)
        polys = kept

    n = max(1, int(round(max(W, H) / step)))
    free = ~rasterise(polys, max(W, H), step=step) if polys else \
        np.ones((n, n), dtype=bool)
    # rasterise works on a square of side max(W, H); trim to the real crop.
    ny, nx = int(round(H / step)), int(round(W / step))
    free = free[:ny, :nx]

    r_cells = max(1, int(round(ROBOT_BODY_RADIUS / step)))
    body = np.ones((2 * r_cells + 1, 2 * r_cells + 1), dtype=bool)
    fits = binary_erosion(free, structure=body, border_value=0)

    seeds = np.zeros_like(fits)
    if exit_polys:
        em = rasterise(exit_polys, max(W, H), step=step)[:ny, :nx]
        seeds = fits & em
    return fits, seeds, free


def check(plan, exits: Sequence[Sequence[Tuple[int, int]]]
          ) -> Tuple[bool, Optional[str]]:
    """(playable, reason it is not). Difficulty zero is playable by default."""
    import numpy as np
    from scipy.ndimage import label

    if not exits:
        return False, "no exits"

    rings, _ = plan.render()
    if not rings:
        # The empty field. Nothing to walk around, so nothing can go wrong.
        return True, None

    fits, seeds, free = passable_mask(plan, exits)
    total = int(fits.size)
    n_fits = int(fits.sum())
    if n_fits == 0:
        return False, "no corridor wide enough for the robot"
    if n_fits / total < MIN_FREE_FRACTION:
        return False, (f"robot-passable space is {n_fits / total:.3f} of the "
                       f"crop, below {MIN_FREE_FRACTION}")
    if not seeds.any():
        return False, "no exit opens onto space the robot can occupy"

    labels, _ = label(fits)
    reachable_ids = set(np.unique(labels[seeds])) - {0}
    reachable = int(np.isin(labels, list(reachable_ids)).sum())
    share = reachable / max(1, n_fits)
    if share < MIN_REACHABLE_SHARE:
        return False, (f"only {share:.2f} of passable space reaches an exit")
    return True, None


def is_playable(plan, exits) -> bool:
    ok, _ = check(plan, exits)
    return ok
