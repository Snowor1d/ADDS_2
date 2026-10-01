"""Stage 4: structural statistics for a layout.

The same measurements are taken on real crops and on generated maps, because
the point is the gap between them. Each one was chosen because it separates
"open ground with obstacles scattered on it", which is what the current
generator makes, from "a street network between building masses", which is
what a real downtown is.

The free-space width statistics are the load-bearing ones, and they are also
the easiest to misread: an empty corner of a crop, whether a park or a hole in
OSM coverage, reports a huge distance to the nearest building and drags the
upper percentiles with it. So the width distribution is reported with
percentiles rather than a mean, and `open_fraction` is reported beside it to
say how much of the crop is nowhere near anything.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Distance from a building beyond which ground counts as simply open rather
# than as a street. Wider than any real street cross-section.
OPEN_GROUND_M = 40.0


def rasterise(polygons: Sequence[Polygon], size_m: float, step: float = 1.0) -> np.ndarray:
    """Boolean occupancy grid at `step` metres per cell."""
    n = max(1, int(round(size_m / step)))
    ys, xs = np.mgrid[0:n, 0:n]
    pts = np.column_stack([(xs.ravel() + 0.5) * step, (ys.ravel() + 0.5) * step])
    occupied = np.zeros(pts.shape[0], dtype=bool)
    for poly in polygons:
        if poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        # Only test cells the bounding box can contain; the containment test is
        # the expensive part and a footprint covers a small slice of the crop.
        cand = ((pts[:, 0] >= minx - step) & (pts[:, 0] <= maxx + step) &
                (pts[:, 1] >= miny - step) & (pts[:, 1] <= maxy + step))
        if not cand.any():
            continue
        ring = np.asarray(poly.exterior.coords)
        hit = MplPath(ring).contains_points(pts[cand])
        idx = np.flatnonzero(cand)[hit]
        occupied[idx] = True
    return occupied.reshape(n, n)


def layout_stats(polygons: Sequence[Polygon], size_m: float,
                 step: float = 1.0) -> Optional[Dict[str, float]]:
    polys = [p for p in polygons if p.is_valid and not p.is_empty and p.area > 0]
    if not polys:
        return None

    area = float(size_m) ** 2
    merged = unary_union(polys)
    footprints = np.array([p.area for p in polys], dtype=float)

    occ = rasterise(polys, size_m, step)
    free = ~occ
    if free.sum() == 0:
        return None

    # Twice the distance to the nearest building is the local width of the
    # free space, which is the street cross-section where there is a street.
    dist_m = distance_transform_edt(free) * step
    widths = 2.0 * dist_m[free]

    # Connected free components. Kept for inspection but NOT a targeting
    # statistic: on a raw crop the count is dominated by interior courtyards
    # sealed by the footprints rather than by street topology, and once those
    # are filled the count rises instead of falling, because simplified fill
    # outlines leave one-cell seams between neighbours. Channelisation is
    # measured by width_p50 and open_fraction, which are stable.
    n_components = int(label(free)[1])

    # Vertex count is the simulation-cost statistic. Navmesh triangulation,
    # the visibility atlas and the all-pairs path table all scale with it, so
    # this is what simplification is tuned against.
    n_vertices = sum(len(p.exterior.coords) - 1 for p in polys)

    return dict(
        n_obstacles=float(len(polys)),
        n_vertices=float(n_vertices),
        coverage=float(merged.area / area),
        footprint_median=float(np.median(footprints)),
        footprint_cv=float(footprints.std() / footprints.mean()) if footprints.mean() else 0.0,
        footprint_max_frac=float(footprints.max() / area),
        width_p10=float(np.percentile(widths, 10)),
        width_p50=float(np.percentile(widths, 50)),
        width_p90=float(np.percentile(widths, 90)),
        open_fraction=float((widths > OPEN_GROUND_M).mean()),
        free_components=float(n_components),
    )


FIELDS = ("n_obstacles", "n_vertices", "coverage", "footprint_median", "footprint_cv",
          "footprint_max_frac", "width_p10", "width_p50", "width_p90",
          "open_fraction", "free_components")

LABELS = {
    "n_obstacles": "obstacles",
    "n_vertices": "obstacle vertices (simulation cost)",
    "coverage": "blocked coverage",
    "footprint_median": "median footprint m2",
    "footprint_cv": "footprint size CV",
    "footprint_max_frac": "largest footprint / area",
    "width_p10": "free width p10 m",
    "width_p50": "free width p50 m",
    "width_p90": "free width p90 m",
    "open_fraction": f"fraction >{OPEN_GROUND_M:.0f} m from an obstacle",
    "free_components": "free-space components (diagnostic only)",
}


def summarise(rows: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Per-field median and 10th/90th percentile across a set of layouts."""
    out: Dict[str, Dict[str, float]] = {}
    for f in FIELDS:
        vals = np.array([r[f] for r in rows if r is not None and f in r], dtype=float)
        if vals.size == 0:
            continue
        out[f] = dict(p10=float(np.percentile(vals, 10)),
                      p50=float(np.percentile(vals, 50)),
                      p90=float(np.percentile(vals, 90)))
    return out
