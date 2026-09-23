"""Stage 7: real downtown crops as playable simulation levels.

Two uses, same conversion. As a zero-shot holdout these are the only levels in
the project that no generator produced, so they are the only honest test of
transfer to real layouts. As imitation maps they can also be trained on, to ask
how much of the gap is closed by simply showing the policy real fabric.

Exits are the researcher's choice, not a property of the map, so a site's exits
come from `sites.py` when given. When they are not given, exits are placed
automatically so the level is at least runnable, and every exported level
records which of the two happened. An automatic exit is a hole in the boundary
at a spot where the crop's free space already reaches the edge; it is not a
claim about where a real street leads.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from osm_corpus.collect import load_corpus
from osm_corpus.sites import by_key

EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "levels")

# Automatic exits. Width is a doorway at street scale rather than at building
# scale, and the depth matches what the generator produces so the simulator's
# exit handling sees nothing unusual.
AUTO_EXIT_WIDTH_M = 10.0
AUTO_EXIT_DEPTH_M = 5.0

# Exits are held one metre clear of the map edge. The simulator resolves a
# point to a navmesh triangle through a dictionary keyed by integer grid cells
# running 0..size-1, so a point at exactly x == size has no entry; an exit
# flush against the wall makes nearest_point_on_exit return such a point and
# the lookup raises. Flushness is a generator convention, not something the
# simulator needs, so the inset costs nothing.
EXIT_BOUNDARY_INSET_M = 1.0
AUTO_EXIT_COUNT = 2
AUTO_EXIT_MIN_SEPARATION_M = 0.35        # as a fraction of the perimeter


def _free_boundary_spans(rings: Sequence[Sequence[Sequence[int]]],
                         size_m: float, samples_per_side: int = 64):
    """Where the crop's free space reaches each edge.

    An exit has to open onto somewhere the crowd can actually stand, so the
    candidate positions are the stretches of boundary that are not covered by
    a building.
    """
    polys = [Polygon(r) for r in rings if len(r) >= 3]
    blocked = unary_union([p for p in polys if p.is_valid and not p.is_empty]) \
        if polys else None

    spans = []
    step = size_m / samples_per_side
    for side in ("left", "right", "bottom", "top"):
        for i in range(samples_per_side):
            t = (i + 0.5) * step
            if side == "left":
                pt = (0.5, t)
            elif side == "right":
                pt = (size_m - 0.5, t)
            elif side == "bottom":
                pt = (t, 0.5)
            else:
                pt = (t, size_m - 0.5)
            from shapely.geometry import Point
            if blocked is not None and blocked.contains(Point(*pt)):
                continue
            spans.append((side, t))
    return spans


def auto_exits(rings, size_m: float, count: int = AUTO_EXIT_COUNT,
               rng_seed: int = 0) -> List[List[Tuple[int, int]]]:
    """Place `count` exits on free stretches of the boundary, kept well apart."""
    import random as _random

    spans = _free_boundary_spans(rings, size_m)
    if not spans:
        return []

    rng = _random.Random(rng_seed)
    rng.shuffle(spans)
    half = AUTO_EXIT_WIDTH_M / 2.0
    depth = AUTO_EXIT_DEPTH_M
    min_sep = AUTO_EXIT_MIN_SEPARATION_M * 4 * size_m

    def perimeter_pos(side, t):
        # Distance around the boundary, so "far apart" means far apart in the
        # crowd's terms and not merely on different walls.
        return {"bottom": t, "right": size_m + t,
                "top": 2 * size_m + (size_m - t), "left": 3 * size_m + (size_m - t)}[side]

    chosen: List[Tuple[str, float]] = []
    for side, t in spans:
        if len(chosen) >= count:
            break
        p = perimeter_pos(side, t)
        if any(min(abs(p - perimeter_pos(s2, t2)),
                   4 * size_m - abs(p - perimeter_pos(s2, t2))) < min_sep
               for s2, t2 in chosen):
            continue
        chosen.append((side, t))

    # Relax the separation rather than return nothing at all.
    if len(chosen) < count:
        for side, t in spans:
            if len(chosen) >= count:
                break
            if (side, t) not in chosen:
                chosen.append((side, t))

    out = []
    inset = EXIT_BOUNDARY_INSET_M
    for side, t in chosen[:count]:
        lo, hi = max(inset, t - half), min(size_m - inset, t + half)
        if hi - lo < 2.0:
            continue
        if side == "left":
            poly = box(inset, lo, inset + depth, hi)
        elif side == "right":
            poly = box(size_m - inset - depth, lo, size_m - inset, hi)
        elif side == "bottom":
            poly = box(lo, inset, hi, inset + depth)
        else:
            poly = box(lo, size_m - inset - depth, hi, size_m - inset)
        coords = list(poly.exterior.coords)[:-1]
        out.append([(int(round(x)), int(round(y))) for x, y in coords])
    return out


# Raster fills come out as one-metre staircases. Left as they are, a filled
# courtyard becomes a 600-vertex outline, the navmesh triangulation of it
# produces degenerate slivers along the crop edge, and the simulator then fails
# resolving a sliver's centroid to a grid cell. Simplifying to roughly the
# robot's own diameter removes the staircase without moving any wall far enough
# to matter.
FILL_SIMPLIFY_TOLERANCE_M = 2.0
FILL_MIN_AREA_M2 = 12.0


def _mask_to_rings(mask, step: float = 1.0):
    """Raster cells to integer rings, merging each row's runs before union.

    Unioning a million unit squares is not viable at 1 km, so each row becomes
    a few rectangles first and only those are merged.
    """
    from shapely.ops import unary_union

    boxes = []
    n_rows, n_cols = mask.shape
    for y in range(n_rows):
        row = mask[y]
        x = 0
        while x < n_cols:
            if not row[x]:
                x += 1
                continue
            x0 = x
            while x < n_cols and row[x]:
                x += 1
            boxes.append(box(x0 * step, y * step, x * step, (y + 1) * step))
    if not boxes:
        return []
    merged = unary_union(boxes)
    parts = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    out = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area < FILL_MIN_AREA_M2:
            continue
        simple = part.simplify(FILL_SIMPLIFY_TOLERANCE_M, preserve_topology=True)
        if simple.is_empty or simple.geom_type != "Polygon":
            simple = part
        if not simple.is_valid:
            simple = simple.buffer(0)
            if simple.geom_type != "Polygon" or simple.is_empty:
                continue
        if simple.area < FILL_MIN_AREA_M2:
            continue
        coords = list(simple.exterior.coords)[:-1]
        ring = [[int(round(cx)), int(round(cy))] for cx, cy in coords]
        # Integer rounding can collapse a simplified sliver; drop what is no
        # longer a polygon rather than hand the triangulator a degenerate one.
        from shapely.geometry import Polygon as _P
        if len(ring) < 3:
            continue
        poly_int = _P(ring)
        if not poly_int.is_valid or poly_int.area < FILL_MIN_AREA_M2:
            continue
        out.append(ring)
    return out


def fill_unreachable(rings, exits, size_m: float, step: float = 1.0):
    """Turn free ground that cannot be reached from an exit into obstacle.

    Real footprints enclose courtyards. A Berlin perimeter block is entered
    through a passage cut into the building, which OSM records as building, so
    in a two-dimensional footprint model the courtyard is sealed. Left as free
    space it is somewhere the simulator will spawn pedestrians who can never
    evacuate, and the episode then runs to MAX_STEPS for a reason that has
    nothing to do with the policy.

    Filling it says the honest thing instead: ground the crowd cannot reach
    from the street is not usable space in this task. The amount filled is
    reported, because a crop where most of the free space is unreachable is a
    crop worth looking at rather than trusting.
    """
    import numpy as np
    from scipy.ndimage import label

    from osm_corpus.stats import rasterise

    polys = [Polygon(r) for r in rings if len(r) >= 3]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if p.geom_type == "Polygon" and not p.is_empty]

    occ = rasterise(polys, size_m, step=step)
    free = ~occ

    # Seed the flood fill from the cells the exits occupy.
    exit_mask = rasterise([Polygon(e) for e in exits], size_m, step=step)
    seed = free & exit_mask
    if not seed.any():
        return rings, 0.0

    labels, n = label(free)
    reachable_ids = set(np.unique(labels[seed])) - {0}
    unreachable = free & ~np.isin(labels, list(reachable_ids))

    filled_fraction = float(unreachable.sum()) / max(1, int(free.sum()))
    if not unreachable.any():
        return rings, 0.0

    return list(rings) + _mask_to_rings(unreachable, step=step), filled_fraction


def _subtract_exits(rings, exits, size_m):
    """Keep buildings from sitting inside an exit mouth."""
    if not exits:
        return rings
    exit_union = unary_union([Polygon(e) for e in exits])
    out = []
    for ring in rings:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        cut = poly.difference(exit_union)
        if cut.is_empty:
            continue
        for part in (cut.geoms if cut.geom_type == "MultiPolygon" else [cut]):
            if part.geom_type == "Polygon" and part.area >= 15.0:
                coords = list(part.exterior.coords)[:-1]
                out.append([[int(round(x)), int(round(y))] for x, y in coords])
    return out


# How far in front of an exit must be kept clear. An exit needs an approach:
# in a real crop a building can sit right up against the boundary beside the
# opening, which leaves no navmesh in front of it, and the simulator then fails
# looking for a mesh triangle at a point on the map edge.
EXIT_APRON_DEPTH_M = 10.0


def _clear_exit_apron(rings, exits, size_m: float,
                      depth: float = EXIT_APRON_DEPTH_M):
    """Subtract an approach corridor in front of each exit from the obstacles.

    Real footprints frequently abut the crop boundary next to where an exit is
    placed, so the free space at the opening is a sliver or absent, the
    navmesh does not cover it, and nearest_point_on_exit returns a point on the
    map edge that no triangle contains. Clearing an apron is the physical
    reading of the same requirement: an exit the crowd cannot walk up to is
    not an exit.
    """
    if not exits:
        return rings

    aprons = []
    for ring in exits:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        eps = 1e-6
        near = EXIT_BOUNDARY_INSET_M + 1e-6
        if x0 <= near:                      # left wall
            aprons.append(box(0, y0, min(size_m, x1 + depth), y1))
        elif x1 >= size_m - near:           # right wall
            aprons.append(box(max(0.0, x0 - depth), y0, size_m, y1))
        elif y0 <= near:                    # bottom wall
            aprons.append(box(x0, 0, x1, min(size_m, y1 + depth)))
        else:                               # top wall
            aprons.append(box(x0, max(0.0, y0 - depth), x1, size_m))

    apron_union = unary_union(aprons)
    out = []
    for ring in rings:
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        cut = poly.difference(apron_union)
        if cut.is_empty:
            continue
        for part in (cut.geoms if cut.geom_type == "MultiPolygon" else [cut]):
            if part.geom_type == "Polygon" and part.area >= 15.0:
                coords = list(part.exterior.coords)[:-1]
                out.append([[int(round(x)), int(round(y))] for x, y in coords])
    return out


def export_row(row: dict, crowd_size: Optional[int] = None) -> Optional[dict]:
    """One corpus crop as a level payload."""
    if row.get("rejected") is not None or not row.get("rings"):
        return None

    size_m = float(row["size_m"])
    site = by_key(row["site"])

    if site.exits:
        exits = [[(int(round(x)), int(round(y))) for x, y in ring]
                 for ring in site.exits]
        exit_source = "sites.py"
    else:
        exits = auto_exits(row["rings"], size_m,
                           rng_seed=abs(hash((row["site"], row["size_m"]))) % (1 << 30))
        exit_source = "auto"

    rings = _subtract_exits(row["rings"], exits, size_m)
    rings = _clear_exit_apron(rings, exits, size_m)
    if not rings or not exits:
        return None

    # Courtyards sealed by real footprints have to be filled before the level
    # is usable; see fill_unreachable. Done after the apron is cleared, so the
    # apron counts as reachable ground.
    rings, filled = fill_unreachable(rings, exits, size_m)

    # Crowd scales with area so density, not headcount, is held constant
    # across crop sizes; otherwise a 1 km crop is a search problem and a 100 m
    # crop is a congestion problem.
    if crowd_size is None:
        per_hectare = 30.0 / ((100.0 * 100.0) / 10000.0)
        crowd_size = int(round(per_hectare * (size_m * size_m) / 10000.0))
        crowd_size = max(10, min(400, crowd_size))

    return {
        "site": row["site"],
        "city": row["city"],
        "country": row["country"],
        "continent": row["continent"],
        "morphology": row["morphology"],
        "lat": row["lat"],
        "lon": row["lon"],
        "source": row["source"],
        "traversability": row.get("traversability", "buildings"),
        "traversable_fraction": row.get("traversable"),
        "width": int(size_m),
        "height": int(size_m),
        "obstacles": rings,
        "exits": [[list(p) for p in ring] for ring in exits],
        "exit_source": exit_source,
        "unreachable_filled_fraction": round(float(filled), 4),
        "crowd_size": crowd_size,
        "stats": row.get("stats"),
    }


def export_all(crop_sizes: Optional[Sequence[int]] = None,
               out_dir: str = EXPORT_DIR) -> dict:
    corpus = load_corpus()
    os.makedirs(out_dir, exist_ok=True)

    levels, skipped = [], []
    for row in corpus["rows"]:
        if crop_sizes and int(row["size_m"]) not in set(int(s) for s in crop_sizes):
            continue
        payload = export_row(row)
        if payload is None:
            skipped.append((row["site"], row["size_m"],
                            row.get("rejected") or "no exits placed"))
            continue
        levels.append(payload)

    index = {
        "_about": ("Playable levels built from real downtown crops. Map "
                   "data from OpenStreetMap, (c) OpenStreetMap contributors, "
                   "ODbL 1.0. Exits marked exit_source=auto were placed by "
                   "osm_corpus.export so the level runs, and are not a claim "
                   "about where real streets lead."),
        "traversability": corpus.get("traversability", "buildings"),
        "simplify_m": corpus.get("simplify_m"),
        "n_levels": len(levels),
        "levels": levels,
    }
    path = os.path.join(out_dir, "real_levels.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    os.replace(tmp, path)

    print(f"exported {len(levels)} levels -> {path}")
    if skipped:
        print(f"skipped {len(skipped)}:")
        for site, size, why in skipped[:20]:
            print(f"  {site:24s} {size:5} m  {why}")
    return index


def load_levels(path: Optional[str] = None) -> List:
    """Real-map levels as `ued.level.Level` objects."""
    from ued.level import Level

    path = path or os.path.join(EXPORT_DIR, "real_levels.json")
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    out = []
    for d in payload["levels"]:
        lv = Level(
            obstacles=[[list(p) for p in ring] for ring in d["obstacles"]],
            exits=[[tuple(p) for p in ring] for ring in d["exits"]],
            crowd_size=int(d["crowd_size"]),
            width=int(d["width"]),
            height=int(d["height"]),
            augmentation="identity",
            generator="osm",
            difficulty=None,
        )
        lv.site_key = d["site"]
        lv.morphology = d["morphology"]
        lv.exit_source = d["exit_source"]
        out.append(lv)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    args = ap.parse_args()
    export_all(crop_sizes=args.sizes)
