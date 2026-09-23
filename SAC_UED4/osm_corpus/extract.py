"""Stage 3: OSM rings to a cropped, metric building layout.

Turns (lon, lat) building rings into shapely polygons in metres, origin at the
crop's south-west corner, clipped to the crop square. The result is the same
shape as the simulator's own obstacle list, so downstream stages can treat a
real layout and a generated one identically.

Crops are quality-filtered rather than taken on trust. A named downtown can
still resolve onto a park, a plaza or a patch where OSM has no building data,
and a near-empty crop would quietly drag every corpus statistic toward the
generator's current behaviour instead of away from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from osm_corpus.fetch import BuildingRings, meters_per_degree

# A crop has to look like built-up fabric to be usable. These are deliberately
# loose: the corpus is supposed to describe the range of real layouts, so the
# filter rejects empty ground, not unusual ground.
MIN_BUILDINGS = 3
MIN_COVERAGE = 0.04
MAX_COVERAGE = 0.85
MIN_FOOTPRINT_AREA_M2 = 15.0

# OSM outlines carry digitising detail the simulator cannot use. A real 400 m
# downtown crop arrives with about 30 times the vertex count of a generated map
# of the same size, and the navmesh triangulation is what pays: 2000 triangles
# instead of 200, and a 90 second environment build.
#
# The tolerance is deliberately below the narrowest real gaps. Measured free
# width at the tenth percentile runs about 3 m in dense fabric, so simplifying
# at 1 m removes staircase detail without closing an alley the crowd uses.
SIMPLIFY_TOLERANCE_M = 1.0


@dataclass
class Crop:
    site_key: str
    size_m: float
    lat: float
    lon: float
    polygons: List[Polygon]
    source: str
    rejected: Optional[str] = None     # why it is unusable, or None
    # Set by the road-based path: the share of the crop a robot can occupy.
    # Not the same as 1 - coverage, which is the whole point of that path.
    traversable_fraction: Optional[float] = None
    n_road_ways: Optional[int] = None
    # Share of the blocked area carrying no building: how much of the place
    # OpenStreetMap does not record. Diagnostic and rejection criterion both.
    unbuilt_blocked: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.rejected is None

    def coverage(self) -> float:
        if not self.polygons:
            return 0.0
        return unary_union(self.polygons).area / (self.size_m ** 2)

    def rings_int(self) -> List[List[List[int]]]:
        """Integer coordinate rings, the form the simulator stores."""
        out = []
        for poly in self.polygons:
            coords = list(poly.exterior.coords)
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
            ring = [[int(round(x)), int(round(y))] for x, y in coords]
            # Rounding can collapse a sliver; drop what is no longer a polygon.
            if len(ring) >= 3 and Polygon(ring).area >= MIN_FOOTPRINT_AREA_M2:
                out.append(ring)
        return out


def project_rings(rings: Sequence[Sequence[Tuple[float, float]]],
                  lat: float, lon: float, size_m: float) -> List[Polygon]:
    """(lon, lat) degrees to metres within the crop, then clip to the square."""
    m_lat, m_lon = meters_per_degree(lat)
    half = size_m / 2.0
    world = box(0, 0, size_m, size_m)

    out: List[Polygon] = []
    for ring in rings:
        if len(ring) < 4:
            continue
        pts = [((p[0] - lon) * m_lon + half, (p[1] - lat) * m_lat + half)
               for p in ring]
        try:
            poly = Polygon(pts)
        except Exception:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        for cand in (poly.geoms if poly.geom_type == "MultiPolygon" else [poly]):
            if cand.geom_type != "Polygon":
                continue
            clipped = cand.intersection(world)
            if clipped.is_empty:
                continue
            for part in (clipped.geoms if clipped.geom_type == "MultiPolygon"
                         else [clipped]):
                if part.geom_type == "Polygon" and part.area >= MIN_FOOTPRINT_AREA_M2:
                    out.append(part)
    return out


def simplify_footprints(polygons: Sequence[Polygon],
                        tolerance: float = SIMPLIFY_TOLERANCE_M) -> List[Polygon]:
    """Drop digitising detail while keeping walls where they are.

    Applied after dissolving, so a whole terrace is simplified as one outline
    rather than each house separately, which is where most of the saving is.
    """
    if tolerance <= 0:
        return list(polygons)

    out: List[Polygon] = []
    for poly in polygons:
        simple = poly.simplify(tolerance, preserve_topology=True)
        if simple.is_empty or simple.geom_type not in ("Polygon", "MultiPolygon"):
            out.append(poly)
            continue
        if not simple.is_valid:
            simple = simple.buffer(0)
        parts = simple.geoms if simple.geom_type == "MultiPolygon" else [simple]
        kept = [p for p in parts
                if p.geom_type == "Polygon" and p.area >= MIN_FOOTPRINT_AREA_M2]
        # Simplifying must not delete a building; fall back if it did.
        out.extend(kept if kept else [poly])
    return out


def dissolve_touching(polygons: Sequence[Polygon]) -> List[Polygon]:
    """Merge footprints that share a wall into one obstacle.

    Terraced and perimeter-block buildings are separate OSM objects but one
    solid mass to walk around, and the simulator's obstacle list is about
    blocked space rather than about ownership. Leaving them separate also
    breaks the minimum-gap rules the generator's validators apply.
    """
    if not polygons:
        return []
    merged = unary_union(polygons)
    parts = merged.geoms if merged.geom_type == "MultiPolygon" else [merged]
    return [p for p in parts if p.geom_type == "Polygon" and p.area > 0]


# A road-built crop has to leave enough connected space to lead a crowd
# through. These are looser than they look: a crop with a tenth of its area as
# street is a real place, and the check is there to catch a centre that landed
# where OSM has no road network at all.
MIN_TRAVERSABLE = 0.05
MAX_TRAVERSABLE = 0.95

# Reject a crop when this much of what the road network calls blocked holds no
# building at all. See config.OSM_MAX_UNBUILT_BLOCKED.
#
# Set from the 400 m crops of all 29 sites, which fall into two groups with a
# wide gap between them. Three sit at 0.82 and above: palermo_soho 0.915,
# chandni_chowk 0.832, khan_el_khalili 0.823. The next is sultanahmet at
# 0.615, and the rest run down to times_square at 0.067. A threshold of 0.70
# sits in that gap and drops exactly those three.
#
# The surprise is which three. It is not the medinas: Marrakesh comes out at
# 0.495 and survives, while the worst crop in the corpus is Palermo Soho, a
# grid, where OpenStreetMap records buildings on 7 per cent of the ground.
# What this measures is how completely a country has been surveyed, not what
# kind of place it is.
#
# Judged once per site on its widest crop, not per crop. The ratio moves a
# great deal with crop size and not even in a consistent direction: Kreuzberg
# reads 0.849 at 100 m against 0.528 at 400 m, while Gastown goes the other
# way, 0.196 to 0.521. A small crop is a small sample where one block swings
# the figure; a large one reaches into whatever is unmapped at the edges.
# Applying a threshold fitted at 400 m to every size would cut single sizes
# out of otherwise sound sites, and the size axis would then mean something
# different at each site, which is exactly what the size-generalisation
# holdout is built to measure.
UNBUILT_BLOCKED_REFERENCE_SIZE = 400.0


def unbuilt_blocked_share(crop, lat: float, lon: float,
                          size_m: float) -> Optional[float]:
    """How much of this crop's blocked area carries no building.

    A direct measure of how much of the place OpenStreetMap does not record.
    The road network is the interpretation and it is applied the same way
    everywhere, so a crop the network describes badly is dropped rather than
    patched; this is what decides which those are.

    Buildings are fetched only for this test. They are not part of the layout
    and never have been under road-derived extraction.
    """
    from shapely.ops import unary_union

    from osm_corpus.fetch import fetch_site

    if not crop.polygons:
        return None
    try:
        blocks = unary_union([p for p in crop.polygons
                              if p.is_valid and not p.is_empty])
        if blocks.is_empty or blocks.area <= 0:
            return None
        b = fetch_site(crop.site_key, lat, lon, size_m)
        polys = project_rings(b.rings, lat, lon, size_m)
        polys = [p if p.is_valid else p.buffer(0) for p in polys]
        polys = [p for p in polys
                 if p.geom_type == "Polygon" and not p.is_empty]
        if not polys:
            return 1.0
        built = unary_union(polys)
        return float(1.0 - blocks.intersection(built).area / blocks.area)
    except Exception:
        # A missing or unreadable building source is not a reason to reject a
        # crop whose road network is fine; it just means this cannot be judged.
        return None


def make_crop(site_key: str, buildings: BuildingRings,
              lat: float, lon: float, size_m: float,
              dissolve: bool = True) -> Crop:
    polys = project_rings(buildings.rings, lat, lon, size_m)
    if dissolve:
        polys = dissolve_touching(polys)
    polys = simplify_footprints(polys)

    crop = Crop(site_key=site_key, size_m=size_m, lat=lat, lon=lon,
                polygons=polys, source=buildings.source)

    if len(polys) < MIN_BUILDINGS:
        crop.rejected = f"only {len(polys)} footprints (need {MIN_BUILDINGS})"
        return crop
    cov = crop.coverage()
    if cov < MIN_COVERAGE:
        crop.rejected = f"coverage {cov:.3f} below {MIN_COVERAGE} (park or missing data)"
    elif cov > MAX_COVERAGE:
        crop.rejected = f"coverage {cov:.3f} above {MAX_COVERAGE} (almost no free space)"
    return crop


def make_road_crop(site_key: str, region: Optional[str],
                   lat: float, lon: float, size_m: float,
                   simplify_m: Optional[float] = None) -> Crop:
    """A crop whose free space is the road network rather than the gaps.

    The obstacles come back as the complement of the traversable corridors, so
    a block is one polygon however many buildings stand on it. That is both the
    honest picture for a robot and far cheaper to simulate: a Covent Garden
    crop drops from hundreds of footprint vertices to about a hundred.
    """
    import os

    from config import OSM_OPEN_GROUND, OSM_SIMPLIFY_M
    from osm_corpus.fetch import extract_path, fetch_site
    from osm_corpus.roads import (fetch_roads_overpass, fetch_roads_pbf,
                                  obstacles_from_traversable,
                                  reachable_open_ground, traversable_polygon)

    simplify_m = OSM_SIMPLIFY_M if simplify_m is None else simplify_m

    if region and os.path.exists(extract_path(region)):
        net = fetch_roads_pbf(region, lat, lon, size_m)
    else:
        net = fetch_roads_overpass(lat, lon, size_m)

    traversable = traversable_polygon(net, size_m)

    # Unbuilt ground beside the streets counts as walkable. How much the road
    # network leaves out varies enormously by city: 83 per cent of what it
    # called blocked at Chandni Chowk holds no building, against 32 per cent
    # at Shibuya, because a wide bazaar is recorded there as one centreline.
    # Without this those places come out as mostly wall.
    if OSM_OPEN_GROUND:
        buildings = fetch_site(site_key, lat, lon, size_m)
        polys = project_rings(buildings.rings, lat, lon, size_m)
        polys = [p if p.is_valid else p.buffer(0) for p in polys]
        polys = [p for p in polys
                 if p.geom_type == "Polygon" and not p.is_empty]
        traversable = reachable_open_ground(traversable, polys, size_m)

    rings = obstacles_from_traversable(traversable, size_m, simplify_m)
    polys = [Polygon(r) for r in rings if len(r) >= 3]
    polys = [p if p.is_valid else p.buffer(0) for p in polys]
    polys = [p for p in polys if p.geom_type == "Polygon" and not p.is_empty]

    crop = Crop(site_key=site_key, size_m=size_m, lat=lat, lon=lon,
                polygons=polys, source=f"{net.source}:roads")
    crop.traversable_fraction = (traversable.area / (size_m ** 2)
                                 if traversable is not None else 0.0)
    crop.n_road_ways = len(net.lines)

    if len(net) == 0:
        crop.rejected = "no road network in the crop"
        return crop

    # The survey-completeness verdict is NOT taken here. It is a property of
    # the site, not of one crop, and is applied in collect.py once every size
    # has been cut. See UNBUILT_BLOCKED_REFERENCE_SIZE: the ratio moves a
    # great deal with crop size and not even in a consistent direction, so
    # judging each crop on its own cuts single sizes out of otherwise sound
    # sites and the size axis then means something different at each one.
    if crop.traversable_fraction < MIN_TRAVERSABLE:
        crop.rejected = (f"traversable {crop.traversable_fraction:.3f} below "
                         f"{MIN_TRAVERSABLE} (no mapped streets here)")
    elif crop.traversable_fraction > MAX_TRAVERSABLE:
        crop.rejected = (f"traversable {crop.traversable_fraction:.3f} above "
                         f"{MAX_TRAVERSABLE} (nothing to guide around)")
    return crop
