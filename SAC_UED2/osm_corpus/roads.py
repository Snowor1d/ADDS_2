"""Traversable space from the road network, not from the gaps between buildings.

Extracting buildings and calling everything else walkable overstates where a
guidance robot can actually go, and it overstates it badly. The space between
buildings in a real city is mostly not street: it is private plots, walled
yards, car parks, planting, water and rail. A crowd being led out of an area
moves along roads, footways and squares, so those are what the simulation's
free space has to be built from.

So the extraction is inverted. Traversable space is the road and path network
widened to its carriageway, plus the pedestrian areas and squares that are
mapped as polygons. Everything else in the crop is obstacle. That produces a
corridor network by construction, which is both what the task is about and
much cheaper to simulate: the obstacle set becomes a few dozen blocks instead
of hundreds of individual footprints.

OSM almost never tags a width in practice, so widths come from the highway
class, refined by `lanes` where it exists. The values are carriageway plus
footway, because the crowd uses both.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

# Total traversable width in metres per highway class: carriageway plus the
# footway either side, since a crowd uses the whole cross-section when leaving.
ROAD_WIDTHS: Dict[str, float] = {
    "primary": 16.0,
    "primary_link": 12.0,
    "secondary": 14.0,
    "secondary_link": 10.0,
    "tertiary": 11.0,
    "tertiary_link": 9.0,
    "residential": 9.0,
    "unclassified": 9.0,
    "living_street": 8.0,
    "pedestrian": 11.0,
    "service": 5.0,
    "footway": 3.5,
    "path": 3.0,
    "cycleway": 3.0,
    "track": 3.5,
    "corridor": 3.0,
}

# A wheeled guidance robot cannot use these, and a crowd being led should not
# be sent onto them, so they are not traversable however they are mapped.
EXCLUDED_CLASSES = frozenset({
    "steps", "motorway", "motorway_link", "trunk", "trunk_link",
    "raceway", "construction", "proposed", "planned", "escalator",
    "elevator", "bridleway", "via_ferrata",
})

# Metres of carriageway per lane, used when a way says how many lanes it has
# but not how wide it is.
METRES_PER_LANE = 3.2
FOOTWAY_ALLOWANCE_M = 3.0

DEFAULT_WIDTH_M = 6.0


@dataclass
class RoadNetwork:
    """Road centrelines with widths, plus any pedestrian areas, in metres."""
    lines: List[Tuple[LineString, float]] = field(default_factory=list)
    areas: List[Polygon] = field(default_factory=list)
    source: str = ""
    n_excluded: int = 0

    def __len__(self) -> int:
        return len(self.lines) + len(self.areas)


def width_for(tags) -> Optional[float]:
    """Traversable width for a way, or None if a robot cannot use it."""
    highway = tags.get("highway") or tags.get("area:highway")
    if highway is None or highway in EXCLUDED_CLASSES:
        return None

    # An explicit width is rare but authoritative when present.
    for key in ("width", "est_width"):
        raw = tags.get(key)
        if raw:
            try:
                return max(2.0, float(str(raw).split()[0]))
            except ValueError:
                pass

    base = ROAD_WIDTHS.get(highway, DEFAULT_WIDTH_M)

    lanes = tags.get("lanes")
    if lanes:
        try:
            n = max(1, int(float(str(lanes).split(";")[0])))
            from_lanes = n * METRES_PER_LANE
            if str(tags.get("sidewalk", "")).lower() not in ("no", "none", ""):
                from_lanes += FOOTWAY_ALLOWANCE_M
            base = max(base, from_lanes)
        except ValueError:
            pass
    return base


def _is_area(tags) -> bool:
    if "area:highway" in tags:
        return True
    return str(tags.get("area", "")).lower() in ("yes", "true", "1")


def fetch_roads_pbf(region: str, lat: float, lon: float, size_m: float) -> RoadNetwork:
    """Road ways from a regional extract, projected into crop metres."""
    import osmium

    from osm_corpus.fetch import extract_path, meters_per_degree

    path = extract_path(region)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no extract for {region}")

    from osm_corpus.fetch import bbox_for

    south, west, north, east = bbox_for(lat, lon, size_m, margin=1.25)
    m_lat, m_lon = meters_per_degree(lat)
    half = size_m / 2.0

    def to_local(lon_deg, lat_deg):
        return ((lon_deg - lon) * m_lon + half, (lat_deg - lat) * m_lat + half)

    net = RoadNetwork(source=f"pbf:{region}")

    class Handler(osmium.SimpleHandler):
        def way(self, w):
            tags = {k: v for k, v in w.tags}
            if "highway" not in tags and "area:highway" not in tags:
                return
            width = width_for(tags)
            if width is None:
                net.n_excluded += 1
                return
            pts, inside = [], False
            for node in w.nodes:
                try:
                    lon_deg, lat_deg = node.lon, node.lat
                except osmium.InvalidLocationError:
                    return
                pts.append(to_local(lon_deg, lat_deg))
                if west <= lon_deg <= east and south <= lat_deg <= north:
                    inside = True
            if not inside or len(pts) < 2:
                return
            if _is_area(tags) and len(pts) >= 4:
                poly = Polygon(pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.geom_type == "Polygon" and not poly.is_empty:
                    net.areas.append(poly)
                    return
            net.lines.append((LineString(pts), width))

    Handler().apply_file(path, locations=True, idx="flex_mem")
    return net


def fetch_roads_overpass(lat: float, lon: float, size_m: float) -> RoadNetwork:
    """Same thing through Overpass, for a site with no extract downloaded."""
    import requests

    from osm_corpus.fetch import USER_AGENT, bbox_for, meters_per_degree

    south, west, north, east = bbox_for(lat, lon, size_m, margin=1.25)
    query = (f'[out:json][timeout:120];'
             f'(way["highway"]({south},{west},{north},{east});'
             f'way["area:highway"]({south},{west},{north},{east}););'
             f'out geom;')
    r = requests.post("https://overpass-api.de/api/interpreter",
                      data={"data": query},
                      headers={"User-Agent": USER_AGENT}, timeout=180)
    r.raise_for_status()

    m_lat, m_lon = meters_per_degree(lat)
    half = size_m / 2.0
    net = RoadNetwork(source="overpass")

    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        width = width_for(tags)
        if width is None:
            net.n_excluded += 1
            continue
        geom = el.get("geometry") or []
        pts = [((p["lon"] - lon) * m_lon + half, (p["lat"] - lat) * m_lat + half)
               for p in geom if p.get("lon") is not None]
        if len(pts) < 2:
            continue
        if _is_area(tags) and len(pts) >= 4:
            poly = Polygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.geom_type == "Polygon" and not poly.is_empty:
                net.areas.append(poly)
                continue
        net.lines.append((LineString(pts), width))
    return net


# ---------------------------------------------------------------------------
# traversable space, and the obstacles that are its complement
# ---------------------------------------------------------------------------

# How far the block outlines may be moved when simplifying. Buffered road
# edges arrive as dense arcs, and the navmesh cost is driven by vertex count,
# so this is the main speed control. It is bounded by what a robot needs: move
# a wall by more than roughly its own width and a corridor that was passable
# can close.
SIMPLIFY_TOLERANCE_M = 1.5

# Rounded corners on a buffer cost vertices and buy nothing here, so the joins
# are mitred and the caps are flat.
BUFFER_JOIN_STYLE = 2
BUFFER_CAP_STYLE = 2

# Obstacle blocks below this are specks left over from the complement; they
# slow the navmesh down and add nothing a crowd would walk around.
MIN_BLOCK_AREA_M2 = 40.0

# Traversable slivers narrower than this cannot be used, and leaving them in
# creates pinch points the navmesh has to resolve. Opening the space by half
# this and closing it again removes them.
MIN_CORRIDOR_M = 2.5


def traversable_polygon(net: RoadNetwork, size_m: float,
                        widen: float = 1.0,
                        height_m: Optional[float] = None):
    """The space a robot and crowd can actually occupy.

    `height_m` defaults to `size_m`, so a square crop needs only one figure.
    The procedural generator shares this function, which is the point: a
    generated layout and a real one are then the same kind of object, built by
    buffering centrelines and taking the complement, rather than two different
    constructions that happen to be measured with the same statistics.
    """
    height_m = size_m if height_m is None else height_m
    world = box(0, 0, size_m, height_m)
    pieces = []

    for line, width in net.lines:
        if line.is_empty:
            continue
        pieces.append(line.buffer(max(1.0, width * widen) / 2.0,
                                  join_style=BUFFER_JOIN_STYLE,
                                  cap_style=BUFFER_CAP_STYLE))
    for area in net.areas:
        pieces.append(area)

    if not pieces:
        return None

    merged = unary_union(pieces).intersection(world)
    if merged.is_empty:
        return None

    # Close the hairline gaps where two buffered ways almost but not quite
    # meet, then reopen, so the network is one connected corridor system
    # rather than a set of nearly touching ribbons.
    closing = MIN_CORRIDOR_M / 2.0
    merged = merged.buffer(closing, join_style=BUFFER_JOIN_STYLE) \
                   .buffer(-closing, join_style=BUFFER_JOIN_STYLE) \
                   .intersection(world)
    return None if merged.is_empty else merged


def obstacles_from_traversable(traversable, size_m: float,
                               simplify_m: float = SIMPLIFY_TOLERANCE_M,
                               height_m: Optional[float] = None
                               ) -> List[List[List[int]]]:
    """Blocks, as the part of the crop the traversable network does not cover.

    Only the outer ring of each block is kept. A block with a hole means a
    piece of traversable space enclosed by it, which nothing can reach from
    the street, and filling it is what the building-based path already does.
    """
    height_m = size_m if height_m is None else height_m
    world = box(0, 0, size_m, height_m)
    if traversable is None:
        return [[[0, 0], [int(size_m), 0],
                 [int(size_m), int(height_m)], [0, int(height_m)]]]

    blocks = world.difference(traversable)
    if blocks.is_empty:
        return []

    parts = blocks.geoms if blocks.geom_type == "MultiPolygon" else [blocks]
    rings: List[List[List[int]]] = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area < MIN_BLOCK_AREA_M2:
            continue
        simple = part.simplify(simplify_m, preserve_topology=True)
        if simple.is_empty or simple.geom_type != "Polygon":
            simple = part
        if not simple.is_valid:
            simple = simple.buffer(0)
            if simple.geom_type != "Polygon" or simple.is_empty:
                continue
        if simple.area < MIN_BLOCK_AREA_M2:
            continue
        coords = list(simple.exterior.coords)[:-1]
        ring = [[int(round(x)), int(round(y))] for x, y in coords]
        if len(ring) < 3:
            continue
        if Polygon(ring).area < MIN_BLOCK_AREA_M2:
            continue
        rings.append(ring)
    return rings


def road_layout(net: RoadNetwork, size_m: float,
                simplify_m: float = SIMPLIFY_TOLERANCE_M,
                height_m: Optional[float] = None):
    """Obstacle rings plus the traversable polygon they were cut from."""
    traversable = traversable_polygon(net, size_m, height_m=height_m)
    rings = obstacles_from_traversable(traversable, size_m, simplify_m,
                                       height_m=height_m)
    return rings, traversable
