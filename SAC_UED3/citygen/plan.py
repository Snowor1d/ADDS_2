"""A level as a street plan, not as a bag of polygons.

This is the representation the curriculum edits. The earlier generators handed
ACCEL a list of obstacle polygons, and the mutation operators moved, rotated
and resized individual polygons. Those edits are morphology-blind: a block
translated off its frontage is no longer part of a street network, so a layout
that started out looking like a downtown stopped looking like one after a few
generations. Since offspring are the entire mechanism by which ACCEL builds
complexity, the progression from an empty field to a city has to be a property
of what a level *is*, not of how it was first drawn.

So a level carries the street network, and every mutation is an edit to that
network: insert a cross street, remove one, widen a corridor, build up a block
or clear it. The polygons are then derived, never edited. Whatever a lineage
does, the result is still a street network.

The derivation is deliberately the same code the real-map corpus uses. Street
centrelines are buffered to their carriageway width, that union is the
traversable space, and the obstacles are its complement inside the crop. A
generated block is therefore solid, exactly as a road-derived real block is
solid, and the generator cannot reintroduce the overstatement of navigable
space that motivated the road-based extraction in the first place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]


@dataclass
class Street:
    """One street, as a centreline polyline and a carriageway width.

    A polyline rather than a segment because organic fabric is not made of
    straight lines, and a medina lane that bends is the structure, not noise.
    """

    points: List[Point]
    width_m: float
    kind: str = "street"          # street | arterial | alley | stub

    def length(self) -> float:
        return sum(math.dist(self.points[i], self.points[i + 1])
                   for i in range(len(self.points) - 1))


# A block's built state is stored as a marker point inside it rather than as an
# index, because blocks are derived: they are the faces the street network cuts
# the crop into, and any edit to the network renumbers them. A point survives
# that. After an edit each derived block takes the state of whichever marker
# falls inside it, and a block with no marker inherits from the nearest one, so
# the two halves of a split block agree with their parent.
@dataclass
class BlockMark:
    x: float
    y: float
    built: bool


@dataclass
class CityPlan:
    """A street network plus which of its blocks are built up."""

    width: int
    height: int
    morphology: str
    development: float                      # 0 = empty field, 1 = full fabric
    streets: List[Street] = field(default_factory=list)
    marks: List[BlockMark] = field(default_factory=list)
    seed: Optional[int] = None

    # ---------------------------------------------------------------- geometry

    def _network(self):
        """The street centrelines as the corpus's RoadNetwork type."""
        from shapely.geometry import LineString

        from osm_corpus.roads import RoadNetwork

        lines = []
        for st in self.streets:
            if len(st.points) < 2:
                continue
            lines.append((LineString(st.points), float(st.width_m)))
        return RoadNetwork(lines=lines, areas=[], source="citygen",
                           n_excluded=0)

    def _street_space(self):
        """Traversable space contributed by the streets alone."""
        from osm_corpus.roads import traversable_polygon

        if not self.streets:
            return None
        return traversable_polygon(self._network(), float(self.width),
                                   height_m=float(self.height))

    def blocks(self) -> List:
        """The faces the street network cuts the crop into, largest first."""
        from shapely.geometry import box

        world = box(0, 0, self.width, self.height)
        street_space = self._street_space()
        if street_space is None:
            return [world]
        rest = world.difference(street_space)
        if rest.is_empty:
            return []
        parts = rest.geoms if rest.geom_type == "MultiPolygon" else [rest]
        out = [p for p in parts if p.geom_type == "Polygon" and not p.is_empty]
        out.sort(key=lambda p: -p.area)
        return out

    def _is_built(self, block) -> bool:
        """Whether a derived block is built, according to the marks."""
        from shapely.geometry import Point as _P

        if not self.marks:
            # No decisions recorded. A network with streets is a built city;
            # a crop with no streets at all is an empty field, and calling its
            # single face "built" would render the whole crop as one obstacle.
            return bool(self.streets)
        for m in self.marks:
            if block.covers(_P(m.x, m.y)):
                return m.built
        # No marker inside: this face did not exist when the marks were last
        # placed, so take the nearest decision rather than inventing one.
        cx, cy = block.representative_point().x, block.representative_point().y
        nearest = min(self.marks, key=lambda m: (m.x - cx) ** 2 + (m.y - cy) ** 2)
        return nearest.built

    def render(self, simplify_m: Optional[float] = None):
        """Obstacle rings plus the traversable polygon, as the corpus does it.

        Unbuilt blocks are traversable, not absent. A real downtown crop holds
        squares, yards and car parks, and at low development almost every block
        is one of those, which is what makes an empty field and a dense city
        the two ends of a single axis rather than two different generators.
        """
        from shapely.geometry import box
        from shapely.ops import unary_union

        from config import OSM_SIMPLIFY_M
        from osm_corpus.roads import obstacles_from_traversable

        simplify_m = OSM_SIMPLIFY_M if simplify_m is None else simplify_m
        world = box(0, 0, self.width, self.height)

        built = [b for b in self.blocks() if self._is_built(b)]
        if not built:
            # An empty field: nothing to walk around, the whole crop is free.
            return [], world

        traversable = world.difference(unary_union(built))
        if traversable.is_empty:
            traversable = None
        rings = obstacles_from_traversable(traversable, float(self.width),
                                           simplify_m,
                                           height_m=float(self.height))
        return rings, traversable

    # ------------------------------------------------------------- diagnostics

    def coverage(self) -> float:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        rings, _ = self.render()
        if not rings:
            return 0.0
        area = unary_union([Polygon(r) for r in rings]).area
        return float(area / (self.width * self.height))

    def summary(self) -> Dict:
        rings, trav = self.render()
        return dict(
            morphology=self.morphology,
            development=round(self.development, 3),
            n_streets=len(self.streets),
            n_blocks=len(rings),
            n_vertices=sum(len(r) for r in rings),
            street_width_median=(
                sorted(s.width_m for s in self.streets)[len(self.streets) // 2]
                if self.streets else 0.0),
            # A traversable polygon of None means nothing is walkable, which is
            # full coverage, not none of it.
            coverage=round(
                1.0 if trav is None else 1.0 - trav.area / (self.width * self.height),
                3),
        )

    # -------------------------------------------------------------- lineage

    def copy(self) -> "CityPlan":
        return CityPlan(
            width=self.width, height=self.height, morphology=self.morphology,
            development=self.development,
            streets=[Street(list(s.points), s.width_m, s.kind)
                     for s in self.streets],
            marks=[BlockMark(m.x, m.y, m.built) for m in self.marks],
            seed=self.seed,
        )

    def place_marks(self, built_fraction: float, rng) -> None:
        """Decide which blocks are built, one marker per block.

        Called once at generation. Which blocks are left open is not uniform
        chance: an unbuilt block beside an arterial reads as a square and one
        in the interior reads as a yard, and both appear in real crops, so the
        choice is by size, with the largest blocks kept open first. That puts
        the open ground where a real crop puts it and keeps the frontage built.
        """
        blocks = self.blocks()
        self.marks = []
        if not blocks:
            return
        n_open = int(round((1.0 - float(built_fraction)) * len(blocks)))
        # Size order with a little noise, so two levels at the same
        # development do not open exactly the same blocks.
        order = sorted(range(len(blocks)),
                       key=lambda i: -blocks[i].area * rng.uniform(0.75, 1.25))
        open_ids = set(order[:n_open])
        for i, b in enumerate(blocks):
            p = b.representative_point()
            self.marks.append(BlockMark(float(p.x), float(p.y),
                                        built=i not in open_ids))
