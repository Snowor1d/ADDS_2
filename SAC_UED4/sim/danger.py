"""The danger zone: the thing the crowd has to get out of, and stay out of.

This replaces the exits. Under the old task a pedestrian was safe once it
reached a hole in the map boundary, so safety was a small set of places and
the crowd converged on them. Here safety is everywhere except one region, so
the crowd diverges instead, and the two produce quite different problems: the
old one is a queueing problem at a few doors, this one is a dispersal problem
with a perimeter.

Two shapes, because they behave differently and both occur. A circle is a
release from a point, a gas leak or an explosion, and its escape directions
are uniform. A rectangle is a stretch of street or a building footprint, and
escaping it sideways can be many times shorter than escaping it lengthwise,
which is the whole difficulty when the crowd starts at the far end.

Everything here is straight-line geometry. Escape distance through a city is
not straight-line, because blocks are in the way, and the geodesic version
lives in the simulator where the navmesh is. The straight-line figures are
still needed: they are what the observation shows the robot, and they are the
lower bound the free-flow estimate is built from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# A zone smaller than this is not a hazard, it is a traffic cone. Below roughly
# the robot's own turning space there is nothing to guide anyone around.
MIN_RADIUS_M = 6.0

# And one larger than this share of the crop leaves nowhere to escape to, so
# the episode measures the map rather than the policy.
MAX_AREA_FRACTION = 0.45


@dataclass
class DangerZone:
    """A circle or an axis-aligned rectangle, in world metres."""

    shape: str                 # "circle" | "rect"
    cx: float
    cy: float
    # Circle: `radius`. Rectangle: half-extents, so `contains` is symmetric in
    # both and a rectangle's centre means the same thing as a circle's.
    radius: float = 0.0
    half_w: float = 0.0
    half_h: float = 0.0
    # Rectangle only: rotation in radians, anticlockwise about the centre.
    #
    # A hazard in a city usually runs along something. A gas main follows the
    # street it was laid under, a crash closes a carriageway, a fire spreads
    # down a terrace, and all three produce a long region lying at whatever
    # angle that street happens to run. An axis-aligned box can only express
    # that when the street happens to run north or east, and the generator
    # rotates its whole street grid by a random angle, so on most levels it
    # could not express it at all.
    angle: float = 0.0

    # ------------------------------------------------------------ rotation

    def _to_local(self, x: float, y: float) -> Tuple[float, float]:
        """A world point in the zone's own frame, centred and unrotated."""
        dx, dy = x - self.cx, y - self.cy
        if not self.angle:
            return dx, dy
        c, s = math.cos(-self.angle), math.sin(-self.angle)
        return dx * c - dy * s, dx * s + dy * c

    def _to_world(self, lx: float, ly: float) -> Tuple[float, float]:
        if not self.angle:
            return self.cx + lx, self.cy + ly
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.cx + lx * c - ly * s, self.cy + lx * s + ly * c

    # ---------------------------------------------------------------- queries

    def contains(self, x: float, y: float) -> bool:
        if self.shape == "circle":
            return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.radius ** 2
        lx, ly = self._to_local(x, y)
        return abs(lx) <= self.half_w and abs(ly) <= self.half_h

    def signed_distance(self, x: float, y: float) -> float:
        """Negative inside, positive outside, zero on the boundary.

        Signed rather than clamped because the robot needs both halves. How
        deep a pedestrian is inside says how urgent it is, and how far outside
        one is says how much margin there is before it wanders back in, which
        is the half the inflow-blocking part of the task depends on.
        """
        if self.shape == "circle":
            return math.hypot(x - self.cx, y - self.cy) - self.radius
        # Standard box distance, in the zone's own frame so a rotated
        # rectangle is still a rectangle: outside is the length of the
        # positive part, inside is the negative of the distance to the
        # nearest face.
        lx, ly = self._to_local(x, y)
        ox = abs(lx) - self.half_w
        oy = abs(ly) - self.half_h
        if ox > 0.0 or oy > 0.0:
            return math.hypot(max(ox, 0.0), max(oy, 0.0))
        return max(ox, oy)

    def escape_distance(self, x: float, y: float) -> float:
        """Straight-line metres to safety. Zero for anyone already out."""
        return max(0.0, -self.signed_distance(x, y))

    def nearest_safe_point(self, x: float, y: float,
                           margin: float = 1.0) -> Tuple[float, float]:
        """The closest point outside, with a little clearance past the edge.

        The margin matters. Aiming exactly at the boundary leaves a pedestrian
        standing on it, where `contains` is true and one social-force jostle
        puts it back inside, so the episode never finishes for a reason that
        has nothing to do with the guidance.
        """
        dx, dy = x - self.cx, y - self.cy
        if self.shape == "circle":
            d = math.hypot(dx, dy)
            if d < 1e-9:
                # Dead centre has no nearest direction; any one will do.
                return self.cx + self.radius + margin, self.cy
            k = (self.radius + margin) / d
            return self.cx + dx * k, self.cy + dy * k

        # Rectangle: leave by the nearest face, decided in the zone's frame
        # and mapped back, so the way out of a rotated hazard is across it
        # rather than along whichever world axis happens to be closer.
        lx, ly = self._to_local(x, y)
        out_x = self.half_w - abs(lx)
        out_y = self.half_h - abs(ly)
        if out_x <= out_y:
            sign = 1.0 if lx >= 0 else -1.0
            return self._to_world(sign * (self.half_w + margin), ly)
        sign = 1.0 if ly >= 0 else -1.0
        return self._to_world(lx, sign * (self.half_h + margin))

    # ------------------------------------------------------------- geometry

    def area(self) -> float:
        if self.shape == "circle":
            return math.pi * self.radius ** 2
        return 4.0 * self.half_w * self.half_h

    def perimeter(self) -> float:
        """Total boundary length. The analogue of exit width.

        The old free-flow estimate divided the crowd by the width of the exits
        it had to pass through. The boundary of a zone plays the same role,
        except that most of it is usually available at once, which is why a
        dispersal takes so much less queueing than an evacuation through doors.
        """
        if self.shape == "circle":
            return 2.0 * math.pi * self.radius
        return 4.0 * (self.half_w + self.half_h)

    def polygon(self, resolution: int = 48):
        """Shapely polygon, for intersection tests against the fabric."""
        from shapely.geometry import Point, box

        if self.shape == "circle":
            return Point(self.cx, self.cy).buffer(self.radius,
                                                  resolution=resolution // 4)
        rect = box(-self.half_w, -self.half_h, self.half_w, self.half_h)
        if self.angle:
            from shapely.affinity import rotate as _rotate
            rect = _rotate(rect, self.angle, origin=(0, 0), use_radians=True)
        from shapely.affinity import translate as _translate
        return _translate(rect, self.cx, self.cy)

    def bounds(self) -> Tuple[float, float, float, float]:
        if self.shape == "circle":
            return (self.cx - self.radius, self.cy - self.radius,
                    self.cx + self.radius, self.cy + self.radius)
        if not self.angle:
            return (self.cx - self.half_w, self.cy - self.half_h,
                    self.cx + self.half_w, self.cy + self.half_h)
        # A rotated rectangle's axis-aligned extent, which is what a raster
        # loop needs to know where to look.
        c, s = abs(math.cos(self.angle)), abs(math.sin(self.angle))
        ex = self.half_w * c + self.half_h * s
        ey = self.half_w * s + self.half_h * c
        return (self.cx - ex, self.cy - ey, self.cx + ex, self.cy + ey)

    def max_escape_distance(self) -> float:
        """The worst case from inside: the deepest point's distance out."""
        if self.shape == "circle":
            return self.radius
        return min(self.half_w, self.half_h)

    # ------------------------------------------------------------ conversion

    def to_dict(self) -> dict:
        return dict(shape=self.shape, cx=self.cx, cy=self.cy,
                    radius=self.radius, half_w=self.half_w,
                    half_h=self.half_h, angle=self.angle)

    @staticmethod
    def from_dict(d: dict) -> "DangerZone":
        return DangerZone(shape=d["shape"], cx=float(d["cx"]),
                          cy=float(d["cy"]), radius=float(d.get("radius", 0.0)),
                          half_w=float(d.get("half_w", 0.0)),
                          half_h=float(d.get("half_h", 0.0)),
                          angle=float(d.get("angle", 0.0)))

    def transformed(self, transform: str, width: float,
                    height: float) -> "DangerZone":
        """The same zone under one of the map's D4 symmetries.

        The simulator rotates and reflects a level's buildings per episode
        (sim/map_augmentation.py) but used to leave the hazard where it was,
        so on seven of the eight orientations the hazard sat over different
        streets from the ones the generator validated it against. `width` and
        `height` are the source dimensions, as for transform_point.
        """
        from sim.map_augmentation import transform_point

        if transform == "identity":
            return self
        cx, cy = transform_point((self.cx, self.cy), width, height, transform)
        angle = self.angle
        if self.shape != "circle":
            ex, ey = transform_point((self.cx + math.cos(self.angle),
                                      self.cy + math.sin(self.angle)),
                                     width, height, transform)
            angle = math.atan2(ey - cy, ex - cx)
        return DangerZone(shape=self.shape, cx=cx, cy=cy, radius=self.radius,
                          half_w=self.half_w, half_h=self.half_h, angle=angle)

    def scaled_to(self, width: float, height: float) -> "DangerZone":
        """The same zone on a resized crop, kept inside the new bounds.

        Map size is a curriculum axis and the canvas operator moves the crop
        edge, so a zone can end up outside the world. Clamping the centre
        rather than rescaling the radius keeps the hazard the same physical
        size, which is the point: a gas leak does not get bigger because the
        map did.
        """
        if self.shape == "circle":
            r = self.radius
        else:
            # The rotated extent, so a long hazard at an angle is kept fully
            # inside rather than half over the edge.
            x0, y0, x1, y1 = self.bounds()
            r = max(x1 - x0, y1 - y0) / 2.0
        return DangerZone(
            shape=self.shape,
            cx=min(max(self.cx, r), max(r, width - r)),
            cy=min(max(self.cy, r), max(r, height - r)),
            radius=self.radius, half_w=self.half_w, half_h=self.half_h,
            angle=self.angle)


def sample_along_street(rng, width: float, height: float,
                        area_fraction: float, streets,
                        cross_margin: float = 4.0) -> Optional["DangerZone"]:
    """A hazard lying along one of the plan's streets.

    Most real hazards in a city run along something rather than sitting on a
    patch of ground: a gas main follows the street it was laid under, a crash
    closes a carriageway, a fire spreads down a terrace. The resulting region
    is long, thin, and at whatever angle that street happens to run.

    That shape is a different problem for the robots, and the shape is the
    whole of what it buys. Escaping a circle is uniform in every direction;
    escaping across a corridor is quick while escaping along it is not, so
    where the crowd happens to be inside decides how hard the episode is, and
    a robot posted at one end of a corridor is doing something quite unlike a
    robot standing beside a circle.

    It does not put the hazard on better ground. Measured over twelve grid
    plans at a 0.15 area fraction, 0.64 of a street-aligned hazard lies on
    walkable ground against 0.61 for an unaligned rectangle, which is the same
    number. An earlier version of this docstring claimed otherwise; it had not
    been measured.

    The width across comes from the street's own carriageway plus a margin,
    not from the requested area. Deriving it from the area instead was the
    first attempt and it was wrong in a way worth recording: at a 0.15 area
    fraction on a 140 m map it produced a corridor 27 m across, wider than any
    street in the plan, so the hazard spilled onto the blocks either side and
    only 0.59 of it lay on ground anyone could stand on, against 0.81 for an
    unaligned rectangle. Following the street means following its width.

    The length then follows from the area, capped by the segment it lies on.
    So a street-aligned hazard is usually smaller than the requested fraction,
    which is simply what a corridor hazard is: a fire in one street is not the
    same size as a gas cloud over a quarter.

    Returns None when the plan has no street long enough to carry one, so the
    caller can fall back to an unaligned shape.
    """
    target = max(1e-4, min(MAX_AREA_FRACTION, float(area_fraction))) \
        * width * height

    usable = []
    for st in streets or ():
        pts = list(getattr(st, "points", ()))
        w = float(getattr(st, "width_m", 0.0))
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            seg = math.hypot(x1 - x0, y1 - y0)
            if seg < 3.0 * MIN_RADIUS_M:
                continue
            usable.append(((x0, y0), (x1, y1), seg, w))
    if not usable:
        return None

    (x0, y0), (x1, y1), seg, street_w = usable[rng.randrange(len(usable))]

    # Across: the carriageway, widened a little so the hazard reaches the
    # frontages rather than stopping at the kerb.
    half_h = max(MIN_RADIUS_M / 2.0, (street_w + cross_margin) / 2.0)
    # Along: whatever length reaches the requested area, but never longer than
    # the segment it lies on.
    half_w = min(seg / 2.0, max(MIN_RADIUS_M, target / (4.0 * half_h)))

    angle = math.atan2(y1 - y0, x1 - x0)
    span = half_w / seg
    t = rng.uniform(span, 1.0 - span) if seg > 2 * half_w else 0.5
    cx = x0 + (x1 - x0) * t
    cy = y0 + (y1 - y0) * t

    zone = DangerZone("rect", cx, cy, half_w=half_w, half_h=half_h,
                      angle=angle)

    # Shrink along the street until the whole thing fits inside the crop.
    #
    # A street segment is drawn across a span the length of the crop's
    # diagonal, so half the segment can be longer than the map: on a 100 m
    # crop this produced a hazard reaching from y = 0 to y = 141.7, of which
    # only 1118 of its 1581 square metres lay inside the world. The area
    # fraction the curriculum asked for was then not what the level had, and
    # the part outside was hazard the crowd could neither be in nor escape.
    for _ in range(12):
        x0, y0, x1, y1 = zone.bounds()
        over = max(-x0, -y0, x1 - width, y1 - height, 0.0)
        if over <= 0.5:
            break
        zone.half_w = max(MIN_RADIUS_M, zone.half_w - over)
        zone = DangerZone("rect", zone.cx, zone.cy, half_w=zone.half_w,
                          half_h=zone.half_h, angle=zone.angle)
    return zone.scaled_to(width, height)


def sample_zone(rng, width: float, height: float,
                area_fraction: float,
                shape: Optional[str] = None) -> DangerZone:
    """A zone covering `area_fraction` of the crop, placed at random.

    Area rather than radius is the parameter because it is what compares
    across shapes and across map sizes: a 20 m circle is a large hazard in a
    70 m crop and a small one in a 200 m crop, and the curriculum needs the
    difficulty to mean the same thing at both.
    """
    frac = max(1e-4, min(MAX_AREA_FRACTION, float(area_fraction)))
    target = frac * width * height
    shape = shape or rng.choice(("circle", "rect"))
    if shape == "street":
        # Handled by sample_along_street, which needs the street network.
        # Falling back rather than failing keeps a caller that asked for it on
        # a level that has none, a real OSM crop for instance, working.
        shape = "rect"

    if shape == "circle":
        radius = max(MIN_RADIUS_M, math.sqrt(target / math.pi))
        cx = rng.uniform(radius, max(radius, width - radius))
        cy = rng.uniform(radius, max(radius, height - radius))
        return DangerZone("circle", cx, cy, radius=radius)

    # Rectangle: draw an aspect ratio, then the extents that hit the area.
    # Elongated rectangles are the interesting case, because escaping across
    # one is quick and escaping along it is not.
    aspect = math.exp(rng.uniform(-1.0, 1.0))       # 0.37 to 2.7
    half_w = max(MIN_RADIUS_M, math.sqrt(target * aspect) / 2.0)
    half_h = max(MIN_RADIUS_M, (target / (4.0 * half_w)))
    cx = rng.uniform(half_w, max(half_w, width - half_w))
    cy = rng.uniform(half_h, max(half_h, height - half_h))
    return DangerZone("rect", cx, cy, half_w=half_w, half_h=half_h)
