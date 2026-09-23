"""Where a pedestrian who knows nothing about the hazard is going.

Before this, an unaware pedestrian picked a walkable navmesh triangle
uniformly at random, walked to it, and picked another. Nothing supports that,
and it is not a harmless placeholder: the resulting distribution of people
across the crop is what decides how many are inside the hazard when it
starts, how far they have to be moved, and therefore how hard the episode is.
The difficulty of the task was resting on an artefact of the triangulation.

What people in a downtown actually do is travel between origins and
destinations along streets, and most of what crosses a two hundred metre crop
is passing through it. So trips run between street mouths, the places where
the walkable space meets the crop boundary, with the rest being local errands
to a point inside. Mouths are weighted by how wide they are, which is the
part a pedestrian count survey can calibrate: counts on a street scale with
its width and class, so width is the available proxy until counts are
attached to a site.

Two things this deliberately does not claim. It is not a gravity model or an
estimated OD matrix; there is no survey behind the weights yet. And the
weights are a property of the map, not of the curriculum, so the design space
is unchanged.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence


class Gate:
    """A street mouth: where walkable ground meets the crop boundary."""

    __slots__ = ("side", "meshes", "cx", "cy", "span")

    def __init__(self, side: str, meshes: list, cx: float, cy: float,
                 span: float):
        self.side = side
        self.meshes = meshes
        self.cx = cx
        self.cy = cy
        self.span = span

    def __repr__(self):
        return (f"Gate({self.side}, {len(self.meshes)} tri, "
                f"({self.cx:.1f},{self.cy:.1f}), {self.span:.1f} m)")


def _centroid(mesh):
    return ((mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0,
            (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0)


def _boundary_edges(model, eps: float = 0.6):
    """Navmesh edges that lie on the crop boundary, with their side and span.

    Taken from the triangle edges rather than from triangle centroids. The
    navmesh inserts points every NAVMESH_SEGMENT_STEP_M, so its triangles are
    tens of metres across and only a handful have a centroid near the
    boundary; the edges themselves are where the walkable space actually
    meets it. On a 120 m crop that is 26 edges covering 330 m of the 480 m
    perimeter, against 7 triangles by the centroid test.
    """
    W = float(model.width)
    H = float(model.height)
    lines = (("left", 0.0, 0), ("right", W, 0),
             ("bottom", 0.0, 1), ("top", H, 1))
    out = []
    for mesh in model.pure_mesh:
        for i in range(3):
            a = mesh[i]
            b = mesh[(i + 1) % 3]
            for side, val, ax in lines:
                if abs(a[ax] - val) < eps and abs(b[ax] - val) < eps:
                    lo = min(a[1 - ax], b[1 - ax])
                    hi = max(a[1 - ax], b[1 - ax])
                    if hi - lo > 1e-6:
                        out.append((side, lo, hi, mesh))
    return out


def gates(model) -> List[Gate]:
    """Street mouths on the crop boundary, cached per geometry.

    A run of boundary edges is one mouth; a gap wider than CROWD_GATE_GAP_M
    starts another, so two streets reaching the same side do not merge. The
    weight is the width of the mouth in metres, because pedestrian volume on
    a street scales with its width and class, and width is what the map gives
    us until counts are attached to a site.
    """
    from config import CROWD_GATE_GAP_M

    key = getattr(model, "obstacles_version", 0)
    cached = getattr(model, "_gate_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    by_side = {}
    for side, lo, hi, mesh in _boundary_edges(model):
        by_side.setdefault(side, []).append((lo, hi, mesh))

    out: List[Gate] = []
    gap = float(CROWD_GATE_GAP_M)
    for side, items in by_side.items():
        items.sort()
        run = [items[0]]
        run_hi = items[0][1]
        for lo, hi, mesh in items[1:]:
            if lo - run_hi > gap:
                out.append(_gate_from_run(model, side, run))
                run = [(lo, hi, mesh)]
                run_hi = hi
            else:
                run.append((lo, hi, mesh))
                run_hi = max(run_hi, hi)
        out.append(_gate_from_run(model, side, run))

    model._gate_cache = (key, out)
    return out


def _gate_from_run(model, side: str, run) -> Gate:
    W = float(model.width)
    H = float(model.height)
    lo = min(r[0] for r in run)
    hi = max(r[1] for r in run)
    span = max(1.0, hi - lo)
    mid = 0.5 * (lo + hi)
    # Placed just inside the boundary, so the destination is a point a
    # pedestrian can stand on rather than the wall itself.
    inset = 2.0
    if side == "left":
        cx, cy = inset, mid
    elif side == "right":
        cx, cy = W - inset, mid
    elif side == "bottom":
        cx, cy = mid, inset
    else:
        cx, cy = mid, H - inset
    meshes = []
    for _, _, mesh in run:
        if mesh not in meshes:
            meshes.append(mesh)
    return Gate(side, meshes, cx, cy, span)


def interior_meshes(model) -> list:
    """Walkable triangles that are not on the boundary, cached per geometry."""
    key = getattr(model, "obstacles_version", 0)
    cached = getattr(model, "_interior_mesh_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    edge = set(model._edge_meshes())
    out = [m for m in model.pure_mesh if m not in edge]
    model._interior_mesh_cache = (key, out)
    return out


def choose_destination(model, xy, rng: Optional[random.Random] = None):
    """A destination triangle for a pedestrian standing at `xy`.

    Through trips are drawn among street mouths, weighted by width and
    excluding any that is too close to be a trip at all. Local errands are
    drawn among interior triangles. Returns None only when the map has no
    walkable ground, which the validator rejects.
    """
    from config import CROWD_MIN_TRIP_M, CROWD_THROUGH_TRIP_SHARE

    rng = rng or random
    gs = gates(model)
    if gs and rng.random() < float(CROWD_THROUGH_TRIP_SHARE):
        far = [g for g in gs
               if math.hypot(xy[0] - g.cx, xy[1] - g.cy) > float(CROWD_MIN_TRIP_M)]
        pool = far or gs
        total = sum(g.span for g in pool)
        if total > 0:
            pick = rng.random() * total
            acc = 0.0
            for g in pool:
                acc += g.span
                if pick <= acc:
                    return g.meshes[rng.randrange(len(g.meshes))]
            return pool[-1].meshes[0]

    inner = interior_meshes(model)
    pool = inner or model.pure_mesh
    if not pool:
        return None
    return pool[rng.randrange(len(pool))]


def gate_mesh_set(model) -> set:
    """Every triangle belonging to a street mouth, cached per geometry."""
    key = getattr(model, "obstacles_version", 0)
    cached = getattr(model, "_gate_mesh_set_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    out = set()
    for g in gates(model):
        out.update(g.meshes)
    model._gate_mesh_set_cache = (key, out)
    return out


def is_gate_mesh(model, mesh) -> bool:
    """Whether this triangle is part of a street mouth."""
    if mesh is None:
        return False
    return mesh in gate_mesh_set(model)
