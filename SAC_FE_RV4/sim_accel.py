"""Vectorised map building for FightingModel.

The SAC_FE_RV3 constructor spent 10-13 s on a 100 m map and about 50 s on a
160-200 m map in pure-Python loops: a Floyd-Warshall over navmesh triangles,
a point-in-triangle test of every grid cell against every triangle, and a
ray-bisection visibility precompute with one shapely call per probe. A
curriculum that builds a fresh level every episode cannot afford that.

Every function here reproduces the original result exactly, not
approximately: the same floating-point operations in the same order and the
same shapely predicates. The original code is still in model.py and
visibility_atlas.py, selected by SIM_FAST_BUILD = False, and
tests/test_sim_equivalence.py builds maps both ways and compares every
structure and a stretch of simulation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Point, Polygon


# ----------------------------------------------------------------------------
# point-in-triangle over the integer grid
# ----------------------------------------------------------------------------

def _sign(p1x, p1y, p2, p3):
    # model.is_point_in_triangle.sign(p1, p2, p3), elementwise over p1.
    return (p1x - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1y - p3[1])


def _in_triangle_mask(px, py, v0, v1, v2):
    """model.is_point_in_triangle for arrays of points."""
    d1 = _sign(px, py, v0, v1)
    d2 = _sign(px, py, v1, v2)
    d3 = _sign(px, py, v2, v0)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _degenerate(v0, v1, v2) -> bool:
    """Exactly collinear vertices. Such a triangle passes the sign test for
    every point on its line, inside its bounding box or not, so it is tested
    against the whole grid rather than its box."""
    return ((v1[0] - v0[0]) * (v2[1] - v0[1])
            - (v2[0] - v0[0]) * (v1[1] - v0[1])) == 0


def grid_points_in_triangle(width: int, height: int, v0, v1, v2) -> List[List[int]]:
    """calculate_internal_coordinates_in_triangle: [x, y] grid points, in the
    original x-major order (for x: for y:)."""
    v0 = (float(v0[0]), float(v0[1]))
    v1 = (float(v1[0]), float(v1[1]))
    v2 = (float(v2[0]), float(v2[1]))
    if _degenerate(v0, v1, v2):
        x0, x1, y0, y1 = 0, width - 1, 0, height - 1
    else:
        # One cell of margin: a point outside the box by a whole cell cannot
        # pass the test of a non-degenerate triangle whatever the rounding.
        xs = (v0[0], v1[0], v2[0])
        ys = (v0[1], v1[1], v2[1])
        x0 = max(0, int(math.floor(min(xs))) - 1)
        x1 = min(width - 1, int(math.ceil(max(xs))) + 1)
        y0 = max(0, int(math.floor(min(ys))) - 1)
        y1 = min(height - 1, int(math.ceil(max(ys))) + 1)
    if x1 < x0 or y1 < y0:
        return []
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.int64),
                         np.arange(y0, y1 + 1, dtype=np.int64), indexing="ij")
    px = gx.ravel()
    py = gy.ravel()
    mask = _in_triangle_mask(px.astype(np.float64), py.astype(np.float64), v0, v1, v2)
    sel = np.nonzero(mask)[0]
    return [[int(px[k]), int(py[k])] for k in sel]


# ----------------------------------------------------------------------------
# navmesh graph and all-pairs shortest paths
# ----------------------------------------------------------------------------

def _centroid(m):
    return ((m[0][0] + m[1][0] + m[2][0]) / 3, (m[0][1] + m[1][1] + m[2][1]) / 3)


def build_navmesh_graph(model, triangles: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        width: int, height: int) -> None:
    """Everything mesh_map builds after triangulation, written into `model`.

    `triangles` are the (v0, v1, v2) rows of the triangulation in its own
    order. Fills mesh_list, mesh, match_grid_to_mesh, obstacle_mesh,
    distance, next_vertex_matrix, adjacent_mesh, pure_mesh,
    match_mesh_to_grid and valid_space with the values, types and orders
    the original produced.
    """
    mesh_list = model.mesh_list
    for v0, v1, v2 in triangles:
        vertices_tuple = tuple(sorted([tuple(v0), tuple(v1), tuple(v2)]))
        mesh_list.append(vertices_tuple)
        model.mesh.append(grid_points_in_triangle(width, height, v0, v1, v2))

    m2g = model.match_grid_to_mesh
    for mesh in mesh_list:
        for x, y in grid_points_in_triangle(width, height, mesh[0], mesh[1], mesh[2]):
            if (x, y) not in m2g:
                m2g[(x, y)] = (mesh[0], mesh[1], mesh[2])

    obstacle_polys = [Polygon(ob) for ob in model.obstacles]
    n = len(mesh_list)
    is_obst = np.zeros(n, dtype=bool)
    for idx, mesh in enumerate(mesh_list):
        mx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
        my = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
        p = Point(mx, my)
        if any(P.contains(p) or P.touches(p) for P in obstacle_polys):
            model.obstacle_mesh.append(mesh)
            is_obst[idx] = True

    # Adjacency: two triangles sharing at least two vertices (exact tuple
    # equality, as set(mesh1) & set(mesh2) compared them).
    by_vertex: Dict[tuple, List[int]] = {}
    for idx, mesh in enumerate(mesh_list):
        for v in set(mesh):
            by_vertex.setdefault(v, []).append(idx)
    shared: Dict[Tuple[int, int], int] = {}
    for members in by_vertex.values():
        for a in members:
            for b in members:
                if a != b:
                    shared[(a, b)] = shared.get((a, b), 0) + 1

    dist = np.full((n, n), math.inf, dtype=np.float64)
    nxt = np.full((n, n), -1, dtype=np.int64)
    idx_all = np.arange(n)
    dist[idx_all, idx_all] = 0.0
    nxt[idx_all, idx_all] = idx_all
    adjacent: Dict[int, List[int]] = {}
    for (a, b), cnt in shared.items():
        if cnt < 2 or is_obst[a] or is_obst[b]:
            continue
        c1 = _centroid(mesh_list[a])
        c2 = _centroid(mesh_list[b])
        dist[a, b] = math.sqrt(pow(c1[0] - c2[0], 2) + pow(c1[1] - c2[1], 2))
        nxt[a, b] = b
        adjacent.setdefault(a, []).append(b)

    # Floyd-Warshall in the original loop order: k over mesh_list, skipping
    # obstacle k and obstacle j. Row k and column k cannot change during
    # their own iteration (dist[k, k] = 0 and the test is strict), so the
    # vectorised update equals the in-place triple loop.
    free_cols = ~is_obst
    for k in range(n):
        if is_obst[k]:
            continue
        cand = dist[:, k:k + 1] + dist[k:k + 1, :]
        upd = cand < dist
        upd &= free_cols[None, :]
        if upd.any():
            dist = np.where(upd, cand, dist)
            nxt = np.where(upd, nxt[:, k:k + 1], nxt)

    model.next_vertex_matrix = {}
    for a, m1 in enumerate(mesh_list):
        drow = dist[a].tolist()
        nrow = nxt[a].tolist()
        model.distance[m1] = {m2: drow[b] for b, m2 in enumerate(mesh_list)}
        model.next_vertex_matrix[m1] = {
            m2: (mesh_list[nrow[b]] if nrow[b] >= 0 else None)
            for b, m2 in enumerate(mesh_list)}
    for a in range(n):
        if a in adjacent:
            model.adjacent_mesh[mesh_list[a]] = [mesh_list[b] for b in sorted(adjacent[a])]

    for idx, mesh in enumerate(mesh_list):
        if not is_obst[idx]:
            model.pure_mesh.append(mesh)

    # match_mesh_to_grid: every pure triangle's grid points, x-major. A
    # point on a shared edge belongs to every triangle it touches.
    for mesh in model.pure_mesh:
        pts = grid_points_in_triangle(width, height, mesh[0], mesh[1], mesh[2])
        if pts:
            model.match_mesh_to_grid[mesh] = pts

    vs = model.valid_space
    for i in range(-10, width + 10):
        for j in range(-10, height + 10):
            vs[(i, j)] = 1 if (0 <= i < width and 0 <= j < height) else 0


# ----------------------------------------------------------------------------
# static raster
# ----------------------------------------------------------------------------

def render_static_map(grid: np.ndarray, obstacles, exits, world_w: float,
                      world_h: float, H: int, W: int) -> None:
    """model._render_static_map, vectorised: obstacle pixels 50, exit
    pixels max(current, 100). Pixel centres are tested with the same
    strict containment as Polygon.contains(Point)."""

    def to_px(x, y):
        ix = int(np.clip(x / world_w * W, 0, W - 1))
        iy = int(np.clip(y / world_h * H, 0, H - 1))
        return ix, iy

    def pixels(poly):
        P = Polygon(poly)
        minx, miny, maxx, maxy = P.bounds
        gx0, gy0 = to_px(minx, miny)
        gx1, gy1 = to_px(maxx, maxy)
        ixs = np.arange(min(gx0, gx1), max(gx0, gx1) + 1)
        iys = np.arange(min(gy0, gy1), max(gy0, gy1) + 1)
        IX, IY = np.meshgrid(ixs, iys, indexing="ij")
        x = (IX + 0.5) * world_w / W
        y = (IY + 0.5) * world_h / H
        inside = shapely.contains_xy(P, x, y)
        return IX[inside], IY[inside]

    for poly in obstacles:
        ix, iy = pixels(poly)
        grid[iy, ix] = 50
    for epoly in exits:
        ix, iy = pixels(epoly)
        grid[iy, ix] = np.maximum(grid[iy, ix], 100)


# ----------------------------------------------------------------------------
# visibility precompute
# ----------------------------------------------------------------------------

def precompute_visibility(atlas, rays_per_poly: int = 64, bsearch_iters: int = 12) -> None:
    """VisibilityAtlas.precompute with every ray of a bisection iteration
    tested in one STRtree query. Probe coordinates are computed with the
    same scalar formulas, so each predicate sees the same segment."""
    from visibility_atlas import _ObstacleIndex

    if atlas._obs_index is None:
        atlas._obs_index = _ObstacleIndex([])
        atlas._obs_version = 0
        atlas._cache.clear()
    tree = atlas._obs_index._tree

    regions = [(i, j) for j in range(atlas.reg_rows) for i in range(atlas.reg_cols)]
    if not regions:
        return
    centers = [atlas.region_center(ij) for ij in regions]
    cx = np.array([c[0] for c in centers], dtype=np.float64)[:, None]
    cy = np.array([c[1] for c in centers], dtype=np.float64)[:, None]

    for R in atlas._radii:
        todo = [r for r, ij in enumerate(regions)
                if (ij[0], ij[1], float(R), atlas._obs_version) not in atlas._cache]
        if not todo:
            continue
        # R*cos(a) as a Python float, then added to the centre, as the
        # original's center[0] + R*math.cos(a).
        off_x = np.array([R * math.cos(2 * math.pi * (k / rays_per_poly))
                          for k in range(rays_per_poly)], dtype=np.float64)[None, :]
        off_y = np.array([R * math.sin(2 * math.pi * (k / rays_per_poly))
                          for k in range(rays_per_poly)], dtype=np.float64)[None, :]
        ccx = cx[todo]
        ccy = cy[todo]
        far_x = ccx + off_x
        far_y = ccy + off_y
        lo = np.zeros(far_x.shape)
        hi = np.ones(far_x.shape)
        hit_x = far_x.copy()
        hit_y = far_y.copy()
        if tree is not None:
            for _ in range(bsearch_iters):
                mid = 0.5 * (lo + hi)
                mx = ccx + (far_x - ccx) * mid
                my = ccy + (far_y - ccy) * mid
                coords = np.stack([
                    np.stack([np.broadcast_to(ccx, mx.shape), np.broadcast_to(ccy, my.shape)], -1),
                    np.stack([mx, my], -1)], axis=-2).reshape(-1, 2, 2)
                rays = shapely.linestrings(coords)
                hit_idx, _ = tree.query(rays, predicate="intersects")
                blocked = np.zeros(rays.shape[0], dtype=bool)
                blocked[hit_idx] = True
                # A zero-length probe is never blocked (it cannot occur:
                # mid >= 2**-bsearch_iters > 0 and R > 0).
                blocked &= shapely.length(rays) != 0
                blocked = blocked.reshape(mx.shape)
                hi = np.where(blocked, mid, hi)
                lo = np.where(blocked, lo, mid)
                hit_x = np.where(blocked, mx, hit_x)
                hit_y = np.where(blocked, my, hit_y)
        for row, r in enumerate(todo):
            i, j = regions[r]
            pts = list(zip(hit_x[row].tolist(), hit_y[row].tolist()))
            atlas._cache[(i, j, float(R), atlas._obs_version)] = Polygon(pts)
