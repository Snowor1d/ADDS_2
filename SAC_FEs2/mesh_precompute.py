# mesh_precompute.py
from __future__ import annotations
import math
import random
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import triangle as tr

from model import (
    generate_segments_with_points,
    _dedup_pslg,
    calculate_internal_coordinates_in_triangle,
    are_meshes_adjacent,
)

def build_mesh_artifacts(width: int, height: int, obstacles: list, *, D: int = 200):
    """
    returns dict with:
      - mesh_list
      - next_vertex_matrix
      - distance
      - pure_mesh
      - valid_space
      - match_grid_to_mesh   (grid (ix,iy) -> mesh triple)
      - match_mesh_to_grid   (mesh -> list of [i,j] grid cells)
    """
    # ----- sanitize + union (네 _sanitize_obstacles + union_obstacles_to_polygons 흐름) -----
    polys = []
    for ob in obstacles:
        if len(ob) < 3:
            continue
        P = Polygon(ob)
        if not P.is_valid:
            P = P.buffer(0)
        if P.is_empty or P.area < 1e-6:
            continue
        polys.append(P)
    if polys:
        U = unary_union(polys)
        if isinstance(U, Polygon):
            polys = [U]
        else:
            polys = list(getattr(U, "geoms", []))
    else:
        polys = []

    new_obs = []
    for P in polys:
        coords = list(P.exterior.coords)[:-1]
        coords = [[round(float(x), 6), round(float(y), 6)] for x, y in coords]
        cleaned = []
        for c in coords:
            if not cleaned or cleaned[-1] != c:
                cleaned.append(c)
        if len(cleaned) >= 3:
            new_obs.append(cleaned)
    obstacles = new_obs

    # ----- PSLG 구성 -----
    map_boundary = [[0, 0], [width, 0], [width, height], [0, height]]
    obstacle_hulls = [np.array(ob, dtype=float) for ob in obstacles]

    vertices = map_boundary.copy()
    for hull in obstacle_hulls:
        vertices.extend(hull.tolist())

    segments = [[i, (i + 1) % 4] for i in range(4)]
    offset = 4
    for hull in obstacle_hulls:
        n = len(hull)
        segments.extend([[i + offset, (i + 1) % n + offset] for i in range(n)])
        offset += n

    vertices2, segments2 = generate_segments_with_points(vertices, segments, max(D // 4, 10))
    vertices2, segments2 = _dedup_pslg(vertices2, segments2)
    tri_in = {"vertices": np.array(vertices2), "segments": np.array(segments2)}

    t = tr.triangulate(tri_in, "p")

    mesh_list = []
    match_grid_to_mesh = {}

    for tri in t["triangles"]:
        v0, v1, v2 = t["vertices"][tri[0]], t["vertices"][tri[1]], t["vertices"][tri[2]]
        mesh = tuple(sorted([tuple(v0), tuple(v1), tuple(v2)]))
        mesh_list.append(mesh)

        internal = calculate_internal_coordinates_in_triangle(width, height, v0, v1, v2, D)
        for (ix, iy) in internal:
            if (ix, iy) not in match_grid_to_mesh:
                match_grid_to_mesh[(ix, iy)] = (mesh[0], mesh[1], mesh[2])

    # match_mesh_to_grid: mesh -> list of [i, j] grid cells (inverse of match_grid_to_mesh)
    match_mesh_to_grid = {}
    for (ix, iy), m in match_grid_to_mesh.items():
        match_mesh_to_grid.setdefault(m, []).append([ix, iy])

    # ----- obstacle mesh 판정 -----
    obstacle_polys = [Polygon(ob) for ob in obstacles]
    obstacle_mesh = set()
    for mesh in mesh_list:
        mx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
        my = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
        p = Point(mx, my)
        if any(P.contains(p) or P.touches(p) for P in obstacle_polys):
            obstacle_mesh.add(mesh)

    # ----- floyd-warshall용 distance / next -----
    distance = {a: {} for a in mesh_list}
    next_vertex_matrix = {a: {b: None for b in mesh_list} for a in mesh_list}

    for i, m1 in enumerate(mesh_list):
        for j, m2 in enumerate(mesh_list):
            if i == j:
                distance[m1][m2] = 0.0
                next_vertex_matrix[m1][m2] = m1
            elif (m1 in obstacle_mesh) or (m2 in obstacle_mesh):
                distance[m1][m2] = math.inf
                next_vertex_matrix[m1][m2] = None
            elif are_meshes_adjacent(m1, m2):
                c1 = ((m1[0][0] + m1[1][0] + m1[2][0]) / 3.0, (m1[0][1] + m1[1][1] + m1[2][1]) / 3.0)
                c2 = ((m2[0][0] + m2[1][0] + m2[2][0]) / 3.0, (m2[0][1] + m2[1][1] + m2[2][1]) / 3.0)
                d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                distance[m1][m2] = d
                next_vertex_matrix[m1][m2] = m2
            else:
                distance[m1][m2] = math.inf
                next_vertex_matrix[m1][m2] = None

    # floyd-warshall
    for k in mesh_list:
        if k in obstacle_mesh:
            continue
        for i in mesh_list:
            if i in obstacle_mesh:
                continue
            dik = distance[i][k]
            if dik == math.inf:
                continue
            for j in mesh_list:
                if j in obstacle_mesh:
                    continue
                alt = dik + distance[k][j]
                if alt < distance[i][j]:
                    distance[i][j] = alt
                    next_vertex_matrix[i][j] = next_vertex_matrix[i][k]

    pure_mesh = [m for m in mesh_list if m not in obstacle_mesh]

    # valid_space (너 코드 그대로)
    valid_space = {}
    for x in range(-10, width + 10):
        for y in range(-10, height + 10):
            valid_space[(x, y)] = 1 if (0 <= x < width and 0 <= y < height) else 0

    return {
        "mesh_list": mesh_list,
        "next_vertex_matrix": next_vertex_matrix,
        "distance": distance,
        "pure_mesh": pure_mesh,
        "valid_space": valid_space,
        "match_grid_to_mesh": match_grid_to_mesh,
        "match_mesh_to_grid": match_mesh_to_grid,
    }
