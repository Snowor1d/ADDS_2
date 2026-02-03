# build_environment.py
import os
import pickle
import numpy as np
import math
import random

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree

import triangle as tr

# ===== model.py에서 가져온 유틸들(그대로 복붙 추천) =====
def _dedup_pslg(vertices, segments, ndigits=6):
    key_to_new = {}
    new_vertices = []
    old_to_new = {}

    for i, (x, y) in enumerate(vertices):
        k = (round(float(x), ndigits), round(float(y), ndigits))
        if k not in key_to_new:
            key_to_new[k] = len(new_vertices)
            new_vertices.append([k[0], k[1]])
        old_to_new[i] = key_to_new[k]

    seen = set()
    new_segments = []
    for a, b in segments:
        ia = old_to_new[int(a)]
        ib = old_to_new[int(b)]
        if ia == ib:
            continue
        e = (ia, ib) if ia < ib else (ib, ia)
        if e in seen:
            continue
        seen.add(e)
        new_segments.append([ia, ib])

    return new_vertices, new_segments

def add_intermediate_points(p1, p2, D):
    dist = np.linalg.norm(np.array(p2) - np.array(p1))
    if dist > D:
        num_points = int(dist // D) + 1
        return np.linspace(p1, p2, num=num_points+1, endpoint=False)[1:].tolist()
    return []

def generate_segments_with_points(vertices, segments, D):
    new_vertices = vertices.copy()
    new_segments = []
    for seg in segments:
        p1 = vertices[seg[0]]
        p2 = vertices[seg[1]]
        new_points = add_intermediate_points(p1, p2, D)
        last_index = seg[0]
        for point in new_points:
            new_vertices.append(point)
            new_index = len(new_vertices) - 1
            new_segments.append([last_index, new_index])
            last_index = new_index
        new_segments.append([last_index, seg[1]])
    return new_vertices, new_segments

def are_meshes_adjacent(mesh1, mesh2):
    common_vertices = set(mesh1) & set(mesh2)
    return len(common_vertices) >= 2

def is_point_in_triangle(p, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])
    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)

def calculate_internal_coordinates_in_triangle(width, height, v0, v1, v2, D):
    pts = []
    for x in range(width):
        for y in range(height):
            if is_point_in_triangle([x, y], v0, v1, v2):
                pts.append([x, y])
    return pts

def union_obstacles_to_polygons(raw_obstacles, min_area=1e-8):
    parts = []
    for ob in raw_obstacles:
        if ob is None or len(ob) < 3:
            continue
        P = Polygon(ob)
        if not P.is_valid:
            P = P.buffer(0)
        if P.is_empty:
            continue
        if isinstance(P, MultiPolygon):
            parts.extend(list(P.geoms))
        else:
            parts.append(P)

    if not parts:
        return [], []

    U = unary_union(parts)
    polys = []
    if isinstance(U, Polygon):
        polys = [U]
    elif isinstance(U, MultiPolygon):
        polys = list(U.geoms)
    else:
        try:
            polys = [g for g in U.geoms if isinstance(g, Polygon)]
        except Exception:
            polys = []

    outer_polys = []
    hole_rings = []
    for P in polys:
        if P.area <= min_area:
            continue
        ext = list(P.exterior.coords)[:-1]
        if len(ext) >= 3:
            outer_polys.append([(float(x), float(y)) for x, y in ext])
        for ring in P.interiors:
            h = list(ring.coords)[:-1]
            if len(h) >= 3:
                hole_rings.append([(float(x), float(y)) for x, y in h])
    return outer_polys, hole_rings

# ===== 전처리 Builder =====
class EnvBuilder:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.obstacles = []
        self.exit_list = []

        self.mesh_list = []
        self.mesh = []
        self.match_grid_to_mesh = {}
        self.match_mesh_to_grid = {}
        self.obstacle_mesh = []
        self.pure_mesh = []

        self.adjacent_mesh = {}
        self.distance = {}
        self.next_vertex_matrix = {}

        self.valid_space = {}
        self.blocked = np.zeros((height, width), dtype=bool)
        self.obstacles_grid_points = []  # 원하면 construct_map를 여기서도 만들 수 있음

    def load_map_from_file(self, map_num: int, base_dir: str = "map_infos"):
        import json
        fname = f"map_{map_num}.json"
        fpath = os.path.join(os.getcwd(), base_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"[map] not found: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            obj = json.load(f)

        obstacles_raw = obj.get("obstacles", []) or []
        obstacles = []
        for poly in obstacles_raw:
            if not poly or len(poly) < 3:
                continue
            norm = [[int(pt[0]), int(pt[1])] for pt in poly]
            obstacles.append(norm)
        self.obstacles = obstacles

        exits_raw = obj.get("exits", []) or []
        exits = []
        for poly in exits_raw:
            if not poly or len(poly) < 3:
                continue
            norm = [(int(pt[0]), int(pt[1])) for pt in poly]
            exits.append(norm)
        self.exit_list = exits

    def _sanitize_obstacles(self, eps=1e-6):
        polys = []
        for ob in self.obstacles:
            if len(ob) < 3:
                continue
            P = Polygon(ob)
            if not P.is_valid:
                P = P.buffer(0)
            if P.is_empty or P.area < eps:
                continue
            polys.append(P)

        if not polys:
            self.obstacles = []
            return

        U = unary_union(polys)
        if isinstance(U, Polygon):
            out = [U]
        elif isinstance(U, MultiPolygon):
            out = list(U.geoms)
        else:
            out = [g for g in getattr(U, "geoms", []) if isinstance(g, Polygon)]

        new_obs = []
        for P in out:
            coords = list(P.exterior.coords)[:-1]
            coords = [[round(x, 6), round(y, 6)] for x, y in coords]
            cleaned = []
            for c in coords:
                if not cleaned or cleaned[-1] != c:
                    cleaned.append(c)
            if len(cleaned) >= 3:
                new_obs.append(cleaned)
        self.obstacles = new_obs

    def mesh_map(self, D=20):
        # 1) obstacle sanitize + union
        self._sanitize_obstacles()
        outer_polys, _ = union_obstacles_to_polygons(self.obstacles)
        self.obstacles = [[list(p) for p in poly] for poly in outer_polys]

        map_boundary = [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        obstacle_hulls = [np.array(ob, dtype=float) for ob in self.obstacles]

        # 2) PSLG vertices/segments
        vertices = map_boundary.copy()
        for hull_points in obstacle_hulls:
            vertices.extend(hull_points.tolist())

        segments = [[i, (i + 1) % 4] for i in range(4)]
        offset = 4
        for hull_points in obstacle_hulls:
            n = len(hull_points)
            segments.extend([[i + offset, (i + 1) % n + offset] for i in range(n)])
            offset += n

        vertices2, seg2 = generate_segments_with_points(vertices, segments, D)
        vertices2, seg2 = _dedup_pslg(vertices2, seg2)

        tdata = {'vertices': np.array(vertices2), 'segments': np.array(seg2)}
        t = tr.triangulate(tdata, 'p')

        self.mesh_list.clear()
        self.mesh.clear()
        self.match_grid_to_mesh.clear()
        self.match_mesh_to_grid.clear()
        self.obstacle_mesh.clear()
        self.pure_mesh.clear()
        self.adjacent_mesh.clear()
        self.distance.clear()
        self.next_vertex_matrix.clear()

        # 3) triangle list
        for tri in t['triangles']:
            v0, v1, v2 = t['vertices'][tri[0]], t['vertices'][tri[1]], t['vertices'][tri[2]]
            tri_tuple = tuple(sorted([tuple(v0), tuple(v1), tuple(v2)]))
            self.mesh_list.append(tri_tuple)

            internal = calculate_internal_coordinates_in_triangle(self.width, self.height, v0, v1, v2, D)
            self.mesh.append(internal)

        # 4) grid->mesh (대표 tri 할당)
        for mesh in self.mesh_list:
            internal = calculate_internal_coordinates_in_triangle(self.width, self.height, mesh[0], mesh[1], mesh[2], D)
            for i in internal:
                k = (i[0], i[1])
                if k not in self.match_grid_to_mesh:
                    self.match_grid_to_mesh[k] = (mesh[0], mesh[1], mesh[2])

        # 5) obstacle tri 판정
        obstacle_polys = [Polygon(ob) for ob in self.obstacles]
        for mesh in self.mesh_list:
            mx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
            my = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
            p = Point(mx, my)
            if any(P.contains(p) or P.touches(p) for P in obstacle_polys):
                self.obstacle_mesh.append(mesh)

        # 6) floyd-warshall 준비
        self.next_vertex_matrix = {s: {e: None for e in self.mesh_list} for s in self.mesh_list}
        for i, mesh1 in enumerate(self.mesh_list):
            self.distance[mesh1] = {}
            for j, mesh2 in enumerate(self.mesh_list):
                if i == j:
                    self.distance[mesh1][mesh2] = 0.0
                    self.next_vertex_matrix[mesh1][mesh2] = mesh1
                elif (mesh1 in self.obstacle_mesh or mesh2 in self.obstacle_mesh):
                    self.distance[mesh1][mesh2] = math.inf
                    self.next_vertex_matrix[mesh1][mesh2] = None
                elif are_meshes_adjacent(mesh1, mesh2):
                    c1 = ((mesh1[0][0]+mesh1[1][0]+mesh1[2][0])/3.0,
                          (mesh1[0][1]+mesh1[1][1]+mesh1[2][1])/3.0)
                    c2 = ((mesh2[0][0]+mesh2[1][0]+mesh2[2][0])/3.0,
                          (mesh2[0][1]+mesh2[1][1]+mesh2[2][1])/3.0)
                    dist = math.hypot(c1[0]-c2[0], c1[1]-c2[1])
                    self.distance[mesh1][mesh2] = dist
                    self.next_vertex_matrix[mesh1][mesh2] = mesh2

                    self.adjacent_mesh.setdefault(mesh1, []).append(mesh2)
                else:
                    self.distance[mesh1][mesh2] = math.inf
                    self.next_vertex_matrix[mesh1][mesh2] = None

        # 7) floyd-warshall (여기가 제일 무거움 → 오프라인으로 빼는게 핵심)
        for k in self.mesh_list:
            if k in self.obstacle_mesh:
                continue
            for i in self.mesh_list:
                if i in self.obstacle_mesh:
                    continue
                dik = self.distance[i][k]
                if not np.isfinite(dik):
                    continue
                for j in self.mesh_list:
                    if j in self.obstacle_mesh:
                        continue
                    alt = dik + self.distance[k][j]
                    if alt < self.distance[i][j]:
                        self.distance[i][j] = alt
                        self.next_vertex_matrix[i][j] = self.next_vertex_matrix[i][k]

        for mesh in self.mesh_list:
            if mesh not in self.obstacle_mesh:
                self.pure_mesh.append(mesh)

        # 8) mesh->grid 매핑(원하면 저장)
        for i in range(self.width):
            for j in range(self.height):
                for mesh in self.pure_mesh:
                    if is_point_in_triangle([i, j], mesh[0], mesh[1], mesh[2]):
                        self.match_mesh_to_grid.setdefault(mesh, []).append([i, j])

        # 9) valid_space
        for i in range(-10, self.width + 10):
            for j in range(-10, self.height + 10):
                self.valid_space[(i, j)] = 1 if (0 <= i < self.width and 0 <= j < self.height) else 0

    def build_blocked_grid(self, obstacles_grid_points=None):
        blocked = np.zeros((self.height, self.width), dtype=bool)
        blocked[0, :] = True
        blocked[-1, :] = True
        blocked[:, 0] = True
        blocked[:, -1] = True

        if obstacles_grid_points:
            for gx, gy in obstacles_grid_points:
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    blocked[gy, gx] = True

        self.blocked = blocked

    def build_exit_union_and_mesh_index_payload(self):
        # 저장은 "좌표 형태"만. 로드 시 shapely 재구성.
        payload = {
            "obstacles": self.obstacles,
            "exits": self.exit_list,
            "mesh_list": self.mesh_list,
            "pure_mesh": self.pure_mesh,
            "obstacle_mesh": self.obstacle_mesh,
            "adjacent_mesh": self.adjacent_mesh,
            "next_vertex_matrix": self.next_vertex_matrix,
            "distance": self.distance,
            "match_grid_to_mesh": self.match_grid_to_mesh,
            "match_mesh_to_grid": self.match_mesh_to_grid,
            "valid_space": self.valid_space,
            "blocked": self.blocked,
            "width": self.width,
            "height": self.height,
        }
        return payload

def cache_path(map_num: int, cache_dir="env_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"env_map_{map_num}.pkl")

def build_and_save(map_num: int, width: int, height: int,
                   map_base_dir="map_infos",
                   cache_dir="env_cache",
                   D=20):
    b = EnvBuilder(width, height)
    b.load_map_from_file(map_num, base_dir=map_base_dir)
    b.mesh_map(D=D)
    # blocked는 너희가 장애물을 grid로 만든 방식이 있으면 그걸 넣으면 됨.
    # 여기서는 blocked만 외벽 세팅
    b.build_blocked_grid(obstacles_grid_points=None)

    payload = b.build_exit_union_and_mesh_index_payload()
    out = cache_path(map_num, cache_dir)
    with open(out, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out

def load_cached(map_num: int, cache_dir="env_cache"):
    path = cache_path(map_num, cache_dir)
    with open(path, "rb") as f:
        return pickle.load(f)
