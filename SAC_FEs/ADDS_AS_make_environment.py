#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_environment.py
- 전역 변수로 설정
- 맵 로드 -> triangulation -> adjacency -> floyd -> grid_to_tri -> npz 저장
- vision_atlas 없음
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, hashlib, time
import numpy as np

# =========================
# GLOBAL CONFIG (NO ARGPARSE)
# =========================
MAP_NUMS = [200]          # 여기에 여러개 넣으면 일괄 생성
MAP_DIR  = "map_infos"    # map_{num}.json 등이 있는 폴더 (프로젝트에 맞게 수정)
OUT_DIR  = "env_cache"

WIDTH  = 100
HEIGHT = 100

# Triangulation/mesh density control
D = 20                   # segments 중간점 간격 (너가 쓰던 개념)
USE_TRIANGLE_LIB = True  # triangle 패키지 사용

# grid_to_tri generation
GRID_STEP = 1            # 1이면 모든 cell 채움. 2면 2칸 간격 (속도/정밀 tradeoff)

# Floyd
INF = 1e15
USE_FLOAT32_DIST = True

# =========================
# Map Data Schema (example)
# =========================
@dataclass
class MapData:
    width: int
    height: int
    obstacles: list  # List[List[List[int]]]  (polygon list)
    exits: list | None = None


# =========================
# Utilities
# =========================
def sha1_of_obstacles(obstacles: list) -> str:
    s = json.dumps(obstacles, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_map_data(map_num: int) -> MapData:
    fp = Path(MAP_DIR) / f"map_{map_num}.json"
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)
    w = int(obj.get("width", WIDTH))
    h = int(obj.get("height", HEIGHT))
    obstacles = obj.get("obstacles", [])
    exits = obj.get("exits", None)
    return MapData(width=w, height=h, obstacles=obstacles, exits=exits)


# =========================
# Geometry
# =========================
def point_in_tri_barycentric(px, py, ax, ay, bx, by, cx, cy) -> bool:
    # barycentric technique
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x*v0x + v0y*v0y
    dot01 = v0x*v1x + v0y*v1y
    dot02 = v0x*v2x + v0y*v2y
    dot11 = v1x*v1x + v1y*v1y
    dot12 = v1x*v2x + v1y*v2y

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0.0:
        return False
    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv
    return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)


def tri_bbox(v0, v1, v2):
    xs = [v0[0], v1[0], v2[0]]
    ys = [v0[1], v1[1], v2[1]]
    return (min(xs), min(ys), max(xs), max(ys))


def centroid(v0, v1, v2):
    return ((v0[0] + v1[0] + v2[0]) / 3.0, (v0[1] + v1[1] + v2[1]) / 3.0)


# =========================
# Triangulation (PSLG -> triangle)
# =========================
def build_pslg_from_map(map_data: MapData, d: float = 20.0):
    """
    최소 버전 PSLG 생성.
    - boundary(사각형) + obstacle polygon edges
    - d 간격으로 edge에 중간점 추가(선택)
    반환:
      vertices: (N,2) float32
      segments: (M,2) int32
    """
    vertices = []
    segments = []

    def add_poly(poly_xy):
        # poly_xy: [[x,y],...], 닫혀있든 아니든 처리
        pts = [(float(x), float(y)) for x, y in poly_xy]
        if len(pts) >= 2 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            return
        base_idx = len(vertices)
        for p in pts:
            vertices.append(p)
        n = len(pts)
        for i in range(n):
            a = base_idx + i
            b = base_idx + ((i + 1) % n)
            segments.append((a, b))

    # boundary rectangle
    W, H = map_data.width, map_data.height
    boundary = [(0.0, 0.0), (W*1.0, 0.0), (W*1.0, H*1.0), (0.0, H*1.0)]
    add_poly(boundary)

    # obstacles
    for poly in map_data.obstacles:
        add_poly(poly)

    vertices = np.array(vertices, dtype=np.float32)
    segments = np.array(segments, dtype=np.int32)

    # (옵션) edge subdivide: 아주 단순히 'vertices/segments'를 늘리는 건
    # project마다 규칙이 달라서 여기서는 생략.
    # 너가 이미 쓰던 generate_segments_with_points(D)가 있으면,
    # 여기에서 그 함수로 치환하면 됨.

    return vertices, segments


def triangulate_pslg(vertices: np.ndarray, segments: np.ndarray):
    """
    triangle 라이브러리 기반 triangulation.
    pip: triangle (Jonathan Shewchuk Triangle wrapper)
    """
    import triangle as tr

    A = {"vertices": vertices, "segments": segments}
    # 'p' = PSLG, 'q' = quality mesh, 'a' = max area (여기선 미사용)
    B = tr.triangulate(A, "p")
    out_v = B["vertices"].astype(np.float32)
    out_t = B["triangles"].astype(np.int32)
    return out_v, out_t


# =========================
# Obstacle test (centroid inside obstacle)
# =========================
def build_obstacle_checker(obstacles: list):
    """
    shapely 있으면 빠르고 정확하게 centroid-in-polygon 판단 가능.
    없으면 fallback으로 ray casting (단순) 구현 가능.
    """
    try:
        from shapely.geometry import Point, Polygon
        from shapely.prepared import prep

        polys = []
        for poly in obstacles:
            pts = [(float(x), float(y)) for x, y in poly]
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            polys.append(Polygon(pts))
        preps = [prep(p) for p in polys]

        def is_inside(x, y) -> bool:
            pt = Point(float(x), float(y))
            for pp in preps:
                if pp.contains(pt):
                    return True
            return False

        return is_inside

    except Exception:
        # Fallback: even-odd ray casting per polygon
        def pip_ray(px, py, poly):
            pts = [(float(x), float(y)) for x, y in poly]
            if len(pts) >= 2 and pts[0] == pts[-1]:
                pts = pts[:-1]
            inside = False
            n = len(pts)
            for i in range(n):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % n]
                cond = ((y1 > py) != (y2 > py))
                if cond:
                    xinters = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
                    if px < xinters:
                        inside = not inside
            return inside

        def is_inside(x, y) -> bool:
            for poly in obstacles:
                if pip_ray(x, y, poly):
                    return True
            return False

        return is_inside


# =========================
# Adjacency via Edge Hashing
# =========================
def build_adjacency_by_edges(triangles: np.ndarray):
    """
    triangles: (T,3) vertex indices
    return:
      neighbors: list[list[int]]  (모든 triangle에 대한 neighbor tri ids)
    """
    T = triangles.shape[0]
    edge_map = {}  # (min_vi, max_vi) -> [tri_ids...]

    for tid in range(T):
        a, b, c = triangles[tid]
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            if u > v:
                u, v = v, u
            edge_map.setdefault((u, v), []).append(tid)

    neighbors = [[] for _ in range(T)]
    for edge, tris in edge_map.items():
        if len(tris) == 2:
            t1, t2 = tris
            neighbors[t1].append(t2)
            neighbors[t2].append(t1)
        # len==1 -> boundary edge
        # len>2 -> 비정상/중복 (triangle 결과에 따라 간혹 생길 수 있음)
    return neighbors


def to_csr(neighbors: list[list[int]]):
    """
    ragged neighbor list -> CSR format arrays (indptr, indices)
    """
    indptr = [0]
    indices = []
    for nbrs in neighbors:
        indices.extend(nbrs)
        indptr.append(len(indices))
    return np.array(indptr, dtype=np.int32), np.array(indices, dtype=np.int32)


# =========================
# Floyd-Warshall on pure triangles
# =========================
def floyd_warshall_pure(tri_centroids: np.ndarray,
                        neighbors: list[list[int]],
                        pure_ids: np.ndarray):
    """
    pure_ids: (P,) triangle ids that are NOT obstacles
    return:
      dist: (P,P) float
      nxt:  (P,P) int (next hop index in pure-index space, -1 if none)
    """
    P = pure_ids.shape[0]
    id_to_p = {int(tid): i for i, tid in enumerate(pure_ids.tolist())}

    dist = np.full((P, P), INF, dtype=np.float32 if USE_FLOAT32_DIST else np.float64)
    nxt  = np.full((P, P), -1, dtype=np.int32)

    # self
    for i in range(P):
        dist[i, i] = 0.0
        nxt[i, i] = i

    # edges
    for tid in pure_ids:
        i = id_to_p[int(tid)]
        cx, cy = tri_centroids[int(tid)]
        for nb in neighbors[int(tid)]:
            if nb not in id_to_p:
                continue
            j = id_to_p[int(nb)]
            nx, ny = tri_centroids[int(nb)]
            w = float(np.hypot(nx - cx, ny - cy))
            if w < dist[i, j]:
                dist[i, j] = w
                nxt[i, j] = j

    # floyd
    for k in range(P):
        dk = dist[:, k].reshape(-1, 1)   # (P,1)
        kj = dist[k, :].reshape(1, -1)   # (1,P)
        alt = dk + kj                     # (P,P)
        improved = alt < dist
        if np.any(improved):
            dist = np.where(improved, alt, dist)
            # next update: nxt[i,j] = nxt[i,k]
            nk = np.broadcast_to(nxt[:, k].reshape(-1, 1), (P, P))
            nxt = np.where(improved, nk, nxt)

    return dist, nxt


# =========================
# grid_to_tri (bbox-based fill)
# =========================
def build_grid_to_tri(vertices: np.ndarray,
                      triangles: np.ndarray,
                      pure_ids: np.ndarray,
                      width: int,
                      height: int,
                      grid_step: int = 1):
    """
    (H,W) int array: each cell maps to pure-index (0..P-1), else -1.
    - bbox 범위의 grid만 검사해서 채움
    """
    P = pure_ids.shape[0]
    triid_to_p = {int(tid): i for i, tid in enumerate(pure_ids.tolist())}
    grid = np.full((height, width), -1, dtype=np.int32)

    for tid in pure_ids:
        tid = int(tid)
        pidx = triid_to_p[tid]
        a, b, c = triangles[tid]
        v0 = vertices[a]
        v1 = vertices[b]
        v2 = vertices[c]
        x0, y0, x1, y1 = tri_bbox(v0, v1, v2)

        # clamp to grid
        gx0 = max(0, int(np.floor(x0)))
        gy0 = max(0, int(np.floor(y0)))
        gx1 = min(width - 1, int(np.ceil(x1)))
        gy1 = min(height - 1, int(np.ceil(y1)))

        ax, ay = float(v0[0]), float(v0[1])
        bx, by = float(v1[0]), float(v1[1])
        cx, cy = float(v2[0]), float(v2[1])

        for y in range(gy0, gy1 + 1, grid_step):
            py = y + 0.5
            for x in range(gx0, gx1 + 1, grid_step):
                px = x + 0.5
                if point_in_tri_barycentric(px, py, ax, ay, bx, by, cx, cy):
                    grid[y, x] = pidx

    if grid_step != 1:
        # 빈칸이 남을 수 있으니, 간단한 최근접 보간(선택)
        # 여기선 생략. 필요하면 추가해줄게.
        pass

    return grid


# =========================
# Main build per map
# =========================
def build_and_save(map_num: int):
    t0 = time.time()
    out_root = ensure_dir(OUT_DIR)

    map_data = load_map_data(map_num)
    obstacles_hash = sha1_of_obstacles(map_data.obstacles)

    # 1) PSLG -> triangulate
    v_pslg, s_pslg = build_pslg_from_map(map_data, d=D)
    v_all, tri = triangulate_pslg(v_pslg, s_pslg)

    # 2) centroid + obstacle tri mark
    is_inside = build_obstacle_checker(map_data.obstacles)

    T = tri.shape[0]
    tri_centroids = np.zeros((T, 2), dtype=np.float32)
    is_obstacle_tri = np.zeros((T,), dtype=np.bool_)

    for tid in range(T):
        a, b, c = tri[tid]
        v0 = v_all[a]; v1 = v_all[b]; v2 = v_all[c]
        cx, cy = centroid(v0, v1, v2)
        tri_centroids[tid] = (cx, cy)
        if is_inside(cx, cy):
            is_obstacle_tri[tid] = True

    pure_ids = np.where(~is_obstacle_tri)[0].astype(np.int32)

    # 3) adjacency (all triangles)
    neighbors_all = build_adjacency_by_edges(tri)
    indptr, indices = to_csr(neighbors_all)

    # 4) floyd on pure
    dist, nxt = floyd_warshall_pure(tri_centroids, neighbors_all, pure_ids)

    # 5) grid_to_tri
    grid_to_tri = build_grid_to_tri(v_all, tri, pure_ids, map_data.width, map_data.height, GRID_STEP)

    # 6) save
    out_path = out_root / f"env_map_{map_num}_W{map_data.width}_H{map_data.height}_D{D}.npz"
    meta = {
        "map_num": map_num,
        "width": map_data.width,
        "height": map_data.height,
        "D": D,
        "grid_step": GRID_STEP,
        "obstacles_hash": obstacles_hash,
        "triangles": int(T),
        "pure_triangles": int(pure_ids.shape[0]),
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    np.savez_compressed(
        out_path,
        vertices=v_all.astype(np.float32),
        triangles=tri.astype(np.int32),
        tri_centroids=tri_centroids.astype(np.float32),
        is_obstacle_tri=is_obstacle_tri.astype(np.bool_),
        pure_ids=pure_ids.astype(np.int32),
        adj_indptr=indptr,
        adj_indices=indices,
        floyd_dist=dist.astype(np.float32 if USE_FLOAT32_DIST else np.float64),
        floyd_next=nxt.astype(np.int32),
        grid_to_tri=grid_to_tri.astype(np.int32),
        meta_json=np.array([json.dumps(meta)], dtype=object),
    )

    dt = time.time() - t0
    print(f"[OK] map={map_num} saved -> {out_path} ({dt:.2f}s)")
    print("     meta:", meta)


def main():
    ensure_dir(OUT_DIR)
    for m in MAP_NUMS:
        build_and_save(int(m))


if __name__ == "__main__":
    main()
