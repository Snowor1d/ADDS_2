# visibility_atlas.py
from __future__ import annotations
import math
import numpy as np
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from typing import Tuple, List, Dict, Optional, Iterable

Vec2 = Tuple[float, float]
Region = Tuple[int, int]

class _ObstacleIndex:
    def __init__(self, polys):
        flats = []
        for p in (polys or []):
            if isinstance(p, Polygon):
                flats.append(p)
            elif isinstance(p, MultiPolygon):
                flats.extend(list(p.geoms))
        self._geoms = flats
        self._tree  = STRtree(self._geoms) if self._geoms else None

    def blocked(self, p0, p1) -> bool:
        if self._tree is None:
            return False
        ray = LineString([p0, p1])
        if ray.is_empty or ray.length == 0:
            return False
        try:
            idxs = self._tree.query(ray, predicate="intersects")  # indices (2.x)
            candidates = self._tree.geometries.take(idxs)
        except TypeError:
            candidates = self._tree.query(ray)  # geometries (1.x)
        for poly in candidates:
            if ray.intersects(poly):  # 경계 스침 포함
                return True
        return False

class VisibilityAtlas:
    """
    2x2 셀을 1개 영역으로 묶어, (region, radius)별 시야 폴리곤을 '미리' 계산/저장.
    스텝 중 polygon_at()은 캐시 조회만 한다.
    """
    def __init__(self, world_w: float, world_h: float, region_cells: int = 2):
        self.world_w = world_w
        self.world_h = world_h
        self.region_size = region_cells  # 영역의 월드 크기(2×2이면 2*cell)
        self.reg_cols = int(math.floor(world_w / self.region_size))
        self.reg_rows = int(math.floor(world_h / self.region_size))
        self._cache: Dict[Tuple[int,int,float,int], Polygon] = {}
        self._radii: List[float] = []
        self._obs_version: int = 0
        self._obs_index: Optional[_ObstacleIndex] = None
        self._polys: List[Polygon] = []

    # ----- 좌표/영역 유틸 -----
    def region_of_xy(self, x: float, y: float) -> Region:
        i = max(0, min(self.reg_cols - 1, int(x // self.region_size)))
        j = max(0, min(self.reg_rows - 1, int(y // self.region_size)))
        return (i, j)

    def region_center(self, ij: Region) -> Vec2:
        cx = (ij[0] + 0.5) * self.region_size
        cy = (ij[1] + 0.5) * self.region_size
        return (cx, cy)

    # ----- 장애물/반경 등록 & 사전계산 -----
    def set_radii(self, radii: Iterable[float]):
        """사전계산 대상 반경 집합 등록 (예: {3+self_agent_vision, 3+self_robot_vision})."""
        self._radii = sorted(set(float(r) for r in radii))

    def rebuild_obstacles(self, polys: List[Polygon], obstacles_version: int):
        """장애물 바뀔 때 호출: 인덱스 재구축 + 캐시 무효화."""
        self._polys = polys or []
        self._obs_index = _ObstacleIndex(self._polys)
        self._obs_version = obstacles_version
        self._cache.clear()

    def _candidate_edges(self, center, R):
        """Obstacle edges that could possibly block a ray from `center`.

        One spatial-index query per region instead of one per ray-step. The
        edges come back as flat arrays so the ray casting below can run as a
        single vectorised operation.
        """
        if self._obs_index is None or self._obs_index._tree is None:
            return None, False, []

        disc = Point(center).buffer(float(R))
        try:
            idxs = self._obs_index._tree.query(disc, predicate="intersects")
            cands = list(self._obs_index._tree.geometries.take(idxs))
        except TypeError:
            cands = list(self._obs_index._tree.query(disc))
        if not cands:
            return None, False, []

        cpt = Point(center)
        inside = any(poly.covers(cpt) for poly in cands)

        ax, ay, bx, by = [], [], [], []
        for poly in cands:
            rings = [poly.exterior] + list(poly.interiors)
            for ring in rings:
                coords = list(ring.coords)
                for k in range(len(coords) - 1):
                    ax.append(coords[k][0]);  ay.append(coords[k][1])
                    bx.append(coords[k + 1][0]); by.append(coords[k + 1][1])
        if not ax:
            return None, inside, cands
        return ((np.asarray(ax), np.asarray(ay), np.asarray(bx),
                 np.asarray(by)), inside, cands)

    def _free_point_in_region(self, ij: Region, cands) -> Optional[Vec2]:
        """A point inside this region that is not inside an obstacle.

        The region centre is used when it is free, and it usually is. When it
        is not, the region still holds walkable ground in any layout with
        streets narrower than the region: a 4 m region straddling a 3 m alley
        has its centre in the building on one side. Everyone standing in that
        alley then got a visibility polygon of radius R/64, saw no neighbours
        at all, and walked through the crowd without feeling it. Measured
        before this fix: 20 per cent of a medina's pedestrians and 10 per cent
        of Shibuya's.
        """
        cx, cy = self.region_center(ij)
        half = self.region_size / 2.0
        # Rings outward from the centre, so the representative point is as
        # close to the nominal centre as the geometry allows.
        offsets = [(0.0, 0.0)]
        for frac in (0.35, 0.7, 0.95):
            d = half * frac
            offsets.extend([(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d),
                            (d, d), (d, -d), (-d, d), (-d, -d)])
        for dx, dy in offsets:
            x = min(max(cx + dx, 0.0), self.world_w)
            y = min(max(cy + dy, 0.0), self.world_h)
            p = Point(x, y)
            if not any(poly.covers(p) for poly in cands):
                return (x, y)
        return None

    def precompute(self, rays_per_poly: int = 64, bsearch_iters: int = 12):
        """
        모든 (region, radius) 조합에 대해 1회성 시야 폴리곤 계산 → 캐시에 저장.
        이후 polygon_at()은 O(1) 조회만 수행.

        Rays are cast analytically against the candidate obstacle edges rather
        than by bisecting with a shapely predicate. The old form built a
        LineString and queried the spatial index once per bisection step, which
        came to 64 rays times `bsearch_iters` steps per region and dominated
        the environment build: 526k queries on a 150x150 map. Solving for the
        ray/segment parameter directly needs one index query per region and one
        vectorised pass over its edges, and it returns the exact first hit
        instead of a bisection estimate, so `bsearch_iters` is no longer used.
        """
        if self._obs_index is None:
            # 장애물 없을 때도 빈 인덱스는 허용
            self._obs_index = _ObstacleIndex([])
            self._obs_version = 0
            self._cache.clear()

        angles = 2.0 * math.pi * (np.arange(rays_per_poly) / float(rays_per_poly))
        dir_x = np.cos(angles)
        dir_y = np.sin(angles)

        for j in range(self.reg_rows):
            for i in range(self.reg_cols):
                center = self.region_center((i, j))
                for R in self._radii:
                    key = (i, j, float(R), self._obs_version)
                    if key in self._cache:
                        continue

                    edges, center_inside, cands = self._candidate_edges(
                        center, R)
                    origin = center

                    if center_inside:
                        # The centre is inside a block, but the region may
                        # still contain walkable ground. Cast from a free
                        # point in the region instead of giving everyone
                        # standing there a blind spot the size of the region.
                        free = self._free_point_in_region((i, j), cands)
                        if free is None:
                            # The whole region is inside an obstacle. Nobody
                            # can stand here, so the degenerate polygon is
                            # correct and is kept as it always was.
                            t = np.full(rays_per_poly,
                                        R / (2.0 ** max(1, int(bsearch_iters))))
                            pts = list(zip(center[0] + dir_x * t,
                                           center[1] + dir_y * t))
                            self._cache[key] = Polygon(pts)
                            continue
                        origin = free
                        edges, _inside_again, _ = self._candidate_edges(
                            origin, R)

                    if edges is None:
                        t = np.full(rays_per_poly, float(R))
                    else:
                        t = self._cast_rays(origin, dir_x, dir_y, float(R),
                                            edges)

                    pts = list(zip(origin[0] + dir_x * t,
                                   origin[1] + dir_y * t))
                    self._cache[key] = Polygon(pts)

    @staticmethod
    def _cast_rays(center, dir_x, dir_y, R, edges):
        """First hit distance along each ray, or R when nothing blocks it.

        Solves C + t*d = A + u*(B-A) for every ray/edge pair at once, keeping
        hits with t in [0, R] and u in [0, 1].
        """
        ax, ay, bx, by = edges
        sx = (bx - ax)[None, :]
        sy = (by - ay)[None, :]
        rx = dir_x[:, None]
        ry = dir_y[:, None]

        denom = rx * sy - ry * sx
        parallel = np.abs(denom) < 1e-12
        safe = np.where(parallel, np.nan, denom)

        qx = (ax - center[0])[None, :]
        qy = (ay - center[1])[None, :]

        t = (qx * sy - qy * sx) / safe
        u = (qx * ry - qy * rx) / safe

        valid = np.isfinite(t) & (t >= 0.0) & (t <= R) & (u >= 0.0) & (u <= 1.0)
        t_masked = np.where(valid, t, np.inf)

        # A ray running exactly along an obstacle edge is parallel to it, so
        # the crossing solve above divides by zero and drops it. That is not a
        # corner case here: obstacles are axis-aligned at integer coordinates
        # and region centres sit on the same lattice, so the four axis-aligned
        # rays land collinear with an edge regularly. The old predicate counted
        # a touch as blocked, so the overlap start is taken as the hit.
        if parallel.any():
            # Collinear when the edge start also lies on the ray's line.
            collinear = parallel & (np.abs(qx * ry - qy * rx) < 1e-9)
            if collinear.any():
                ta = qx * rx + qy * ry                      # projection of A
                tb = ((bx - center[0])[None, :] * rx
                      + (by - center[1])[None, :] * ry)     # projection of B
                lo = np.minimum(ta, tb)
                hi = np.maximum(ta, tb)
                # Overlap of [lo, hi] with the ray's [0, R]; the near end is
                # the first contact.
                overlap = collinear & (hi >= 0.0) & (lo <= R)
                t_col = np.where(overlap, np.maximum(lo, 0.0), np.inf)
                t_masked = np.minimum(t_masked, t_col)

        first = t_masked.min(axis=1)
        return np.where(np.isfinite(first), first, R)

    # ----- 런타임 조회(O(1)) -----
    def polygon_at(self, x: float, y: float, radius: float, obstacles_version: int) -> Polygon:
        """
        스텝 중에는 캐시 조회만. (미사전계산된 반경이면 바로 반환)
        사전에 set_radii([...]) + precompute()를 호출해야 함.
        """
        if obstacles_version != self._obs_version:
            # 모델이 장애물 버전을 올렸는데 precompute를 안 돌린 경우 대비: 빈 폴리곤 반환
            # (또는 예외를 던져 강제할 수도 있음)
            return Polygon()

        R = float(radius)
        ij = self.region_of_xy(x, y)
        
        key = (ij[0], ij[1], R, self._obs_version)
        poly = self._cache.get(key)
        if poly is None:
            # 등록되지 않은 반경이 들어오면 가까운 반경으로 스냅하거나, 빈 폴리곤 반환.
            # 여기서는 안전하게 빈 폴리곤 반환.
            return Polygon()
        return poly
