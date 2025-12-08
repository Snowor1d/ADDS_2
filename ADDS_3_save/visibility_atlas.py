# visibility_atlas.py
from __future__ import annotations
import math
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

    def precompute(self, rays_per_poly: int = 64, bsearch_iters: int = 12):
        """
        모든 (region, radius) 조합에 대해 1회성 시야 폴리곤 계산 → 캐시에 저장.
        이후 polygon_at()은 O(1) 조회만 수행.
        """
        if self._obs_index is None:
            # 장애물 없을 때도 빈 인덱스는 허용
            self._obs_index = _ObstacleIndex([])
            self._obs_version = 0
            self._cache.clear()

        for j in range(self.reg_rows):
            for i in range(self.reg_cols):
                center = self.region_center((i, j))
                for R in self._radii:
                    key = (i, j, float(R), self._obs_version)
                    if key in self._cache:
                        continue
                    pts: List[Vec2] = []
                    for k in range(rays_per_poly):
                        a = 2*math.pi * (k / rays_per_poly)
                        far = (center[0] + R*math.cos(a), center[1] + R*math.sin(a))
                        lo, hi = 0.0, 1.0
                        hit = far
                        for _ in range(bsearch_iters):
                            mid = 0.5*(lo+hi)
                            mx = center[0] + (far[0]-center[0]) * mid
                            my = center[1] + (far[1]-center[1]) * mid
                            if self._obs_index.blocked(center, (mx, my)):
                                #print("blocked됨")
                                hi = mid
                                hit = (mx, my)
                            else:
                                lo = mid
                        pts.append(hit)
                    poly = Polygon(pts)
                    self._cache[key] = poly

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
