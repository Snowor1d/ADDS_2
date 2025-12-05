# shadow_fov.py
from __future__ import annotations
import numpy as np
from typing import Tuple

Cell = Tuple[int, int]

class ShadowFOV:
    """
    blocked[y, x] == True 인 셀은 시야를 막는 벽으로 취급하는
    grid 기반 shadow casting FOV.
    """
    def __init__(self, blocked: np.ndarray):
        assert blocked.ndim == 2
        self.blocked = blocked.astype(bool)
        self.h, self.w = self.blocked.shape

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.w and 0 <= y < self.h

    def is_blocked(self, x: int, y: int) -> bool:
        if not self.in_bounds(x, y):
            return True
        return bool(self.blocked[y, x])

    def compute_fov(self, cx: int, cy: int, radius: int) -> np.ndarray:
        """
        (cx, cy)를 중심으로 반경 radius 안의 visible mask 반환.
        visible[y, x] == True 이면 보이는 셀.
        """
        vis = np.zeros_like(self.blocked, dtype=bool)
        if not self.in_bounds(cx, cy):
            return vis

        vis[cy, cx] = True

        for octant in range(8):
            self._cast_octant(cx, cy, radius, octant, 1, 1.0, 0.0, vis)

        return vis
    def _cast_octant(
        self,
        cx: int,
        cy: int,
        radius: int,
        octant: int,
        row: int,
        start_slope: float,
        end_slope: float,
        vis: np.ndarray,
    ):
        if start_slope < end_slope or row > radius:
            return

        radius_sq = radius * radius
        blocked_in_row = False
        new_start_slope = start_slope

        # 🔥 핵심 수정: col을 row에서 0까지 감소시키는 방향으로
        for col in range(row, -1, -1):
            l_slope = (col - 0.5) / (row + 0.5)
            r_slope = (col + 0.5) / (row - 0.5) if row > 0 else start_slope

            # 🔥 슬로프 범위 체크 조건 수정
            if l_slope < end_slope:
                break
            if r_slope > start_slope:
                continue

            dx, dy = self._transform_octant(octant, row, col)
            x = cx + dx
            y = cy + dy

            if (dx*dx + dy*dy) <= radius_sq and self.in_bounds(x, y):
                vis[y, x] = True

            tile_blocked = self.is_blocked(x, y)

            if blocked_in_row:
                if tile_blocked:
                    new_start_slope = r_slope
                    continue
                else:
                    blocked_in_row = False
                    start_slope = new_start_slope
            else:
                if tile_blocked and row < radius:
                    blocked_in_row = True
                    self._cast_octant(
                        cx, cy, radius, octant,
                        row + 1,
                        start_slope,
                        l_slope,
                        vis,
                    )
                    new_start_slope = r_slope

        if not blocked_in_row:
            self._cast_octant(
                cx, cy, radius, octant,
                row + 1,
                start_slope,
                end_slope,
                vis,
            )

    @staticmethod
    def _transform_octant(octant: int, row: int, col: int) -> Cell:
        if   octant == 0:  dx, dy = +col, -row
        elif octant == 1:  dx, dy = +row, -col
        elif octant == 2:  dx, dy = +row, +col
        elif octant == 3:  dx, dy = +col, +row
        elif octant == 4:  dx, dy = -col, +row
        elif octant == 5:  dx, dy = -row, +col
        elif octant == 6:  dx, dy = -row, -col
        else:              dx, dy = -col, -row
        return dx, dy
