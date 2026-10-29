"""Walking distance to the nearest exit, for agent.danger (DANGER_DISTANCE).

RV3's agent.danger is model.distance_to_exit, the straight-line distance to
the nearest exit polygon: it ignores walls, so in a dead end behind a wall it
is far below the distance a person has to walk (map 1504: 21.5 m straight,
about 56 m walking). reward_b sums it, so leading people out of such a dead
end first makes the penalty larger.

With DANGER_DISTANCE = "geodesic", model.danger_distance reads this field
instead. It is computed once per environment: the map rasterised on a
DANGER_GEO_RES grid (pixel centres inside polygons), then a multi-source
Dijkstra from the exit cells over the free cells with a 16-neighbourhood
(orthogonal, diagonal and knight moves; a move is allowed only if the cells
it passes through are free, so it never cuts a wall corner). The
16-neighbourhood keeps the grid length within about 3 % of the straight-line
length in open space, against 8 % for 8 neighbours. Points are read by
bilinear interpolation; wall cells (a person's centre can fall in one next to
a wall) take the value of the nearest free cell plus the distance to it. An
exit too small to cover a cell centre is given the free cell nearest to it.

Values:
    route to an exit     the walking distance
    no route             `unreachable`: the farthest reachable distance in the
                         map plus its diagonal. Farther than any place with a
                         route, but finite and on the scale of the map, so
                         reward_b stays bounded and the simulator's
                         `danger > 1e5 -> removed` rule (RV3's "no exit")
                         does not remove people who are merely shut in
    map without exits    inf, as RV3's straight-line distance
"""

from __future__ import annotations

import math
import numpy as np

WALL = 50
EXIT = 100

# (dx, dy) of each undirected edge, and the cells it passes through relative
# to the start cell (besides start and end). Each edge is listed once; the
# graph is symmetric.
_MOVES = (
    ((1, 0), ()),
    ((0, 1), ()),
    ((1, 1), ((1, 0), (0, 1))),
    ((1, -1), ((1, 0), (0, -1))),
    ((2, 1), ((1, 0), (1, 1))),
    ((1, 2), ((0, 1), (1, 1))),
    ((2, -1), ((1, 0), (1, -1))),
    ((1, -2), ((0, -1), (1, -1))),
)


def _render_static_map(grid, obstacles, exits, world_w, world_h, H, W):
    """Obstacle pixels WALL, then exit pixels max(current, EXIT); a pixel is
    in a polygon when its centre is strictly inside (the same rule as
    SAC_FE_RV4's observation rasteriser)."""
    import shapely
    from shapely.geometry import Polygon

    def to_px(x, y):
        return (int(np.clip(x / world_w * W, 0, W - 1)), int(np.clip(y / world_h * H, 0, H - 1)))

    def pixels(poly):
        P = Polygon(poly)
        minx, miny, maxx, maxy = P.bounds
        gx0, gy0 = to_px(minx, miny)
        gx1, gy1 = to_px(maxx, maxy)
        IX, IY = np.meshgrid(np.arange(min(gx0, gx1), max(gx0, gx1) + 1),
                             np.arange(min(gy0, gy1), max(gy0, gy1) + 1), indexing="ij")
        inside = shapely.contains_xy(P, (IX + 0.5) * world_w / W, (IY + 0.5) * world_h / H)
        return IX[inside], IY[inside]

    for poly in obstacles:
        ix, iy = pixels(poly)
        grid[iy, ix] = WALL
    for poly in exits:
        ix, iy = pixels(poly)
        grid[iy, ix] = np.maximum(grid[iy, ix], EXIT)


class ExitDistanceField:
    """Walking distance (m) to the nearest exit at any point of one map."""

    def __init__(self, obstacles, exits, width: float, height: float, res: float = 0.5):
        from scipy.ndimage import distance_transform_edt
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra

        self.res = float(res)
        self.width = float(width)
        self.height = float(height)
        nx = int(math.ceil(self.width / self.res))
        ny = int(math.ceil(self.height / self.res))
        grid = np.zeros((ny, nx), dtype=np.uint8)            # [y, x]
        _render_static_map(grid, obstacles, exits, nx * self.res, ny * self.res, ny, nx)
        grid = grid.T                                        # [x, y]
        free = grid != WALL
        idx = -np.ones((nx, ny), dtype=np.int64)
        idx[free] = np.arange(int(free.sum()))
        n = int(free.sum())
        field = np.full((nx, ny), np.inf)
        is_exit = (grid == EXIT) & free
        if n:
            is_exit |= self._uncovered_exit_cells(exits, free, nx, ny)
        src = idx[is_exit]
        self.has_exit = bool(len(exits))
        if n and src.size:
            rows, cols, wts = [], [], []
            for (dx, dy), via in _MOVES:
                x0, x1 = max(0, -dx), nx - max(0, dx)
                y0, y1 = max(0, -dy), ny - max(0, dy)
                ok = free[x0:x1, y0:y1] & free[x0 + dx:x1 + dx, y0 + dy:y1 + dy]
                for vx, vy in via:
                    ok &= free[x0 + vx:x1 + vx, y0 + vy:y1 + vy]
                a = idx[x0:x1, y0:y1][ok]
                b = idx[x0 + dx:x1 + dx, y0 + dy:y1 + dy][ok]
                rows.append(a)
                cols.append(b)
                wts.append(np.full(a.size, math.hypot(dx, dy) * self.res))
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            w = np.concatenate(wts)
            g = coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
            d = dijkstra(g, directed=False, indices=src, min_only=True)
            field[free] = d
            # wall cells: the nearest free cell's value (inf if that cell has
            # no route) + the distance to it
            dist, (ix, iy) = distance_transform_edt(~free, return_indices=True)
            wall = ~free
            field[wall] = field[ix[wall], iy[wall]] + dist[wall] * self.res
        self.field = field
        finite = field[free & np.isfinite(field)]
        self.unreachable = float((finite.max() if finite.size else 0.0)
                                 + math.hypot(self.width, self.height))
        self.unreachable_free_cells = int((free & ~np.isfinite(field)).sum())

    def _uncovered_exit_cells(self, exits, free, nx, ny) -> np.ndarray:
        """For each exit that covers no cell centre, the free cell nearest
        to it (a resolution artefact, not a closed exit)."""
        import shapely
        from scipy.ndimage import distance_transform_edt
        from shapely.geometry import Polygon
        out = np.zeros((nx, ny), dtype=bool)
        near = None
        for poly in exits:
            P = Polygon(poly)
            minx, miny, maxx, maxy = P.bounds
            ix = np.arange(max(0, int(minx / self.res) - 1), min(nx, int(maxx / self.res) + 2))
            iy = np.arange(max(0, int(miny / self.res) - 1), min(ny, int(maxy / self.res) + 2))
            IX, IY = np.meshgrid(ix, iy, indexing="ij")
            if IX.size and shapely.contains_xy(P, (IX + 0.5) * self.res, (IY + 0.5) * self.res).any():
                continue
            q = P.representative_point()
            i = int(np.clip(q.x / self.res, 0, nx - 1))
            j = int(np.clip(q.y / self.res, 0, ny - 1))
            if not free[i, j]:
                if near is None:
                    near = distance_transform_edt(~free, return_indices=True)[1]
                i, j = int(near[0][i, j]), int(near[1][i, j])
            out[i, j] = True
        return out

    def __call__(self, x: float, y: float) -> float:
        """Walking distance at (x, y) in metres; `unreachable` where there is
        no route to an exit; inf if the map has no exit."""
        if not self.has_exit:
            return math.inf
        nx, ny = self.field.shape
        fx = min(max(x / self.res - 0.5, 0.0), nx - 1.0)
        fy = min(max(y / self.res - 0.5, 0.0), ny - 1.0)
        f = self.field
        cx = min(max(int(x / self.res), 0), nx - 1)
        cy = min(max(int(y / self.res), 0), ny - 1)
        if not math.isfinite(f[cx, cy]):
            return self.unreachable
        i0, j0 = int(fx), int(fy)
        i1, j1 = min(i0 + 1, nx - 1), min(j0 + 1, ny - 1)
        tx, ty = fx - i0, fy - j0
        num = den = 0.0
        for v, wgt in ((f[i0, j0], (1 - tx) * (1 - ty)), (f[i1, j0], tx * (1 - ty)),
                       (f[i0, j1], (1 - tx) * ty), (f[i1, j1], tx * ty)):
            if math.isfinite(v) and wgt > 0:
                num += v * wgt
                den += wgt
        return num / den if den > 0 else float(f[cx, cy])
