"""What a robot can know, and the one function that turns it into network input.

docs/outdoor_madrl_redesign.md sections 1 and 4. Three kinds of information,
kept apart because they have different owners:

  static    the city map and this episode's hazard, shared with the team in
            advance: buildings, hazard extent, walking distances to the
            hazard boundary. Known everywhere, observed or not.
  measured  what each robot's own sensor saw at each decision instant: the
            people within ROBOT_VISION metres and in line of sight, and the
            cells it could see. Anything else is *unobserved*, not empty.
  shared    what teammates' messages delivered: their pose, mode and
            measurements, subject to COMM_DELAY_DECISIONS and COMM_DROP_PROB.

The simulator's true crowd reaches only the critic, as `priv`, and the actor
only in the declared full-information experiment (ACTOR_GLOBAL_CROWD_TRUTH).

Storage and rebuilding follow section 6. A `DecisionRecord` holds exactly
what is stored per decision instant (about 6 KB for three robots), and
`build_observations` rebuilds every actor input from a window of records plus
the static layers. The rollout worker, the replay buffer, the viewer and the
zero-shot evaluator all call `build_observations`; nothing else constructs an
actor input.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Channel layouts. Changing any of these changes OBSERVATION_SCHEMA_VERSION.
EGO_STATIC = ("obstacle", "hazard", "off_map")
EGO_DYNAMIC = ("own_crowd", "own_observed")           # per history frame
MID_STATIC = ("walkable", "hazard", "hazard_boundary", "path_to_boundary",
              "in_map")
MID_DYNAMIC = ("team_crowd", "team_observed", "team_age")
GLOBAL_STATIC = ("obstacle", "hazard", "path_to_boundary")
GLOBAL_DYNAMIC = ("team_crowd", "team_observed")
OWN_STATE = ("x", "y", "log_w", "log_h", "signed_hazard_distance",
             "mode_off", "mode_guide", "mode_direct", "signal_x", "signal_y",
             "own_visible_crowd", "team_seen_in_hazard",
             "team_observed_hazard_fraction")
TEAMMATE_STATE = ("present", "dx", "dy", "mode_off", "mode_guide",
                  "mode_direct", "signal_x", "signal_y", "message_age")


def ego_channels(cfg) -> int:
    return len(EGO_STATIC) + len(EGO_DYNAMIC) * int(cfg.OBS_HISTORY_DECISIONS)


def mid_channels(cfg) -> int:
    return len(MID_STATIC) + len(MID_DYNAMIC)


def global_channels(cfg) -> int:
    return len(GLOBAL_STATIC) + len(GLOBAL_DYNAMIC)


def state_dim(cfg) -> int:
    return len(OWN_STATE) + len(TEAMMATE_STATE) * (int(cfg.MAX_ROBOTS) - 1)


def obs_shapes(cfg) -> Dict[str, Tuple[int, ...]]:
    e, m, g = int(cfg.EGO_MAP_SIZE), int(cfg.OBS_MID_SIZE), int(cfg.OBS_GLOBAL_SIZE)
    return {
        "ego": (ego_channels(cfg), e, e),
        "mid": (mid_channels(cfg), m, m),
        "glob": (global_channels(cfg), g, g),
        "state": (state_dim(cfg),),
        "priv": (1, g, g),
    }


# ------------------------------------------------------------ resampling

def area_resample(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Exact area average of a piecewise-constant raster onto a coarser grid.

    Works for non-integer ratios (a 100 m map onto 64 cells is 1.5625 m per
    cell) by interpolating the integral image at fractional cell edges, so
    each output cell is the mean of exactly the area it covers.
    """
    arr = np.asarray(arr, dtype=np.float64)
    h, w = arr.shape
    integ = np.zeros((h + 1, w + 1), dtype=np.float64)
    integ[1:, 1:] = arr.cumsum(0).cumsum(1)
    ys = np.linspace(0.0, h, out_h + 1)
    xs = np.linspace(0.0, w, out_w + 1)

    def sample(yv, xv):
        # Bilinear interpolation of the integral image is exact for a
        # piecewise-constant source.
        y0 = np.clip(np.floor(yv).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(xv).astype(int), 0, w - 1)
        fy = (yv - y0)[:, None]
        fx = (xv - x0)[None, :]
        a = integ[y0][:, x0]
        b = integ[y0][:, x0 + 1]
        c = integ[y0 + 1][:, x0]
        d = integ[y0 + 1][:, x0 + 1]
        return (a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx
                + c * fy * (1 - fx) + d * fy * fx)

    S = sample(ys, xs)
    total = S[1:, 1:] - S[:-1, 1:] - S[1:, :-1] + S[:-1, :-1]
    area = np.outer(np.diff(ys), np.diff(xs))
    return (total / np.maximum(area, 1e-12)).astype(np.float32)


def _crop(arr: np.ndarray, row0: int, col0: int, size: int,
          pad: float, out: Optional[np.ndarray] = None) -> np.ndarray:
    """size x size window starting at (row0, col0), padded outside. Written
    into `out` when given."""
    h, w = arr.shape[-2:]
    if out is None:
        out = np.empty(arr.shape[:-2] + (size, size), dtype=np.float32)
    out.fill(pad)
    r0, r1 = max(0, row0), min(h, row0 + size)
    c0, c1 = max(0, col0), min(w, col0 + size)
    if r0 < r1 and c0 < c1:
        out[..., r0 - row0:r1 - row0, c0 - col0:c1 - col0] = \
            arr[..., r0:r1, c0:c1]
    return out


# ------------------------------------------------------------ static layers

@dataclass
class StaticLayers:
    """The part of the observation that is fixed for an episode.

    Keyed by map geometry and hazard placement together: the same OSM crop
    with a different hazard seed is a different entry.
    """

    key: str
    width: int
    height: int
    zone: Optional[dict]
    obstacle: np.ndarray        # (H, W) fraction of each 1 m cell built up
    hazard: np.ndarray          # (H, W) fraction inside the hazard
    path: np.ndarray            # (H, W) walking metres to the hazard boundary
    mid_world: np.ndarray       # (len(MID_STATIC), H2, W2) at OBS_MID_RES_M
    glob: np.ndarray            # (len(GLOBAL_STATIC), G, G)
    hazard_area_m2: float
    zone_ref_m: float

    def arrays(self) -> Dict[str, np.ndarray]:
        return {"obstacle": self.obstacle, "hazard": self.hazard,
                "path": self.path, "mid_world": self.mid_world,
                "glob": self.glob}

    def meta(self) -> dict:
        return {"key": self.key, "width": self.width, "height": self.height,
                "zone": self.zone, "hazard_area_m2": self.hazard_area_m2,
                "zone_ref_m": self.zone_ref_m}

    @staticmethod
    def from_parts(meta: dict, arrays: Dict[str, np.ndarray]) -> "StaticLayers":
        return StaticLayers(
            key=meta["key"], width=int(meta["width"]),
            height=int(meta["height"]), zone=meta["zone"],
            obstacle=arrays["obstacle"], hazard=arrays["hazard"],
            path=arrays["path"], mid_world=arrays["mid_world"],
            glob=arrays["glob"], hazard_area_m2=float(meta["hazard_area_m2"]),
            zone_ref_m=float(meta["zone_ref_m"]))

    def zone_obj(self):
        if self.zone is None:
            return None
        from sim.danger import DangerZone
        return DangerZone.from_dict(self.zone)


def static_key(model) -> str:
    """Identity of a map geometry plus hazard placement."""
    h = hashlib.sha256()
    h.update(json.dumps([int(model.width), int(model.height)]).encode())
    for poly in model.obstacles:
        h.update(np.round(np.asarray(poly, dtype=np.float64), 4).tobytes())
    zone = getattr(model, "danger_zone", None)
    h.update(json.dumps(None if zone is None else zone.to_dict(),
                        sort_keys=True).encode())
    return h.hexdigest()[:20]


def _zone_mask(zone, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if zone is None:
        return np.zeros(np.broadcast(xs, ys).shape, dtype=bool)
    if zone.shape == "circle":
        return (xs - zone.cx) ** 2 + (ys - zone.cy) ** 2 <= zone.radius ** 2
    dx, dy = xs - zone.cx, ys - zone.cy
    c, s = math.cos(-zone.angle), math.sin(-zone.angle)
    lx, ly = dx * c - dy * s, dx * s + dy * c
    return (np.abs(lx) <= zone.half_w) & (np.abs(ly) <= zone.half_h)


def _path_to_boundary(walkable: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """Walking metres from every walkable cell to the hazard boundary.

    Multi-source Dijkstra on the 8-connected walkable grid, seeded at every
    walkable cell that touches a walkable cell on the other side of the
    boundary. Inside the hazard this is the way out; outside it is the way
    in, which is what `escape_distance` (inside only) does not provide.
    Unreachable cells are +inf.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    h, w = walkable.shape
    idx = -np.ones((h, w), dtype=np.int64)
    n = int(walkable.sum())
    out = np.full((h, w), np.inf, dtype=np.float32)
    if n == 0:
        return out
    idx[walkable] = np.arange(n)
    rows, cols, wts = [], [], []
    seeds = np.zeros((h, w), dtype=bool)
    for dy, dx, cost in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)),
                         (1, -1, math.sqrt(2))):
        ys0 = slice(0, h - dy)
        ys1 = slice(dy, h)
        if dx >= 0:
            xs0, xs1 = slice(0, w - dx), slice(dx, w)
        else:
            xs0, xs1 = slice(-dx, w), slice(0, w + dx)
        a = walkable[ys0, xs0] & walkable[ys1, xs1]
        if dx != 0 and dy != 0:
            # No diagonal squeeze between two blocked corners.
            if dx > 0:
                a &= walkable[ys0, xs1] | walkable[ys1, xs0]
            else:
                a &= walkable[ys0, xs1] | walkable[ys1, xs0]
        ia = idx[ys0, xs0][a]
        ib = idx[ys1, xs1][a]
        rows.append(ia)
        cols.append(ib)
        wts.append(np.full(ia.shape, cost))
        cross = a & (inside[ys0, xs0] != inside[ys1, xs1])
        seeds[ys0, xs0] |= cross
        seeds[ys1, xs1] |= cross
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    wv = np.concatenate(wts)
    graph = coo_matrix((wv, (r, c)), shape=(n, n)).tocsr()
    src = idx[seeds]
    if src.size == 0:
        return out
    dist = dijkstra(graph, directed=False, indices=src, min_only=True)
    out[walkable] = dist.astype(np.float32)
    return out


def build_static_layers(model, cfg) -> StaticLayers:
    """Rasterise the map and hazard once per geometry/hazard key."""
    from shapely import contains_xy, prepare
    from shapely.ops import unary_union
    from shapely.geometry import Polygon

    W, H = int(model.width), int(model.height)
    # 2 x 2 samples per 1 m cell: fractions in quarters, enough to tell a
    # wall edge from a wall.
    sub = (np.arange(2) + 0.5) / 2.0
    gx = (np.arange(W)[:, None] + sub[None, :]).reshape(-1)
    gy = (np.arange(H)[:, None] + sub[None, :]).reshape(-1)
    X, Y = np.meshgrid(gx, gy)
    polys = [Polygon(p) for p in model.obstacles if len(p) >= 3]
    if polys:
        union = unary_union([p.buffer(0) for p in polys])
        prepare(union)
        blocked = contains_xy(union, X, Y)
    else:
        blocked = np.zeros(X.shape, dtype=bool)
    obstacle = blocked.reshape(H, 2, W, 2).mean(axis=(1, 3)).astype(np.float32)
    zone = getattr(model, "danger_zone", None)
    hz = _zone_mask(zone, X, Y)
    hazard = hz.reshape(H, 2, W, 2).mean(axis=(1, 3)).astype(np.float32)

    walkable = obstacle < 0.5
    inside = hazard >= 0.5
    path = _path_to_boundary(walkable, inside)
    scale = float(cfg.OBS_PATH_SCALE_M)
    path_norm = np.where(np.isfinite(path), np.minimum(1.0, path / scale),
                         1.0).astype(np.float32)

    # Boundary cells at 1 m: hazard cells with a non-hazard 4-neighbour.
    b = np.zeros_like(inside)
    b[1:, :] |= inside[1:, :] != inside[:-1, :]
    b[:-1, :] |= inside[1:, :] != inside[:-1, :]
    b[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    b[:, :-1] |= inside[:, 1:] != inside[:, :-1]
    boundary = (b & inside).astype(np.float32)

    res = float(cfg.OBS_MID_RES_M)
    H2 = int(math.ceil(H / res))
    W2 = int(math.ceil(W / res))

    def to_mid(a, fill):
        # Pad to a whole number of mid cells, then exact area average.
        ph, pw = int(round(H2 * res)), int(round(W2 * res))
        padded = np.full((ph, pw), fill, dtype=np.float32)
        padded[:H, :W] = a
        return area_resample(padded, H2, W2)

    walk_f = 1.0 - obstacle
    in_map = np.ones((H, W), dtype=np.float32)
    mid_world = np.stack([to_mid(walk_f, 0.0), to_mid(hazard, 0.0),
                          to_mid(boundary, 0.0), to_mid(path_norm, 1.0),
                          to_mid(in_map, 0.0)]).astype(np.float32)

    G = int(cfg.OBS_GLOBAL_SIZE)
    glob = np.stack([area_resample(obstacle, G, G),
                     area_resample(hazard, G, G),
                     area_resample(path_norm, G, G)]).astype(np.float32)
    zref = 1.0 if zone is None else max(1.0, float(zone.max_escape_distance()))
    return StaticLayers(
        key=static_key(model), width=W, height=H,
        zone=None if zone is None else zone.to_dict(),
        obstacle=obstacle, hazard=hazard, path=path, mid_world=mid_world,
        glob=glob, hazard_area_m2=float(hazard.sum()), zone_ref_m=zref)


# ------------------------------------------------------------ measurement

@dataclass
class DecisionRecord:
    """Everything stored about one team at one decision instant.

    Arrays are padded to MAX_ROBOTS; `n_robots` says how many are real.
    `avail[r, s]` has bit j set when receiver r holds sender s's record from
    j decision instants ago (its own record always).
    """

    n_robots: int
    anchors: np.ndarray        # (R, 2) int32, ego-grid origin (col, row) in 1 m px
    counts: np.ndarray         # (R, E, E) uint8, people per 1 m cell
    observed: np.ndarray       # (R, E, E) bool
    pose: np.ndarray           # (R, 2) float32 world xy
    mode: np.ndarray           # (R,) int8 index into ROBOT_MODES
    signal: np.ndarray         # (R, 2) float32
    avail: np.ndarray          # (R, R) uint8 bitmask over lags
    priv: np.ndarray           # (G, G) uint8 true people per global cell


def crowd_positions(model) -> np.ndarray:
    """(N, 2) positions of everyone still in the crop."""
    pts = [(a.xy[0], a.xy[1]) for a in model.crowds
           if not a.dead and a.type in (0, 1, 2)]
    return (np.asarray(pts, dtype=np.float64) if pts
            else np.zeros((0, 2), dtype=np.float64))


def measure_robot(model, robot, cfg, positions: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One robot's measurement: (anchor, counts, observed) on its ego grid.

    A person is counted when it is within ROBOT_VISION of the robot and its
    1 m cell is in the robot's line of sight, so counts are never reported
    for an unobserved cell. Only the people in the crop are candidates; a
    person behind a building is simply absent from what the robot records.
    """
    from shapely import contains_xy, prepare

    E = int(cfg.EGO_MAP_SIZE)
    R = float(cfg.ROBOT_VISION)
    x, y = float(robot.xy[0]), float(robot.xy[1])
    ix, iy = int(math.floor(x)), int(math.floor(y))
    col0, row0 = ix - E // 2, iy - E // 2
    poly = model.vision_atlas.exact_polygon(x, y, R)
    cx = col0 + np.arange(E) + 0.5
    cy = row0 + np.arange(E) + 0.5
    CX, CY = np.meshgrid(cx, cy)
    in_map = (CX >= 0) & (CX < model.width) & (CY >= 0) & (CY < model.height)
    in_range = (CX - x) ** 2 + (CY - y) ** 2 <= R * R
    if poly.is_empty:
        observed = np.zeros((E, E), dtype=bool)
    else:
        prepare(poly)
        observed = contains_xy(poly, CX, CY) & in_map & in_range
    counts = np.zeros((E, E), dtype=np.int32)
    P = crowd_positions(model) if positions is None else positions
    if P.shape[0]:
        near = (P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2 <= R * R
        P = P[near]
        c = np.floor(P[:, 0]).astype(int) - col0
        r = np.floor(P[:, 1]).astype(int) - row0
        ok = (c >= 0) & (c < E) & (r >= 0) & (r < E)
        c, r = c[ok], r[ok]
        seen = observed[r, c]
        np.add.at(counts, (r[seen], c[seen]), 1)
    return (np.array([col0, row0], dtype=np.int32),
            np.minimum(counts, 255).astype(np.uint8), observed)


def truth_counts(model, cfg, positions: Optional[np.ndarray] = None
                 ) -> np.ndarray:
    """True people per global cell. Critic and full-information runs only."""
    G = int(cfg.OBS_GLOBAL_SIZE)
    out = np.zeros((G, G), dtype=np.int32)
    P = crowd_positions(model) if positions is None else positions
    if P.shape[0]:
        c = np.clip((P[:, 0] * G / model.width).astype(int), 0, G - 1)
        r = np.clip((P[:, 1] * G / model.height).astype(int), 0, G - 1)
        np.add.at(out, (r, c), 1)
    return np.minimum(out, 255).astype(np.uint8)


class CommChannel:
    """Which teammate records each robot holds, per decision instant.

    Delay is in decision intervals; a dropped message is lost for good (no
    retransmission). The draw uses its own random stream so enabling loss
    does not change the crowd's randomness.
    """

    def __init__(self, cfg, n_robots: int, seed: int = 0):
        self.delay = int(cfg.COMM_DELAY_DECISIONS)
        self.drop = float(cfg.COMM_DROP_PROB)
        self.history = int(cfg.OBS_HISTORY_DECISIONS)
        self.n = int(n_robots)
        self.rng = np.random.default_rng(seed)
        # delivered[(t_sent, receiver, sender)] -> bool, decided when sent.
        self._delivered: Dict[Tuple[int, int, int], bool] = {}

    def send(self, t: int) -> None:
        for r in range(self.n):
            for s in range(self.n):
                if r == s:
                    self._delivered[(t, r, s)] = True
                else:
                    self._delivered[(t, r, s)] = bool(
                        self.rng.random() >= self.drop)
        old = t - self.history - self.delay - 1
        for key in [k for k in self._delivered if k[0] < old]:
            del self._delivered[key]

    def avail(self, t: int, max_robots: int) -> np.ndarray:
        out = np.zeros((max_robots, max_robots), dtype=np.uint8)
        for r in range(self.n):
            for s in range(self.n):
                bits = 0
                for j in range(self.history):
                    ts = t - j
                    if ts < 0:
                        break
                    if r != s and j < self.delay:
                        continue
                    if self._delivered.get((ts, r, s), False):
                        bits |= 1 << j
                out[r, s] = bits
        return out


def record_decision(model, cfg, t: int, comm: CommChannel) -> DecisionRecord:
    """Measure every robot and note which messages have arrived."""
    Rm = int(cfg.MAX_ROBOTS)
    E = int(cfg.EGO_MAP_SIZE)
    robots = list(model.robots)[:Rm]
    n = len(robots)
    anchors = np.zeros((Rm, 2), dtype=np.int32)
    counts = np.zeros((Rm, E, E), dtype=np.uint8)
    observed = np.zeros((Rm, E, E), dtype=bool)
    pose = np.zeros((Rm, 2), dtype=np.float32)
    mode = np.zeros((Rm,), dtype=np.int8)
    signal = np.zeros((Rm, 2), dtype=np.float32)
    modes = tuple(cfg.ROBOT_MODES)
    P = crowd_positions(model)
    for i, rb in enumerate(robots):
        anchors[i], counts[i], observed[i] = measure_robot(model, rb, cfg, P)
        pose[i] = (float(rb.xy[0]), float(rb.xy[1]))
        mode[i] = modes.index(getattr(rb, "mode", "off"))
        signal[i] = getattr(rb, "signal_dir", (0.0, 0.0))
    comm.send(t)
    return DecisionRecord(n_robots=n, anchors=anchors, counts=counts,
                          observed=observed, pose=pose, mode=mode,
                          signal=signal, avail=comm.avail(t, Rm),
                          priv=truth_counts(model, cfg, P))


# ------------------------------------------------------------ building inputs

@dataclass
class ObsRequest:
    """One actor input to build: robot `receiver` at the newest record of
    `window` (oldest first, None where the episode had not started)."""

    static: StaticLayers
    window: Sequence[Optional[DecisionRecord]]
    receiver: int


_RC_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}


def _grid_rc(E: int):
    if E not in _RC_CACHE:
        _RC_CACHE[E] = np.nonzero(np.ones((E, E), dtype=bool))
    return _RC_CACHE[E]


def _union_key(req: ObsRequest, cfg):
    """Receivers holding the same set of records share one team union."""
    newest = req.window[-1]
    r = req.receiver
    share = bool(cfg.TEAM_SHARE_OBSERVATIONS)
    held = tuple(int(newest.avail[r, s]) if (share or s == r) else 0
                 for s in range(newest.n_robots))
    return (id(newest),) + tuple(id(w) for w in req.window) + held


def _team_union(req: ObsRequest, cfg):
    """Deduplicated team knowledge at 1 m: (gx, gy, count, lag) arrays.

    Per world cell the newest available measurement wins. Two robots that
    saw the same cell at the same instant saw the same people, so the cell is
    taken once, never summed.
    """
    H = int(cfg.OBS_HISTORY_DECISIONS)
    E = int(cfg.EGO_MAP_SIZE)
    newest = req.window[-1]
    r = req.receiver
    share = bool(cfg.TEAM_SHARE_OBSERVATIONS)
    gxs, gys, cnts, lags = [], [], [], []
    rr, cc = _grid_rc(E)
    for j in range(H):
        rec = req.window[-1 - j] if j < len(req.window) else None
        if rec is None:
            continue
        for s in range(newest.n_robots):
            if s != r and not share:
                continue
            if not (int(newest.avail[r, s]) >> j) & 1:
                continue
            obs = rec.observed[s]
            if not obs.any():
                continue
            sel = obs.reshape(-1)
            gxs.append((rec.anchors[s, 0] + cc)[sel])
            gys.append((rec.anchors[s, 1] + rr)[sel])
            cnts.append(rec.counts[s].reshape(-1)[sel].astype(np.int32))
            lags.append(np.full(int(sel.sum()), j, dtype=np.int32))
    if not gxs:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z.astype(np.int32), z.astype(np.int32)
    gx = np.concatenate(gxs).astype(np.int64)
    gy = np.concatenate(gys).astype(np.int64)
    cnt = np.concatenate(cnts)
    lag = np.concatenate(lags)
    W = int(req.static.width)
    lin = gy * (W + 64) + gx          # unique per cell, off-map cells included
    order = np.lexsort((-cnt, lag, lin))
    lin_s = lin[order]
    keep = np.ones(lin_s.shape[0], dtype=bool)
    keep[1:] = lin_s[1:] != lin_s[:-1]
    sel = order[keep]
    return gx[sel], gy[sel], cnt[sel], lag[sel]


def _bin(values_rc, weights, shape) -> np.ndarray:
    r, c = values_rc
    h, w = shape
    ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
    return np.bincount(r[ok] * w + c[ok], weights=weights[ok],
                       minlength=h * w).reshape(h, w)


def build_observations(requests: Sequence[ObsRequest], cfg,
                       out: Optional[Dict[str, np.ndarray]] = None,
                       rows: Optional[Sequence[int]] = None
                       ) -> Dict[str, np.ndarray]:
    """Actor inputs for a batch of requests.

    Returns float32 arrays: ego (B, Ce, E, E), mid (B, Cm, M, M),
    glob (B, Cg, G, G), state (B, D). The only constructor of actor input.

    With `out` and `rows`, request i is written into row rows[i] of the
    given arrays instead, which lets the replay buffer build a joint batch in
    place. Every ego, mid and glob channel is overwritten; `state` rows must
    arrive zeroed, because empty teammate slots are left untouched.
    """
    B = len(requests)
    shapes = obs_shapes(cfg)
    if out is None:
        out = {k: np.empty((B,) + shapes[k], dtype=np.float32)
               for k in ("ego", "mid", "glob")}
        out["state"] = np.zeros((B,) + shapes["state"], dtype=np.float32)
        rows = range(B)
    H = int(cfg.OBS_HISTORY_DECISIONS)
    E = int(cfg.EGO_MAP_SIZE)
    M = int(cfg.OBS_MID_SIZE)
    G = int(cfg.OBS_GLOBAL_SIZE)
    res_e = float(cfg.OBS_EGO_RES_M)
    res_m = float(cfg.OBS_MID_RES_M)
    sat = float(cfg.OBS_DENSITY_SATURATION)
    truth = bool(cfg.ACTOR_GLOBAL_CROWD_TRUTH)
    modes = tuple(cfg.ROBOT_MODES)
    Rm = int(cfg.MAX_ROBOTS)
    unions: Dict[tuple, tuple] = {}
    for b, req in zip(rows, requests):
        st = req.static
        rec = req.window[-1]
        r = int(req.receiver)
        Wm, Hm = int(st.width), int(st.height)
        px, py = float(rec.pose[r, 0]), float(rec.pose[r, 1])

        # ---------------- local branch
        col0, row0 = int(math.floor(px)) - E // 2, int(math.floor(py)) - E // 2
        ego = out["ego"][b]
        _crop(st.obstacle, row0, col0, E, 1.0, out=ego[0])
        _crop(st.hazard, row0, col0, E, 0.0, out=ego[1])
        ego[2].fill(1.0)
        r0, r1 = max(0, row0), min(Hm, row0 + E)
        c0, c1 = max(0, col0), min(Wm, col0 + E)
        if r0 < r1 and c0 < c1:
            ego[2, r0 - row0:r1 - row0, c0 - col0:c1 - col0] = 0.0
        first = next(w for w in req.window if w is not None)
        for j in range(H):
            frame = req.window[-1 - j] if j < len(req.window) else None
            frame = frame if frame is not None else first
            base = len(EGO_STATIC) + 2 * j
            ego[base] = np.minimum(
                1.0, frame.counts[r].astype(np.float32)
                / (res_e * res_e * sat))
            ego[base + 1] = frame.observed[r].astype(np.float32)

        # ---------------- team knowledge
        if truth:
            gx = gy = None
        else:
            key = _union_key(req, cfg)
            if key not in unions:
                unions[key] = _team_union(req, cfg)
            gx, gy, cnt, lag = unions[key]

        # ---------------- middle branch
        c2 = int(math.floor(px / res_m)) - M // 2
        r2 = int(math.floor(py / res_m)) - M // 2
        mid = out["mid"][b]
        pads = (0.0, 0.0, 0.0, 1.0, 0.0)
        for k in range(len(MID_STATIC)):
            _crop(st.mid_world[k], r2, c2, M, pads[k], out=mid[k])
        base = len(MID_STATIC)
        cell_px = res_m * res_m
        if truth:
            tc = _truth_mid(rec, st, cfg, c2, r2)
            mid[base] = np.minimum(1.0, tc / (cell_px * sat))
            mid[base + 1] = mid[len(MID_STATIC) - 1]      # in-map cells
            mid[base + 2] = 0.0
        else:
            mr = (np.floor((gy + 0.5) / res_m).astype(np.int64) - r2)
            mc = (np.floor((gx + 0.5) / res_m).astype(np.int64) - c2)
            crowd = _bin((mr, mc), cnt.astype(np.float64), (M, M))
            seen = _bin((mr, mc), np.ones(mr.shape[0]), (M, M))
            # Newest observation per mid cell, as an age in [0, 1].
            newest_lag = np.full((M, M), H, dtype=np.float64)
            ok = (mr >= 0) & (mr < M) & (mc >= 0) & (mc < M)
            np.minimum.at(newest_lag, (mr[ok], mc[ok]), lag[ok])
            mid[base] = np.minimum(1.0, crowd / (cell_px * sat))
            mid[base + 1] = np.minimum(1.0, seen / cell_px)
            mid[base + 2] = np.where(seen > 0,
                                     newest_lag / max(1, H - 1), 1.0)

        # ---------------- global branch
        glob = out["glob"][b]
        glob[:len(GLOBAL_STATIC)] = st.glob
        cw, ch = Wm / G, Hm / G
        base = len(GLOBAL_STATIC)
        if truth:
            glob[base] = np.minimum(1.0, rec.priv.astype(np.float32)
                                    / (cw * ch * sat))
            glob[base + 1] = 1.0
        else:
            gr = np.clip(np.floor((gy + 0.5) / ch).astype(np.int64), 0, G - 1)
            gc = np.clip(np.floor((gx + 0.5) / cw).astype(np.int64), 0, G - 1)
            inmap = (gx >= 0) & (gx < Wm) & (gy >= 0) & (gy < Hm)
            crowd = _bin((gr[inmap], gc[inmap]),
                         cnt[inmap].astype(np.float64), (G, G))
            seen = _bin((gr[inmap], gc[inmap]),
                        np.ones(int(inmap.sum())), (G, G))
            glob[base] = np.minimum(1.0, crowd / (cw * ch * sat))
            glob[base + 1] = np.minimum(1.0, seen / (cw * ch))

        # ---------------- scalar state
        s = out["state"][b]
        zone = st.zone_obj()
        s[0] = px / max(1.0, Wm)
        s[1] = py / max(1.0, Hm)
        s[2] = math.log(max(1.0, Wm) / float(cfg.MAP_SIZE_REFERENCE))
        s[3] = math.log(max(1.0, Hm) / float(cfg.MAP_SIZE_REFERENCE))
        s[4] = 0.0 if zone is None else max(
            -2.0, min(2.0, zone.signed_distance(px, py) / st.zone_ref_m))
        s[5 + int(rec.mode[r])] = 1.0
        s[8], s[9] = float(rec.signal[r, 0]), float(rec.signal[r, 1])
        own = float(rec.counts[r].sum())
        s[10] = min(2.0, own / 20.0)
        ref_people = max(1.0, st.hazard_area_m2 * 0.05)
        if truth:
            inside_truth = _truth_inside(rec, st, cfg)
            s[11] = min(2.0, inside_truth / ref_people)
            s[12] = 1.0
        else:
            inm = (gx >= 0) & (gx < Wm) & (gy >= 0) & (gy < Hm)
            hz = np.zeros(gx.shape[0], dtype=bool)
            hz[inm] = st.hazard[gy[inm], gx[inm]] >= 0.5
            s[11] = min(2.0, float(cnt[hz].sum()) / ref_people)
            s[12] = min(1.0, float(hz.sum()) / max(1.0, st.hazard_area_m2))

        # ---------------- teammates, nearest first
        base = len(OWN_STATE)
        width = len(TEAMMATE_STATE)
        mates = []
        for m in range(rec.n_robots):
            if m == r:
                continue
            bits = int(rec.avail[r, m])
            if bits == 0:
                continue
            lagm = (bits & -bits).bit_length() - 1   # newest delivered lag
            src = req.window[-1 - lagm]
            mx, my = float(src.pose[m, 0]), float(src.pose[m, 1])
            mates.append((math.hypot(mx - px, my - py), m, lagm, src))
        mates.sort(key=lambda t: (t[0], t[1]))
        scale = float(cfg.OBS_TEAM_DISTANCE_SCALE_M)
        for k, (_d, m, lagm, src) in enumerate(mates[:Rm - 1]):
            o = base + k * width
            s[o] = 1.0
            s[o + 1] = (float(src.pose[m, 0]) - px) / scale
            s[o + 2] = (float(src.pose[m, 1]) - py) / scale
            s[o + 3 + int(src.mode[m])] = 1.0
            s[o + 6] = float(src.signal[m, 0])
            s[o + 7] = float(src.signal[m, 1])
            s[o + 8] = lagm / max(1, H - 1)
    return out


def _truth_mid(rec, st, cfg, c2, r2) -> np.ndarray:
    """Full-information mid crowd: the global truth resampled. Upper bound
    experiment only; resolution is the global cell, not the mid cell."""
    G = int(cfg.OBS_GLOBAL_SIZE)
    M = int(cfg.OBS_MID_SIZE)
    res_m = float(cfg.OBS_MID_RES_M)
    cw, ch = st.width / G, st.height / G
    ys = (r2 + np.arange(M) + 0.5) * res_m
    xs = (c2 + np.arange(M) + 0.5) * res_m
    gr = np.clip((ys / ch).astype(int), 0, G - 1)
    gc = np.clip((xs / cw).astype(int), 0, G - 1)
    dens = rec.priv.astype(np.float32) / (cw * ch)
    out = dens[gr][:, gc] * res_m * res_m
    inside = ((ys >= 0) & (ys < st.height))[:, None] & \
             ((xs >= 0) & (xs < st.width))[None, :]
    return np.where(inside, out, 0.0)


def _truth_inside(rec, st, cfg) -> float:
    G = int(cfg.OBS_GLOBAL_SIZE)
    hz = area_resample(st.hazard, G, G) >= 0.5
    return float(rec.priv[hz].sum())


def critic_privileged(records: Sequence[DecisionRecord], cfg,
                      statics: Sequence[StaticLayers]) -> np.ndarray:
    """(B, 1, G, G) true crowd density for the critic, zeros if disabled."""
    G = int(cfg.OBS_GLOBAL_SIZE)
    out = np.zeros((len(records), 1, G, G), dtype=np.float32)
    if not bool(cfg.CRITIC_PRIVILEGED_CROWD):
        return out
    sat = float(cfg.OBS_DENSITY_SATURATION)
    for b, (rec, st) in enumerate(zip(records, statics)):
        cw, ch = st.width / G, st.height / G
        out[b, 0] = np.minimum(1.0, rec.priv.astype(np.float32)
                               / (cw * ch * sat))
    return out


# ------------------------------------------------------------ online history

class ObservationHistory:
    """A rollout's rolling window of records, for building actor input live.

    The replay buffer rebuilds the same windows from stored records, so the
    observation a worker acted on and the one the learner trains on are the
    same array (tests/test_observation.py checks this).
    """

    def __init__(self, cfg, static: StaticLayers, n_robots: int, seed: int = 0):
        self.cfg = cfg
        self.static = static
        self.comm = CommChannel(cfg, n_robots, seed=seed)
        self.records: List[DecisionRecord] = []
        self.t = -1

    def record(self, model) -> DecisionRecord:
        self.t += 1
        rec = record_decision(model, self.cfg, self.t, self.comm)
        self.records.append(rec)
        H = int(self.cfg.OBS_HISTORY_DECISIONS)
        if len(self.records) > H:
            self.records = self.records[-H:]
        return rec

    def window(self) -> List[Optional[DecisionRecord]]:
        H = int(self.cfg.OBS_HISTORY_DECISIONS)
        pad = [None] * (H - len(self.records))
        return pad + list(self.records)

    def team_observations(self) -> Dict[str, np.ndarray]:
        """Actor input for every real robot of the newest record."""
        rec = self.records[-1]
        reqs = [ObsRequest(self.static, self.window(), i)
                for i in range(rec.n_robots)]
        return build_observations(reqs, self.cfg)
