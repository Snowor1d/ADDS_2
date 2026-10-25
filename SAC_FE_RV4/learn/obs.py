"""The policy's observation, built the same way for training, evaluation and
replay.

Per decision instant (every ACTION_SCALE steps) one frame pair is rendered:

    ego   (EGO_MAP_SIZE, EGO_MAP_SIZE) uint8, centred on the robot
    glob  (DOWNSAMPLE_MAP_SIZE, DOWNSAMPLE_MAP_SIZE) uint8, the whole map

from a raster of the map (walls 50, exits 100, people 150, robot 255). The
network input is a stack of the last FRAME_STACK frames, newest first, as
float32 / 255, plus an optional static channel (geodesic closeness to the
nearest exit). With the default settings this is exactly what SAC_FE_RV3's
workers fed the policy, so RV3 checkpoints run unchanged.

The robot state vector is always five numbers,

    [x / width, y / height, follower flag, scale_w, scale_h]

and each network takes the first three plus, if configured, the two scale
values (MAP_SCALE_TO_ACTOR / MAP_SCALE_TO_CRITIC). scale = MAP_SCALE_N *
side / MAP_SCALE_UNIT_M, so a 50 m side is n and a 100 m side 2n.

`d4_*` apply one of the eight map symmetries to batched observations,
states and actions consistently; learn/sac.py uses them for replay
augmentation and tests/test_obs.py checks them against the simulator's own
geometric transform.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

STATE_DIM = 5
WALL = 50
EXIT = 100
PERSON = 150
ROBOT = 255


def _to_px(x, y, world_w, world_h, W, H) -> Tuple[int, int]:
    # model.return_current_image.to_px
    ix = int(np.clip(x / world_w * W, 0, W - 1))
    iy = int(np.clip(y / world_h * H, 0, H - 1))
    return ix, iy


def ego_crop(full: np.ndarray, center_px: Tuple[int, int], size: int,
             pad_value: int = WALL) -> np.ndarray:
    """ADDS_AS_reinforcement.ego_crop_from_full_map."""
    H, W = full.shape
    cx, cy = center_px
    half = size // 2
    x0, x1 = cx - half, cx - half + size
    y0, y1 = cy - half, cy - half + size
    sx0, sx1 = max(0, x0), min(W, x1)
    sy0, sy1 = max(0, y0), min(H, y1)
    crop = np.full((size, size), pad_value, dtype=full.dtype)
    dx0 = sx0 - x0
    dy0 = sy0 - y0
    crop[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = full[sy0:sy1, sx0:sx1]
    return crop


def downsample_max(full: np.ndarray, target: int) -> np.ndarray:
    """F.adaptive_max_pool2d(full, target). With an exact divisor it is a
    block max, computed in numpy without the torch round trip; otherwise the
    torch call is used so the bins are the same."""
    H, W = full.shape
    if H % target == 0 and W % target == 0:
        fh, fw = H // target, W // target
        return full.reshape(target, fh, target, fw).max(axis=(1, 3))
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(full).float()[None, None]
    return F.adaptive_max_pool2d(x, (target, target))[0, 0].byte().numpy()


def _geodesic_closeness(static: np.ndarray) -> np.ndarray:
    """uint8 field: 255 at exits falling to 0 at one map diagonal of walking
    (8-connected pixel graph), 0 on walls and on ground with no route out."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    H, W = static.shape
    free = static != WALL
    idx = -np.ones((H, W), dtype=np.int64)
    idx[free] = np.arange(int(free.sum()))
    rows, cols, wts = [], [], []
    # Each undirected 8-neighbour edge once: (y, x) -> (y + dy, x + dx).
    for dy, dx, w in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
        a = idx[0:H - dy, max(0, -dx):W - max(0, dx)]
        b = idx[dy:H, max(0, dx):W + min(0, dx)]
        m = (a >= 0) & (b >= 0)
        rows.append(a[m]); cols.append(b[m]); wts.append(np.full(int(m.sum()), w))
    n = int(free.sum())
    if n == 0:
        return np.zeros_like(static)
    r = np.concatenate(rows); c = np.concatenate(cols); w = np.concatenate(wts)
    g = coo_matrix((np.r_[w, w], (np.r_[r, c], np.r_[c, r])), shape=(n, n)).tocsr()
    src = idx[(static == EXIT) & free]
    out = np.zeros((H, W), dtype=np.uint8)
    if src.size == 0:
        return out
    d = dijkstra(g, directed=False, indices=src, min_only=True)
    dmax = math.hypot(H, W)
    close = np.clip(1.0 - d / dmax, 0.0, 1.0)
    close[~np.isfinite(d)] = 0.0
    out[free] = np.round(close * 255.0).astype(np.uint8)
    return out


class ObservationBuilder:
    """Renders one environment's frames. Make one per episode (it caches the
    static rasters of the map it was built for)."""

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.H = int(cfg.MAP_H)
        self.W = int(cfg.MAP_W)
        self.ego = int(cfg.EGO_MAP_SIZE)
        self.down = int(cfg.DOWNSAMPLE_MAP_SIZE)
        self.follower_px = cfg.OBS_FOLLOWER_PIXEL
        self.metric = cfg.OBS_EGO_MODE == "metric"
        self.mpp = float(cfg.OBS_EGO_METERS_PER_PX)
        self.static_channel = bool(cfg.OBS_STATIC_DISTANCE_CHANNEL)
        self.world_w = float(env.width)
        self.world_h = float(env.height)

        # The normalised static raster. RV3 rendered env.static_grid at world
        # resolution and drew people with MAP_H x MAP_W coordinates into it,
        # which only agrees at 100 m; it is rendered here at MAP_H x MAP_W.
        if env.static_grid.shape == (self.H, self.W) and env.width == self.W and env.height == self.H:
            self.static_full = env.static_grid.copy()
        else:
            self.static_full = self._render_static(self.H, self.W)
        if self.metric:
            mh = int(math.ceil(self.world_h / self.mpp))
            mw = int(math.ceil(self.world_w / self.mpp))
            if (mh, mw) == (self.H, self.W) and self.world_w == self.W and self.world_h == self.H:
                self.static_metric = self.static_full
            else:
                self.static_metric = self._render_static(mh, mw, metric=True)
        else:
            self.static_metric = None
        self.glob_static = None
        self.dist_full = None
        self.dist_metric = None
        if self.static_channel:
            self.dist_full = _geodesic_closeness(self.static_full)
            self.glob_static = downsample_max(self.dist_full, self.down)
            if self.metric:
                self.dist_metric = (self.dist_full if self.static_metric is self.static_full
                                    else _geodesic_closeness(self.static_metric))

        n = float(cfg.MAP_SCALE_N)
        unit = float(cfg.MAP_SCALE_UNIT_M)
        self.scale = (n * self.world_w / unit, n * self.world_h / unit)
        self.reward_side = max(self.world_w, self.world_h)

    # -- rasters -----------------------------------------------------------

    def _render_static(self, H, W, metric=False):
        import sim_accel
        grid = np.zeros((H, W), dtype=np.uint8)
        if metric:
            # pixel centre (ix + 0.5) * mpp in world metres
            sim_accel.render_static_map(grid, self.env.obstacles, self.env.exit_list,
                                        W * self.mpp, H * self.mpp, H, W)
        else:
            sim_accel.render_static_map(grid, self.env.obstacles, self.env.exit_list,
                                        self.world_w, self.world_h, H, W)
        return grid

    def _draw_agents(self, img, H, W, ww, wh):
        fp = self.follower_px
        for ag in self.env.crowds:
            if ag.dead:
                continue
            ix, iy = _to_px(ag.xy[0], ag.xy[1], ww, wh, W, H)
            img[iy, ix] = fp if (fp is not None and ag.type == 0) else PERSON
        for rb in getattr(self.env, "robots", []):
            ix, iy = _to_px(rb.xy[0], rb.xy[1], ww, wh, W, H)
            img[iy, ix] = ROBOT

    def full_image(self) -> np.ndarray:
        """model.return_current_image(MAP_H, MAP_W) (people 150, robot 255)."""
        img = self.static_full.copy()
        self._draw_agents(img, self.H, self.W, self.world_w, self.world_h)
        return img

    def robot_px(self) -> Tuple[int, int]:
        rx, ry = self.env.robot.xy
        return _to_px(rx, ry, self.world_w, self.world_h, self.W, self.H)

    def frames(self):
        """(ego uint8, glob uint8, ego_static uint8 | None) for the current
        state."""
        full = self.full_image()
        glob = downsample_max(full, self.down)
        if self.metric:
            mh, mw = self.static_metric.shape
            img = self.static_metric.copy()
            ww, wh = mw * self.mpp, mh * self.mpp
            self._draw_agents(img, mh, mw, ww, wh)
            rx, ry = self.env.robot.xy
            c = _to_px(rx, ry, ww, wh, mw, mh)
            ego = ego_crop(img, c, self.ego, pad_value=WALL)
            ego_static = (ego_crop(self.dist_metric, c, self.ego, pad_value=0)
                          if self.static_channel else None)
        else:
            c = self.robot_px()
            ego = ego_crop(full, c, self.ego, pad_value=WALL)
            ego_static = (ego_crop(self.dist_full, c, self.ego, pad_value=0)
                          if self.static_channel else None)
        return ego, glob, ego_static

    def robot_state(self) -> np.ndarray:
        env = self.env
        follower = 0.0 if env.agents_near_robot_num() == 0 else 1.0
        return np.array([env.robot.xy[0] / self.world_w, env.robot.xy[1] / self.world_h,
                         follower, self.scale[0], self.scale[1]], dtype=np.float32)


class StackedObservation:
    """The acting-side frame stack: uint8 frames, newest first."""

    def __init__(self, stack: int):
        self.stack = int(stack)
        self.ego = None
        self.glob = None
        self.ego_static = None

    def reset(self, ego, glob, ego_static=None):
        self.ego = np.repeat(ego[None], self.stack, axis=0)
        self.glob = np.repeat(glob[None], self.stack, axis=0)
        self.ego_static = ego_static

    def push(self, ego, glob, ego_static=None):
        self.ego = np.concatenate([ego[None], self.ego[:-1]], axis=0)
        self.glob = np.concatenate([glob[None], self.glob[:-1]], axis=0)
        self.ego_static = ego_static

    def network_input(self, glob_static=None):
        """(ego, glob) float32 arrays, C x H x W, values / 255."""
        ego = self.ego
        glob = self.glob
        if self.ego_static is not None:
            ego = np.concatenate([ego, self.ego_static[None]], axis=0)
        if glob_static is not None:
            glob = np.concatenate([glob, glob_static[None]], axis=0)
        return ego.astype(np.float32) / 255.0, glob.astype(np.float32) / 255.0


def network_state(state5, use_scale: bool):
    """The slice of the 5-vector a network reads (numpy or torch)."""
    return state5[..., :5] if use_scale else state5[..., :3]


# ----------------------------------------------------------------------------
# D4 symmetries on batched tensors
# ----------------------------------------------------------------------------
# Index k in 0..7: bit 2 = reflect first (x -> width - x), k & 3 = quarter
# turns as in map_augmentation.transform_point ((x, y) -> (y, width - x)).
# On an image indexed [row = y, col = x] a world quarter turn is
# torch.rot90(img, 1, dims=(-2, -1)) and the reflection is a flip of the
# column axis.

D4_NAMES = ("identity", "rotate_90", "rotate_180", "rotate_270",
            "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270")


def d4_image(x, k: int):
    import torch
    if k & 4:
        x = torch.flip(x, dims=(-1,))
    q = k & 3
    return torch.rot90(x, q, dims=(-2, -1)) if q else x


def d4_state(s, k: int):
    """s (..., 5): position in [0, 1] and the two scale values."""
    import torch
    u, v, f, sw, sh = s.unbind(-1)
    if k & 4:
        u = 1.0 - u
    for _ in range(k & 3):
        u, v = v, 1.0 - u
        sw, sh = sh, sw
    return torch.stack([u, v, f, sw, sh], dim=-1)


def d4_action(a, k: int):
    import torch
    dx, dy = a.unbind(-1)
    if k & 4:
        dx = -dx
    for _ in range(k & 3):
        dx, dy = dy, -dx
    return torch.stack([dx, dy], dim=-1)
