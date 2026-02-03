#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
env_precomputed.py
- make_environment.py가 만든 npz를 로드해서 runtime에서 즉시 사용
- find_tri_id(x,y): grid_to_tri 기반 O(1)
- next_hop(from_pure_idx, to_pure_idx): floyd_next 기반 O(1)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


@dataclass
class PrecomputedEnv:
    vertices: np.ndarray        # (N,2)
    triangles: np.ndarray       # (T,3)
    tri_centroids: np.ndarray   # (T,2)
    is_obstacle_tri: np.ndarray # (T,)
    pure_ids: np.ndarray        # (P,) triangle-id list (global tri id)
    grid_to_tri: np.ndarray     # (H,W) pure-index or -1
    floyd_dist: np.ndarray      # (P,P)
    floyd_next: np.ndarray      # (P,P)
    meta: dict

    def find_pure_index(self, x: float, y: float) -> int:
        """
        Returns pure-index (0..P-1) or -1 if outside/blocked.
        """
        xi = int(x)
        yi = int(y)
        H, W = self.grid_to_tri.shape
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            return -1
        return int(self.grid_to_tri[yi, xi])

    def pure_index_to_tri_id(self, pidx: int) -> int:
        if pidx < 0 or pidx >= self.pure_ids.shape[0]:
            return -1
        return int(self.pure_ids[pidx])

    def next_hop(self, from_pidx: int, to_pidx: int) -> int:
        """
        Returns next pure-index (0..P-1) on shortest path, or -1.
        """
        if from_pidx < 0 or to_pidx < 0:
            return -1
        if from_pidx >= self.floyd_next.shape[0] or to_pidx >= self.floyd_next.shape[1]:
            return -1
        return int(self.floyd_next[from_pidx, to_pidx])

    def distance(self, from_pidx: int, to_pidx: int) -> float:
        if from_pidx < 0 or to_pidx < 0:
            return float("inf")
        return float(self.floyd_dist[from_pidx, to_pidx])


def load_precomputed_env(cache_dir: str, map_num: int, width: int, height: int, D: int) -> PrecomputedEnv:
    p = Path(cache_dir) / f"env_map_{map_num}_W{width}_H{height}_D{D}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Precomputed env not found: {p}")

    z = np.load(p, allow_pickle=True)
    meta_json = str(z["meta_json"][0])
    meta = json.loads(meta_json)

    return PrecomputedEnv(
        vertices=z["vertices"],
        triangles=z["triangles"],
        tri_centroids=z["tri_centroids"],
        is_obstacle_tri=z["is_obstacle_tri"],
        pure_ids=z["pure_ids"],
        grid_to_tri=z["grid_to_tri"],
        floyd_dist=z["floyd_dist"],
        floyd_next=z["floyd_next"],
        meta=meta,
    )
