# mesh_cache.py
from __future__ import annotations
import os, json, hashlib, pickle, gzip
from typing import Any, Dict, List, Tuple

Mesh = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

def _round_obstacles(obstacles, nd=6):
    out = []
    for poly in obstacles:
        out.append([[round(float(x), nd), round(float(y), nd)] for x, y in poly])
    return out

def obstacles_fingerprint(obstacles, nd=6) -> str:
    """
    장애물 좌표를 정규화(반올림) 후 sha1.
    같은 장애물 → 같은 fingerprint.
    """
    norm = _round_obstacles(obstacles, nd=nd)
    s = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def cache_path(base_dir: str, map_num: int, width: int, height: int, obs_fp: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"mesh_map{map_num}_w{width}_h{height}_obs{obs_fp}.pkl.gz")

def save_mesh_cache(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with gzip.open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)

def load_mesh_cache(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)
