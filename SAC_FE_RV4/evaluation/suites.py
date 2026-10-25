"""Frozen evaluation suites: build, check for leakage, load.

A suite file (data/suites/<name>.json) lists scenarios; each holds its map in
full (generated geometry, or a numbered map id), the crowd size, the D4
transform and the initial-placement seed. Building is deterministic given
the spec, but a suite is built once and committed: results are only
comparable on the same file, which `digest` identifies.

Leakage. A map counts as seen in training when its 50 x 50 raster (free /
obstacle / exit) differs by less than LEAKAGE_PIXEL_THRESHOLD of its pixels
from a training source map under any of the eight symmetries. RV3's 300
training maps hold such copies of 57 of maps 150-249 and of the real-building
maps 300, 303, 306 and 307; "auto" numbered suites drop them, and near-copies
inside a suite are kept once.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RASTER = 50
REAL_BUILDINGS = (130, 131, 132, 133, 134)


def suite_path(name: str, cfg) -> str:
    d = cfg.SUITE_DIR if os.path.isabs(cfg.SUITE_DIR) else os.path.join(ROOT, cfg.SUITE_DIR)
    return os.path.join(d, f"{name}.json")


def placement_seed(suite: str, map_key: str, k: int) -> int:
    h = hashlib.sha256(f"{suite}/{map_key}/{k}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31 - 1)


# ----------------------------------------------------------------------------
# rasters and leakage
# ----------------------------------------------------------------------------

def raster(obstacles, exits, W, H, n: int = RASTER) -> np.ndarray:
    xs = (np.arange(n) + 0.5) * W / n
    ys = (np.arange(n) + 0.5) * H / n
    X, Y = np.meshgrid(xs, ys)
    img = np.zeros((n, n), np.uint8)
    for o in obstacles:
        if len(o) >= 3:
            P = Polygon(o)
            P = P if P.is_valid else P.buffer(0)
            img[shapely.contains_xy(P, X, Y)] = 1
    for e in exits:
        if len(e) >= 3:
            img[shapely.contains_xy(Polygon(e), X, Y)] = 2
    return img


def d4_rasters(img: np.ndarray) -> List[np.ndarray]:
    out = []
    for flip in (False, True):
        base = np.fliplr(img) if flip else img
        for q in range(4):
            out.append(np.rot90(base, q))
    return out


class LeakageIndex:
    def __init__(self, maps: Sequence[int], cfg):
        import sim_maps
        self.ids = list(maps)
        imgs = []
        for m in self.ids:
            g = sim_maps.map_geometry(m, cfg)
            imgs.append(raster(g["obstacles"], g["exits"], g["width"], g["height"]))
        self.imgs = np.stack(imgs) if imgs else np.zeros((0, RASTER, RASTER), np.uint8)

    def nearest(self, img: np.ndarray) -> Tuple[float, Optional[int]]:
        if not len(self.ids):
            return 1.0, None
        best, who = 1.0, None
        for v in d4_rasters(img):
            d = (self.imgs != v[None]).mean(axis=(1, 2))
            j = int(d.argmin())
            if d[j] < best:
                best, who = float(d[j]), self.ids[j]
        return best, who


# ----------------------------------------------------------------------------
# building
# ----------------------------------------------------------------------------

def geometry_digest(g: dict) -> str:
    """Hash of a map's size, obstacles and exits, so a suite notices when a
    numbered map file it refers to has been edited since it was built."""
    blob = json.dumps([g["width"], g["height"], g["obstacles"],
                       [[list(p) for p in e] for e in g["exits"]]], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _group_numbered(m: int) -> str:
    import sim_maps
    if m in REAL_BUILDINGS:
        return "real_building"
    if m in sim_maps.CODED_MAPS:
        return "legacy_coded"
    return "handmade"


def _gen_one(args):
    seed, size, diff, crowd, max_density = args
    from ued.level import generate_level
    lv = generate_level(seed, size, diff, crowd, timeout_s=60.0, max_density=max_density)
    return seed, (lv.to_json() if lv is not None else None)


def _build_generated(name: str, spec: dict, cfg, leak: LeakageIndex, log,
                     workers: int = 6) -> List[dict]:
    """n_maps per size, split evenly over the difficulties. Candidate seeds
    are fixed per (difficulty, size): seed_base + 10000 * d_index + 2000 *
    size_index + j, taken in order, so the suite does not depend on timing."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from ued.level import Level

    lo, hi = spec["difficulty"]
    diffs = list(range(int(lo), int(hi) + 1))
    n = int(spec["n_maps"])
    quota = [n // len(diffs) + (1 if i < n % len(diffs) else 0) for i in range(len(diffs))]
    out = []
    thr = float(cfg.LEAKAGE_PIXEL_THRESHOLD)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for si, size in enumerate(spec["sizes_m"]):
            for di, diff in enumerate(diffs):
                base = int(spec["seed_base"]) + 10000 * di + 2000 * si
                need = quota[di]
                got = 0
                j = 0
                while got < need:
                    if j >= 2000:
                        raise RuntimeError(f"{name}: could not generate {need} maps at "
                                           f"{size} m, difficulty {diff}")
                    batch = [(base + j + k, int(size), diff, int(spec["crowd"]),
                              float(cfg.UED_MAX_OBSTACLE_DENSITY)) for k in range(workers * 2)]
                    j += len(batch)
                    for seed, lvj in ex.map(_gen_one, batch):
                        if got >= need or lvj is None:
                            continue
                        lv = Level.from_json(lvj)
                        img = raster(lv.obstacles, lv.exits, lv.width, lv.height)
                        d, who = leak.nearest(img) if int(size) == 100 else (1.0, None)
                        if d < thr:
                            log(f"  {name}: seed {seed} matches training map {who} ({d:.3f}); skipped")
                            continue
                        key = f"gen{seed}"
                        for k in range(int(spec["crowd_seeds"])):
                            out.append(dict(id=f"{name}/{key}/s{k}", map_key=key,
                                            group=f"gen_d{diff}", map_num=None,
                                            level=lv.to_json(), width=lv.width, height=lv.height,
                                            crowd=int(spec["crowd"]), augmentation="identity",
                                            seed=placement_seed(name, key, k), difficulty=diff,
                                            size_m=int(size), nearest_train_diff=round(d, 4)))
                        got += 1
                log(f"  {name}: {size} m, difficulty {diff}: {got} maps")
    return out


def _build_numbered(name: str, spec: dict, cfg, leak: LeakageIndex, log) -> List[dict]:
    import sim_maps
    if spec["maps"] == "auto":
        sources = set(cfg.LEAKAGE_SOURCE_MAPS) | set(cfg.DATASET_MAPS)
        cands = [m for m in sim_maps.all_numbered_maps()
                 if m != 0 and m not in sources and sim_maps.map_size(m, cfg) == (100, 100)]
    else:
        cands = list(spec["maps"])
    kept: List[Tuple[int, np.ndarray]] = []
    out = []
    thr = float(cfg.LEAKAGE_PIXEL_THRESHOLD)
    for m in cands:
        g = sim_maps.map_geometry(m, cfg)
        if not g["exits"]:
            log(f"  {name}: map {m} has no exit; skipped")
            continue
        img = raster(g["obstacles"], g["exits"], g["width"], g["height"])
        d, who = leak.nearest(img)
        if d < thr:
            log(f"  {name}: map {m} is a copy of training map {who} ({d:.3f}); skipped")
            continue
        dup = None
        for m2, img2 in kept:
            if min((v != img2).mean() for v in d4_rasters(img)) < thr:
                dup = m2
                break
        if dup is not None:
            log(f"  {name}: map {m} duplicates map {dup} already in the suite; skipped")
            continue
        kept.append((m, img))
        key = f"map{m}"
        for k in range(int(spec["crowd_seeds"])):
            out.append(dict(id=f"{name}/{key}/s{k}", map_key=key, group=_group_numbered(m),
                            map_num=int(m), level=None, width=g["width"], height=g["height"],
                            crowd=int(spec["crowd"]), augmentation="identity",
                            seed=placement_seed(name, key, k), difficulty=None,
                            map_digest=geometry_digest(g), nearest_train_diff=round(d, 4)))
    return out


def _build_subset(name: str, spec: dict, cfg, log) -> List[dict]:
    rng = random.Random(int(spec.get("seed", 0)))
    out = []
    for src, n in spec["sources"]:
        s = load_suite(src, cfg)
        by_map: Dict[str, List[dict]] = {}
        for sc in s["scenarios"]:
            by_map.setdefault(sc["map_key"], []).append(sc)
        keys = sorted(by_map)
        if int(n) >= len(keys):
            chosen = keys
        else:
            # real buildings first when the source has them
            real = [k for k in keys if by_map[k][0]["group"] == "real_building"]
            rest = [k for k in keys if k not in real]
            rng.shuffle(rest)
            chosen = (real + rest)[: int(n)]
        for k in chosen:
            for sc in by_map[k][: int(spec.get("crowd_seeds", 1))]:
                d = dict(sc)
                d["source_id"] = sc["id"]
                out.append(d)
    return out


def digest(scenarios: List[dict]) -> str:
    blob = json.dumps(scenarios, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_suite(name: str, cfg, log=print) -> dict:
    spec = cfg.SUITE_SPECS[name]
    kind = spec["kind"]
    if kind == "subset":
        scen = _build_subset(name, spec, cfg, log)
    else:
        leak = LeakageIndex(sorted(set(cfg.LEAKAGE_SOURCE_MAPS)), cfg)
        scen = (_build_generated if kind == "generated" else _build_numbered)(name, spec, cfg, leak, log)
    return {"name": name, "version": 1, "spec": spec,
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "leakage_sources": [min(cfg.LEAKAGE_SOURCE_MAPS), max(cfg.LEAKAGE_SOURCE_MAPS),
                                len(set(cfg.LEAKAGE_SOURCE_MAPS))],
            "leakage_threshold": float(cfg.LEAKAGE_PIXEL_THRESHOLD),
            "n_scenarios": len(scen), "n_maps": len({s["map_key"] for s in scen}),
            "digest": digest(scen), "scenarios": scen}


def save_suite(suite: dict, cfg) -> str:
    path = suite_path(suite["name"], cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(suite, fh, indent=1)
    return path


def load_suite(name: str, cfg) -> dict:
    path = suite_path(name, cfg) if not name.endswith(".json") else name
    with open(path, encoding="utf-8") as fh:
        s = json.load(fh)
    if digest(s["scenarios"]) != s["digest"]:
        raise ValueError(f"suite {name}: contents do not match its digest (edited by hand?)")
    import sim_maps
    checked = {}
    for sc in s["scenarios"]:
        want = sc.get("map_digest")
        m = sc.get("map_num")
        if want is None or m is None:
            continue
        if m not in checked:
            checked[m] = geometry_digest(sim_maps.map_geometry(int(m), cfg))
        if checked[m] != want:
            raise ValueError(f"suite {name}: map {m} has changed since the suite was built; "
                             "rebuild it (python3 -m cli.build_suites --only ... --force)")
    return s


def scenario_spec(sc: dict):
    from learn.rollout import EnvSpec
    from ued.level import Level
    lv = Level.from_json(sc["level"]) if sc.get("level") else None
    return EnvSpec(crowd=int(sc["crowd"]), map_num=sc.get("map_num"), level=lv,
                   augmentation=sc.get("augmentation", "identity"), seed=int(sc["seed"]))
