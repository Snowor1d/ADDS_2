"""Numbered maps: their size and their geometry without building a model.

A numbered map is either a JSON file (map_infos/map_<n>.json) or one of the
maps written into model.FightingModel.extract_map / make_random_exit. The
geometry function runs those methods on a stand-in object, so the map
leakage check and the suite builder see exactly the obstacles and exits the
simulator would use, without a navmesh build.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(ROOT, "map_infos")

# Numbers handled in code by FightingModel.extract_map (map 0 is the random
# generator and is not a fixed map).
CODED_MAPS = (6, 7, 8, 26, 50, 51, 53, 54) + tuple(range(100, 126)) + (130, 131, 132, 133, 134)


def json_maps() -> List[int]:
    out = []
    for name in os.listdir(MAP_DIR):
        if name.startswith("map_") and name.endswith(".json"):
            try:
                out.append(int(name[4:-5]))
            except ValueError:
                pass
    return sorted(out)


def all_numbered_maps() -> List[int]:
    return sorted(set(json_maps()) | set(CODED_MAPS))


@lru_cache(maxsize=None)
def _json(map_num: int) -> dict:
    with open(os.path.join(MAP_DIR, f"map_{map_num}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def map_size(map_num: int, cfg=None) -> Tuple[int, int]:
    """World (width, height) in metres the simulator should use."""
    if cfg is None:
        from configs import cfg as _cfg
        cfg = _cfg
    if map_num in cfg.HARDCODED_MAP_SIZES:
        s = cfg.HARDCODED_MAP_SIZES[map_num]
        return (int(s), int(s)) if not isinstance(s, (tuple, list)) else (int(s[0]), int(s[1]))
    path = os.path.join(MAP_DIR, f"map_{map_num}.json")
    if cfg.MAP_SIZE_FROM_FILE and map_num not in CODED_MAPS and os.path.exists(path):
        d = _json(map_num)
        return int(d.get("width", 100)), int(d.get("height", 100))
    return 100, 100


class _Stub:
    """Enough of FightingModel for extract_map and make_random_exit."""

    def __init__(self, map_num, w, h):
        self.map_num = map_num
        self.width = w
        self.height = h
        self._source_map_width = w
        self._source_map_height = h
        self.map_augmentation = "identity"
        self.obstacles = []
        self.exit_list = []

    def _apply_map_augmentation(self):
        pass

    def load_map_from_file(self, map_num, base_dir="map_infos"):
        import model
        return model.FightingModel.load_map_from_file(self, map_num, base_dir=base_dir)


def map_geometry(map_num: int, cfg=None) -> Dict:
    """{"width", "height", "obstacles", "exits"} as the simulator builds them
    (before augmentation and obstacle union)."""
    import model
    w, h = map_size(map_num, cfg)
    stub = _Stub(map_num, w, h)
    if map_num == 0:
        raise ValueError("map 0 is the random generator, not a fixed map")
    model.FightingModel.extract_map(stub, map_num)
    model.FightingModel.make_random_exit(stub)
    return {"width": w, "height": h,
            "obstacles": [[list(p) for p in poly] for poly in stub.obstacles],
            "exits": [[tuple(p) for p in poly] for poly in stub.exit_list]}
