"""The held-out evaluation set.

Zero-shot transfer is the claim UED makes, so it is the only number that
settles whether the curriculum worked. The set has to be fixed across every
condition and every seed, which is why the procedural half is generated from
hard-coded seeds and cached to disk rather than drawn per run.

Two halves, measuring different things:

* hand/JSON maps from `map_infos/`, outside the id range the DR baseline trains
  on, which test transfer to geometry no generator produced;
* procedural maps at fixed seeds spread over the full difficulty range, which
  test transfer within the generator's own distribution and are the direct
  evidence for the generalisation claim.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from config import MAP_H, MAP_W

# Fixed forever. Changing these invalidates comparisons against runs already
# measured, so add new seeds rather than editing existing ones.
HOLDOUT_SEEDS = {
    1: (900001, 900002, 900003),
    2: (900011, 900012, 900013),
    3: (900021, 900022, 900023),
    4: (900031, 900032, 900033),
    5: (900041, 900042, 900043),
    6: (900051, 900052, 900053),
}

# Crowd size is held at the midpoint of the training range so evaluation does
# not silently vary with the curriculum's crowd axis.
HOLDOUT_CROWD_SIZE = 30

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "ued_holdout_levels.json")

_cached: Optional[List] = None


def _build() -> List:
    from random_map import RandomMapSpec, generate_map
    from ued.level import level_from_map_data

    levels = []
    for difficulty, seeds in sorted(HOLDOUT_SEEDS.items()):
        for seed in seeds:
            data = generate_map(
                RandomMapSpec(width=MAP_W, height=MAP_H, difficulty=difficulty, seed=seed)
            )
            levels.append(
                level_from_map_data(
                    data,
                    crowd_size=HOLDOUT_CROWD_SIZE,
                    # Identity, so evaluation measures the map as generated.
                    augmentation="identity",
                    difficulty=difficulty,
                )
            )
    return levels


def _serialise(levels) -> list:
    return [
        {
            "obstacles": lv.obstacles,
            "exits": [[list(p) for p in poly] for poly in lv.exits],
            "crowd_size": lv.crowd_size,
            "width": lv.width,
            "height": lv.height,
            "difficulty": lv.difficulty,
            "source_seed": lv.source_seed,
        }
        for lv in levels
    ]


def _deserialise(raw: list) -> List:
    from ued.level import Level

    out = []
    for d in raw:
        out.append(
            Level(
                obstacles=[[list(p) for p in poly] for poly in d["obstacles"]],
                exits=[[tuple(p) for p in poly] for poly in d["exits"]],
                crowd_size=int(d["crowd_size"]),
                width=int(d["width"]),
                height=int(d["height"]),
                augmentation="identity",
                difficulty=d.get("difficulty"),
                source_seed=d.get("source_seed"),
            )
        )
    return out


def holdout_levels(rebuild: bool = False) -> List:
    """Return the procedural held-out levels, building and caching on first use."""
    global _cached
    if _cached is not None and not rebuild:
        return _cached

    if os.path.exists(CACHE_PATH) and not rebuild:
        try:
            with open(CACHE_PATH, "r") as f:
                _cached = _deserialise(json.load(f))
                return _cached
        except Exception as e:
            print(f"[UED] holdout cache unreadable ({e}); rebuilding")

    levels = _build()
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(_serialise(levels), f)
    except Exception as e:
        print(f"[UED] could not cache holdout levels: {e}")
    _cached = levels
    return _cached


if __name__ == "__main__":
    lv = holdout_levels(rebuild=True)
    print(f"built {len(lv)} holdout levels -> {CACHE_PATH}")
    for x in lv:
        print(" ", x.summary(), "difficulty", x.difficulty, "seed", x.source_seed)
