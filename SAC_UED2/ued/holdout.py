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

# Fixed forever from here on. Changing these invalidates comparisons against
# runs already measured, so add new seeds rather than editing existing ones.
#
# Already invalidated once, deliberately. These seeds addressed the scatter
# generator, which no longer exists, so every level they named is gone and any
# zero-shot number measured before that replacement is not comparable with one
# measured after it. The seeds themselves are kept so the set stays
# reproducible from here.
HOLDOUT_SEEDS = {
    1: (900001, 900002, 900003),
    2: (900011, 900012, 900013),
    3: (900021, 900022, 900023),
    4: (900031, 900032, 900033),
    5: (900041, 900042, 900043),
    6: (900051, 900052, 900053),
}

# World sizes the held-out set is evaluated at, with the label used in the
# metric names. Two of these sit inside UED_MAP_SIZE_RANGE and two outside it
# on either side, because interpolating within the training sizes and
# extrapolating beyond them are different claims and a single average hides
# which one the policy can actually do.
HOLDOUT_SIZES = (
    (50, "below"),
    (85, "inside"),
    (125, "inside"),
    (180, "above"),
)

# Difficulty 0 is an empty room; it is a useful floor check during training but
# not representative of anything worth evaluating transfer on.
HOLDOUT_SIZE_SEED_OFFSET = 7000

# Crowd size is held at the midpoint of the training range so evaluation does
# not silently vary with the curriculum's crowd axis.
HOLDOUT_CROWD_SIZE = 30

CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "ued_holdout_levels.json")

_cached: Optional[List] = None


def _build() -> List:
    """The held-out evaluation set: every morphology, at every size band.

    Morphology is now an axis of the set, not just difficulty and size. That
    is the point of generating per-morphology rather than from one continuous
    realism scale: a zero-shot number can be reported against the morphology
    it was measured on, and held beside the real sites that share the tag,
    instead of against an average no real place resembles.

    Difficulty 0 is excluded. An empty field is a useful floor check during
    training but measures nothing about transfer to real layouts.
    """
    import random

    from citygen.morphology import MORPHOLOGIES
    from ued.level import generate_city_level

    morphologies = sorted(MORPHOLOGIES)
    levels = []
    for size, band in HOLDOUT_SIZES:
        for difficulty, seeds in sorted(HOLDOUT_SEEDS.items()):
            for k, seed in enumerate(seeds):
                # Offset per size so the same difficulty at two sizes is two
                # different maps rather than the same layout stretched.
                size_seed = seed + HOLDOUT_SIZE_SEED_OFFSET * (size % 97) + k
                # Morphology cycles with the seed index, so each difficulty
                # and size sees a spread of patterns rather than one.
                morph = morphologies[(difficulty * len(seeds) + k)
                                     % len(morphologies)]
                try:
                    level = generate_city_level(
                        rng=random.Random(size_seed),
                        difficulty=difficulty,
                        crowd_size=HOLDOUT_CROWD_SIZE,
                        width=size, height=size, seed=size_seed,
                        morphology=morph,
                    )
                except RuntimeError:
                    # Some size and morphology pairs are not satisfiable: a
                    # 50 m crop cannot hold a superblock. Skip rather than
                    # weaken the rules for the evaluation set.
                    continue
                # Identity augmentation, so evaluation measures the map as
                # generated.
                level.augmentation = "identity"
                level.size_band = band
                level.morphology = morph
                levels.append(level)
    return levels


# Bumped whenever _build changes what it produces. A cached set built by a
# different generator is not the same evaluation set, and silently reusing one
# would compare two runs against two different holdouts while reporting a
# single metric name.
CACHE_VERSION = 2


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
            "size_band": getattr(lv, "size_band", "inside"),
            # Morphology is a reporting axis now, so it has to survive the
            # cache round trip or per-morphology results silently become
            # per-nothing results.
            "morphology": getattr(lv, "morphology", None),
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
        out[-1].size_band = d.get("size_band", "inside")
        out[-1].morphology = d.get("morphology")
    return out


def holdout_levels(rebuild: bool = False) -> List:
    """Return the procedural held-out levels, building and caching on first use."""
    global _cached
    if _cached is not None and not rebuild:
        return _cached

    if os.path.exists(CACHE_PATH) and not rebuild:
        try:
            with open(CACHE_PATH, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and int(payload.get("version", 0)) == CACHE_VERSION:
                _cached = _deserialise(payload["levels"])
                return _cached
            print("[UED] holdout cache was built by an earlier generator; "
                  "rebuilding")
        except Exception as e:
            print(f"[UED] holdout cache unreadable ({e}); rebuilding")

    levels = _build()
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({"version": CACHE_VERSION,
                       "levels": _serialise(levels)}, f)
    except Exception as e:
        print(f"[UED] could not cache holdout levels: {e}")
    _cached = levels
    return _cached


if __name__ == "__main__":
    lv = holdout_levels(rebuild=True)
    print(f"built {len(lv)} holdout levels -> {CACHE_PATH}")
    for x in lv:
        print(" ", x.summary(), "difficulty", x.difficulty, "seed", x.source_seed)
