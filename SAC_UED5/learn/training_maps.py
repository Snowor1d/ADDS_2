"""Training levels from the real OSM corpus (TRAIN_MAP_SOURCE = "dataset").

Each episode draws one (site, size) pair uniformly from
DATASET_SITES x DATASET_SIZES_M and turns the stored crop into a task with
the parameters in configs/training/dataset.py:

  * a fresh hazard, area from DATASET_DANGER_AREA_RANGE and shape from
    DATASET_DANGER_SHAPES, accepted only when the level validator finds
    walkable ground inside it and a way out (retried otherwise);
  * a crowd sized from the crop's walkable area and the target density
    (DATASET_DENSITY_BY_SIZE, else CROWD_DENSITY_RANGE), not the stored
    headcount, which the export capped at 800 on every 400 m crop;
  * perceptibility, prior warning, team size and one D4 symmetry from the
    DATASET_* ranges.

The crops never change during a run, so each is loaded and measured once per
worker.
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Tuple

HAZARD_TRIES = 30


class OsmTrainingMaps:
    def __init__(self, cfg):
        from osm_corpus.export import load_levels
        from sim.crowd_density import walkable_area

        self.cfg = cfg
        wanted = {(s, int(z)) for s in cfg.DATASET_SITES
                  for z in cfg.DATASET_SIZES_M}
        self.crops: Dict[Tuple[str, int], object] = {}
        self.walkable: Dict[Tuple[str, int], float] = {}
        for lv in load_levels():
            key = (lv.site_key, int(lv.width))
            if key in wanted:
                self.crops[key] = lv
        missing = sorted(wanted - set(self.crops))
        if missing:
            raise ValueError(f"OSM crops missing from the corpus: {missing}")
        for key, lv in self.crops.items():
            self.walkable[key] = float(walkable_area(lv.width, lv.height,
                                                     lv.obstacles))
        self.pairs: List[Tuple[str, int]] = sorted(self.crops)

    def sample(self, rng: random.Random):
        """A ready-to-run level. Tries other pairs if a crop cannot take a
        hazard of the drawn size."""
        for _ in range(10):
            key = self.pairs[rng.randrange(len(self.pairs))]
            level = self.make_level(key, rng)
            if level is not None:
                return level
        raise RuntimeError("could not place a hazard on any sampled OSM crop")

    def make_level(self, key, rng: random.Random):
        from citygen.plan import CityPlan
        from citygen.validate import check
        from sim.crowd_density import crowd_size_for
        from sim.danger import sample_zone

        cfg = self.cfg
        base = self.crops[key]
        level = copy.deepcopy(base)
        # A stand-in plan so the validator can rasterise the crop; an OSM crop
        # has no street plan of its own.
        plan = CityPlan(width=int(level.width), height=int(level.height),
                        morphology="osm", development=1.0)
        rings = [[list(p) for p in ring] for ring in level.obstacles]
        plan.render = lambda simplify_m=None, _r=rings: (_r, None)
        lo_a, hi_a = cfg.DATASET_DANGER_AREA_RANGE
        shapes = tuple(cfg.DATASET_DANGER_SHAPES)
        zone = None
        for _ in range(HAZARD_TRIES):
            area = rng.uniform(float(lo_a), float(hi_a))
            shape = shapes[rng.randrange(len(shapes))]
            cand = sample_zone(rng, float(level.width), float(level.height),
                               area, shape)
            ok, _why = check(plan, (), cand)
            if ok:
                zone = cand
                break
        if zone is None:
            return None
        walk = self.walkable[key]
        level.crowd_size = crowd_size_for(walk, rng=rng, size_m=key[1],
                                          table=cfg.DATASET_DENSITY_BY_SIZE)
        level.crowd_density = float(level.crowd_size) / max(1.0, walk)
        level.danger = zone
        level.exits = []
        lo_f, hi_f = cfg.DATASET_DANGER_INSIDE_FRACTION
        level.inside_fraction = rng.uniform(float(lo_f), float(hi_f))
        lo_p, hi_p = cfg.DATASET_DANGER_PERCEPTIBILITY
        level.perceptibility = rng.uniform(float(lo_p), float(hi_p))
        lo_i, hi_i = cfg.DATASET_PRIOR_INFORMED_FRACTION
        level.prior_informed_fraction = rng.uniform(float(lo_i), float(hi_i))
        lo_r, hi_r = cfg.DATASET_ROBOT_RANGE
        level.robot_num = rng.randint(int(lo_r), int(hi_r))
        if cfg.DATASET_AUGMENTATION:
            transforms = tuple(cfg.DATASET_AUGMENTATION_TRANSFORMS)
            level.augmentation = transforms[rng.randrange(len(transforms))]
        else:
            level.augmentation = "identity"
        level.level_id = -1
        level.is_replay = False
        level.site_key = key[0]
        return level

    def site_index(self, site: str) -> int:
        return list(self.cfg.DATASET_SITES).index(site)
