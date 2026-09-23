"""Level representation for the ACCEL-style curriculum.

A level carries the map geometry itself rather than a generation seed. ACCEL
mutates geometry directly, so a seed would not survive the first edit; storing
the polygons also means replaying a level never pays the generator's cost
(measured at 0.8-2.1 s on average and up to 8.6 s at difficulty 6).
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from config import (
    MAP_AUGMENTATION_TRANSFORMS,
    MAP_H,
    MAP_W,
    UED_CROWD_RANGE,
    UED_DIFFICULTY_RANGE,
    UED_MAP_SIZE_RANGE,
    UED_MAP_SQUARE,
)

# Polygons are kept in the exact shapes model.py expects after extract_map():
# obstacles as lists of [x, y] pairs, exits as lists of (x, y) tuples.
Obstacle = List[List[int]]
Exit = List[Tuple[int, int]]

_id_counter = itertools.count(1)


def _next_level_id() -> int:
    return next(_id_counter)


def reserve_level_ids(min_next: int) -> None:
    """Restart the id counter above `min_next`.

    Ids are the population's only handle on a level, and they travel to the
    workers and back inside every transition. After a restart the counter would
    otherwise begin at 1 again and collide with the ids of restored levels,
    silently crediting one level's episodes to another.
    """
    global _id_counter
    _id_counter = itertools.count(max(1, int(min_next)))


@dataclass
class Level:
    """One training scenario the curriculum can select, score and mutate."""

    obstacles: List[Obstacle]
    exits: List[Exit]
    crowd_size: int
    width: int = MAP_W
    height: int = MAP_H

    # One D4 transform fixed per level. The simulator normally redraws this
    # every episode, which would make the same level appear with eight
    # different faces and inflate the number of trials needed to estimate a
    # success rate. Mutation already supplies geometric diversity.
    augmentation: str = "identity"

    # Which generator family produced this level. A real field rather than an
    # attribute set after construction, because the validation rules differ by
    # family and `child()` has to carry it to every descendant; as a stray
    # attribute it silently vanished on the first mutation and the child was
    # then judged by the wrong family's rules.
    generator: str = "citygen"

    # The street plan the obstacles were rendered from, for levels the city
    # generator produced. This is what makes the progression from an empty
    # field to a downtown survive mutation: the curriculum edits the plan and
    # re-derives the polygons, so a child of a city is still a city. Levels
    # that came from elsewhere, a real OSM crop for instance, have no plan and
    # are not mutated.
    plan: Optional[object] = None

    # Set by the runner when it hands the level out: True if this came from the
    # population, False if it is a freshly generated one. The replay-only
    # update rule reads it in the main process, so it has to survive the trip
    # through the worker's queue as a real field rather than a stray attribute.
    is_replay: bool = False

    # Provenance, used for the complexity-over-time plots and for debugging
    # which mutation produced an unplayable level.
    level_id: int = field(default_factory=_next_level_id)
    parent_id: Optional[int] = None
    generation: int = 0
    mutation_ops: Tuple[str, ...] = ()
    source_seed: Optional[int] = None
    difficulty: Optional[int] = None

    def child(
        self,
        obstacles: Sequence[Obstacle],
        exits: Sequence[Exit],
        crowd_size: int,
        ops: Sequence[str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        plan: Optional[object] = None,
    ) -> "Level":
        """Build a mutated descendant, carrying the lineage forward."""
        return Level(
            obstacles=[[list(p) for p in poly] for poly in obstacles],
            exits=[[tuple(p) for p in poly] for poly in exits],
            crowd_size=int(crowd_size),
            width=int(self.width if width is None else width),
            height=int(self.height if height is None else height),
            augmentation=self.augmentation,
            generator=self.generator,
            plan=plan if plan is not None else self.plan,
            parent_id=self.level_id,
            generation=self.generation + 1,
            mutation_ops=tuple(ops),
            source_seed=self.source_seed,
            difficulty=self.difficulty,
        )

    def obstacle_area(self) -> float:
        """Total obstacle area, the cheap complexity proxy used for logging."""
        total = 0.0
        for poly in self.obstacles:
            if len(poly) < 3:
                continue
            acc = 0.0
            for i in range(len(poly)):
                x1, y1 = poly[i][0], poly[i][1]
                x2, y2 = poly[(i + 1) % len(poly)][0], poly[(i + 1) % len(poly)][1]
                acc += x1 * y2 - x2 * y1
            total += abs(acc) * 0.5
        return total

    def density(self) -> float:
        area = float(self.width * self.height)
        return self.obstacle_area() / area if area > 0 else 0.0

    def summary(self) -> dict:
        return {
            "level_id": self.level_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "n_obstacles": len(self.obstacles),
            "n_exits": len(self.exits),
            "crowd_size": self.crowd_size,
            "density": round(self.density(), 4),
            "ops": list(self.mutation_ops),
            "generator": self.generator,
            "morphology": getattr(self.plan, "morphology", None),
            "development": (round(self.plan.development, 3)
                            if self.plan is not None else None),
        }


def level_from_map_data(
    data,
    crowd_size: int,
    augmentation: str = "identity",
    difficulty: Optional[int] = None,
    generator: str = "random_map",
) -> Level:
    """Wrap a `random_map.MapData` into a Level.

    Kept for the numbered maps in `map_infos/` and the GUI editors, which
    still go through `random_map`. The curriculum does not use this path: its
    levels come from `generate_city_level` and carry a street plan.
    """
    return Level(
        obstacles=[[list(p) for p in poly] for poly in data.obstacles],
        exits=[[tuple(p) for p in poly] for poly in data.exits],
        crowd_size=int(crowd_size),
        width=int(data.width),
        height=int(data.height),
        augmentation=augmentation,
        generator=generator,
        source_seed=int(data.seed_used),
        difficulty=difficulty,
    )


def sample_augmentation(rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return rng.choice(list(MAP_AUGMENTATION_TRANSFORMS))


def sample_map_size(rng: Optional[random.Random] = None):
    """Draw a world size in metres for a fresh level."""
    rng = rng or random
    lo, hi = UED_MAP_SIZE_RANGE
    w = rng.randint(int(lo), int(hi))
    h = w if UED_MAP_SQUARE else rng.randint(int(lo), int(hi))
    return w, h


def generate_random_level(
    rng: Optional[random.Random] = None,
    difficulty: Optional[int] = None,
    crowd_size: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    generator: Optional[str] = None,
) -> Level:
    """Draw a fresh level from the procedural generator.

    This is the curriculum's source of novelty; mutation supplies the rest.
    Generation is expensive, so callers run it on the dedicated producer thread
    rather than inside a rollout worker.
    """
    rng = rng or random
    if width is None or height is None:
        sampled_w, sampled_h = sample_map_size(rng)
        width = sampled_w if width is None else width
        height = sampled_h if height is None else height
    lo, hi = UED_DIFFICULTY_RANGE
    if difficulty is None:
        difficulty = rng.randint(lo, hi)
    if crowd_size is None:
        c_lo, c_hi = UED_CROWD_RANGE
        crowd_size = rng.randint(c_lo, c_hi)

    seed = rng.randrange(0, 2**31 - 1)
    # `generator` exists so a test or a study can pin the family. There is one
    # family now: `citygen` lays out a street network and decides which of its
    # blocks are built up. It replaced `random_map`, which scattered obstacles
    # on open ground, and `street_map`, which laid streets and then split each
    # block into separate footprints.
    #
    # Both were dropped for the same reason. Their obstacles were polygons the
    # curriculum edited directly, and the edits were morphology-blind, so a
    # layout that started out looking like a downtown stopped looking like one
    # within a few generations. `street_map` had a second problem: the gaps it
    # left between footprints inside a block were free space, which overstates
    # where a robot can go in exactly the way that road-derived extraction was
    # introduced to stop.
    family = generator or "citygen"
    if family != "citygen":
        raise ValueError(
            f"unknown generator family {family!r}. The scatter and street-map "
            "families were replaced by 'citygen'; see citygen/plan.py.")

    return generate_city_level(
        rng=rng, difficulty=int(difficulty), crowd_size=int(crowd_size),
        width=int(width), height=int(height), seed=seed)


# How many attempts to draw a playable level before giving up. Generation is
# not the expensive part any more, and a failure here costs an episode.
GENERATION_TRIES = 6


def generate_city_level(rng: random.Random, difficulty: int, crowd_size: int,
                        width: int, height: int, seed: Optional[int] = None,
                        morphology: Optional[str] = None) -> Level:
    """A fresh level from the city generator, with its street plan attached."""
    from citygen.generate import _place_exits, generate_city_plan
    from citygen.morphology import MORPHOLOGIES
    from citygen.validate import check

    if morphology is None:
        morphology = sorted(MORPHOLOGIES)[rng.randrange(len(MORPHOLOGIES))]

    last_why = None
    for _ in range(GENERATION_TRIES):
        plan = generate_city_plan(rng, morphology, difficulty=int(difficulty),
                                  width=int(width), height=int(height))
        exits = _place_exits(plan, 2, rng)
        ok, why = check(plan, exits)
        if not ok:
            last_why = why
            continue
        rings, _ = plan.render()
        return Level(
            obstacles=[[list(p) for p in ring] for ring in rings],
            exits=[[tuple(p) for p in ring] for ring in exits],
            crowd_size=int(crowd_size),
            width=int(width), height=int(height),
            augmentation=sample_augmentation(rng),
            generator="citygen",
            plan=plan,
            source_seed=None if seed is None else int(seed),
            difficulty=int(difficulty),
        )
    raise RuntimeError(
        f"could not draw a playable {morphology} level at difficulty "
        f"{difficulty}, {width}x{height} in {GENERATION_TRIES} tries"
        + (f" (last: {last_why})" if last_why else ""))
