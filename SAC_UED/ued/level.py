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
    ) -> "Level":
        """Build a mutated descendant, carrying the lineage forward."""
        return Level(
            obstacles=[[list(p) for p in poly] for poly in obstacles],
            exits=[[tuple(p) for p in poly] for poly in exits],
            crowd_size=int(crowd_size),
            width=self.width,
            height=self.height,
            augmentation=self.augmentation,
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
        }


def level_from_map_data(
    data,
    crowd_size: int,
    augmentation: str = "identity",
    difficulty: Optional[int] = None,
) -> Level:
    """Wrap a `random_map.MapData` into a Level."""
    return Level(
        obstacles=[[list(p) for p in poly] for poly in data.obstacles],
        exits=[[tuple(p) for p in poly] for poly in data.exits],
        crowd_size=int(crowd_size),
        width=int(data.width),
        height=int(data.height),
        augmentation=augmentation,
        source_seed=int(data.seed_used),
        difficulty=difficulty,
    )


def sample_augmentation(rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return rng.choice(list(MAP_AUGMENTATION_TRANSFORMS))


def generate_random_level(
    rng: Optional[random.Random] = None,
    difficulty: Optional[int] = None,
    crowd_size: Optional[int] = None,
    width: int = MAP_W,
    height: int = MAP_H,
) -> Level:
    """Draw a fresh level from the procedural generator.

    This is the curriculum's source of novelty; mutation supplies the rest.
    Generation is expensive, so callers run it on the dedicated producer thread
    rather than inside a rollout worker.
    """
    from random_map import RandomMapSpec, generate_map

    rng = rng or random
    lo, hi = UED_DIFFICULTY_RANGE
    if difficulty is None:
        difficulty = rng.randint(lo, hi)
    if crowd_size is None:
        c_lo, c_hi = UED_CROWD_RANGE
        crowd_size = rng.randint(c_lo, c_hi)

    seed = rng.randrange(0, 2**31 - 1)
    data = generate_map(
        RandomMapSpec(width=width, height=height, difficulty=int(difficulty), seed=seed)
    )
    return level_from_map_data(
        data,
        crowd_size=crowd_size,
        augmentation=sample_augmentation(rng),
        difficulty=int(difficulty),
    )
