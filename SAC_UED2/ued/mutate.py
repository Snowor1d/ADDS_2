"""Breeding a child level, by editing its street plan.

ACCEL builds complexity entirely through offspring, so whatever mutation
preserves is what a lineage converges on. This module used to hold the
operators themselves: translate a polygon, rotate it, resize it, delete it,
add one. Those were morphology-blind. A block translated off its frontage is
no longer part of a street network, so a layout that started out looking like
a downtown stopped looking like one within a few generations, and the
curriculum could not build city-like fabric no matter what it selected for.

The operators now live in `citygen/mutate.py` and edit the street plan a level
carries, after which the polygons are re-derived. This module is what is left:
the entry point the population calls, and the rule that a level without a plan
cannot be bred from.

The global-RNG guard is gone with the old operators. They borrowed shape
samplers from `random_map` that drew from the `random` module directly, so a
mutation was only reproducible if the global stream happened to sit in the
same place, and the fix was to seed and restore it around every call. The plan
operators take their generator as an argument and touch nothing global, so
reproducibility now follows from the signature.
"""

from __future__ import annotations

import random
from typing import Optional

from config import UED_MUTATE_MAX_TRIES
from ued.level import Level


class MutationFailed(RuntimeError):
    """Raised when no valid child was produced within the retry budget."""


def mutate_level(
    parent: Level,
    rng: Optional[random.Random] = None,
    n_ops: Optional[int] = None,
    max_tries: int = UED_MUTATE_MAX_TRIES,
) -> Level:
    """Breed one valid child from `parent`.

    Raises MutationFailed if the retry budget is exhausted, which the caller
    should treat as "skip breeding this episode" rather than as an error.
    """
    from citygen.mutate import mutate_plan

    rng = rng or random.Random()
    if parent.plan is None:
        # Levels from a real OSM crop have no plan. They are held out rather
        # than trained on, and there is no sound way to mutate one anyway:
        # editing polygons traced from a real city produces something that is
        # neither that city nor a generated layout.
        raise MutationFailed(
            f"level {parent.level_id} has no street plan to mutate "
            f"(generator={parent.generator!r})")

    exits = [[tuple(p) for p in ring] for ring in parent.exits]
    child_plan, ops = mutate_plan(parent.plan, exits, rng, n_ops=n_ops,
                                  max_tries=max_tries)
    if child_plan is None:
        raise MutationFailed(
            f"no playable child of level {parent.level_id} in {max_tries} tries")

    rings, _ = child_plan.render()
    return parent.child(
        obstacles=[[list(p) for p in ring] for ring in rings],
        exits=exits,
        crowd_size=parent.crowd_size,
        ops=ops,
        width=child_plan.width,
        height=child_plan.height,
        plan=child_plan,
    )


def is_playable(level: Level) -> bool:
    """Validate a level that did not come from `mutate_level`.

    A level with a plan is checked the way the robot experiences it: shrink
    the free space by the robot's radius and require what remains to be
    connected and to reach an exit. A level without one, a real OSM crop, is
    checked on its rendered polygons through the same test, which is why the
    check takes geometry rather than a plan.
    """
    from citygen.plan import CityPlan
    from citygen.validate import check

    exits = [[tuple(p) for p in ring] for ring in level.exits]
    if level.plan is not None:
        ok, _ = check(level.plan, exits)
        return ok

    # No plan: wrap the rendered obstacles in a plan-shaped object so the same
    # validator applies. It has no streets, so `render` would call it an empty
    # field; the rings are supplied directly instead.
    plan = CityPlan(width=int(level.width), height=int(level.height),
                    morphology="unknown", development=1.0)
    rings = [[list(p) for p in ring] for ring in level.obstacles]
    plan.render = lambda simplify_m=None, _r=rings: (_r, None)  # type: ignore
    ok, _ = check(plan, exits)
    return ok
