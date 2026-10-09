"""Unsupervised environment design for the crowd-evacuation guidance task.

The package holds everything the curriculum needs and nothing the simulator
needs, so `model.py` only ever sees a plain `Level` and never imports the
population or the scoring code.
"""

from ued.level import Level, level_from_map_data, generate_random_level
from ued.mutate import mutate_level, MutationFailed
from ued.population import LevelPopulation
from ued.runner import UEDRunner, make_value_fn

__all__ = [
    "Level",
    "level_from_map_data",
    "generate_random_level",
    "mutate_level",
    "MutationFailed",
    "LevelPopulation",
    "UEDRunner",
    "make_value_fn",
]
