"""Compatibility re-export of the configuration files.

The settings live in configs/environment.py, configs/simulation_run.py and
the three files under configs/training/, each owned by exactly one of them;
see configs/__init__.py. This module exists so the many `from config import
*` callers keep working while they move to the ResolvedConfig object an
entry point builds with `configs.resolve_config()`. Do not define settings
here.
"""

from configs.environment import *        # noqa: F401,F403
from configs.training.common import *    # noqa: F401,F403
from configs.training.ued import *       # noqa: F401,F403
from configs.training.dataset import *   # noqa: F401,F403
from configs.simulation_run import *     # noqa: F401,F403
