"""Compatibility shim: the settings live in configs/ (see configs/__init__.py).

`from config import *` keeps working for the simulator and the older
scripts; it exports the merged, validated settings including any
RV4_PRESET / RV4_OVERRIDES. Define nothing here.
"""
from configs import *  # noqa: F401,F403
from configs import cfg  # noqa: F401
