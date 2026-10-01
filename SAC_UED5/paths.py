"""Where the project's data lives.

Modules moved into `sim/`, `viz/`, `learn/` and `cli/` during the layout
tidy-up, and several of them located data by walking up from their own
`__file__`. That walk is now one level deeper, so the images folder, the
numbered maps and the validation results all pointed at directories that do
not exist. One definition of the root instead, imported by anything that
needs to find a file rather than a module.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def at(*parts: str) -> str:
    """A path inside the project, e.g. at("images", "robot.png")."""
    return os.path.join(ROOT, *parts)
