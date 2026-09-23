"""Synthetic geometries for the verification tests.

Built as ordinary Levels with hand-written obstacle polygons and no hazard, so
they run through exactly the same simulator, navmesh and social force as a
training episode. A scenario that reimplemented the movement would verify the
reimplementation.
"""

from __future__ import annotations

from typing import List, Optional

from ued.level import Level


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def corridor_level(length: float = 60.0, width: float = 5.0,
                   margin: float = 6.0) -> Level:
    """A straight corridor of the given clear width, walled above and below.

    The world is larger than the corridor so the walls are real obstacles
    rather than the map boundary, which carries its own repulsion.
    """
    W = length + 2 * margin
    H = width + 2 * margin
    obstacles = [
        _rect(0, 0, W, margin),
        _rect(0, margin + width, W, H),
    ]
    return Level(obstacles=obstacles, exits=[], crowd_size=1,
                 width=int(W), height=int(H), augmentation="identity",
                 generator="validation")


def corridor_band(level: Level, width: float = 5.0, margin: float = 6.0):
    """The clear band of a corridor level as (y_low, y_high)."""
    return margin, margin + width


def bottleneck_level(room: float = 20.0, door_width: float = 1.2,
                     corridor: float = 12.0, margin: float = 4.0) -> Level:
    """A square room whose only opening is a door of the given width.

    The door leads into a short corridor, so the flow being measured is
    through a real opening rather than into open space, which is what the
    published specific-flow figures are measured on.
    """
    W = margin + room + corridor + margin
    H = margin + room + margin
    x_wall = margin + room
    y_mid = margin + room / 2.0
    half = door_width / 2.0
    obstacles = [
        # room walls
        _rect(0, 0, W, margin),                       # bottom
        _rect(0, margin + room, W, H),                # top
        _rect(0, 0, margin, H),                       # left
        # the wall with the door in it, as two pieces
        _rect(x_wall, margin, x_wall + 1.0, y_mid - half),
        _rect(x_wall, y_mid + half, x_wall + 1.0, margin + room),
    ]
    return Level(obstacles=obstacles, exits=[], crowd_size=1,
                 width=int(W), height=int(H), augmentation="identity",
                 generator="validation")


def bottleneck_geometry(room: float = 20.0, door_width: float = 1.2,
                        margin: float = 4.0):
    """Where the door is, for the measurement: (x_wall, y_mid, door_width)."""
    return margin + room, margin + room / 2.0, door_width
