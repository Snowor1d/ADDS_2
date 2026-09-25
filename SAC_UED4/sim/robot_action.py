"""The robot's action: where it moves, and what it signals.

One definition, because four places act on it. The worker applies it during
training, the zero-shot evaluator applies it, the human-play script builds it
from the keyboard, and the viewer draws it. Before the signal existed the
action was two numbers and duplicating that was harmless; a structured action
duplicated four times is how the trainer and the environment come to disagree
about what the policy chose.

Layout, with USE_DIRECT (seven numbers):

    0:2   where to move, as a direction in [-2, 2] per axis
    2:4   the heading to signal, same range, only read in "direct" mode
    4:7   a one-hot over config.ROBOT_MODES ("off", "guide", "direct")

and without it, the default (four numbers):

    0:2   where to move
    2:4   a one-hot over ("off", "guide")

The heading exists only for "direct", so it is dropped with it rather than
left as two numbers the policy would have to learn to ignore. CONT_DIM is how
many of the numbers are continuous (the squashed Gaussian part of the
policy); the rest are the mode.

The mode is one-hot rather than an index because it is fed to a critic as part
of the action vector, and an index would tell the network that the modes are
ordered.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from config import ROBOT_MODES

HAS_SIGNAL = "direct" in ROBOT_MODES
MOVE = slice(0, 2)
SIGNAL = slice(2, 4) if HAS_SIGNAL else None
CONT_DIM = 4 if HAS_SIGNAL else 2
MODE = slice(CONT_DIM, CONT_DIM + len(ROBOT_MODES))
ACTION_DIM = CONT_DIM + len(ROBOT_MODES)


def encode(move: Sequence[float], mode: str,
           signal: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Build an action vector from its parts."""
    if mode not in ROBOT_MODES:
        raise ValueError(f"unknown robot mode {mode!r}; expected one of "
                         f"{ROBOT_MODES}")
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[MOVE] = (float(move[0]), float(move[1]))
    if SIGNAL is not None:
        out[SIGNAL] = (float(signal[0]), float(signal[1]))
    out[MODE][ROBOT_MODES.index(mode)] = 1.0
    return out


def decode(vec: Sequence[float]) -> Tuple[Tuple[float, float], str,
                                          Tuple[float, float]]:
    """Split an action vector into (move, mode, signalled heading)."""
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[0] < ACTION_DIM:
        # A two-number action, from before the signal existed. Read as a plain
        # move with no signal, so old scripts and saved trajectories still
        # mean something.
        return (float(v[0]), float(v[1])), "off", (0.0, 0.0)
    mode = ROBOT_MODES[int(np.argmax(v[MODE]))]
    heading = ((float(v[2]), float(v[3])) if SIGNAL is not None
               else (0.0, 0.0))
    return (float(v[0]), float(v[1])), mode, heading


def apply_to(robot, vec):
    """Apply an action vector to one robot. Returns the move it accepted."""
    move, mode, signal = decode(vec)
    robot.set_signal(mode, signal[0], signal[1])
    return robot.receive_action([move[0], move[1]])


def random_action(rng=None, scale: float = 2.0) -> np.ndarray:
    """A uniformly random action, for the exploration phase.

    The mode is drawn uniformly too. Exploring the movement while leaving the
    signal fixed would leave the policy no experience of what the modes do,
    and the mode is the half of the action the crowd responds to.
    """
    import random as _random

    rng = rng or _random
    move = (rng.uniform(-scale, scale), rng.uniform(-scale, scale))
    signal = (rng.uniform(-scale, scale), rng.uniform(-scale, scale))
    mode = ROBOT_MODES[rng.randrange(len(ROBOT_MODES))]
    return encode(move, mode, signal)
