"""Walking distance to exits (exit_distance.py, config.DANGER_DISTANCE)."""

import math
import os
import pathlib
import random
import sys
import time

import numpy as np
import pytest

PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
os.chdir(PROJECT_DIR)

from exit_distance import ExitDistanceField  # noqa: E402

EXIT_RIGHT = [[[96.0, 45.0], [100.0, 45.0], [100.0, 55.0], [96.0, 55.0]]]


def _straight(x, y, exits=EXIT_RIGHT):
    from shapely.geometry import Point, Polygon
    return min(Polygon(e).distance(Point(x, y)) for e in exits)


def test_open_space_matches_straight_line():
    f = ExitDistanceField([], EXIT_RIGHT, 100, 100, 0.5)
    rng = np.random.default_rng(0)
    # grid metric: within 3 % in any direction, plus about half a cell
    # because exit cells are counted from their centres
    for x, y in rng.uniform(5, 90, size=(300, 2)):
        s = _straight(x, y)
        assert abs(f(x, y) - s) <= 0.03 * s + 0.5, (x, y)


def test_detour_around_wall_end():
    wall = [[[50.0, 0.0], [51.0, 0.0], [51.0, 80.0], [50.0, 80.0]]]
    f = ExitDistanceField(wall, EXIT_RIGHT, 100, 100, 0.5)
    p = (40.0, 50.0)
    exact = math.hypot(50 - 40, 80 - 50) + 1.0 + math.hypot(96 - 51, 80 - 55)
    assert exact > 1.4 * _straight(*p)
    assert abs(f(*p) - exact) <= 0.03 * exact + 0.5


def test_shut_in_area_gets_the_finite_cap():
    wall = [[[0.0, 60.0], [60.0, 0.0], [60.6, 0.6], [0.6, 60.6]]]
    f = ExitDistanceField(wall, EXIT_RIGHT, 100, 100, 0.5)
    assert f.unreachable_free_cells > 0
    reach_max = max(f(x, y) for x in range(35, 95, 5) for y in range(35, 95, 5))
    d = f(10.0, 10.0)
    assert d == f.unreachable
    assert reach_max < d < 1e5


def test_no_exit_is_infinite():
    assert ExitDistanceField([], [], 100, 100, 0.5)(50.0, 50.0) == math.inf


def test_exit_smaller_than_a_cell_still_counts():
    tiny = [[[99.8, 49.9], [100.0, 49.9], [100.0, 50.1], [99.8, 50.1]]]
    f = ExitDistanceField([], tiny, 100, 100, 0.5)
    assert f.unreachable_free_cells == 0
    assert abs(f(50.0, 50.0) - 49.8) <= 0.03 * 49.8 + 0.5


def test_build_time():
    t0 = time.time()
    ExitDistanceField([], EXIT_RIGHT, 200, 200, 0.5)
    assert time.time() - t0 < 5.0


def _env(map_num=1504, seed=3):
    import model
    random.seed(seed)
    np.random.seed(seed)
    return model.FightingModel(30, 100, 100, model_num=map_num, robot="Q")


def test_euclidean_default_is_rv3():
    import model
    assert model.DANGER_DISTANCE == "euclidean"
    env = _env()
    for _ in range(3):
        env.step()
    for a in env.crowds:
        if not a.dead:
            assert env.danger_distance(a.xy) == pytest.approx(env.distance_to_exit(a.xy), abs=1e-9)


def test_geodesic_in_the_map_1504_dead_end(monkeypatch):
    import model
    monkeypatch.setattr(model, "DANGER_DISTANCE", "geodesic")
    env = _env()
    env.step()
    a = next(a for a in env.crowds if not a.dead)
    a.xy = [40.5, 97.1]
    a.vel = [0.0, 0.0]
    env.step()
    assert env.distance_to_exit(a.xy) == pytest.approx(21.5, abs=0.6)
    assert 50.0 < a.danger < 62.0                    # around the wall: about 56 m
    # danger is taken before the agent moves in the step (< 1 m)
    assert abs(a.danger - env.danger_distance(a.xy)) < 1.0
    tot = -sum(x.danger for x in env.crowds if x.type in (0, 1, 2) and not x.dead)
    assert env.reward_based_all_agents_danger() == pytest.approx(tot, abs=1e-9)


def test_bad_setting_is_rejected(monkeypatch):
    import model
    monkeypatch.setattr(model, "DANGER_DISTANCE", "manhattan")
    env = _env()
    with pytest.raises(ValueError):
        env.danger_distance((50.0, 50.0))
