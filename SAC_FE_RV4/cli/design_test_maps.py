"""Design the hand-made-style test maps 1505, 1506, 1508, 1509, 1510.

    python3 -m cli.design_test_maps            # check and draw only
    python3 -m cli.design_test_maps --write    # write map_infos/map_<n>.json

They complete the unknown test set 1500-1510 next to the hand-drawn
1500-1504 and 1507: 100 m square, building-like layouts (blocks against the
outer wall, large free-standing buildings with rounded or curved outlines,
small kiosks), two 4 x 8 m exits flush with the outer wall, obstacle density
0.3-0.5.

Before writing, every map must pass ued.level.check_level (all floor
reachable from an exit with a 0.5 m body clearance), differ from every
training source map (LEAKAGE_SOURCE_MAPS) and from the other test maps by at
least MIN_DIFF of their pixels under all eight symmetries, and keep integer
vertices (the map loader truncates to int). The old map_1506.json (160 m)
is moved to map_infos/archive/ instead of being overwritten.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXISTING = (1500, 1501, 1502, 1503, 1504, 1507)
NEW = (1505, 1506, 1508, 1509, 1510)
MIN_DIFF = 0.10


def rounded(poly, r):
    """Round the convex corners by r metres."""
    return poly.buffer(-r, join_style=1).buffer(r, join_style=1)


def chaikin(points, n=2):
    """Smooth a closed outline (curved walls like the editor's)."""
    pts = np.asarray(points, float)
    for _ in range(n):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        pts = np.empty((2 * len(q), 2))
        pts[0::2] = q
        pts[1::2] = r
    return Polygon(pts)


def exit_rect(side, start, along=8, depth=4, W=100, H=100):
    if side == "left":
        return [(0, start), (depth, start), (depth, start + along), (0, start + along)]
    if side == "right":
        return [(W - depth, start), (W, start), (W, start + along), (W - depth, start + along)]
    if side == "bottom":
        return [(start, 0), (start + along, 0), (start + along, depth), (start, depth)]
    return [(start, H - depth), (start + along, H - depth), (start + along, H), (start, H)]


def designs():
    D = {}

    # 1505  campus quad: wall-attached wings left and top, a rounded hall in
    # the middle, a curved lecture building lower right, a pavilion, a kiosk.
    D[1505] = dict(
        obstacles=[
            box(0, 8, 20, 50),
            rounded(box(22, 80, 76, 104), 4).intersection(box(0, 0, 100, 100)),
            rounded(box(30, 36, 64, 66), 6),
            chaikin([(60, 4), (94, 4), (94, 34), (76, 40), (62, 28)], 2),
            box(80, 54, 92, 68),
            chaikin([(16, 64), (24, 62), (26, 70), (18, 72)], 2),
        ],
        exits=[exit_rect("left", 56), exit_rect("right", 42)])

    # 1506  L-shaped office block along the top-left, a long curved building
    # bottom right, a row of parked blocks, a round annex; exits top and bottom.
    D[1506] = dict(
        obstacles=[
            box(0, 64, 50, 100).union(box(0, 40, 20, 100)),
            chaikin([(36, 6), (86, 6), (94, 22), (90, 46), (58, 44), (40, 30)], 2),
            box(28, 52, 38, 58), box(46, 52, 56, 58), box(64, 52, 74, 58),
            chaikin([(64, 68), (88, 66), (94, 84), (70, 90)], 2),
            box(0, 0, 12, 28),
        ],
        exits=[exit_rect("top", 60), exit_rect("bottom", 20)])

    # 1508  two long buildings with a street between them, a round pavilion,
    # wall blocks on the right, a kiosk; exits left and top.
    D[1508] = dict(
        obstacles=[
            rounded(box(14, 58, 72, 78), 3),
            rounded(box(14, 26, 66, 46), 3),
            Point(86, 50).buffer(9, 16),
            box(86, 76, 100, 100), box(80, 0, 100, 20),
            chaikin([(30, 4), (48, 2), (52, 16), (34, 18)], 2),
            box(4, 84, 16, 94),
        ],
        exits=[exit_rect("left", 48), exit_rect("top", 30)])

    # 1509  courtyard ring: four buildings round an open court, corner gaps
    # of 8 m or more (a 4 m gap trapped walkers on the navmesh), wall blocks
    # in two corners; exits right and bottom.
    D[1509] = dict(
        obstacles=[
            rounded(box(18, 62, 56, 80), 2),
            rounded(box(64, 36, 80, 72), 2),
            rounded(box(30, 14, 64, 28), 2),
            rounded(box(10, 34, 24, 54), 2),
            box(0, 84, 16, 100),
            box(82, 82, 100, 100),
            box(0, 0, 14, 14),
        ],
        exits=[exit_rect("right", 18), exit_rect("bottom", 72)])

    # 1510  shopping street: a large building with a curved front, a row of
    # shops on the bottom wall with 14 m wide gaps (narrow dead-end alleys
    # trapped walkers), a fountain, wall blocks; exits left and right.
    D[1510] = dict(
        obstacles=[
            chaikin([(24, 50), (80, 50), (88, 66), (80, 88), (52, 90), (24, 84)], 2),
            box(6, 0, 20, 16), box(34, 0, 48, 16), box(62, 0, 76, 16),
            Point(50, 34).buffer(6, 16),
            box(0, 86, 16, 100),
            box(88, 26, 100, 42),
        ],
        exits=[exit_rect("left", 40), exit_rect("right", 70)])
    return D


def to_points(P):
    """Integer vertices, closing point dropped, collinear runs merged."""
    P = P.simplify(0.4, preserve_topology=True)
    pts = [[int(round(x)), int(round(y))] for x, y in list(P.exterior.coords)[:-1]]
    out = []
    for p in pts:
        if not out or out[-1] != p:
            out.append(p)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def build():
    maps = {}
    for n, d in designs().items():
        obs = []
        for P in d["obstacles"]:
            P = P.intersection(box(0, 0, 100, 100))
            pts = to_points(P)
            if not Polygon(pts).is_valid:
                raise SystemExit(f"map {n}: an obstacle became invalid after rounding")
            obs.append(pts)
        maps[n] = {"width": 100, "height": 100, "obstacles": obs,
                   "exits": [[list(p) for p in e] for e in d["exits"]]}
    return maps


def main(argv=None):
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--png", default=None)
    args = ap.parse_args(argv)
    from configs import cfg
    import sim_maps
    from evaluation.suites import LeakageIndex, d4_rasters, raster
    from ued.level import Level, check_level
    from ued.render import render_levels

    maps = build()
    leak = LeakageIndex(sorted(set(cfg.LEAKAGE_SOURCE_MAPS)), cfg)
    others = {m: sim_maps.map_geometry(m, cfg) for m in EXISTING}
    imgs = {m: raster(g["obstacles"], g["exits"], 100, 100) for m, g in others.items()}
    ok_all = True
    for n, g in maps.items():
        lv = Level(n, 100, 100, g["obstacles"], [[tuple(p) for p in e] for e in g["exits"]], 30)
        ok, why = check_level(lv, max_density=0.55)
        img = raster(g["obstacles"], g["exits"], 100, 100)
        d_train, who = leak.nearest(img)
        d_test = min(min((v != imgs[m]).mean() for v in d4_rasters(img)) for m in imgs)
        dens = lv.obstacle_density()
        flag = ok and d_train >= MIN_DIFF and d_test >= MIN_DIFF and 0.25 <= dens <= 0.55
        ok_all &= flag
        print(f"map {n}: valid={ok} ({why}) density={dens:.2f} obstacles={len(g['obstacles'])} "
              f"nearest training {d_train:.3f} (map {who}), nearest test {d_test:.3f} "
              f"-> {'OK' if flag else 'REJECT'}")
        imgs[n] = img
    levels = [Level(m, 100, 100, g["obstacles"], [[tuple(p) for p in e] for e in g["exits"]], 30)
              for m, g in sorted({**others, **maps}.items())]
    png = args.png or os.path.join(ROOT, "docs", "test_maps_1500_1510.png")
    render_levels(levels, png, [f"map {lv.level_id}" for lv in levels], cols=4)
    print(png)
    if not args.write:
        return
    if not ok_all:
        raise SystemExit("not written: a map failed its checks")
    md = os.path.join(ROOT, "map_infos")
    old = os.path.join(md, "map_1506.json")
    if os.path.exists(old) and json.load(open(old)).get("width") != 100:
        os.makedirs(os.path.join(md, "archive"), exist_ok=True)
        dst = os.path.join(md, "archive", "map_1506_160m.json")
        shutil.move(old, dst)
        print(f"moved the old 160 m map 1506 to {dst}")
    for n, g in maps.items():
        path = os.path.join(md, f"map_{n}.json")
        if os.path.exists(path) and json.load(open(path)) != g:
            raise SystemExit(f"{path} exists with other content; not overwriting")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(g, fh)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
