"""Summarise cli.trace_episode outputs: where the evacuation stalls and what
the robot, the policy and the remaining people do from then on.

    python3 -m cli.analyze_traces ~/RV4_eval/traces/augL-ep3800_fail [--png]

Window: a failed episode from its last evacuation to the step cap (the
stall); a successful one over its last 25 %. Writes <dir>/analysis.json
(one entry per scenario) and, with --png, <id>_stall.png (map with the
robot path in the window and the remaining people; counts over time).

Per scenario:
    stall_s            time from the last evacuation to the cap
    followers_mean     people following the robot (type 0) in the window
    robot_speed        m/s, from position change
    straight20s        net displacement / path length over 20 s (1 = straight)
    cells10m           distinct 10 m cells the robot visits in the window
    std_u              policy pre-squash std (exp(0.5) = 1.65 is the cap)
    dir_R              length of the mean unit vector of the policy's mean
                       action: 0 = no preferred direction, 1 = always the same
    cos_euclid         mean cosine between the mean action and the straight
                       direction to the nearest exit
    cos_geodesic       same for the walking direction (descent of the
                       geodesic distance field, 0.5 m grid, walls respected)
    robot_euclid_mean, robot_geo_mean   robot's straight / walking distance
                       to the nearest exit (m); a large gap = behind a wall
Per remaining person: type at the end, share of the window spent following
the robot, distances to the exit (straight / walking) and to the robot.
"""

import heapq
import json
import os
import sys

import numpy as np

RES = 0.5   # m per cell of the geodesic field


def geodesic_field(row):
    """Walking distance (m) to the nearest exit on a RES grid, 8-connected,
    obstacles blocked; inf where unreachable. Indexed [x_cell, y_cell]."""
    from matplotlib.path import Path
    W, H = row["width"], row["height"]
    nx, ny = int(round(W / RES)), int(round(H / RES))
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * RES, (np.arange(ny) + 0.5) * RES, indexing="ij")
    pts = np.c_[X.ravel(), Y.ravel()]
    free = np.ones(nx * ny, bool)
    for poly in row["obstacles"]:
        free &= ~Path(np.array(poly)).contains_points(pts, radius=1e-9)
    goal = np.zeros(nx * ny, bool)
    for poly in row["exits"]:
        goal |= Path(np.array(poly)).contains_points(pts, radius=0.3)
    free = free.reshape(nx, ny)
    goal = goal.reshape(nx, ny) & free
    dist = np.full((nx, ny), np.inf)
    pq = [(0.0, i, j) for i, j in zip(*np.where(goal))]
    for _, i, j in pq:
        dist[i, j] = 0.0
    heapq.heapify(pq)
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        for di, dj, w in steps:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b] and d + w * RES < dist[a, b]:
                dist[a, b] = d + w * RES
                heapq.heappush(pq, (dist[a, b], a, b))
    return dist


def _cells(field, xy):
    xy = np.atleast_2d(xy)
    i = np.clip((xy[:, 0] / RES).astype(int), 0, field.shape[0] - 1)
    j = np.clip((xy[:, 1] / RES).astype(int), 0, field.shape[1] - 1)
    return i, j


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _exit_vectors(row, xy):
    """Vector from each point to the nearest point of the nearest exit polygon."""
    from shapely.geometry import Point, Polygon
    polys = [Polygon(p) for p in row["exits"]]
    out = []
    for q in np.atleast_2d(xy):
        p = Point(*q)
        best = min(polys, key=lambda pl: pl.distance(p))
        if best.distance(p) == 0:
            out.append([0.0, 0.0])
            continue
        c = best.exterior.interpolate(best.exterior.project(p))
        out.append([c.x - q[0], c.y - q[1]])
    return np.array(out)


def analyse_one(row, z):
    dt = 0.5
    alive = z["alive"].astype(int)
    T = len(alive)
    total = int(z["crowd_out"].shape[1])
    drops = np.where(np.diff(np.r_[total, alive]) < 0)[0]
    t_last = int(drops[-1]) if len(drops) else 0
    s0 = t_last if not row["success"] else int(T * 0.75)
    field = geodesic_field(row)
    rxy = z["robot_xy"].astype(np.float64)
    spd = np.r_[0.0, np.linalg.norm(np.diff(rxy, axis=0), axis=1) / dt]
    W, H = row["width"], row["height"]
    rw = rxy[s0:]
    k = z["dec_step"].astype(int)
    sel = k >= s0
    k0 = int(np.argmax(sel)) if sel.any() else len(k)
    dxy = rxy[np.clip(k[sel] - 1, 0, T - 1)]

    # policy direction against the exit directions
    gx, gy = np.gradient(np.where(np.isfinite(field), field, np.nan), RES)
    ii, jj = _cells(field, dxy)
    geo_dir = -_unit(np.c_[gx[ii, jj], gy[ii, jj]])
    eu_dir = _unit(_exit_vectors(row, dxy))
    u = _unit(z["mean_action"][sel][:, :2].astype(np.float64))   # direction only (waypoint: + speed)
    ok = np.isfinite(geo_dir).all(1)

    L = 40   # 20 s
    if len(rw) > L:
        net = np.linalg.norm(rw[L:] - rw[:-L], axis=1)
        path = np.convolve(np.r_[0, np.linalg.norm(np.diff(rw, axis=0), axis=1)], np.ones(L), "valid")[:len(net)]
        straight = float(np.mean(net / np.maximum(path, 1e-6)))
    else:
        straight = float("nan")
    rg = field[_cells(field, rw)]
    cxy, ctype, cout = z["crowd_xy"], z["crowd_type"], z["crowd_out"]
    people = []
    for i in np.where(~cout[-1])[0]:
        p = cxy[k0:, i]
        dr = np.linalg.norm(p - dxy, axis=1) if len(p) == len(dxy) and len(p) else np.array([np.nan])
        people.append(dict(
            type_end=int(ctype[-1, i]),
            frac_following=float(np.mean(ctype[k0:, i] == 0)) if len(p) else float("nan"),
            moved_m=float(np.linalg.norm(p[-1] - p[0])) if len(p) else float("nan"),
            exit_euclid=float(np.linalg.norm(_exit_vectors(row, cxy[-1, i])[0])),
            exit_geo=float(field[_cells(field, cxy[-1, i])][0]),
            min_robot_d=float(np.nanmin(dr)), frac_within10=float(np.nanmean(dr < 10)),
            xy_end=[round(float(v), 1) for v in cxy[-1, i]]))
    return dict(
        id=row["id"], map_key=row["map_key"], success=bool(row["success"]),
        left=int(alive[-1]), t100_s=row["evac_time_100"] * dt, t_last_evac_s=t_last * dt,
        stall_s=0.0 if row["success"] else (T - t_last) * dt,
        followers_mean=float(z["followers"][s0:].mean()), followers_max=int(z["followers"][s0:].max()),
        robot_speed=float(spd[s0:].mean()), straight20s=straight,
        cells10m=len({(int(x // 10), int(y // 10)) for x, y in rw}),
        near_edge=float(np.mean((rw[:, 0] < 3) | (rw[:, 0] > W - 3) | (rw[:, 1] < 3) | (rw[:, 1] > H - 3))),
        collisions=int(z["collision"][s0:].sum()),
        std_u=float(z["std_u"][sel].mean()) if sel.any() else float("nan"),
        dir_R=float(np.linalg.norm(u.mean(0))) if len(u) else float("nan"),
        cos_euclid=float(np.mean(np.sum(u * eu_dir, 1))) if len(u) else float("nan"),
        cos_geodesic=float(np.mean(np.sum(u[ok] * geo_dir[ok], 1))) if ok.any() else float("nan"),
        robot_euclid_mean=float(np.mean(np.linalg.norm(_exit_vectors(row, rw[::8]), axis=1))),
        robot_geo_mean=float(np.mean(rg[np.isfinite(rg)])) if np.isfinite(rg).any() else float("nan"),
        replay_ok=row.get("replay_matches_eval"), people=people), s0, k0


def draw(row, z, r, s0, k0, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    dt = 0.5
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    a = ax[0]
    for poly in row["obstacles"]:
        a.add_patch(Polygon(poly, closed=True, fc="#999", ec="#555"))
    for poly in row["exits"]:
        a.add_patch(Polygon(poly, closed=True, fc="#59a14f", ec="#2e6b2a"))
    rxy = z["robot_xy"]
    a.plot(rxy[:s0, 0], rxy[:s0, 1], color="#ccc", lw=0.6, zorder=1)
    sc = a.scatter(rxy[s0:, 0], rxy[s0:, 1], c=np.arange(len(rxy) - s0) * dt, s=1.5, cmap="viridis", zorder=2)
    plt.colorbar(sc, ax=a, fraction=0.04, label="s into the window")
    cxy, cout = z["crowd_xy"], z["crowd_out"]
    for i in np.where(~cout[-1])[0]:
        a.plot(cxy[k0:, i, 0], cxy[k0:, i, 1], color="#e15759", lw=0.8, alpha=0.7)
        a.scatter(cxy[-1, i, 0], cxy[-1, i, 1], marker="x", color="#e15759", s=40, zorder=3)
    a.set_xlim(0, row["width"])
    a.set_ylim(0, row["height"])
    a.set_aspect("equal")
    a.set_title(f"{r['id']}  left={r['left']}  window from {s0 * dt:.0f}s", fontsize=9)
    b = ax[1]
    t = np.arange(len(z["alive"])) * dt
    b.plot(t, z["alive"], label="alive", color="k")
    b.plot(t, z["followers"], label="following robot", color="#ffa500")
    b.plot(t, z["my_way"], label="my-way (type 1)", color="#4e79a7", lw=0.8)
    b.plot(t, z["neighbor_followers"], label="neighbor-following (type 2)", color="#76b7b2", lw=0.8)
    spd = np.r_[0.0, np.linalg.norm(np.diff(rxy, axis=0), axis=1) / dt]
    b.plot(t, np.convolve(spd, np.ones(40) / 40, "same") * 5, label="robot speed x5 (20 s avg)",
           color="#e15759", lw=0.7)
    b.axvline(s0 * dt, color="gray", ls="--", lw=0.8)
    b.legend(fontsize=7, loc="upper right")
    b.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    png = "--png" in argv
    d = os.path.expanduser([a for a in argv if not a.startswith("--")][0])
    rows = {}
    with open(os.path.join(d, "summary.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["id"]] = r          # last row wins on repeated runs
    out = []
    for i, row in rows.items():
        name = i.replace("/", "__")
        z = np.load(os.path.join(d, name + ".npz"))
        r, s0, k0 = analyse_one(row, z)
        out.append(r)
        if png:
            draw(row, z, r, s0, k0, os.path.join(d, name + "_stall.png"))
    keys = ["left", "stall_s", "followers_mean", "robot_speed", "straight20s", "cells10m", "std_u",
            "dir_R", "cos_euclid", "cos_geodesic", "robot_euclid_mean", "robot_geo_mean", "collisions"]
    print(f"{'id':20s} ok " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for r in sorted(out, key=lambda x: x["id"]):
        print(f"{r['id']:20s} {int(r['success'])}  " + " ".join(f"{r[k]:9.2f}" for k in keys))
    with open(os.path.join(d, "analysis.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
