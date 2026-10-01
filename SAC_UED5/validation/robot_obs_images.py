"""Draw what one robot actually receives as input, next to the scene.

    python3 -m validation.robot_obs_images --sites soho_nyc hongdae mitte

For each site: robots guide toward the hazard for a while, then the actor
input of robot 0 at one decision instant is drawn channel by channel, with
the world at that moment and the ego and mid windows marked on it. The
arrays are exactly what build_observations returns, the same call the
trainer uses.
"""

from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np


def capture(site: str, decision: int, seed: int = 1):
    from configs import resolve_config
    from learn.rollout import run_episode
    from sim import robot_action as ra
    from validation.behaviour import _level
    import sim.model as M

    cfg = resolve_config(check_data=False)
    lv = _level(cfg, site, seed, 0.5)
    random.seed(seed)
    np.random.seed(seed)
    model = M.FightingModel(int(lv.crowd_size), lv.width, lv.height,
                            robot="Q", level=lv)
    z = model.danger_zone
    grabbed = {}

    def act(obs, rec):
        k = len(grabbed.get("seen", []))
        grabbed.setdefault("seen", []).append(k)
        if k == decision:
            grabbed["obs"] = {key: v.copy() for key, v in obs.items()}
            grabbed["robots"] = [(float(rb.xy[0]), float(rb.xy[1]), rb.mode)
                                 for rb in model.robots]
            grabbed["people"] = [(float(a.xy[0]), float(a.xy[1]), a.awareness,
                                  a.type) for a in model.crowds
                                 if not a.dead and a.type != 3]
            grabbed["step"] = int(model.step_count)
        rows = []
        for rb in model.robots:
            dx, dy = float(z.cx) - rb.xy[0], float(z.cy) - rb.xy[1]
            d = math.hypot(dx, dy)
            mv = (dx / d, dy / d) if d > 3 else (0.0, 0.0)
            rows.append(ra.encode(mv, "guide"))
        return np.stack(rows)

    run_episode(model, cfg, act, gamma=0.99,
                max_steps=(decision + 1) * int(cfg.ACTION_SCALE), needs_obs=True)
    return cfg, model, lv, grabbed


def draw(site: str, out_dir: str, decision: int = 40):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon, Rectangle
    from sim.observation import (EGO_DYNAMIC, EGO_STATIC, GLOBAL_DYNAMIC,
                                 GLOBAL_STATIC, MID_DYNAMIC, MID_STATIC,
                                 own_state_layout, teammate_state_layout)

    cfg, model, lv, g = capture(site, decision)
    obs = g["obs"]
    r = 0
    ego, mid, glob, state = (obs["ego"][r], obs["mid"][r], obs["glob"][r],
                             obs["state"][r])
    W, H = float(model.width), float(model.height)
    E, Mz = int(cfg.EGO_MAP_SIZE), int(cfg.OBS_MID_SIZE)
    res_m = float(cfg.OBS_MID_RES_M)
    rx, ry, _ = g["robots"][r]

    fig = plt.figure(figsize=(26, 24))
    gs = fig.add_gridspec(5, 9, height_ratios=[1, 1, 1, 1, 1.05],
                          hspace=0.35, wspace=0.25)
    fig.suptitle(f"{site} 200 m - robot 0 actor input at decision {decision} "
                 f"(step {g['step']}), {len(g['robots'])} robots, guide mode",
                 fontsize=16)

    # --- world
    ax = fig.add_subplot(gs[0:2, 0:4])
    for poly in lv.obstacles:
        ax.add_patch(Polygon(poly, closed=True, color="#444"))
    zone = model.danger_zone
    ax.add_patch(Circle((zone.cx, zone.cy), getattr(zone, "radius", 0),
                        color="#d62728", alpha=0.25))
    colors = {"unaware": "#1f77b4", "milling": "#ff7f0e", "acting": "#2ca02c",
              "nonresponsive": "#9467bd"}
    for aw, col in colors.items():
        pts = [(x, y) for x, y, a, t in g["people"] if a == aw]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=4, c=col, label=f"{aw} ({len(pts)})")
    fol = [(x, y) for x, y, a, t in g["people"] if t == 0]
    if fol:
        xs, ys = zip(*fol)
        ax.scatter(xs, ys, s=14, facecolors="none", edgecolors="k",
                   linewidths=0.6, label=f"following a robot ({len(fol)})")
    for i, (x, y, mode) in enumerate(g["robots"]):
        ax.scatter([x], [y], s=120, marker="*",
                   c="red" if i == r else "orange", edgecolors="k", zorder=5)
        ax.annotate(f"R{i}", (x, y), xytext=(4, 4), textcoords="offset points")
    ax.add_patch(Rectangle((math.floor(rx) - E // 2, math.floor(ry) - E // 2),
                           E, E, fill=False, ec="red", lw=1.5, label=f"ego {E} m"))
    m0x = (math.floor(rx / res_m) - Mz // 2) * res_m
    m0y = (math.floor(ry / res_m) - Mz // 2) * res_m
    ax.add_patch(Rectangle((m0x, m0y), Mz * res_m, Mz * res_m, fill=False,
                           ec="purple", lw=1.5, ls="--", label="mid 128 m"))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title("world (true state, not observed)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              markerscale=2)

    def panel(spec, img, title, vmax=None, cmap="viridis"):
        a = fig.add_subplot(spec)
        im = a.imshow(img, origin="lower", cmap=cmap, vmin=0,
                      vmax=vmax if vmax is not None else max(1e-6, float(img.max())))
        a.set_title(title, fontsize=9)
        a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    # --- ego: static + newest and oldest frames
    hist = int(cfg.OBS_HISTORY_DECISIONS)
    ego_items = [(ego[i], f"ego {n}") for i, n in enumerate(EGO_STATIC)]
    ego_items += [(ego[3], "ego own_crowd t"), (ego[4], "ego own_observed t"),
                  (ego[3 + 2 * (hist - 1)], f"ego own_crowd t-{hist - 1}"),
                  (ego[4 + 2 * (hist - 1)], f"ego own_observed t-{hist - 1}")]
    slots = [(0, 6), (0, 7), (0, 8), (1, 5), (1, 6), (1, 7), (1, 8)]
    for (rr, cc), (img, title) in zip(slots, ego_items):
        panel(gs[rr, cc], img, title, vmax=1.0)

    # --- mid
    mid_names = list(MID_STATIC) + list(MID_DYNAMIC)
    for k, name in enumerate(mid_names):
        panel(gs[2, k], mid[k], f"mid {name}", vmax=1.0)

    # --- global
    glob_names = list(GLOBAL_STATIC) + list(GLOBAL_DYNAMIC)
    for k, name in enumerate(glob_names):
        panel(gs[3, k], glob[k], f"glob {name}", vmax=1.0)

    # --- state vector
    own = own_state_layout(cfg)
    mate = teammate_state_layout(cfg)
    names = list(own)
    for j in range(int(cfg.MAX_ROBOTS) - 1):
        names += [f"m{j + 1}.{n}" for n in mate]
    a = fig.add_subplot(gs[4, :])
    cols = []
    for i in range(len(names)):
        cols.append("#1f77b4" if i < len(own)
                    else ("#2ca02c" if ((i - len(own)) // len(mate)) % 2 == 0
                          else "#ff7f0e"))
    a.bar(range(len(names)), state, color=cols)
    for i, v in enumerate(state):
        a.text(i, v + (0.03 if v >= 0 else -0.08), f"{v:.2f}", ha="center",
               fontsize=7, rotation=90)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    a.axhline(0, color="k", lw=0.5)
    a.set_title("state vector (blue: own, green/orange: teammate slots)")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"robot_obs_{site}.png")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="*", default=["soho_nyc", "hongdae", "mitte"])
    ap.add_argument("--out", default="validation/results/robot_obs")
    ap.add_argument("--decision", type=int, default=40)
    args = ap.parse_args()
    for s in args.sites:
        print(draw(s, args.out, args.decision), flush=True)


if __name__ == "__main__":
    main()
