"""Scale of every reward term under rew-v6-pbrs, from the actual TaskReward.

States are reached with every robot on the shuttle baseline. From each state
robot 0 takes one of six actions, the others keep the shuttle action they had,
and the episode runs H steps under common random numbers (M replicates).
Per term (weighted, discounted over the H steps, robot 0's share for the
per-robot approach term): the level, the sd between actions (corrected for
replicate noise) and the replicate noise.

    python3 docs/review_20261013/scale_probe.py run LEVEL      # 0, 3, 8
    python3 docs/review_20261013/scale_probe.py report
"""
import os, sys, copy, random, json
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(k, "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import numpy as np
OUT = os.path.join(HERE, "results", "scale_probe")
T0S = (48, 160, 320)          # steps: 24 s, 80 s, 160 s
HS = (8, 24)                  # 4 s (report 5.3) and 12 s (a typical event hold)
M = 3
TEAM = ("person_time", "remaining_path", "reentry", "shape_progress")


def run(li):
    from configs import resolve_config
    from learn.zero_shot import validation_levels, build_eval_model, validation_seed
    from sim import robot_action as ra
    from sim.rewards import TaskReward
    from validation.shuttle_baseline import ShuttlePolicy, _centre
    cfg = resolve_config({"REWARD_VERSION": "rew-v6-pbrs"})
    g = cfg.gamma() ** (1.0 / int(cfg.ACTION_SCALE))
    w_appr = float(cfg.REWARD_W_SHAPE_APPROACH)
    name, lv = validation_levels(cfg)[li]
    seed = validation_seed(name)
    random.seed(seed); np.random.seed(seed % 2**32)
    m = build_eval_model(lv, seed); m.should_finish = lambda: False
    rew = TaskReward(m, cfg)
    sp = ShuttlePolicy(m, len(m.robots), cfg=cfg)
    zone = m.danger_zone
    lim = float(cfg.ROBOT_WAYPOINT_RANGE_M)
    recs, t = [], 0
    for t0 in T0S:
        while t < t0:
            if t % 8 == 0:
                for i, rb in enumerate(m.robots):
                    ra.apply_to(rb, sp._act_one(i, rb))
            m.step(); rew.step(m); t += 1
        base = [sp._act_one(i, rb) for i, rb in enumerate(m.robots)]
        rb0 = m.robots[0]; x, y = float(rb0.xy[0]), float(rb0.xy[1])

        def toward(tx, ty, speed, mode):
            dx, dy = tx - x, ty - y
            s = max(1.0, abs(dx) / lim, abs(dy) / lim)
            return ra.encode((2 * dx / (lim * s), 2 * dy / (lim * s)), mode, speed=speed)
        ex = sp._exit_mesh(m.find_mesh((x, y)))
        exc = _centre(ex) if ex is not None else (x, y)
        cands = {"shuttle": base[0],
                 "in_off": toward(zone.cx, zone.cy, 1.0, "off"),
                 "in_guide": toward(zone.cx, zone.cy, 0.6, "guide"),
                 "out_guide": toward(exc[0], exc[1], 0.6, "guide"),
                 "out_off": toward(exc[0], exc[1], 1.0, "off"),
                 "still_guide": ra.encode((0.0, 0.0), "guide", speed=0.0)}
        for rep in range(M):
            for nm, a0 in cands.items():
                mm = copy.deepcopy(m)
                rr = copy.copy(rew); rr._clear = set(rew._clear)
                rr._phi_appr = list(rew._phi_appr)
                s = 10_000 * rep + 7 + seed + t0
                random.seed(s); np.random.seed(s % 2**32)
                acts = list(base); acts[0] = a0
                for i, rb in enumerate(mm.robots):
                    ra.apply_to(rb, acts[i])
                acc = {k: 0.0 for k in TEAM + ("collision0", "shape_approach0")}
                disc = 1.0
                for k in range(max(HS)):
                    p0 = rr._phi_appr[0]
                    mm.step(); c = rr.step(mm)
                    for term in TEAM:
                        acc[term] += disc * c[term]
                    acc["collision0"] += disc * (-float(cfg.REWARD_W_COLLISION)
                                                 * int(bool(getattr(mm.robots[0], "collision_check", 0))))
                    acc["shape_approach0"] += disc * w_appr * (g * rr._phi_appr[0] - p0)
                    disc *= g
                    if k + 1 in HS:
                        recs.append({"level": name, "t0": t0, "rep": rep, "action": nm,
                                     "H": k + 1, **acc})
        print(f"[{name}] t0={t0} done", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(recs, open(os.path.join(OUT, f"L{li}.json"), "w"))


def report():
    import glob
    recs = sum((json.load(open(f)) for f in sorted(glob.glob(os.path.join(OUT, "L*.json")))), [])
    terms = TEAM + ("collision0", "shape_approach0")
    for H in HS:
        rows = [r for r in recs if r["H"] == H]
        states = sorted({(r["level"], r["t0"]) for r in rows})
        print(f"--- H = {H} steps ({H / 2:.0f} s), {len(states)} states x 6 actions x {M} reps")
        print(f"{'term':16s} {'mean':>8s} {'|mean|':>8s} {'between-action sd':>18s} {'noise sd':>9s} {'ratio':>6s}")
        tot = {"task": [], "learn": []}
        for term in terms + ("TASK", "LEARN"):
            b2, n2, lv, alv = [], [], [], []
            for st in states:
                sr = [r for r in rows if (r["level"], r["t0"]) == st]
                acts = sorted({r["action"] for r in sr})
                def val(r):
                    if term == "TASK":
                        return sum(r[k] for k in ("person_time", "remaining_path", "reentry", "collision0"))
                    if term == "LEARN":
                        return sum(r[k] for k in terms)
                    return r[term]
                X = np.array([[val(r) for r in sr if r["action"] == a] for a in acts])  # (A, M)
                n2.append(X.var(axis=1, ddof=1).mean())
                b2.append(max(0.0, X.mean(axis=1).var(ddof=1) - n2[-1] / X.shape[1]))
                lv.append(X.mean()); alv.append(abs(X.mean()))
            b, n = np.sqrt(np.mean(b2)), np.sqrt(np.mean(n2))
            print(f"{term:16s} {np.mean(lv):8.3f} {np.mean(alv):8.3f} {b:18.3f} {n:9.3f} {b / max(n, 1e-9):6.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]))
    else:
        report()
