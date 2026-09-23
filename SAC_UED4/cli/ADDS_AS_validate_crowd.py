"""Verification and validation of the crowd model.

Runs the standard pedestrian-dynamics test cases against this simulator and
prints what it measured beside what the literature reports. See
docs/crowd_validation.md for the sources and for what each result means.

    python3 -m cli.ADDS_AS_validate_crowd all
    python3 -m cli.ADDS_AS_validate_crowd fd
    python3 -m cli.ADDS_AS_validate_crowd bottleneck --door 1.2
"""

from __future__ import annotations

import argparse
import json
import os
import time

from paths import at

REPORT_DIR = at("validation", "results")


def _save(name, payload):
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  saved -> {path}")


def run_fd(args):
    from validation.measure import fundamental_diagram, weidmann_speed

    rows = fundamental_diagram(seed=args.seed)
    free = rows[0]["flow_speed_ms"]
    print("\nFundamental diagram: walking speed against density")
    print(f"{'density':>9} {'n':>5} {'speed m/s':>10} {'Weidmann':>9} "
          f"{'ratio':>7}")
    for r in rows:
        expect = weidmann_speed(r["density_ped_m2"], free)
        r["weidmann_ms"] = expect
        print(f"{r['density_ped_m2']:9.2f} {r['n']:5d} "
              f"{r['flow_speed_ms']:10.3f} {expect:9.3f} "
              f"{r['flow_speed_ms'] / max(expect, 1e-9):7.2f}")
    falls = all(a["flow_speed_ms"] >= b["flow_speed_ms"] - 1e-6
                for a, b in zip(rows, rows[1:]))
    print(f"\n  monotone decreasing with density: {'yes' if falls else 'NO'}")
    _save("fundamental_diagram", rows)
    return rows


def run_free(args):
    from validation.measure import REFERENCE, free_speed

    r = free_speed(seed=args.seed)
    print("\nFree walking speed")
    print(f"  configured mean      {r['configured_mean_ms']:.2f} m/s")
    print(f"  drawn mean           {r['drawn_mean_ms']:.2f} m/s")
    print(f"  achieved mean        {r['achieved_mean_ms']:.2f} m/s "
          f"(sd {r['achieved_sd']:.2f})")
    print(f"  Weidmann free speed  {REFERENCE['free_speed_mean_ms']:.2f} m/s")
    _save("free_speed", r)
    return r


def run_bottleneck(args):
    from validation.measure import REFERENCE, bottleneck_flow

    rows = []
    for w in (args.door,) if args.door else (0.8, 1.2, 2.0):
        r = bottleneck_flow(n=args.n, door_width=w, seed=args.seed)
        rows.append(r)
        print(f"\nBottleneck, door {w:.1f} m")
        print(f"  crossed              {r['crossed']} of {r['n']}")
        print(f"  flow                 {r.get('flow_ps', 0.0):.2f} persons/s")
        print(f"  specific flow        {r['specific_flow_pms']:.2f} "
              f"persons/m/s")
        print(f"  literature           {REFERENCE['specific_flow_pms']:.2f} "
              f"persons/m/s")
    _save("bottleneck", rows)
    return rows


def run_fis(args):
    from validation.measure import faster_is_slower

    rows = faster_is_slower(n=args.n, seed=args.seed)
    print("\nFaster is slower: desired speed against how long the room takes")
    print(f"{'scale':>6} {'t90 steps':>10} {'specific flow':>14}")
    for r in rows:
        print(f"{r['speed_scale']:6.1f} {str(r['t90_steps']):>10} "
              f"{r['specific_flow_pms']:14.2f}")
    worse = any(b["t90_steps"] and a["t90_steps"] and b["t90_steps"] > a["t90_steps"]
                for a, b in zip(rows, rows[1:]))
    print(f"\n  slower at some higher speed: {'yes' if worse else 'NO'}")
    _save("faster_is_slower", rows)
    return rows


def run_lanes(args):
    from validation.measure import lane_formation

    r = lane_formation(seed=args.seed)
    print("\nLane formation in counterflow")
    print(f"  order at start       {r['order_start']:.3f}")
    print(f"  order at end         {r['order_end']:.3f}")
    print(f"  (0.5 is a mixed crowd, 1.0 is complete separation)")
    print(f"\n  lanes formed: "
          f"{'yes' if r['order_end'] > r['order_start'] + 0.05 else 'NO'}")
    _save("lane_formation", r)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["all", "fd", "free", "bottleneck",
                                     "fis", "lanes"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--door", type=float, default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.what in ("all", "free"):
        run_free(args)
    if args.what in ("all", "fd"):
        run_fd(args)
    if args.what in ("all", "bottleneck"):
        run_bottleneck(args)
    if args.what in ("all", "fis"):
        run_fis(args)
    if args.what in ("all", "lanes"):
        run_lanes(args)
    print(f"\ndone in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
