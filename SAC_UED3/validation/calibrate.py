"""Sweep the social force constants against the standard test cases.

Run: python3 -m validation.calibrate

The constants live in config and are read through agent.py's namespace, so a
sweep sets them on that module between runs. Nothing here writes to config:
the output is a table, and choosing from it is a deliberate edit.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List

import numpy as np


def _apply(**kw):
    """Set social force constants for the next model built."""
    import agent

    for k, v in kw.items():
        setattr(agent, k, v)


def fd_error(seed: int = 0, densities=(0.5, 1.0, 2.0),
             length: float = 20.0, width: float = 3.0) -> Dict:
    """Root mean square gap between the model's curve and Weidmann's."""
    from validation.measure import fundamental_diagram, weidmann_speed

    rows = fundamental_diagram(densities=(0.1,) + tuple(densities),
                               length=length, width=width, warmup=15,
                               measure=25, seed=seed, verbose=False)
    free = rows[0]["flow_speed_ms"]
    errs = []
    got = {}
    for r in rows[1:]:
        d = r["density_ped_m2"]
        want = weidmann_speed(d, free)
        got[d] = r["flow_speed_ms"]
        errs.append(r["flow_speed_ms"] - want)
    return {"free": free, "rmse": float(np.sqrt(np.mean(np.square(errs)))),
            "speeds": got}


def sweep(grid: Dict[str, List[float]], seed: int = 0) -> List[Dict]:
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, combo))
        _apply(**kw)
        t0 = time.perf_counter()
        r = fd_error(seed=seed)
        row = dict(kw)
        row.update(r)
        row["seconds"] = time.perf_counter() - t0
        out.append(row)
        speeds = " ".join(f"{d}:{v:.2f}" for d, v in sorted(r["speeds"].items()))
        print("  " + "  ".join(f"{k}={v}" for k, v in kw.items())
              + f"  free={r['free']:.2f}  rmse={r['rmse']:.3f}  {speeds}",
              flush=True)
    out.sort(key=lambda r: r["rmse"])
    return out


def main():
    from config import SF_ANISOTROPY, SF_K_AGENT, SF_LAMBDA_A
    from validation.measure import weidmann_speed

    print("target curve (Weidmann, at the model's own free speed):")
    for d in (0.5, 1.0, 2.0, 3.0):
        print(f"    {d}: {weidmann_speed(d, 1.49):.2f} m/s")

    print("\nsweep:")
    rows = sweep({
        "SF_K_AGENT": [10.0, 20.0, 40.0, 60.0],
        "SF_LAMBDA_A": [0.3, 0.45, 0.6, 0.8],
        "SF_ANISOTROPY": [0.2],
    })
    print("\nbest:")
    for r in rows[:5]:
        print(f"  K={r['SF_K_AGENT']:6.1f} lambda={r['SF_LAMBDA_A']:.2f} "
              f"aniso={r['SF_ANISOTROPY']:.2f}  rmse={r['rmse']:.3f}")
    print(f"\ncurrent config: K={SF_K_AGENT} lambda={SF_LAMBDA_A} "
          f"aniso={SF_ANISOTROPY}")


if __name__ == "__main__":
    main()
