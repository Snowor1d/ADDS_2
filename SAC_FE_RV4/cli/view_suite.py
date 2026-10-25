"""Draw a suite's maps as PNG grids (obstacles grey, exits green).

    python3 -m cli.view_suite test                # -> ~/RV4_eval/suite_views/test_*.png
    python3 -m cli.view_suite human --per-page 10
"""

import argparse
import os
import sys


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--per-page", type=int, default=24)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    from configs import cfg
    import sim_maps
    from evaluation.runner import results_root
    from evaluation.suites import load_suite
    from ued.level import Level
    from ued.render import render_levels

    s = load_suite(args.suite, cfg)
    seen, levels, titles = set(), [], []
    for sc in s["scenarios"]:
        if sc["map_key"] in seen:
            continue
        seen.add(sc["map_key"])
        if sc.get("level"):
            lv = Level.from_json(sc["level"])
        else:
            g = sim_maps.map_geometry(sc["map_num"], cfg)
            lv = Level(0, g["width"], g["height"], g["obstacles"], g["exits"], sc["crowd"])
        levels.append(lv)
        titles.append(f"{sc['map_key']} ({sc['group']})")
    out = args.out or os.path.join(results_root(cfg), "suite_views")
    os.makedirs(out, exist_ok=True)
    k = args.per_page
    for i in range(0, len(levels), k):
        p = os.path.join(out, f"{args.suite}_{i // k:02d}.png")
        render_levels(levels[i:i + k], p, titles[i:i + k], cols=6)
        print(p)


if __name__ == "__main__":
    main()
