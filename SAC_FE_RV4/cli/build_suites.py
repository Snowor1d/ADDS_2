"""Build the frozen evaluation suites (data/suites/*.json).

    python3 -m cli.build_suites                 # every suite that is missing
    python3 -m cli.build_suites --only test     # one suite
    python3 -m cli.build_suites --force         # rebuild (changes the digests!)

Rebuilding changes what a suite contains; results are comparable only on
the same file, so rebuild only on purpose.
"""

import argparse
import os
import sys
import time

ORDER = ("val", "tune", "test", "test_scale", "human")


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up under nohup
    from configs import apply_cli_overrides
    argv = apply_cli_overrides(list(sys.argv[1:] if argv is None else argv))
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from configs import cfg
    from evaluation.suites import build_suite, save_suite, suite_path
    names = [n for n in ORDER if n in cfg.SUITE_SPECS] + \
        [n for n in cfg.SUITE_SPECS if n not in ORDER]
    if args.only:
        names = [n for n in names if n in args.only]
    for name in names:
        path = suite_path(name, cfg)
        if os.path.exists(path) and not args.force:
            print(f"[suites] {name}: exists ({path}); --force to rebuild")
            continue
        t = time.time()
        print(f"[suites] building {name} ...", flush=True)
        s = build_suite(name, cfg, log=lambda m: print(m, flush=True))
        save_suite(s, cfg)
        print(f"[suites] {name}: {s['n_maps']} maps, {s['n_scenarios']} scenarios, "
              f"digest {s['digest']} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
