#!/usr/bin/env python3
"""Look at UED levels without running training.

Three things you cannot see from the scalars alone:

  difficulty   what each difficulty tier actually looks like, which is what
               UED_DIFFICULTY_RANGE should be chosen from
  lineage      whether editing compounds complexity from an empty room
  population   a saved curriculum from a finished or running experiment

Examples
    python3 ADDS_AS_view_ued.py difficulty --out /tmp/tiers.png
    python3 ADDS_AS_view_ued.py lineage --difficulty 0 --steps 8
    python3 ADDS_AS_view_ued.py population --state ~/Log_SAC_UED/ued_curriculum.pkl
"""

from __future__ import annotations

import argparse
import os
import random

from ued import render
from ued.level import generate_random_level
from ued.mutate import MutationFailed, mutate_level


def cmd_difficulty(args) -> None:
    rng = random.Random(args.seed)
    levels, titles = [], []
    for difficulty in args.difficulties:
        for _ in range(args.per_difficulty):
            level = generate_random_level(rng, difficulty=difficulty)
            levels.append(level)
            titles.append(f"d{difficulty} o{len(level.obstacles)} {level.density():.2f}")
    render.save_grid(
        args.out, levels, titles=titles, cols=args.per_difficulty,
        suptitle="one row per difficulty tier",
    )
    print(f"wrote {args.out} ({len(levels)} levels)")


def cmd_lineage(args) -> None:
    rng = random.Random(args.seed)
    current = generate_random_level(rng, difficulty=args.difficulty)
    levels, titles = [current], ["origin"]
    failures = 0
    for _ in range(args.steps):
        try:
            current = mutate_level(current, rng=rng)
        except MutationFailed:
            failures += 1
            continue
        levels.append(current)
        titles.append(
            f"g{current.generation} o{len(current.obstacles)} "
            f"[{','.join(current.mutation_ops)}]"
        )
    render.save_grid(
        args.out, levels, titles=titles, cols=min(len(levels), 5),
        suptitle=f"editing from difficulty {args.difficulty} ({failures} failed edits)",
    )
    print(f"wrote {args.out} ({len(levels)} generations, {failures} failed edits)")


def cmd_population(args) -> None:
    # Loaded through the runner so the level id counter is restored too, which
    # matters if anything is bred from the loaded population afterwards.
    from ued.runner import UEDRunner

    if not os.path.exists(args.state):
        raise SystemExit(f"no curriculum state at {args.state}")

    runner = UEDRunner(value_fn=None, rng=random.Random(args.seed))
    if not runner.enabled:
        raise SystemExit("UED_ENABLED is False in config.py, so nothing will load")
    if not runner.load(args.state):
        raise SystemExit(f"could not load {args.state}")

    population = runner.population
    print(f"population: {len(population)} levels, episode {population.episode}")
    for name, value in sorted(population.stats().items()):
        print(f"  {name}: {value}")

    levels, titles = render.population_sample(population, n=args.count, top=not args.random)
    if not levels:
        raise SystemExit("population has no scored levels yet")
    render.save_grid(
        args.out, levels, titles=titles, cols=4,
        suptitle=f"{'random' if args.random else 'top-scoring'} levels @ episode {population.episode}",
    )
    print(f"wrote {args.out}")

    best = render.best_scored_level_id(population)
    if best is not None:
        chain, chain_titles = render.lineage(population, best, max_depth=8)
        if len(chain) > 1:
            path = args.out.replace(".png", "_lineage.png")
            render.save_grid(
                path, chain, titles=chain_titles, cols=len(chain),
                suptitle=f"lineage of best level {best}",
            )
            print(f"wrote {path}")
        else:
            print(f"best level {best} has no surviving ancestors to draw")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("difficulty", help="one row of levels per difficulty tier")
    p.add_argument("--difficulties", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    p.add_argument("--per-difficulty", type=int, default=4)
    p.add_argument("--out", default="ued_difficulty.png")
    p.set_defaults(func=cmd_difficulty)

    p = sub.add_parser("lineage", help="repeatedly edit one level and draw each generation")
    p.add_argument("--difficulty", type=int, default=0)
    p.add_argument("--steps", type=int, default=9)
    p.add_argument("--out", default="ued_lineage.png")
    p.set_defaults(func=cmd_lineage)

    p = sub.add_parser("population", help="draw a saved curriculum")
    p.add_argument("--state", default=os.path.expanduser("~/Log_SAC_UED/ued_curriculum.pkl"))
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--random", action="store_true", help="sample at random instead of taking the top")
    p.add_argument("--out", default="ued_population.png")
    p.set_defaults(func=cmd_population)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
