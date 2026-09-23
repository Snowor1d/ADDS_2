"""Stage 6: compare the corpus against the generator, and say which knob to turn.

The output is a report for a person to act on, not an optimiser. Automated
distribution matching would need a single scalar objective over these
statistics, and choosing that scalar is the whole problem: weight the width
percentiles too heavily and the generator learns to carve corridors by making
everything enormous. So the report puts the two distributions side by side,
flags each field as low, high or covered, and names the generator parameter
that moves it.

"Covered" is the goal rather than "matched". For a curriculum the generator's
range has to contain the real range with margin, so that the editing stage has
somewhere to go and the policy is not asked to extrapolate on the very axis
the corpus is drawn from.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from osm_corpus.collect import load_corpus
from osm_corpus.stats import FIELDS, LABELS, layout_stats, summarise

# Which generator parameters move each statistic. These are the knobs in
# random_map._params_from_difficulty_base plus the config ranges.
KNOBS: Dict[str, str] = {
    "n_obstacles": "large_count_min/max, small_count_min/max",
    "n_vertices": "not a generator target; it is the simulation-cost budget "
                  "and the lever is config.OSM_SIMPLIFY_M",
    "coverage": "density_min/max",
    "footprint_median": "small_rect_*_min/max, rect_*_min/max, large_bw/bh_*",
    "footprint_cv": "the spread between large_* and small_* size ranges",
    "footprint_max_frac": "large_bw_max, large_bh_max, large_conv_scale_max",
    "width_p10": "min_obstacle_gap, wall_clearance",
    "width_p50": "density_min/max together with obstacle counts",
    "width_p90": "large_count_max and how much of the map a block can cover",
    "open_fraction": "density_min; open ground is unbuilt area",
    # Not actionable: see the note in stats.layout_stats. The channelisation
    # question it was meant to answer is answered by width_p50 and
    # open_fraction instead, and street_map.py is the generator built for it.
    "free_components": "not a target; read width_p50 and open_fraction instead",
}

# A generated range is "covered" when it spans the corpus range with this much
# slack on each side, measured in corpus interquantile widths.
MARGIN = 0.15

# Crops this open are worth a look before they are trusted as downtown fabric.
# The quality filter in extract.py is deliberately permissive, because a real
# corpus contains genuinely open types such as tower-in-park superblocks, so
# the check is a flag in this report rather than a rejection there. Real sites
# measured so far run from about 0.13 to 0.58 built coverage.
OPEN_CROP_COVERAGE = 0.15
OPEN_CROP_FREE_GROUND = 0.45

# Road-derived crops need the mirror-image check. There the blocked share is
# normally high, so a thin crop is one whose street network barely covers the
# ground, and a suspicious crop is one where almost nothing is blocked, which
# usually means the extract is missing the road classes for that country
# rather than that the place is a plaza. Measured street shares run from about
# 0.15 at Palermo Soho to about 0.44 at Kreuzberg.
THIN_STREET_SHARE = 0.10
BARE_STREET_SHARE = 0.70


def generator_sample(n: int = 40, size_m: int = 200,
                     difficulties: Sequence[int] = (0, 1, 2, 3, 4, 5, 6),
                     morphology: Optional[str] = None) -> List[dict]:
    """Layout statistics for a sample of generated levels.

    Drawn from `citygen`, which is the only family now. Pass `morphology` to
    compare one pattern against the real sites carrying that tag, which is the
    comparison the per-morphology design was for; leave it out for a spread
    across all seven.
    """
    import random

    from shapely.geometry import Polygon

    from citygen.generate import generate_city_plan
    from citygen.morphology import MORPHOLOGIES

    morphs = [morphology] if morphology else sorted(MORPHOLOGIES)
    rows: List[dict] = []
    for i in range(n):
        rng = random.Random(20000 + i)
        d = difficulties[i % len(difficulties)]
        m = morphs[i % len(morphs)]
        plan = generate_city_plan(rng, m, difficulty=int(d),
                                  width=int(size_m), height=int(size_m))
        rings, _ = plan.render()
        if not rings:
            # Difficulty 0 is an open field. It has no layout to measure, and
            # including it as a row of zeros would drag every percentile.
            continue
        st = layout_stats([Polygon(r) for r in rings], float(size_m))
        if st:
            st["morphology"] = m
            rows.append(st)
    return rows


def corpus_rows(corpus: dict, size_m: Optional[int] = None) -> List[dict]:
    out = []
    for row in corpus["rows"]:
        if row.get("rejected") is not None or not row.get("stats"):
            continue
        if size_m is not None and int(row["size_m"]) != int(size_m):
            continue
        out.append(row["stats"])
    return out


def verdict(real: Dict[str, float], gen: Dict[str, float]) -> str:
    """Whether the generator's range covers the corpus range."""
    span = max(1e-9, real["p90"] - real["p10"])
    slack = MARGIN * span
    below = gen["p10"] <= real["p10"] + slack
    above = gen["p90"] >= real["p90"] - slack
    if below and above:
        return "covered"
    if not above and below:
        return "generator TOO LOW at the top"
    if above and not below:
        return "generator TOO HIGH at the bottom"
    return "generator range MISSES both ends"


def render(size_m: int = 200, n_generated: int = 40) -> str:
    corpus = load_corpus()
    real_rows = corpus_rows(corpus, size_m)
    if not real_rows:
        raise RuntimeError(f"no usable corpus crops at {size_m} m")
    gen_rows = generator_sample(n=n_generated, size_m=size_m)

    real = summarise(real_rows)
    gen = summarise(gen_rows)

    lines = []
    lines.append(f"Corpus vs generator at {size_m} m crops")
    lines.append(f"  corpus: {len(real_rows)} real downtown crops, traversable "
                 f"space from {corpus.get('traversability', 'buildings')}")
    lines.append(f"  generator: {len(gen_rows)} maps across difficulties 0-6")
    lines.append("")
    lines.append(f"{'statistic':34s} {'real p10..p50..p90':>26s} "
                 f"{'generated p10..p50..p90':>26s}  verdict")
    lines.append("-" * 118)
    for f in FIELDS:
        if f not in real or f not in gen:
            continue
        r, g = real[f], gen[f]
        lines.append(
            f"{LABELS[f]:34s} "
            f"{r['p10']:8.2f}{r['p50']:9.2f}{r['p90']:9.2f} "
            f"{g['p10']:8.2f}{g['p50']:9.2f}{g['p90']:9.2f}  {verdict(r, g)}")

    lines.append("")
    lines.append("What to turn for each gap:")
    for f in FIELDS:
        if f not in real or f not in gen:
            continue
        if f == "free_components":
            continue
        v = verdict(real[f], gen[f])
        if v != "covered":
            lines.append(f"  {LABELS[f]:34s} {v}")
            lines.append(f"    knob: {KNOBS.get(f, 'n/a')}")

    # Per site, so an unusually open crop is visible rather than averaged away.
    # The filter lets these through on purpose; this is where they surface.
    rows_at_size = [r for r in corpus["rows"]
                    if r.get("rejected") is None and r.get("stats")
                    and int(r["size_m"]) == int(size_m)]
    mode = corpus.get("traversability", "buildings")

    def reasons(st):
        why = []
        if mode == "roads":
            free = 1.0 - st["coverage"]
            if free < THIN_STREET_SHARE:
                why.append("hardly any street")
            if free > BARE_STREET_SHARE:
                why.append("hardly anything blocked")
        else:
            if st["coverage"] < OPEN_CROP_COVERAGE:
                why.append("sparse")
            if st["open_fraction"] > OPEN_CROP_FREE_GROUND:
                why.append("wide open")
        return why

    flagged = [r for r in rows_at_size if reasons(r["stats"])]
    lines.append("")
    if mode == "roads":
        criterion = (f"street share under {THIN_STREET_SHARE:.0%} or over "
                     f"{BARE_STREET_SHARE:.0%} of the crop")
    else:
        criterion = (f"built coverage under {OPEN_CROP_COVERAGE} or more than "
                     f"{OPEN_CROP_FREE_GROUND:.0%} of the area far from a "
                     f"building")
    lines.append(f"Crops worth a look ({len(flagged)} of {len(rows_at_size)} at "
                 f"{size_m} m): {criterion}.")
    if flagged:
        lines.append(f"  {'site':24s} {'morphology':14s} {'coverage':>9s} "
                     f"{'open':>6s} {'obstacles':>10s}  why")
        for r in sorted(flagged, key=lambda r: r["stats"]["coverage"]):
            st = r["stats"]
            why = reasons(st)
            lines.append(f"  {r['site']:24s} {r['morphology']:14s} "
                         f"{st['coverage']:9.3f} {st['open_fraction']:6.2f} "
                         f"{st['n_obstacles']:10.0f}  {', '.join(why)}")
    else:
        lines.append("  none")

    # The other tail: the densest crops, which set the top of the range the
    # generator has to reach.
    dense = sorted(rows_at_size, key=lambda r: -r["stats"]["coverage"])[:5]
    if dense:
        lines.append("")
        lines.append(f"Densest crops at {size_m} m, which set the top of the range:")
        for r in dense:
            st = r["stats"]
            lines.append(f"  {r['site']:24s} {r['morphology']:14s} "
                         f"{st['coverage']:9.3f} {st['open_fraction']:6.2f} "
                         f"{st['n_obstacles']:10.0f}")

    # By morphology, so a gap can be traced to the kind of fabric causing it.
    by_morph: Dict[str, List[dict]] = {}
    for row in corpus["rows"]:
        if row.get("rejected") is not None or not row.get("stats"):
            continue
        if int(row["size_m"]) != int(size_m):
            continue
        by_morph.setdefault(row["morphology"], []).append(row["stats"])
    if by_morph:
        lines.append("")
        lines.append(f"{'morphology':16s} {'n':>3s} {'coverage':>9s} "
                     f"{'width p50':>10s} {'obstacles':>10s} {'components':>11s}")
        lines.append("-" * 64)
        for m, rows in sorted(by_morph.items()):
            s = summarise(rows)
            lines.append(f"{m:16s} {len(rows):3d} {s['coverage']['p50']:9.2f} "
                         f"{s['width_p50']['p50']:10.1f} "
                         f"{s['n_obstacles']['p50']:10.0f} "
                         f"{s['free_components']['p50']:11.0f}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--n-generated", type=int, default=40)
    ap.add_argument("--out", default=None, help="also write the report here")
    args = ap.parse_args()
    text = render(size_m=args.size, n_generated=args.n_generated)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.out}")
