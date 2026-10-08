"""Fit the morphology tables to the corpus, and print them to paste back.

The generator is pinned to two measured numbers per morphology: how far apart
the streets run, and how much of the crop is street. Both are read off the
road-derived corpus here rather than chosen, and the width every pattern uses
is derived from them.

Spacing is recovered as size / sqrt(block count). That is exact for a square
grid and approximate for everything else, which is the right trade: the corpus
counts blocks, and a block count is a far more stable measurement than any
attempt to identify individual streets in a rendered layout.

Run it after re-collecting:

    python3 -m citygen.fit                 # every size the corpus holds
    python3 -m citygen.fit --size 200      # one crop size

The printed table carries the crop count behind each figure, because two of
the seven morphologies rest on a single site and a number fitted from one crop
should not read the same as one fitted from nine.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, Optional, Sequence, Tuple


def measurements(size_m: Optional[int] = None) -> Dict[str, List[dict]]:
    """Per-morphology rows of (site, size, blocks, street share, spacing)."""
    from osm_corpus.collect import load_corpus

    corpus = load_corpus()
    if corpus.get("traversability") != "roads":
        raise SystemExit(
            f"corpus was built as '{corpus.get('traversability', 'buildings')}'.\n"
            "The generator is fitted to road-derived traversable space, and a\n"
            "building-derived corpus measures something else. Re-run:\n"
            "  python3 -m cli.ADDS_AS_osm_pipeline collect")

    out: Dict[str, List[dict]] = {}
    for row in corpus["rows"]:
        if row.get("rejected") is not None or not row.get("stats"):
            continue
        if size_m is not None and int(row["size_m"]) != int(size_m):
            continue
        st = row["stats"]
        n_blocks = float(st["n_obstacles"])
        if n_blocks < 1:
            continue
        size = float(row["size_m"])
        share = 1.0 - float(st["coverage"])
        spacing = size / math.sqrt(n_blocks)
        width = spacing * (1.0 - math.sqrt(max(0.0, 1.0 - share)))
        out.setdefault(row["morphology"], []).append(dict(
            site=row["site"], size=int(size), blocks=int(n_blocks),
            share=share, spacing=spacing, width=width,
            vertices=float(st.get("n_vertices", 0.0))))
    return out


def fitted(size_m: Optional[int] = None
           ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    """Median spacing and share per morphology, plus how many crops back them.

    Median rather than mean throughout. One crop centred where OSM has sparse
    road data drags a mean a long way, and the corpus deliberately keeps such
    crops rather than filtering them out.
    """
    rows = measurements(size_m)
    spacing, share, support = {}, {}, {}
    for morph, rs in rows.items():
        spacing[morph] = statistics.median(r["spacing"] for r in rs)
        share[morph] = statistics.median(r["share"] for r in rs)
        support[morph] = len(rs)
    return spacing, share, support


def render(size_m: Optional[int] = None) -> str:
    from citygen.morphology import BLOCK_SPACING_M, STREET_SHARE

    rows = measurements(size_m)
    spacing, share, support = fitted(size_m)

    lines: List[str] = []
    scope = "all crop sizes" if size_m is None else f"{size_m} m crops"
    lines.append(f"Morphology tables fitted to the corpus ({scope})")
    lines.append("")
    lines.append(f"{'morphology':16s} {'crops':>5s} {'spacing m':>10s} "
                 f"{'share':>7s} {'width m':>8s}   {'in use now':>22s}")
    lines.append("-" * 82)
    for morph in sorted(rows):
        sp, sh = spacing[morph], share[morph]
        w = sp * (1.0 - math.sqrt(max(0.0, 1.0 - sh)))
        now = (f"{BLOCK_SPACING_M.get(morph, float('nan')):.0f} m / "
               f"{STREET_SHARE.get(morph, float('nan')):.3f}")
        lines.append(f"{morph:16s} {support[morph]:5d} {sp:10.1f} {sh:7.3f} "
                     f"{w:8.1f}   {now:>22s}")

    lines.append("")
    lines.append("Per crop, so an outlier is visible rather than averaged away:")
    for morph in sorted(rows):
        lines.append(f"  {morph}")
        for r in sorted(rows[morph], key=lambda r: (r["site"], r["size"])):
            lines.append(f"    {r['site']:24s} {r['size']:5d} m  "
                         f"blocks {r['blocks']:4d}  share {r['share']:.3f}  "
                         f"spacing {r['spacing']:6.1f} m  "
                         f"width {r['width']:5.1f} m")

    lines.append("")
    lines.append("Paste into citygen/morphology.py:")
    lines.append("")
    lines.append("BLOCK_SPACING_M: Dict[str, float] = {")
    for morph in sorted(rows):
        lines.append(f"    {morph!r}: {spacing[morph]:.1f},"
                     f"  # {support[morph]} crops")
    lines.append("}")
    lines.append("")
    lines.append("STREET_SHARE: Dict[str, float] = {")
    for morph in sorted(rows):
        lines.append(f"    {morph!r}: {share[morph]:.3f},"
                     f"  # {support[morph]} crops")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=None,
                    help="fit to one crop size only")
    args = ap.parse_args()
    print(render(size_m=args.size))
