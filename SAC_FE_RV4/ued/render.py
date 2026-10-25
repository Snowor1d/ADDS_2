"""Pictures of curriculum levels (W&B image of the highest-scoring levels)."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def render_levels(levels: Sequence, path: str, titles: Optional[Sequence[str]] = None,
                  cols: int = 4) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    n = max(1, len(levels))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(levels):
            ax.axis("off")
            continue
        lv = levels[i]
        for o in lv.obstacles:
            ax.add_patch(MplPolygon(o, closed=True, color="#555555"))
        for e in lv.exits:
            ax.add_patch(MplPolygon(e, closed=True, color="#2ca02c"))
        ax.set_xlim(0, lv.width)
        ax.set_ylim(0, lv.height)
        ax.set_aspect("equal")
        t = titles[i] if titles else (f"id {lv.level_id} g{lv.generation} d{lv.difficulty} "
                                      f"n{lv.crowd}")
        ax.set_title(t, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path
