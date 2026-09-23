"""Drawing levels, for looking at what the curriculum is actually doing.

Renders the stored polygons rather than the 100x100 observation raster, so
exits, obstacles and the free space are visually distinct instead of being
three grey levels. Uses the Agg backend and never opens a window, so the same
code serves a training-time TensorBoard snapshot and an offline PNG.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # before pyplot; the trainer has no display of its own
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

# Deliberately readable when printed in greyscale, and distinct from the
# observation encoding so nobody mistakes one for the other.
COLOR_FREE = "#f7f7f4"
COLOR_OBSTACLE = "#3d4451"
COLOR_EXIT = "#2e9e5b"
COLOR_WALL = "#20242c"


def _zone_over_free_space(level, zone):
    """The hazard minus the blocks, as (exterior, holes) pairs.

    Holes are kept. A block sitting entirely inside the hazard cuts a hole in
    the difference, and returning only the outer ring fills that hole straight
    back in: measured, a difficulty-6 crop put about five per cent of the wash
    back on top of masonry that way.

    Falls back to the raw shape if the subtraction fails, so a rendering
    problem never costs the picture the one thing it most needs to show.
    """
    try:
        from shapely.geometry import Polygon as ShPoly
        from shapely.ops import unary_union

        shape = zone.polygon()
        blocks = [ShPoly([(float(p[0]), float(p[1])) for p in ring])
                  for ring in level.obstacles if len(ring) >= 3]
        blocks = [b if b.is_valid else b.buffer(0) for b in blocks]
        blocks = [b for b in blocks if not b.is_empty]
        if blocks:
            shape = shape.difference(unary_union(blocks))
        if shape.is_empty:
            return []
        parts = shape.geoms if shape.geom_type == "MultiPolygon" else [shape]
        return [(list(q.exterior.coords),
                 [list(h.coords) for h in q.interiors])
                for q in parts if q.geom_type == "Polygon" and not q.is_empty]
    except Exception:
        return [(list(zone.polygon().exterior.coords), [])]


def _hazard_patch(exterior, holes, **kw):
    """A filled patch with holes, which a plain Polygon cannot express."""
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    verts, codes = [], []
    for ring in [exterior] + list(holes):
        if len(ring) < 3:
            continue
        pts = list(ring)
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        verts.extend(pts)
        codes.extend([Path.MOVETO] + [Path.LINETO] * (len(pts) - 2)
                     + [Path.CLOSEPOLY])
    if not verts:
        return None
    return PathPatch(Path(verts, codes), **kw)


def draw_level(ax, level, title: Optional[str] = None, show_axes: bool = False,
               common_extent: Optional[float] = None) -> None:
    """Draw one level onto a matplotlib axis.

    `common_extent` sets the same axis range on every tile of a contact sheet,
    so a small world visibly fills less of its tile than a large one. Without
    it each level is drawn to fit and every map looks the same size, which
    hides the axis the curriculum is now allowed to vary.
    """
    w, h = float(level.width), float(level.height)
    limit = float(common_extent) if common_extent else None

    ax.set_xlim(0, limit or w)
    ax.set_ylim(0, limit or h)
    ax.set_aspect("equal")
    if limit is None:
        ax.set_facecolor(COLOR_FREE)
    else:
        ax.set_facecolor("none")
        ax.add_patch(
            MplPolygon(
                [(0, 0), (w, 0), (w, h), (0, h)],
                closed=True, facecolor=COLOR_FREE, edgecolor="none",
            )
        )

    for poly in level.obstacles:
        if len(poly) >= 3:
            ax.add_patch(
                MplPolygon(
                    [(float(p[0]), float(p[1])) for p in poly],
                    closed=True,
                    facecolor=COLOR_OBSTACLE,
                    edgecolor=COLOR_OBSTACLE,
                    linewidth=0.4,
                )
            )

    # The hazard, over the fabric and under the exits. This is the level's
    # task: the obstacles say where a robot can go and the zone says which
    # part of that the crowd must not be standing in, so a contact sheet
    # without it shows maps rather than problems.
    zone = getattr(level, "danger", None)
    if zone is not None:
        from config import (DANGER_EDGE_COLOR, DANGER_FILL_ALPHA,
                            DANGER_FILL_COLOR)

        # Clipped to the ground the hazard actually covers, so red never sits
        # on top of masonry. The raw outline is kept as a dashed line, because
        # that is the boundary the task is defined by and the clipped fill
        # alone would not show where it runs behind a block.
        for exterior, holes in _zone_over_free_space(level, zone):
            patch = _hazard_patch(exterior, holes,
                                  facecolor=DANGER_FILL_COLOR,
                                  edgecolor="none", alpha=DANGER_FILL_ALPHA)
            if patch is not None:
                ax.add_patch(patch)

        # Drawn from the zone's own polygon, the same object the fill was
        # clipped out of. Rebuilding it from cx/cy and the half-extents is
        # what the first version did, and it drew an unrotated rectangle for a
        # rotated hazard: the dashed outline sat at a different angle from the
        # red it was supposed to bound.
        ax.add_patch(
            MplPolygon(list(zone.polygon().exterior.coords), closed=True,
                       facecolor="none", edgecolor=DANGER_EDGE_COLOR,
                       linewidth=0.9, linestyle="--")
        )

    for poly in level.exits:
        if len(poly) >= 3:
            ax.add_patch(
                MplPolygon(
                    [(float(p[0]), float(p[1])) for p in poly],
                    closed=True,
                    facecolor=COLOR_EXIT,
                    edgecolor=COLOR_EXIT,
                    # Exits are only 7-10 by 4-5 units, so at contact-sheet
                    # scale they need help to stay findable.
                    linewidth=1.6,
                )
            )

    if limit is not None:
        # The spines sit at the axis limits, not at the world edge, so with a
        # shared range they would draw every world the same size regardless of
        # how big it is. Draw the actual boundary instead.
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.add_patch(
            MplPolygon(
                [(0, 0), (w, 0), (w, h), (0, h)],
                closed=True, fill=False,
                edgecolor=COLOR_WALL, linewidth=1.0,
            )
        )
    else:
        for spine in ax.spines.values():
            spine.set_color(COLOR_WALL)
            spine.set_linewidth(1.0)

    if not show_axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=6.5, pad=2, linespacing=1.15)


def default_title(level, score: Optional[float] = None) -> str:
    """The things you compare levels by, short enough to fit over a tile.

    Tiles are about 1.5 inches wide, so anything longer than roughly six
    fields runs into its neighbour; extra fields go on a second line instead.
    """
    head = [f"#{level.level_id}"]
    if level.difficulty is not None:
        head.append(f"d{level.difficulty}")
    head.append(f"g{level.generation}")
    head.append(f"o{len(level.obstacles)}")
    head.append(f"{level.density():.2f}")
    title = " ".join(head)
    # World size is a design variable, so it belongs on the label even when the
    # common axis range already shows it relatively.
    tail = [f"{level.width}x{level.height}", f"n{level.crowd_size}"]
    if score is not None:
        tail.append(f"s{score:.3f}")
    return title + "\n" + " ".join(tail)


def render_grid(
    levels: Sequence,
    titles: Optional[Sequence[str]] = None,
    cols: Optional[int] = None,
    tile_inches: float = 1.5,
    dpi: int = 110,
    suptitle: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Render levels as a contact sheet and return it as an HWC uint8 array.

    The array form is what TensorBoard's add_image wants, so the training-time
    snapshot and the offline PNG come out of the same call.
    """
    levels = list(levels)
    if not levels:
        return None

    if cols is None:
        cols = max(1, min(6, int(math.ceil(math.sqrt(len(levels))))))
    rows = int(math.ceil(len(levels) / cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * tile_inches, rows * (tile_inches + 0.34) + (0.3 if suptitle else 0.0)),
        dpi=dpi,
    )
    axes = np.atleast_1d(np.asarray(axes)).reshape(-1)

    # One axis range for the whole sheet, so relative world size is visible.
    common_extent = max(
        max(float(lv.width), float(lv.height)) for lv in levels
    )

    try:
        for i, ax in enumerate(axes):
            if i < len(levels):
                title = titles[i] if titles is not None and i < len(titles) else default_title(levels[i])
                draw_level(ax, levels[i], title=title, common_extent=common_extent)
            else:
                ax.axis("off")

        if suptitle:
            fig.suptitle(suptitle, fontsize=9)
        fig.tight_layout(pad=0.6, h_pad=1.1)
        if suptitle:
            # tight_layout does not reserve room for a suptitle, so without
            # this it lands on top of the first row's per-tile titles.
            fig.subplots_adjust(top=1.0 - 0.5 / max(1.0, fig.get_figheight()))
        return _figure_to_array(fig)
    finally:
        plt.close(fig)


def save_grid(path: str, levels: Sequence, **kwargs) -> Optional[str]:
    """Write a contact sheet to a PNG. Returns the path, or None if empty."""
    img = render_grid(levels, **kwargs)
    if img is None:
        return None
    import imageio.v2 as imageio

    imageio.imwrite(path, img)
    return path


def _figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(buf[:, :, :3])


# ---------------------------------------------------------------------------
# population views
# ---------------------------------------------------------------------------

def population_sample(population, n: int = 12, top: bool = True) -> Tuple[List, List[str]]:
    """The n highest-scoring levels, or a random n.

    Scored levels only. An unscored level's rank comes from its parent, so
    including them would show the curriculum levels it has not measured yet.
    """
    with population._lock:
        records = [r for r in population._records.values() if r.trials > 0]
        scored = [(population._raw_score(r), r) for r in records]
    if not scored:
        return [], []

    if top:
        scored.sort(key=lambda sr: sr[0], reverse=True)
        chosen = scored[:n]
    else:
        chosen = population.rng.sample(scored, min(n, len(scored)))

    levels = [r.level for _, r in chosen]
    titles = [
        f"{default_title(r.level, score)} p{r.success_rate:.2f} t{r.trials}"
        .replace("\n", "\n")
        for score, r in chosen
    ]
    return levels, titles


def lineage(population, level_id: int, max_depth: int = 8) -> Tuple[List, List[str]]:
    """Walk a level's ancestors, oldest first.

    The chain stops wherever an ancestor has already been evicted, so a
    lineage can come back shorter than its generation number implies. That is
    the honest picture: the population does not keep every ancestor alive.
    """
    with population._lock:
        records = dict(population._records)

    chain = []
    current = records.get(level_id)
    while current is not None and len(chain) < max_depth:
        chain.append(current)
        parent_id = current.level.parent_id
        current = records.get(parent_id) if parent_id is not None else None

    chain.reverse()
    levels = [r.level for r in chain]
    titles = []
    for r in chain:
        ops = ",".join(r.level.mutation_ops) if r.level.mutation_ops else "origin"
        titles.append(f"g{r.level.generation} o{len(r.level.obstacles)} [{ops}]")
    return levels, titles


def best_scored_level_id(population) -> Optional[int]:
    with population._lock:
        records = [r for r in population._records.values() if r.trials > 0]
        if not records:
            return None
        best = max(records, key=lambda r: population._raw_score(r))
        return best.level.level_id
