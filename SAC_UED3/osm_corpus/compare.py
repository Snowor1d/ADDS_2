"""Put the extracted layout next to the real map, so it can be checked by eye.

Every other stage reports numbers, and numbers do not catch the kind of
mistake that matters most here: building relations silently dropped, a crop
centred on a plaza instead of a street, a courtyard filled that should have
stayed open. Those are obvious in a second when the simulation layout sits
beside the map it came from, and invisible in a coverage figure.

Tiles come from the OpenStreetMap standard layer, cached on disk so a re-run
costs nothing and the tile servers are hit once per view. Their usage policy
asks for an identifying User-Agent and no bulk downloading; a contact sheet of
the whole corpus is a few hundred tiles, which is why the cache is not
optional.
"""

from __future__ import annotations

import io
import math
import os
import time
from typing import List, Optional, Sequence, Tuple

import requests

USER_AGENT = ("ADDS-research/0.1 (crowd evacuation RL corpus verification; "
              "map tiles (c) OpenStreetMap contributors, ODbL)")
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_SIZE = 256

MIN_TILE_INTERVAL_S = 0.12          # polite spacing between uncached fetches
_last_fetch = [0.0]

# Target on-screen resolution for a crop. Fine enough to read individual
# buildings, coarse enough that a 1 km crop does not need hundreds of tiles.
TARGET_PIXELS = 700
MAX_ZOOM = 19
MIN_ZOOM = 13


def cache_dir() -> str:
    """Where downloaded map tiles are kept.

    Beside the regional extracts, for the same reason: a re-downloadable cache
    rather than part of the project. Only a few megabytes, but splitting the
    two would leave the rule unclear. Read through a function so importing
    this module does not pull in the training configuration.
    """
    from config import OSM_CACHE_DIR

    return os.path.join(OSM_CACHE_DIR, "tiles")


def deg2tile(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Fractional tile coordinates, so a bbox can be cropped exactly."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n
    return x, y


def resolution_m_per_px(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2.0 ** zoom)


def zoom_for(lat: float, size_m: float, target_px: int = TARGET_PIXELS) -> int:
    """The zoom whose pixel size renders `size_m` near `target_px` wide."""
    for zoom in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
        if size_m / resolution_m_per_px(lat, zoom) <= target_px:
            return zoom
    return MIN_ZOOM


def _fetch_tile(zoom: int, x: int, y: int):
    from PIL import Image

    path = os.path.join(cache_dir(), str(zoom), str(x), f"{y}.png")
    if os.path.exists(path):
        return Image.open(path).convert("RGB")

    gap = time.time() - _last_fetch[0]
    if gap < MIN_TILE_INTERVAL_S:
        time.sleep(MIN_TILE_INTERVAL_S - gap)
    _last_fetch[0] = time.time()

    url = TILE_URL.format(z=zoom, x=x, y=y)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=40)
    r.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def map_image(lat: float, lon: float, size_m: float,
              zoom: Optional[int] = None):
    """The standard OSM layer over exactly the crop's extent.

    Stitches whole tiles and then crops to the bbox, so the result lines up
    with the extracted layout pixel for pixel rather than approximately.
    """
    from PIL import Image

    from osm_corpus.fetch import bbox_for

    zoom = zoom or zoom_for(lat, size_m)
    # The crop itself, not the margin the building query uses.
    south, west, north, east = bbox_for(lat, lon, size_m, margin=1.0)

    left_f, top_f = deg2tile(north, west, zoom)
    right_f, bottom_f = deg2tile(south, east, zoom)
    # deg2tile's y grows southward, so north gives the smaller y.
    x_lo, x_hi = int(math.floor(left_f)), int(math.ceil(right_f))
    y_lo, y_hi = int(math.floor(top_f)), int(math.ceil(bottom_f))

    canvas = Image.new("RGB", ((x_hi - x_lo) * TILE_SIZE,
                               (y_hi - y_lo) * TILE_SIZE), (255, 255, 255))
    for tx in range(x_lo, x_hi):
        for ty in range(y_lo, y_hi):
            try:
                tile = _fetch_tile(zoom, tx, ty)
            except Exception:
                continue        # a missing tile leaves white rather than failing
            canvas.paste(tile, ((tx - x_lo) * TILE_SIZE, (ty - y_lo) * TILE_SIZE))

    px_left = int(round((left_f - x_lo) * TILE_SIZE))
    px_top = int(round((top_f - y_lo) * TILE_SIZE))
    px_right = int(round((right_f - x_lo) * TILE_SIZE))
    px_bottom = int(round((bottom_f - y_lo) * TILE_SIZE))
    return canvas.crop((px_left, px_top, max(px_right, px_left + 1),
                        max(px_bottom, px_top + 1)))


# ---------------------------------------------------------------------------
# side by side
# ---------------------------------------------------------------------------

def _level_for(site: str, size: int):
    from osm_corpus.export import load_levels

    for level in load_levels():
        if level.site_key == site and int(level.width) == int(size):
            return level
    return None


def _corpus_entry(site: str, size: int):
    from osm_corpus.collect import load_corpus

    for row in load_corpus()["rows"]:
        if row["site"] == site and int(row["size_m"]) == int(size):
            return row
    return None


def _modes():
    """How the levels were built, and how the corpus index was built."""
    import json

    from osm_corpus.collect import CORPUS_INDEX
    from osm_corpus.export import EXPORT_DIR

    def read(path, key):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get(key) or "buildings"
        except Exception:
            return "unknown"

    return (read(os.path.join(EXPORT_DIR, "real_levels.json"), "traversability"),
            read(CORPUS_INDEX, "traversability"))


def compare(sites: Sequence[str], size: int = 200,
            out: str = "osm_compare.png", cols: int = 3) -> str:
    """A sheet of site pairs: the real map, then what the simulator will run.

    The pair is the point. Read alone, a layout looks plausible whatever went
    wrong upstream; read against its map, a dropped building relation or a crop
    centred on a square is immediately visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from ued.render import draw_level

    levels_mode, corpus_mode = _modes()
    if levels_mode != corpus_mode:
        # The sheet would show layouts built one way, labelled with statistics
        # measured the other way, and look fine.
        print(f"  WARNING: exported levels came from '{levels_mode}' but the "
              f"corpus index says '{corpus_mode}'. Re-run export.")

    pairs = []
    for site in sites:
        level = _level_for(site, size)
        row = _corpus_entry(site, size)
        if level is None:
            print(f"  {site}: no exported level at {size} m, skipped")
            continue
        if row is None:
            # The corpus index is what carries the coordinates and the
            # statistics, so a level without one cannot be placed on a map.
            # This means the index was rebuilt over a narrower set of sites
            # than the levels were exported from.
            print(f"  {site}: exported level exists but the corpus index has "
                  f"no {size} m row; re-run collect for this site")
            continue
        try:
            tiles = map_image(row["lat"], row["lon"], float(size))
        except (OSError, requests.RequestException, ValueError) as e:
            # Deliberately narrow. A bare `except Exception` here once
            # reported every site as "tiles unavailable" when the real fault
            # was a missing corpus row, which sent the search in the wrong
            # direction entirely.
            print(f"  {site}: tiles unavailable ({type(e).__name__}: {e}), skipped")
            continue
        pairs.append((site, row, level, tiles))

    if not pairs:
        raise SystemExit("nothing to compare")

    rows = int(math.ceil(len(pairs) / cols))
    # Two-line titles above every panel plus a suptitle, so the per-row height
    # has to leave room for both or they land on the images.
    fig, axes = plt.subplots(rows, cols * 2,
                             figsize=(cols * 2 * 2.6, rows * 3.4 + 0.5), dpi=110)
    axes = np.atleast_1d(np.asarray(axes)).reshape(rows, cols * 2)

    for idx, (site, row, level, tiles) in enumerate(pairs):
        r, c = divmod(idx, cols)
        ax_map, ax_sim = axes[r][c * 2], axes[r][c * 2 + 1]

        ax_map.imshow(np.asarray(tiles))
        ax_map.set_xticks([]); ax_map.set_yticks([])
        st = row.get("stats") or {}
        ax_map.set_title(f"{site}\n{row['city']}, {row['country']}", fontsize=7)

        draw_level(ax_sim, level)
        ax_sim.set_title(
            f"simulator  {size} m\n"
            f"o{len(level.obstacles)}  cov {st.get('coverage', 0):.2f}  "
            f"n{level.crowd_size}", fontsize=7)

    for idx in range(len(pairs), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c * 2].axis("off")
        axes[r][c * 2 + 1].axis("off")

    fig.suptitle(
        f"real map vs extracted layout, {size} m crops   "
        f"(map tiles and building data (c) OpenStreetMap contributors, ODbL)",
        fontsize=9)
    fig.tight_layout(pad=0.6, h_pad=1.8)
    fig.subplots_adjust(top=1.0 - 0.55 / max(1.0, fig.get_figheight()))
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    import argparse

    from osm_corpus.sites import SITES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", default=None,
                    help="site keys; default is every exported site")
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--out", default="osm_compare.png")
    args = ap.parse_args()

    keys = args.sites or [s.key for s in SITES]
    path = compare(keys, size=args.size, out=args.out, cols=args.cols)
    print(f"wrote {path}")
