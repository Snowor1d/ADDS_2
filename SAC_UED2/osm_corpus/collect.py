"""Stage 5: run the corpus, crop by crop, and record what happened.

Writes one row per (site, crop size) with the structural statistics and, for
usable crops, the obstacle rings. The rejected rows are kept rather than
dropped: a corpus that silently discards a third of its sites is a corpus with
an unexamined bias, and the reason for each rejection is the thing that tells
you whether the site needs a better query or is genuinely atypical.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Sequence

from osm_corpus import sites as sites_mod
from osm_corpus.extract import Crop, make_crop, make_road_crop
from osm_corpus.fetch import GEOFABRIK_REGIONS, download_extract, extract_path, fetch_site
from osm_corpus.resolve import Resolved, load_manifest
from osm_corpus.stats import layout_stats

CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "corpus")
CORPUS_INDEX = os.path.join(CORPUS_DIR, "index.json")


def ensure_extracts(keys: Sequence[str], skip_download: bool = False) -> Dict[str, str]:
    """Download each region a site needs, once. Returns region -> status."""
    wanted = {}
    for k in keys:
        region = GEOFABRIK_REGIONS.get(k)
        if region:
            wanted.setdefault(region, []).append(k)

    status = {}
    for region, users in sorted(wanted.items()):
        path = extract_path(region)
        if os.path.exists(path):
            status[region] = f"present ({os.path.getsize(path) >> 20} MiB)"
            continue
        if skip_download:
            status[region] = "missing (will fall back to Overpass)"
            continue
        print(f"  downloading {region} for {', '.join(users)}")
        try:
            download_extract(region)
            status[region] = f"downloaded ({os.path.getsize(path) >> 20} MiB)"
        except Exception as e:
            status[region] = f"failed: {type(e).__name__}: {e}"
    return status


def collect(keys: Optional[Sequence[str]] = None,
            crop_sizes: Optional[Sequence[int]] = None,
            skip_download: bool = False,
            mode: Optional[str] = None,
            simplify_m: Optional[float] = None,
            pause_s: float = 1.5) -> dict:
    from config import OSM_SIMPLIFY_M, OSM_TRAVERSABILITY

    mode = OSM_TRAVERSABILITY if mode is None else mode
    if mode not in ("roads", "buildings"):
        raise ValueError(f"unknown traversability mode {mode!r}")
    simplify_m = OSM_SIMPLIFY_M if simplify_m is None else float(simplify_m)

    manifest = load_manifest()
    if not manifest:
        raise RuntimeError("no site manifest; run `python3 -m osm_corpus.resolve` first")

    keys = list(keys) if keys else [s.key for s in sites_mod.SITES]
    crop_sizes = list(crop_sizes) if crop_sizes else list(sites_mod.CROP_SIZES)

    print(f"extracts for {len(keys)} sites:")
    extract_status = ensure_extracts(keys, skip_download=skip_download)
    for region, st in sorted(extract_status.items()):
        print(f"    {region:44s} {st}")

    os.makedirs(CORPUS_DIR, exist_ok=True)
    rows: List[dict] = []
    print(f"\ncropping {len(keys)} sites at {crop_sizes} m"
          f", traversability from {mode}, simplify {simplify_m} m:")

    for key in keys:
        entry = manifest.get(key)
        if entry is None:
            print(f"  {key:24s} not in manifest, skipped")
            continue
        for size in crop_sizes:
            row = dict(site=key, size_m=size, lat=entry.lat, lon=entry.lon,
                       city=entry.city, country=entry.country,
                       continent=entry.continent, morphology=entry.morphology,
                       traversability=mode)
            try:
                if mode == "roads":
                    crop = make_road_crop(key, GEOFABRIK_REGIONS.get(key),
                                          entry.lat, entry.lon, float(size),
                                          simplify_m=simplify_m)
                else:
                    buildings = fetch_site(key, entry.lat, entry.lon, float(size))
                    crop = make_crop(key, buildings, entry.lat, entry.lon,
                                     float(size))
                row["source"] = crop.source
                row["traversable"] = crop.traversable_fraction
                row["rejected"] = crop.rejected
                if crop.ok:
                    st = layout_stats(crop.polygons, float(size))
                    row["stats"] = st
                    row["rings"] = crop.rings_int()
                else:
                    row["stats"] = None
                    row["rings"] = None
            except Exception as e:
                row["source"] = "error"
                row["rejected"] = f"{type(e).__name__}: {e}"
                row["stats"] = None
                row["rings"] = None

            rows.append(row)
            mark = "ok " if row["rejected"] is None else "REJ"
            n = (row["stats"] or {}).get("n_obstacles", 0)
            cov = (row["stats"] or {}).get("coverage", 0.0)
            verts = (row["stats"] or {}).get("n_vertices", 0)
            print(f"  {key:24s} {size:5d} m  {mark} n={n:5.0f} cov={cov:.3f}"
                  f" verts={verts:5.0f}  {row['rejected'] or ''}")
            if row["source"] == "overpass":
                time.sleep(pause_s)

    # Merge into whatever is already on disk rather than replacing it.
    #
    # A partial run is the normal way to work: one site is re-cropped after its
    # query is fixed, or two are cropped to check a change. Writing only those
    # rows used to leave an index holding two sites where the exported levels
    # still held twenty-nine, and nothing downstream noticed until the compare
    # sheet reported every site as a tile failure. Rows are kept only when the
    # run agrees with them about how traversable space is defined; a mode or
    # tolerance change invalidates every measurement in the file.
    kept: List[dict] = []
    fresh = {(r["site"], int(r["size_m"])) for r in rows}
    if os.path.exists(CORPUS_INDEX):
        try:
            with open(CORPUS_INDEX, encoding="utf-8") as f:
                old = json.load(f)
        except Exception as e:
            print(f"\nexisting index unreadable ({type(e).__name__}), replaced")
            old = None
        if old is not None:
            same = (old.get("traversability", "buildings") == mode
                    and abs(float(old.get("simplify_m") or 0.0) - simplify_m) < 1e-9)
            if same:
                kept = [r for r in old.get("rows", [])
                        if (r["site"], int(r["size_m"])) not in fresh]
                if kept:
                    print(f"\nkept {len(kept)} rows from the existing index")
            elif old.get("rows"):
                print(f"\ndropped {len(old['rows'])} rows built as "
                      f"'{old.get('traversability', 'buildings')}' at "
                      f"{old.get('simplify_m')} m; this run is '{mode}' at "
                      f"{simplify_m} m")
    order = {s.key: i for i, s in enumerate(sites_mod.SITES)}
    rows = sorted(kept + rows,
                  key=lambda r: (order.get(r["site"], 1 << 30), int(r["size_m"])))

    payload = {
        "_about": ("Real downtown layouts for the SAC_UED2 corpus. Built by "
                   "osm_corpus.collect. Map data from OpenStreetMap, "
                   "(c) OpenStreetMap contributors, ODbL 1.0."),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crop_sizes_m": crop_sizes,
        "traversability": mode,
        "simplify_m": simplify_m,
        "extract_status": extract_status,
        "rows": rows,
    }
    tmp = CORPUS_INDEX + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, CORPUS_INDEX)

    ok = sum(1 for r in rows if r["rejected"] is None)
    print(f"\n{ok} of {len(rows)} crops usable -> {CORPUS_INDEX}")
    return payload


def load_corpus(path: str = CORPUS_INDEX) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"no corpus at {path}; run `python3 -m osm_corpus.collect`")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", nargs="+", default=None)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--skip-download", action="store_true",
                    help="use Overpass instead of downloading regional extracts")
    ap.add_argument("--mode", choices=["roads", "buildings"], default=None,
                    help="where traversable space comes from (default: config)")
    ap.add_argument("--simplify", type=float, default=None,
                    help="block outline simplification in metres (default: config)")
    args = ap.parse_args()
    collect(keys=args.sites, crop_sizes=args.sizes, skip_download=args.skip_download,
            mode=args.mode, simplify_m=args.simplify)
