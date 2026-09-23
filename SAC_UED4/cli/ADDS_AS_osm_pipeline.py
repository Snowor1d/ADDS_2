#!/usr/bin/env python3
"""The real-map pipeline, start to finish.

The stages exist as separate, re-runnable steps because that is the shape of
the question: the generator is being tuned toward a target distribution, and
the only way to know whether a change helped is to re-measure the gap. Each
stage writes its output to disk so a later stage never silently depends on a
memory state that is gone.

  resolve   place names -> coordinates, with the geocoder's answer recorded
  download  Geofabrik regional extracts, once per region
  collect   crop each site at each size, measure it, keep or reject it
  report    corpus against generator, per statistic, with the knob to turn
  export    usable crops -> playable levels for holdout or imitation training
  compare   the real map beside the layout, to check the extraction by eye

Typical first run:

    python3 -m cli.ADDS_AS_osm_pipeline resolve
    python3 -m cli.ADDS_AS_osm_pipeline download
    python3 -m cli.ADDS_AS_osm_pipeline collect
    python3 -m cli.ADDS_AS_osm_pipeline report --size 200
    python3 -m cli.ADDS_AS_osm_pipeline export

After changing a generator parameter, only the last two need repeating:

    python3 -m cli.ADDS_AS_osm_pipeline report --size 200

Which downtown, and where, is recorded in osm_corpus/sites.py and resolved
into osm_corpus/site_manifest.json. Building data is from OpenStreetMap,
(c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import os
import sys


def cmd_resolve(args):
    from osm_corpus.resolve import MANIFEST_PATH, resolve_all
    from osm_corpus.sites import SITES

    got = resolve_all(force=args.force)
    print(f"\n{len(got)} of {len(SITES)} sites resolved -> {MANIFEST_PATH}")


def cmd_download(args):
    from osm_corpus.fetch import GEOFABRIK_REGIONS, download_extract, extract_path

    regions = sorted(set(GEOFABRIK_REGIONS.values()))
    if args.regions:
        regions = [r for r in regions if any(a in r for a in args.regions)]
    total_mb = 0
    for i, region in enumerate(regions, 1):
        path = extract_path(region)
        if os.path.exists(path) and not args.force:
            mb = os.path.getsize(path) >> 20
            total_mb += mb
            print(f"[{i:2d}/{len(regions)}] have {region} ({mb} MiB)")
            continue
        print(f"[{i:2d}/{len(regions)}] downloading {region}")
        try:
            download_extract(region, force=args.force)
            total_mb += os.path.getsize(path) >> 20
        except Exception as e:
            print(f"    FAILED {region}: {type(e).__name__}: {e}")
    print(f"\n{total_mb/1024:.1f} GiB of extracts present")


def cmd_collect(args):
    from osm_corpus.collect import collect

    collect(keys=args.sites, crop_sizes=args.sizes,
            skip_download=args.skip_download,
            mode=args.mode, simplify_m=args.simplify)


def cmd_report(args):
    from osm_corpus.report import render

    text = render(size_m=args.size, n_generated=args.n_generated)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.out}")


def cmd_export(args):
    from osm_corpus.export import export_all

    export_all(crop_sizes=args.sizes)


def cmd_compare(args):
    from osm_corpus.compare import compare
    from osm_corpus.sites import SITES

    keys = args.sites or [s.key for s in SITES]
    path = compare(keys, size=args.size, out=args.out, cols=args.cols)
    print(f"wrote {path}")


def cmd_status(args):
    """Where the pipeline currently stands."""
    from osm_corpus.fetch import GEOFABRIK_REGIONS, extract_path
    from osm_corpus.resolve import MANIFEST_PATH, load_manifest
    from osm_corpus.sites import CROP_SIZES, SITES, summary

    print(summary())
    print()

    manifest = load_manifest()
    print(f"resolve : {len(manifest)} of {len(SITES)} sites -> "
          f"{'present' if manifest else 'MISSING, run resolve'}")

    regions = sorted(set(GEOFABRIK_REGIONS.values()))
    have = [r for r in regions if os.path.exists(extract_path(r))]
    size_gb = sum(os.path.getsize(extract_path(r)) for r in have) / 2**30
    print(f"download: {len(have)} of {len(regions)} regions, {size_gb:.1f} GiB")

    try:
        from osm_corpus.collect import load_corpus
        corpus = load_corpus()
        rows = corpus["rows"]
        ok = [r for r in rows if r.get("rejected") is None]
        print(f"collect : {len(ok)} usable of {len(rows)} crops "
              f"(built {corpus.get('built_at')})")
        print(f"          traversability={corpus.get('traversability', 'buildings')}"
              f" simplify={corpus.get('simplify_m', 1.0)} m")
        by_size = {}
        for r in ok:
            by_size[r["size_m"]] = by_size.get(r["size_m"], 0) + 1
        if by_size:
            print("          usable per size: " +
                  ", ".join(f"{k} m={v}" for k, v in sorted(by_size.items())))
        rejected = [r for r in rows if r.get("rejected") is not None]
        if rejected:
            print(f"          {len(rejected)} rejected; worst offenders:")
            for r in rejected[:8]:
                print(f"            {r['site']:24s} {r['size_m']:5} m  {r['rejected']}")
    except Exception as e:
        print(f"collect : none ({type(e).__name__})")

    from paths import at

    levels = at("osm_corpus", "data", "levels", "real_levels.json")
    if os.path.exists(levels):
        import json
        with open(levels, encoding="utf-8") as f:
            payload = json.load(f)
        lv_mode = payload.get("traversability") or "buildings"
        print(f"export  : {payload['n_levels']} playable levels, "
              f"traversability={lv_mode}")
        # The two files are written by different stages, so they can disagree
        # about how traversable space is defined. Nothing downstream notices:
        # the levels run, the statistics read, and every number describes a
        # different map from the one being simulated.
        try:
            corpus_mode = corpus.get("traversability", "buildings")
        except NameError:
            corpus_mode = None
        if corpus_mode and lv_mode != corpus_mode:
            print(f"          STALE: corpus is '{corpus_mode}'. "
                  f"Run: python3 -m cli.ADDS_AS_osm_pipeline export")
    else:
        print("export  : none")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("resolve", help="place names -> coordinates")
    p.add_argument("--force", action="store_true", help="re-resolve everything")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("download", help="Geofabrik regional extracts")
    p.add_argument("--regions", nargs="+", default=None,
                   help="only regions whose path contains one of these")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("collect", help="crop, measure, keep or reject")
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    p.add_argument("--skip-download", action="store_true",
                   help="use Overpass where an extract is missing")
    p.add_argument("--mode", choices=["roads", "buildings"], default=None,
                   help="traversable space from the road network or from the "
                        "gaps between buildings (default: config)")
    p.add_argument("--simplify", type=float, default=None,
                   help="block outline tolerance in metres (default: config)")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("report", help="corpus against generator")
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--n-generated", type=int, default=40)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("export", help="crops -> playable levels")
    p.add_argument("--sizes", nargs="+", type=int, default=None)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("compare", help="real map beside the extracted layout")
    p.add_argument("--sites", nargs="+", default=None)
    p.add_argument("--size", type=int, default=200)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--out", default="osm_compare.png")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("status", help="where the pipeline stands")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
