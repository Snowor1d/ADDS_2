"""Stage 1: turn a place name into a coordinate, and record how.

The corpus has to be reproducible and auditable: months later it must be
possible to see which place each crop came from, what OSM object answered the
query, and when. So the geocoder's answer is written to a manifest rather than
being folded into a coordinate literal, and a site pinned by hand is marked as
pinned so it is never mistaken for a geocoded one.

Nominatim's usage policy asks for an identifying User-Agent and at most one
request per second, both of which are honoured here. Results are cached, so a
re-run costs nothing and the corpus does not depend on the geocoder staying
reachable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import requests

from osm_corpus.sites import SITES, Site, by_key

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "ADDS-research/0.1 (crowd evacuation RL corpus; OSM data (c) OpenStreetMap contributors, ODbL)"
MIN_REQUEST_INTERVAL_S = 1.1     # Nominatim asks for <= 1 req/s

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "site_manifest.json")

_last_request = [0.0]


@dataclass
class Resolved:
    key: str
    query: str
    city: str
    country: str
    continent: str
    morphology: str
    notes: str
    lat: float
    lon: float
    source: str                  # "nominatim" | "pinned"
    osm_type: Optional[str] = None
    osm_id: Optional[int] = None
    display_name: Optional[str] = None
    resolved_at: Optional[str] = None


def _throttle():
    gap = time.time() - _last_request[0]
    if gap < MIN_REQUEST_INTERVAL_S:
        time.sleep(MIN_REQUEST_INTERVAL_S - gap)
    _last_request[0] = time.time()


def geocode(query: str, timeout: float = 30.0) -> Optional[dict]:
    _throttle()
    r = requests.get(
        NOMINATIM,
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    r.raise_for_status()
    hits = r.json()
    return hits[0] if hits else None


def resolve_site(site: Site) -> Optional[Resolved]:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    common = dict(key=site.key, query=site.query, city=site.city,
                  country=site.country, continent=site.continent,
                  morphology=site.morphology, notes=site.notes,
                  resolved_at=stamp)

    if site.center is not None:
        lat, lon = site.center
        return Resolved(lat=float(lat), lon=float(lon), source="pinned", **common)

    hit = geocode(site.query)
    if hit is None:
        return None
    return Resolved(
        lat=float(hit["lat"]), lon=float(hit["lon"]), source="nominatim",
        osm_type=hit.get("osm_type"), osm_id=hit.get("osm_id"),
        display_name=hit.get("display_name"), **common,
    )


def load_manifest(path: str = MANIFEST_PATH) -> Dict[str, Resolved]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: Resolved(**v) for k, v in raw.get("sites", {}).items()}


def save_manifest(entries: Dict[str, Resolved], path: str = MANIFEST_PATH) -> None:
    payload = {
        "_about": (
            "Resolved centres for the real-map corpus. Written by "
            "osm_corpus.resolve; edit osm_corpus/sites.py rather than this "
            "file. Place data from OpenStreetMap, (c) OpenStreetMap "
            "contributors, ODbL 1.0."
        ),
        "sites": {k: asdict(v) for k, v in sorted(entries.items())},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def resolve_all(force: bool = False, path: str = MANIFEST_PATH) -> Dict[str, Resolved]:
    """Resolve every site, skipping ones already in the manifest."""
    entries = {} if force else load_manifest(path)
    failures = []
    for site in SITES:
        if site.key in entries and not force:
            continue
        try:
            got = resolve_site(site)
        except Exception as e:
            failures.append((site.key, f"{type(e).__name__}: {e}"))
            continue
        if got is None:
            failures.append((site.key, "no geocoder match"))
            continue
        entries[site.key] = got
        print(f"  {site.key:24s} {got.lat:9.5f} {got.lon:10.5f}  {got.source}"
              f"  {(got.display_name or '')[:52]}")
    save_manifest(entries, path)
    if failures:
        print("\nunresolved (pin a centre in sites.py to fix):")
        for key, why in failures:
            print(f"  {key:24s} {why}")
    return entries


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    got = resolve_all(force=force)
    print(f"\n{len(got)} of {len(SITES)} sites resolved -> {MANIFEST_PATH}")
