"""Stage 2: get OSM building data for a site.

Two sources behind one interface.

Regional extracts (.osm.pbf from Geofabrik) are the primary path. They are the
only practical way to collect the hundreds of crops the statistics stage needs,
they cost one download per region however many crops come out of it, and a
pinned file makes the corpus reproducible without depending on a live server.

The Overpass API is the fallback for a site whose extract has not been
downloaded. It is fine for a handful of crops and unusable for bulk: rapid
sequential queries start returning HTTP 429 after about five requests.

Either way the output is the same: building rings in WGS84 degrees, which the
extract stage projects and crops.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

USER_AGENT = ("ADDS-research/0.1 (crowd evacuation RL corpus; "
              "OSM data (c) OpenStreetMap contributors, ODbL)")
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

# Built artefacts live with the project; downloaded data does not. The corpus
# index and the exported levels are small and are what everything downstream
# reads, so they stay here. The regional extracts are a 12.5 GiB cache that
# only the collect stage touches, so they live under config.OSM_CACHE_DIR.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def extract_dir() -> str:
    """Where the regional .osm.pbf files are kept.

    A function rather than a module constant so importing this module does not
    pull in `config`, which carries the whole training configuration down to
    the CUDA device. The rest of the package reads config the same way, inside
    the function that needs it.
    """
    from config import OSM_CACHE_DIR

    return os.path.join(OSM_CACHE_DIR, "extracts")

# Geofabrik regions, chosen per site so a download covers what it needs and no
# more. Keys are site keys; values are paths under the Geofabrik download root.
# A site missing from here falls back to Overpass.
GEOFABRIK_REGIONS: Dict[str, str] = {
    "shibuya": "asia/japan/kanto",
    "dotonbori": "asia/japan/kansai",
    "times_square": "north-america/us/new-york",
    "soho_nyc": "north-america/us/new-york",
    # Geofabrik serves the UK under "united-kingdom"; the "great-britain"
    # path still answers with HTTP 200 and a zero-length body, so a naive
    # existence check passes and the download silently produces nothing.
    "covent_garden": "europe/united-kingdom/england/greater-london",
    "pigalle": "europe/france/ile-de-france",
    "eixample": "europe/spain",
    "gracia": "europe/spain",
    "la_latina": "europe/spain",
    "mitte": "europe/germany/berlin",
    "kreuzberg": "europe/germany/berlin",
    "myeongdong": "asia/south-korea",
    "hongdae": "asia/south-korea",
    "gangnam_superblock": "asia/south-korea",
    "vila_madalena": "south-america/brazil/sudeste",
    "nanjing_road": "asia/china",
    "chandni_chowk": "asia/india/northern-zone",
    "khao_san": "asia/thailand",
    "sultanahmet": "europe/turkey",
    "trastevere": "europe/italy/centro",
    "grachtengordel": "europe/netherlands",
    "khan_el_khalili": "africa/egypt",
    "marrakesh_medina": "africa/morocco",
    "maboneng": "africa/south-africa",
    "french_quarter": "north-america/us/louisiana",
    "gastown": "north-america/canada/british-columbia",
    "centro_historico_cdmx": "north-america/mexico",
    "palermo_soho": "south-america/argentina",
    "surry_hills": "australia-oceania/australia",
}

GEOFABRIK_ROOT = "https://download.geofabrik.de"


@dataclass
class BuildingRings:
    """Building outlines as (lon, lat) rings, plus where they came from."""
    rings: List[List[Tuple[float, float]]]
    source: str            # "pbf:<region>" | "overpass"
    n_raw: int


def meters_per_degree(lat: float) -> Tuple[float, float]:
    """Local metres per degree of latitude and longitude.

    Good to well under a metre over a 1 km crop, which is all the corpus
    needs, and avoids a pyproj dependency.
    """
    p = math.radians(lat)
    m_lat = 111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    m_lon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
    return m_lat, m_lon


def bbox_for(lat: float, lon: float, size_m: float, margin: float = 1.15):
    """South, west, north, east for a crop, with margin for clipped buildings."""
    m_lat, m_lon = meters_per_degree(lat)
    half = size_m * margin / 2.0
    dlat, dlon = half / m_lat, half / m_lon
    return lat - dlat, lon - dlon, lat + dlat, lon + dlon


# ---------------------------------------------------------------------------
# Overpass
# ---------------------------------------------------------------------------

def fetch_overpass(lat: float, lon: float, size_m: float,
                   retries: int = 4, pause_s: float = 3.0) -> BuildingRings:
    s, w, n, e = bbox_for(lat, lon, size_m)
    query = (f'[out:json][timeout:120];'
             f'(way["building"]({s},{w},{n},{e});'
             f'relation["building"]({s},{w},{n},{e}););'
             f'out geom;')

    last = None
    for attempt in range(retries):
        endpoint = OVERPASS_ENDPOINTS[attempt % len(OVERPASS_ENDPOINTS)]
        try:
            r = requests.post(endpoint, data={"data": query},
                              headers={"User-Agent": USER_AGENT}, timeout=180)
            if r.status_code == 429:
                # Rate limited. Backing off is the only correct response; the
                # public endpoint starts refusing after a few rapid queries.
                time.sleep(pause_s * (2 ** attempt))
                last = "429"
                continue
            r.raise_for_status()
            elements = r.json().get("elements", [])
            rings = []
            for el in elements:
                if not el.get("tags", {}).get("building"):
                    continue
                geom = el.get("geometry") or []
                if len(geom) < 4:
                    continue
                rings.append([(p["lon"], p["lat"]) for p in geom
                              if p.get("lon") is not None])
            return BuildingRings(rings=rings, source="overpass",
                                 n_raw=len(elements))
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(pause_s * (2 ** attempt))
    raise RuntimeError(f"overpass failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------
# Regional extracts
# ---------------------------------------------------------------------------

def extract_path(region: str) -> str:
    return os.path.join(extract_dir(),
                        region.replace("/", "_") + "-latest.osm.pbf")


def extract_url(region: str) -> str:
    return f"{GEOFABRIK_ROOT}/{region}-latest.osm.pbf"


def download_extract(region: str, force: bool = False) -> str:
    """Download a Geofabrik region once; later crops reuse the file."""
    path = extract_path(region)
    if os.path.exists(path) and not force:
        return path
    os.makedirs(extract_dir(), exist_ok=True)
    url = extract_url(region)
    tmp = path + ".part"
    with requests.get(url, headers={"User-Agent": USER_AGENT},
                      stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        last_mark = [0]
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                # Every 16 MiB, not every chunk: a per-chunk line turns a
                # hundred-megabyte download into a hundred lines of log.
                if total and (done >> 24) != (last_mark[0] >> 24):
                    last_mark[0] = done
                    print(f"    {region}: {done>>20} / {total>>20} MiB "
                          f"({100.0*done/total:4.1f}%)", flush=True)
        if total:
            print(f"    {region}: {total>>20} / {total>>20} MiB (100.0%)")
    os.replace(tmp, path)
    return path


def fetch_pbf(region: str, lat: float, lon: float, size_m: float) -> BuildingRings:
    """Read building ways inside the crop bbox out of a regional extract."""
    path = extract_path(region)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no extract for {region}; run the download step first ({extract_path(region)})")

    s, w, n, e = bbox_for(lat, lon, size_m)
    rings = _pbf_ways(path, (w, s, e, n))
    return BuildingRings(rings=rings, source=f"pbf:{region}", n_raw=len(rings))


def _pbf_ways(path: str, bbox: Tuple[float, float, float, float]):
    """Building outer rings inside bbox (west, south, east, north).

    Reads areas rather than ways. A building mapped as a multipolygon relation
    is not a closed way, so a way-only pass drops it: at one Berlin test crop
    that was four relations carrying a quarter of the built area, and the
    missing pieces are systematically the large complex buildings. osmium's
    area handler assembles closed ways and multipolygon relations into the
    same kind of object, so both arrive by the same path.
    """
    import osmium

    west, south, east, north = bbox
    rings: List[List[Tuple[float, float]]] = []

    # with_areas() yields every object type, so areas have to be picked out;
    # the key filter alone still lets the underlying nodes and ways through.
    processor = osmium.FileProcessor(path).with_areas() \
        .with_filter(osmium.filter.KeyFilter("building"))
    for area in processor:
        if not isinstance(area, osmium.osm.Area):
            continue
        if "building" not in area.tags:
            continue
        for outer in area.outer_rings():
            pts = []
            inside = False
            for node in outer:
                lon, lat = node.lon, node.lat
                pts.append((lon, lat))
                if west <= lon <= east and south <= lat <= north:
                    inside = True
            if inside and len(pts) >= 4:
                rings.append(pts)
    return rings


def fetch_site(key: str, lat: float, lon: float, size_m: float,
               prefer: str = "pbf") -> BuildingRings:
    """Buildings for one crop, from an extract if we have one."""
    region = GEOFABRIK_REGIONS.get(key)
    if prefer == "pbf" and region and os.path.exists(extract_path(region)):
        return fetch_pbf(region, lat, lon, size_m)
    return fetch_overpass(lat, lon, size_m)
