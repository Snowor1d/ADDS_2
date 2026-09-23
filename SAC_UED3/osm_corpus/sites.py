"""The corpus: which downtown, and where.

This file is the record of what the real-map corpus is made of. Every site is
named by a human-readable place query rather than by raw coordinates, and the
resolve stage turns that query into a coordinate through Nominatim and writes
the result, including the OSM element it matched, into the manifest. So the
provenance of every crop is auditable: you can see which place was asked for,
what OSM object answered, and when.

Coordinates are therefore deliberately NOT hardcoded here. A `center` is only
filled in when a query resolves to the wrong spot and has to be pinned by hand,
and when that happens the reason belongs in `notes`.

Site choice follows the brief: famous downtown streets in major world cities.
Queries name a street rather than a district on purpose. A district name
geocodes to its centroid, and a downtown centroid is very often a plaza, a
park or a station forecourt: "Mitte, Berlin" lands near Alexanderplatz and
yields one building in a 100 m crop. A street name lands inside the fabric the
corpus is meant to describe.
They are also spread deliberately across urban morphologies, because the point
of the corpus is to describe the range of real layouts the generator has to
cover, and a corpus of twenty grid-plan downtowns would understate that range.
The `morphology` tag records the intent so a later bias in the statistics can
be traced back to the sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass
class Site:
    key: str                      # stable id, used in filenames
    query: str                    # what gets sent to the geocoder
    city: str
    country: str
    continent: str
    morphology: str               # see MORPHOLOGIES
    notes: str = ""

    # Filled in only to override the geocoder. (lat, lon) in degrees.
    center: Optional[Tuple[float, float]] = None

    # Exit polygons are the researcher's choice, not a property of the map, so
    # they are specified per site when wanted. Each entry is a ring of (x, y)
    # in metres within the crop, origin at the crop's south-west corner.
    # Left empty, the export stage places exits automatically so the level is
    # at least runnable, and says so in the output.
    exits: List[List[Tuple[float, float]]] = field(default_factory=list)


# Why each class is here, in terms of what it does to the free space the robot
# has to guide a crowd through.
MORPHOLOGIES = {
    "grid": "regular blocks, wide straight streets, long sight lines",
    "organic": "irregular medieval or unplanned fabric, narrow winding streets",
    "medina": "very high coverage, alley-scale gaps, dead ends",
    "superblock": "few very large footprints, wide setbacks, open interiors",
    "boulevard": "angled radial intersections cutting across blocks",
    "colonial_grid": "grid with large courtyard blocks and arcaded frontage",
    "lowrise_dense": "many small footprints, fine-grained street network",
}

SITES: Sequence[Site] = (
    # --- named in the brief ---
    Site("shibuya", "Shibuya Crossing, Shibuya, Tokyo", "Tokyo", "Japan",
         "Asia", "lowrise_dense",
         "Dense fine-grained blocks around a major crossing."),
    Site("times_square", "Times Square, Manhattan, New York", "New York", "USA",
         "North America", "grid",
         "Manhattan grid with very large footprints."),
    Site("covent_garden", "Neal Street, Covent Garden, London", "London", "UK",
         "Europe", "organic",
         "Irregular pre-grid London fabric."),
    Site("pigalle", "Pigalle, Paris", "Paris", "France",
         "Europe", "boulevard",
         "Haussmann blocks meeting angled boulevards."),
    Site("eixample", "Passeig de Gracia, Eixample, Barcelona", "Barcelona", "Spain",
         "Europe", "grid",
         "Cerda grid, chamfered corners, uniform block size. The query keeps "
         "the article on purpose: 'Eixample, Barcelona' resolves to a "
         "same-named district in Alella, which is in Barcelona province, and "
         "adding 'Catalunya, Spain' does not help."),
    Site("mitte", "Rosenthaler Strasse, Mitte, Berlin", "Berlin", "Germany",
         "Europe", "grid",
         "Large perimeter blocks with courtyards."),
    Site("myeongdong", "Myeongdong, Jung-gu, Seoul", "Seoul", "South Korea",
         "Asia", "lowrise_dense",
         "Very fine-grained commercial blocks, alley network."),
    Site("vila_madalena", "Rua Aspicuelta, Vila Madalena, Sao Paulo", "Sao Paulo", "Brazil",
         "South America", "lowrise_dense",
         "Hilly low-rise grid, irregular parcel sizes."),

    # --- added for morphological and geographic spread ---
    Site("dotonbori", "Dotonbori, Osaka", "Osaka", "Japan",
         "Asia", "lowrise_dense", "Canal-side dense commercial strip."),
    Site("nanjing_road", "Nanjing Road Pedestrian Street, Shanghai", "Shanghai", "China",
         "Asia", "grid", "Wide pedestrian spine with large blocks."),
    # Query traps met while building this list, recorded so they are not
    # re-introduced: 'Divanyolu' matches a parking company, 'Avenida Francisco
    # I. Madero' matches a street in a far suburb, and Buenos Aires streets
    # carry no 'Calle' prefix in OSM.
    Site("chandni_chowk", "Chandni Chowk, Delhi", "Delhi", "India",
         "Asia", "medina", "Extremely high coverage, alley-scale gaps."),
    Site("khao_san", "Khao San Road, Bangkok", "Bangkok", "Thailand",
         "Asia", "lowrise_dense", "Low-rise dense with soi side-alleys."),
    Site("hongdae", "Hongik-ro, Mapo-gu, Seoul", "Seoul", "South Korea",
         "Asia", "lowrise_dense",
         "Contrast with Myeongdong at similar grain. The query resolves to a "
         "shop inside the district rather than its centroid; that is inside "
         "the fabric we want, but pin a centre here if the POI ever moves."),
    Site("sultanahmet", "Küçük Ayasofya Caddesi, Fatih, Istanbul", "Istanbul", "Turkey",
         "Europe", "organic", "Ottoman fabric with monumental open spaces."),
    Site("gracia", "Carrer Gran de Gracia, Barcelona", "Barcelona", "Spain",
         "Europe", "organic", "Pre-annexation village fabric; contrast with Eixample."),
    Site("kreuzberg", "Oranienstrasse, Kreuzberg, Berlin", "Berlin", "Germany",
         "Europe", "grid", "Contrast with Mitte at similar block scale."),
    Site("trastevere", "Via della Lungaretta, Trastevere, Rome", "Rome", "Italy",
         "Europe", "organic", "Narrow irregular streets, high coverage."),
    Site("grachtengordel", "Herengracht, Amsterdam", "Amsterdam", "Netherlands",
         "Europe", "lowrise_dense", "Canal ring; water breaks the free space."),
    Site("la_latina", "Calle de la Cava Baja, Madrid", "Madrid", "Spain",
         "Europe", "organic", "Dense irregular old-town fabric."),
    Site("khan_el_khalili", "Khan el-Khalili, Cairo", "Cairo", "Egypt",
         "Africa", "medina", "Souk fabric, alley widths near body scale."),
    Site("marrakesh_medina", "Jemaa el-Fnaa, Marrakesh", "Marrakesh", "Morocco",
         "Africa", "medina", "Medina alleys opening onto a large square."),
    Site("maboneng", "Fox Street, Jeppestown, Johannesburg", "Johannesburg", "South Africa",
         "Africa", "grid", "Industrial-to-mixed grid, large sheds."),
    Site("soho_nyc", "Spring Street, SoHo, Manhattan, New York", "New York", "USA",
         "North America", "grid", "Cast-iron loft blocks; contrast with Times Square."),
    Site("french_quarter", "Bourbon Street, New Orleans", "New Orleans", "USA",
         "North America", "colonial_grid", "Courtyard blocks on a colonial grid."),
    Site("gastown", "Water Street, Gastown, Vancouver", "Vancouver", "Canada",
         "North America", "grid", "Small-block grid with angled streets."),
    Site("centro_historico_cdmx", "Calle Madero, Centro, Mexico City", "Mexico City", "Mexico",
         "North America", "colonial_grid", "Spanish colonial grid, courtyard blocks."),
    Site("palermo_soho", "Thames, Palermo, Buenos Aires", "Buenos Aires", "Argentina",
         "South America", "grid", "Regular manzana grid, low-rise."),
    Site("surry_hills", "Crown Street, Surry Hills, Sydney", "Sydney", "Australia",
         "Oceania", "lowrise_dense", "Terrace-house fabric, fine grain."),
    Site("gangnam_superblock", "Gangnam-daero, Gangnam-gu, Seoul", "Seoul", "South Korea",
         "Asia", "superblock", "Large towers with wide setbacks."),
)

# Crop sizes in metres. The same centre at two sizes is two different
# structural problems, not the same one rescaled, so the corpus is cut at every
# scale the policy will meet. The training range is currently 70-140 m.
CROP_SIZES = (100, 200, 400)

# 1 km is planned but not collected. A crop that wide needs the multi-scale
# global observation that is still a TODO, so the levels would not be runnable,
# and each one costs minutes of extract scanning for statistics nothing reads
# yet. Add it back to CROP_SIZES when that observation lands, or ask for it on
# the command line:
#   python3 ADDS_AS_osm_pipeline.py collect --sizes 1000
TODO_CROP_SIZES = (1000,)


def by_key(key: str) -> Site:
    for s in SITES:
        if s.key == key:
            return s
    raise KeyError(f"no site named {key!r}; known: {[s.key for s in SITES]}")


def summary() -> str:
    lines = [f"{len(SITES)} sites, crop sizes {CROP_SIZES} m", ""]
    from collections import Counter
    for field_name in ("continent", "morphology"):
        counts = Counter(getattr(s, field_name) for s in SITES)
        lines.append(field_name + ": " + ", ".join(
            f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    for s in SITES:
        pinned = "" if s.center is None else f"  [pinned {s.center}]"
        print(f"  {s.key:24s} {s.morphology:14s} {s.city}, {s.country}{pinned}")
