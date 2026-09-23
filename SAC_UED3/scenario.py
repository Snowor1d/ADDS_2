"""Which level to watch or play, decided by config.

One place, because two programs ask the same question. run_sim.py had these
three functions and ADDS_AS_HumanPlay.py had none: it built a numbered map
with its own hardcoded map id, its own step limit and the crowd size from
config, so the viewer and the human-play script were showing different tasks.
Human play was still on the boundary-exit formulation that the hazard zone
replaced.

SIM_SOURCE and the SIM_* settings in config decide the level, and everything
downstream takes the world size, the crowd size and the robot count from the
level rather than from constants.
"""

from __future__ import annotations

from config import (SIM_DIFFICULTY, SIM_MORPHOLOGY, SIM_REAL_SITE,
                    SIM_REAL_SIZE, SIM_ROBOTS, SIM_SEED, SIM_SIZE,
                    SIM_SOURCE)


def load_real_level(site: str, size: int):
    """Pick one crop out of the exported real-map levels."""
    from osm_corpus.export import load_levels

    levels = load_levels()
    for lv in levels:
        if lv.site_key == site and int(lv.width) == int(size):
            return lv
    by_site = {}
    for l in levels:
        by_site.setdefault(l.site_key, []).append(int(l.width))
    lines = [f"no exported real level '{site}' at {size} m.",
             "Set SIM_REAL_SITE and SIM_REAL_SIZE in config.py to one of:"]
    for key in sorted(by_site):
        lines.append(f"  {key:24s} sizes {sorted(by_site[key])}")
    lines.append("If the site is missing entirely, run:")
    lines.append("  python3 ADDS_AS_osm_pipeline.py collect")
    lines.append("  python3 ADDS_AS_osm_pipeline.py export")
    raise SystemExit("\n".join(lines))


def attach_hazard(level, rng=None):
    """Put a hazard on a level that has none, and drop its boundary exits.

    Real crops are exported as geometry, not as tasks: the export stage places
    boundary exits because that is what the old formulation needed, and it
    knows nothing about hazards. Rather than re-export the corpus, the hazard
    is drawn here the same way the generator draws one, so a real downtown can
    be watched under the current task.

    Retried, because a zone can land where the crop has no walkable ground
    inside it, or in a pocket whose only way out is too narrow for the robot.
    """
    import random as _random

    from citygen.plan import CityPlan
    from citygen.validate import check
    from config import (SIM_DIFFICULTY, SIM_ROBOTS, UED_DANGER_AREA_RANGE,
                        UED_DANGER_INSIDE_FRACTION, UED_DANGER_SHAPES,
                        UED_DIFFICULTY_RANGE)
    from danger import sample_zone

    if getattr(level, "danger", None) is not None:
        return level

    rng = rng or _random.Random()
    lo_d, hi_d = UED_DIFFICULTY_RANGE
    t = (float(SIM_DIFFICULTY) - lo_d) / max(1.0, float(hi_d - lo_d))
    lo_a, hi_a = UED_DANGER_AREA_RANGE
    area = lo_a + (hi_a - lo_a) * t

    # A stand-in plan so the validator can rasterise the crop's obstacles. The
    # real crop has no street plan of its own: its geometry came from OSM, not
    # from the generator.
    plan = CityPlan(width=int(level.width), height=int(level.height),
                    morphology="osm", development=1.0)
    rings = [[list(p) for p in ring] for ring in level.obstacles]
    plan.render = lambda simplify_m=None, _r=rings: (_r, None)

    for _ in range(30):
        shape = UED_DANGER_SHAPES[rng.randrange(len(UED_DANGER_SHAPES))]
        zone = sample_zone(rng, float(level.width), float(level.height),
                           area, shape)
        ok, _why = check(plan, (), zone)
        if ok:
            level.danger = zone
            level.exits = []
            lo_f, hi_f = UED_DANGER_INSIDE_FRACTION
            level.inside_fraction = rng.uniform(lo_f, hi_f)
            level.robot_num = int(SIM_ROBOTS)
            return level
    raise SystemExit(
        f"could not place a hazard on {getattr(level, 'site_key', 'this crop')} "
        f"covering {area:.2f} of it; try a smaller SIM_DIFFICULTY")


def build_curriculum_level():
    """A level of the kind training runs on: city fabric plus a hazard.

    The viewer had no path to one. It could show a real OSM crop or a numbered
    map from map_infos, and both of those carry boundary exits and no hazard,
    so running it showed the task that was replaced rather than the one being
    trained.
    """
    import random as _random

    from config import (SIM_DIFFICULTY, SIM_MORPHOLOGY, SIM_ROBOTS, SIM_SEED,
                        SIM_SIZE)
    from ued.level import generate_city_level

    seed = _random.randrange(1 << 30) if SIM_SEED is None else int(SIM_SEED)
    rng = _random.Random(seed)
    level = generate_city_level(
        rng, difficulty=int(SIM_DIFFICULTY),
        crowd_size=None,   # density on the walkable ground; crowd_density.py
        width=int(SIM_SIZE), height=int(SIM_SIZE),
        seed=seed, morphology=SIM_MORPHOLOGY, robot_num=int(SIM_ROBOTS))
    print(f"[run_sim] curriculum level seed={seed} "
          f"morphology={level.plan.morphology} difficulty={SIM_DIFFICULTY} "
          f"{level.width} m, {len(level.obstacles)} blocks, "
          f"{level.crowd_size} pedestrians "
          f"({level.crowd_density:.3f}/m2 walkable), "
          f"{level.robot_num} robot(s), "
          f"{level.danger.shape} hazard covering "
          f"{level.danger.area() / (level.width * level.height):.2f} of the crop")
    return level

def level_from_config(source: str = None, site: str = None,
                      size: int = None):
    """The level the SIM_* settings describe, and a tag naming it.

    Returns (level, tag). The level is None for "numbered", where the
    simulator loads the map itself by id from map_infos and there is no
    hazard: that is the task this project replaced, kept for the map editors.
    The tag names the scenario for output folders, so results from a real
    crop and from a generated level cannot land in the same directory.
    """
    source = source or SIM_SOURCE
    if source == "curriculum":
        level = build_curriculum_level()
        return level, f"curriculum_d{int(SIM_DIFFICULTY)}_{int(SIM_SIZE)}m"
    if source == "real":
        key = site or SIM_REAL_SITE
        metres = int(size or SIM_REAL_SIZE)
        level = attach_hazard(load_real_level(key, metres))
        return level, f"real_{key}_{metres}m"
    if source == "numbered":
        from config import MAP_NUM

        return None, f"map_{int(MAP_NUM)}"
    raise SystemExit(f"unknown SIM_SOURCE {source!r}; expected "
                     f"'real', 'curriculum' or 'numbered'")
