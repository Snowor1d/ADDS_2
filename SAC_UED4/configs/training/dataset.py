"""Training on the real OSM corpus (TRAIN_MAP_SOURCE = "dataset").

Every episode draws one (site, size) pair uniformly from
DATASET_SITES x DATASET_SIZES_M and turns the stored crop into a task with
the parameters below: a fresh hazard, a crowd at the target density, how
perceptible the hazard is, how many were warned, team size and symmetry.
The curriculum mode has its own copy of these in configs/training/ued.py;
names here start with DATASET_.

The final zero-shot site, and any crop that overlaps it on the ground, may
not be listed; nor may an auxiliary evaluation site. Every (site, size) pair
must exist in the corpus (surry_hills has no 400 m crop). The stored
headcount is not used: the export capped every 400 m crop at 800 people.

Constants only; importing this module has no side effects.
"""

# --------------- WHICH MAPS ---------------
DATASET_SITES = (
    "dotonbori", "hongdae", "gangnam_superblock", "nanjing_road", "khao_san",
    "covent_garden", "kreuzberg", "mitte", "eixample", "gracia", "la_latina",
    "trastevere", "grachtengordel", "sultanahmet", "soho_nyc",
    "french_quarter", "gastown", "centro_historico_cdmx", "maboneng",
    "vila_madalena",
)
DATASET_SIZES_M = (200,)
# Persons per square metre of walkable ground per size. None means the
# environment's CROWD_DENSITY_RANGE.
DATASET_DENSITY_BY_SIZE = {100: None, 200: None}

# --------------- TASK PARAMETERS PER EPISODE ---------------
# Hazard area as a share of the crop, and its shape. A tuple of "circle" and
# "rect"; "street" is refused on OSM crops, which have no street plan to lay
# it along. A one-element tuple needs its comma.
DATASET_DANGER_AREA_RANGE = (0.04, 0.2)
DATASET_DANGER_SHAPES = ("circle",)
# Only read when CROWD_SPAWN = "inside_fraction".
DATASET_DANGER_INSIDE_FRACTION = (0.5, 1.0)
# How detectable the hazard is (near 1 a fire, near 0 a gas leak); see
# docs/crowd_awareness_design.md.
DATASET_DANGER_PERCEPTIBILITY = (0.5, 0.5)
# Share of the crowd already cued when the episode starts.
DATASET_PRIOR_INFORMED_FRACTION = (0.0, 0.3)
# Robots per episode, inclusive, capped by MAX_ROBOTS.
DATASET_ROBOT_RANGE = (3, 3)
# Geometric augmentation: whether each episode's crop is rotated or reflected.
# True draws one symmetry per episode from the list below; False always uses
# the crop as exported ("identity"). The hazard is transformed with the
# buildings either way.
DATASET_AUGMENTATION = False
DATASET_AUGMENTATION_TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)
