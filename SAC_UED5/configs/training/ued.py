"""Training on curriculum levels (TRAIN_MAP_SOURCE = "ued").

Everything the ACCEL-style curriculum needs, and the task parameters every
generated citygen level is drawn with: map size and crowd density, hazard
area and shape, how perceptible the hazard is, how many people were warned in
advance, team size and symmetry. The dataset mode has its own copy of the
task parameters in configs/training/dataset.py, so the two can be tuned
independently; names here start with UED_.

Constants only; importing this module has no side effects.
"""

# --------------- UNSUPERVISED ENVIRONMENT DESIGN -----------------
# ACCEL-style evolutionary curriculum: keep a population of levels, score them,
# and breed from the ones the agent is on the edge of solving. The plan
# document says why the minimax-regret framing is not adopted.
# "accel" breeds and replays levels; "dr" draws a fresh random level every
# episode (plain domain randomisation over citygen) with the same task
# parameters below.
UED_METHOD = "accel"             # "dr" (uniform over generated levels) | "accel"
# "learnability" is p*(1-p) over a level's success history and needs several
# trials; "maxmc" scores a single episode from the SAC critic; "hybrid" uses
# maxmc until UED_MIN_TRIALS trials exist, covering a fresh child's cold start.
UED_SCORE = "hybrid"             # "learnability" | "maxmc" | "hybrid"
UED_MIN_TRIALS = 5               # fewer cannot separate a frontier level from noise
UED_TRIAL_HISTORY = 10           # sliding window used for the success rate
# Robust-PLR's replay-only rule cannot hold exactly against a shared off-policy
# buffer. An experiment axis rather than a silent choice: when True,
# transitions from fresh levels are scored but not stored.
UED_REPLAY_ONLY_UPDATES = False
# Sized by how many episodes a level needs before its score means anything.
# Population size, UED_TRIAL_HISTORY and UED_TRIAL_MAX_AGE have to stay
# consistent or no level ever reaches the breeding bar; CurriculumTimingTest
# holds that invariant.
UED_POP_SIZE = 200
# Children start on their parent's score, so without a cap one lucky ancestor
# crowds the population with its own lineage.
UED_MAX_CHILDREN_PER_LEVEL = 3
UED_P_NEW_START = 0.5            # probability of a fresh level early on
UED_P_NEW_END = 0.1              # ... once the population has filled out
UED_P_NEW_DECAY_EPISODES = 5000
UED_STALENESS_COEF = 0.2         # rho: weight on the staleness distribution
UED_SCORE_TEMPERATURE = 1.0      # rank-based sampling temperature
# Breed only from the top quarter by score percentile. Half turned the
# population over faster than levels could accumulate trials.
UED_MUTATE_THRESHOLD = 0.75
UED_MUTATIONS_PER_CHILD = (1, 3)
UED_MUTATE_MAX_TRIES = 40
# A level counts as solved when everyone is out within this multiple of its own
# free-flow estimate. Normalised per level, because a hand-picked constant
# makes p saturate and takes learnability with it. Calibrate from the logged
# UED/mean_evac_ratio; raw times are stored per level, so a new K can be
# applied to the existing history.
UED_SUCCESS_K = 5.0
UED_DIFFICULTY_RANGE = (0, 6)    # 0 is an empty field, 6 real downtown density

# --------------- DANGER ZONE (the task the robots are given) ---------------
# The crowd must leave a hazard region and be kept from wandering back in.
# This replaced the boundary exits, and it is a different problem: safety is
# everywhere except one region, so the crowd disperses instead of converging.
# The zone is static within an episode; a spreading hazard has no stable
# free-flow reference to normalise the success criterion by.
#
# Size as a share of crop area rather than a radius, because that is what
# compares across map sizes.
UED_DANGER_AREA_RANGE = (0.04, 0.30)
# A circle escapes uniformly; a rectangle is a stretch of street, where
# leaving sideways is quick and lengthwise is not. "street" lies along one of
# the plan's own streets, at its width and angle, which is the shape most real
# hazards in a city take.
# A tuple: ("circle") without the comma is the string "circle", and one
# character of it was drawn as the shape, which the zone sampler then treated
# as a rectangle. The resolved configuration now refuses anything else.
UED_DANGER_SHAPES = ("circle",)
# Only read when CROWD_SPAWN = "inside_fraction".
UED_DANGER_INSIDE_FRACTION = (0.5, 1.0)

# --------------- CROWD HAZARD AWARENESS ---------------
# Design and justification: docs/crowd_awareness_design.md and
# docs/crowd_awareness_implementation.md.
#
# How detectable the hazard is, 0 to 1: a property of the hazard, so a
# curriculum variable. Near 1 is a fire, near 0 a gas leak, where the only
# channels are word of mouth and the robots. It sets both how fast direct
# perception fires and whether perceiving gives a global picture or only the
# spot touched. Watch for degeneration: the curriculum can raise difficulty by
# starving the crowd of information instead of by geometry, so if the logged
# distribution piles up at the floor, narrow this or close the axis.
UED_DANGER_PERCEPTIBILITY = (0.05, 0.95)
# Share of the crowd already cued at the start: an alarm, the news, someone
# off-map. Zero is a hazard nobody has announced.
UED_PRIOR_INFORMED_FRACTION = (0.0, 0.3)
# How many robots a fresh level gets, inclusive, capped by MAX_ROBOTS. A
# curriculum axis, so the mask is exercised rather than a formality.
UED_ROBOT_RANGE = (1, 3)
# World sizes a fresh level is drawn at, and how often. Empty means uniform
# over UED_MAP_SIZE_RANGE instead.
UED_MAP_SIZES_M = (100, 200, 400)
UED_MAP_SIZE_WEIGHTS = (0.6, 0.3, 0.1)
# Persons per square metre of walkable ground per size. None means the
# environment's CROWD_DENSITY_RANGE.
UED_DENSITY_BY_SIZE = {100: None, 200: None, 400: None}
# The continuous size range: the fallback above, and the bounds the canvas
# mutation may resize a level within.
UED_MAP_SIZE_RANGE = (70, 140)
# Square only for now: aspect ratio interacts with the D4 augmentation, which
# swaps width and height on the odd rotations.
UED_MAP_SQUARE = True
# The one generator family: citygen lays out a street network for one of the
# seven morphologies, then decides which blocks are built up. It replaced
# random_map and street_map, whose obstacle polygons the curriculum edited
# directly; those edits were morphology-blind, so a generated downtown stopped
# looking like one within a few generations. Every training, validation and
# viewer map is a citygen level or an OSM crop; the numbered maps are gone.
UED_GENERATOR = "citygen"
# Uniform, so the curriculum's own score decides where to spend episodes.
# Weight a morphology up only to study it deliberately.
UED_MORPHOLOGY_WEIGHTS = None    # None = uniform over citygen.MORPHOLOGIES
UED_WARMUP_EPISODES = 500        # pure DR until the population has some mass
# Level scores are recorded only once exploration has decayed below this. A
# success rate measured while actions are mostly random describes what a random
# walker solves, not where the policy's ability ends. Episodes still run and
# still train; only their outcomes are withheld from the curriculum.
UED_SCORE_MAX_EPSILON = 0.25
# Outcomes older than this many episodes are dropped from a level's history, so
# a score stays attached to a recognisable policy. A fixed trial count spanned
# more episodes than the whole epsilon decay.
UED_TRIAL_MAX_AGE = 3000
UED_LEVEL_QUEUE_SIZE = 4         # per-worker prefetch depth
UED_PRODUCER_TARGET = 64         # fresh levels kept ready on the producer thread
UED_SNAPSHOT_EPISODES = 200      # how often population pictures go to TensorBoard


# Geometric augmentation: whether a fresh level is rotated or reflected.
# True draws one symmetry per level from the list below; False always uses
# the level as generated ("identity"). Evaluation always runs a level as
# generated, whatever this says.
UED_AUGMENTATION = True
UED_AUGMENTATION_TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)
