import os

# ------------- BASIC PARAMETERS ------------------
LR = 1e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1 #intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500
DEVICE = "cuda"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000
MAP_H = 100
MAP_W = 100 

#weight decay
WD_Q = 3e-4
WD_PI = 0.0

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 30
CROWD_NUMBER_MAX = 30
MAP_NUM = -1 #if not used, -2, if random, -1
#learning
# 50x50 : 6,7,26
# 70x70 : 50,53,54
# 100x100 : 105, 108, 128
# unseen maps : 500, 501, 502, 503
MAP_NUM_RANDOM = list(range(1000, 1300))

# MAP_NUM_RANDOM = [1001, 1004, 1007, 1010, 1013, 1016, 1019, 1022, 1025, 1028,
#  1031, 1034, 1037, 1040, 1043, 1046, 1049, 1052, 1055, 1058,
#  1061, 1064, 1067, 1070, 1073, 1076, 1079, 1082, 1085, 1088,
#  1091, 1094, 1097, 1100, 1103, 1106, 1109, 1112, 1115, 1118,
#  1121, 1124, 1127, 1130, 1133, 1136, 1139, 1142, 1145, 1148,
#  1151, 1154, 1157, 1160, 1163, 1166, 1169, 1172, 1175, 1178,
#  1181, 1184, 1187, 1190, 1193, 1196, 1199, 1202, 1205, 1208,
#  1211, 1214, 1217, 1220, 1223, 1226, 1229, 1232, 1235, 1238,
#  1241, 1244, 1247, 1250, 1253, 1256, 1259, 1262, 1265, 1268,
#  1271, 1274, 1277, 1280, 1283, 1286, 1289, 1292, 1295, 1298] # 100 maps

# MAP_NUM_RANDOM = [1006, 1018, 1031, 1047, 1059, 1072, 1085, 1093, 1108, 1124,
#  1136, 1149, 1162, 1175, 1188, 1201, 1213, 1227, 1239, 1254,
#  1266, 1273, 1281, 1287, 1289, 1293, 1296, 1298, 1299, 1300] # 30 maps

# MAP_NUM_RANDOM = [1014, 1042, 1068, 1095, 1127, 1153, 1186, 1219, 1264, 1298] # 10 maps



#MAP_NUM_RANDOM = [105, 108, 128]
#MAP_NUM_RANDOM = [1506]

# Apply one random geometric symmetry whenever a simulation map is created.
# Disable this for evaluation when the original JSON/map orientation is needed.
MAP_DATA_AUGMENTATION = True
MAP_AUGMENTATION_TRANSFORMS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "reflect",
    "reflect_rotate_90",
    "reflect_rotate_180",
    "reflect_rotate_270",
)

SCALE_CHECK = 0 # want to check reward scale?
ACTION_SCALE = 4
MAX_STEPS = 6000

ROBOT_BODY_RADIUS = 1 # m
AGENT_BODY_RADIUS = 0.5 # m
ROBOT_VISION = 10 # m
AGENT_VISION = 10 # m
EXIT_CONFIRM_RADIUS = 10
EXIT_CONFIRM_RADIUS_BONUS = 0
AGENT_SPEED_MEAN = 1.5 # m/s 
ROBOT_SPEED_MAX = 2 # m/s 
AGENT_TIME_STEP = 0.5
ROBOT_TIME_STEP = 0.5

# --------------- EPSILON-EXPLORATION ------------------
EPSILON_MIN = 0
START_EPSILON = 1 
SCHEDULER_TYPE = "l"
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 3000
START_DECAY_STEP = 500
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0
DECAY_MODE = 'episode'
RANDOM_EXIT = False


# --------------- REAL-MAP EXTRACTION (osm_corpus) ---------------
# How the traversable space of a real crop is decided.
#
# "roads" builds it from the road and footway network, widened to its
# carriageway, plus mapped pedestrian areas. Everything else is obstacle. This
# is what a guidance robot can actually drive on.
#
# "buildings" keeps the earlier behaviour: building footprints are the
# obstacles and everything else is walkable. That overstates where the robot
# can go, because the space between buildings in a real city is largely
# private plots, walled yards, car parks, planting, water and rail.
OSM_TRAVERSABILITY = "roads"      # "roads" | "buildings"

# Where the downloaded OpenStreetMap data is kept.
#
# Outside the project, because it is a cache and not a part of it. The regional
# extracts run to about 12.5 GiB for the 29 sites: one download covers a whole
# country or region so a single 200 m crop can be cut from it, which means
# three Seoul sites pull 274 MiB of South Korea. Keeping that inside the source
# tree made the project directory 13 GiB, of which everything that is actually
# the project came to under 20 MiB.
#
# Nothing downstream needs these files. The corpus index and the exported
# levels are built once and stand alone, so the simulator, the generator fit
# and the comparison sheet all run with this directory empty. It is needed
# again only to re-cut crops: adding a site, collecting the 1 km size, or
# changing OSM_SIMPLIFY_M.
#
# Set ADDS_OSM_CACHE to move it elsewhere, a larger disk for instance.
OSM_CACHE_DIR = os.environ.get(
    "ADDS_OSM_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "adds-osm"))

# How far block outlines may move when simplified, in metres. Buffered road
# edges arrive as dense arcs and the navmesh cost follows vertex count, so
# this is the main lever on simulation speed. Bounded by what a robot needs:
# move a wall further than about its own width and a passable corridor closes.
OSM_SIMPLIFY_M = 1.5

# --------------- NAVMESH RESOLUTION (simulation cost) ---------------
# Long obstacle and boundary edges get intermediate points inserted before
# triangulation, every this many metres, so the navmesh has triangles to route
# through rather than a few enormous ones.
#
# This is the strongest lever on environment construction cost, because the
# all-pairs shortest-path table over the mesh graph is cubic in the triangle
# count. Raising it coarsens the routing graph: the crowd still moves
# continuously, but waypoints sit further apart. Lower it for finer paths in
# small maps, raise it if the largest maps become too slow to build.
NAVMESH_SEGMENT_STEP_M = 20.0

# --------------- SIMULATION VIEWER (run_sim.py) ---------------
# Everything run_sim.py needs, so watching a map is a config edit rather than
# a command line to remember. The command line still overrides these when it
# is more convenient.
#
# Set SIM_REAL_SITE to a site key from osm_corpus/sites.py to watch a real
# downtown crop, or leave it None to drive the numbered maps in map_infos as
# before. Run `python3 ADDS_AS_osm_pipeline.py status` to see which sites have
# been exported.
SIM_REAL_SITE = "mitte"   # e.g. "kreuzberg", "mitte", "shibuya"
SIM_REAL_SIZE = 200           # crop size in metres, must be one that exists

# -------------- PATH -------------------
# Kept distinct from SAC_FE_RV3 so the UED run and the baseline run can share a
# machine without clobbering each other's logs, checkpoints or tensorboard port.
LOG_DIR = "Log_SAC_UED2"
#LOG_DIR = "SOTA_MODELS"
#LOG_DIR = "Log_test"
PORT_NUM = 7755

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20
ALPHA_START = 0.2 # in SAC
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHAPING -----------------

REWARD_A = 2 #reward_based_alived
REWARD_B = 0.003 #reward_based_all_agents_danger
REWARD_D = 2 #reward_based_penalty
REWARD_K = 6 #reward_penalty_collsion
REWARD_J = 0 #reward_based_all_agents_danger_root
REWARD_L = 1 #reward_based_farthest_agent_distance
REWARD_FIXED = -0.5

REWARD_I = 0 #reward_based_alived_root
REWARD_C = 0 #reward_based_gain
REWARD_E = 0 #reward_based_evacuated_with_robot
REWARD_F = 0 #reward_based_distance_from_near_agents
REWARD_G = 0 #reward_based_distance_from_near_agent_gain
REWARD_H = 0 #reward_based_gain_with_time_bonus
FINISHED_BONUS = 0


USING_TRAINED_MODEL = True
SHOW_CONTROLLED_CROWD = False
ENABLE_TIMER = True

N_ENVS = 4
UPDATES_PER_TRANSITION = 1
POLICY_BROADCAST_INTERVAL = 10

EGO_MAP_SIZE = 25
DOWNSAMPLE_MAP_SIZE = 50
FiLM_USE = True
EGO_USE = True

# Zero-shot evaluation (kept identical to DREAMER_MARL17_m for comparison).
# SAC_FE_RV3 is a single-robot policy, so only robot_num=1 is supported.
# Held-out JSON maps: 100x100, outside the 1000-1299 range the DR baseline
# trains on, so they test transfer to geometry no generator produced. Never
# add one of these to MAP_NUM_RANDOM.
ZSG_MAP = [1500, 1501, 1502, 1503]
ZSG_CYCLE_EPISODE = 500
ZSG_ITERATION = 10
ZSG_ROBOT_NUM = [1]

# The procedural half of the held-out set, built from fixed seeds by
# ued/holdout.py. This is the direct evidence for the generalisation claim, so
# it is evaluated alongside ZSG_MAP. Fewer iterations because there are 18 of
# them and every episode runs up to MAX_STEPS.
ZSG_HOLDOUT_LEVELS = True
# 72 held-out levels now, four size bands times six difficulties times three
# seeds, so each pass is 72 * this many episodes.
ZSG_HOLDOUT_ITERATION = 2




# --------------- UNSUPERVISED ENVIRONMENT DESIGN -----------------
# ACCEL-style evolutionary curriculum: keep a population of levels, score them,
# and breed new ones by editing the levels the agent is currently on the edge of
# solving. See the plan document for why the minimax-regret framing is not
# adopted here.

UED_ENABLED = True
UED_METHOD = "accel"          # "dr" (uniform over generated levels) | "accel"

# Score function. "learnability" is p*(1-p) over a level's success history and
# needs several trials per level; "maxmc" scores a single episode from the SAC
# critic; "hybrid" uses maxmc until UED_MIN_TRIALS trials have accumulated and
# learnability after that, which covers the cold start of freshly bred levels.
UED_SCORE = "hybrid"          # "learnability" | "maxmc" | "hybrid"
# With Beta(1,1) smoothing, three trials only distinguish four success rates,
# so p*(1-p) cannot tell a frontier level from a noisy one. Five is still
# coarse but affordable at the smaller population size.
UED_MIN_TRIALS = 5
UED_TRIAL_HISTORY = 10        # sliding window used for the success rate

# Robust-PLR's replay-only update rule cannot hold exactly against a shared
# off-policy buffer. Kept as an experiment axis rather than a silent choice:
# when True, transitions from freshly sampled levels are scored but not stored.
UED_REPLAY_ONLY_UPDATES = False

# Sized by how many episodes a level needs before its score means anything,
# not by how many levels we could store. Learnability needs several episodes
# per level and an episode here runs up to MAX_STEPS steps of social-force
# simulation, so a population of thousands would leave almost every level with
# one trial or none and the curriculum would be steering on noise. A few
# hundred live levels, each revisited every few hundred episodes, actually
# accumulate a success rate.
# 200, not 400, because the population size and the score window have to be
# consistent: each live level must collect more than UED_MIN_TRIALS trials
# inside UED_TRIAL_MAX_AGE, or the window expires faster than a level can earn
# a score and nothing ever reaches the breeding bar. At 200 levels, a 3000
# episode window and a 90% replay rate that is about 13 trials per window,
# roughly 2.7x the bar.
UED_POP_SIZE = 200

# No single level may produce more than this many descendants. Children start
# on their parent's score, so without a cap one lucky ancestor can crowd the
# population with its own lineage.
UED_MAX_CHILDREN_PER_LEVEL = 3
UED_P_NEW_START = 0.5         # probability of a fresh level early in training
UED_P_NEW_END = 0.1           # ... once the population has filled out
UED_P_NEW_DECAY_EPISODES = 5000
UED_STALENESS_COEF = 0.2      # rho: weight on the staleness distribution
UED_SCORE_TEMPERATURE = 1.0   # rank-based sampling temperature

# Breed only from the top quarter by score percentile. Half was too
# permissive: it produced a child on roughly every other episode, so the
# population turned over faster than levels could accumulate trials.
UED_MUTATE_THRESHOLD = 0.75
UED_MUTATIONS_PER_CHILD = (1, 3)
UED_MUTATE_MAX_TRIES = 40

# Difficulty 0 is an empty room with two exits: the tier ACCEL is designed to
# start from, where all map complexity has to be earned by editing. Narrow this
# to (0, 2) for a canonical ACCEL run, where the generator only ever supplies
# easy levels and the complexity-growth curve is therefore evidence about the
# curriculum rather than about the generator. The held-out evaluation set spans
# difficulty 1-6 either way.
# Success criterion for the curriculum's score. A level counts as solved when
# every pedestrian is out within UED_SUCCESS_K times the level's own free-flow
# evacuation estimate, rather than within the global MAX_STEPS cap. MAX_STEPS
# stays a hard timeout, but it no longer defines difficulty: a constant chosen
# by hand makes p saturate at 0 or 1 and takes learnability with it.
#
# K needs calibrating against a trained policy. The ratio of actual to
# free-flow time is logged as UED/mean_evac_ratio, so pick K from that curve
# rather than guessing; raw evacuation times are stored per level, so a new K
# can be applied to the existing history without rerunning anything.
UED_SUCCESS_K = 5.0

# Reference world size the scale scalars in the robot state are measured
# against. log(width / reference) is 0 at the reference, symmetric between
# shrinking and growing, and bounded over any realistic size range, unlike the
# width/DOWNSAMPLE_MAP_SIZE form which grows without limit.
MAP_SIZE_REFERENCE = 100.0

# Specific flow through an exit, persons per metre of width per second. 1.2 is
# the usual figure in the evacuation literature.
EVAC_SPECIFIC_FLOW = 1.2

UED_DIFFICULTY_RANGE = (0, 6)
UED_CROWD_RANGE = (10, 100)

# World size in metres, sampled per fresh level. This is the design axis that
# map-size generalisation is about; the held-out set deliberately reaches
# outside it so interpolation and extrapolation can be told apart.
UED_MAP_SIZE_RANGE = (70, 140)
# Square worlds only for now. Opening the aspect ratio interacts with the D4
# augmentation, which swaps width and height on the odd rotations.
UED_MAP_SQUARE = True

# Which generator family a fresh level comes from.
#
# One family: "citygen" lays out a street network for one of the seven
# morphologies the corpus contains, then decides which of its blocks are built
# up. Difficulty is how much of it is built, so difficulty 0 is an open field
# with nothing to walk around and difficulty 6 is real downtown density.
#
# It replaced two earlier families. "random_map" scattered obstacles on open
# ground; "street_map" laid streets and split each block into separate
# footprints. Both were dropped because the curriculum edited their obstacle
# polygons directly, and those edits were morphology-blind: a block moved off
# its frontage is no longer part of a street network, so a layout that started
# out looking like a downtown stopped looking like one within a few
# generations. Since ACCEL builds complexity entirely through offspring, that
# put city-like fabric out of the curriculum's reach whatever it selected for.
#
# "street_map" had a second fault. The gaps it left between footprints inside
# a block were free space, which overstates where a robot can go in exactly
# the way road-derived extraction was introduced to stop.
#
# Both modules remain for the numbered maps in map_infos/ and the GUI editors.
UED_GENERATOR = "citygen"

# Relative frequency of each morphology among fresh levels. Uniform, so the
# curriculum meets every pattern equally often and its own score decides where
# to spend episodes. Weight a morphology up only to study it deliberately.
UED_MORPHOLOGY_WEIGHTS = None     # None = uniform over citygen.MORPHOLOGIES

UED_WARMUP_EPISODES = 500     # pure DR until the population has some mass

# Level scores are only recorded once exploration has decayed below this.
# START_EPSILON is 1.0 and decays to 0 over LINEARLY_DECAY_STEP episodes from
# START_DECAY_STEP, so for thousands of episodes the actions are mostly random.
# A success rate measured then describes what a random walker happens to solve,
# not where the policy's ability ends, and breeding from it picks parents for
# the wrong reason. Episodes still run and still train during that period;
# only their outcomes are withheld from the curriculum.
UED_SCORE_MAX_EPSILON = 0.25

# Outcomes older than this many episodes are dropped from a level's history.
# The window used to be a fixed count of trials, and at a few hundred live
# levels ten trials span more episodes than the whole epsilon decay, so one
# level's success rate averaged over policies that no longer exist. Age-bounding
# it keeps a score attached to a recognisable policy.
UED_TRIAL_MAX_AGE = 3000
UED_LEVEL_QUEUE_SIZE = 4      # per-worker prefetch depth
UED_PRODUCER_TARGET = 64      # fresh levels kept ready on the producer thread

# How often to push population pictures into TensorBoard. Each snapshot renders
# 12 levels plus a lineage, so it is cheap next to an episode but not free.
UED_SNAPSHOT_EPISODES = 200

# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight
