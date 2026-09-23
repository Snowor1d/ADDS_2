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


# -------------- PATH -------------------
# Kept distinct from SAC_FE_RV3 so the UED run and the baseline run can share a
# machine without clobbering each other's logs, checkpoints or tensorboard port.
LOG_DIR = "Log_SAC_UED"
#LOG_DIR = "SOTA_MODELS"
#LOG_DIR = "Log_test"
PORT_NUM = 7754

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
ZSG_HOLDOUT_ITERATION = 3




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
UED_MIN_TRIALS = 3
UED_TRIAL_HISTORY = 10        # sliding window used for the success rate

# Robust-PLR's replay-only update rule cannot hold exactly against a shared
# off-policy buffer. Kept as an experiment axis rather than a silent choice:
# when True, transitions from freshly sampled levels are scored but not stored.
UED_REPLAY_ONLY_UPDATES = False

UED_POP_SIZE = 3000
UED_P_NEW_START = 0.5         # probability of a fresh level early in training
UED_P_NEW_END = 0.1           # ... once the population has filled out
UED_P_NEW_DECAY_EPISODES = 5000
UED_STALENESS_COEF = 0.2      # rho: weight on the staleness distribution
UED_SCORE_TEMPERATURE = 1.0   # rank-based sampling temperature

UED_MUTATE_THRESHOLD = 0.5    # breed from levels above this score percentile
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
UED_SUCCESS_K = 3.0

# Specific flow through an exit, persons per metre of width per second. 1.2 is
# the usual figure in the evacuation literature.
EVAC_SPECIFIC_FLOW = 1.2

UED_DIFFICULTY_RANGE = (0, 6)
UED_CROWD_RANGE = (20, 40)

UED_WARMUP_EPISODES = 500     # pure DR until the population has some mass
UED_LEVEL_QUEUE_SIZE = 4      # per-worker prefetch depth
UED_PRODUCER_TARGET = 64      # fresh levels kept ready on the producer thread

# How often to push population pictures into TensorBoard. Each snapshot renders
# 12 levels plus a lineage, so it is cheap next to an episode but not free.
UED_SNAPSHOT_EPISODES = 200

# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight
