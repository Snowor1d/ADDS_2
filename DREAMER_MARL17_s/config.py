# DreamerV3 for multi-agent reinforcement learning.
#
# Experiment variant: DREAMER_MARL17_s
# - Based on DREAMER_MARL15.
# - Removes robot coordinates from the RSSM transition input so replay and
#   imagination use the same action/mask-conditioned dynamics.
# - Masks inactive robot slots in ego reconstruction, actor log-probability,
#   and analytic policy entropy.
# - Advances the online RSSM posterior at every action boundary, including
#   boundaries where epsilon exploration replaces every policy action.
# - Aligns RMSNorm, LaProp beta2, AGC, and CUDA mixed precision more closely
#   with the official DreamerV3 implementation.
# - Keeps MARL15's encoder, block-spatial decoder, two-hot heads, and policy
#   distribution unchanged to isolate the correctness fixes.
# - Preserves continuous velocity magnitudes in RSSM dynamics by mapping
#   environment actions from [spd_min, spd_max] to [-1, 1] instead of
#   saturating every action component whose absolute value exceeds 1.
# - Small-size model based on DREAMER_MARL17_m. Scales only the four model
#   width dimensions used by the official DreamerV3 size presets: MLP hidden
#   units, recurrent units, base CNN channels, and codes per latent.
# - Uses model dimension M=64 (the official size1m width), with recurrent=8*M
#   and CNN/codes=M/16. All training and environment settings remain identical
#   to DREAMER_MARL17_m.
# ------------- DREAMER V3 ------------------
DREAMER_REPLAY_CAPACITY = 1000000
DREAMER_BATCH_SIZE = 16
DREAMER_SEQUENCE_LENGTH = 64
DREAMER_REPLAY_CONTEXT = 8
DREAMER_HORIZON = 15
DREAMER_LAMBDA = 0.95
DREAMER_MODEL_LR = 4e-5
DREAMER_ACTOR_LR = 4e-5
DREAMER_VALUE_LR = 4e-5
DREAMER_LR_WARMUP_STEPS = 1000
DREAMER_TARGET_VALUE_TAU = 0.02
DREAMER_RETURN_NORM_DECAY = 0.99
DREAMER_DETER_SIZE = 512
DREAMER_STOCH_SIZE = 32
DREAMER_DISCRETE_SIZE = 4 # codes per latent
DREAMER_HIDDEN_SIZE = 64
DREAMER_EMBED_SIZE = 256
DREAMER_ACTION_EMBED_SIZE = 256
DREAMER_CONV_DEPTH = 4
DREAMER_VECTOR_ENCODER_LAYERS = 3
DREAMER_RSSM_BLOCKS = 8
DREAMER_IMAG_LAST = 0  # 0: all posterior states, >0: last K states, -1: random single state
DREAMER_TWOHOT_BINS = 255
DREAMER_POLICY_LAYERS = 3
DREAMER_VALUE_LAYERS = 3
DREAMER_REWARD_LAYERS = 1
DREAMER_CONTINUE_LAYERS = 1
DREAMER_SLOW_VALUE_TARGET = False
DREAMER_CONTDISC = True
DREAMER_USE_AMP = True
DREAMER_DECODER_BSPACE = 8
DREAMER_VECTOR_DECODER_LAYERS = 3
DREAMER_DECODER_CHUNK_SIZE = 128

# ------------- BASIC PARAMETERS ------------------
START_UPDATE_EPISODE = 500
DEVICE = "cuda"

DISCOUNT_HORIZON = 333
GAMMA_START = 1.0 - 1.0 / DISCOUNT_HORIZON
GAMMA_END = GAMMA_START
GAMMA_SCHEDULE_STEP = 1000
THE_NUMBER_OF_ROBOTS = 1

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 30
CROWD_NUMBER_MAX = 30
MAP_NUM = -1 #if not used, -2, if random, -1
# learning
# 50x50 : 6,7,26
# 70x70 : 50,53,54
# 100x100 : 105, 108, 128
# unseen maps : 500, 501, 502, 503
# MAP_NUM_RANDOM = [105, 108, 128]

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


#MAP_NUM_RANDOM = list(range(1000, 1300))
#MAP_NUM_RANDOM = list(range(1000, 1300))
MAP_NUM_RANDOM = list(range(1000, 1300))
#MAP_NUM_RANDOM = [1502]

# Apply one random geometric symmetry whenever a simulation map is loaded.
# The default keeps the previous behavior.  Enable this for training and turn
# it off for evaluation when results must use the map orientation from JSON.
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
ACTION_STEP = 4
MAX_STEPS = 6000

ROBOT_BODY_RADIUS = 1 # m
AGENT_BODY_RADIUS = 0.5 # m
ROBOT_VISION = 20 # m
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
LOG_DIR = "DREAMER_MARL17_s_aug"
#LOG_DIR = "SOTA_MODELS"
#LOG_DIR = "Log_test"
PORT_NUM = 8541

# --------------- REWARD SHAPING -----------------

REWARD_A = 0.5 #reward_based_alived
REWARD_B = 0.001 #reward_based_all_agents_danger
REWARD_D = 0 #reward_based_penalty
REWARD_K = 3 #reward_penalty_collsion
REWARD_FIXED = -0.1

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

N_ENVS = 6
UPDATES_PER_TRANSITION = 0.2
POLICY_BROADCAST_INTERVAL = 10

EGO_MAP_SIZE = 25
DOWNSAMPLE_MAP_SIZE = 50
USE_MARL_BASELINE = False


SPD_MIN = -2
SPD_MAX = 2
FRAME_STEP = 4

ZSG_MAP = [1500, 1501, 1502, 1503]
ZSG_CYCLE_EPISODE = 500
ZSG_ITERATION = 10
ZSG_ROBOT_NUM = [1]

BLACK_SHEEP_WALL = True


# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight
