# Soft Actor Critic for Multi-Agent Reinforcement Learning with 
# collision check error solved
# ------------- BASIC PARAMETERS ------------------
LR = 1e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1 #intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500
DEVICE = "cpu"

GAMMA_START = 0.999
GAMMA_END = 0.999
GAMMA_SCHEDULE_STEP = 1000
MAP_H = 100
MAP_W = 100
THE_NUMBER_OF_ROBOTS = 1

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
#MAP_NUM_RANDOM = [105, 108, 128]

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
MAP_NUM_RANDOM = [0]
#MAP_NUM_RANDOM = list(range(1000, 1300))
#MAP_NUM_RANDOM = [1506]

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
LOG_DIR = "Log_MARL_zsgtest"
#LOG_DIR = "SOTA_MODELS"
#LOG_DIR = "Log_test"
PORT_NUM = 8150

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20
ALPHA_START = 0.2 # in SAC
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

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

N_ENVS = 4
UPDATES_PER_TRANSITION = 1
POLICY_BROADCAST_INTERVAL = 10

EGO_MAP_SIZE = 25
DOWNSAMPLE_MAP_SIZE = 50
USE_MARL_BASELINE = False


R_MIN = 2
R_MAX = 8
SPD_MIN = 1
SPD_MAX = 2
DIRECTION_N= 12
FRAME_STEP = 4

ZSG_MAP = [1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509]
ZSG_CYCLE_EPISODE = 500
ZSG_ITERATION = 10
ZSG_ROBOT_NUM = [1]


# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight