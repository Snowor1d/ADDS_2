# ------------- BASIC PARAMETERS ------------------
LR = 1e-4
INTRINSIC_ETA = 0.1 #intrinsic reward
DEVICE = "cuda"

GAMMA = 0.99
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
MAP_DATA_AUGMENTATION = False
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

# -------------- PATH -------------------
LOG_DIR = "Log_PPO_FE_RV"
#LOG_DIR = "SOTA_MODELS"
#LOG_DIR = "Log_test"
PORT_NUM = 7756

# --------------- PPO ALGORITHM PARAMETERS ---------------
# PPO uses its stochastic Gaussian policy for exploration.  There is no
# epsilon-greedy phase and no replay-buffer warm-up.
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20
PPO_ROLLOUT_STEPS_PER_ENV = 512
PPO_EPOCHS = 10
PPO_MINIBATCH_SIZE = 256
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_VALUE_CLIP_EPS = 0.2
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5
PPO_TARGET_KL = 0.02
PPO_ADVANTAGE_NORMALIZATION = True
# CPU rollout과 GPU learner의 convolution/reduction 순서 차이로 생기는 작은
# float32 오차는 허용한다. Policy version 검사는 별도로 항상 수행된다.
PPO_LOGPROB_WARN_TOL = 1e-3
PPO_LOGPROB_FAIL_TOL = 1e-2
PPO_CHECKPOINT_INTERVAL_EPISODES = 100
PPO_CHECKPOINT_INTERVAL_UPDATES = 5
PPO_MAX_EPISODES = 9999999
PPO_BASE_SEED = 1234
PPO_MODEL_LOAD = 3  # 1: fresh, 2: named checkpoint, 3: latest
PPO_NAMED_CHECKPOINT = "ppo_checkpoint_ep_200.pth"

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
EVALUATION_DETERMINISTIC = True
SHOW_CONTROLLED_CROWD = False
ENABLE_TIMER = True

N_ENVS = 4

EGO_MAP_SIZE = 25
DOWNSAMPLE_MAP_SIZE = 50
FiLM_USE = True
EGO_USE = True
ROBOT_STATE_EMBEDDING = True
ROBOT_STATE_DIM = 3

# Zero-shot evaluation (kept identical to DREAMER_MARL17_m for comparison).
# PPO_FE_RV is a single-robot policy, so only robot_num=1 is supported.
ZSG_MAP = [1500, 1501, 1502, 1503]
ZSG_CYCLE_EPISODE = 500
ZSG_ITERATION = 10
ZSG_ROBOT_NUM = [1]




# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight
