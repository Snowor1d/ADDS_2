# ------------- BASIC PARAMETERS ------------------
LR = 2e-4
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

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 30
CROWD_NUMBER_MAX = 30
MAP_NUM = -1 #if not used, -2, if random, -1
#learning
# 50x50 : 6,7,26
# 70x60 : 50,53,54
# 100x100 : 100, 104, 105, 108, 110, 117, 118, 119, 120, 121
# unseen maps : 102, 113, 114, 115, 116
# RE maps : 105, 108, 110, 119, 120, 121, 122, 123, 124, 125
# RE 100x100 3maps : 105, 108, 122
# resal test map : 130, 131, 132, 133, 134


#MAP_NUM_RANDOM = [6, 7, 26]


#MAP_NUM_RANDOM = [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
# 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170
#MAP_NUM_RANDOM = [155]
MAP_NUM_RANDOM = [2]
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
LOG_DIR = "SOTA_MODELS"
PORT_NUM = 6007

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


# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight