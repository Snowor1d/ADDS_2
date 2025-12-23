# ------------- BASIC PARAMETERS ------------------
LR = 2e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1 #intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500
DEVICE = "cpu"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000
MAP_H = 100
MAP_W = 100

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 30
CROWD_NUMBER_MAX = 30
MAP_NUM = -1 #if not used, -2, if random, -1
#MAP_NUM_RANDOM = [50, 51, 53]
#MAP_NUM_RANDOM = [51]
#MAP_NUM_RANDOM = [100, 101, 102]
MAP_NUM_RANDOM = [106]
SCALE_CHECK = 0 # want to check reward scale?
ACTION_SCALE = 4
MAX_STEPS = 4000

ROBOT_BODY_RADIUS = 1
AGENT_BODY_RADIUS = 0.5
ROBOT_VISION = 10
AGENT_VISION = 10
EXIT_CONFIRM_RADIUS = 10

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


# -------------- PATH -------------------
LOG_DIR = "Log_FU_100_101_103"
PORT_NUM = 6007

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20

ALPHA_START = 0.2 # in SAC
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHAPING -----------------

REWARD_A = 1 #reward_based_alived
REWARD_B = 0.001 #reward_based_all_agents_danger
REWARD_D = 2 #reward_based_penalty
REWARD_K = 1 #reward_penalty_collsion
REWARD_J = 0 #reward_based_all_agents_danger_root
REWARD_L = 0.5 #reward_based_farthest_agent_distance
REWARD_FIXED = -1

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