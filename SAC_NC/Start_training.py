import multiprocessing
import os
import time
import subprocess

def print_banner():
    print("""
 █████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝
███████║██║  ██║██║  ██║███████╗
██╔══██║██║  ██║██║  ██║╚════██║
██║  ██║██████╔╝██████╔╝███████║
╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝
""")


# ------------- BASIC PARAMETERS ------------------
LR = 2e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1 #intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_STEP = 500
DEVICE = "cuda"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 20
CROWD_NUMBER_MAX = 20
MAP_NUM = -1 #if not used, -2, if random, -1
MAP_NUM_RANDOM = [1, 2, 3, 6, 7, 8, 23, 24, 25, 26]
SCALE_CHECK = 0 # want to check reward scale?
ACTION_SCALE = 4
MAX_STEPS = 4000


ROBOT_RADIUS = 10
EXIT_CONFIRM_RADIUS = 10
NEIGHBOR_RADIUS = 10
VISION_RADIUS = 10
# --------------- EPSILON-EXPLORATION ------------------
EPSILON_MIN = 0.1
START_EPSILON = 1 
SCHEDULER_TYPE = "l"
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 3000
START_DECAY_STEP = 500
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0

# -------------- PATH -------------------
LOG_DIR = "Log_SAC_NC_6007"
PORT_NUM = 6007

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20

ALPHA_START = 0.2 # in SAC
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHAPING -----------------

REWARD_A = 2 #reward_based_alived
REWARD_B = 0.006 #reward_based_all_agents_danger
REWARD_D = 4 #reward_based_penalty
REWARD_K = 1 #reward_penalty_collsion
REWARD_J = 0 #reward_based_all_agents_danger_root
REWARD_FIXED = -2

REWARD_I = 0 #reward_based_alived_root
REWARD_C = 0 #reward_based_gain
REWARD_E = 0 #reward_based_evacuated_with_robot
REWARD_F = 0 #reward_based_distance_from_near_agents
REWARD_G = 0 #reward_based_distance_from_near_agent_gain
REWARD_H = 0 #reward_based_gain_with_time_bonus
FINISHED_BONUS = 0


USING_TRAINED_MODEL = True

# --------------- crowd evacuation parameter -----------------
K1 = 1 # distance weight
K2 = 0.000001 # density weight (currently not used ; there is only one exit)
K3 = 1 # width weight

def main():
    print_banner()
    while True:
        process = multiprocessing.Process(target=run_reinforcement_learning)
        process.start()
        process.join()
        if process.exitcode != 0:
            print("segmentation fault detected, restarting...")
            time.sleep(3)


def run_reinforcement_learning():
    subprocess.run([
        "python3", "ADDS_AS_reinforcement.py",
    ])


if __name__ == "__main__":
    main()