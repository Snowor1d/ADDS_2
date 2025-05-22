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
LR = 1e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1 #intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_STEP = 150
DEVICE = "cpu"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 20
CROWD_NUMBER_MAX = 20
MAP_NUM = -1 #if not used, -2, if random, -1
MAP_NUM_RANDOM = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#MAP_NUM_RANDOM = [5]
SCALE_CHECK = 0 # want to check reward scale?
ACTION_SCALE = 4
MAX_STEPS = 6000

# --------------- EPSILON-EXPLORATION ------------------
EPSILON_MIN = 0.1
START_EPSILON = 1 
SCHEDULER_TYPE = "l"
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 2000
START_DECAY_STEP = 150
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0

# -------------- PATH -------------------
LOG_DIR = "Log_SAC"
PORT_NUM = 6007

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20

ALPHA_START = 0.2 # in SAC
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHPAING -----------------

REWARD_A = 5 #reward_based_alived
REWARD_B = 0.002 #reward_based_all_agents_danger
REWARD_D = 3 #reward_based_penalty
REWARD_K = 1 #reward_penalty_collsion
REWARD_E = 10 #reward_based_evacuated_with_robot
REWARD_F = 0.01 #reward_based_distance_from_near_agents
REWARD_L = 1 #reward_based_heatmap

REWARD_FIXED = -0.1
FINISHED_BONUS = 100

REWARD_J = 0 #reward_based_all_agents_danger_log
REWARD_I = 0 #reward_based_alived_root
REWARD_C = 0 #reward_based_gain
REWARD_G = 0 #reward_based_distance_from_near_agent_gain
REWARD_H = 0 #reward_based_gain_with_time_bonus



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