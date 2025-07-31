import multiprocessing
import os
import time
import subprocess
from typing import Tuple

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
START_UPDATE_STEP = 150
DEVICE = "cuda"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000

# ---------------- SIMULATION ENVIRONMENT ---------------------
CROWD_NUMBER_MIN = 20
CROWD_NUMBER_MAX = 20
MAP_NUM = -1 #if not used, -2, if random, -1
MAP_NUM_RANDOM = [6, 7, 8]
SCALE_CHECK = 0 # want to check reward scale?
ACTION_SCALE = 4
MAX_STEPS = 3000

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
LOG_DIR = "Log_DREAMER"
PORT_NUM = 6007

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -10.0

ALPHA_START = 0.1

ALPHA_END = 0.1
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHAPING -----------------

REWARD_A = 2 #reward_based_alived
REWARD_B = 0.006 #reward_based_all_agents_danger
REWARD_D = 0.3 #reward_based_penalty
REWARD_K = 0.2 #reward_penalty_collsion
REWARD_FIXED = -0.3

REWARD_J = 0 #reward_based_all_agents_danger_log
REWARD_I = 0 #reward_based_alived_root
REWARD_C = 0 #reward_based_gain
REWARD_E = 0 #reward_based_evacuated_with_robot
REWARD_F = 0 #reward_based_distance_from_near_agentsW
REWARD_G = 0 #reward_based_distance_from_near_agent_gain
REWARD_H = 0 #reward_based_gain_with_time_bonus
FINISHED_BONUS = 100


OBS_SHAPE: Tuple[int, int, int] = (1, 50, 50)  # (C,H,W)
ACTION_DIM = 2
LATENT_DIM = 64          # deterministic hidden size h_t
STOCH_DIM = 32           # stochastic latent size z_t
RSSM_DEPTH = 2           # number of GRU layers
IMAG_HORIZON = 15        # imagination rollout length
BATCH_SIZE = 32
SEQ_LEN = 50             # length of sequences sampled from buffer
REPLAY_CAPACITY = 200_000
DISCOUNT = 0.99
LR_WORLD = 3e-4
LR_ACTOR = 2e-5
LR_CRITIC = 4e-5
SYML_LOG_BASE = 10
FREE_NATS = 1.0
GRAD_CLIP = 10
DREAMER_UPDATE_FREQ = 2
DREAMER_START_UPDATE_EP= 100
WORLD_MODEL_FIRST_LEARN = 1_000_000 # /1000->1000
ENTROPY_COEFF = 0.003
LAMBDA= 0.95

# --------------- Dreamer parameters -----------------


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