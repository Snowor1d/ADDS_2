import multiprocessing
import os
import time
import subprocess

lr = 2e-4
buffer_size = 100000
batch_size = 128

EPSILON_MIN = 0.99 #min epsilon
START_EPSILON = 1 #start epsilon
SCHEDULER_TYPE = "l" 
DECAY_VALUE = 0.9
LINEARLY_DECAY_STEP = 100
START_UPDATE_STEP = 1000
START_DECAY_STEP = START_UPDATE_STEP
LOG_DIR = "learning_log_guide_game_sac4_cnn"

long_epsilon_min = 0
start_long_epsilon = 0

log_std_max = 0.5
log_std_min = -20
alpha = 0.2 # in SAC
device = "cpu" 
start_batch_times = 1 #when start update?
gamma = 0.995
port_num = 6007

REWARD_A = 0 #reward_based_alived
REWARD_B = 0 #reward_based_all_agents_danger
REWARD_C = 0 #reward_based_gain
REWARD_D = 2 #reward_based_penalty
REWARD_E = 0 #reward_based_evacuated_with_robot
REWARD_F = 0 #reward_based_distance_from_near_agents
REWARD_G = 0 #reward_based_distance_from_near_agent_gain
REWARD_H = 0 #reward_based_gain_with_time_bonus
REWARD_I = 2 #reward_based_alived_root
REWARD_J = 0.5 #reward_based_all_agents_danger_log
REWARD_K = 1 #reward_based_near_agents_exist
CROWD_NUMBER_MIN = 0
CROWD_NUMBER_MAX = 20


finished_bonus = 50
scale_check = 0 # want to check reward scale?
action_scale = 4
max_steps = 6000



def run_reinforcement_learning():
    subprocess.run([
        "python3", "ADDS_AS_reinforcement.py",
        "--lr", str(lr),
        "--buffer_size", str(buffer_size),
        "--batch_size", str(batch_size),
        "--log_std_max", str(log_std_max),
        "--log_std_min", str(log_std_min),
        "--alpha", str(alpha),
        "--device", str(device),
        "--finished_bonus", str(finished_bonus),
        "--scale_check", str(scale_check),
        "--action_scale", str(action_scale),
        "--start_batch_times", str(start_batch_times),
        "--max_steps", str(max_steps),
        "--gamma", str(gamma),
        "--port_num", str(port_num),
        "--long_epsilon_min", str(long_epsilon_min),
        "--start_long_epsilon", str(start_long_epsilon)
    ])


if __name__ == "__main__":
    while True:
        process = multiprocessing.Process(target=run_reinforcement_learning)
        process.start()
        process.join()
        if process.exitcode != 0:
            print("segmentation fault detected, restarting...")
            time.sleep(3)