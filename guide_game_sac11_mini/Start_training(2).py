import multiprocessing
import os
import time
import subprocess

lr = 1e-4
decay_value = 0.999 
buffer_size = 100000
batch_size = 128

epsilon_min = 0.1 #min epsilon
start_epsilon = 1 #start epsilon

long_epsilon_min = 0
start_long_epsilon = 0.1

log_dir = "learning_log_guide_game_sac15_mini"
log_std_max = 1
log_std_min = -20
alpha = 0.2 # in SAC
device = "cpu" 
start_batch_times = 1 #when start update?
gamma = 0.99
port_num = 6007

reward_A = 1 #reward_based_alived
reward_B = 1 #reward_based_all_agents_danger
reward_C = 1 #reward_based_gain
reward_D = 1 #reward_based_penalty
reward_E = 1 #reward_based_evacuated_with_robot
reward_F = 1 #reward_based_distance_from_near_agents
reward_G = 1 #reward_based_distance_from_near_agent_gain
reward_H = 1 #reward_based_gain_with_time_bonus
reward_I = 1 #reward_based_alived_root
reward_J = 1 #reward_based_distance_from_all_agents
finished_bonus = 50
scale_check = 1 # want to check reward scale?
action_scale = 4
max_steps = 2000
crowd_number = 20


def run_reinforcement_learning():
    subprocess.run([
        "python3", "ADDS_AS_reinforcement.py",
        "--lr", str(lr),
        "--decay_value", str(decay_value),
        "--buffer_size", str(buffer_size),
        "--batch_size", str(batch_size),
        "--start_epsilon", str(start_epsilon),
        "--epsilon_min", str(epsilon_min),
        "--log_dir", str(log_dir),
        "--log_std_max", str(log_std_max),
        "--log_std_min", str(log_std_min),
        "--alpha", str(alpha),
        "--device", str(device),
        "--reward_A", str(reward_A),
        "--reward_B", str(reward_B),
        "--reward_C", str(reward_C),
        "--reward_D", str(reward_D),
        "--reward_E", str(reward_E),
        "--reward_F", str(reward_F),
        "--reward_G", str(reward_G),
        "--reward_H", str(reward_H),
        "--reward_I", str(reward_I),
        "--reward_J", str(reward_J),
        "--finished_bonus", str(finished_bonus),
        "--scale_check", str(scale_check),
        "--action_scale", str(action_scale),
        "--start_batch_times", str(start_batch_times),
        "--max_steps", str(max_steps),
        "--crowd_number", str(crowd_number),
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