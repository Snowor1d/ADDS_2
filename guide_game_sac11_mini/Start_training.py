import multiprocessing
import os
import time
import subprocess

lr = 1e-4
decay_value = 0.999
buffer_size = 50000
batch_size = 64
epsilon_min = 0
start_epsilon = 0
log_dir = "learning_log_guide_game_sac11_mini"
log_std_max = 1
log_std_min = -0.5
alpha = 0.2
device = "cpu"

reward_A = 0 #reward_based_alived
reward_B = 0 #reward_distance_from_all_agents_danger
reward_C = 0 #reward_based_gain
reward_D = 0 #reward_based_penalty
reward_E = 0 #reward_based_evacuated_with_robot
reward_F = 0 
finished_bonus = 0
scale_check = True

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
        "finished_bonus", str(finished_bonus),
        "scale_check", str(True)
    ])


if __name__ == "__main__":
    while True:
        process = multiprocessing.Process(target=run_reinforcement_learning)
        process.start()
        process.join()
        if process.exitcode != 0:
            print("segmentation fault detected, restarting...")
            time.sleep(3)