import multiprocessing
import os
import time
import subprocess

lr = 1e-4
decay_value = 0.999
buffer_size = 50000
batch_size = 64
alpha = 0.2
start_epsilon = 1.0
epsilon_min = 0.1
expert_buffer_size = 50000
device = "cpu"
use_gail = False

def run_reinforcement_learning():
    subprocess.run(["python3", "ADDS_AS_reinforcement.py" , "--lr", str(lr), "--decay_value", str(decay_value), "--buffer_size", str(buffer_size), "--batch_size", str(batch_size), "--alpha", str(alpha), "--start_epsilon", str(start_epsilon), "--epsilon_min", str(epsilon_min), "--device", device, "--use_gail", str(use_gail), "--expert_buffer_size", str(expert_buffer_size)])

if __name__ == "__main__":
    while True:
        process = multiprocessing.Process(target=run_reinforcement_learning)
        process.start()
        process.join()
        if process.exitcode != 0:
            print("segmentation fault detected, restarting...")
            time.sleep(3)