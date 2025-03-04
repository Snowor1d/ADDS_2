import multiprocessing
import os
import time
import subprocess

lr = 2e-4
decay_value = 0.0005
buffer_size = 20000
batch_size = 32


def run_reinforcement_learning():
    subprocess.run(["python3", "ADDS_AS_reinforcement.py" , "--lr", str(lr), "--decay_value", str(decay_value), "--buffer_size", str(buffer_size), "--batch_size", str(batch_size)])


if __name__ == "__main__":
    while True:
        process = multiprocessing.Process(target=run_reinforcement_learning)
        process.start()
        process.join()
        if process.exitcode != 0:
            print("segmentation fault detected, restarting...")
            time.sleep(3)