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

#MAP_NUM_RANDOM = [1, 2, 3, 6, 7, 8, 23, 24, 25, 26]
#MAP_NUM_RANDOM = [1, 28, 29, 6, 7, 8, 30, 24, 25, 26]

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