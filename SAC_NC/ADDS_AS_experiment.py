import os
import sys
import time
import math
import shutil
import numpy as np
import faulthandler
faulthandler.enable()

import model  # FightingModel 이 여기 있다고 가정

# ------------------------- #
# 실험 파라미터
visualization_mode = 'off'
run_iteration      = 1
number_of_agents   = 20
max_step_num       = 3000
robot_version      = 'R' # 'T' : direct to goal, 'R' : 가장 먼 agent와 출구 사이를 왔다갔다, 'Q' : 학습된 모델 사용
robot_learned_model = 'sac_checkpoint_ep_15000.pth'
test_num           = 5
map_list           = [6, 7, 8]

MAP_WIDTH  = 50
MAP_HEIGHT = 50

EXP_NAME = "try1"   # 최상위 결과 폴더 이름


# ------------------------- #

def home_path(*parts):
    return os.path.join(os.path.expanduser("~"), *parts)

def init_result_root(exp_name: str) -> str:
    """최상위 결과 폴더는 이미 있으면 삭제하지 않고 사용"""
    root = home_path(f"Result_data_{exp_name}")
    os.makedirs(root, exist_ok=True)
    return root

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def recreate_dir(path: str):
    """세부 실험 폴더는 있으면 지우고 새로 만든다"""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def write_txt(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def write_list_txt(path: str, values):
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(list(values)))

def save_step_series(path: str, series):
    with open(path, "w", encoding="utf-8") as f:
        for v in series:
            f.write(f"{v}\n")

def run_one_episode(map_id: int):
    step_num = 0
    env = model.FightingModel(
        number_of_agents,
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        model_num=map_id,
        robot=robot_version
    )
    if(robot_version == 'Q'):
        env.use_model(robot_learned_model)
    episode_log = []
    try:
        while True:
            env.step()
            step_num += 1
            alive = env.alived_agents()
            episode_log.append(alive)

            if alive < 1:
                evacuated_all = True
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log
            if step_num > max_step_num:
                evacuated_all = False
                all_life = env.calculate_all_agents_life_time()
                return step_num, evacuated_all, all_life, episode_log
    finally:
        del env

def aggregate_episode_logs(logs):
    if not logs:
        return [], [], []
    max_len = max(len(log) for log in logs)
    means, mins, maxs = [], [], []
    for i in range(max_len):
        bucket = [log[i] for log in logs if len(log) > i]
        means.append(float(np.mean(bucket)))
        mins.append(int(np.min(bucket)))
        maxs.append(int(np.max(bucket)))
    return means, mins, maxs

def main():
    result_root = init_result_root(EXP_NAME)
    print(f"[INFO] Result root at: {result_root}")

    global_test_index = 0

    for j in range(run_iteration):
        print(f"\n=== Iteration {j+1}/{run_iteration} ===")
        for map_id in map_list:
            map_dir = os.path.join(result_root, f"Result_{map_id}")
            ensure_dir(map_dir)

            map_robot_dir = os.path.join(map_dir, f"Result_{map_id}_{robot_version}")
            ensure_dir(map_robot_dir)

            evac_times, life_times, episode_logs = [], [], []

            for test_i in range(test_num):
                unique_test_i = global_test_index
                test_dirname = f"Result_{map_id}_{robot_version}_{unique_test_i}"
                test_dir = os.path.join(map_robot_dir, test_dirname)

                # 세부폴더는 항상 새로 생성
                recreate_dir(test_dir)

                steps, evacuated_all, all_life, ep_log = run_one_episode(map_id)

                evacuation_100_time = steps if evacuated_all else max_step_num
                all_agents_life_time = all_life

                evac_times.append(evacuation_100_time)
                life_times.append(all_agents_life_time)
                episode_logs.append(ep_log)

                write_txt(
                    os.path.join(test_dir, "metrics.txt"),
                    f"evacuation_100_time={evacuation_100_time}\nall_agents_life_time={all_agents_life_time}\n"
                )
                write_list_txt(os.path.join(test_dir, "episode_log.txt"), ep_log)

                print(f"[Map {map_id}] Test {unique_test_i} -> steps={steps}, evacuated_all={evacuated_all}")

                global_test_index += 1

            avg_evac = float(np.mean(evac_times)) if evac_times else float('nan')
            avg_life = float(np.mean(life_times)) if life_times else float('nan')
            write_txt(
                os.path.join(map_robot_dir, "avg_metrics.txt"),
                f"avg_evacuation_100_time={avg_evac}\navg_all_agents_life_time={avg_life}\n"
            )

            mean_series, min_series, max_series = aggregate_episode_logs(episode_logs)
            save_step_series(os.path.join(map_robot_dir, "episode_log_mean.txt"), mean_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_min.txt"), min_series)
            save_step_series(os.path.join(map_robot_dir, "episode_log_max.txt"), max_series)

            print(f"[Map {map_id}] Summary saved at {map_robot_dir}")

if __name__ == "__main__":
    main()
