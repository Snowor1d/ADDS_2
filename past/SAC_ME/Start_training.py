import multiprocessing
import os
import time
import subprocess
import signal
from config import *

HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat.txt")

STALL_SEC = 300        # 300초 동안 global_episode가 안 늘면 “멈춤”으로 판단
POLL_SEC = 60          # heartbeat 체크 주기

def print_banner():
    print(r"""
 █████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝
███████║██║  ██║██║  ██║███████╗
██╔══██║██║  ██║██║  ██║╚════██║
██║  ██║██████╔╝██████╔╝███████║
╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝
""")

def read_heartbeat_mtime():
    try:
        return os.path.getmtime(HEARTBEAT_PATH)
    except FileNotFoundError:
        return None

def kill_process_tree(p: subprocess.Popen):
    # start_new_session=True로 띄우면 p.pid가 프로세스 그룹 리더가 됨
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(2)
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass

def run_reinforcement_learning_with_watchdog():
    # heartbeat 파일이 과거 값이면 혼동될 수 있으니 시작 전에 지워두는 것도 좋음
    try:
        os.remove(HEARTBEAT_PATH)
    except FileNotFoundError:
        pass

    p = subprocess.Popen(
        ["python3", "ADDS_AS_reinforcement.py"],
        start_new_session=True,   # 프로세스 그룹 생성(자식들까지 kill하기 위함)
    )

    last_mtime = None
    last_progress_t = time.time()

    while True:
        # 1) 프로세스가 죽었는지 확인
        ret = p.poll()
        if ret is not None:
            return ret  # exitcode 반환

        # 2) heartbeat 갱신 확인
        mtime = read_heartbeat_mtime()
        now = time.time()

        if mtime is None: # 기다리기
            time.sleep(POLL_SEC)
            continue

        if last_mtime is None or mtime > last_mtime:
            last_mtime = mtime
            last_progress_t = now

        # 3) 정체 판단
        if now - last_progress_t > STALL_SEC:
            print(f"[Watchdog] global_episode stalled for > {STALL_SEC}s. Killing RL process...")
            kill_process_tree(p)
            return 999  

        time.sleep(POLL_SEC)

def main():
    print_banner()
    while True:
        code = run_reinforcement_learning_with_watchdog()
        if code != 0:
            print(f"RL exited (code={code}). restarting...")
            time.sleep(3)

if __name__ == "__main__":
    main()
