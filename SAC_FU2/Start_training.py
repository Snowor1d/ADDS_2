import multiprocessing
import os
import time
import subprocess
import signal
from config import *

TARGET_SCRIPT = "ADDS_AS_reinforcement.py"
home_dir = os.path.expanduser("~")
HEARTBEAT_PATH = os.path.join(home_dir, LOG_DIR, "heartbeat.txt")

STALL_SEC = 300        # 300초 동안 global_episode가 안 늘면 “멈춤”으로 판단
POLL_SEC = 60          # heartbeat 체크 주기

def print_banner():
    print("""
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
        ["python3", TARGET_SCRIPT],
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
    
    try:
        while True:
            exit_code = run_reinforcement_learning_with_watchdog()

            if exit_code == 0:
                print("[Main] Process finished successfully (code 0). Exiting")
                break

            elif exit_code == 999:
                print(f"[Main] Stalled detected. Restarting {TARGET_SCRIPT}...")
                time.sleep(3)
            else:
                print(f"[Main] Process exited with error (Code {exit_code}). Restarting...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\n[Watchdog] KeyboardInterrupt (Ctrl+C) received.")
        print("[Watchdog] Force killing all related processes...")

        try:
            subprocess.run(["pkill", "-9", "-f", TARGET_SCRIPT])
        except Exception:
            pass
        
    finally:
        print("System Shutdown")

if __name__ == "__main__":
    main()
