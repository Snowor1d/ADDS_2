import multiprocessing
import os
import time
import subprocess
import signal
from config import *

# The trainer moved into learn/ with the layout tidy-up, so it is launched
# as a module. It is stopped through its own process group, never by name: a
# name pattern also matches a sibling copy's trainer (SAC_UED6 launches the
# same module) and misses the spawned workers, whose command line is
# multiprocessing's.
TARGET_MODULE = "learn.ADDS_AS_reinforcement"
TARGET_SCRIPT = TARGET_MODULE
home_dir = os.path.expanduser("~")
HEARTBEAT_PATH = os.path.join(home_dir, LOG_DIR, "heartbeat.txt")

STALL_SEC = 2000        # 300초 동안 global_episode가 안 늘면 “멈춤”으로 판단
POLL_SEC = 100          # heartbeat 체크 주기
STOP_GRACE_SEC = 90     # Ctrl+C 후 학습이 스스로 정리(worker 종료, W&B 마감)할 시간

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

def stop_process_tree(p: subprocess.Popen, grace_sec: float = STOP_GRACE_SEC):
    """Ctrl+C for the trainer, which runs in its own session and so never
    sees the terminal's. SIGINT goes to its whole process group: the trainer's
    finally block stops the workers and closes W&B, and the workers stop too.
    Whatever is still in the group after grace_sec (or a second Ctrl+C) is
    killed. SIGKILL on the trainer alone, as before, skipped that finally block
    and left every worker running with no parent."""
    try:
        os.killpg(p.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    print(f"[Watchdog] Stopping the trainer (up to {grace_sec:.0f}s; "
          "Ctrl+C again to kill it now)...")
    try:
        p.wait(timeout=grace_sec)
    except subprocess.TimeoutExpired:
        print("[Watchdog] Trainer did not stop in time; killing its process group.")
    except KeyboardInterrupt:
        print("[Watchdog] Second Ctrl+C; killing its process group.")
    # Workers keep the group after the trainer is gone, so this still finds them.
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

def run_reinforcement_learning_with_watchdog():
    # heartbeat 파일이 과거 값이면 혼동될 수 있으니 시작 전에 지워두는 것도 좋음
    try:
        os.remove(HEARTBEAT_PATH)
    except FileNotFoundError:
        pass

    p = subprocess.Popen(
        ["python3", "-m", TARGET_MODULE],
        start_new_session=True,   # 프로세스 그룹 생성(자식들까지 kill하기 위함)
    )
    try:
        return _watch(p)
    except KeyboardInterrupt:
        stop_process_tree(p)
        raise

def _watch(p: subprocess.Popen):
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

    finally:
        print("System Shutdown")

if __name__ == "__main__":
    main()
