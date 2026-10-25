"""Training with a watchdog: restarts the trainer when it crashes or when
global_episode stops advancing (heartbeat file) for STALL_SEC.

    python3 Start_training.py [--preset FILE] [--set KEY=VALUE ...]

Arguments are passed through to the trainer, and --preset / --set go into
the environment (RV4_PRESET / RV4_OVERRIDES), so every restart and every
worker uses the same configuration. Ctrl+C sends SIGINT to the trainer's
process group; it stops its workers, saves a checkpoint and closes W&B.
"""
import os
import signal
import subprocess
import sys
import time

STALL_SEC = 2000
POLL_SEC = 100
STOP_GRACE_SEC = 90


def print_banner():
    print("""
 █████╗ ██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝
███████║██║  ██║██║  ██║███████╗
██╔══██║██║  ██║██║  ██║╚════██║
██║  ██║██████╔╝██████╔╝███████║
╚═╝  ╚═╝╚═════╝ ╚═════╝ ╚══════╝   SAC_FE_RV4
""")


def main():
    from configs import apply_cli_overrides
    rest = apply_cli_overrides(sys.argv[1:])
    from configs import cfg
    heartbeat = os.path.join(os.path.expanduser("~"), cfg.LOG_DIR, "heartbeat.txt")
    here = os.path.dirname(os.path.abspath(__file__))
    print_banner()
    while True:
        try:
            os.remove(heartbeat)
        except FileNotFoundError:
            pass
        p = subprocess.Popen([sys.executable, os.path.join(here, "ADDS_AS_reinforcement.py")] + rest,
                             cwd=here, start_new_session=True)
        last_m, last_t = None, time.time()
        try:
            while True:
                ret = p.poll()
                if ret is not None:
                    break
                try:
                    m = os.path.getmtime(heartbeat)
                except FileNotFoundError:
                    m = None
                now = time.time()
                if m is not None and (last_m is None or m > last_m):
                    last_m, last_t = m, now
                if m is not None and now - last_t > STALL_SEC:
                    print(f"[watchdog] no progress for {STALL_SEC}s; restarting", flush=True)
                    os.killpg(p.pid, signal.SIGINT)
                    try:
                        p.wait(timeout=STOP_GRACE_SEC)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                    ret = 999
                    break
                time.sleep(POLL_SEC if m is not None else 5)
        except KeyboardInterrupt:
            print("\n[watchdog] Ctrl+C: stopping the trainer ...", flush=True)
            try:
                os.killpg(p.pid, signal.SIGINT)
                p.wait(timeout=STOP_GRACE_SEC)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            print("System Shutdown")
            return
        if ret == 0:
            print("[watchdog] trainer finished")
            return
        print(f"[watchdog] trainer exited ({ret}); restarting in 3 s", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
