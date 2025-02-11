import time
from config import ENABLE_TIMER

class Timer:
    def __init__(self):
        self.elapsed_time = 0.0
        self.start_time = None

    def start(self):
        if ENABLE_TIMER:  # 타이머 활성화 여부 확인
            self.start_time = time.time()

    def stop(self):
        if ENABLE_TIMER and self.start_time is not None:
            self.elapsed_time += time.time() - self.start_time
            self.start_time = None

    def get_time(self):
        return self.elapsed_time

    def reset(self):
        self.elapsed_time = 0.0
        self.start_time = None
