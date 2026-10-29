from __future__ import annotations
from typing import Tuple
import pygame
import time
from config import ENABLE_TIMER


def draw_cell(surface: pygame.Surface, x: int, y: int, scale: int, color: Tuple[int,int,int]) -> None:
    rect = pygame.Rect(x*scale, y*scale, scale, scale)
    surface.fill(color, rect)

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
