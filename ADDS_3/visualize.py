# ──────────────────────────────────────────────────────────────────────────────
# crowdlib/visualize.py
# ──────────────────────────────────────────────────────────────────────────────
import pygame
from typing import Callable, Tuple

class Viewer:
    """Continuous viewer with play/pause/speed/step; draws circles at float coords."""
    def __init__(self, width: int, height: int, scale: int = 12,
                 step_fn: Callable[[], None] | None = None,
                 draw_fn: Callable[[pygame.Surface, int], None] | None = None,
                 reset_fn: Callable[[], None] | None = None) -> None:
        pygame.init()
        self.scale = scale
        self.w_px = int(width * scale)
        self.h_px = int(height * scale)
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("CrowdSim Viewer v0.2 — Continuous")
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.speed = 1.0
        self.paused = False
        self.step_fn = step_fn
        self.draw_fn = draw_fn
        self.reset_fn = reset_fn

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT and self.paused and self.step_fn:
                    self.step_fn()
                elif event.key == pygame.K_UP:
                    self.speed = min(8.0, self.speed * 2)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.25, self.speed / 2)
                elif event.key == pygame.K_r and self.reset_fn:
                    self.reset_fn()
        return True

    def loop(self) -> None:
        running = True
        while running:
            running = self._handle_events()
            if (not self.paused) and self.step_fn:
                n = max(1, int(self.speed))
                for _ in range(n):
                    self.step_fn()
            if self.draw_fn:
                self.draw_fn(self.screen, self.scale)
            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()
# ──────────────────────────────────────────────────────────────────────────────
import pygame
from typing import Callable

class Viewer:
    """Tiny Pygame viewer with play/pause/speed/step controls.
    Keys: SPACE toggle pause, RIGHT step once (when paused),
          UP/DOWN speed x2 / x0.5, R reset callback.
    Provide draw_fn(surface) and step_fn() from your model.
    """
    def __init__(self, width: int, height: int, scale: int = 12,
                 step_fn: Callable[[], None] | None = None,
                 draw_fn: Callable[[pygame.Surface], None] | None = None,
                 reset_fn: Callable[[], None] | None = None) -> None:
        pygame.init()
        self.scale = scale
        self.w_px = width * scale
        self.h_px = height * scale
        self.screen = pygame.display.set_mode((self.w_px, self.h_px))
        pygame.display.set_caption("CrowdSim Viewer v0.1")
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.speed = 1.0
        self.paused = False
        self.step_fn = step_fn
        self.draw_fn = draw_fn
        self.reset_fn = reset_fn

    def _handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_RIGHT and self.paused and self.step_fn:
                    self.step_fn()
                elif event.key == pygame.K_UP:
                    self.speed = min(8.0, self.speed * 2)
                elif event.key == pygame.K_DOWN:
                    self.speed = max(0.25, self.speed / 2)
                elif event.key == pygame.K_r and self.reset_fn:
                    self.reset_fn()
        return True

    def loop(self) -> None:
        running = True
        while running:
            running = self._handle_events()
            if (not self.paused) and self.step_fn:
                # Run N steps per frame based on speed multiplier
                n = max(1, int(self.speed))
                for _ in range(n):
                    self.step_fn()
            if self.draw_fn:
                self.draw_fn(self.screen)
            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()
