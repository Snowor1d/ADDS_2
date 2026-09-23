"""Time-lapse video of a training episode, for W&B.

A worker asked to record draws every `frame_every` simulation steps with the
viewer's renderer (viz/continuous_renderer.py) and pipes the frames to the
system ffmpeg, so the clip plays VIDEO_SPEEDUP times faster than simulated
time:

    frame_every = VIDEO_SPEEDUP / (VIDEO_FPS * AGENT_TIME_STEP)

With the defaults (40x, 20 fps, 0.5 s per step) that is one frame every four
steps, and a 2,000-step episode (1,000 simulated seconds) becomes a 25 s clip.
The worker writes the mp4 under LOG_DIR/videos and reports the path; only
the main process logs it to W&B.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np


def frame_every(cfg) -> int:
    return max(1, int(round(float(cfg.VIDEO_SPEEDUP)
                            / (float(cfg.VIDEO_FPS)
                               * float(cfg.AGENT_TIME_STEP)))))


class EpisodeRecorder:
    """Call `step(model, sim_step)` after every simulation step, then
    `close()`; returns the mp4 path, or None if nothing was written."""

    def __init__(self, cfg, model, path: str, title: str = ""):
        import matplotlib.pyplot as plt
        plt.switch_backend("Agg")          # workers have no display
        from viz.continuous_renderer import ContinuousRenderer

        self.cfg = cfg
        self.path = path
        self.title = title
        self.every = frame_every(cfg)
        self.speed = self.every * float(cfg.AGENT_TIME_STEP) * float(cfg.VIDEO_FPS)
        # Bodies drawn at their true size are under a pixel on a 200 m map, so
        # they are scaled with the map for legibility (drawing only; the
        # simulation is untouched) and the overlay says so.
        span = max(float(model.width), float(model.height))
        crowd_r = max(float(cfg.AGENT_BODY_RADIUS), 0.004 * span)
        robot_r = max(float(cfg.ROBOT_BODY_RADIUS), 0.012 * span)
        self.body_scale = crowd_r / float(cfg.AGENT_BODY_RADIUS)
        self.renderer = ContinuousRenderer(
            world_size=(float(model.width), float(model.height)),
            dpi=int(cfg.VIDEO_DPI),
            crowd_radius=crowd_r, robot_radius=robot_r,
            crowd_colors={0: "#ffa500", 1: "#4e79a7", 2: "#4e79a7"},
            robot_color="#e15759", show_agent_heading=False,
            show_robot_heading=False, trail_target="none",
            single_color_edges=True)
        self._proc: Optional[subprocess.Popen] = None
        self._shape = None
        self.frames = 0
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _open(self, h: int, w: int) -> None:
        # Even dimensions for yuv420p.
        self._shape = (h - h % 2, w - w % 2)
        self._proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{self._shape[1]}x{self._shape[0]}",
             "-r", str(int(self.cfg.VIDEO_FPS)), "-i", "-",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
             "-crf", "28", self.path],
            stdin=subprocess.PIPE)

    def _overlay(self, img: np.ndarray, model, sim_step: int) -> np.ndarray:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        t = sim_step * float(self.cfg.AGENT_TIME_STEP)
        modes = ",".join(getattr(rb, "mode", "?") for rb in model.robots)
        text = (f"{self.title}  step {sim_step}  t={t:.0f}s  x{self.speed:.0f}\n"
                f"in hazard {model.agents_in_danger()}  robots [{modes}]"
                + (f"  bodies drawn x{self.body_scale:.1f}"
                   if self.body_scale > 1.05 else ""))
        d.rectangle([0, 0, im.width, 30], fill=(255, 255, 255))
        d.multiline_text((4, 2), text, fill=(0, 0, 0))
        return np.asarray(im)

    def step(self, model, sim_step: int) -> None:
        if sim_step % self.every:
            return
        img = self.renderer.draw(model, step=sim_step)
        img = self._overlay(img, model, sim_step)
        if self._proc is None:
            self._open(*img.shape[:2])
        h, w = self._shape
        self._proc.stdin.write(np.ascontiguousarray(img[:h, :w, :3]).tobytes())
        self.frames += 1

    def close(self) -> Optional[str]:
        import matplotlib.pyplot as plt
        try:
            plt.close(self.renderer.fig)
        except Exception:
            pass
        if self._proc is None:
            return None
        self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0 or not os.path.exists(self.path):
            return None
        return self.path
