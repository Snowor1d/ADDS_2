"""Record one episode as an H.264 mp4 (ffmpeg), for W&B."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np


class _FFmpeg:
    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = int(fps)
        self.proc = None

    def write(self, rgb: np.ndarray):
        if self.proc is None:
            h, w = rgb.shape[:2]
            h2, w2 = h - h % 2, w - w % 2
            self.size = (h2, w2)
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w2}x{h2}", "-r", str(self.fps), "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path],
                stdin=subprocess.PIPE)
        h2, w2 = self.size
        self.proc.stdin.write(np.ascontiguousarray(rgb[:h2, :w2, :3]).astype(np.uint8).tobytes())

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=60)


def record_policy_episode(cfg, scenario: dict, policy_path: str, out_path: str,
                          every: int = 4, fps: int = 20, deterministic: bool = True) -> Optional[dict]:
    from continuous_renderer import ContinuousRenderer
    from evaluation.suites import scenario_spec
    from learn.policy_io import load_policy
    from learn.rollout import PolicyController, build_env, eval_step_cap, run_episode
    import random

    actor, pcfg, _ = load_policy(policy_path)
    env = build_env(scenario_spec(scenario), cfg, robot="Q")
    random.seed(int(scenario["seed"]) + 1)
    np.random.seed((int(scenario["seed"]) + 1) % (2 ** 32))
    rend = ContinuousRenderer(world_size=(float(env.width), float(env.height)),
                              crowd_colors={0: "#ffa500", 1: "#4e79a7", 2: "#4e79a7"},
                              robot_color="#e15759", show_agent_heading=False,
                              show_robot_heading=False, trail_target="robot", trail_style="fade",
                              max_trail=400, single_color_edges=True, exit_size=5.0,
                              snap_exit_to_boundary=False)
    writer = _FFmpeg(out_path, fps)

    def on_step(e, step):
        if step % every == 0:
            writer.write(rend.draw(e, step=step))

    try:
        res = run_episode(env, PolicyController(actor, pcfg, deterministic=deterministic),
                          pcfg, max_steps=eval_step_cap(cfg, env), on_step=on_step)
    finally:
        writer.close()
    return res if os.path.exists(out_path) else None
