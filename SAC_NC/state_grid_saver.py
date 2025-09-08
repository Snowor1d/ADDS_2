# state_grid_saver.py
import os
from typing import Optional
import numpy as np
from PIL import Image

def save_grid_image(grid_2d: np.ndarray, save_path: str, scale: int = 10):
    """
    grid_2d: (H, W) uint8, 0~255, 흑백(L) 이미지
    save_path: '.../step_000250.png' 같은 파일 경로
    scale: 몇 배 확대할지 (기본 10배 → 50x50 → 500x500 저장)
    """
    if grid_2d.dtype != np.uint8:
        grid_2d = np.clip(grid_2d, 0, 255).astype(np.uint8)

    img = Image.fromarray(grid_2d, mode="L")

    # 반시계 방향으로 90도 회전
    img = img.transpose(Image.ROTATE_90)

    # 확대 (nearest → 격자 선명 / bilinear → 부드럽게)
    if scale > 1:
        w, h = img.size  # PIL은 (width, height) 순서
        img = img.resize((w * scale, h * scale), resample=Image.NEAREST)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path, format="PNG")

class GridStateSaver:
    """
    지정한 간격으로 env의 50x50 state 이미지를 저장.
    """
    def __init__(self, out_dir: str, every_steps: int = 25, prefix: str = "step"):
        self.out_dir = out_dir
        self.every = max(1, int(every_steps))
        self.prefix = prefix
        os.makedirs(out_dir, exist_ok=True)

    def maybe_save(self, step: int, env):
        """
        step이 간격(every)에 맞으면 env.return_current_image()를 저장.
        """
        if step % self.every != 0:
            return
        grid = env.return_current_image()  # (50, 50) uint8
        fname = f"{self.prefix}_{step:06d}.png"
        save_grid_image(grid, os.path.join(self.out_dir, fname))
