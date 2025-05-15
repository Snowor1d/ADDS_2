#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heat_map.py  (PNG + NPY snapshot 지원 버전)
-----------------------------------------
• HeatMapLogger.update / flush_episode : 누적 count 저장
• HeatMapLogger.snapshot(episode)      : heat_<map>_ep_<N>.{npy,png} 둘 다 저장
• CLI(--watch)는 유지, --compare 등 복잡 기능 제거
"""

import argparse, time, os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import matplotlib
matplotlib.use("Agg")                   # GUI 없어도 savefig 가능
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────
# Start_training.LOG_DIR 가져오기 (없으면 무시)
try:
    from Start_training import LOG_DIR as _DEFAULT_LOG_DIR
except ModuleNotFoundError:
    _DEFAULT_LOG_DIR = None

__all__ = ["HeatMapLogger"]


class HeatMapLogger:
    """로봇 위치 heat-map 누적·시각화·스냅샷(PNG/NPY)"""

    # -------------------------------------------------- 초기화
    def __init__(
        self,
        save_root: Union[str, Path],
        map_size: Tuple[int, int] = (50, 50),
        known_maps: Optional[List[int]] = None,
        resume: bool = True,
    ) -> None:
        self.save_root = Path(save_root).expanduser().resolve()
        self.save_root.mkdir(parents=True, exist_ok=True)

        self.w, self.h = map_size
        self._episode_buf: Dict[int, np.ndarray] = {}   # 에피소드별 카운트
        self._aggregate:   Dict[int, np.ndarray] = {}   # 누적 카운트
        self.scan_existing_files()

        known_maps = known_maps or []
        for m in known_maps:
            self._aggregate.setdefault(m,
                np.zeros((self.w, self.h), dtype=np.int64))
            self._episode_buf.setdefault(m,
                np.zeros((self.w, self.h), dtype=np.int64))

    # -------------------------------------------- 학습 중 호출
    def update(self, map_id: int, x: int, y: int) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self._episode_buf.setdefault(
                map_id, np.zeros((self.w, self.h), dtype=np.int64)
            )[x, y] += 1
            self._aggregate.setdefault(
                map_id, np.zeros((self.w, self.h), dtype=np.int64)
            )

    # flush_episode
    def flush_episode(self) -> None:
        for m, epi in self._episode_buf.items():
            if epi.sum():
                self._aggregate[m] += epi
                tmp = self._file_path(m).with_suffix(".tmp")
                with tmp.open("wb") as f:          # ← 수정
                    np.save(f, self._aggregate[m])
                tmp.replace(self._file_path(m))
                epi.fill(0)

        # snapshot
    def snapshot(self, episode: int, *, cmap: str = "jet") -> None:
        """
        • heat_<map>_ep_<N>.npy : 맵별 누적 행렬
        • heat_maps_ep_<N>.png  : 모든 맵 한 Figure 에 저장   ← ★ NEW
        """
        # 1) NPY 개별 저장 (변경 없음)
        for m, mat in self._aggregate.items():
            if mat.sum() == 0:
                continue
            npy_path = self.save_root / f"heat_{m}_ep_{episode}.npy"
            with npy_path.open("wb") as f:
                np.save(f, mat)

        # 2) 통합 PNG 저장  (map_id=None → 모든 맵 자동 그리드)
        png_path = self.save_root / f"heat_maps_ep_{episode}.png"
        fig_all  = self.visualise(
            map_id=None,          # ← 모든 맵
            cmap=cmap,
            normalise=True,
            blocking=False,
            fig=None              # 새 Figure
        )
        fig_all.set_size_inches(10, 10)
        fig_all.savefig(png_path, dpi=150)
        plt.close(fig_all)
    # ---------------------------------------------- 시각화
    def visualise(
        self,
        map_id: Optional[int] = None,
        *,
        cmap: str = "jet",
        normalise: bool = True,
        blocking: bool = False,
        fig: Optional["plt.Figure"] = None,
    ) -> "plt.Figure":
        """fig 재사용하여 1개 또는 모든 맵을 subplot 으로 그림."""
        maps = [map_id] if map_id is not None else sorted(self._aggregate)
        if not maps:
            return fig or plt.figure()

        # Figure 생성/재사용
        if fig is None or not plt.fignum_exists(fig.number):
            fig = plt.figure(figsize=(5, 5))
        else:
            fig.clf()

        n, cols = len(maps), int(np.ceil(np.sqrt(len(maps))))
        rows = int(np.ceil(n / cols))
        axes = fig.subplots(rows, cols, squeeze=False)

        for idx, m in enumerate(maps):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            data = self._load(m)
            if normalise and data.max() > 0:
                data = data / data.max()
            im = ax.imshow(data.T, origin="lower", cmap=cmap, interpolation="nearest")
            ax.set_title(f"map {m}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04)

        # 남는 subplot 감춤
        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].set_visible(False)

        fig.tight_layout()
        if blocking:
            plt.show()
        else:
            fig.canvas.draw_idle(); plt.pause(0.001)
        return fig

    # -------------------------------------------- 파일 helper
    def _file_path(self, m: int) -> Path:
        return self.save_root / f"heat_{m}.npy"

    def _load(self, m: int) -> np.ndarray:
        p = self._file_path(m)
        return np.load(p) if p.exists() else np.zeros((self.w, self.h), dtype=np.int64)

    def scan_existing_files(self) -> None:
        """save_root 에 존재하는 heat_*.npy 자동 로드."""
        for f in self.save_root.glob("heat_*.npy"):
            try:
                mid = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            self._aggregate[mid] = np.load(f)
            self._episode_buf[mid] = np.zeros((self.w, self.h), dtype=np.int64)


# ──────────────────────────────────────────────────────────
# 간단 CLI  (watch 전용)
# ──────────────────────────────────────────────────────────
def _cli() -> None:
    pa = argparse.ArgumentParser("heat-map viewer (PNG 저장 기능은 학습 코드에서 호출)")
    default_root = (
        str(Path.home() / _DEFAULT_LOG_DIR / "heat_maps")
        if _DEFAULT_LOG_DIR else None
    )
    pa.add_argument("--root", default=default_root,
                    help="heat_maps 폴더 (생략 시 Start_training.LOG_DIR 사용)")
    pa.add_argument("--map", type=int, help="단일 map id ( 없으면 전체 )")
    pa.add_argument("--watch", action="store_true", help="주기 갱신 모드")
    pa.add_argument("--interval", type=float, default=5, help="watch 주기(sec)")
    pa.add_argument("--cmap", default="jet")
    args = pa.parse_args()

    if not args.root:
        pa.error("--root 를 지정하거나 Start_training.LOG_DIR 을 설정하세요.")

    logger = HeatMapLogger(args.root, resume=True)
    fig = logger.visualise(args.map, cmap=args.cmap, blocking=not args.watch)

    if args.watch:
        print("watching…  Ctrl-C or close window to exit")
        try:
            while plt.fignum_exists(fig.number):
                time.sleep(args.interval)
                fig = logger.visualise(
                    args.map, cmap=args.cmap, fig=fig, blocking=False
                )
        except KeyboardInterrupt:
            pass
        finally:
            plt.close("all")


if __name__ == "__main__":
    _cli()
