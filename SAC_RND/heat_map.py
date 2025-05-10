import argparse
import time
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["HeatMapLogger"]


class HeatMapLogger:
    """Record robot positions per-map and build cumulative heat-maps.

    Parameters
    ----------
    save_root : str or Path
        Directory in which *.npy aggregate files will be stored.
    map_size : Tuple[int, int]
        (width, height) of a map in grid cells.
    known_maps : List[int] or None
        List of map-ids expected at start-up.  New unknown ids will be created
        on-the-fly.
    resume : bool, default True
        If *True* existing ``heat_<map>.npy`` files are loaded so training can
        be resumed without losing statistics.

    - 로봇이 지날 때마다 좌표 count (update)
    - 에피소드 단위로 count를 합산하여 저장 (flush_episode)
    - 저장된 누적 heat map을 시각화 (visualise)
    """

    def __init__(
        self,
        save_root: Union[str, Path],
        map_size: Tuple[int, int] = (50, 50),
        known_maps: Union[List[int], None] = None,
        resume: bool = True,
    ) -> None:
        self.save_root = Path(save_root).expanduser().resolve()
        self.save_root.mkdir(parents=True, exist_ok=True) ## 폴더 없으면 생성
        self.w, self.h = map_size ## 맵 사이즈

        self._episode_buf: Dict[int, np.ndarray] = {} ## 현재 에피소드 내 방문 횟수
        self._aggregate: Dict[int, np.ndarray] = {} ## 누적 방문 횟수
        
        self.scan_existing_files() ## 기존 파일(heat_<map>.npy) 읽고 _aggregate에 로드

        if known_maps is None:
            known_maps = []

        for m in known_maps:
            self._episode_buf[m] = np.zeros((self.w, self.h), dtype=np.int64)
            self._aggregate[m] = (
                np.load(self._file_path(m)) if resume and self._file_path(m).exists() else np.zeros((self.w, self.h), dtype=np.int64)
            ) 

    # ---------------------------------------------------------------------
    # Logging during training 
    # ---------------------------------------------------------------------

    def update(self, map_id: int, x: int, y: int) -> None:
        """Increment visit-counter for *(x, y)* on *map_id*."""
        if not (0 <= x < self.w and 0 <= y < self.h): ## 좌표가 맵 크기 안에 있는가
            return  # ignore out-of-bounds
        if map_id not in self._episode_buf: ## 에피소드 버퍼에 map_id가 없으면 생성 / 로드
            # lazily create buffers for unseen map
            self._episode_buf[map_id] = np.zeros((self.w, self.h), dtype=np.int64)
            self._aggregate[map_id] = (
                np.load(self._file_path(map_id)) if self._file_path(map_id).exists() else np.zeros((self.w, self.h), dtype=np.int64)
            )
        self._episode_buf[map_id][x, y] += 1 ## 해당 셀 방문 횟수 +1

    def flush_episode(self) -> None:
        """Commit episode-level counts to disk (add & save)."""
        for m, epi_mat in self._episode_buf.items():
            if epi_mat.sum() == 0: ## 누적 합이 없으면 skip
                continue
            self._aggregate[m] += epi_mat ## _aggregate에 에피소드 버퍼의 방문 횟수 합산
            # atomic save → write temp then rename
            tmp_path = self._file_path(m).with_suffix(".tmp") ## .tmp로 저장
            # write raw file to avoid automatic .npy addition
            with tmp_path.open('wb') as f:
                np.save(f, self._aggregate[m])
            tmp_path.replace(self._file_path(m))
            epi_mat.fill(0) ## 에피소드 카운트 초기화

    # ------------------------------------------------------------------
    # Visualisation helpers (can be called from a separate process)
    # ------------------------------------------------------------------

    def visualise(
        self,
        map_id: Optional[int] = None, ##특정 맵을 그릴지/ None 은 모든 맵
        *,
        cmap: str = "jet",
        normalise: bool = True, ## 0~1 로 정규화 여부
        blocking: bool = True, ## plt.show() 호출 시 블로킹 여부
        fig: Optional["plt.Figure"] = None, ## 기존 figure 재사용 핸들
    ) -> "plt.Figure":
        """
        Figure 하나를 재사용해 깜빡임 없이 갱신.
        • map_id 가 None 이면 저장된 모든 맵을 서브플롯으로 배치.
        • 반환값: 사용한 Figure 핸들 (watch 모드에서 재사용).
        """
        maps = [map_id] if map_id is not None else sorted(self._aggregate) ## 그릴 맵 리스트
        if not maps:
            print("No heat‑map data found.") ## 없으면 빈 figure 반환
            return fig if fig is not None else plt.figure()

        # ── Figure 재사용 / 생성 ───────────────────────────
        if fig is None or not plt.fignum_exists(fig.number):
            fig = plt.figure(figsize=(5, 5))
        else:
            fig.clf()

        n = len(maps) ## subplot
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        axes = fig.subplots(rows, cols, squeeze=False)

        for idx, m in enumerate(maps):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            data = self._load(m) ## (50, 50) count matrix, >=0
            if normalise and data.max() > 0:
                data = data / data.max() ## 0~1 정규화, max 가 1임
            im = ax.imshow( ## 전치
                data.T, origin="lower", cmap=cmap, interpolation="nearest"
            )
            ax.set_title(f"map {m}")
            ax.set_xticks([]), ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04) ## 컬러바

        # 남는 subplot 숨김
        for j in range(n, rows * cols):
            r, c = divmod(j, cols)
            axes[r][c].set_visible(False)

        fig.tight_layout()
        if blocking:
            plt.show()
        else:
            fig.canvas.draw_idle()
            plt.pause(0.001)
        return fig
    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _file_path(self, map_id: int) -> Path:
        return self.save_root / f"heat_{map_id}.npy"

    def _load(self, map_id: int) -> np.ndarray:
        path = self._file_path(map_id)
        if path.exists():
            return np.load(path)
        return np.zeros((self.w, self.h), dtype=np.int64)
    
    # ------------------------------------------------------------------
    # Scan_existing_files
    # ------------------------------------------------------------------

    def scan_existing_files(self) -> None:
        """Look for ``heat_*.npy`` files in *save_root* and load them."""
        for f in self.save_root.glob("heat_*.npy"):
            try:
                map_id = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue                     # 파일명이 예상 형식이 아니면 건너뜀
            if map_id not in self._aggregate:
                self._aggregate[map_id] = np.load(f)
                self._episode_buf[map_id] = np.zeros((self.w, self.h), dtype=np.int64)


# ---------------------------------------------------------------------------
# CLI utility – inspect heat-maps while training is running
# ---------------------------------------------------------------------------

def _cli():
    p = argparse.ArgumentParser(description="Visualise or watch accumulated heat-maps produced by HeatMapLogger.")
    p.add_argument("--root", required=True, help="Directory that contains heat_<map>.npy files (the 'heat_maps' folder).")
    p.add_argument("--map", type=int, default=None, help="Single map-id to display.  Omit to show all.")
    p.add_argument("--watch", action="store_true", help="Refresh the plot every --interval seconds.")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between refresh when --watch is set.")
    p.add_argument("--cmap", default="jet", help="Matplotlib colour-map name.")
    args = p.parse_args()

    logger = HeatMapLogger(args.root, resume=True) ## HeatMapLogger 생성

    fig = logger.visualise(args.map, cmap=args.cmap, blocking=not args.watch)
    
    if args.watch: ## --watch 모드면 지정간격마다 visualise 호출
        print("Watching for updates…  Press Ctrl-C to stop.")
        try:
            while plt.fignum_exists(fig.number):
                time.sleep(args.interval)
                fig = logger.visualise(
                    args.map, cmap=args.cmap, blocking=False, fig=fig
                )
        except KeyboardInterrupt:
            pass
        finally:
            plt.close("all")


if __name__ == "__main__":
    _cli()


