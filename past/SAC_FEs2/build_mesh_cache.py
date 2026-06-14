# build_mesh_cache.py
from __future__ import annotations

import json
import os
from typing import List, Tuple

from mesh_cache import obstacles_fingerprint, cache_path, save_mesh_cache
from mesh_precompute import build_mesh_artifacts

PointT = Tuple[int, int]

# ============================================================
# ✅ USER CONFIG (전역변수로만 설정)
# ============================================================
MAPS = [4001]   # 캐시 만들 map_num 목록
OUT_DIR = "mesh_cache"                           # 저장 폴더
D = 200                                          # mesh_map에서 쓰던 D
# ============================================================


# 너 프로젝트에서 "맵 로드만" 하는 함수를 하나 쓰는 걸 권장.
# 가장 쉬운 방법: FightingModel의 extract_map/load_map_from_file를 복사해
# 여기서 "obstacles만" 채우는 함수로 분리해두기.
def load_map_geometry(map_num: int, base_dir: str = "map_infos"):
    """
    스크립트 위치 기준 (SAC_FEs2) 또는 현재 작업 디렉토리 기준:
    map_infos/map_{map_num}.json

    JSON 포맷:
    {
    "width": 100,
    "height": 100,
    "obstacles": [ [[x,y],...], ... ],
    "exits":     [ [[x,y],...], ... ]   # json에는 tuple이 없으니 list로 저장됨
    }
    """
    # Resolve base relative to this script's directory (SAC_FEs2) so it works from any cwd
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    fname = f"map_{map_num}.json"
    fpath = os.path.join(_script_dir, base_dir, fname)

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"[map] not found: {fpath}")

    with open(fpath, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # --- width/height ---
    w = obj.get("width", None)
    h = obj.get("height", None)
    # if w is not None and h is not None:
    #     self.width = int(w)
    #     self.height = int(h)

    # --- obstacles: [[[x,y],...], ...] 형태 유지 ---
    obstacles_raw = obj.get("obstacles", []) or []
    obstacles: List[List[List[int]]] = []
    for poly in obstacles_raw:
        if not poly or len(poly) < 3:
            continue
        norm_poly: List[List[int]] = []
        for pt in poly:
            # pt가 [x,y] 형태라고 가정 (혹시 dict/tuple 섞여도 대비)
            x, y = pt[0], pt[1]
            norm_poly.append([int(x), int(y)])
        obstacles.append(norm_poly)

    # --- exits: 내부에서는 (x,y) 튜플 리스트로 맞춤 ---
    exits_raw = obj.get("exits", []) or []
    #print(exits_raw)
    exits: List[List[PointT]] = []
    for poly in exits_raw:
        if not poly or len(poly) < 3:
            continue
        norm_poly: List[PointT] = []
        for pt in poly:
            x, y = pt[0], pt[1]
            norm_poly.append((int(x), int(y)))
        exits.append(norm_poly)
    #self.exit_list = exits
    #(self.exit_list)
    return w, h, obstacles

def build_one(map_num: int, out_dir: str = OUT_DIR, D_: int = D):
    w, h, obs = load_map_geometry(map_num)

    fp = obstacles_fingerprint(obs)
    path = cache_path(out_dir, map_num, w, h, fp)

    artifacts = build_mesh_artifacts(w, h, obs, D=D_)
    payload = {
        "map_num": map_num,
        "width": w,
        "height": h,
        "obs_fp": fp,
        "artifacts": artifacts,
    }
    save_mesh_cache(path, payload)
    print(f"[OK] saved: {path}")


def main():
    for m in MAPS:
        build_one(m, out_dir=OUT_DIR, D_=D)


if __name__ == "__main__":
    main()
