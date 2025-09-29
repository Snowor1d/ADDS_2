#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dataset.py ─ offline_dataset_dql.npz 무결성 검사  (Py 3.9 호환)
· 2025-07 수정:  source_id 필드가 없는 버퍼와 호환
"""

from __future__ import annotations
import sys, json, pathlib, numpy as np
import model          # FightingModel  ─ 시뮬레이터

DEF_PATH = pathlib.Path.home() / "Log_DQL" / "offline_dataset_dql.npz"


def _py(v):
    """np.* → 파이썬 기본형 (json 직렬화용)"""
    return v.item() if isinstance(v, np.generic) else v


def verify(path: pathlib.Path):
    f        = np.load(path, allow_pickle=False)
    report   = {}
    ok: bool = True

    # ① 필수 키 -------------------------------------------------------
    exp = {"states", "next_states", "actions", "rewards", "dones", "size", "capacity"}
    keys_ok = exp <= set(f.files)
    ok &= keys_ok
    report["keys_ok"] = keys_ok

    # ② 길이 일치 -----------------------------------------------------
    N  = int(f["size"][0])                  # ← [0] 로 스칼라 추출
    sz_ok = (N == len(f["states"]) == len(f["actions"]))
    ok &= sz_ok
    report["size_match"] = sz_ok
    report["size"]       = N

    # ③ dtype / 범위 / 유한성 ----------------------------------------
    act       = f["actions"]
    dtype_ok  = (act.dtype == np.float16)
    range_ok  = np.abs(act).max() <= 2.0001
    finite_ok = np.isfinite(act).all()
    ok &= dtype_ok and range_ok and finite_ok
    report.update({
        "actions_dtype" : str(act.dtype),
        "dtype_ok"      : dtype_ok,
        "range_ok"      : range_ok,
        "finite_ok"     : finite_ok,
    })

    # ④ states 배열 검증 ---------------------------------------------
    st, nst = f["states"], f["next_states"]
    states_ok = (st.shape == nst.shape and st.dtype == np.uint8)
    ok &= states_ok
    report["state_shape"] = list(st.shape)
    report["states_ok"]   = states_ok

    # ⑤ env.receive_action 무작위 샘플 테스트 -------------------------
    try:
        env = model.FightingModel(20, 50, 50, model_num=-1, robot="Q")
        for i in np.random.choice(N, min(5, N), replace=False):
            env.robot.receive_action(act[i])
        env_ok = True
    except Exception as e:
        env_ok, ok = False, False
        report["env_error"] = str(e)
    report["env_push_ok"] = env_ok

    # numpy → 파이썬 형으로 정리
    report = {k: _py(v) for k, v in report.items()}
    return ok, report


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) == 2 else DEF_PATH)
    if not path.exists():
        sys.exit(f"[ERROR] file not found: {path}")

    ok, rep = verify(path)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\n" + ("✅ ALL CHECKS PASSED" if ok else "❌ SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
