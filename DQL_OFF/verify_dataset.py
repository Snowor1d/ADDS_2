#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_dataset.py ─ offline_dataset_dql.npz 무결성 검사 (Py 3.9 호환)
"""

from __future__ import annotations
import sys, json, pathlib, numpy as np
import model                       # FightingModel

DEF_PATH = pathlib.Path.home() / "Log_DQL" / "offline_dataset_dql.npz"


# ────────────────── 헬퍼 ──────────────────
def to_py(x):
    """np.*  →  파이썬 기본 타입으로 변환 (json 직렬화 가능)"""
    if isinstance(x, (np.generic,)):   # np.bool_, np.int64 등
        return x.item()
    return x


def verify(path: pathlib.Path):
    f            = np.load(path)
    report       = {}
    ok: bool     = True               # 전체 통과 여부 플래그

    # ① 키 존재 여부 --------------------------------------------------
    exp_keys = {"states", "next_states", "actions", "rewards",
                "dones", "source_id", "size", "capacity"}
    keys_ok  = exp_keys <= set(f.files)
    ok       = ok and keys_ok
    report["keys_ok"] = keys_ok

    # ② 길이 일치 ------------------------------------------------------
    N            = int(f["size"])
    size_match   = (N == len(f["actions"]) == len(f["states"]))
    ok           = ok and size_match
    report["size_match"] = size_match

    # ③ dtype / 범위 / finite -----------------------------------------
    act         = f["actions"]
    dtype_ok    = (act.dtype == np.float32)
    range_ok    = (np.abs(act).max() <= 2.0001)
    finite_ok   = np.isfinite(act).all()
    ok          = ok and dtype_ok and range_ok and finite_ok
    report.update({
        "actions_dtype": str(act.dtype),
        "dtype_ok": dtype_ok,
        "range_ok": range_ok,
        "finite_ok": finite_ok,
    })

    # ④ state 배열 형태 ------------------------------------------------
    st, nst = f["states"], f["next_states"]
    states_ok = (st.shape == nst.shape and st.dtype == np.uint8)
    ok        = ok and states_ok
    report["state_shape"] = list(st.shape)
    report["states_ok"]   = states_ok

    # ⑤ 소스 비율 -----------------------------------------------------
    src, cnt     = np.unique(f["source_id"], return_counts=True)
    report["ratio"] = {int(s): float(c) / cnt.sum() for s, c in zip(src, cnt)}

    # ⑥ env.push 테스트 -----------------------------------------------
    try:
        env = model.FightingModel(20, 50, 50, model_num=-1, robot="Q")
        for i in np.random.choice(N, min(5, N), replace=False):
            env.robot.receive_action(act[i])
        env_ok = True
    except Exception as e:
        env_ok, ok = False, False
        report["env_error"] = str(e)
    report["env_push_ok"] = env_ok

    # numpy 타입 → 파이썬 타입 변환
    for k, v in list(report.items()):
        if isinstance(v, dict):
            report[k] = {kk: to_py(vv) for kk, vv in v.items()}
        else:
            report[k] = to_py(v)

    return ok, report


def main():
    path = pathlib.Path(sys.argv[1] if len(sys.argv) == 2 else DEF_PATH)
    if not path.exists():
        sys.exit(f"[ERROR] 파일이 없습니다: {path}")

    ok, rep = verify(path)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print("\n" + ("✅ ALL CHECKS PASSED" if ok else "❌ SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
