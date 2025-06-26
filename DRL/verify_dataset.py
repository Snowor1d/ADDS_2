# verify_dataset.py
import sys, numpy as np, pathlib, json, textwrap, model

def main(path):
    f = np.load(path)
    ok = True
    report = {}

    # ① 파일/키
    expected = {"states","next_states","actions","rewards","dones","source_id","size","capacity"}
    report["keys_ok"] = ok = ok and (expected <= set(f.files))

    # ② 길이
    N = int(f["size"])
    report["size_match"] = ok = ok and (N == len(f["actions"]) == len(f["states"]))

    # ③ dtype & 범위
    act = f["actions"]
    report["dtype_ok"]    = ok = ok and (act.dtype == np.float32)
    report["range_ok"]    = ok = ok and (np.abs(act).max() <= 2.0001)

    # ④ NaN / Inf
    report["finite_ok"]   = ok = ok and np.isfinite(act).all()

    # ⑤ 소스 비율
    src, cnt = np.unique(f["source_id"], return_counts=True)
    total = cnt.sum()
    report["ratio"] = {int(s): float(c)/total for s,c in zip(src,cnt)}

    # ⑥ 랜덤 5개 검증 실행
    try:
        env = model.FightingModel(20, 50, 50, robot="Q")
        for i in np.random.choice(N, 5, replace=False):
            _ = env.robot.receive_action(act[i])
        report["env_push_ok"] = True
    except Exception as e:
        report["env_push_ok"] = False
        report["env_error"]   = str(e)
        ok = False

    print(json.dumps(report, indent=2))
    if ok:
        print("\n✅  ALL CHECKS PASSED")
    else:
        print("\n❌  SOME CHECKS FAILED")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_dataset.py <offline_dataset_dql.npz>")
        sys.exit(1)
    main(sys.argv[1])
