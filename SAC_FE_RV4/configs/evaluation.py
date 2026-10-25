"""평가: 테스트 스위트, 베이스라인, 베이스라인 튜닝, 통계.

시나리오 = (맵, 대칭 변환, 군중 수, 초기 배치 시드). 같은 시나리오를 모든
조건(로봇 없음, 출구 직행 로봇, 셔틀 2종, 학습 정책, 사람)이 공유한다. 초기
배치가 같으므로 조건 간 비교는 짝지은(paired) 비교가 된다.

스위트는 cli.build_suites 가 한 번 만들어 data/suites/<이름>.json 으로
고정한다. 다시 만들면 내용이 바뀌므로, 논문용 수치는 커밋된 파일로 낸다.

상수만 정의한다.
"""

# ---------------- 스위트 ----------------
SUITE_DIR = "data/suites"
# 스위트 정의 (cli.build_suites 가 읽는다).
#   kind "generated": random_map 생성기로 새 맵. seed_base 부터 시드를 쓴다.
#                     시드 구간은 학습 생성기가 쓰지 않는 UED_RESERVED_SEED_RANGES
#                     안에 있어야 한다.
#   kind "numbered" : map_infos 및 코드에 박힌 번호 맵. maps="auto" 면 100 m 번호
#                     맵 중 학습 소스(LEAKAGE_SOURCE_MAPS)와 D4 대칭까지
#                     LEAKAGE_PIXEL_THRESHOLD 이내로 겹치는 맵을 뺀 전부.
#   kind "subset"   : 다른 스위트의 시나리오 일부 (사람 실험용).
SUITE_SPECS = {
    # 학습 중 모델 선택용. 테스트와 시드가 다르다.
    "val": dict(kind="generated", n_maps=32, sizes_m=(100,), difficulty=(1, 6),
                seed_base=9_100_000, crowd_seeds=1, crowd=30),
    # 베이스라인 속도 튜닝용. 검증·테스트와 시드가 다르다.
    "tune": dict(kind="generated", n_maps=48, sizes_m=(100,), difficulty=(1, 6),
                 seed_base=9_200_000, crowd_seeds=2, crowd=30),
    # 미지 환경 테스트: 수작업 스타일 맵 1500-1510 (2026-10-25 확정).
    # 1500-1504, 1507 은 RV3 의 기존 맵, 1505, 1506, 1508-1510 은
    # cli/design_test_maps.py 로 새로 만든 맵 (기존 160 m 1506 은
    # map_infos/archive/ 로 이동). 맵당 초기 배치 12 개 (RV3 실험과 같은 반복 수).
    "test": dict(kind="numbered", maps=tuple(range(1500, 1511)), crowd_seeds=12, crowd=30),
    # 맵 크기 일반화 (맵 스케일 입력 실험용). 군중 30 명 고정.
    "test_scale": dict(kind="generated", n_maps=20, sizes_m=(60, 140, 200),
                       difficulty=(1, 6), seed_base=9_500_000, crowd_seeds=2,
                       crowd=30),
    # 사람 조작 실험용 부분집합: test 의 11 개 맵에서 배치 1 개씩 (test 와 같은
    # 시나리오라 RL·베이스라인 결과와 그대로 짝지어진다).
    "human": dict(kind="subset", sources=(("test", 11),), crowd_seeds=1, seed=7),
}
# 미지 환경 성능의 헤드라인 스위트
EVAL_HEADLINE_SUITES = ("test",)
# 누수 검사: 이 번호 맵들과 D4 대칭 포함 픽셀 차이 비율이 임계값 미만이면
# 같은 맵으로 본다 (50x50 래스터, 장애물/출구/빈칸 3값).
LEAKAGE_SOURCE_MAPS = tuple(range(1000, 1300)) + (105, 108, 128)
LEAKAGE_PIXEL_THRESHOLD = 0.02

# ---------------- 실행 ----------------
EVAL_RESULTS_DIR = "RV4_eval"        # ~/RV4_eval/<실행 이름>/
EVAL_WORKERS = 8
# 에피소드 상한 스텝. 100 m 기준 3000 스텝(1500 s)이 RV3 실험·사람 실험과
# 같다. EVAL_MAX_STEPS_SCALE 이면 맵 한 변/100 m 에 비례해 늘린다.
EVAL_MAX_STEPS = 3000
EVAL_MAX_STEPS_SCALE = True
EVAL_POLICY_MODE = "deterministic"   # "deterministic" | "stochastic"
EVAL_CONDITIONS = ("no_robot", "exit_robot", "shuttle_random", "shuttle_nearest",
                   "policy")
# 부트스트랩 신뢰구간
EVAL_BOOTSTRAP_SAMPLES = 5000
EVAL_CI = 0.95
EVAL_WANDB = True                    # 평가 결과를 W&B 에 job_type="eval" 로 올린다

# ---------------- 베이스라인 ----------------
# no_robot        로봇 없음 (RV3 'N')
# exit_robot      가장 가까운 출구로 곧장 가서 멈추는 로봇 (RV3 'T', 그대로)
# shuttle_random  살아 있는 사람 1명을 무작위로 골라 navmesh 로 다가가서
#                 (PICKUP 반경 안에 들면) 가장 가까운 출구로 데려가고, 출구에서
#                 잠시 기다린 뒤 다시 고른다. 반복.
# shuttle_nearest 위와 같고, 사람을 로봇에서 가장 가까운(측지 거리) 사람으로 고른다.
# 두 셔틀은 접근 속도와 호송 속도를 따로 튜닝한다 (cli.tune_baselines).
# 결정 간격은 학습 정책과 같은 ACTION_SCALE 스텝이고, 행동 공간도 같다
# (축별 [-2, 2] m/s; 대각선으로는 최대 2.83 m/s).
BASELINE_DECISION_INTERVAL = 4
SHUTTLE_PICKUP_RADIUS_M = 2.0
SHUTTLE_EXIT_REACH_M = 2.0
SHUTTLE_EXIT_WAIT_STEPS = 8          # 출구 도착 후 대기 (4 s)
SHUTTLE_APPROACH_TIMEOUT_STEPS = 600 # 목표에 못 닿으면 다른 사람을 고른다
# 출구에서 이 거리(m) 안에 있는 사람은 혼자 나간다고 보고 고르지 않는다(위치만으로
# 판단). 없으면 로봇이 출구 모서리에 서서 바로 앞 사람을 막고, 그 사람을 다시
# "가장 가까운 사람"으로 고르는 교착이 생긴다(맵 128 에서 확인). 고를 사람이
# 없으면 로봇은 출구에서 비켜나 맵 중앙 쪽으로 간다.
SHUTTLE_IGNORE_NEAR_EXIT_M = 5.0
# "alive": 아직 대피하지 않은 모든 사람 (정책과 같은 정보: 위치만)
# "uninformed": 출구를 모르고 로봇을 따라오지도 않는 사람만 (특권 정보)
SHUTTLE_TARGET_FILTER = "alive"      # "alive" | "uninformed"
SHUTTLE_DISTANCE = "geodesic"        # nearest 기준: "geodesic" | "euclidean"
# 튜닝 전 기본 속도 (m/s). 튜닝 결과 파일이 있으면 그 값을 쓴다.
SHUTTLE_APPROACH_SPEED = 2.0
SHUTTLE_ESCORT_SPEED = 1.2
SHUTTLE_TUNED_FILE = "data/baselines/tuned_shuttle.json"

# ---------------- 베이스라인 튜닝 ----------------
# 목적: tune 스위트에서 평균 100% 대피 시간 최소화 (미완료는 상한으로 계산).
# successive halving (Jamieson & Talwalkar 2016): 모든 조합을 적은 시나리오로
# 돌리고 상위 1/ETA 만 남겨 시나리오를 ETA 배로 늘린다. 모든 조합이 같은
# 시나리오(같은 초기 배치)를 쓴다.
TUNE_SUITE = "tune"
# 속도 상한(ROBOT_SPEED_NORM_LIMIT) 2.0 에서는 2.0 을 넘는 값이 모두 2.0 이 된다.
# 상한 없이(None) 튜닝할 때는 (1.0, 1.5, 2.0, 2.5, 2.83) 을 썼다.
TUNE_APPROACH_SPEEDS = (1.0, 1.25, 1.5, 1.75, 2.0)
TUNE_ESCORT_SPEEDS = (0.6, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0)
TUNE_ETA = 3
TUNE_FIRST_RUNG_SCENARIOS = 12
