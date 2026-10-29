"""시뮬레이터 물리·측정과 관측(observation) 계약.

군중 행동 모델과 로봇 동역학은 SAC_FE_RV3와 같다. 사람 조작(H) 실험 결과와
비교하려면 이 파일의 물리 값은 바꾸지 않는다. 관측 옵션(OBS_*)은 바꿔도
시뮬레이션 자체는 그대로이고, 정책이 보는 입력만 달라진다.

상수만 정의한다. import해도 부작용이 없다. 병합·검증은 configs/__init__.py.
"""

# ---------------- 관측 래스터 (RV3와 동일) ----------------
# 맵 전체를 월드 크기와 무관하게 MAP_H x MAP_W 픽셀로 정규화해 그린다.
# 벽/장애물 50, 출구 100, 군중 150, 로봇 255 (uint8).
# 100 m 맵이면 1 px = 1 m, 200 m 맵이면 1 px = 2 m 가 된다.
MAP_H = 100
MAP_W = 100
EGO_MAP_SIZE = 25            # 로봇 중심 ego crop (픽셀)
DOWNSAMPLE_MAP_SIZE = 50     # 전역 맵 max-pool 해상도 (픽셀)
FRAME_STACK = 4              # 결정 시점(ACTION_SCALE 스텝)마다 1장씩 쌓는다

# ego crop을 어떻게 자를지.
#   "normalized"  RV3와 동일: 정규화 래스터(MAP_H x MAP_W)에서 25 px를 자른다.
#                 100 m 맵에서 25 m, 200 m 맵에서 50 m 범위를 본다.
#   "metric"      월드 미터 기준(OBS_EGO_METERS_PER_PX)으로 자른다. 맵 크기가
#                 달라도 ego가 보는 물리적 범위가 같다. 100 m 맵에서는
#                 "normalized"와 같은 그림이 나온다.
OBS_EGO_MODE = "normalized"          # "normalized" | "metric"
OBS_EGO_METERS_PER_PX = 1.0

# 로봇을 따라오는 사람(type 0)을 다른 밝기로 그릴지. None이면 RV3처럼 모든
# 사람이 150. 정수(예: 200)면 추종자만 그 값으로 그린다. 로봇이 자기를
# 따라오는 사람을 지각한다는 가정이며, 관측 분포가 바뀌므로 RV3 체크포인트와
# 호환되지 않는다.
OBS_FOLLOWER_PIXEL = None

# ego/전역 입력에 정적 채널 1개(가장 가까운 출구까지의 측지(geodesic)
# 거리장, 0~1 정규화)를 추가할지. 맵의 위상(어디가 막혀 있고 어느 쪽이
# 출구로 이어지는지)을 네트워크가 원시 점유 격자에서 추론하지 않아도 되게
# 한다. 입력 채널 수가 바뀐다(4 -> 5).
OBS_STATIC_DISTANCE_CHANNEL = False

# 맵 스케일 입력: 한 변이 MAP_SCALE_UNIT_M 미터인 맵의 스케일을 MAP_SCALE_N
# 으로 둔다. 즉 50 m x 50 m 가 n 이면 100 m x 100 m 는 2n.
# 값은 (가로, 세로) 2개 스칼라로 들어간다(정사각형 맵에서는 같다). 정규화
# 래스터는 축마다 미터/픽셀이 다를 수 있기 때문이다. 액터/크리틱에 넣을지는
# configs/training/common.py 의 MAP_SCALE_TO_ACTOR / MAP_SCALE_TO_CRITIC.
MAP_SCALE_UNIT_M = 50.0
MAP_SCALE_N = 1.0

# ---------------- 맵 크기 ----------------
# 번호 맵 JSON(map_infos/map_<n>.json)의 width/height를 월드 크기로 쓸지.
# RV3는 JSON 값을 무시하고 모든 맵을 100 x 100으로 만들었다. 학습 맵
# 1000-1299 와 대부분의 테스트 맵은 100 x 100 이라 결과가 같고, 1506(160),
# 2000(200), 4000-4003(400) 만 달라진다.
MAP_SIZE_FROM_FILE = True
# 코드에 박힌 번호 맵(0, 6-134 등)의 크기. 비어 있으면 모두 100 m (RV3).
HARDCODED_MAP_SIZES = {}

# ---------------- 레거시 스크립트용 맵 선택 ----------------
# FightingModel(model_num=-1) 을 쓰는 옛 스크립트(run_sim.py,
# ADDS_AS_experiment.py 등)만 읽는다. RV4 학습·평가는 맵을 명시적으로 넘긴다.
MAP_NUM = -1                 # 고정 맵 번호, -1 이면 MAP_NUM_RANDOM 에서 무작위
MAP_NUM_RANDOM = list(range(1000, 1300))
# 옛 스크립트에서 맵을 만들 때마다 대칭 변환을 무작위로 적용할지.
# RV3 는 True 였고, 그래서 RV3 평가도 회전·반사된 맵에서 돌았다. RV4 학습은
# DATASET_AUGMENTATION / UED_AUGMENTATION 으로, 평가는 시나리오에 적힌
# 변환(기본 identity)으로 정한다.
MAP_DATA_AUGMENTATION = False
MAP_AUGMENTATION_TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)
CROWD_NUMBER_MIN = 30        # 옛 스크립트의 군중 수
CROWD_NUMBER_MAX = 30
RANDOM_EXIT = False

# ---------------- 시뮬레이션 물리 (RV3와 동일, 바꾸지 말 것) ----------------
ACTION_SCALE = 4             # 로봇 결정 간격 (시뮬레이션 스텝). 4 x 0.5 s = 2 s
MAX_STEPS = 6000             # 학습 에피소드 최대 스텝 (3000 s)
SCALE_CHECK = 0
ROBOT_BODY_RADIUS = 1        # m
AGENT_BODY_RADIUS = 0.5      # m
ROBOT_VISION = 10            # m
AGENT_VISION = 10            # m
EXIT_CONFIRM_RADIUS = 10
EXIT_CONFIRM_RADIUS_BONUS = 0
AGENT_SPEED_MEAN = 1.5       # m/s
ROBOT_SPEED_MAX = 2          # m/s. 행동 (dx, dy) in [-2, 2]^2 가 곧 속도(m/s)
# 로봇 명령 속도의 크기 상한 (m/s). 행동은 축별 [-2, 2] 이라 대각선으로는
# 2.83 m/s 까지 나가지만, 사람 조작(조이스틱)은 크기 2.0 m/s 에서 멈춘다.
# "사람보다 잘한다"를 같은 이동 능력에서 주장하려면 2.0 으로 둔다. 학습·평가·
# 베이스라인·사람 조작 모두 receive_action 을 거치므로 똑같이 적용된다.
# None = RV3 (제한 없음). 셔틀 튜닝 결과는 상한별로 따로 저장된다
# (data/baselines/tuned_shuttle.json 의 "shuttle_<종류>@<상한>").
# 2026-10-25: 사람 조작과 같은 이동 능력으로 비교하기 위해 기본값 2.0.
ROBOT_SPEED_NORM_LIMIT = 2.0
# 로봇이 맵 가장자리 1 m 띠(외곽 반발 경계가 W-1, H-1 에 있음)에 들어가면 RV3
# 동역학의 벽 반발력이 바깥쪽을 향해 로봇이 영구히 벽에 붙는다(어떤 입력으로도
# 못 빠져나옴; 셔틀 베이스라인에서 관찰, 2026-10-25). True 면 그 띠에서만 반발
# 방향을 안쪽으로 고친다. 경계 안쪽 동작은 RV3 와 같다. False = RV3.
ROBOT_EDGE_TRAP_FIX = True
# 사람·로봇의 agent.danger (출구까지 거리)를 재는 방법.
#   "euclidean" RV3 그대로: 가장 가까운 출구 다각형까지의 직선거리
#               (model.distance_to_exit). 벽을 무시해서, 출구 옆 막다른
#               골목에서는 실제 걷는 거리의 절반 이하가 된다(맵 1504: 직선
#               21.5 m, 걷는 거리 약 56 m).
#   "geodesic"  장애물을 돌아가는 걷는 거리 (learn/exit_distance.py: 맵마다
#               DANGER_GEO_RES 격자 거리장을 한 번 계산, 열린 공간에서는 직선
#               거리와 3 % 안으로 같다). 출구로 가는 길이 없는 곳은 "그 맵에서
#               도달 가능한 가장 먼 거리 + 맵 대각선": 어느 도달 가능한 곳보다
#               멀지만 유한해서 reward_b 가 발산하지 않고, 갇힌 사람이 아래
#               "danger > 1e5 이면 제거" 규칙에 걸리지 않는다.
# agent.danger 를 쓰는 곳: reward_b (생존자 거리 합), reward_d (로봇이 출구
# 5 m 안인지), reward_j, 그리고 "출구가 없는 맵이면 제거"(danger > 1e5,
# RV3; 출구가 없는 맵에서는 두 방식 모두 inf).
# 군중의 이동 결정은 danger 를 쓰지 않으므로 시뮬레이션 결과(대피 시간)는
# 바뀌지 않고 보상만 바뀐다.
DANGER_DISTANCE = "euclidean"      # "euclidean" | "geodesic"
DANGER_GEO_RES = 0.5               # m
AGENT_TIME_STEP = 0.5        # s
ROBOT_TIME_STEP = 0.5        # s
# 출구 점수 가중치 (crowd exit choice)
K1 = 1                       # distance
K2 = 0.000001                # density
K3 = 1                       # width

SHOW_CONTROLLED_CROWD = False
ENABLE_TIMER = True
USING_TRAINED_MODEL = True

# ---------------- 시뮬레이터 구현 ----------------
# 맵 생성(navmesh 최단경로, 격자-삼각형 대응, 가시성 사전계산)을 numpy로
# 벡터화한 구현을 쓸지. 결과는 RV3 구현과 비트 단위로 같다
# (tests/test_sim_equivalence.py). 100 m 맵 생성이 약 11 s -> 1~2 s.
SIM_FAST_BUILD = True
# 매 스텝 DataCollector 기록(에이전트별 위치·속도). 학습·평가는 읽지 않고
# 에피소드마다 수만 행이 쌓인다. 옛 분석 스크립트가 필요하면 True.
SIM_COLLECT_STATS = False
