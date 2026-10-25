"""TRAIN_MAP_SOURCE = "ued": 비지도 환경 설계(Unsupervised Environment Design).

레벨 = (맵 크기, 장애물 다각형, 출구 다각형, 군중 수). 새 레벨은
random_map.py 의 절차적 생성기(난이도 1-6)로 만든다.

방법 (UED_METHOD):
  "dr"     Domain Randomization. 매 에피소드 새 레벨. 커리큘럼 없음.
  "plr"    Robust Prioritized Level Replay (Jiang et al. 2021a,b). 점수가
           높은 레벨을 우선 재생(rank 우선순위 + staleness). 확률
           1-UED_P_REPLAY 로 새 레벨을 시도한다.
  "accel"  ACCEL (Parker-Holder et al. 2022). PLR 에 더해, 점수 상위 레벨을
           조금씩 편집(장애물 이동·크기·추가·삭제, 출구 이동·폭, 군중 수)한
           자식 레벨을 개체군에 넣는다. 난이도가 에이전트의 능력 경계를
           따라 점진적으로 올라간다.

점수 (UED_SCORE):
  "learnability"   p(1-p), p = 성공률 (SFL, Rutherford et al. 2024).
                   성공 = 100% 대피 시간이 기준(UED_SUCCESS_REFERENCE)의
                   UED_SUCCESS_K 배 이내. Beta(1,1) 평활.
  "maxmc"          mean_t (R_max - V(s_t)) (Jiang et al. 2021b). SAC 크리틱으로
                   계산. 시도 1번으로 점수가 나온다.
  "pvl"            Positive Value Loss: mean_t max(0, GAE_t) (Jiang et al. 2021a).
  "hybrid"         시도 UED_MIN_TRIALS 회 전에는 maxmc, 이후 learnability.
  "shuttle_regret" (T_agent - T_shuttle)/T_shuttle 의 평균. 같은 레벨에서
                   튜닝된 셔틀 베이스라인보다 얼마나 못하는가 = 베이스라인을
                   antagonist 로 둔 regret 근사(PAIRED 의 변형). 새 레벨마다
                   셔틀 에피소드를 1번 더 돌린다.

상수만 정의한다.
"""

UED_METHOD = "accel"             # "dr" | "plr" | "accel"
UED_SCORE = "learnability"       # "learnability" | "maxmc" | "pvl" | "hybrid" | "shuttle_regret"
# learnability 의 성공 기준.
#   "freeflow"  모두가 출구를 알고 곧장 걷는다고 할 때의 시간(측지 거리 /
#               개인 희망 속도의 최댓값)의 K 배. 레벨마다 정규화된다.
#   "shuttle"   튜닝된 nearest-person 셔틀 베이스라인 시간의 K 배.
#               K = 1 이면 "셔틀보다 빨리 끝냈는가" 가 성공이다.
UED_SUCCESS_REFERENCE = "freeflow"   # "freeflow" | "shuttle"
# 튜닝된 셔틀 베이스라인의 (실제 / freeflow) 비율은 tune 스위트에서 중앙값
# 3.4-3.6, 10-90 백분위 2.0-5.4 였다 (2026-10-24). K=3.5 면 "성공"은 대략
# "베이스라인 중앙값만큼 해냄" 이고, 이 경계 근처 레벨에서 p(1-p) 가 커진다.
# 로그의 ued/evac_ratio_mean 을 보고 조정한다.
UED_SUCCESS_K = 3.5
UED_MIN_TRIALS = 3               # hybrid 전환·번식 자격에 필요한 시도 수
UED_TRIAL_HISTORY = 10           # 성공률을 계산할 최근 시도 수
UED_TRIAL_MAX_AGE = 4000         # 이보다 오래된(에피소드) 시도는 버린다
UED_PVL_LAMBDA = 0.95            # pvl 의 GAE lambda
UED_SCORE_MAX_STATES = 256       # maxmc/pvl 계산에 쓰는 에피소드당 최대 상태 수

# 개체군 / 샘플링 (PLR)
UED_POP_SIZE = 400
UED_P_REPLAY = 0.5               # plr: 재생 확률 (나머지는 새 레벨)
UED_P_NEW_START = 0.5            # accel: 새 레벨 확률 (초기)
UED_P_NEW_END = 0.1              # accel: 새 레벨 확률 (UED_P_NEW_DECAY_EPISODES 후)
UED_P_NEW_DECAY_EPISODES = 5000
UED_SCORE_TEMPERATURE = 0.3      # rank 우선순위 온도 beta (PLR/ACCEL 0.3)
UED_STALENESS_COEF = 0.3         # rho
# Robust PLR(PLR-perp): 새 레벨에서 나온 transition 은 점수만 매기고
# 리플레이에 넣지 않는다. off-policy SAC 에서는 정확히 같은 효과가 아니다.
UED_REPLAY_ONLY_UPDATES = False

# ACCEL 번식
UED_MUTATE_THRESHOLD = 0.75      # 점수 백분위가 이 이상인 레벨만 번식
UED_MAX_CHILDREN_PER_LEVEL = 3
UED_MUTATIONS_PER_CHILD = (1, 3) # 자식 하나에 적용할 편집 수 (양끝 포함)
UED_MUTATE_MAX_TRIES = 30
# 편집 종류별 가중치
UED_MUTATION_WEIGHTS = {
    "move_obstacle": 3.0,
    "scale_obstacle": 2.0,
    "add_obstacle": 2.0,
    "remove_obstacle": 1.0,
    "move_exit": 1.5,
    "resize_exit": 1.0,
    "crowd": 1.0,
}
UED_MUTATE_MOVE_M = (2.0, 8.0)       # 장애물 평행이동 거리 범위
UED_MUTATE_SCALE = (0.8, 1.25)       # 장애물 크기 배율 범위
UED_MUTATE_ADD_SIZE_M = (4, 15)      # 추가 장애물 한 변 범위
UED_MUTATE_EXIT_SHIFT_M = (3.0, 15.0)
UED_MUTATE_EXIT_ALONG_M = (5, 12)    # 출구 폭 범위
UED_MUTATE_CROWD_STEP = 5            # 군중 수 편집 폭 (+/-)
# 장애물 면적 비율 상한 (편집이 맵을 막아 버리지 않게)
UED_MAX_OBSTACLE_DENSITY = 0.55

# 새 레벨 생성기 (random_map.generate_map)
UED_MAP_SIZES_M = (100,)         # 맵 한 변 후보 (m). 예: (70, 100, 140)
UED_MAP_SIZE_WEIGHTS = None      # None 이면 균등
UED_DIFFICULTY_RANGE = (1, 6)    # random_map 난이도 (양끝 포함, 균등)
# 생성기의 장애물 크기 파라미터는 미터 단위 절댓값이라 작은 맵(<100 m)에서
# 고난이도 생성이 매우 느리다(70 m, 난이도 6: 평균 58 s, 최대 150 s).
# 이 시간을 넘기면 포기하고 다른 시드로 다시 뽑는다.
UED_GEN_TIMEOUT_S = 20.0
UED_CROWD_RANGE = (30, 30)       # 새 레벨의 군중 수 (양끝 포함)
UED_CROWD_LIMITS = (10, 60)      # 편집으로 바뀔 수 있는 군중 수 범위
# 새 레벨에 D4 대칭을 무작위 적용할지 (학습 시). 평가는 항상 원본.
UED_AUGMENTATION = True
UED_AUGMENTATION_TRANSFORMS = (
    "identity", "rotate_90", "rotate_180", "rotate_270",
    "reflect", "reflect_rotate_90", "reflect_rotate_180", "reflect_rotate_270",
)

# 실행
UED_WARMUP_EPISODES = 500        # 이 전에는 DR (개체군을 채우는 기간)
# epsilon 이 이보다 클 때의 결과는 점수에 쓰지 않는다 (무작위 행동이 푼
# 레벨을 정책의 능력으로 착각하지 않게).
UED_SCORE_MAX_EPSILON = 0.25
UED_GENERATION_WORKERS = 2       # 새 레벨 생성 프로세스 수
UED_FRESH_QUEUE = 32             # 미리 만들어 둘 새 레벨 수
UED_LEVEL_QUEUE_SIZE = 2         # 워커별 레벨 대기열 깊이
UED_SNAPSHOT_EPISODES = 500      # 개체군 그림(PNG)을 W&B 에 올리는 주기 (0 = 끔)
# 학습 생성기가 절대 쓰지 않는 시드 구간 (검증·튜닝·테스트 스위트용).
UED_RESERVED_SEED_RANGES = ((9_000_000, 10_000_000),)
