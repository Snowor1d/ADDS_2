"""학습·평가 공통 설정.

configs/training/ 의 세 파일:

    common.py    이 파일: 학습 맵 소스(TRAIN_MAP_SOURCE), SAC, 네트워크,
                 일반화 기법, 보상, 리플레이, 워커, 검증, 체크포인트, 로깅·W&B
    dataset.py   TRAIN_MAP_SOURCE = "dataset" 일 때: 학습 맵 목록, 증강
    ued.py       TRAIN_MAP_SOURCE = "ued" 일 때: 커리큘럼(DR / PLR / ACCEL)

기본값은 "RV4 권장 설정"이다. RV3와 똑같이 돌리려면
presets/rv3_parity.json 을 쓴다 (README 의 프리셋 표 참고).

상수만 정의한다. 자격 증명(W&B 키)은 여기 두지 않는다. W&B 는 환경변수나
~/.netrc 에서 키를 읽는다.
"""

# =====================================================================
# 학습 맵 소스
# =====================================================================
#   "dataset"  번호 맵 목록(configs/training/dataset.py DATASET_MAPS)에서
#              에피소드마다 균등 추출. RV3의 300맵 학습이 이 모드다.
#   "ued"      절차적으로 생성한 레벨을 커리큘럼(configs/training/ued.py
#              UED_METHOD: dr | plr | accel)이 골라 준다.
#   "mixed"    매 에피소드 MIXED_DATASET_PROB 확률로 dataset, 나머지는 ued.
TRAIN_MAP_SOURCE = "dataset"            # "dataset" | "ued" | "mixed"
MIXED_DATASET_PROB = 0.3

# =====================================================================
# 기본 학습 파라미터 (RV3 값)
# =====================================================================
LR = 1e-4                    # 액터·크리틱 공통 학습률
LR_ACTOR = None              # None 이면 LR
LR_CRITIC = None             # None 이면 LR
# 결정 시점(transition) 개수. RV3는 상태·다음상태 스택을 따로 저장해
# 1M 에 약 25 GB 였다. RV4는 프레임을 한 번만 저장하고 스택은 인덱스로
# 다시 만든다(learn/replay.py): 1M 에 약 4 GB.
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 128
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500   # 이 에피소드 수 이후부터 업데이트 시작
DEVICE = "cuda"
GAMMA_START = 0.99           # 결정(2 s)당 할인율. 에피소드에 따라 START -> END
GAMMA_END = 0.99             # 로 GAMMA_SCHEDULE_STEP 에피소드 동안 선형 변화 (RV3)
GAMMA_SCHEDULE_STEP = 1000
# 타깃 네트워크 소프트 업데이트: target <- TAU*target + (1-TAU)*online
# (RV3 표기 그대로, TAU 는 "유지 비율")
TAU = 0.995
WD_Q = 3e-4                  # AdamW weight decay (critic)
WD_PI = 0.0                  # AdamW weight decay (actor)
GRAD_CLIP_NORM = None        # None 이면 클리핑 없음 (RV3)

# =====================================================================
# 탐험 (RV3 값): 결정마다 확률 epsilon 으로 균등 무작위 행동
# =====================================================================
EPSILON_MIN = 0
START_EPSILON = 1
SCHEDULER_TYPE = "l"         # "l" 선형 | "e" 지수
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 3000   # START_DECAY_STEP 부터 이 에피소드 수에 걸쳐 1 -> 0
START_DECAY_STEP = 500
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0
DECAY_MODE = "episode"

# =====================================================================
# SAC
# =====================================================================
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20            # RV3. 액터가 포화되면 -5 를 권장 (UED6 경험)
# ALPHA_AUTO=False 일 때 alpha 는 ALPHA_START -> ALPHA_END 로 ALPHA_DECAY_STEPS
# 에피소드 동안 선형 변화한다 (RV3 에 정의만 있고 호출되지 않던 스케줄).
ALPHA_START = 0.2
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000
# 엔트로피 온도 자동 조정 (Haarnoja et al. 2018b). False 면 ALPHA_START 고정
# (RV3: 고정 0.2, alpha optimizer 는 있었지만 쓰이지 않았다).
ALPHA_AUTO = False
ALPHA_LR = 3e-4
ALPHA_TARGET_ENTROPY = -2.0  # 행동 차원 2 -> -2
ALPHA_MIN = 0.0
# 크리틱 앙상블 (REDQ, Chen et al. 2021). 2/2 가 일반 SAC 의 twin-Q.
# 예: 10/2 면 10개 크리틱, 타깃은 무작위 2개의 min, 액터는 전체 평균.
CRITIC_ENSEMBLE = 2
CRITIC_TARGET_SUBSET = 2
# n-step 반환 (1 = RV3). 에피소드 끝에서는 남은 스텝만큼만 합산한다.
N_STEP = 1
# 크리틱 업데이트 k 번마다 액터 1번 (1 = RV3)
POLICY_DELAY = 1
# 시간 제한(MAX_STEPS)으로 끊긴 에피소드를 종료(terminal)로 취급하지 않고
# 부트스트랩한다 (Pardo et al. 2018 "Time Limits in RL"). RV3 는 끊긴
# 마지막 상태의 가치를 0 으로 학습했다(False 가 RV3 동작).
TIMEOUT_BOOTSTRAP = True
# 시그모이드 스쿼시의 log-Jacobian 을 log-sigmoid 형태로 계산한다. RV3 식
# log(4 s (1-s) + 1e-8) 은 |u| 가 크면 log(1e-8) 에 붙어 기울기가 0 이 된다.
SAC_STABLE_LOGPROB = True

# =====================================================================
# 네트워크
# =====================================================================
# 인코더 계열:
#   "legacy_rv3"  RV3 네트워크 그대로 (BatchNorm CNN, flatten). RV3 체크포인트
#                 로딩·재현용. NET_SIZE/NET_NORM 등은 무시된다.
#   "cnn"         stride-2 합성곱 3층 -> (flatten | pool) -> 임베딩. 바닐라 CNN.
#   "impala"      IMPALA-CNN (Espeholt et al. 2018): 좌표 채널 + (conv,
#                 max-pool, 잔차블록 2개) x 3 -> 4x4 평균 풀링 -> 임베딩.
#                 Procgen 에서 미지 레벨 일반화가 크게 좋았고 폭을 넓힐수록
#                 더 좋아졌다 (Cobbe et al. 2020).
# 크기 s / m / l / xl. 파라미터 수와 학습 비용 (이 PC RTX 4060 8 GB 실측,
# 2026-10-24, 크리틱 2개+타깃+액터+AdamW, ENCODER_SHARING="separate",
# python3 -m cli.measure_networks 로 다시 잴 수 있다):
#
#   계열/크기     액터     크리틱1개  업데이트1회   학습 VRAM          워커추론
#                                    (배치128)   배치128 / 배치256    (CPU 1스레드)
#   legacy_rv3    4.6 M    8.9 M      32 ms       0.7 / 1.0 GB        1.5 ms
#   cnn s         0.7 M    0.8 M      25 ms       0.2 / 0.3 GB        0.5 ms
#   cnn m         2.8 M    3.0 M      18 ms       0.4 / 0.6 GB        0.8 ms
#   cnn l         6.2 M    6.6 M      32 ms       0.6 / 0.9 GB        1.8 ms
#   cnn xl       11.0 M   11.5 M      48 ms       1.0 / 1.4 GB        3.3 ms
#   impala s      0.6 M    0.9 M      39 ms       0.2 / 0.4 GB        1.2 ms
#   impala m      1.7 M    2.0 M      45 ms       0.4 / 0.7 GB        2.3 ms
#   impala l      3.9 M    4.3 M      79 ms       0.7 / 1.2 GB        4.2 ms
#   impala xl     7.0 M    7.4 M     106 ms       1.0 / 1.6 GB        7-15 ms
#
#   학습 VRAM = PyTorch 예약 피크. nvidia-smi 로는 CUDA context 약 0.1-0.3 GB
#   가 더 보인다. 활성값이 대부분이라 배치에 거의 비례한다(배치 512 면 약 2배).
#   모든 조합이 배치 512 까지 8 GB GPU 에 들어간다. 병목은 VRAM 이 아니라 업데이트
#   시간이다: UPDATES_PER_TRANSITION=1 이면 초당 transition 처리량이
#   1 / 업데이트시간 으로 묶인다(legacy 약 30/s, impala xl 약 9/s). impala l/xl
#   이나 앙상블(CRITIC_ENSEMBLE>2)·DrQ K/M>1 을 쓰면 UPDATES_PER_TRANSITION 을
#   낮추거나 ENCODER_SHARING="critic" 으로 비용을 줄인다.
#   impala 는 50x50 전역 입력에 stride-2 stem 을 둔다(없으면 l/xl 가 2배 느림).
NET_ENCODER = "cnn"                 # "legacy_rv3" | "cnn" | "impala"
NET_SIZE = "l"                      # "s" | "m" | "l" | "xl"
# cnn 계열의 정규화. "batch" 는 RV3 와 같지만, 워커는 eval 모드(running
# 통계)·학습기는 train 모드(배치 통계)로 서로 다른 함수를 계산하고 타깃
# 네트워크 통계도 따로 놀아 off-policy RL 에서 불안정 요인이다.
NET_NORM = "group"                  # "batch" | "group" | "layer" | "none"
# cnn 계열의 공간 처리: "flatten" 은 위치별 선형층(파라미터 대부분이 여기),
# "pool" 은 4x4 평균 풀링 후 임베딩(위치 불변성↑, 파라미터↓).
NET_SPATIAL = "flatten"             # "flatten" | "pool"
NET_COORD_CHANNELS = False          # cnn 입력에 좌표 채널 2개 추가 (impala 는 항상)
# 크리틱 MLP 은닉층에 LayerNorm (Ball et al. 2023 RLPD: off-policy 안정성과
# 외삽 억제에 핵심) / Dropout (DroQ, Hiraoka et al. 2022).
CRITIC_LAYERNORM = False
CRITIC_DROPOUT = 0.0
# 인코더 공유:
#   "separate"  액터와 각 크리틱이 자기 인코더를 가진다 (RV3).
#   "critic"    크리틱들이 인코더 하나를 공유하고 크리틱 손실로만 학습한다.
#               액터는 그 특징을 detach 해서 쓴다 (SAC-AE / DrQ 방식,
#               Yarats et al. 2019, 2021). 업데이트 비용이 크게 준다.
ENCODER_SHARING = "separate"        # "separate" | "critic"
FiLM_USE = True                     # 크리틱: (행동, 로봇상태)로 이미지 특징 FiLM
EGO_USE = True
# 맵 스케일 입력 (값의 정의는 configs/environment.py MAP_SCALE_UNIT_M/N).
# 크기가 다양한 맵에서 학습할 때 켠다. 로봇 상태 벡터에 2개 스칼라가 붙는다.
MAP_SCALE_TO_ACTOR = False
MAP_SCALE_TO_CRITIC = False

# =====================================================================
# 데이터 증강 (리플레이에서 배치를 뽑을 때)
# =====================================================================
# 무작위 평행이동 (RAD / DrQ, Laskin et al. 2020; Kostrikov et al. 2021):
# 패딩(벽 값) 후 원래 크기로 무작위 crop. 상태와 다음 상태에 독립 적용.
AUG_RANDOM_SHIFT = False
AUG_SHIFT_PAD_EGO = 1               # 픽셀
AUG_SHIFT_PAD_GLOBAL = 2            # 픽셀
# DrQ 평균: 타깃을 K 개 증강의 평균, Q 손실을 M 개 증강의 평균으로.
AUG_DRQ_K = 1
AUG_DRQ_M = 1
# 리플레이 D4 대칭 증강: 샘플마다 회전·반사 8개 중 하나를 관측, 로봇 위치,
# 맵 스케일, 행동에 일관되게 적용한다. 보상은 대칭에 불변이다.
AUG_REPLAY_D4 = False

# =====================================================================
# 가소성(plasticity) 유지
# =====================================================================
# 업데이트 RESET_EVERY_UPDATES 번마다 네트워크를 재초기화한다 (primacy bias,
# Nikishin et al. 2022). 리플레이는 유지한다. 0 이면 끔.
#   RESET_SCOPE "heads": 액터·크리틱의 MLP 헤드만 / "all": 인코더 포함 전체
# SHRINK_PERTURB_ALPHA > 0 이면 재초기화 대신 shrink & perturb
# (theta <- alpha*theta + (1-alpha)*theta_init, Ash & Adams 2020).
RESET_EVERY_UPDATES = 0
RESET_SCOPE = "heads"               # "heads" | "all"
SHRINK_PERTURB_ALPHA = 0.0

# =====================================================================
# 보상 (RV3 가중치)
# =====================================================================
REWARD_A = 2                 # -살아있는 인원 비율
REWARD_B = 0.003             # -모든 생존자의 출구까지 거리 합
REWARD_D = 2                 # 추종자 없이 출구 근처에 있으면 -1
REWARD_K = 6                 # 벽 충돌 -1
REWARD_J = 0                 # all_agents_danger_root
REWARD_L = 1                 # -가장 먼 생존자 거리 / 맵 대각선
REWARD_FIXED = -0.5          # 결정당 상수
REWARD_I = 0                 # alived_root
REWARD_C = 0                 # gain
REWARD_E = 0                 # evacuated_with_robot
REWARD_F = 0                 # distance_from_near_agents
REWARD_G = 0                 # distance_from_near_agent_gain
REWARD_H = 0                 # gain_with_time_bonus
FINISHED_BONUS = 0
INTRINSIC_ETA = 0.1          # 사용 안 함 (RV3 잔재)
# 충돌 벌점 계산: "last_step" 은 RV3 동작(결정 4 스텝 중 마지막 스텝만 봄),
# "sum" 은 4 스텝 모두 합산.
REWARD_COLLISION_MODE = "last_step"   # "last_step" | "sum"
# 거리 합 항(REWARD_B)을 맵 한 변/100 m 로 나눠 크기가 다른 맵 사이 보상
# 크기를 맞춘다. 100 m 맵에서는 값이 같다. 여러 크기 맵으로 학습할 때 권장.
REWARD_SCALE_INVARIANT = False

# =====================================================================
# 워커 (롤아웃 프로세스)
# =====================================================================
N_ENVS_WARMUP = 10           # START_UPDATE_EPISODE 전 (업데이트 없음)
N_ENVS = 4                   # 업데이트 시작 후
UPDATES_PER_TRANSITION = 1   # transition 하나당 gradient update 수
POLICY_BROADCAST_INTERVAL = 10   # 에피소드마다 워커에 정책 전송
# 워커가 transition 을 몇 개씩 묶어 보낼지 (IPC 오버헤드 절감)
WORKER_SEND_CHUNK = 8

# =====================================================================
# 주기적 검증과 모델 선택
# =====================================================================
# 별도 프로세스가 고정 스냅샷 정책을 검증 스위트(data/suites/<이름>.json)에서
# 돌린다. 학습 맵·테스트 맵과 겹치지 않는 시드로 생성된 레벨이다.
PERIODIC_VALIDATION = True
VALIDATION_SUITE = "val"
VALIDATION_CYCLE_EPISODE = 500
VALIDATION_WORKERS = 4
VALIDATION_POLICY_MODE = "deterministic"   # "deterministic" | "stochastic"
# best_validation.pth 선택 지표: 평균 100% 대피 시간(작을수록 좋음)
VALIDATION_SELECT_METRIC = "evac_time_100_mean"

# =====================================================================
# 체크포인트와 재개
# =====================================================================
CHECKPOINT_EVERY_EPISODES = 100
KEEP_LAST_CHECKPOINTS = 5
# "latest": LOG_DIR 의 최신 체크포인트에서 재개(설정 fingerprint 가 다르면
# 시작하지 않음) / "fresh": 무시하고 새로 시작
RESUME_MODE = "latest"
# 리플레이 버퍼 저장. RV3 는 100 에피소드마다 수 GB 를 압축 저장해 학습이
# 멈췄다. 기본은 저장하지 않고, 재개 시 워밍업 없이 바로 다시 채운다.
SAVE_REPLAY = False
# 리플레이 없이 재개했을 때, 업데이트를 다시 시작하기 전에 모을 transition 수
# (작은 버퍼에 과적합하지 않게).
RESUME_REFILL_TRANSITIONS = 20000
MAX_EPISODES = 0             # 0 = 제한 없음 (RV3: 9999999)
# 메인 루프 한 바퀴에 최대 업데이트 수 (메시지 처리가 밀리지 않게)
MAX_UPDATES_PER_LOOP = 64

# =====================================================================
# 로깅과 W&B
# =====================================================================
LOG_DIR = "Log_SAC_FE_RV4"           # ~/<LOG_DIR>
PORT_NUM = 7754
EXPERIMENT_ID = "fe-rv4"
LOG_TRAIN_EVERY_UPDATES = 200
LOG_TXT_COMPAT = True                # RV3 형식 txt (total_reward.txt 등)도 계속 기록
TENSORBOARD = True
# W&B. "online" | "offline" | "disabled". W&B 실패는 학습을 멈추지 않는다.
WANDB_MODE = "online"
WANDB_PROJECT = "adds-sac-fe-rv4"
WANDB_ENTITY = None                  # None = 계정 기본 entity
WANDB_GROUP = None                   # None = EXPERIMENT_ID
WANDB_RUN_NAME = "SAC_FE_RV3_aug_L"                # None = 모드·인코더·시각으로 자동 생성
WANDB_TAGS = ()
WANDB_UPLOAD_CHECKPOINTS = "selected"   # "none" | "selected" (검증으로 고른 모델만)
# 검증 때 정책 에피소드 1개를 영상으로 W&B 에 올릴지 (continuous_renderer)
VALIDATION_VIDEO = False
