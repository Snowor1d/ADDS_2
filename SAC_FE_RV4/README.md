# SAC_FE_RV4

실내 화재 대피 로봇(단일 로봇, SAC) RV3의 후속판입니다. 목표는 **처음 보는 맵(unknown 환경)에서 사람이 조작한 로봇보다 빨리 모두를 대피시키는 것**이고, 이를 위해 학습 맵 생성(UED), 평가, 로깅을 다시 만들었습니다.

시뮬레이터(군중 행동 모델, 로봇 동역학, 보상 함수)는 RV3와 같습니다. 사람 조작(H) 실험과 비교하려면 이 부분이 같아야 하기 때문입니다. 맵 생성만 결과가 같은 벡터화 구현으로 바꿔 약 16배 빨라졌습니다.

## 1. RV3에서 발견해 고친 문제

| 문제 | 영향 | RV4 |
| --- | --- | --- |
| 학습 300맵(1000-1299) 중 69개가 맵 150-249의 57개와 실제 건물 맵 300/303/306/307의 사본 (D4 대칭 포함 픽셀 2% 이내) | 이 맵들에서의 "unknown" 평가는 실제로는 학습 맵 평가 | 테스트 스위트를 만들 때 학습 소스와의 중복을 자동 검사해 제외 (`evaluation/suites.py`) |
| unknown 테스트가 맵 3개(1500/1502/1503) × 12회 | 표준편차 200-430 step이라 개선 여부를 판정할 수 없음 | 수작업 스타일 맵 1500-1510 (11개, 5개 신규) × 초기 배치 12 = 132 시나리오, 맵 단위 클러스터 부트스트랩 신뢰구간 |
| 시간 제한(6000 step)을 종료 상태로 처리 | 끊긴 마지막 상태의 가치를 0으로 학습 (Pardo et al. 2018) | `TIMEOUT_BOOTSTRAP=True` (종료와 시간 제한을 분리 저장) |
| 리플레이가 상태·다음상태 스택을 따로 저장 (1M ≈ 25 GB), 100에피소드마다 압축 저장 | 이 PC(RAM 31 GB)에서 위험, 저장 중 학습 정지 | 프레임을 한 번만 저장하고 스택은 인덱스로 재구성 (1M ≈ 4 GB), 저장은 선택 |
| BatchNorm (워커 eval 모드 / 학습기 train 모드) | 행동 정책과 학습 정책이 다른 함수 | 기본 GroupNorm, RV3 재현용 `legacy_rv3`만 BatchNorm |
| 로봇 없음('N') 모드가 맵 생성 단계에서 크래시 (`self.robot` 참조) | 'N' 베이스라인을 현재 코드로 돌릴 수 없음 | 수정 (결과 동일) |
| 로봇의 외곽 벽 반발 경계가 W-1, H-1에 있어, 가장자리 1 m 띠에 들어간 로봇은 바깥쪽으로 밀려 영구히 벽에 붙음 | RL·셔틀·사람 어느 조작으로도 탈출 불가, 에피소드 실패 | `ROBOT_EDGE_TRAP_FIX=True` (그 띠에서만 반발 방향을 안쪽으로; 경계 안 동작은 동일) |
| 시그모이드 스쿼시의 log-Jacobian `log(4s(1-s)+1e-8)` | 포화 영역에서 기울기 0 | `SAC_STABLE_LOGPROB=True` (log-sigmoid 형태) |
| 정책 브로드캐스트가 `episode % 10 == 0`인 동안 매 transition마다 반복 | 불필요한 state_dict 복사 | 에피소드 단위로 한 번 |
| 맵 생성 10-13 s(100 m), 약 50 s(160-200 m) | 레벨을 계속 새로 만드는 UED에 부적합 | 0.6 s / 1.5 s, RV3 구현과 비트 단위 동일 (`tests/test_sim_equivalence.py`) |

## 2. 폴더

| 위치 | 내용 |
| --- | --- |
| `configs/` | 설정 (아래 3절). `config.py`는 옛 `from config import *`용 재내보내기 |
| `model.py`, `agent.py`, `space.py`, `core.py`, `visibility_atlas.py` | 시뮬레이터 (RV3) |
| `sim_accel.py` | 맵 생성 벡터화 (`SIM_FAST_BUILD`) |
| `sim_maps.py` | 번호 맵의 크기·형상 |
| `learn/` | 관측, 네트워크, 리플레이, SAC, 워커, 트레이너, W&B 로거, 검증, 체크포인트 로딩 |
| `ued/` | 레벨, 생성기, ACCEL 편집, 개체군(PLR/ACCEL), 작업 배분 |
| `evaluation/` | 테스트 스위트, 베이스라인, 병렬 실행, 통계, 리포트, 영상 |
| `cli/` | `build_suites`, `tune_baselines`, `evaluate`, `compare`, `measure_networks`, `view_suite` |
| `presets/` | 실험 프리셋 (JSON) |
| `data/suites/` | 고정된 테스트 스위트 (커밋해서 쓸 것) |
| `data/baselines/tuned_shuttle.json` | 셔틀 베이스라인 튜닝 결과 |
| `tests/` | 회귀 테스트 |

## 3. 설정

| 파일 | 담당 |
| --- | --- |
| `configs/environment.py` | 시뮬레이터 물리(바꾸지 말 것), 관측 옵션(`OBS_*`, 맵 스케일 정의) |
| `configs/training/common.py` | `TRAIN_MAP_SOURCE`, SAC, 네트워크(크기별 VRAM 표), 일반화 기법, 보상, 워커, 검증, 체크포인트, W&B |
| `configs/training/dataset.py` | dataset 모드: 맵 목록, `DATASET_AUGMENTATION` |
| `configs/training/ued.py` | ued 모드: `UED_METHOD`(dr/plr/accel), 점수, 생성기, 편집 |
| `configs/evaluation.py` | 스위트 정의, 베이스라인, 튜닝, 통계 |

모든 설정은 한 파일에만 정의되고, 시작할 때 병합·검증됩니다. 오타나 모순(예: `legacy_rv3`에 맵 스케일 입력)이 있으면 시작하지 않습니다. 파일을 고치지 않고 실험하려면:

```
python3 Start_training.py --preset presets/ued_dr.json --set LR=3e-4 --set NET_SIZE=l
```

`--preset`/`--set`은 환경변수(`RV4_PRESET`, `RV4_OVERRIDES`)로 워커·검증 프로세스에도 그대로 전달됩니다. 설정 fingerprint가 체크포인트와 다르면 재개를 거부하므로, 프리셋마다 `LOG_DIR`이 따로 잡혀 있습니다.

### 요청하신 선택 항목

| 항목 | 설정 |
| --- | --- |
| 학습 맵: UED / dataset | `TRAIN_MAP_SOURCE = "ued" \| "dataset" \| "mixed"` (+ `UED_METHOD`) |
| 맵 스케일 입력 | `MAP_SCALE_TO_ACTOR`, `MAP_SCALE_TO_CRITIC` (값: `MAP_SCALE_N × 한 변/MAP_SCALE_UNIT_M`, 50 m = n, 100 m = 2n; 가로·세로 2개) |
| 네트워크 크기 / 구조 | `NET_SIZE = s\|m\|l\|xl`, `NET_ENCODER = cnn\|impala\|legacy_rv3`. 크기별 파라미터·VRAM은 `configs/training/common.py` 표 |
| dataset 증강 | `DATASET_AUGMENTATION` (+ `DATASET_AUGMENTATION_TRANSFORMS`) |
| 기타 일반화 기법 | 4절 |

## 4. 일반화 기법 (config에서 선택)

| 기법 | 설정 | 근거 | 비용·주의 |
| --- | --- | --- | --- |
| Domain randomization | `TRAIN_MAP_SOURCE="ued"`, `UED_METHOD="dr"` | 학습 레벨 수가 많을수록 일반화 (Cobbe et al. 2019); RV3에서도 3 < 30 < 100 < 300맵 | 생성 약 1 s/레벨 (백그라운드) |
| Robust PLR | `UED_METHOD="plr"`, `UED_SCORE="maxmc"\|"pvl"`, `UED_REPLAY_ONLY_UPDATES` | Jiang et al. 2021a, 2021b | 점수는 크리틱으로 계산 |
| ACCEL | `UED_METHOD="accel"` (+ `UED_MUTATION_WEIGHTS` 등) | Parker-Holder et al. 2022 | 편집 결과는 연결성 검사 통과 필수 |
| Learnability 점수 | `UED_SCORE="learnability"`, `UED_SUCCESS_K` | SFL, Rutherford et al. 2024 | K는 `ued/evac_ratio_mean` 로그로 조정 |
| 베이스라인 regret 점수 | `UED_SCORE="shuttle_regret"` | PAIRED (Dennis et al. 2020)의 antagonist를 튜닝된 셔틀로 대체 | 새 레벨마다 셔틀 1에피소드 추가 |
| dataset + UED 혼합 | `TRAIN_MAP_SOURCE="mixed"` | | |
| D4 증강 (환경) | `DATASET_AUGMENTATION`, `UED_AUGMENTATION` | RV3 | |
| D4 증강 (리플레이) | `AUG_REPLAY_D4` | 보상이 대칭 불변이라 정확한 가상 transition. 관측·위치·스케일·행동을 함께 변환 (테스트로 검증) | |
| 무작위 평행이동 / DrQ | `AUG_RANDOM_SHIFT`, `AUG_DRQ_K/M` | RAD (Laskin et al. 2020), DrQ (Kostrikov et al. 2021) | K, M 배로 업데이트 비용 증가 |
| IMPALA-CNN, 폭 확장 | `NET_ENCODER="impala"`, `NET_SIZE` | Espeholt et al. 2018; Cobbe et al. 2020 | 업데이트가 cnn의 1.5-2.5배 |
| 위치 불변 특징 | `NET_SPATIAL="pool"`, `NET_COORD_CHANNELS` | | |
| 정규화 | `NET_NORM` | BatchNorm의 RL 문제 회피 | |
| 크리틱 LayerNorm / Dropout | `CRITIC_LAYERNORM`, `CRITIC_DROPOUT` | RLPD (Ball et al. 2023), DroQ (Hiraoka et al. 2022) | |
| 크리틱 앙상블 | `CRITIC_ENSEMBLE`, `CRITIC_TARGET_SUBSET` | REDQ (Chen et al. 2021) | 크리틱 수에 비례 |
| 인코더 공유 | `ENCODER_SHARING="critic"` | SAC-AE (Yarats et al. 2019), DrQ | 비용 감소 |
| 가소성 유지 | `RESET_EVERY_UPDATES`, `RESET_SCOPE`, `SHRINK_PERTURB_ALPHA` | Nikishin et al. 2022; Ash & Adams 2020 | 리셋 직후 성능 하락 |
| n-step 반환 | `N_STEP` | Rainbow 계열 | |
| 엔트로피 자동 조정 | `ALPHA_AUTO`, `ALPHA_TARGET_ENTROPY` | Haarnoja et al. 2018 | |
| 관측 확장 | `OBS_STATIC_DISTANCE_CHANNEL`(출구까지 측지 거리장), `OBS_FOLLOWER_PIXEL`, `OBS_EGO_MODE="metric"` | 맵 위상을 명시적으로 제공 | 관측이 바뀌어 RV3 체크포인트와 비호환 |
| 맵 크기 다양화 | `UED_MAP_SIZES_M` + 맵 스케일 입력 + `REWARD_SCALE_INVARIANT` | | 작은 맵 고난이도 생성은 느림 (`UED_GEN_TIMEOUT_S`) |

`presets/`에 하나씩 켜는 프리셋과 조합 후보(`candidate_full`, 검증 안 됨)가 있습니다.

## 5. 평가

### 테스트 스위트 (`data/suites/`, 한 번 만들고 고정)

| 스위트 | 내용 | 용도 |
| --- | --- | --- |
| `val` | 생성 맵 32개 | 학습 중 모델 선택 (`best_validation.pth`) |
| `tune` | 생성 맵 48개 × 2 | 베이스라인 속도 튜닝 |
| `test` | 맵 1500-1510 × 초기 배치 12 = 132 | **unknown 테스트 (헤드라인)** |
| `test_scale` | 60/140/200 m 생성 맵 각 20개 × 2 | 맵 크기 일반화 (참고용) |
| `human` | `test`의 11개 맵 × 배치 1 (같은 시나리오) | 사람 조작 실험 |

`test`의 1500-1504, 1507은 RV3의 기존 수작업 맵이고, 1505, 1506, 1508, 1509, 1510은 같은 스타일로 새로 만든 맵입니다(`cli/design_test_maps.py`, 그림 `docs/test_maps_1500_1510.png`). 기존 160 m 맵 1506은 `map_infos/archive/map_1506_160m.json`으로 옮겼습니다. 11개 모두 학습 소스(1000-1299, 105/108/128)와 D4 대칭 포함 픽셀 24% 이상, 새 맵은 다른 테스트 맵과도 31% 이상 다릅니다. 맵이 11개뿐이라 맵 단위 신뢰구간이 넓으니, 방법 간 비교는 짝지은 차이(`cli.compare`)로 보세요.

생성 맵 시드는 학습 생성기가 쓰지 않는 구간(`UED_RESERVED_SEED_RANGES`)에 있고, 스위트끼리도 겹치지 않습니다. 같은 시나리오는 모든 조건에서 같은 초기 배치(군중·로봇)로 시작하므로 조건 간 비교는 짝지은 비교입니다.

### 조건

| 조건 | 설명 |
| --- | --- |
| `no_robot` | 로봇 없음 (RV3 N) |
| `exit_robot` | 가장 가까운 출구로 가서 멈춤 (RV3 T, 그대로) |
| `shuttle_random` | 아직 대피하지 않은 사람 1명을 무작위로 골라 접근 → 2 m 안에 들면 가장 가까운 출구로 호송 → 4 s 대기 → 반복 |
| `shuttle_nearest` | 같은 방식, 로봇에서 측지 거리로 가장 가까운 사람 |
| `best_baseline` | 시나리오마다 두 셔틀 중 더 빠른 쪽 (사후 선택, 가장 강한 비교 대상) |
| `policy_det` / `policy_stoch` | 학습 정책의 평균 행동 / 샘플 행동 (RV3 실험은 샘플 행동) |
| `human` | `ADDS_AS_HumanPlay.py` 스위트 모드 결과 |

셔틀은 정책과 같은 결정 간격(2 s)과 행동 공간(축별 ±2 m/s)을 쓰고 navmesh 경로를 따라 움직입니다. 출구 5 m 안의 사람은 고르지 않습니다(위치만으로 판단; 없으면 로봇이 출구 모서리에서 바로 앞 사람을 막는 교착이 생김).

### 셔틀 튜닝 (`python3 -m cli.tune_baselines`)

`tune` 스위트에서 평균 100% 대피 시간을 최소화하는 (접근 속도, 호송 속도)를 successive halving으로 찾았습니다 (접근 1.0-2.83 × 호송 0.6-2.0, 35조합). 결과는 `data/baselines/tuned_shuttle.json`. 호송 속도 약 1.0 m/s가 일관되게 최적이었고, 접근 속도의 차이는 신뢰구간 안이었습니다.

### 지표와 통계

- 100% 대피 시간(초) = 모두 나올 때까지의 스텝 × 0.5 s, 미완료는 상한(100 m 기준 3000 스텝, 크기에 비례)
- 평균, 중앙값, IQM(Agarwal et al. 2021), 성공률, 맵 단위 클러스터 부트스트랩 95% 신뢰구간
- 짝지은 비교: 평균 차이와 신뢰구간, 더 빨랐던 시나리오 비율, 시간 비율의 기하평균
- 체크포인트가 학습한 맵(및 그 사본)은 자동으로 헤드라인에서 제외 (`--train-maps`로 RV3 파일에도 지정 가능)

```
python3 -m cli.evaluate --checkpoint ~/Log_SAC_FE_RV4_ued_accel/best_validation.pth --name accel-v1
python3 -m cli.evaluate --checkpoint ~/SOTA_MODELS/300maps.pth --name rv3-300maps --train-maps 1000-1299
python3 -m cli.evaluate --baselines-only
```

결과: `~/RV4_eval/<이름>/summary.md`, `summary.json`, 시나리오별 CSV, 그림, W&B eval run. 베이스라인 결과는 캐시되어 스위트당 한 번만 돌아갑니다.

### 사람 조작 실험

```
python3 ADDS_AS_HumanPlay.py --scenario 0 --participant P01     # human 스위트 0-9번
python3 ADDS_AS_HumanPlay.py --legacy                           # RV3 방식 (MAP_NUM_FOR_RUN)
```

**이동 능력의 공정성**: 행동은 축별 ±2 m/s라 RL과 셔틀은 대각선으로 2.83 m/s까지 나가지만, 사람 조이스틱은 크기 2.0 m/s에서 멈춥니다. 그래서 기본값을 `ROBOT_SPEED_NORM_LIMIT = 2.0`(configs/environment.py)으로 두어 모든 조작(RL, 셔틀, 출구 직행, 사람)의 명령 크기를 2.0 m/s로 제한합니다. 모든 조작이 `receive_action`을 거치므로 한 곳에서 적용됩니다. RV3와 같은 조건(제한 없음)은 `None`이고, 셔틀 튜닝 결과는 상한별로 따로 저장됩니다.

결과는 `~/RV4_eval/human/<참가자>.jsonl`에 쌓이고 `cli.evaluate --suites human ...`이 `human` 조건으로 짝지어 비교합니다. 기존 결과 폴더 형식으로도 함께 저장합니다. 기존 사람 결과(맵 105/108/128, 약 740 step)는 RV3 3maps 모델의 학습 맵이라 unknown 비교가 아닙니다.

## 6. 실행

```
python3 -m pytest tests -q                 # 회귀 테스트 42개 (약 2-3분)
python3 -m cli.build_suites                # 없을 때만 생성 (이미 생성·고정됨)
python3 -m cli.tune_baselines              # 시뮬레이터/베이스라인 코드를 바꿨을 때만
python3 Start_training.py --preset presets/ued_accel.json
python3 -m cli.evaluate --checkpoint <ckpt>
python3 -m cli.measure_networks            # 네트워크별 파라미터·VRAM 재측정
python3 -m cli.compare <평가폴더1> <평가폴더2> --labels A B   # 방법 간 짝지은 비교
python3 -m cli.view_suite test             # 스위트 맵 그림
```

학습 시드가 여러 개면 같은 `--labels` 이름을 주면 시나리오별로 평균한 뒤 비교합니다.

`Start_training.py`는 워치독입니다(정지/크래시 시 재시작, Ctrl+C 시 워커 정리·체크포인트·W&B 마감). 재개는 같은 `LOG_DIR`의 최신 `ckpt_ep_*.pth`에서, 설정 fingerprint가 같을 때만 합니다.

## 7. W&B

프로젝트 `WANDB_PROJECT`(기본 `adds-sac-fe-rv4`), 그룹 `EXPERIMENT_ID`. 축은 `global_episode`(episode/*, eval/*, key/*, ued/*)와 `global_update`(train/*).

| 키 | 내용 |
| --- | --- |
| `episode/evac_time_100`, `episode/total_reward`, `episode/reward/<항>` | 학습 에피소드 |
| `episode/evac_over_freeflow`, `episode/regret_vs_shuttle` | 레벨 기준 정규화 시간 |
| `train/q_loss`, `train/actor_loss`, `train/alpha`, `train/entropy`, `train/updates_per_s` | 학습기 |
| `ued/score_mean`, `ued/success_rate_mean`, `ued/evac_ratio_mean`, `ued/mean_generation`, `ued/top_levels` | 커리큘럼 |
| `key/val_evac_time_100`, `key/val_win_rate_vs_shuttle_nearest` | 검증 헤드라인 |
| `eval/val/video` | 검증 영상 (`VALIDATION_VIDEO=True`) |

설정 파일 원문과 병합된 값은 `config` 아티팩트로, 검증으로 고른 모델은 `model` 아티팩트로 올라갑니다. W&B 실패는 학습을 멈추지 않습니다(로컬 `events.jsonl`이 원본).

## 8. 권장 실험 순서

1. `rv3_parity` (dataset 300맵, RV3 네트워크): RV4 파이프라인이 RV3를 재현하는지
2. `dataset300` 대 `ued_dr`: 고정 300맵 대 무한 생성 맵
3. `ued_dr` 대 `ued_plr` 대 `ued_accel` 대 `ued_accel_shuttle`: 커리큘럼 효과
4. 가장 좋은 커리큘럼 위에서 단일 기법 하나씩: `d4aug`, `impala_m`, `obs_rich`, `critic_ln_dropout`, `nstep3`, `drq`, `redq`, `resets`
5. 효과가 있는 것끼리 조합 → `human` 스위트로 사람 실험

각 비교는 `test`에서, 학습 시드 3개 이상으로 보는 것을 권합니다(학습 시드 간 분산이 큼).

## 9. 현재 기준선

`test`(맵 1500-1510, 132 시나리오), 속도 상한 2.0 m/s, 가장자리 갇힘 수정 기준 (2026-10-25):

| 조건 | 평균 (s) | 95% CI (s) | 중앙값 (s) | 성공률 |
| --- | --- | --- | --- | --- |
| 로봇 없음 | 826.8 | 720.6-933.4 | 802.5 | 93.9% |
| 출구 직행 로봇 | 765.3 | 668.7-875.8 | 722.8 | 96.2% |
| 셔틀 (무작위 사람, 접근 1.5 / 호송 1.0 m/s) | **364.2** | 322.1-405.5 | 361.2 | 100% |
| 셔틀 (최근접 사람, 접근 2.0 / 호송 1.2 m/s) | 390.1 | 341.4-440.0 | 354.8 | 99.2% |
| 시나리오별 최선 셔틀 | 310.3 | 272.2-350.8 | 290.5 | 100% |
| RV3 3maps 정책 (평균 행동) | 1160.5 | 995.0-1311.7 | 1500 (상한) | 38.6% |

원자료와 맵별 표: `~/RV4_eval/reference-test-limit2-1025-1311/`, `docs/RV4_notes.md` 9.1절. 학습 정책이 넘어야 할 기준은 무작위 셔틀 364 s(그리고 사람 실험 결과)입니다.

처음 만든 스위트(생성 맵 150개, 번호 맵 92개, 속도 상한 없음)에서 잰 값은 참고용입니다. 튜닝된 무작위 셔틀이 평균 404-476 s로, 기존 사람 기록(알려진 맵 약 370 s)과 RV3 300maps 모델(1500-1503에서 348-417 s) 수준이었습니다. 학습 정책의 목표는 사람뿐 아니라 이 셔틀들까지 넘는 것으로 잡는 게 맞습니다.

## 10. 알려진 한계

- **군중 모델의 교착**: (1) 출구가 보이면 경로 계획 없이 출구의 가장 가까운 점으로 직진하고 이 신념이 이웃에게 전파되어, 오목한 공간에서는 벽에 막혀 멈출 수 있습니다(맵 130). (2) 무작위 탐색은 navmesh 삼각형 중심으로 직진하므로, 좁은 틈(4 m)이나 막다른 골목(폭 8 m)에서 모서리에 걸리거나 제자리를 맴돌 수 있습니다. 새 테스트 맵은 건물 간격 8 m 이상, 막다른 골목 없이 설계했고 시드 6개로 확인했습니다. 어떤 로봇 조건에서도 생길 수 있으므로 평균과 함께 성공률·중앙값을 보세요. 사람 실험과의 비교를 위해 군중 모델은 고치지 않았습니다.
- 생성기(`random_map.py`)의 장애물 크기가 미터 절댓값이라 작은 맵(<100 m)의 고난이도 생성이 느립니다.
- 셔틀 베이스라인은 위치 정보만 씁니다(`SHUTTLE_TARGET_FILTER="alive"`). 출구를 아는지 여부까지 쓰는 특권 버전은 `"uninformed"`.
- 실제 학습 결과(UED가 300맵보다 나은지)는 아직 없습니다. 파이프라인 검증용 짧은 실행만 했습니다.
