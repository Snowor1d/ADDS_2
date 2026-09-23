# 실외 MADRL 재설계 구현 기록 (2026-09-23)

`docs/outdoor_madrl_redesign.md`의 구현 기록이다. 재설계 직전 상태는 `docs/redesign_baseline.md`, 계측 결과는 `docs/replay_memory_budget.md`에 있다.

## 파일

| 파일 | 역할 |
| --- | --- |
| `configs/environment.py`, `training.py`, `simulation_run.py` | 설정 3분할. 설정마다 한 파일만 소유한다 |
| `configs/__init__.py` | `resolve_config()`와 불변 `ResolvedConfig`, 시작 시 검증 |
| `config.py` | 기존 호출부를 위한 재내보내기. 설정을 정의하지 않는다 |
| `learn/metrics_logger.py` | 이벤트 하나를 JSONL(원본), TXT(호환), TensorBoard, W&B에 기록 |
| `sim/observation.py` | 정적 계층, 로봇 측정, 통신, 다중 해상도 입력의 유일한 생성기 |
| `sim/rewards.py` | rew-v2 보상 |
| `learn/rollout.py` | 에피소드 실행과 k스텝 보상 적산. 학습·평가·테스트가 공유한다 |
| `learn/networks.py` | 공유 actor와 중앙 critic |
| `learn/sac.py` | 팀 Q 목적함수, 7차원 탐험, 스키마 검사 체크포인트 |
| `learn/replay.py` | 기록 기반 리플레이, 정적 저장소, 배치 프리페치 |
| `learn/zero_shot.py` | 검증, 보조 OSM, 최종 제로샷의 평가 |
| `learn/ADDS_AS_reinforcement.py` | 학습 진입점(워커, 메인, 검증 프로세스) |
| `cli/final_zero_shot.py` | 최종 제로샷 사전등록 검사와 1회 실행 |
| `cli/measure_budget.py` | 크기별 비용·메모리 계측 |

## 단계별 통과 조건

`python3 -m pytest tests/ -q`: 257개 통과. 재설계 전부터 실패하던 1건도 지금은 통과한다.

1. **기준선·설정·로거** (`tests/test_configs_logging.py`)
   - 설정마다 소유 파일이 하나이고, import에는 부작용이 없다.
   - 다음 경우 시작을 거부한다.
     - 지원하지 않는 크롭 크기
     - 로봇 수 범위 위반
     - 최종 사이트가 보조 사이트에 포함된 경우와 공간 중첩
     - 검증·최종 시드가 학습 예약 범위 밖에 있는 경우
     - OSM 크롭의 `street` 위험구역
     - 키 형태의 값
     - 전체정보 actor를 별도 실험 없이 켠 경우
   - 같은 이벤트가 TXT·JSONL·TensorBoard·W&B에 같은 값과 축으로 들어간다.
   - W&B 실패 시에도 로컬 로그가 계속된다.
   - 카운터가 되돌아가면 새 run을 같은 그룹에 만든다.
   - 실제 W&B `disabled`/`offline`에서 동작한다.
2. **학습 목적함수** (`tests/test_madrl.py::TeamQTest`, `UpdateTest`)
   - 목표·critic·actor가 모두 팀 평균 Q를 쓰고, min은 팀 평균 뒤에 취한다.
   - critic은 로봇 순서에 동변이고, 패딩 슬롯의 영향을 받지 않는다.
   - 탐험은 7차원이다.
   - 저장 모델 실행 시 모든 로봇을 구동한다.
   - 1·2·3대 팀이 섞인 버퍼에서 갱신이 동작한다.
   - "기존 모델·새 코드 비교 실행"은 하지 않았다. 이 폴더에는 학습된 체크포인트가 없고, 새 스키마는 옛 체크포인트를 거부하기 때문이다. 옛 목적함수와의 차이는 `docs/redesign_baseline.md`에 적었다.
3. **보상·시간** (`RewardTimeTest`)
   - 첫 행동 구간과 짧게 끝난 마지막 구간을 기록하고, k는 실제 유지 스텝 수다.
   - 과제 종료만 terminal로 표시한다.
   - 유출은 재진입이 아니다.
   - 사람-시간 규모가 100 m와 200 m에서 같다.
4. **부분관측·통신·다중 해상도** (`tests/test_observation.py`)
   - 보이지 않는 사람을 옮기면 actor 입력은 그대로이고 critic 입력만 바뀐다.
   - 팀원의 관측이 다른 로봇의 공유 입력에 반영된다.
   - 겹친 관측을 중복 계수하지 않는다.
   - 건물 뒤의 사람은 측정되지 않는다.
   - 통신 지연을 반영하고, 오래된 관측은 만료된다.
5. **리플레이·성능** (`ReplayTest`, `docs/replay_memory_budget.md`)
   - 재구성한 입력이 워커가 행동한 입력과 비트 단위로 같다.
   - 창이 에피소드를 넘지 않고, 덮어쓴 이력은 샘플하지 않는다.
   - 정적 계층을 디스크에서 다시 불러올 수 있다.
   - 스키마가 다른 버퍼·체크포인트는 거부한다.
   - 시점당 6.4 kB다.
   - 100/200/400 m의 비용과 RAM·GPU 피크를 계측했다. 400 m 목표 밀도는 비용상 가능하다고 판단했다.
6. **평가** (`cli/final_zero_shot.py`)
   - 사전등록 검사를 실행했다. 명동 400 m, 인원을 1,776명으로 재산정했고, 위험구역 시드 6개가 모두 통과했다.
   - 최종 실행과 검증 기반 모델 선택은 학습된 모델이 있어야 하므로 아직 하지 않았다.

## 설계 문서와 달리 정한 것

- **재진입 보상에 여유 히스테리시스를 두었다.**
  - 보상과 `reentries_after_clear`는 안전 여유(2 m) 밖까지 나갔던 사람이 다시 들어온 경우만 센다.
  - 이유: 경계선에 선 사람이 사회력으로 밀려 안팎을 오가면, 100 m 생성 지도 120스텝 시험에서 146건이 나왔다. 로봇이 통제할 수 없는 값이다.
  - 경계 통과를 모두 세는 기존 `reentries`는 평가 지표로 그대로 둔다.
- **스텝별 보상을 저장하고, 할인은 학습기가 한다.** 문서의 "유지 길이 k 저장"에 더해 k개의 원시 보상을 저장한다. 그래서 γ 스케줄이 있어도 워커와 학습기의 할인이 어긋나지 않는다.
- **시간 제한은 terminal이 아니다.** `MAX_STEPS` 도달 시에도 부트스트랩한다. 과제가 에피소드를 끝낸 경우(`DANGER_TERMINATION`)만 terminal이다.
- **검증 규모를 줄였다.** 3 크기 × 난이도 3개(2·4·6) × 시드 1개 × 로봇 1·2·3대로 했다. 문서는 규모를 정하지 않았다. 400 m 에피소드가 약 40분이라 주기 평가 비용을 고려했다.
- **보조 OSM 4곳은 주기 평가에서 뺐다.** 모델 선택은 생성 지도 검증만으로 한다.

## 함께 고친 기존 결함

- 커리큘럼 레벨의 D4 증강이 건물만 회전·반사하고 위험구역은 그대로 두었다. `DangerZone.transformed`를 추가해 함께 변환하도록 고쳤다.
- `LOG_DIR`가 SAC_UED3와 같은 `Log_SAC_UED3`였다. `Log_SAC_UED4_madrl`로 바꿨다.
- `SIM_REAL_SIZE=150`은 코퍼스에 없는 크기였다. 100으로 바꿨다.
- 로봇 시야 반경을 가시성 아틀라스에 등록하지 않고 있었다. 로봇 측정은 영역 중심이 아니라 로봇 위치에서 광선을 쏘도록 했다.
- `model.step()`이 정책이 없어도 매 스텝 전체 지도를 래스터화했다. 이를 없앴다.
- 쓰이지 않는 `DataCollector` 누적을 기본값에서 껐다.
- 옛 절차 홀드아웃의 실제 생성 시드는 `seed + 7000×(size%97)`인데, 이를 예약한 것처럼 적은 설정을 바로잡았다.

## 학습 지도: 실제 OSM 크롭 (2026-09-23 추가)

- 학습 설정은 `configs/training/` 아래 세 파일로 나뉜다.
  - `common.py`: 공통 설정과 모드 스위치 `TRAIN_MAP_SOURCE`(`dataset` | `ued`)
  - `dataset.py`: OSM 크롭 모드의 지도 목록과 과제 파라미터(`DATASET_*`)
  - `ued.py`: 커리큘럼 모드의 설정과 과제 파라미터(`UED_*`)
- 과제 파라미터(위험구역 면적·형태, 인지도, 사전 경보, 로봇 수, 증강, 크기별 밀도)는 두 모드가 각자 가진다. 별도 스위치였던 `UED_ENABLED`는 `TRAIN_MAP_SOURCE == "ued"`로 대체했다.
- 검증 레벨은 citygen으로 만들므로 `UED_*` 과제 파라미터를 쓴다. 이 값이 바뀌면 검증 캐시를 다시 만든다.
- `TRAIN_MAP_SOURCE = "dataset"`이 기본값이다. 매 에피소드 `DATASET_SITES` × `DATASET_SIZES_M`에서 (사이트, 크기) 하나를 균등하게 뽑는다(`learn/training_maps.py`).
- 뽑은 크롭에는 매번 새로 다음을 정한다.
  - 위험구역: 면적 `DATASET_DANGER_AREA_RANGE`, 형태 `DATASET_DANGER_SHAPES`. 레벨 검증기를 통과할 때까지 다시 뽑는다.
  - 인원: 보행 가능 면적 × 목표 밀도. 저장된 400 m 크롭은 모두 800명으로 잘려 있어서 저장값을 쓰지 않는다.
  - 인지도, 사전 경보 비율, 로봇 수, D4 증강: `DATASET_*` 범위에서 뽑는다.
- 기본 목록은 20곳이다. 코퍼스 26곳에서 다음을 뺐다.
  - 최종 제로샷 명동
  - 400 m 크롭이 없는 surry_hills
  - 원래의 보조 평가 4곳
- 설정 검증은 다음을 거부한다.
  - 명동, 또는 명동 크롭과 지리적으로 겹치는 크롭
  - 보조 평가 사이트와 겹치는 사이트
  - 코퍼스에 없는 (사이트, 크기) 조합
  - OSM에서의 `street` 형태
- 크기를 균등하게 뽑는다. 400 m를 넣으면 에피소드의 1/3이 400 m가 되어, 에피소드당 비용이 `docs/replay_memory_budget.md`의 추정(생성 지도 0.1 비중 기준)보다 크다.
- 검증(모델 선택)은 여전히 citygen 생성 지도로 한다.

## 지도: 도시 기반으로 통일

- 학습, 검증, 뷰어, 사람 조작의 모든 지도가 도시 기반이다. citygen 생성 레벨과 OSM 크롭만 쓴다.
- 레벨 없이 시뮬레이터를 만들면 요청한 크기의 citygen 레벨을 생성한다.
- 번호 지도 `map_infos/`(JSON 428개)는 관련 코드와 함께 삭제했다.
  - 코드: `load_map_from_file`, 하드코딩 지도, 출구 생성기
  - 설정: `MAP_NUM`, `MAP_NUM_RANDOM`, `CROWD_NUMBER_*`, `MAP_DATA_AUGMENTATION`, `RANDOM_EXIT`, `SIM_SOURCE="numbered"`
- 삭제 전 사본은 작업 세션의 임시 스냅샷에만 있다.

## 남은 일

- 실제 학습을 돌리고, 검증으로 모델을 선택한 뒤 `cli.final_zero_shot run`을 한 번 실행해야 한다.
- W&B `online` 모드는 이 세션에서 실제 동기화까지 확인하지 않았다(`offline`/`disabled`만 확인). 이 기계의 `~/.netrc`에 W&B 항목은 있다.
- 통신 지연·손실(`COMM_DELAY_DECISIONS`, `COMM_DROP_PROB`)은 구현했지만 기본값은 0이다. 별도 실험축으로 남긴다.
- 기존 `from config import *` 호출부는 아직 재내보내기를 거친다. 새 코드(관측·보상·리플레이·로깅·평가)는 `ResolvedConfig`를 명시적으로 받는다. 워커는 설정 파일이 실행 중 바뀌면 시작을 거부한다.
