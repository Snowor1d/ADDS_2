# SAC_UED4

도심 **실외** 위험 구역에서 군중을 대피시키고 재진입을 막는 다중 로봇 강화학습 프로젝트입니다. 학습 분포는 ACCEL 계열 무감독 환경 설계로 생성합니다. 건물 출구 대피는 현재 과제가 아닙니다.

## 폴더

| 폴더 | 내용 |
| --- | --- |
| `sim/` | 시뮬레이터. 모델, 보행자, 공간, 위험 구역, 통행 구조, 로봇 행동 정의 |
| `configs/` | 설정 파일(`environment.py`, `simulation_run.py`, `training/` 세 파일)과 이를 합쳐 검증하는 `ResolvedConfig` |
| `learn/` | 학습기. 분산 비동기 SAC, 중앙집중 팀 크리틱, 리플레이, 롤아웃, 로깅, 평가 |
| `ued/` | 커리큘럼. 레벨 표현, 개체군, 변이, 홀드아웃 |
| `citygen/` | 도시 레이아웃 생성기. 형태별 가로망과 블록 |
| `osm_corpus/` | 실제 도심 크롭 추출 파이프라인 |
| `validation/` | 군중 모델 검증. 기본도, 병목, 창발 현상 |
| `viz/` | 뷰어, 렌더러, 사람 조작 |
| `cli/` | 명령줄 도구 |
| `tests/` | 시뮬레이터·학습·평가 회귀 테스트 |
| `docs/` | 설계와 검증 문서 |

설정은 다섯 파일로 나뉘고, 각 설정은 한 파일에만 정의됩니다.

| 파일 | 담당 |
| --- | --- |
| `configs/simulation_run.py` | 뷰어·수동 실행 한 번의 선택: 열어 볼 지도와 크기, 로봇 수, 체크포인트, 렌더링 속도 |
| `configs/environment.py` | 시뮬레이터 물리·측정과 관측 계약: 속도·반경, 안전 여유, 시야·통신, 관측 채널·해상도 |
| `configs/training/common.py` | 학습 공통: SAC, 탐험, 보상, 리플레이, 검증·최종 제로샷 지도와 시드, 로깅·W&B, 학습 지도 모드 `TRAIN_MAP_SOURCE` (`dataset` \| `ued`) |
| `configs/training/dataset.py` | `dataset` 모드: 학습할 OSM 도시(`DATASET_SITES`)·크기와, 에피소드마다 뽑는 과제 파라미터(위험구역 면적·형태, 인지도, 사전 경보, 로봇 수, 증강, 밀도) |
| `configs/training/ued.py` | `ued` 모드: 커리큘럼 파라미터와, 생성 레벨의 과제 파라미터(`UED_*`, 위와 같은 항목) |

진입점은 `configs.resolve_config()`로 세 파일을 한 번 합쳐 검증합니다. 모순이 있으면 시작하지 않습니다. `config.py`는 기존 `from config import *` 호출을 위한 재내보내기일 뿐이고, 설정을 정의하지 않습니다. `paths.py`는 데이터 위치를 정의합니다.

모든 지도는 도시 기반입니다. 학습은 기본값 `dataset` 모드에서 OSM 실제 크롭(`DATASET_SITES` × `DATASET_SIZES_M`, 균등)을 쓰고, 검증은 citygen 생성 레벨, 최종 제로샷은 명동 OSM 크롭을 씁니다. 옛 번호 지도 `map_infos/`는 삭제했습니다.

## 실행

프로젝트 루트에서 모듈로 실행합니다.

```
python3 Start_training.py                      # 학습 (워치독 포함)
python3 -m learn.ADDS_AS_reinforcement         # 학습 (워치독 없이)
python3 -m cli.final_zero_shot check           # 최종 제로샷 사전등록 검사 (정책 불필요)
python3 -m cli.final_zero_shot run --checkpoint <best_validation.pth>
python3 -m cli.measure_budget --final          # 크기별 비용·메모리 계측
python3 -m viz.run_sim                         # 시뮬레이션 보기
python3 -m viz.ADDS_AS_HumanPlay               # 사람이 직접 조작
python3 -m viz.ADDS_AS_view_ued difficulty     # 커리큘럼 레벨 그림
python3 -m cli.ADDS_AS_osm_pipeline status     # OSM 코퍼스 상태
python3 -m cli.ADDS_AS_validate_crowd all      # 군중 모델 검증
python3 -m pytest tests/ -q                    # 테스트
```

무엇을 보고 무엇을 플레이할지는 `configs/simulation_run.py`의 `SIM_SOURCE`(`real` | `curriculum`)와 `SIM_*`이 정하고, 뷰어와 사람 조작이 같은 정의를 씁니다. 학습 기록은 `~/<LOG_DIR>/events.jsonl`(원본), TXT, TensorBoard, W&B에 같은 값으로 남습니다.

## 문서

| 문서 | 내용 |
| --- | --- |
| `docs/crowd_awareness_design.md` | 군중의 위험 인지 모델 설계와 문헌 근거 |
| `docs/crowd_awareness_implementation.md` | 그 구현 |
| `docs/crowd_density.md` | 군중 규모를 밀도로 정의한 근거 |
| `docs/crowd_od.md` | 위험을 모르는 보행자의 통행 구조 |
| `docs/crowd_validation.md` | 표준 시험 대비 측정 결과와 보정 |
| `docs/robot_action.md` | 로봇의 이동과 신호 모드 |
| `docs/zero_shot_evaluation.md` | 실외 위험 구역 제로샷 지표와 대조군 |
| `docs/outdoor_human_calibration.md` | 실외 사람 실험에 따른 행동 보정과 인용 |
| `docs/outdoor_post_safety_mobility.md` | 위험 구역 이탈 후 이동·도시 유출입의 근거와 미보정 가정 |
| `docs/outdoor_madrl_redesign.md` | 실외 MADRL 재설계안(관측·보상·설정·로깅) |
| `docs/outdoor_madrl_implementation.md` | 그 구현과 각 단계의 통과 조건 확인 결과 |
| `docs/redesign_baseline.md` | 재설계 직전 기준선 |
| `docs/replay_memory_budget.md` | 크기별 비용·메모리 계측 |
| `docs/known_issues.md` | 미해결 항목 |
