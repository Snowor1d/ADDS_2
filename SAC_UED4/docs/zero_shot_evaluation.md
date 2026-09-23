# 도심 실외 위험 구역 제로샷 평가

이 평가의 목표는 건물 출구 도달이 아니라 고정된 실외 위험 구역을
비우고 사람들의 재진입을 억제하는 것이다. 학습 루프의 과제 정의는
`config.py`의 `DANGER_TERMINATION`과 `sim/model.py`의
`agents_in_danger`, `is_cleared_and_held`에 있다.

## 평가 대상과 대조군

평가는 세 종류이고 목적이 다르다. 모두 `configs/training/common.py`에 선언한다.
시작할 때 `configs.resolve_config()`가 세 종류가 겹치지 않는지 검사한다.
검사 항목은 장소, 크롭의 공간 중첩, 생성 시드, 위험구역 시드다.

- **검증 (모델 선택 전용)**: 고정 시드의 citygen 생성 지도다.
  크기 100·200·400 m, 난이도 2·4·6, 로봇 1·2·3대다(`VALIDATION_*`).
  `VALIDATION_CYCLE_EPISODE`마다 고정한 정책 사본을 **별도 프로세스**에서
  평가한다. 결과는 메인 프로세스의 로거로 기록한다. 선택 점수는 같은
  시드의 대조군 대비 위험구역 사람-스텝 감소율의 평균이다. 최고 점수
  모델은 `best_validation.pth`로 저장한다. 검증 시드는
  `TRAIN_RESERVED_SEED_RANGES`에 있어 학습 레벨이 뽑을 수 없다.
- **보조 OSM 보고**: 시부야·타임스스퀘어·피갈·마라케시 메디나 100 m다.
  고정 난수로 위험구역을 한 번 배치한다. 보고용이며 모델 선택에 쓰지 않는다.
- **최종 제로샷**: 사전등록한 명동 400 m 지도 한 곳이다
  (`FINAL_ZERO_SHOT_*`, 2026-09-23 등록). 보조 사이트에 없고, 명동의
  다른 크기 크롭도 어디에도 쓰지 않는다. 순서는 다음과 같다.
  1. `python3 -m cli.final_zero_shot check`는 정책 없이 실행한다.
     사전등록한 위험구역 시드마다 세 가지를 검사하고, 통과·탈락과
     사유를 `preregistration_check.json`에 기록한다.
     - 위험구역 안 보행 가능 면적
     - 안의 모든 보행 가능 삼각형에서 안전지대까지 가는 경로
     - 유효 밀도
  2. `python3 -m cli.final_zero_shot run --checkpoint <선택된 모델>`는
     확정된 모델 하나로 한 번 실행한다. 체크포인트 SHA-256을 기록하고,
     다른 모델로 다시 실행하는 것은 `--register-new-model` 없이는 거부한다.
     이 옵션을 쓴 재실행도 기록에 남는다.
  - 저장된 명동 레벨은 800명, 0.0203명/m²다. 인원은 보행 가능 면적
    39,467 m²에 목표 밀도 0.045를 곱해 1,776명으로 다시 산정한다.
    `FINAL_ZERO_SHOT_DENSITY=None`이면 저장된 밀도로 실행하고
    `low_density`로 표시한다.
  - 위험구역 형태는 원과 사각형뿐이다. OSM 크롭의 `street` 위험구역은
    거리망을 따르지 않는 사각형으로 대체되므로 쓰지 않는다. 기록되는
    형태는 실제로 배치된 형태다.
- 각 시나리오에서 로봇 1·2·3대의 학습 정책과 **이동 명령 0·신호 꺼짐**
  대조군을 같은 지도·위험 배치·시드로 비교한다. 대조군의 로봇 몸체는
  남아 있어 물리적 장애물 효과가 포함되고, 벽 반발로 미세하게 움직일 수
  있다. 이것을 무로봇 또는 완전 정지 실험이라고 표기해서는 안 된다.
  대조군 결과는 정책과 무관하므로 검증에서는 한 번 계산해 캐시한다.
- 정책은 결정적 행동을 쓰며 이동과 신호를 모두 적용한다. 관측은 학습
  워커와 같은 생성기(`sim/observation.py`)로 만든다. 로봇 시야 10 m와
  가시선으로 측정하고 팀원 메시지로 공유한다. 실제 전역 군중은 actor에
  들어가지 않는다.
- 모든 평가 에피소드는 `MAX_STEPS`까지 진행하고, 학습의 종료 규칙은
  적용하지 않는다. 일찍 비워도 재진입 가능성을 끝까지 관찰한다.

## 기록 지표

`learn/zero_shot.py`가 각 실행을 기록한다. 검증은
`<LOG_DIR>/validation/validation_metrics.jsonl`, 보조 OSM은
`auxiliary_osm_metrics.jsonl`, 최종은
`<LOG_DIR>/final_zero_shot/final_metrics.jsonl`이다. 아래 모두 에피소드
당 값이다. 추론 시간, 명령 크기, 실제 속도, 모드 전환율, 실제 밀도도
함께 기록한다.

| 지표 | 의미 |
| --- | --- |
| `first_empty_step` | 최초 일시적으로 위험 구역 인원이 0이 된 스텝 |
| `held_clear_step` | 연속 60스텝 비움이 성립한 구간의 시작 스텝 |
| `held_clear_success` | 그 조건을 에피소드 안에 달성했는지 |
| `mean_occupancy` | 위험 구역 인원/누적 유입 인원의 시간평균(유입 시 희석 가능) |
| `mean_active_occupancy` | 위험 구역 인원/현재 살아 있는 인원의 시간평균 |
| `hazard_person_steps` | 위험 구역 안 인원 수를 스텝마다 더한 절대 노출량 |
| `final_occupancy` | 종료 시 위험 구역 인원/누적 유입 인원 |
| `reentries` | 구역 밖에 있던 살아 있는 보행자가 다시 진입한 횟수(경계 흔들림 포함) |
| `reentries_after_clear` | 안전 여유(2 m) 밖까지 나갔던 사람의 재진입 횟수. rew-v2 보상의 정의 |
| `inflows`, `outflows` | 크롭 경계로 들어온/나간 누적 보행자 수 |
| `evacuation_departures` | 위험 인지 후 지역 이탈 의도를 택해 입구로 나간 수 |
| `informed_trip_outflows` | 위험을 인지했지만 일상 통행을 이어가다 나간 수 |
| `background_departures`, `stuck_releases` | 무인지 통행 유출/경계 끼임 구제 유출 |
| `informed_departure_fraction` | 인지 후 유출 전체/행동 개시 경험자 수 |

비움에 실패한 실행의 `held_clear_step`은 null이다. TensorBoard의
`clear_step_censored` 평균에서만 `MAX_STEPS`로 대체한다.
시간과 함께 성공률을 항상 보고해야 한다. 시간만 평균하면 실패를
숨길 수 있다. `reentries`는 경계를 넘은 **사건 수**이며 같은
사람의 여러 재진입을 각각 센다. 크롭 밖으로 유출되어 죽은
보행자는 재진입으로 세지 않는다.

기본 실외 설정에서는 유입과 유출이 모두 켜져 있다. 자연 통행만으로
비움 성공률이 높아질 수 있으므로 **정책과 같은 시드의 신호 꺼짐
대조군의 차이**를 함께 보고해야 한다. `outflows` 자체를 로봇
성과로 해석하지 않는다. 유출 사유의 합계는 `outflows`와
일치해야 한다. 이동 의도의 가정과 사람 대상 근거는
`docs/outdoor_post_safety_mobility.md`에 기록했다.

## 근거와 해석 범위

- 2024년 체계적 문헌고찰은 대피 시간만이 아니라 밀도, 흐름,
  공간 분포 등의 다중 지표와 실측 자료 검증이 필요하다고 정리한다.
  Senanayake et al., “Agent-based simulation for pedestrian evacuation:
  A systematic literature review,” *International Journal of Disaster Risk
  Reduction* 111, 104705. https://doi.org/10.1016/j.ijdrr.2024.104705
- 이 연구 과제의 비움 유지 지표는 문헌에 제시된 표준 단일 지표가
  아니라 **실외 국소 위험 구역 방어**라는 문제 정의에서 나온
  프로젝트 지표다. 기존 출구 도달 시간과 성능 수치를 직접
  비교해서는 안 된다.
- OSM은 도로·건물 기하를 주지만 실제 대피 궤적이나 사람의
  로봇 신뢰를 검증하지 않는다. 실제 실외 행동 보정의 현 상태와
  인용은 `docs/outdoor_human_calibration.md`에 기록한다.

## 기존 결과와의 호환성

옛 `ZeroShot/`·`ZeroShotTask/` 값과 옛 절차 홀드아웃(50/85/125/180 m)
결과는 관측·보상 스키마가 다르므로 새 `eval/` 값과 섞지 않는다. 새 결과는
관측·보상·행동 스키마 버전과 함께 기록된다.
