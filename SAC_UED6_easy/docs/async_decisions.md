# 로봇별 비동기 결정 (2026-10-27)

`ROBOT_DECISION_SCOPE = "robot"`(기본): 각 로봇은 **자기** 결정이 끝날 때만 다시 결정하고, 팀원은 하던 일을 계속한다. `"team"`은 이전 방식(한 로봇의 결정이 끝나면 모두 다시 결정)이다.

## 왜 바꿨나

이전 구현은 팀 전체가 함께 결정했다. 어느 로봇이든 목표에 도착하거나 벽에 막히면 세 로봇 모두 새 행동을 받았다. 그 결과:

- 결정 하나가 평균 약 8–9 s밖에 이어지지 않았다. `ROBOT_DECISION_MAX_S_WAYPOINT`를 20 s에서 60 s로 올려도 에피소드당 결정 수가 그대로였다(fg5jloea: 54–60개, 대조군 61–65개).
- 반사실 probe에서 분기 후 첫 결정의 76%가 20 s 상한 전에 끊겼다(중앙값 8 s).
- 로봇 0이 목표를 60–120 s 유지하게 하자 결정 하나의 신호 대 잡음비가 0.31에서 0.60으로 올랐다(`review_20261022/critic_diagnosis.md` 7장).

다른 로봇의 도착 때문에 각 로봇의 결정이 계속 끊기는 것이 결정 하나의 효과를 작게 만든 원인 중 하나였다.

## 결정 규칙

관측 프레임(`OBS_FRAME_PERIOD_S`)마다, 로봇 i는 다음 중 하나일 때 결정한다.

- 아직 결정한 적이 없다
- 자기 결정이 `ROBOT_DECISION_MAX_S_<MODE>` 동안 이어졌다
- `ROBOT_DECISION_ON_EVENTS_<MODE>`가 켜져 있고, 자기 waypoint에 도착했거나 벽에 막혔다

`"team"`이면 한 로봇이라도 결정할 때 모두 결정한다. 정책은 모든 로봇의 행동을 계산하지만 결정하는 로봇의 행만 적용된다. 이름이 `deciding`인 매개변수를 가진 행동 함수(shuttle baseline)는 누가 결정하는지 전달받는다. 같은 규칙이 학습 rollout(`learn/rollout.py`), 검증·zero-shot 평가, 뷰어(`sim/model.py`의 `_policy_tick`)에 쓰인다.

## 데이터

`run_episode`는 프레임마다 한 번씩 내보낸다(`learn/rollout.Transition`).

| 필드 | 뜻 |
|---|---|
| `record` | 그 프레임의 관측 |
| `step_rewards`, `robot_penalties` | 그 프레임부터 다음 프레임까지 각 step의 학습 보상: 팀 몫과, 로봇별 자기 몫(충돌, approach shaping 등) |
| `action` | 누군가 결정한 프레임에서 모든 로봇의 현재 행동. 결정한 로봇은 새 행동, 결정하지 않은 로봇은 지금 위치에서 자기 waypoint까지 남은 offset으로 다시 표현한 진행 중 행동 |
| `decided` | 그 프레임에서 결정한 로봇 |
| `terminal` | 그 step들 안에서 과제가 끝났다 |
| `final` | 마지막 step 뒤의 상태. 모든 로봇의 결정이 여기서 끝난다 |

replay(`learn/replay.py`, 형식 2)는 프레임을 저장하고, 로봇 j가 결정한 프레임에서 j의 다음 결정 프레임(또는 final)으로 링크한다(`next_dec_r`, `hold_r`, `term_r`). 형식 1(팀 결정) replay는 읽지 않는다.

## 학습 (`learn/sac.py`)

샘플 하나는 **로봇 j의 결정 하나**다.

- **보상:** j의 결정부터 다음 결정까지의 step 보상(팀 몫 + PER_ROBOT_VERSIONS에서는 j의 자기 몫)을 학습기의 γ로 step마다 할인한 합. replay가 프레임을 따라가며 계산한다.
- **critic:** 입력은 결정 프레임의 모든 로봇 행동(j의 새 행동과 팀원의 진행 중 행동). j의 출력 Q_j만 j의 target에 맞춘다.
- **target:** y = R_j + γ_step^k (1 − terminal) V_j(s'). s'는 j의 다음 결정 프레임이다. 거기서 결정하는 로봇(j 포함)의 행동만 현재 정책으로 다시 뽑는다(모드는 정확한 기댓값, 연속 부분은 모드별 샘플). 나머지 로봇은 진행 중 행동을 유지한다. clipped double-Q의 최솟값은 기댓값 안에서 취하고, entropy 항은 j 자신의 것만 쓴다.
- **actor:** 샘플 프레임에서 결정한 로봇(`ACTOR_UPDATE_ROBOTS="all"`) 또는 샘플된 로봇 하나(`"one"`)만 학습한다. 각자 자기 Q로, 팀원은 그 프레임의 행동으로 고정한다. `ACTOR_TEAMMATE_ACTIONS="current"`는 그 프레임에서 함께 결정한 팀원만 다시 뽑는다.
- 이전의 팀 평균 target 경로(rew-v2 전용)는 없앴다. 팀 보상만 쓰는 보상 버전도 각 로봇의 Q가 팀 return을 학습한다.

## 업데이트 수

`UPDATES_PER_TRANSITION`은 "누군가 결정한 프레임(과 final 프레임)당 업데이트 수"다. 한 업데이트는 로봇 결정 `BATCH_SIZE`개를 학습한다. 이전에는 한 샘플이 팀 결정 하나(로봇 N개의 출력)였으므로, 같은 배치 크기에서 업데이트당 critic target 수는 1/N로 준다. 결정 프레임은 대개 로봇 한 대만 결정하므로, 로봇 결정 하나당 critic target 수는 이전과 비슷하다.

## 검증

- 테스트: `tests/test_madrl.py::AsyncDecisionTest`
  - 도착한 로봇만 다시 결정하고, 멀리 가는 팀원은 상한까지 자기 waypoint를 유지한다
  - team 범위에서는 함께 결정한다
  - 진행 중 행동이 현재 위치에서 같은 waypoint를 가리킨다
  - 뷰어가 로봇별로 결정한다
  - shuttle이 결정하는 로봇만 진행한다
- 테스트: `AlphaAutoTest::test_own_collision_is_charged_to_the_robot_that_hit`
  - 로봇별 링크, 구간 길이, 할인 return, 자기 몫 부과를 정확한 값으로 검증한다
- 테스트: `tests/test_hybrid_expectation.py::test_target_redraws_only_the_deciding_robots`
  - target이 결정하는 로봇만 다시 뽑고 자기 entropy만 쓰는지 손 계산과 대조한다
- W&B 지표: `episode/decision_seconds_mean`, `episode/decisions_per_robot`(검증에는 `decision_seconds_mean`)
