# 재설계 직전 기준선 (2026-09-23)

`docs/outdoor_madrl_redesign.md` 9절 1단계의 "현재 지표와 체크포인트 형식 기록"이다. 아래는 재설계를 적용하기 직전 코드의 상태다. 비교를 위해 그 코드의 사본을 작업 세션의 임시 스냅샷으로 보관했고, 저장소에는 넣지 않았다. SAC_UED4는 git에 추적되지 않는다.

## 테스트

`python3 -m pytest tests/ -q` 결과: 210개 중 209개 통과, 1개 실패(`CrowdFlowTest::test_outflow_removes_a_pedestrian_stuck_at_the_edge`, 기대 30 / 실제 29). 이 실패는 재설계 전부터 있었다.

## 관측 (옛 형식, 이름 없는 스키마)

- 한 장의 회색조 래스터(1 m/픽셀)에 벽 50, 군중 150, 위험구역 200, 로봇 255를 덮어 그린다.
- actor 입력은 세 가지다.
  - 로봇 주변 25×25 크롭의 최근 4 행동 시점 스택
  - 전체 지도를 50×50으로 **최댓값 풀링**한 4프레임 스택(모든 로봇이 공유)
  - 12차원 로봇 상태
- 12차원 상태에는 전역 정보 누출이 두 가지 있다.
  - `danger_occupancy()`: 위험구역 안 인원 비율의 실제값
  - `agents_near_robot_num_robot_index`: 시야 밖 사람까지 포함한 추종자 여부
- 군중은 시야·가시선과 무관하게 전부 래스터에 그려진다. 즉 actor가 전역 정답을 본다.

## 행동과 탐험

- 7차원 행동은 이동 2, 신호 방향 2, 모드 one-hot 3이다. 워커의 ε 탐험은 7차원 `random_action`을 썼다.
- `SACAgent.select_action`에는 `EXPLORATION_TYPE == 0`일 때 2차원 `[dx, dy]`만 반환하는 분기가 남아 있었다. 이 분기는 뷰어 경로에서 사용됐다.
- 저장 모델의 일반 실행(`FightingModel.step`, `using_model`)은 `self.robot`, 즉 0번 로봇에만 정책을 적용했다.

## 목적함수

- critic 손실: 로봇별 Q를 마스크 평균한 값을 목표와 비교한다.
- 목표: 로봇별로 `min(Q1', Q2')`를 먼저 취한 뒤 평균한다(min → mean).
- actor 손실: 행동을 바꾼 로봇의 **로봇별** `min(Q1, Q2)`를 쓴다. 팀 평균 Q가 아니다.
- 즉 세 곳의 팀 Q 정의가 서로 달랐다.

## 보상과 시간

- 가중합 성분은 `reward_based_alived`×2, `all_agents_danger`×0.003, `reward_penalty`×2, 충돌×6(0번 로봇만), `farthest_agent_distance`×1, 고정 -0.5다.
- 점유율 분모로 유입에 따라 증가하는 누적 `total_agents`를 썼다.
- 보상은 `ACTION_SCALE` 구간의 마지막 스텝 값 한 번만 계산했다. 충돌 항은 매 스텝 0으로 초기화돼 경계 스텝 외에는 사라졌다.
- 첫 행동 구간(`step <= ACTION_SCALE`)은 전송하지 않았다. 리플레이의 `delta_t`는 항상 1이었고 부트스트랩은 γ¹이었다.
- `MAX_STEPS` 도달(시간 제한)을 `done`으로 처리해 부트스트랩을 끊었다.

## 체크포인트와 리플레이

- 체크포인트 `sac_checkpoint_ep_{N}.pth`의 키: `q1, q2, q1_target, q2_target, policy, q1_opt, q2_opt, policy_opt`. 스키마 버전 정보는 없다.
- 리플레이 `replay_buffer.npz`(압축): 전이마다 `joint_ego (3,4,25,25)`, `global (4,50,50)`, 그리고 그 다음 상태 사본을 uint8로 저장한다. 전이당 35,000 B이고, `BUFFER_SIZE=1,000,000`이면 약 35 GB다.
- 커리큘럼 상태 `ued_curriculum.pkl` version 1: 스키마 정보가 없다.
- `LOG_DIR = "Log_SAC_UED3"`: SAC_UED3과 같은 폴더라, 같은 기계에서 SAC_UED3 학습과 로그·체크포인트를 공유할 수 있었다.

## 로깅

- 워커 통계를 메인 프로세스가 지표별 TXT 파일에 쓴다.
- 지표마다 감시 스레드를 두고, 이 스레드가 TXT를 다시 읽어 TensorBoard에 기록한다. 이때 축은 줄 번호다.
- W&B는 없었다.

## 지도

- 학습은 UED(citygen) 레벨과 `map_infos/` 번호 지도(`MAP_NUM_RANDOM` 1000–1299) 폴백을 섞어 썼다.
- 주기적 제로샷은 5,000 에피소드마다 OSM 4곳(시부야·타임스스퀘어·피갈·마라케시 100 m)과 옛 절차 홀드아웃(50/85/125/180 m)을 조회했다.
- 커리큘럼 레벨의 D4 증강은 건물만 변환하고 위험구역은 변환하지 않았다. 그래서 8개 방향 중 7개에서 위험구역이 검증된 위치와 다른 거리 위에 놓였다.
- `DataCollector`가 매 스텝 모든 에이전트의 행을 쌓았지만, 이를 읽는 코드는 없었다.
