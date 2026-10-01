# 군중 행동 모델: 현재 구현, 선행 연구와의 비교, 정당화 계획

이 문서는 세 가지를 한곳에 둡니다. 지금 코드에 구현된 군중 행동 모델이 무엇인지, 최신 군중 대피 연구가 같은 문제를 어떻게 다루는지와 비교하면 우리 모델이 어디에 서 있는지, 그리고 이 모델이 현실과 닮았다는 것을 어떻게 보일 것인지입니다.

이동 층(social force)의 보정과 측정은 [crowd_validation.md](crowd_validation.md), 군중 밀도의 근거는 [crowd_density.md](crowd_density.md)에 있습니다. 여기서는 그 위의 인지와 판단 층을 주로 다룹니다.

작성 시점: 2026-09-29. 코드 기준은 재진입 수정(`_on_reentry`, `_release_from_robot`)이 들어간 상태입니다.

---

## 1. 현재 구현

### 1.1 전체 구조

보행자는 매 스텝(0.5초) 두 가지를 따로 결정합니다.

- 인지 상태: 위험을 얼마나 알고 있는가 (`CrowdAgent.update_awareness`)
- 목표 지점: 어디로 갈 것인가 (`CrowdAgent.which_goal_agent_want`)

그다음 social force 모델로 실제로 움직입니다. 로봇은 인지 상태와 목표 지점 모두에 개입합니다. 코드는 [sim/agent.py](../sim/agent.py), 평소 이동의 출발지와 목적지는 [sim/od.py](../sim/od.py), 설정값은 [configs/environment.py](../configs/environment.py)에 있습니다.

### 1.2 인지 상태: 네 단계

`unaware` → `cued` → `milling` → `acting`

신호를 받는 세 경로(unaware일 때 매 스텝 검사):

| 경로 | 조건 | 설정 |
| --- | --- | --- |
| 로봇 | 신호를 켠 로봇이 10 m 안에 있고 보임 | `ROBOT_SIGNAL_RADIUS_M`, `ROBOT_SIGNAL_REQUIRES_SIGHT` |
| 직접 감지 | 구역 안: 스텝당 0.35 × 현저성, 구역이 10 m 안에 보임: 0.06 × 현저성 | `AWARENESS_P_INSIDE`, `AWARENESS_P_VISIBLE` |
| 주변 사람 | 행동 중인 이웃이 보이면 0.12 × min(1, 이웃 수 / 4) | `AWARENESS_P_SOCIAL`, `AWARENESS_SOCIAL_SATURATION` |

현저성은 위험의 감지 가능성(perceptibility)에서 계산하며, 0.25 미만이면 직접 감지 경로가 닫힙니다(`PERCEPTIBILITY_SENSORY_FLOOR`, 예: 무취 가스).

망설임(milling):

- 행동 전 지연을 로그정규분포에서 뽑습니다. 중앙값 16.7스텝(약 8초), σ 0.7(`PREMOVEMENT_MEDIAN_STEPS`, `PREMOVEMENT_SIGMA`). 야외 홍수 VR 실험(NHESS 2026)의 두 조건 평균 사이 값에 맞춘 잠정치입니다.
- 로봇 신호로 인지하면 지연이 0.25배가 됩니다(`MILLING_ROBOT_SPEEDUP`).
- 망설이는 동안에도 행동 중인 이웃이나 가까운 로봇이 카운트다운을 앞당깁니다(로봇이 있으면 스텝당 +4).

위험 기억:

- 보행자는 실제 구역 모양을 모르고, 직접 감지한 지점만 기억합니다. 최대 12개, 서로 4 m 이상 떨어진 점입니다(`HAZARD_MEMORY_MAX_POINTS`).
- 감지 가능성 0.7 이상이면 구역 중심도 함께 기억합니다(`PERCEPTIBILITY_GLOBAL_CUE`, 멀리서 보이는 연기).
- 도망칠 때는 기억한 지점에서 12 m 안에 있을 때만 밀려납니다(`HAZARD_MEMORY_RADIUS_M`).

### 1.3 목표 결정의 우선순위

1. 행동 타입 결정. 15~35스텝마다 다시 정하고, 신호 중인 로봇이 보이면 즉시 정합니다.
   - type 0 (로봇 추종): 로봇 지시를 수락. 수락 확률 = 기본 0.75 × 개인 성향 U(2/3, 4/3) × 상황 계수 × 로봇 형태 계수. 같은 로봇의 같은 모드 지시에는 한 번만 판단하고, 10스텝 이상 안 보이다 다시 만나면 새로 판단합니다.
   - type 2 (이웃 추종): 로봇이 없으면 70% 확률로 이웃을 따릅니다. 탈출 방향을 아는 이웃을 우선하고, 같으면 가까운 쪽을 고릅니다.
   - type 1 (자기 판단): 나머지.
2. type 0이면 로봇 지시를 따릅니다. 자기 판단보다 우선합니다. guide 모드는 로봇 위치로, direct 모드는 신호 방향 12 m 앞으로 향합니다(현재 설정에서 direct는 꺼져 있음). 따라가는 모습이 주변 사람에게 탈출 방향 신호(점수 2)가 됩니다.
3. 행동 중(acting)이면:
   - 구역을 벗어났으면(2 m 여유 밖, 한 번 벗어난 뒤에는 몸 반경만큼 안쪽까지 허용) 다음 행동을 하나 고릅니다. 떠남 50%(도로 입구로), 멈춤 20%(30초~2분 뒤 떠남이나 계속으로), 계속 이동 30%(기억한 위험 지점을 피하는 목적지로). `CROWD_POSTSAFE_INTENT_WEIGHTS`
   - 구역 안이고 기억이 있으면 기억 지점 반대 방향으로 도망칩니다. 막히면 navmesh에서 20 m 안의 가장 안전한 삼각형으로 우회하고, 더 안전한 곳이 없으면 현재 삼각형 중앙에 섭니다.
   - 행동 중인데 기억이 없으면(말로만 들은 경우) 아래 4, 5로 넘어갑니다.
4. type 1: 평소 이동. 70%는 맵 경계의 도로 입구로 가는 통과 이동(입구 폭에 비례, 25 m 이상 떨어진 곳), 30%는 맵 안쪽 볼일 이동입니다. 도착하면 5~30초 머뭅니다. `CROWD_THROUGH_TRIP_SHARE`, `CROWD_MIN_TRIP_M`, `CROWD_DWELL_STEPS`
5. type 2: 따라가는 이웃의 현재 위치를 향합니다.

### 1.4 재진입 처리 (2026-09-29 추가)

구역을 벗어난 사람이 다음 목적지를 고를 때 기억한 위험 지점과의 직선 거리만 검사하고, 실제로는 navmesh 경로로 걸어가 구역을 다시 지나는 문제가 있었습니다. 로봇 신호 없이 1,200스텝을 돌리면 200 m crop에서 군중의 약 15%가 경계를 수십 번씩 오가며 구역에 남았습니다(soho_nyc 한 사람당 경계 통과 중앙값 27.5회).

- `_on_reentry`: 벗어났다가 다시 들어오면 들어온 지점을 기억에 추가하고, 목적지를 버리고, 이웃 추종을 끊습니다. 그 에피소드 동안 이웃을 따라가지 않습니다.
- `_release_from_robot`: 로봇 추종이 풀리면(로봇이 사라짐, 신호 끊김, 지시 거절) 로봇을 만나기 전에 고른 목적지를 버립니다. 로봇이 구역 밖으로 데리고 나와도 풀리는 순간 옛 목적지로 구역을 다시 가로지르던 문제를 막습니다.
- 로봇을 따라 구역에 들어간 경우는 재진입으로 보지 않습니다. 로봇이 사람을 위험 쪽으로 이끄는 것은 정책의 책임이고 reward가 다룹니다.

효과(1,200스텝 후 위험구역 안 인원, 맵당 시드 1개):

| 조건 | 맵 | 수정 전 | 수정 후 |
| --- | --- | --- | --- |
| 로봇 신호 없음 | soho_nyc | 80 | 8 |
| | hongdae | 96 | 15 |
| | eixample | 16 | 4 |
| | kreuzberg | 47 | 7 |
| 로봇이 들어가 데리고 나온 뒤 신호 끔 | soho_nyc | 69 | 16 |
| | hongdae | 138 | 21 |

재진입 판정 기준을 넓히면(2 m 여유 밖까지 나가야 "벗어남") 경계에서 1~2 m씩 오가는 사람을 놓쳐 남는 인원이 약 4배로 늘었습니다. 그래서 `cleared`와 같은 좁은 기준을 씁니다. 대가로 진단용 카운터 `reentries`에는 경계 위에서 흔들리는 경우도 포함되지만, reward는 이 카운터를 쓰지 않고 [sim/rewards.py](../sim/rewards.py)에서 따로 셉니다.

### 1.5 이동

Helbing 계열 social force입니다. 목표 방향 구동력, 사람 간 지수 반발(앞쪽에 더 크게 반응, 뒤쪽 가중치 0.2), 몸이 겹칠 때의 접촉력(스프링, 감쇠, 마찰), 벽 여유 0.15 m의 약한 반발과 벽 접촉력을 더합니다. 벽을 향하면 벽을 따라 미끄러지는 방향으로 틀고, 필요하면 navmesh 경로로 우회합니다. 희망 속도는 N(1.5, 0.2) m/s이고 인지 상태와 무관합니다. 몸 반경은 0.25 m입니다. 로봇도 이웃으로서 반발과 접촉 계산에 들어갑니다.

### 1.6 인원 생성과 유입·유출

- 초기 배치: 보행 가능 면적에 고르게, 밀도 0.04~0.05명/m² ([crowd_density.md](crowd_density.md)).
- 처음부터 알고 있는 사람: 학습 설정에서 0~30%가 망설임 상태로 시작(`DATASET_PRIOR_INFORMED_FRACTION`).
- 유입: 100스텝마다 평균 4명이 도로 입구로 들어옵니다. 미리 경고받은 비율은 0.
- 유출: 도로 입구에 도착하면 맵 밖으로 나갑니다. 경계 3 m 띠 안에서 20초 동안 0.5 m도 못 움직이면 끼인 것으로 보고 제거합니다(`stuck_release`).

### 1.7 근거가 약한 값

| 값 | 설정 | 근거 |
| --- | --- | --- |
| 이웃 추종 확률 0.7 | `which_goal_agent_want` 안의 상수 | 없음 |
| 로봇 기본 수락률 0.75 | `ROBOT_GUIDE_BASE_COMPLIANCE` | 없음 (config 주석에도 보정 근거 없음) |
| 벗어난 뒤 행동 비율 50/20/30 | `CROWD_POSTSAFE_INTENT_WEIGHTS` | 없음 |
| 통과 이동 비율 0.7 | `CROWD_THROUGH_TRIP_SHARE` | 없음 |
| 도로 입구 가중치 | 입구 폭 | 보행량 조사 대신 폭을 대리 지표로 사용 |
| 망설임 지연 | 로그정규, 중앙값 8초 | 야외 홍수 VR 실험, 잠정치 |

---

## 2. 선행 연구와의 비교

### 2.1 분석 틀

- 3단계 구분(Hoogendoorn & Bovy, 2004): 전략(무엇을 하고 어디로 갈지), 전술(어떤 경로로), 운영(한 걸음씩 어떻게 움직일지).
- 심리 층 구조(Vadere 그룹, Köster 등): 자극을 지각하고, 인지 층에서 해석하고, 행동 층에서 실행합니다. 우리 모델의 `update_awareness` → `which_goal_agent_want` → social force가 정확히 이 형태입니다.
- 어느 층이 가장 중요한가(Haghani & Sarvi, 2021): 대피 시간 추정은 이동 층, 특히 병목 유출률 파라미터에 압도적으로 민감합니다. "If a model does not produce bottleneck flowrates accurately, efforts to refine other aspects of simulation might be in vain."

### 2.2 운영 층: 이동 모델

| 계열 | 대표 | 특징 |
| --- | --- | --- |
| 힘 기반 | Social force (Helbing), 일반화 원심력 모델 (Chraibi 등) | 가장 널리 쓰임. 파라미터 보정이 핵심 |
| 속도 기반 | Collision-free speed model (Tordeux 등, JuPedSim) | 충돌이 원천적으로 없고 파라미터가 적음 |
| 걸음 기반 | Optimal Steps Model (Seitz & Köster, Vadere) | 원형 영역 안에서 다음 발 위치를 최적화 |
| 인지 휴리스틱 | Moussaïd, Helbing & Theraulaz (2011) | 시선 방향별 장애물 거리로 속도·방향을 정하는 두 규칙 |
| 속도 장애물 | RVO / ORCA | 로봇·그래픽스에서 주로 사용 |
| 데이터 기반 | 궤적 예측 신경망 | 학습한 장면 밖에서 일반화가 약함 |

우리 모델은 social force를 기본도와 병목에 맞춰 보정했습니다. 다만 3.2절처럼 병목 유량은 아직 문헌에 미달합니다. 위험구역은 이동에 물리적 영향(속도 저하 등)을 주지 않습니다. 홍수 연구(Shirvani, Kesserwani & Richmond, 2020)는 물 깊이에 따른 속도와 넘어짐을 모델링합니다.

### 2.3 행동 전 단계: 인지와 결정

- PADM(Lindell & Perry, 2012): 환경 신호, 사회적 신호, 경고를 받아 위협 인식, 대응 방법 인식, 정보원 신뢰도를 판단한 뒤 행동을 결정하는 다단계 모델. 여러 대피 ABM이 이를 구현합니다.
- Kuligowski(NIST, SFPE 핸드북): 신호 지각 → 해석 → 결정 → 행동의 과정. 모델들이 데이터 없이 비현실적인 가정을 한다고 비판합니다.
- 행동 전 시간 데이터베이스(Lovreglio, Kuligowski, Gwynne & Boyce, 2019): 화재 9건과 훈련 103건, 16개국 13,591명.

우리 모델은 PADM을 단순화한 네 단계입니다. 차이는 신호를 받으면 결국 모두 행동한다는 점(현실에서는 위협을 낮게 보거나 정보를 더 찾다가 행동하지 않는 사람이 상당함)과 위협 해석·정보원 신뢰도의 개인차가 없다는 점입니다.

### 2.4 정보 전파와 사회적 영향

- 체계적 리뷰(Templeton, Xie, Gwynne, Hunt, Thompson & Köster, 2023): 70편 분석, 사회적 상호작용 8종 분류, 의사소통 모델 17편은 공간 반경·소셜 네트워크·외부 통신으로 정보를 전달. 결론은 "assumed reasons for interactions and portrayal of them may be overly simple".
- 정보 기반 대피 모델(Zhao 등, 2026, 실내 가스 누출): 모름 / 직접 인지 / 전달받음 상태, 정보 수용 = 인지 능력 × 신뢰도 × 전달 감쇠. 저·중밀도에서는 정보 전파 범위가 대피를 크게 앞당기고, 고밀도에서는 혼잡이 지배해 효과가 줄어듭니다. 우리 모델과 구조가 가장 가깝습니다.
- 군집 추종과 패닉 신화 논쟁: Helbing, Farkas & Vicsek(2000)은 개인 판단과 군집 추종의 혼합이 최적이라고 봤고, 사회심리학(Drury 등)은 집단 패닉이 근거 없는 신화이며 실제로는 공동의 정체성과 돕기 행동이 흔하다고 비판합니다. 이를 반영한 모델로 von Sivers 등(2016)이 있습니다.
- 감정 전염: Durupinar 등(2016), Tsai의 ESCAPES(2011). 주로 그래픽스 쪽이고 검증이 약합니다.

우리 모델은 들은 사람이 위험의 존재만 알고 위치는 모른다는 점에서 "직접 인지 vs 전달받음" 구분과 같습니다. 약점은 근거 없는 70% 이웃 추종(재진입 순환의 3분의 2가 여기서 나왔음), 일행과 돕기 행동의 부재, 정보원 신뢰도와 거리 감쇠의 부재입니다.

### 2.5 전술 층: 경로 선택과 공간 지식

- 인지 지도(Andresen, Haensel, Chraibi & Seyfried, 2016): 각자 부정확하고 불완전한 공간 지식으로 경로를 정하고 탐색하며 지식을 쌓습니다. 대피 모델이 "전체 구조를 안다"고 가정하는 것을 비판합니다.
- 이산 선택 모델(Haghani & Sarvi, 2017 등): 거리, 혼잡, 연기, 조명이 출구 선택에 미치는 영향을 실험과 설문으로 추정하고 제한된 합리성을 전제합니다.

우리 모델은 위험을 부분 지식으로 다루지만 도시 지도는 모두가 완전히 압니다(navmesh 최단 경로표). 이 비대칭이 가장 큰 논리적 틈입니다. 또 목적지를 고를 때 위험을 경로 비용이 아니라 직선 거리로만 검사하며, 이것이 재진입 문제의 근본 원인입니다.

### 2.6 전략 층: 도시 규모와 안전 이후 행동

도시 규모 대피 ABM은 쓰나미(와이키키, 이키케), 홍수, 지진 연구가 많고, 대부분 "모두가 대피소로 간다"는 목적지 고정 구조입니다. 우리 모델의 열린 경계, 평소 통과·볼일 이동, 벗어난 뒤 행동 선택은 문헌에서 드문 설정이고, 로봇의 구역 방어 과제가 성립하는 이유이기도 합니다. 대신 검증 데이터가 거의 없습니다.

### 2.7 유도, 수락, 로봇

- 리더-추종 모델: 유도자의 영향 범위 안에서 확률적으로 따라가며, 유도자 수에는 최적값이 있습니다.
- Mayr & Köster(2022, Vadere): 수락을 모두에게 같은 단일 확률로 두고 한계로 인정. 혼잡이 적은 경로를 추천하는 전략에서는 약 20% 수락률로도 혼잡을 막았습니다.
- 로봇 과신(Robinette 등, 2016): 연기와 경보 속에서 26명 전원이 로봇을 따랐습니다. 절반은 몇 분 전 로봇의 안내 실패를 봤습니다.
- 로봇 대 군중 충돌(Nayyar & Wagner): 로봇 지시와 반대로 군중이 뛰어가면 군중을 따르는 경우가 많았습니다. 설명을 덧붙이면(내용이 없어도) 따르는 비율이 올랐습니다. 실제 실험 14명으로 로봇을 따르는 대피자의 궤적 모델도 학습했습니다(평균 오차 9.9 cm, 다른 환경에서는 크게 증가).
- 로봇이 흐름을 바꾸는 방식: Zheng 등(2023/24)은 social force에 로봇 힘을 넣은 미시 모델과 밀도 방정식의 거시 모델을 함께 씁니다. Wan 등(2020)은 병목 앞에서 흐름을 조절하는 로봇을 심층 강화학습으로 학습했습니다. Chen·Jiang·Guo의 실험에서는 로봇이 있으면 보행 속도가 느려지고 로봇이 빠를수록 더 느려졌습니다.

우리 모델은 개인차, 지시 단위 판단, 시야 조건, 추종자를 통한 간접 전파까지 있어 Mayr & Köster보다 정교합니다. 빠진 것은 군중과 로봇이 충돌하는 상황(주변 군중이 반대로 가도 수락 확률이 변하지 않음)과 신뢰의 변화(로봇이 실수해도 이후 수락률이 그대로)입니다. 기본 수락률 0.75는 Robinette(거의 100%)와 Nayyar(군중과 충돌하면 낮음) 사이의 값일 뿐 근거가 없습니다.

### 2.8 새로운 흐름

- 강화학습 기반 대피: 대부분 보행자 에이전트나 경로 계획을 학습합니다(리더·추종자 두 층, 계층형 강화학습 등). 로봇이 군중을 유도하는 다중 에이전트 강화학습은 드물어 이 프로젝트의 차별점입니다.
- LLM 에이전트: LLM을 개별 에이전트의 의사결정기로 쓰는 대피 ABM(2025), 대화를 이동 결정에 연결한 연구(Liu 등, 2025). 공통 한계는 계산 비용과 정량 검증 부족이며, 수백만 스텝이 필요한 강화학습 환경에는 아직 맞지 않습니다.

### 2.9 종합 비교

| 항목 | 최신 연구의 주류 | 우리 모델 | 평가 |
| --- | --- | --- | --- |
| 층 구조 | 지각 → 인지 → 행동 | 같은 구조 | 일치 |
| 이동 모델 | SF, 속도 기반, 걸음 기반 + RiMEA/IMO 검증 | SF + 기본도 보정 | 기본도는 통과, 병목 미달 (3.2절) |
| 행동 전 단계 | PADM, 로그정규 행동 전 시간 | 4단계, 로그정규 | 신호 받으면 모두 행동. 해석·신뢰 차이 없음 |
| 위험 지식 | 부분 지식, 직접 인지 vs 전달받음 | 직접 본 지점 기억 | 일치 |
| 공간 지식 | 인지 지도, 부분 지식 | 도시 지도는 완전히 앎 | 비대칭 |
| 경로 판단 | 위험을 경로 비용에 반영 | 직선 거리 검사 | 재진입의 근본 원인 |
| 정보 전파 | 채널 구분, 신뢰도·감쇠 | 이웃 수 포화 확률 | 단순하지만 방향은 맞음 |
| 사회적 영향 | 정체성, 일행, 돕기 | 70% 이웃 추종 | 군집 추종 가정이 강함 |
| 로봇 수락 | 대부분 단일 확률 | 개인차 + 지시 단위 판단 | 문헌보다 정교 |
| 로봇 vs 군중 충돌 | 군중을 따르는 경향 관찰 | 없음 | 핵심 현상 누락 |
| 신뢰 변화 | 과신, 실수 후 변화 | 없음 | 누락 |
| 도시 규모 흐름 | 목적지 고정 대피 | 열린 경계, 평소 이동, 벗어난 뒤 행동 | 드문 강점, 검증 데이터 없음 |
| 위험의 물리 효과 | 홍수 속도 저하 등 | 없음 | 위험 종류에 따라 필요 |

### 2.10 개선 후보 (우선순위 순)

1. 위험을 반영한 경로 선택: 목적지와 출구를 고를 때 navmesh 경로가 기억한 위험 지점을 지나면 비용을 매깁니다. 재진입 문제의 근본 해결입니다.
2. 군중과 충돌할 때 수락률 감소: 주변 다수가 로봇 지시와 반대로 가면 수락 확률을 낮춥니다(Nayyar & Wagner).
3. 근거 없는 파라미터의 민감도 분석(4장).
4. 이웃 추종을 약하게 하거나 일행 단위 이동으로 바꾸기.
5. 공간 지식 제한(지역 주민 vs 방문자 비율)은 실험 설계 차원의 선택 사항.
6. 복잡도 관리: 상위 행동 층을 더 늘리기보다 지금 층의 근거와 민감도를 먼저 확보합니다.

---

## 3. 검증: 문헌의 방법과 우리 현황

### 3.1 문헌이 쓰는 방법

1. 검증(verification)과 타당성 확인(validation)의 구분: 모델이 설계대로 동작하는가 vs 실제 사람과 닮았는가. NIST TN 1822(Ronchi, Kuligowski 등, 2013)가 구성 요소 테스트, 창발 현상 테스트, 불확실성 분석을 묶은 절차를 제안했고, ISO 20414가 이런 절차를 표준화했습니다.
2. 표준 테스트 케이스: RiMEA 14개 시나리오, IMO MSC.1/Circ.1533 12개 테스트(직선 복도 속도, 모퉁이, 행동 전 지연 분포, 출구 선택, 병목 등).
3. 이동 층의 정량 비교: 기본도(Weidmann 1993, Seyfried 등 2005), 병목 유량, 차선 형성과 faster-is-slower 같은 창발 현상. 율리히 연구소의 공개 궤적 데이터 아카이브가 공용 기준이며, Wolinski 등(2014)은 실측 궤적에 맞게 파라미터를 자동 추정한 뒤 모델을 같은 지표로 비교하는 틀을 제안했습니다.
4. 행동 층: 행동 전 시간 분포(Lovreglio 등 2019), VR 실험(Kinateder & Warren 2016은 대피 시작의 긍정적 사회 영향이 실제와 비슷하게 나왔지만 경보 반응과 부정적 영향은 VR에서 약했다고 보고), 실제 사고 재구성(뒤스부르크 러브 퍼레이드: Pretorius 등 2015, 2020년 J. R. Soc. Interface 연구, 하지 순례: Helbing 등 2007), 생존자 인터뷰(von Sivers 등 2016이 런던 테러 생존자 연구를 근거로 규칙 설정).
5. 방법론 원칙: 패턴 중심 모델링(Grimm 등, 2005, 여러 규모의 여러 패턴을 동시에 재현하는지로 구조와 파라미터를 거름), 불확실성과 민감도 분석, 이동 층부터 보정(Haghani & Sarvi, 2021).
6. 로봇 유도 연구: 제가 본 문헌에서는 대부분 기존 social force 계열을 그대로 쓰고 군중 모델을 따로 검증하지 않았으며(Wan 등 2020, Zheng 등 2023/24), 사람의 반응은 소규모 실험으로 보완했습니다(Robinette 등 26명, Nayyar 등 14명).

### 3.2 우리 현황

`validation/` 패키지가 자유 보행 속도, 기본도, 병목, 차선 형성, faster-is-slower를 잽니다. 아래는 [crowd_validation.md](crowd_validation.md)와 `validation/results/`의 2026-09-17 측정값입니다.

| 항목 | 우리 값 | 문헌 | 판정 |
| --- | --- | --- | --- |
| 자유 보행 속도 | 1.49 m/s | 1.34 m/s | 통과 |
| 1명/m²에서 속도 | 1.09 m/s | 1.15 m/s | 통과 |
| 2명/m²에서 속도 | 0.68 m/s | 0.66 m/s | 통과 |
| 병목 비유량, 문 1.2 m | 0.18명/m/s | 약 1.2 | 실패 (약 15%) |
| 병목 비유량, 문 3 m | 0.67명/m/s | 약 1.2 | 실패 (약 55%) |
| 역류 차선 형성 | 없음 | 있어야 함 | 실패 |
| faster-is-slower | 보류 | | 여러 시드 필요 |

- Haghani & Sarvi가 가장 중요하다고 한 병목 유량이 문헌의 15~55%입니다. crowd_validation.md는 원인 후보로 시간 간격 0.5초(문 통과량을 약 3분의 1 깎음)와 속도 기반 충돌 회피 항의 부재(차선 형성)를 듭니다.
- 이 측정은 이후의 코드 변경(벽 처리, 재진입 수정 등) 이전이므로 다시 재야 합니다.
- 행동 층은 망설임 지연만 VR 실험 값에 기반하고, 나머지는 근거 데이터가 없으며 검증도 아직 없습니다.

---

## 4. 정당화 계획

### 4.1 우선순위

1. 병목 유량을 해결하고 다시 측정합니다. 이게 맞지 않으면 다른 층의 검증은 의미가 약합니다.
2. 율리히 공개 궤적 데이터로 복도·병목 조건을 재현해 기본도, 유량, 궤적 오차를 같은 지표로 비교합니다.
3. 행동 층은 패턴 중심 검증과 민감도 분석으로 대응합니다(4.2~4.5).
4. 여력이 있으면 공개 영상이 있는 야외 대피 사례 재구성이나 소규모 VR 실험(로봇 수락률, 군중 충돌 효과)을 합니다.

### 4.2 패턴 중심 검증의 논리

행동 층은 개인의 반응 시각을 맞출 데이터가 없습니다. 그래서 현실에서 반복적으로 관찰된 모양을 재현하는지 봅니다. 여러 패턴을 동시에 통과할수록 우연이 아니라 구조가 맞을 가능성이 커집니다.

- 주장할 수 있는 것: 모델이 알려진 패턴과 모순되지 않고, 근거 없는 파라미터를 흔들어도 결론(로봇 정책 > 기준선)이 유지된다.
- 주장할 수 없는 것: 모델이 현실을 정확히 예측한다.

이 과제의 주장은 "군중을 정확히 예측한다"가 아니라 "로봇이 대피를 개선한다"이므로, 행동 층은 절대 정확도보다 결론의 견고성을 보이는 편이 설득력이 큽니다. 각 패턴에는 민감도 분석이 짝으로 붙습니다.

### 4.3 패턴 1: 행동 전 시간 분포

- 측정: `cued_at`부터 `awareness == "acting"`이 된 스텝까지(반응 단계). 위험 발생부터 신호를 받기까지(인지 단계)도 따로 잽니다. 설정에서 뽑는 분포가 아니라 시뮬레이션에서 실현된 분포를 재야 합니다. 이웃과 로봇이 카운트다운을 앞당기기 때문입니다.
- 비교 대상: Lovreglio 등(2019) 데이터베이스(주로 건물 화재·훈련, 경보가 있는 조건), config에 인용된 야외 홍수 VR 실험(평균 7.5초와 13.8초).
- 판정: 절대값보다 모양과 경향. 오른쪽 꼬리가 긴 분포인가, 행동하는 이웃이 많을수록 짧아지는가(Kinateder & Warren 2016), 로봇이 있을 때 짧아지는가, 중앙값이 두 참고 데이터가 만드는 범위 안에 드는가.
- 주의: 건물 데이터를 그대로 목표로 쓰면 안 됩니다. 두 데이터 사이를 민감도 구간으로 씁니다.
- 짝 민감도: 망설임 중앙값 4 / 8 / 16 / 32초.

### 4.4 패턴 2: 로봇 없이 구역이 비워지는 곡선

- 측정: 로봇 신호를 끈 조건에서 스텝마다 구역 안 인원과 벗어난 사람의 누적 비율. 요약 지표는 `t50`, `t90`, 꼬리 비율 `t90 / t50`, 평탄 구간 여부.
- 비교 대상(정성적 패턴): S자 누적 곡선(지연 → 빠른 감소 → 긴 꼬리), 감지 가능성이 높을수록 빠름, 미리 알던 비율이 높을수록 초반 기울기가 큼, 고밀도에서 정보 전파 이득이 포화(Zhao 등 2026).
- 판정: 조건을 바꿔 가며 경향의 방향이 문헌과 같은지(예: 감지 가능성 0.3 / 0.5 / 0.9에서 `t50` 단조 감소). 곡선이 줄다가 평탄해지면 결함을 의심합니다.
- 이미 효과를 본 사례: 재진입 버그가 이 검사로 드러났습니다. 로봇 없는 곡선이 약 15%에서 평탄해진 것은 현실에서 설명할 수 없는 모양이었습니다.
- 주의: 열린 경계라 사람이 계속 드나듭니다. 구역 안 인원과 벗어난 누적 인원을 따로 보고, 시드 여러 개로 평균과 분산을 봅니다.
- 짝 민감도: 감지 가능성, 미리 알던 비율, 이웃 추종 확률.

### 4.5 패턴 3: 로봇 유도를 따르는 비율

- 측정: 로봇 신호를 본 사람 중 type 0이 된 비율. 조건별로 나눕니다. 1:1 마주침 vs 주변 군중이 있는 경우, 주변 군중이 로봇 지시와 같은 방향 vs 반대 방향, 이미 도망 중이던 사람 vs 모르던 사람.
- 비교 대상: Robinette 등(2016, 1:1 비상 상황에서 26명 전원, 수락률 상한의 근거), Nayyar & Wagner(군중이 반대로 가면 군중을 따르는 경향, 설명이 수락을 높임), Mayr & Köster(2022, 약 20% 수락률로도 효과, 하한 쪽 근거).
- 판정: 1:1 조건의 실현 수락률이 높은 쪽에 있는가(현재 평균 0.75, 범위 0.5~1.0이라 대체로 맞음). 반대 방향 군중이 있을 때 수락률이 떨어지는가(현재 모델에는 이 조건이 없어 실패가 예상되며, 2.10절 개선 후보 2가 됨).
- 주의: 이 실험들은 모두 실내, 소수, 로봇 1대 조건입니다. 도심 야외 다수 로봇 상황에는 범위와 방향성의 근거로만 씁니다. "따라간 비율"의 정의를 문헌과 맞춥니다.
- 짝 민감도: 기본 수락률 0.2 / 0.5 / 0.75 / 1.0.

### 4.6 요약

| 패턴 | 참고 근거 | 예상 결과 | 짝 민감도 |
| --- | --- | --- | --- |
| 행동 전 시간 분포 | Lovreglio 2019, 야외 홍수 VR | 모양·경향은 맞을 가능성이 높고 범위는 확인 필요 | 망설임 중앙값 4~32초 |
| 로봇 없는 대피 곡선 | S자 곡선, 감지 가능성·밀도 효과 | 재진입 수정 후 재확인 필요 | 감지 가능성, 미리 알던 비율 |
| 로봇 추종 비율 | Robinette, Nayyar, Mayr & Köster | 1:1은 맞고 군중 충돌은 실패 예상 | 수락률 0.2~1.0 |

실패한 패턴을 숨기지 않고 "알려진 한계 + 민감도 분석으로 결론 유지"로 보고하면, 군중 모델 검증이 거의 없는 로봇 유도 분야의 기존 연구보다 강한 근거가 됩니다. 측정은 `validation/`에 행동 층 모듈(예: `behaviour.py`)로 추가해 기존 이동 층 검증과 같은 형식의 결과 JSON을 내도록 할 계획입니다.

---

## 참고 문헌

이동 층과 보정

- Hoogendoorn, S. P., & Bovy, P. H. L. (2004). Pedestrian route-choice and activity scheduling theory and models. *Transportation Research Part B*, 38, 169–190. https://www.sciencedirect.com/science/article/abs/pii/S0191261503000079
- Haghani, M., & Sarvi, M. (2021). Calibrating parameters of crowd evacuation simulation at strategic, tactical and operational levels: Which one matters most? https://arxiv.org/abs/2109.02885
- Moussaïd, M., Helbing, D., & Theraulaz, G. (2011). How simple rules determine pedestrian behavior and crowd disasters. *PNAS*, 108(17), 6884–6888. https://www.pnas.org/doi/full/10.1073/pnas.1016507108
- Seyfried, A., Steffen, B., Klingsch, W., & Boltes, M. (2005). The fundamental diagram of pedestrian movement revisited. *J. Stat. Mech.*, P10002. https://arxiv.org/pdf/physics/0506170
- Tordeux, A., Chraibi, M., & Seyfried, A. Collision-free speed model for pedestrian dynamics. https://arxiv.org/pdf/1512.05597
- Chraibi, M., Seyfried, A., & Schadschneider, A. Generalized centrifugal-force model for pedestrian dynamics. https://www.semanticscholar.org/paper/Generalized-centrifugal-force-model-for-pedestrian-Chraibi-Seyfried/3e2b54825c7b71fa1fbf0dedfe3c7e95cf996acc
- Optimal Steps Model. https://pedestriandynamics.org/models/optimal_steps_model/
- Kleinmeier, B. 등. Vadere: An open-source simulation framework. https://arxiv.org/pdf/1907.09520
- Shirvani, M., Kesserwani, G., & Richmond, P. (2020). Agent-based modelling of pedestrian responses during flood emergency. *Journal of Hydroinformatics*, 22(5), 1078–1092. https://arxiv.org/abs/2004.10589

인지, 결정, 사회적 영향

- Lindell, M. K., & Perry, R. W. (2012). The Protective Action Decision Model. *Risk Analysis*. https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1539-6924.2011.01647.x
- Kuligowski, E. The Process of Human Behavior in Fires. NIST. https://www.nist.gov/publications/process-human-behavior-fires
- Lovreglio, R., Kuligowski, E., Gwynne, S., & Boyce, K. (2019). A pre-evacuation database for use in egress simulations. *Fire Safety Journal*. https://www.nist.gov/publications/pre-evacuation-database-use-egress-simulations
- Templeton, A., Xie, H., Gwynne, S., Hunt, A., Thompson, P., & Köster, G. (2023). Agent-based models of social behaviour and communication in evacuations: A systematic review. *Safety Science*. https://arxiv.org/abs/2310.15761
- Zhao, D. 등 (2026). Information-driven behavioural dynamics in indoor gas-leak evacuation. *Physica A*, 688. https://www.sciencedirect.com/science/article/abs/pii/S0378437126001457
- Helbing, D., Farkas, I., & Vicsek, T. (2000). Simulating dynamical features of escape panic. *Nature*, 407, 487–490.
- Representing crowd behaviour in emergency planning guidance: 'mass panic' or collective resilience? https://www.tandfonline.com/doi/full/10.1080/21693293.2013.765740
- von Sivers, I. 등 (2016). Modelling social identification and helping in evacuation simulation. *Safety Science*. https://arxiv.org/abs/1602.00805
- Durupinar, F. 등 (2016). Psychological parameters for crowd simulation: From audiences to mobs.
- Tsai, J. 등 (2011). ESCAPES. https://www.researchgate.net/publication/221455296
- Andresen, E., Haensel, D., Chraibi, M., & Seyfried, A. (2016). Wayfinding and cognitive maps for pedestrian models. https://arxiv.org/abs/1602.01971
- Haghani, M., & Sarvi, M. (2017). Stated and revealed exit choices of pedestrian crowd evacuees. https://sciencedirect.com/science/article/abs/pii/S0191261516306762

유도와 로봇

- Mayr, C. M., & Köster, G. (2022). Guiding crowds when facing limited compliance: Simulating strategies. *PLOS ONE*. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0276229
- Robinette, P., Li, W., Allen, R., Howard, A. M., & Wagner, A. R. (2016). Overtrust of robots in emergency evacuation scenarios. *HRI 2016*.
- Nayyar, M., & Wagner, A. R. Exploring the effect of explanations during robot-guided emergency evacuation. https://link.springer.com/chapter/10.1007/978-3-030-62056-1_2
- Nayyar, M. 등 (2023). Learning evacuee models from robot-guided emergency evacuation experiments. https://arxiv.org/abs/2306.17824
- Evacuee behavior modeling during robot-guided evacuations (2025). *International Journal of Social Robotics*. https://link.springer.com/article/10.1007/s12369-025-01259-w
- Zheng, T. 등 (2023/24). Multi-robot-guided crowd evacuation: Two-scale modeling and control. https://arxiv.org/abs/2302.14752
- Wan, Z. 등 (2020). Robot-assisted pedestrian regulation based on deep reinforcement learning. *IEEE Trans. Cybernetics*, 50(4), 1669–1682. https://pubmed.ncbi.nlm.nih.gov/30475740/
- Pedestrian-robot interaction experiments in an exit corridor. https://arxiv.org/pdf/1802.05730
- Sakour, I., & Hu, H. (2017). Robot-assisted crowd evacuation under emergency situations: A survey. *Robotics*, 6(2), 8. https://doi.org/10.3390/robotics6020008

새로운 흐름

- When agents learn to think: LLM-enhanced agent-based modeling for crowd evacuation (2025). *Reliability Engineering & System Safety*. https://www.sciencedirect.com/science/article/abs/pii/S0951832025012554
- Liu, Y., Shatzel, L., Haworth, B., & Schneider, T. (2025). Emergent crowd dynamics from language-driven multi-agent interactions. https://arxiv.org/abs/2508.15047

검증

- Ronchi, E., Kuligowski, E. D., Reneke, P. A., Peacock, R. D., & Nilsson, D. (2013). The process of verification and validation of building fire evacuation models. NIST TN 1822. https://www.nist.gov/publications/process-verification-and-validation-building-fire-evacuation-models?pub_id=913642
- IMO MSC.1/Circ.1533 (2016). https://www.traffgo-ht.com/downloads/pedestrians/downloads/documents/MSC.1,Circ.1533,2016.pdf
- RiMEA: A way to define a standard for evacuation calculations. https://www.researchgate.net/publication/300661139
- Wolinski, D. 등 (2014). Parameter estimation and comparative evaluation of crowd simulations. *Computer Graphics Forum*, 33, 303–312. https://onlinelibrary.wiley.com/doi/10.1111/cgf.12328
- Jülich Pedestrian Dynamics Data Archive. https://www.re3data.org/repository/r3d100013370
- Grimm, V. 등 (2005). Pattern-oriented modeling of agent-based complex systems. *Science*, 310, 987–991. https://www.science.org/doi/10.1126/science.1116681
- Kinateder, M., & Warren, W. H. (2016). Social influence on evacuation behavior in real and virtual environments. *Frontiers in Robotics and AI*. https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00043/full
- Analysis of the use of behavioral data from virtual reality for calibration of agent-based evacuation models (2023). *Heliyon*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10015235/
- Pretorius, M. 등 (2015). Large crowd modelling: An analysis of the Duisburg Love Parade disaster. *Fire and Materials*. https://onlinelibrary.wiley.com/doi/abs/10.1002/fam.2214
- Assessing crowd management strategies for the 2010 Love Parade disaster (2020). *J. R. Soc. Interface*. https://royalsocietypublishing.org/doi/10.1098/rsif.2020.0116
