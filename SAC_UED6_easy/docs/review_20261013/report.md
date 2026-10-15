# SAC_UED6_easy 학습 정체 원인과 돌파 방안 — 2026-10-13

## 0. 결론 먼저

1. **[Certain] 학습이 "안 되는" 것이 아니라 에피소드 약 1000에서 같은 지점으로 수렴한 뒤 거의 움직이지 않는다.** 고정 검증 9개 시나리오에서 결정적 정책의 위험구역 사람-스텝은 네 변형(stored/current × Gumbel/expectation) 모두 43.7k–48.9k다. 로봇 끔 60.1k 대비 −25~27%, 손으로 짠 shuttle 34.0k(−43%)에 못 미친다. 알고리즘 세부 변형은 병목이 아니다.
2. **[Certain] 학습된 정책은 "위험구역 근처에서 guide를 켜고 서 있거나 천천히 움직이는" 국소 해다.** 결정적 속도 비율 하위 25%가 0.04(정지), 검증 평균 속도 0.39–0.47 m/s(shuttle 1.08). 위험구역 깊은 곳 체류자의 사람-스텝을 shuttle은 64% 줄이고 정책은 28–31%만 줄인다.
3. **[Certain] 한 결정의 실제 효과는 크고, critic은 그것을 약 1/10로 줄여서 안다.** 같은 시뮬레이터 상태를 복제해 로봇 하나의 4초 행동만 바꾼 반사실 롤아웃에서 200 s 할인 return의 행동 간 표준편차는 잡음 보정 후 10.5(합동 F 검정 p = 3 × 10⁻⁵), critic이 같은 행동들에 매긴 Q의 표준편차는 1.08, 상관 +0.39다(상태 6개 × 행동 6개 × 반복 3회, 상태별 편차 큼). 한 결정의 4초 즉시 보상은 행동과 무관하다(행동 간 sd 0.06 vs 잡음 0.12). 같은 데이터로 학습한 q1과 q2의 행동 순위 상관도 평균 0.39뿐이다.
4. **[Likely, 10/13 추가 확인 후 수정] 압축의 주된 원인은 정보 부재가 아니라 학습 신호 부족이다.** 한 결정이 즉시 바꾸는 것은 추종자 집합이다(guide 4초면 0 → 9–38명, off면 즉시 0명). 그러나 추종자 수는 "로봇 주변 3.5/5/10 m 인원 × guide 여부"만으로 처음 보는 레벨에서도 R² 0.82(MAE 2명)로 예측된다. 이 정보는 이미 ego 입력과 상태 벡터에 있다. 학습된 actor·critic의 내부 특징으로는 처음 보는 레벨에서 R² 0.2 이하다(같은 레벨 안에서는 0.56–0.82). 입력에 있는 정보를 네트워크가 일반화되는 형태로 뽑지 못한 것이다. 효과가 수십~수백 초 뒤에 나타나고 즉시 보상의 행동 신호가 없으며 탐색이 내부에 닿지 않아 그 표현을 배울 압력이 약하다고 본다. (처음 판에서 FE_RV3의 `agent_num`을 근거로 들었으나 그것은 개수가 아니라 boolean이고, 사용자 ablation에서 효과가 작았다. 아래 5.6절대로 FE_RV3에서는 추종 여부가 이미지로 거의 정해져 중복 정보였을 것이므로 그 결과와도 맞는다.)
5. **[Certain] 두 번째 병목은 탐색 범위다.** 지금의 4초 무작위 warm-up은 로봇을 위험구역 안에 5–43%의 시간만 두고 대부분 경계 근처에 머문다. 같은 무작위라도 목표를 도착까지 유지하면 80–97%를 안에서 보내고, 정책 없이도 체류자 사람-스텝을 세 시나리오 모두에서 학습 정책보다 더 줄인다(예: 5.4k vs 18.2–22.3k). 깊은 진입의 결과가 replay에 거의 없으니 critic이 그 가치를 배울 수 없다.
6. **[Certain] FE_RV3가 잘 된 이유는 알고리즘이 아니라 문제 구조다.** 하이퍼파라미터(LR, 배치, γ, τ, UPT)는 같다. 다른 것은 확률 1 추종, 출구에서 영구 제거(단조 진행), 전원 대피 시 종료, 로봇 중심 dense shaping("가장 먼 사람까지 거리", 출구에서 빈손 패널티), 단일 로봇, 이미지에 그려진 로봇, 고대비 군중 픽셀이다. UED6_easy에는 하나도 없다.
7. **돌파 방향.** 원인이 둘(행동 효과를 배울 신호가 약함, 탐색이 내부에 닿지 않음)이므로 둘을 함께 겨냥한다. (R4) 목표를 도착까지 유지하는 시간 추상화(원래 비-easy 설정의 이벤트 구동을 되살리고 범위 40 m, warm-up도 유지형), (R2) 로봇별 접근 잠재함수 shaping(한 결정 신호 비율 7.2, 기본 보상 0.5) + 진행 잠재함수, (R3) 밀도 채널 스케일·로봇 위치 채널. 이 셋을 한 run으로 묶어 돌파 여부를 먼저 확인하고 ablation으로 나눈다. (R1) 추종자 수·숨은 군중 상태의 명시 입력은 값싼 보조 수단으로 낮췄다. (R5) shuttle 사전정책은 병렬 진단 run, (R6) 그래도 안 되면 critic 기울기에 의존하지 않는 MAPPO. 판정 기준은 10장.
8. **[Certain] 지금 바로 정리할 운영 문제:** 충돌한 이전 run의 고아 워커 24개가 남아 있고 그중 12개가 각 25% CPU(합 약 3코어)를 쓰고 있다. 학습기 처리량(3.9 update/s)이 전체를 112 에피소드/시간으로 묶는다(warm-up 245/시간).

## 1. 근거와 범위

- W&B `437snowcap-sungkyunkwan-university`의 `adds-sac-ued6-261011ENV` 5개 run과 `adds-sac-ued6` 4개 run의 history 전체를 공식 API로 받았다(스칼라만).
- 로컬에서 돌고 있는 `sanity-stored-fullinfo-Gumbel`(czvwyevm)의 `~/Log_SAC_UED6_easy_fullinfo/`: 시나리오별 검증 원기록, 체크포인트 ep2300·ep2400, replay buffer(302k records).
- 코드: `SAC_FE_RV3` 학습기·모델·군중 에이전트, `SAC_UED6_easy`의 `learn/`, `sim/`(rewards, observation, robot_action, agent의 로봇-군중 상호작용, model의 step·유입·유출·배치), `validation/shuttle_baseline.py`, 설정 세 파일.
- 새로 돌린 읽기 전용 진단(스크립트·결과는 이 폴더):
  1. `critic_probe.py`: critic의 행동 민감도
  2. `critic_consistency.py`: q1/q2, 체크포인트 간 행동 선호 일치도
  3. `obs_probe.py`: 관측 스케일과 결정적 정책 출력
  4. `true_adv.py`: 반사실 롤아웃(시뮬레이터 deepcopy + 공통 난수)
  5. `shaping_probe.py`: 한 결정 동안의 기본 보상 vs 잠재함수 신호, 추종자 변화
  6. `explore_cov.py`: 행동 parameterization별 무작위 탐색이 닿는 범위
  7. `follower_decode.py`, `follower_fit.py`: 추종자 수가 현재 입력·학습된 특징으로 읽히는지(10/13 추가)
  8. `prof_step.py`: 시뮬레이터 step 프로파일
- 학습 코드·설정은 바꾸지 않았고 돌고 있는 학습도 건드리지 않았다(nice로 남는 CPU에서 실행).
- 하지 않은 것: 새 학습 run, FE_RV3 재학습(로컬에 FE_RV3 로그 없음), 전체 테스트 스위트.

## 2. 증상: 실패가 아니라 조기 정체

### 2.1 고정 검증 (9 시나리오 × 1 시드, 결정적 정책, 위험구역 사람-스텝, 낮을수록 좋음)

로봇 끔 60,123 / shuttle 34,045.

| run | ep500 | ep1000 | ep1500 | ep2000 | ep2500 | ep3000 |
|---|---:|---:|---:|---:|---:|---:|
| sanity-stored-fullinfo-ModeExpectation | 104,666 | 44,571 | 44,878 | 44,319 | 43,739 | 43,975 |
| sanity-stored-fullinfo-Gumbel (진행 중) | 47,595 | 45,086 | 46,212 | 46,200 | - | - |
| sanity-current-fullinfo-ModeExpectation | 78,993 | 46,990 | 48,878 | 46,672 | 47,452 | 46,033 |
| sanity-stored-fullinfo-ModeExpectation-Epsilon | 59,905 | - | - | - | - | - |

- **[Certain]** ep1000 이후 네 run이 같은 띠 안에 있다. 변형 간 차이는 띠 안의 잡음 수준이다.
- **[Certain]** 학습 곡선 총보상은 느리게 오른다(6e4ys8kv: 501–1000 −1372 → 3001–3500 −1249, 약 9%, 구간 SE 약 15). 검증 총보상도 −1278 → −1185. 정지가 아니라 매우 느린 개선이다.
- **[Certain]** 검증은 9쌍 1시드라 정책 간 8% 미만 차이는 판별하기 어렵다(ep1000 vs ep2000 시나리오별 차이 ±5,000, 평균 차이 SE 약 2,000).
- 10/09–10/10 이전 환경의 run도 같은 모양(학습 시작 후 500–1000 에피소드 안에 크게 오르고 그 뒤 완만)이었다. 10/11 환경 변경은 절대 수치만 바꿨다.

## 3. 정책이 실제로 하는 일

### 3.1 검증 시나리오 9개 평균 (로컬 run)

| 지표 | 끔 | shuttle | 정책 ep1000 | 정책 ep2000 |
|---|---:|---:|---:|---:|
| 위험구역 사람-스텝 | 60,123 | 34,045 | 45,086 | 46,200 |
| 그중 활동 체류자 몫 | 21,112 | 7,668 | 14,666 | 15,237 |
| 행동을 시작한 사람(ever_acted) | 108* | 412 | 385 | 391 |
| 로봇 정보로 떠난 사람(informed_departures) | 96 | 350 | 324 | 298 |
| 추종해서 떠난 사람(followed_departures) | 41 | 161 | 126 | 108 |
| 로봇 평균 속도 (m/s) | 0 | 1.08 | 0.39 | 0.47 |
| guide 비율 | 0 | 0.64 | 0.51 | 0.62 |

\* 끔 조건의 ever_acted는 covent_garden_1과 hongdae_0의 자연 인지 때문이다.

- **[Certain]** 정책은 사람을 행동하게 만드는 데는 shuttle에 근접한다(385–391 vs 412). 차이는 **체류자**에서 크다: shuttle −64%, 정책 −28~31%. hongdae_2(초기 내부 191명)에서 체류자 사람-스텝 43.0k / 11.6k / 33.7–37.5k(끔/shuttle/정책).
- **[Certain]** shuttle은 위험구역 반경 절반 지점까지 신호를 끈 채 들어가 guide로 끌고 나와 바깥에서 10 s 기다린다. 체류자는 30–600 s 그 자리에 있으므로 로봇이 10 m 안까지 가야 반응한다. 정책은 그 거리까지 가지 않는다. hongdae_2에서 로봇 0은 160 s가 지나도 위험구역 밖에 있었다(5.2–5.3절, 가장 가까운 미접촉 내부 인원까지 약 37 m).

### 3.2 결정적 정책의 행동 분포 (replay 768 로봇-결정)

- 속도 비율 분위: 10% 0.015, 25% 0.038, 50% 0.62, 75% 0.93. **약 1/3이 사실상 정지.** 속도 차원의 사전-squash |μ| = 2.66, log-std 0.13으로 가장 확신에 찬 차원이 "설지 갈지"다.
- 이동 목표 크기 중앙값 13.3 m, guide 비율 0.44. 이동 차원 log-std 0.40–0.43(상한 0.5): 방향은 거의 무작위.
- **[Likely]** 배운 것은 "guide를 켜고 서 있으면 10 m 안의 사람이 다가오고, 3.5 m 안에서 10 s 머물면 절반이 위험 정보를 받는다"는 국소 메커니즘(정보 키오스크)이다. 이것만으로 끔 대비 −25%가 나온다. 진입-유도-대기 순환은 없다.

## 4. critic 진단

### 4.1 행동 민감도 (`critic_probe.py`, ep2400, replay 512 상태)

같은 결정 시점·같은 팀 행동에서 로봇 j의 행동만 바꿔 j의 Q(min(q1, q2))를 비교했다.

| 양 | 평균 | 중앙값 | p90 |
|---|---:|---:|---:|
| Q(s, a_stored) | −344.9 | −288.0 | −114.2 |
| 상태 간 Q 표준편차 | 234 | | |
| TD 오차 q − y | −3.55 (sd 7.74) | −1.25 | 3.17 |
| 4 s 보상 합 | −11.1 (sd 8.4) | | |
| **무작위 행동 32개에 대한 Q sd** | **0.49** | 0.32 | 1.08 |
| 무작위 행동 32개의 Q 범위 | 1.77 | 1.17 | 3.85 |
| **정책 샘플 32개에 대한 Q sd** | **0.16** | 0.13 | 0.30 |
| guide − off (같은 연속 행동) | +0.08 (절댓값 0.31) | | |
| ∂Q/∂a_cont 크기 | 0.48 | 0.30 | 1.09 |

- **[Certain]** 정책 log-std: x 0.40, y 0.43(상한 0.5), 속도 0.13. α는 update 5,200부터 하한 0.05에 붙어 학습 로그의 97.9–98.7%에서 하한이다. 엔트로피 목표 −3, 실제 +2.3 안팎.

### 4.2 q1·q2와 체크포인트 간 일치도 (`critic_consistency.py`)

상태 128개 × 로봇 0의 무작위 후보 행동 24개에서 상태 평균을 뺀 Q의 상태별 상관.

| 비교 | 평균 상관 | 중앙값 |
|---|---:|---:|
| q1 vs q2 (ep2400) | 0.39 | 0.69 |
| q1 vs q2 (ep2300) | 0.34 | 0.61 |
| q1 ep2300 vs ep2400 | 0.65 | 0.81 |
| q2 ep2300 vs ep2400 | 0.67 | 0.84 |

- **[Certain]** q1 − q2의 행동 성분 sd(0.68)가 각 critic의 행동 성분 sd(0.66, 0.75)와 같은 크기다. 행동 의존 성분의 절반가량이 데이터가 아니라 초기값과 잡음으로 정해진다.
- actor는 min(q1, q2)를 미분하므로 서로 다른 잡음 지형 두 개의 최솟값을 따라간다. 방향 엔트로피가 최대 근처에 머무는 현상과 맞는다.

### 4.3 관측 스케일 (`obs_probe.py`)

| 채널 | 0이 아닌 셀 비율 | 0이 아닌 값 중앙값 |
|---|---:|---:|
| ego/own_crowd (1 m) | 1.9% | 0.50 |
| mid/team_crowd (2 m) | 5.0% | **0.051** |
| glob/team_crowd (3.1 m) | 5.8% | **0.051** |
| critic priv (3.1 m) | 5.8% | **0.051** |
| obstacle / walkable / hazard | 25–77% | 0.9–1.0 |

- **[Certain]** `OBS_DENSITY_SATURATION = 2.0` 인/m²이 1.0이다. 실제 밀도 0.04–0.05에서 mid·global·priv의 사람 한 명은 0.05이고, 정적 채널(0–1)과 같은 첫 합성곱에 섞여 약 20배 약한 신호가 된다. FE_RV3는 사람을 150/255 = 0.59로 그리고 max-pool로 줄였다.
- **[Certain]** 로봇 자신·팀원 위치는 어떤 지도 채널에도 없다(상태 벡터 스칼라뿐). FE_RV3는 로봇을 255로 그렸다. 합성곱은 좌표↔픽셀 변환에 약하다(CoordConv, Liu et al. 2018).
- **[Certain]** mid·global 군중 채널은 최신 1 프레임뿐이라 12.5 m 밖의 이동 방향이 안 보인다. FE_RV3의 global은 4 프레임이었다.
- 행동도 지도와 정렬되어 있지 않다. 로봇 기준 (dx, dy) 벡터가 critic의 MLP·FiLM으로 들어간다.

## 5. 반사실 롤아웃: 한 결정의 실제 효과는 크고, critic은 그것을 압축한다

### 5.1 방법 (`true_adv.py`)

검증 시나리오에서 학습된 확률적 정책으로 t0개 결정을 진행한 뒤 시뮬레이터 전체를 deepcopy한다. 로봇 0만 한 결정(4 s) 동안 여섯 행동 중 하나를 하고(팀원은 같은 정책 행동으로 고정), 이후 모든 로봇이 확률적 정책을 따른다. 행동끼리 같은 난수열(CRN)을 쓰고 난수열을 바꿔 3회 반복했다. 지표는 학습과 같은 할인(γ = 0.99/2 s)의 200 s 할인 return, 즉 critic이 배워야 하는 Q^π(s, a) 차이다. 같은 상태·행동의 critic Q도 계산했다.

행동: `policy_det`(정책의 결정적 행동), `in_off`·`in_guide`(위험구역 중심 쪽, 속도 1.0·0.6), `out_guide`·`out_off`(가장 가까운 바깥 지점 쪽), `still_guide`(정지, guide).

### 5.2 결과

200 s 할인 return의 `policy_det` 대비 차이(3회 평균), 행동 간 sd, 잡음 보정 sd(이원분산분석, 행동 × 반복), critic Q의 행동 간 sd와 상관.

| 상태 | 위험구역 인원 / N_ref | 로봇0 | in_off | in_guide | out_guide | out_off | still_guide | 행동 간 sd | 잡음 보정 sd | 반복 잡음 sd | critic sd | 상관 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dotonbori_0, t=40 s | 158 / 111 | 안 | −57 | −61 | −108 | −21 | −37 | 37.3 | 33.9 | 26.9 | 0.46 | +0.66 |
| dotonbori_0, t=160 s | 66 / 111 | 안 | +6 | 0 | +11 | −3 | +9 | 5.6 | 3.1 | 8.2 | 1.61 | +0.23 |
| covent_garden_0, t=40 s | 97 / 82 | 밖 | +23 | −16 | −18 | −23 | −16 | 17.3 | 11.8 | 21.9 | 0.18 | +0.15 |
| covent_garden_0, t=160 s | 67 / 82 | 안 | −3 | −28 | −19 | −5 | +10 | 13.7 | 9.2 | 17.5 | 2.33 | +0.48 |
| hongdae_2, t=40 s | 245 / 191 | 밖 | −8 | +6 | +10 | +8 | +9 | 6.9 | 1.6 | 11.7 | 1.03 | +0.35 |
| hongdae_2, t=160 s | 165 / 191 | 밖 | −12 | −6 | −9 | −5 | −14 | 5.2 | 3.3 | 6.9 | 0.89 | +0.48 |
| **평균** | | | | | | | | **14.3** | **10.5** | **15.5** | **1.08** | **+0.39** |

- 여섯 상태를 합친 행동 효과 검정: F(30, 60) = 3.38, p = 3 × 10⁻⁵. 한 결정의 효과는 통계적으로 실재한다. 상태 하나씩은 반복이 3회라 dotonbori t=40 s(F = 5.8)만 단독으로 뚜렷하다.
- 행동 간 최대−최소 차이 평균: 실제 39.7, critic 2.66.
- critic과 실제의 상관은 여섯 상태 모두 양수(+0.15 ~ +0.66)다. 순서는 어느 정도 알지만 크기를 약 1/10로 줄여서 안다.

### 5.3 한 결정 동안의 신호 (`shaping_probe.py`, 9 상태 × 6 행동 × 3 반복)

| 신호 (한 결정, 4 s) | 행동 간 sd | 반복 잡음 sd | 비율 |
|---|---:|---:|---:|
| 기본 과제 보상 R(4 s) | 0.059 | 0.117 | **0.50** |
| 진행 잠재함수 F_prog (가중치 1) | 0.005 | 0.003 | 1.30 |
| 접근 잠재함수 F_appr (가중치 1, 로봇 0) | 0.051 | 0.007 | **7.20** |

로봇 0의 추종자 수, 4 s 후:

| 상태 | 시작 | guide 행동들 | off 행동들 |
|---|---:|---:|---:|
| dotonbori t=24 s | 22 | 27–28 | 0 |
| dotonbori t=80 s | 0 | 9–12 | 0 |
| covent_garden t=80 s | 38 | 38 | 0 |
| covent_garden t=160 s | 0 | 8–10 | 0 |
| hongdae_2 t=24 s | 6 | 7–9 | 0 |

### 5.4 해석

- **[Certain]** 한 결정이 200 s return을 행동 간 sd 약 10(잡음 보정, 상태별 1.6–33.9)만큼 바꾼다. 그 결정의 4 s 즉시 보상에는 행동 정보가 없다(비율 0.50). 효과는 전부 뒤에 나타나 TD로 전파되어야 한다.
- **[Certain]** critic의 같은 행동들 간 Q sd는 1.08로 실제의 약 1/10이다. 순서는 어느 정도 맞다(상관 +0.39, 여섯 상태 모두 양수).
- **[Likely]** 한 결정이 즉시 바꾸는 것은 추종자 집합이다. guide 4 s로 9–38명이 붙고 off로 바꾸면 즉시 모두 떨어진다. 이 사람들이 이후 로봇을 따라 위험구역 안팎으로 움직이며 수십 보상 단위를 만든다. 추종자 수 자체는 입력에서 읽을 수 있지만(5.6절) 학습된 네트워크는 그것을 레벨을 넘어 쓰는 형태로 표현하지 못했다. 효과가 수십~수백 초 뒤에 오고, 한 결정의 즉시 보상에는 행동 정보가 없으며(비율 0.50), 200 s return의 반복 잡음(sd 15.5)이 효과(sd 10.5)보다 크다. 1단계 TD가 이 잡음 속에서 약 50 결정을 거슬러 효과를 전파해야 하므로 함수 근사가 행동 효과를 0 쪽으로 줄인다(회귀 희석).
- **[Likely]** 접근 잠재함수는 한 결정 안에서 깨끗한 행동 신호(비율 7.2)를 준다. 진행 잠재함수는 사람들 스스로의 이동에 섞여 신호가 약하다(1.3).

### 5.5 무작위 탐색이 닿는 범위 (`explore_cov.py`)

검증과 같은 레벨·시드, 1000 step, 로봇 3대. 정책을 쓰지 않는 세 가지 행동 생성기를 비교했다.

- `4 s 무작위`: 지금의 warm-up 탐색(`exploration_action`, 4 s마다 균등 무작위)
- `유지 무작위`: 위험구역 15 m 이내 보행 가능 지점을 무작위로 골라 도착 또는 30 s까지 유지, 구간마다 모드·속도 무작위(시간 추상화된 무작위 탐색)
- `경계 guide`: 가장 가까운 위험구역 경계 바깥 1–4 m 지점에 가서 guide로 서 있기(학습 정책이 가까이 간 국소 해)

| 시나리오 (반경) | 조건 | 로봇 위험구역 체류 비율 | 최대 진입 깊이 m (로봇 3대) | 체류자 사람-스텝 | 전체 사람-스텝 |
|---|---|---:|---|---:|---:|
| dotonbori_0 (38.7 m) | 4 s 무작위 | 0.43 | 27 / 26 / 8 | 21.8k | 75.9k |
| | 유지 무작위 | **0.97** | 37 / 39 / 38 | **7.8k** | 66.5k |
| | 경계 guide | 0.34 | 0 / 14 / 6 | 20.5k | 113.5k |
| | 참고: 끔 / shuttle / 정책 ep1000–2000 | | | 32.1k / 11.9k / 14.8–22.5k | 78.0k / 48.4k / 49.2–58.6k |
| covent_garden_0 (40.1 m) | 4 s 무작위 | 0.05 | 0 / 10 / 0 | 26.9k | 63.6k |
| | 유지 무작위 | **0.86** | 35 / 35 / 35 | **5.4k** | 54.3k |
| | 경계 guide | 0.00 | 0 / 0 / 0 | 12.4k | 45.4k |
| | 참고 | | | 25.6k / 8.8k / 18.2–22.3k | 64.2k / 35.3k / 50.6–54.1k |
| hongdae_2 (46.7 m) | 4 s 무작위 | 0.15 | 0 / 9 / 21 | 34.3k | 127.0k |
| | 유지 무작위 | **0.80** | 40 / 34 / 43 | **16.2k** | 136.0k |
| | 경계 guide | 0.00 | 0 / 0 / 0 | 32.4k | 127.4k |
| | 참고 | | | 43.0k / 11.6k / 33.7–37.5k | 149.4k / 75.1k / 99.5–119.6k |

- **[Certain]** 지금의 warm-up 탐색은 로봇을 위험구역 안에 5–43%의 시간만 두고 대부분 경계 근처에 머문다. 같은 무작위성이라도 목표를 도착까지 유지하면 80–97%의 시간을 안에서 보내고 중심까지 간다.
- **[Certain]** 정책 없이 무작위로 고른 목표를 유지하기만 해도 체류자 사람-스텝이 세 시나리오 모두에서 **학습 정책보다 낮다**(7.8k vs 14.8–22.5k, 5.4k vs 18.2–22.3k, 16.2k vs 33.7–37.5k). 두 곳에서는 shuttle보다도 낮다. 전체 사람-스텝이 나쁜 이유는 무작위 모드로 사람을 안으로 끌고 들어가기 때문이다.
- **[Likely]** 4 s 무작위 행동은 확산(random walk)이라 "깊이 들어가 체류자를 만나는" 경험이 replay에 드물다. critic이 그 가치를 배울 데이터가 없다. 5.4절의 critic 압축과 별개인 두 번째 병목이며, 시간 추상화(R4)가 직접 해결한다.
- **[Certain]** 경계 guide 대기는 dotonbori에서 끔보다 크게 나쁘다(113.5k vs 78.0k). guide는 양날이고, 학습 정책이 가까이 간 국소 해가 항상 안전한 것도 아니다.
- **[Likely]** 진행 중인 `Epsilon` run의 무작위 행동도 4 s 무작위라서 위 표의 첫 줄과 같은 범위밖에 탐색하지 못한다.

### 5.6 추종자 정보는 이미 입력에 있는가 (`follower_decode.py`, `follower_fit.py`)

10/13 사용자 지적(FE_RV3의 `agent_num`은 boolean이고 ablation 효과가 작았다)을 받아 직접 확인했다. 학습된 확률적 정책(ep2400)으로 검증 레벨 4개(dotonbori_0, covent_garden_0, covent_garden_2, hongdae_2)를 100 결정씩 돌려 로봇-결정 1,200개를 모았다. 추종자 수는 평균 7.0명(sd 14.1, 0보다 큰 비율 0.39, 최대 69)이다. 세 레벨로 학습한 decoder를 나머지 한 레벨에 적용했다.

| 입력 | 추종자 수, 처음 보는 레벨 R² | 같은 레벨 섞은 5-fold R² |
|---|---:|---:|
| 단순 관측량: 로봇 주변 3.5/5/10 m 인원, 모드, 위험구역 안 여부(ego 입력·상태 벡터로 계산 가능) | **0.82** (MAE 2.0명, ridge) / 0.83 (GBM) | - |
| 학습된 actor 특징(64차원) | −0.06 ~ 0.19 (ridge·GBM) | 0.56 |
| 학습된 critic 특징(행동 무관, 1042차원) | 0.12 ~ 0.22 | 0.82 |

- **[Certain]** 추종자 수는 지금 입력만으로 거의 정해진다. "숨은 변수라서 못 본다"는 처음 판의 가설은 틀렸다.
- **[Certain]** 학습된 네트워크의 특징은 그 정보를 레벨을 넘어 일반화되는 형태로 담지 않는다. 같은 레벨 안에서만 맞는 것은 레벨별 단서를 외운 것에 가깝다.
- **[Likely]** FE_RV3에서는 출구가 안 보이는 사람이 10 m 안에 로봇이 보이면 확률 1로 따라오므로 추종 여부가 이미지(로봇·사람·출구·벽 위치)로 거의 결정된다. boolean 플래그가 중복 정보라 ablation 차이가 작았던 것과 맞는다.
- 결론: 추종자 수를 명시 입력으로 주는 것(R1)은 정보를 새로 주는 것이 아니라 네트워크가 못 뽑은 특징을 대신 계산해 주는 지름길이다. 효과가 있을 수는 있으나 근본 처방은 학습 신호(R2)와 탐색·전파 깊이(R4)다.

## 6. FE_RV3는 왜 됐고 UED6_easy는 왜 안 되나

코드에서 직접 확인한 차이. 학습 난이도에 영향이 큰 순서.

| 항목 | SAC_FE_RV3 | SAC_UED6_easy | 영향 |
|---|---|---|---|
| 로봇 영향의 확실성 | 출구가 안 보이는 사람은 로봇이 시야 10 m 안이면 **확률 1**로 추종(`P_robot_following = 1`), 로봇은 항상 GUIDE | 인지 확률 (1 − d/10)/step, 시야 필요, logit 선택 기본 0.75 × 개인계수 0.67–1.33. off/guide를 정책이 골라야 하고 off면 추종자 즉시 해제. guide로 들어가면 사람을 위험구역으로 끌고 들어감 | 한 행동 결과의 분산과 즉시성 |
| 진행의 단조성 | 출구에 닿으면 **영구 제거**(`dead`) | 위험구역 밖으로 데려와도 크롭 안에 남는다. 추종 해제 후 정보를 못 받은 사람은 원래 경로로 복귀(재진입 가능). 유입 4명/100 step, 통과 통행 70%, 체류 30–600 s | 좋은 행동의 효과가 후속 행동과 무관하게 남는가 |
| 에피소드 종료 | 전원 대피 시 `done` | 항상 1000 step | 좋은/나쁜 정책의 return 차이 |
| 보상 | −생존비율×2, −Σdanger×0.003, **−가장 먼 사람까지 거리**(매 결정), **출구 근처 빈손이면 −2**, 충돌 −6, −0.5 | 사람-시간×5/N_ref, 남은 경로×1/N_ref, 재진입×2/N_ref, 충돌 0.05. 로봇 중심 항 없음 | 한 결정의 신용 할당 |
| 추종 상태 관측 | 로봇 상태에 추종자 유무 boolean(`agent_num`). 사용자 ablation에서 효과 작음 | 없음. 단, 주변 인원 × 모드로 R² 0.82 예측 가능(5.6절) | 두 환경 모두 정보는 입력에 있음. 차이는 아님 |
| 로봇 위치 표현 | 이미지에 255 픽셀 | 지도 채널에 없음 | 공간 관계 계산 |
| 군중 표현 | 150/255 픽셀, max-pool, global 4 프레임 | 밀도/2 인/m² → 0.05, mid·global 1 프레임 | 군중을 보는 능력 |
| 에이전트 수 | 1 | 3, 팀 보상(충돌만 개인) | 기여 희석 |
| 행동 | 속도 (dx, dy) ∈ [−2,2]², 2 s | waypoint ±20 m + 속도 + off/guide, 4 s, 건물 안 목표는 투영 | 행동→결과 사상 |
| 엔트로피 | α = 0.2 고정(`alpha_optimizer` 생성만, 갱신 없음) | 자동 α, 목표 −3, 하한 0.05에 고정 | 아래 참고 |
| 탐색 | ε 1 → 0, 에피소드 500–3500 선형 | ε = 1(500 에피소드) 후 0 | |
| 타임아웃 | `done`에 포함(엄밀히는 틀림) | 부트스트랩(올바름) | FE_RV3 쪽 오류는 학습을 막지 않음 |

### 6.1 하이퍼파라미터는 거의 같다

| 항목 | FE_RV3 | UED6_easy |
|---|---|---|
| LR / batch / buffer | 1e-4 / 128 / 1M | 1e-4 / 128(×3 로봇) / 1M |
| γ / τ | 0.99 per 2 s / 0.005 | 0.99 per 2 s (4 s 결정당 0.98) / 0.005 |
| UPT / 워커 | 1 / 4(warm-up 10) | 1 / 12 |
| 네트워크 | CNN 32-64-128 + BN, FC 512-256-64 | CNN 32-64-128 + GN × 3 branch, actor 5.6M / critic 8.6M |
| log-std 범위 | [−20, 0.5] | [−5, 0.5] |
| 에피소드 | ≤ 6000 step, 결정 ≤ 1500 | 1000 step, 결정 125 |
| 지도·군중 | 100 m 실내, 30명 | 200 m 실외, 약 600명, 위험구역 반경 36–50 m |

- **[Certain]** 상대적 엔트로피 비중은 오히려 FE_RV3가 컸다(α 0.2 × 엔트로피 1–2 대 결정당 보상 −3~−6, UED6는 α 0.05 × 2.3 대 약 −11). "α가 커서 안 배운다"는 설명은 맞지 않는다. 문제는 α 대비 critic의 행동 차이가 작다는 것이다.
- **[Likely]** FE_RV3에서는 로봇이 무작위로 움직여도 따라온 사람은 계속 따라오고, 출구를 보면 영구히 빠진다. 무작위 후속 행동이 앞선 좋은 행동의 효과를 지우지 못하고, "가장 먼 사람까지 거리" 항이 매 결정 방향을 알려 준다. UED6에서는 셋 다 없다.
- **[Likely]** easy 설정에서 없앤 어려움(부분관측, 가변 결정, 큰 위험구역)은 이 차이의 핵심이 아니었다. 오히려 고정 4 s 결정은 하나의 유효 순환(진입-유도-대기, 약 45–55 s)을 11–14 결정으로 쪼개 TD 전파 깊이를 늘렸을 가능성이 있다.

## 7. 숨겨진 오류와 운영 문제

학습 실패를 직접 설명하는 코드 버그는 찾지 못했다. SAC 목표·critic·actor 손실·온도 갱신의 부호, 가변 hold 할인, replay의 창·다음 상태 연결, 로봇 순서 섞기(`permute_robots`) 일관성, 결정 기록 시점과 보상 적산 구간을 확인했다.

| # | 내용 | 판정 | 영향·조치 |
|---|---|---|---|
| H1 | **고아 프로세스 24개.** 10/11 21:42 시작 워커 12개(PID 333857–333868, PPID 1550)가 각 25% CPU(합 약 3코어), 10/12 23:06 시작 12개(479300, 479320–479329, 479349)는 유휴. 모두 cwd가 SAC_UED6_easy | [Certain] | 현재 학습의 CPU를 뺏는다. 종료 권장 |
| H2 | 군중 밀도 채널 스케일(4.3절) | [Certain] | 스키마 변경 필요 |
| H3 | 로봇 위치가 지도에 없음, 행동이 지도와 비정렬, mid·global 1 프레임 | [Certain] | 공간 관계·이동 방향 학습 부담 |
| H4 | 추종·인지·정보 상태가 명시 입력에 없음. 추종자 수는 주변 인원 × 모드로 R² 0.82 예측되지만, 정보 수신·인지·체류 상태는 위치로 알 수 없다 | [Certain] | 처음 판에서 핵심 원인 후보로 봤으나 5.6절로 낮춤. 정보 수신·인지 상태는 critic privileged로 줄 가치가 남아 있다 |
| H5 | α가 하한에 98% 고정 → 사실상 고정 α 0.05 | [Certain] | 원인은 critic 쪽 |
| H6 | 엔트로피 목표 −3(연속 3차원)에 mode 엔트로피가 섞임 | [Certain] | 작음. 연속/이산 온도 분리 권장 |
| H7 | 검증 9쌍 1시드 | [Certain] | 8% 미만 개선 판별 불가 |
| H8 | 학습기 3.9 update/s → learning 단계 112 에피소드/시간(warm-up 245/시간, 순수 시뮬레이션 용량 약 300/시간) | [Certain] | 실험 1회 약 하루의 주된 이유. 배치 재구성(CPU)과 GPU 갱신이 각 약 0.18 s |
| H9 | step 평균 143 ms. 프로파일에서 `is_free_point`가 호출마다 shapely `box`를 만들어(60 step에 29만 회) step 시간의 약 30% | [Certain] | 거리 변환 격자 + 경계 근처만 정확 검사로 교체하면 단축 가능 |
| H10 | 워커는 에피소드 시작에만 정책을 받고 방송은 10 에피소드마다 | [Certain] | off-policy라 치명적이지 않음 |
| H11 | 반사실 비교의 공통 난수는 초기 조건만 맞춘다 | [Certain] | 검증·이번 probe 공통. 잡음 추정에 반영 |

## 8. MARL 구조 검토

- **CTDE(공유 actor + 중앙 critic, 로봇별 출력)** 자체는 표준(MADDPG 계열)이다. stored/current 차이가 없었으므로 팀원 행동 처리 방식은 병목이 아니다. [Certain]
- **혼합 행동(Gumbel vs 정확 기대값)** 차이 없음. [Certain] critic이 모드 차이를 잡음 속에서만 보면 추정기 개선의 이득이 드러나지 않는다.
- **팀 보상 희석.** 한 로봇의 4 s 행동은 팀 사람-시간의 일부만 바꾸고, 다른 두 로봇과 군중 자율 이동의 효과에 섞인다. 로봇별 잠재함수와 차이 보상이 이것을 직접 다룬다(Devlin et al. 2014). [Likely]
- **역할 분담.** shuttle은 로봇 번호별 진입 방향을 고정해 퍼진다. 학습 정책은 로봇 ID가 없고 팀원 정보는 스칼라뿐이다. 의도를 지도로 공유하면 협동이 나아진다는 결과가 있다(Wu et al. 2021). [Likely]
- **결정 동기화.** 이벤트 구동으로 바꾸면 한 로봇의 도착이 팀 전체 재결정을 부른다. 매크로 행동 MARL(Xiao et al. 2019, 2022)이 비동기 결정의 replay와 학습 규칙을 준다.

## 9. 환경 특이성 (학습 관점)

행동 모델은 문헌 근거를 잘 갖추고 있고, 이 검토는 그것을 바꾸자는 것이 아니다. RL 입장에서 어려움을 만드는 특성만 적는다.

1. **효과 경로가 좁고 확률적이다.** 로봇 → (guide, 10 m, 시야, 인지 확률) → 행동 시작·추종(약 0.75) → (3.5 m에서 10 s, 50%) 위험 정보 → 회피. 추종만 하고 정보를 못 받은 사람은 로봇이 떠나면 원래 경로로 돌아간다.
2. **모드와 방향의 상호작용이 결과의 부호를 바꾼다.** 들어갈 때 guide면 손해, 나올 때 guide면 이득(10/01 측정: guide로 들어갔다 나오는 스크립트는 모든 조건에서 악화). off는 추종자를 즉시 해제한다.
3. **체류자.** 초기 인원의 30%가 30–600 s 체류하고 끔 기준 위험구역 사람-스텝의 35%(21.1k/60.1k)가 이들에서 나온다.
4. **자연 감소와 재충전.** 위험구역은 로봇 없이도 비고, 유입·통과로 다시 찬다. 로봇 효과는 그 위의 증분이다.
5. **결과의 분산.** 에피소드 총보상의 표준편차(학습 후반 약 440)가 로봇 효과(shuttle − 끔 약 770)와 같은 자릿수다.

## 10. 돌파 방안

원칙: run 1회가 약 하루이므로 원인에 직접 닿는 변경을 **한 run에 묶어 먼저 돌파 여부를 확인**하고, 성공하면 하나씩 빼는 ablation으로 기여를 나눈다. 동시에 "현재 학습기가 좋은 행동을 유지·개선할 수 있는가"를 묻는 진단 run을 둔다.

### R1. 추종·군중 상태의 명시 입력 (보조, 5.6절로 우선순위 낮춤)

- actor 자기 상태: 자기 추종자 수(예: /20, 1에서 포화), 그중 위험구역 안 인원. 팀원 슬롯에 같은 두 값. 실제 로봇도 자기 뒤를 따라오는 사람은 lidar 추적으로 셀 수 있으므로 부분관측 단계에서도 유지 가능한 입력이다.
- critic privileged: 셀별 '추종 중', '위험구역 안 미접촉', '로봇에게 정보 받음', '체류 중' 인원 채널, 로봇 토큰에 추종자 수. 비대칭 actor-critic(Pinto et al. 2018).
- 구현: `record_decision`에서 집계해 `DecisionRecord`에 필드 추가, `own_state_layout`·`teammate_state_layout`·`critic_privileged` 확장, `OBSERVATION_SCHEMA_VERSION` 올림.
- 근거와 한계: 한 결정의 즉시 효과가 추종 상태에 집중된다(5.3절). 그러나 추종자 수는 이미 입력에서 R² 0.82로 읽힌다(5.6절). FE_RV3의 boolean 플래그는 사용자 ablation에서 효과가 작았다. 따라서 추종자 수는 값싼 지름길 정도로 기대하고, 위치로 알 수 없는 정보 수신·인지·체류 상태만 critic privileged로 주는 것이 정보 측면에서 의미가 있다(R2a의 '미접촉' 판정에 필요).

### R2. 잠재함수 기반 shaping (학습 보상 전용)

F = γ_step·Φ(s′) − Φ(s)를 매 step 더한다. hold k 동안 할인 적산하면 γ^k Φ(s_k) − Φ(s_0)로 telescoping하므로 결정 단위에서도 최적 정책이 바뀌지 않는다(Ng et al. 1999; 다중 에이전트 Devlin & Kudenko 2011, Devlin et al. 2014).

- **R2a 로봇별 접근:** Φ_appr,j = −min(d(로봇 j, 위험구역 안 미접촉 사람), 50 m)/50 m, 모두 접촉했으면 0. d는 직선거리로 시작해도 되지만 navmesh 측지 거리가 더 정확하다. 미접촉 = 추종 중 아님 + 로봇 정보 받지 않음. 로봇별 항이라 충돌처럼 그 로봇의 학습 보상에만 넣는다. 한 결정 신호 비율 7.2(5.3절). FE_RV3의 "가장 먼 사람까지 거리"와 같은 역할을 정책 불변 형태로 한다.
- **R2b 팀 진행:** Φ_prog = −Σ_{위험구역 안 i} min(1, esc_i/D_ref)/N_ref. 비율 1.3으로 약하지만 guide인 채 안으로 끌고 들어가는 결정에 즉시 음의 신호를 준다.
- 크기: 단호한 이동 한 결정의 |F|가 0.5–2가 되도록. 5.3절 기준 가중치 1에서 F_appr 행동 간 sd 0.05이므로 w_appr ≈ 10–40부터 시작.
- 주의: '정보 받음'은 위치로 알 수 없는 숨은 상태라, 그것에 의존하는 Φ는 critic이 그 상태를 privileged 채널로 봐야(R1 일부) 잡음 없이 학습된다. 관측 가능한 근사(예: guide 중인 로봇 5 m 안에 없는 위험구역 안 사람)로 정의해도 된다. `REWARD_VERSION`을 올리고 F는 별도 항으로 기록, 검증 지표에는 넣지 않는다. 구현 스케치는 부록 A.

### R3. 관측 표현

- `OBS_DENSITY_SATURATION` 2.0 → 0.25 인/m² 또는 log1p(count) 스케일.
- mid·global에 자기·팀원 위치 채널(이벤트 구동을 쓰면 팀원 목표점도).
- mid·global 군중 채널 2–4 프레임.

### R4. 목표를 도착까지 유지하는 시간 추상화 (R1–R3와 함께)

- **이벤트 구동 waypoint (설정만):** `ROBOT_DECISION_ON_EVENTS_WAYPOINT = True`, `ROBOT_DECISION_MAX_S_WAYPOINT = 16–20`, `ROBOT_WAYPOINT_RANGE_M = 40`(위험구역 반경 36–50 m를 한두 결정으로 가로지르도록). 5.5절에서 같은 무작위성이라도 유지형이 내부 체류 80–97%, 4 s형이 5–43%였다. 한 순환이 3–5 결정으로 줄어 TD 전파 깊이가 짧아지고, 에피소드당 update가 줄어 H8도 완화된다. 원래 비-easy 설정이 이것이었으므로 easy에서 고정 4 s로 바꾼 결정을 되돌리는 셈이다.
- 주의: 현재 구현은 한 로봇의 도착·막힘이 팀 전체 재결정을 부르는 동기식 근사다. 엔트로피가 결정 단위로 붙는 semi-MDP 목적이 된다는 10/07 검토의 지적도 그대로 적용된다. 결정 간격이 바뀌므로 관측 history의 한 칸이 몇 초인지 모르는 문제가 커진다. 경과 시간(초)을 history와 함께 넣는 것을 권한다.
- warm-up 탐색도 같은 방식으로 바꾼다: `exploration_action`을 4 s마다 새로 뽑지 말고 '무작위 목표를 도착 또는 시간 한도까지 유지'로(5.5절의 `유지 무작위`).
- 또는 n-step(5 결정 ≈ 20 s) 목표. replay의 prev/next 연결로 구현 가능하다.
- 가장 강한 형태: Spatial Action Maps(Wu et al. 2020)처럼 mid 지도 셀 × {off, guide}에 대한 dense Q를 완전 합성곱으로 내고 최댓값 셀로 navmesh 이동. 행동이 관측과 공간 정렬된다. 이산 행동이 되므로 double DQN/n-step 또는 이산 SAC, 팀원 목표점은 지도 채널(Spatial Intention Maps), 매크로 행동 replay(Xiao et al. 2019).

### R5. shuttle 사전정책 (진단과 가속)

- 워커 에피소드의 20–30%를 shuttle(작은 잡음)로 돌려 같은 replay에 넣는다(RLPD식 50:50 샘플링이면 더 강함), 초기 BC 항. 또는 JSRL(앞 h 결정만 shuttle), 또는 잔차(shuttle 목표 + 학습 잔차).
- 진단 의미: 이 run이 shuttle 수준을 유지·개선하면 critic은 좋은 행동의 가치를 학습할 수 있고 탐색이 주된 병목이다. 45k로 돌아가면 critic이 좋은 데이터가 있어도 행동 가치를 표현하지 못하는 것이며 R2·R3와 critic 쪽 수정이 필수다.
- shuttle은 군중 위치를 보지 않는다. 사람이 어디 있는지 보는 학습 정책이 shuttle을 넘는 것은 연구 주장으로도 자연스럽다.

### R6. 프레임워크 대안: MAPPO

SAC의 actor는 critic의 ∂Q/∂a만 본다. 5장처럼 실제 효과는 큰데 critic이 압축하면 actor 신호가 사라진다. PPO/MAPPO는 GAE(λ 0.95–0.98)로 실제 return을 쓰므로 critic이 압축해도 편향 없는 신호를 받는다(분산은 큼). 학습기 병목이 없어 에피소드/시간이 약 2배다. MAPPO는 협동 MARL 벤치마크에서 off-policy 기법과 같거나 나은 표본 효율을 보고했다(Yu et al. 2022). Run L이 실패할 때의 대안 축으로 둔다. `PPO_FE_RV`의 코드를 출발점으로 쓸 수 있다.

### 하지 말 것 / 우선순위가 낮은 것

- α 하한만 낮추기: critic 행동 차이가 잡음 속이라 잡음 기울기를 따라 붕괴할 위험(10/05의 α 0.001 붕괴 이력). R1–R2 이후 판단.
- 네트워크 확대, LR 변경, 장기 ε: 원인과 무관. 진행 중인 `Epsilon` run은 4 s 무작위 행동을 더 섞을 뿐이라 깊은 진입을 만들지 못할 것으로 예상한다[Likely](5.5절).
- 행동 모델 수정: 연구 타당성을 해친다. 위 수정은 모두 학습·인터페이스 쪽이다. 단, 학습 가능성 진단용으로 순응률·정보 확률을 결정적으로 바꾼 run(FE_RV3식 군중)은 원인 분리에 유용하며, 효과가 확인되면 UED 축으로 순응 파라미터를 쉬운 값에서 현실 값으로 옮기는 커리큘럼이 가능하다.

### 실행 계획과 판정 기준

| 단계 | 내용 | 비용 | 통과 기준 |
|---|---|---|---|
| 0 | 고아 프로세스 정리. 검증을 27쌍(9 시나리오 × 3 시드)으로, 최소한 정책 조건만이라도. 진단 로그: 로봇별 위험구역 내부 시간·최대 깊이, 추종자 수, 로봇 간 평균 거리 | 수 시간 | - |
| 1 | **Run L:** R4(이벤트 구동, 유지형 warm-up) + R2a + R2b + R3 + critic privileged 정보 수신·인지 채널(R1 일부), 새 LOG_DIR | 구현 약 1일 + 학습 1일 | 최소 신호: ep500–1000 검증이 현재 정체선(44–47k)을 분명히 넘음. 돌파: ep1000 검증이 shuttle 수준(≤ 34k, 끔 대비 −43%)에 도달. 로봇 평균 속도 > 0.9 m/s, 체류자 사람-스텝 −50% 이상. 이 폴더의 `true_adv.py`를 새 체크포인트에 다시 돌려 critic Q sd / 실제 sd 비율이 지금(약 1/10)보다 크게 개선 |
| 1′ | **Run P (병렬 가능):** 현 설정 + R5(shuttle 25%) | 구현 수 시간 + 1일 | 34k 이하 유지·개선이면 탐색 병목 확정. 45k 복귀면 critic 병목 확정 |
| 2 | L 성공 시 ablation: −R4, −R2a, −R3, −privileged 순. 추종자 수 명시 입력은 +R1로 별도 비교 | 각 1일 | 기여 분해 |
| 3 | L 실패 시 spatial action map + 매크로 행동(R4의 강한 형태) 또는 R6(MAPPO) | 수일 | - |
| 4 | 어려움 복원(부분관측 → 위험구역 크기 → UED) | | 10/10 문서 순서 |

## 11. 선행연구와 적용 판단

원문 또는 공식 페이지를 확인한 것만 적었다. "적용"은 이 프로젝트에 대한 판단이다.

### 11.1 신용 할당과 보상 설계
| 연구 | 핵심 | 적용 |
|---|---|---|
| Ng, Harada, Russell (ICML 1999), [Policy invariance under reward transformations](https://mlanthology.org/icml/1999/ng1999icml-policy/) | γΦ(s′) − Φ(s) 형태만 최적 정책 보존 | R2 이론 근거. 가변 hold에서는 γ^k |
| Devlin, Yliniemi, Kudenko, Tumer (AAMAS 2014), [Potential-based difference rewards](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2014/aamas/p165.pdf) | 차이 보상 + 잠재함수, 다중 에이전트 Nash 균형 보존, 수렴 최대 23.8배 | R2a(로봇별)의 근거 |
| Arjona-Medina et al. (NeurIPS 2019), [RUDDER](https://arxiv.org/abs/1806.07857) | 지연 보상 재분배로 Q 추정을 즉시 보상 평균으로 | R2가 부족할 때. 구현 비용 큼 |
| Cheng, Kolobov, Swaminathan (NeurIPS 2021), [HuRL](https://arxiv.org/abs/2106.02757) | 휴리스틱 가치로 보상을 바꾸고 할인을 줄여 지평 단축, 점차 원 문제로 | shuttle 가치를 휴리스틱으로 쓰는 R5 변형 |
| Bellemare et al. (AAAI 2016), [Increasing the action gap](https://arxiv.org/abs/1512.04860) | 행동 간 가치 차가 작으면 근사 오차가 정책을 망침 | 4–5장 진단의 이론 배경 |

### 11.2 시간 추상화와 공간 행동
| 연구 | 핵심 | 적용 |
|---|---|---|
| Xiao, Hoffman, Amato (CoRL 2019), [Macro-action-based deep MARL](https://arxiv.org/abs/2004.08646) | 길이가 다른 매크로 행동의 분산·중앙 Q 학습, 전용 replay | R4 |
| Xiao, Tan, Amato (NeurIPS 2022), [Asynchronous actor-critic for MARL](https://arxiv.org/abs/2209.10113) | 매크로 행동 비동기 actor-critic, 실제 로봇 | R4 비동기 확장 |
| Wu et al. (RSS 2020), [Spatial Action Maps](https://spatial-action-maps.cs.princeton.edu/) | 행동 = 입력 지도와 정렬된 이동 목표 픽셀, 조향 명령 대비 훨씬 빠른 학습 | R4의 가장 강한 형태 |
| Wu et al. (ICRA 2021), [Spatial Intention Maps](https://spatial-intention-maps.cs.princeton.edu/) | 팀원 의도를 지도에 그려 공유 → 협동 개선 | 팀원 목표점을 지도 채널로 |
| Nguyen et al. (2020), [Hierarchical RL for swarm shepherding](https://arxiv.org/abs/2004.11543) | 몰기·끌기 하위 정책 분리 | 진입/유도 분해 |
| Zhi & Lien (2020), [Learning to herd amongst obstacles](https://arxiv.org/abs/2005.09476) | 장애물 환경 양치기 DRL, 규칙 기반보다 우위 | 같은 계열 |
| (2025), [Decentralized shepherding of non-cohesive swarms](https://arxiv.org/abs/2511.21405) | 저수준 몰이(PPO) + 분산 목표 할당, 최소 시나리오에서 배워 다중으로 전이 | 응집하지 않는 군중과 가장 가까운 설정 |

### 11.3 사전정책 활용
| 연구 | 핵심 | 적용 |
|---|---|---|
| Uchendu et al. (ICML 2023), [Jump-Start RL](https://proceedings.mlr.press/v202/uchendu23a.html) | 안내 정책이 앞부분을 굴려 시작 상태 커리큘럼, 탐색 표본 복잡도 지수→다항 | R5 |
| Ball et al. (ICML 2023), [RLPD](https://arxiv.org/abs/2302.02948) | 오프라인 50% + 온라인 50%, critic LayerNorm, 앙상블 | R5 |
| Silver et al. (2018), [Residual Policy Learning](https://arxiv.org/abs/1812.06298) | 기존 제어기 위 잔차 학습, 장기·희소 과제에서 RL 단독보다 우위 | R5 |

### 11.4 critic과 표현
| 연구 | 핵심 | 적용 |
|---|---|---|
| Pinto et al. (RSS 2018), [Asymmetric actor critic](https://arxiv.org/abs/1710.06542) | critic은 전체 상태, actor는 관측 | R1 |
| Liu et al. (NeurIPS 2018), [CoordConv](https://arxiv.org/abs/1807.03247) | 합성곱의 좌표 변환 약점, 좌표 채널로 해결 | R3 |
| Farebrother et al. (ICML 2024), [Stop Regressing (HL-Gauss)](https://proceedings.mlr.press/v235/farebrother24a.html) | 가치 회귀를 분류로, 잡음 목표에 강함 | 후순위 |
| Nikishin et al. (ICML 2022), [Primacy bias](https://proceedings.mlr.press/v162/nikishin22a.html) | 초기 데이터 과적합, 마지막 층 리셋 | [Speculation] 수준 저비용 시험 |

### 11.5 MARL·혼합 행동
| 연구 | 핵심 | 적용 |
|---|---|---|
| Yu et al. (NeurIPS 2022), [MAPPO](https://papers.nips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) | 중앙 가치함수 PPO가 협동 MARL에서 off-policy와 같거나 나은 표본 효율 | R6 |
| Peng et al. (NeurIPS 2021), [FACMAC](https://arxiv.org/abs/2003.06709) | 공동 행동 공간의 중앙 정책 기울기 | `current`가 이 방향, 차이 없었음 |
| Delalleau et al. (2019), [Hybrid SAC](https://arxiv.org/abs/1912.11077) | 이산·연속 혼합 SAC | 현 구현 근거, 온도 분리 미적용 |

### 11.6 로봇 유도 대피
| 연구 | 핵심 |
|---|---|
| [Multi-Robot-Guided Crowd Evacuation: Two-Scale Modeling and Control](https://arxiv.org/abs/2302.14752) | 군중을 연속체로, 소수 로봇의 미시 상호작용으로 밀도를 목표로 |
| [Multi-Robot-Assisted Human Crowd Evacuation using Navigation Velocity Fields](https://arxiv.org/abs/2209.09795) | 속도장 기반 제어 |
| [Adversarial RL for evacuation guidance robots in fire scenarios](https://www.researchgate.net/publication/386998162_Adversarial_Reinforcement_Learning_for_Enhanced_Decision-Making_of_Evacuation_Guidance_Robots_in_Intelligent_Fire_Scenarios) | MARL + 적대적 RL, 군집·기억이 있는 인간 모델 |

로봇 유도 대피에서 DRL 사례는 적고 대부분 제어·최적화다. 개방 경계·체류·확률적 순응을 함께 다룬 DRL 사례는 찾지 못했다.

## 부록 A. R2 구현 스케치 (`sim/rewards.py`, 가칭 `rew-v6-pbrs`)

```python
# __init__ 끝
self.g_step = cfg.gamma() ** (1.0 / int(cfg.ACTION_SCALE))
self.w_prog = float(cfg.REWARD_W_SHAPE_PROGRESS)
self.w_appr = float(cfg.REWARD_W_SHAPE_APPROACH)
self._phi_prog = self._phi_progress(model)
self._phi_appr = [self._phi_approach(model, rb) for rb in model.robots]

def _phi_progress(self, model):
    zone = model.danger_zone
    s = sum(min(1.0, model.escape_distance(a.xy) / self.d_ref)
            for a in self._alive(model) if zone.contains(*a.xy))
    return -s / self.n_ref

def _phi_approach(self, model, rb, cap=50.0):
    zone = model.danger_zone
    d = [math.hypot(a.xy[0] - rb.xy[0], a.xy[1] - rb.xy[1])
         for a in self._alive(model)
         if zone.contains(*a.xy) and a.type != 0
         and not getattr(a, "informed_by_robot", False)]
    return -(min(min(d), cap) / cap) if d else 0.0

# step() 안, 기존 항 계산 뒤
p = self._phi_progress(model)
shape_team = self.w_prog * (self.g_step * p - self._phi_prog); self._phi_prog = p
shape_robot = []
for j, rb in enumerate(robots):
    q = self._phi_approach(model, rb)
    shape_robot.append(self.w_appr * (self.g_step * q - self._phi_appr[j]))
    self._phi_appr[j] = q
# shape_team → 학습용 step_rewards, shape_robot[j] → robot_penalties[j].
# res.total_reward·검증 지표에는 넣지 않고 shape_progress / shape_approach로 따로 기록.
```

hold 동안의 할인 적산이 F를 γ^k Φ(s_k) − Φ(s_0)로 바꾸므로 `learn/sac.py`의 `discounted_return`과 γ^k 부트스트랩을 그대로 쓴다.

### A.1 구현에서 바뀐 점 (2026-10-15, 실제 `rew-v6-pbrs`)

위 스케치의 `_phi_approach`(가장 가까운 미접촉자까지 거리)는 그대로 쓰지 않았다. 처음 구현(navmesh 삼각형 중심 경유 거리 + 최근접)을 검증 시나리오 2개 × shuttle·무작위 로봇으로 재보니, |F| > 0.6인 step이 전체의 19–27%였고 그 jump가 분산의 98–99%를 차지했다. 원인은 두 가지다. (1) 삼각형 중심 경유 거리는 로봇이나 사람이 삼각형 경계를 넘을 때 불연속이다. 직선거리로 바꾸면 jump step이 6–11%로 줄었다. (2) 최근접 하나만 쓰면 그 사람이 위험구역 경계를 넘나들거나, 스스로 위험을 감지하거나, 접촉될 때마다 다음 사람의 거리로 튄다. 직선거리로 바꿔도 남는 jump 분산 77–94%는 이쪽이다.

실제 구현: Φ_j = 1 − exp(−S_j/κ), S_j = Σ_{위험구역 안 미접촉 i} max(0, 1 − ‖로봇 j − i‖/50 m). 미접촉 = 추종 중 아님, 로봇에게 정보 받지 않음, 위험 기억(hazard_memory) 없음. 사람 한 명이 집합에 들어오거나 나가도 Φ_j는 최대 1/κ만 변한다(`tests/test_madrl.py` ShapingTest). κ = 10은 `shaping_kernel.py`로 정했다: κ=5는 Φ > 0.9 포화가 33–44%, κ=10은 12–25%이고, 움직이는 로봇의 4 s 접근항 sd 1.7–1.8 대 서 있는 로봇 0.6이다. 서 있는 로봇의 결정당 접근항 sd는 2.0에서 0.4로 줄었다. 계산 비용은 step당 0.05 ms다(model.step 약 165 ms). Φ ≥ 0이어서 매 step −(1−g)Φ의 작은 drift가 생기고, 에피소드 합 `shape_approach`는 음수로 기록된다. 전체가 −Φ(s_0)로 telescoping하므로 최적 정책은 바뀌지 않는다. 진행 potential(`escape_distance`)은 삼각형 단위 상수라 계단형이지만, 사람별 기여가 1/N_ref로 작아 그대로 두었다(4 s당 sd 0.8–1.7).

## 부록 B. 재현

```bash
cd docs/review_20261013
python3 wandb_dump.py wandb_dump     # W&B history 다운로드(~/.netrc 인증 사용, 스칼라만)
python3 wandb_summary.py             # run별 검증·학습·learner 요약 (results/wandb_runs_summary.txt)
CKPT=~/Log_SAC_UED6_easy_fullinfo/sac_checkpoint_ep_2400.pth
OMP_NUM_THREADS=4 python3 critic_probe.py $CKPT
python3 critic_consistency.py        # ep2300·ep2400 체크포인트 사용
python3 obs_probe.py $CKPT
# 반사실: 시나리오 번호(0 dotonbori_0, 3 covent_garden_0, 8 hongdae_2), t0(결정 수), 체크포인트, 출력(절대경로)
H=400 M=3 python3 true_adv.py 0 10 $CKPT $PWD/results/adv/s0_t10.json
T0S=6,20,40 M=3 python3 shaping_probe.py 0 $CKPT $PWD/results/shp/s0.json
python3 explore_cov.py 0 random4s $PWD/results/exp/s0_random4s.json   # random4s | random_commit | beacon_edge
python3 cf_table.py; python3 cf_anova.py; python3 analyze_shp.py
DEC=100 python3 follower_decode.py 0 $CKPT $PWD/results/fol/s0.npz        # 레벨 0, 3, 5, 8
python3 follower_fit.py
```

probe 스크립트는 프로젝트 루트로 chdir하므로 출력 경로는 절대경로로 준다. 반사실 한 건(시나리오 하나 × t0 하나)은 CPU 1개로 15–45분 걸린다. Run L의 판정에는 같은 `true_adv.py`를 새 체크포인트에 돌리면 된다.
