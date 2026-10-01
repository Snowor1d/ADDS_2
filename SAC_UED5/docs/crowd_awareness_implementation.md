# 군중 위험 인지 모델: 구현과 근거

설계는 `crowd_awareness_design.md`에 있다. 이 문서는 무엇을 어떻게 구현했고,
각 선택이 어떤 문헌에 근거하며, 어떤 수치로 확인했는지를 남긴다. 논문의
방법론 절에 그대로 옮길 수 있는 수준을 목표로 했다.

## 1. 인지 상태 기계

### 구현

`agent.py`의 `CrowdAgent`가 네 상태를 갖는다.

```
unaware → cued → milling → acting
```

`update_awareness`가 매 스텝 호출되어 상태를 전이시킨다. `cued`는 즉시
`milling`으로 넘어가므로 실질적으로 세 상태다. 분리해 둔 이유는 단서 수신과
확인 탐색이 개념적으로 다른 단계이고, 나중에 둘 사이에 다른 처리를 넣을 수
있게 하기 위해서다.

### 근거

Lindell, M. K., & Perry, R. W. (2012). The Protective Action Decision Model:
Theoretical Modifications and Additional Evidence. *Risk Analysis*, 32(4),
616-632.

PADM은 보호 행동을 단일 판단이 아니라 단계로 본다. 단서 수신, 주의, 이해,
위협 인식, 보호 행동 결정이고 각 단계에서 사람이 멈출 수 있다. 위험 인지를
이진 플래그로 두면 이 구조가 사라지고, 특히 사라지는 단계가 로봇이 실제로
개입할 수 있는 단계다.

Wood, M. M., Mileti, D. S., Bean, H., Liu, B. F., Sutton, J., & Madden, S.
(2018). Milling and Public Warnings. *Environment and Behavior*, 50(5),
535-566.

밀링을 물리적 모임이 아니라 방향을 구하는 심리 상태로 정의한다. 더 긴 경보
메시지가 확인 탐색 욕구를 줄여 반응 지연을 단축한다는 것을 보인다. 이 구현에서
로봇이 밀링을 단축하는 것이 같은 기제다.

### 확인

`test_prior_informed_start_milling_not_acting`. 사전 인지자는 `milling`으로
시작하고 `acting`으로 시작하지 않는다. 경보를 들었다는 것과 움직이기 시작했다는
것은 다르고, 그 사이 간격이 로봇이 작업하는 대상이다.

## 2. 정보 채널 셋

### 구현

PADM의 단서 출처 분류를 따른다.

**환경 단서.** `_sense_hazard`. 구역 안이면 스텝당 `AWARENESS_P_INSIDE`,
시야 안이면 `AWARENESS_P_VISIBLE`. 시야 판정은 로봇이 쓰는 것과 같은
가시성 아틀라스를 쓰므로 블록 뒤의 위험은 감지되지 않는다.

**사회적 단서.** `_social_cue`. 시야 안에 `acting` 상태인 이웃이 있으면
확률적으로 `cued`가 된다. 이웃 수에 대해 포화한다. 한 명이 뛰는 것과 열 명이
뛰는 것은 다르다.

**권위 단서.** 로봇의 유도 반경 안이면 즉시 `cued`가 되고 밀링 시간이
`MILLING_ROBOT_SPEEDUP` 배로 줄어든다. 정책이 통제하는 유일한 채널이다.

### 근거

Mileti, D. S., & Sorensen, J. H. (1990). *Communication of Emergency Public
Warnings: A Social Science Perspective and State-of-the-Art Assessment*.
Oak Ridge National Laboratory, ORNL-6609, 145 pp.

경보 시스템과 경보 반응 연구 200편 이상을 종합한 FEMA 제출 보고서다. 실험
논문이 아니므로 인용 시 "종합했다"로 써야 한다. 경보의 출처, 채널, 일관성,
신뢰성, 정확성, 이해 가능성, 반복 빈도가 반응을 좌우한다는 것이 결론이고,
효과적 메시지의 다섯 요소(출처, 위험, 위치, 지침, 시간)는 실무에서
"Mileti Model"로 불린다.

Nilsson, D., & Johansson, A. (2009). Social influence during the initial phase
of a fire evacuation. *Fire Safety Journal*, 44(1), 71-79.

주변이 움직이지 않으면 개인도 움직이지 않는다는 것을 보인다. 사회적 채널을
주 경로로 두고 포화 형태로 모델링한 근거다.

### 확인

`test_word_of_mouth_spreads_from_a_seed`. 감각 신호가 없는 위험에서 사전
인지자 15%로 시작해도 인지자 수가 늘어난다. 사회적 전파가 유일하게 작동하는
경로인 상황이다.

## 3. 지각 가능성과 감각 문턱

### 구현

`perceptibility ∈ [0, 1]`이 레벨 속성이고 UED 설계 공간 변수다.
`PERCEPTIBILITY_SENSORY_FLOOR = 0.25` 아래에서는 환경 단서 채널이 아예 닫힌다.
그 위에서는 문턱 위 거리에 비례해 감지 확률이 올라간다.

### 이것이 문턱인 이유

처음에는 선형으로 구현했다. 지각 가능성 0.05에 스텝당 0.0175를 주는 형태였다.
측정해보니 800스텝 에피소드에서 감지가 거의 확실했고, 지각 가능성 0.05와 0.90의
최종 상태가 같았다. 축의 낮은 쪽이 통째로 높은 쪽으로 붕괴했다.

물리적으로도 틀렸다. 부취제를 넣지 않은 가스 누출은 낮은 확률로 냄새가 나는
것이 아니라 냄새가 나지 않는다. 확률이 아니라 채널의 유무다.

문턱을 넣은 뒤 800스텝에서 27명 중 지각 가능성 0.05는 17명이 구역 안에
남고 0.90은 1명이 남는다. 인지 절반 도달 시각도 135스텝과 36스텝으로
갈린다.

### 근거

Jin, T. (1978). Visibility through fire smoke. *Journal of Fire and
Flammability*, 9, 135-155.

Frantzich, H., & Nilsson, D. (2003). 연기 속 이동 실측.

연기는 시야와 보행 속도를 직접 떨어뜨린다. 위험이 감각에 도달하는 정도가
위험마다 다르다는 것의 대표 사례다. 반대편 사례인 무취 가스나 구조적 붕괴
위험에는 감각 경로가 없다.

## 4. 전 이동 지연

### 구현

`_draw_premovement`가 로그정규 분포에서 뽑는다. 현재 중앙값
`PREMOVEMENT_MEDIAN_STEPS = 16.7`, 로그 표준편차 `PREMOVEMENT_SIGMA = 0.7`.
실외 도로 대피 VR 자료에 따른 2026-09-23 잠정 보정이다. 수치의 계산,
적용 범위와 원문 인용은 `outdoor_human_calibration.md`에 기록했다.
밀링 중 주변에 `acting` 이웃이 많거나 로봇이 가까우면 소진 속도가 빨라진다.

### 근거

Purser, D. A., & Bensilum, M. (2001). Quantification of behaviour for
engineering design standards and escape time calculations. *Safety Science*,
38(2), 157-182.

PD 7974-6 (BSI). 전 이동 시간 분포, 최초 점유자와 최후 점유자의 구분.

화재 안전 공학은 대피 시간을 전 이동 시간과 이동 시간으로 나누고, 전 이동
시간을 상수가 아니라 넓고 오른쪽으로 치우친 분포로 본다. 꼬리가 중요한 이유는
대피 시간을 결정하는 것이 마지막에 움직이는 사람이기 때문이다. 평균을 쓰면
설계 대상 자체가 사라진다.

### 확인

`test_the_delay_is_lognormal_and_right_skewed`. 4000회 추출에서 중앙값이
설정값의 15% 이내이고, 평균이 중앙값보다 크며, 최대값이 중앙값의 3배를 넘는다.

## 5. 위험 지식의 국소성

### 구현

보행자마다 `hazard_memory`를 갖는다. 자기가 감지한 지점들의 목록이고 최대
`HAZARD_MEMORY_MAX_POINTS = 12`개다. `hazard_repulsion`이 그 기억으로부터
밀어낸다. 실제 구역이 아니라 기억으로부터다.

`PERCEPTIBILITY_GLOBAL_CUE = 0.7` 이상이면 감지 시 구역 중심도 함께 기억한다.
연기 기둥처럼 멀리서도 개략적 위치를 알려주는 위험이다. 그 아래에서는 접촉한
지점만 기억한다.

### 근거

Golledge, R. G. (ed.) (1999). *Wayfinding Behavior: Cognitive Mapping and
Other Spatial Processes*. Johns Hopkins University Press.

Siegel, A. W., & White, S. H. (1975). The development of spatial
representations of large-scale environments. *Advances in Child Development
and Behavior*, 10, 9-55.

Lynch, K. (1960). *The Image of the City*. MIT Press.

Sime, J. D. (1983). Affiliative behaviour during escape to building exits.
*Journal of Environmental Psychology*, 3(1), 21-41.

공간 인지 문헌은 랜드마크, 경로, 측량 지식의 순서를 말한다. 대부분의 사람은
지도 같은 전역 지식이 아니라 경로 지식만 갖는다. 대피 연구는 사람들이 최단
출구가 아니라 익숙한 경로로 가고, 공간을 잘 모를수록 국소적으로 보이는 정보에
더 의존한다고 보고한다. 도심을 지나가는 보행자가 그 블록 구조의 지도를 갖고
있다고 가정할 근거가 없다.

### 이 선택이 만드는 것

아는 경계에서 도망친 사람이 블록을 돌아 한 번도 본 적 없는 같은 위험의 다른
면으로 걸어 들어간다. 결함이 아니라 현실이고, 로봇이 막아야 할 상황이다.
전역 지식을 가정하면 이 상황이 사라지고 로봇의 역할이 그만큼 줄어든다.

### 확인

`test_hazard_knowledge_is_local`, `test_a_low_perceptibility_hazard_conveys_no_global_picture`.
지각 가능성이 문턱 아래인 위험에서는 아무도 구역 중심을 알지 못한다.

## 6. 측지 탈출장은 로봇의 것이다

### 구현

`which_goal_agent_want`는 `nearest_safe_goal`도 `mesh_danger`도 호출하지
않는다. 테스트가 소스를 검사해 이를 고정한다.

### 이유

이전 구현에서 군중이 측지 탈출장을 내려갔고, 그 결과 로봇 없이도 6회 중 6회가
스스로 비워졌다. 정책이 자기가 하지 않은 일로 채점되는 상태였다. 측지장은
전체 배치에 대한 시뮬레이터의 지식이고 군중이 가질 수 없는 것이다.

### 확인

`test_the_crowd_never_uses_the_geodesic_escape_field`.

## 7. UED 설계 공간과 퇴화 감시

`UED_DANGER_PERCEPTIBILITY = (0.05, 0.95)`와
`UED_PRIOR_INFORMED_FRACTION = (0.0, 0.3)`이 커리큘럼 변수다. 위험의 속성이지
보행자의 속성이 아니므로 SAMPLR의 CICS 논증이 보호하는 고정 배포 분포에 속하지
않는다.

감시가 필요하다. 커리큘럼은 기하를 어렵게 만드는 대신 정보를 굶겨서 난이도를
올릴 수 있다. 개체군의 `perceptibility` 분포가 하한으로 몰리면 그렇게 하고
있다는 뜻이다. 이 분포를 텐서보드에 올리고, 몰리면 범위를 좁히거나 축을 닫는다.

Jiang, M., Dennis, M., Parker-Holder, J., Foerster, J., Grefenstette, E., &
Rocktäschel, T. (2022). Grounding Aleatoric Uncertainty for Unsupervised
Environment Design. *NeurIPS*. (SAMPLR, CICS)

## 8. 로봇 관측에 인지 상태를 넣지 않는 이유

실제 유도 로봇은 누가 아직 모르는지 볼 수 없다. 보는 것은 사람들의 위치와
움직임뿐이고, 누가 아직 반응하지 않았는지는 거기서 추론해야 한다. 관측에
넣으면 현장에 없는 센서를 가정하게 되고 제로샷 전이 주장이 그만큼 약해진다.

`ROBOT_STATE_DIM`은 7로 유지된다.

## 9. 부수적으로 고친 결함

### 이웃 선택 루프

`which_goal_agent_want`의 이웃 선택 루프에서 `max_score`가 갱신되지 않았다.
모든 이웃이 `score > -99999`를 통과하므로 `follow_agent_id`는 항상 목록의
마지막 이웃이 됐다. "가장 믿을 만한 이웃을 고른다"는 로직이 한 번도 작동한 적이
없다. `test_the_most_credible_neighbour_is_chosen`이 고정한다.

## 9b. 구현 중 드러난 다른 결함

### 도로 정렬 위험 구역이 맵 밖으로 나갔다

거리 구간은 크롭 대각선 길이만큼 그어지므로 그 절반이 맵보다 길 수 있다.
100 m 크롭에서 y로 0부터 141.7까지 뻗은 위험 구역이 나왔고, 면적 1581 중
1118만 세계 안에 있었다. 커리큘럼이 요청한 면적 비율이 실제 레벨의 것이 아니게
되고, 밖으로 나간 부분은 군중이 들어갈 수도 빠져나올 수도 없는 위험이다.

생성 후 맵에 들어갈 때까지 긴 축을 줄이도록 고쳤다. 20개 시드 세 크기에서
60개 표본 전부가 크롭 안에 들어간다. `test_the_hazard_is_visible_in_the_raster`가
래스터에 칠해진 픽셀 수를 맵 안쪽 면적과 대조하면서 잡아냈다.

### 테스트의 시드 민감성

레벨 생성기가 `perceptibility`와 `prior_informed_fraction`을 같은 난수 스트림에서
뽑게 되면서, 같은 시드가 다른 직물을 만든다. 고정 좌표에 의존하던 테스트 세 개가
그 때문에 실패했다. 하나는 (0.3, 60)이 더 이상 자유 공간이 아니어서 보행자가
블록 안에 놓였고, 그것이 벽 반발력이 작동하지 않는 것처럼 보였다.

교훈은 시뮬레이션 상태에 의존하는 테스트가 좌표나 시드에 의존하면 안 된다는
것이다. 조건을 만족하는 지점을 찾게 하거나, 위험 구역의 모양과 크기를 명시적으로
고정해야 한다. 세 테스트 모두 그렇게 고쳤다.

### 보상 성분 이름이 등록되지 않았다

재진입 페널티 `reward_n`을 보상 합계와 전송 딕셔너리에는 넣었는데
`REWARD_COMPONENT_NAMES`에 넣지 않았다. 워커가 전이를 보낼 때마다
`KeyError('reward_n')`이 났고, 단 하나의 전이도 리플레이 버퍼에 도달하지
못했다.

가장 나쁜 종류의 실패다. 예외 추적이 찍히지 않고, 학습 루프는 계속 돌고,
로그는 정상으로 보인다. 아무것도 학습하지 않으면서 그렇다.

단위 테스트는 이것을 구조적으로 잡을 수 없다. 에피소드별 성분 딕셔너리를
만드는 것은 워커 루프뿐이고 테스트는 그 경로를 타지 않는다. 학습 루프를
실제로 돌려봐야만 드러난다.

교훈은 단위 테스트가 전부를 덮지 못한다는 것이다. 구성 요소를 각각 확인하는
것과 전체가 함께 도는 것을 확인하는 것은 다르고, 후자를 대신할 수 있는
단위 테스트는 없다. 큰 변경 뒤에는 스모크런이 선택이 아니다.

## 10. 측정 요약

로봇 없이, 40명, 140 m 격자, 800스텝.

| 지각 가능성 | 구역 내 시작 | 구역 내 종료 | 인지 절반 도달 |
|---|---|---|---|
| 0.05 | 27 | 17 | 135 스텝 |
| 0.40 | 27 | 2 | 45 스텝 |
| 0.90 | 27 | 1 | 36 스텝 |

## 11. 아직 하지 않은 것

- 정보 확산 곡선이 S자인지 정량 확인
- 로봇의 기여가 밀링 단축에서 오는지 방향 제공에서 오는지 분리 측정
- 전 이동 시간 분포를 문헌 보고 범위와 대조
- 퇴화 지표를 텐서보드에 연결
