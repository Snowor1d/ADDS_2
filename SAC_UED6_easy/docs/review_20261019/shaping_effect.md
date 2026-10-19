# Reward shaping 변경 효과 분석 — 2026-10-19

이 문서는 [최신 run 분석](report.md)에서 분리한 shaping 전용 분석이다. 같은 W&B 스냅샷을 사용한다. 최신 run은 학습 ep5530, 완료 검증은 ep5000까지다. 관련 run은 [Approach20](https://wandb.ai/437snowcap-sungkyunkwan-university/adds-sac-ued6-261011ENV/runs/d774mrrb), [Progress50](https://wandb.ai/437snowcap-sungkyunkwan-university/adds-sac-ued6-261011ENV/runs/cx8bbli5), [Approach20+Progress150](https://wandb.ai/437snowcap-sungkyunkwan-university/adds-sac-ued6-261011ENV/runs/vsa06hsg), [EventDecision](https://wandb.ai/437snowcap-sungkyunkwan-university/adds-sac-ued6-261011ENV/runs/fi2kcteu)다.

## 1. 결론

- **[Certain] Progress150을 포함한 최신 설정은 Approach20보다 안쪽 진입·추종자 수·재진입 지표가 나아졌지만, 전체 위험 노출의 평균 차이는 작고 체류자 노출 개선은 일관되지 않다.** 같은 ep1000–4000의 7개 완료 검증에서 전체 노출 -0.56%, 체류자 노출 +3.26%, 재진입 횟수 -6.79%다. 이 7시점은 같은 정책 학습 경로와 과제를 반복 평가한 것으로 독립적인 학습 seed 7개가 아니다.
- **[Certain] 큰 총보상 차이를 shaping 성과라고 볼 수 없다.** ep3001–4000 학습 보상 차이 +140.70 중 117.23(83.3%)은 이전 설정에만 있던 path·reentry 비용의 규모에 해당한다. 공통 항목(person_time+collision)의 차이는 +23.47이다. 서로 다른 궤적의 평균에 대한 회계적 분해이며, 같은 궤적에서의 인과 효과 추정은 아니다.
- **[Certain] 현재 Approach는 접촉·추종 성공 자체에 양의 추가 보너스를 주는 함수가 아니다.** 미접촉 대상에 접근할 때 potential이 커지지만, 사람이 추종을 시작하면 대상에서 빠져 potential이 내려갈 수 있다. production 함수로 만든 단순 기하 예시에서 이 순간의 shaping은 음수다.
- **[Likely] shaping이 행동 분포를 바꿀 여지는 있지만, 지금 자료만으로 P150의 추가 효과나 A20과의 시너지를 입증하지 못했다.** 더 큰 계수로 올리는 근거도 아직 부족하다.

## 2. 각 run에서 실제로 달랐던 것

| 변형 | Approach | Progress | 기본 path | 기본 reentry | 비교 제약 |
| --- | ---: | ---: | ---: | ---: | --- |
| Approach20 | 20 | 0 | 1 | 2 | obs-v5-clock, expectation, UPT2, 처음부터 시작 |
| Progress50 | 0 | 50 | 1 | 2 | 현재 metadata에 ep2000 resume; 완료 검증은 ep1500까지 |
| 최신 조합 | 20 | 150 | 0 | 0 | obs-v5-clock, expectation, UPT2, 처음부터 시작 |
| 이전 EventDecision | 0 | 0 | 1 | 2 | obs-v4-lean, Gumbel, UPT1 등도 다름 |

세 shaping run의 person_time weight는 5, approach cap은 50 m, kappa는 10이다. 평가 `total_reward`에는 shaping을 넣지 않지만 기본 path/reentry 항목은 들어간다. 따라서 shaping이 평가 보상에 직접 더해져 점수가 오른 것은 아니며, **기본 보상 정의가 바뀌어** 비교 기준이 달라졌다.

Progress50과 Approach20의 현재 공개 config는 shaping·run 이름 외에 **OSM_CACHE_DIR 경로**가 다르다. 이 경로도 dynamics fingerprint에 포함되므로, fingerprint 차이만으로 사람의 행동 동역학이 달라졌다고 단정하면 안 된다. 원격 cache의 동일성·resume 이전 설정·실행 코드 동일성은 확인하지 않았다. 현재 config를 이전 모든 로그의 확정된 설정으로 소급 적용하지 않는다. 학습 로그에서 progress shaping 평균이 약 +49~53, approach는 0인 것은 직접 확인했다.

## 3. 에피소드를 맞춘 검증 비교

고정 과제 9개, 결정적 정책의 위험 구역 사람-스텝 평균, 낮을수록 좋음.

| 평가 ep | 이전 EventDecision | Approach20 | Progress50 | A20+P150 |
| --- | ---: | ---: | ---: | ---: |
| 1000 | 48,460 | 49,994 | 51,352 | 46,519 |
| 1500 | 48,713 | 47,429 | 48,514 | 46,043 |
| 2000 | 45,564 | 45,117 | — | 46,790 |
| 2500 | 45,135 | 43,147 | — | 44,052 |
| 3000 | 45,848 | 45,110 | — | 45,372 |
| 3500 | 44,304 | 42,339 | — | 43,713 |
| 4000 | 45,319 | 43,443 | — | 42,327 |
| 5000 | 44,033 | — | — | 42,040 |

- A20+P150은 ep1000·1500에는 A20보다 낫지만, ep2000–3500에서는 A20보다 노출이 높다. ep4000에서 다시 2.57% 낮다. 일관된 학습 가속이라고 해석할 수 없다.
- P50은 ep1000·1500에서 A20보다 우수하지 않다. 하지만 완료 검증이 짧고 resume 이력도 있어 장기 성능 순위를 매길 수 없다.
- EventDecision의 ep5500은 39,651로 최신 조합의 완료 검증 최저값 42,040보다 낮다. shaping이 없으면 이 성능대에 도달할 수 없다는 주장과 맞지 않는다. 반대로 다른 설정이 섞여 있으므로 shaping이 성능을 낮췄다는 증거도 아니다.

### A20과 A20+P150의 공통 7시점 평균(ep1000–4000)

| 지표 | A20 | A20+P150 | 조합의 변화 |
| --- | ---: | ---: | ---: |
| 전체 위험 노출, 사람-스텝 | 45,226 | 44,974 | -0.56% |
| 활동 체류자 위험 노출 | 13,639 | 14,084 | +3.26% |
| 재진입 횟수 | 216.27 | 201.59 | -6.79% |
| 로봇 위험 구역 체류 비율 | 9.13% | 14.59% | +5.46%p |
| 평균 진입 깊이 | 1.20 m | 2.19 m | +0.99 m |
| 평균 속도 | 0.75 m/s | 0.69 m/s | -8.63% |
| 로봇당 추종자 | 10.10 | 11.84 | +17.21% |

**[Likely]** 조합 설정은 더 안쪽에서 더 많은 사람과 상호작용하지만, 그것이 바깥으로의 수송·유도 완료로 충분히 이어지지 않는 모습이다. 실제 이동 순서나 모집–이탈–재접촉을 확인하려면 궤적/영상 진단이 추가로 필요하다. 진입 시간과 추종자 수 자체를 성공으로 평가하지 않는다.

## 4. 총보상 차이의 회계적 분해

ep3001–4000 학습 에피소드 각각 1000개 평균이다. shaping은 원래 평가/로그 총보상에서 제외되어 있다.

| 항목 | A20 | A20+P150 |
| --- | ---: | ---: |
| 로그 total_reward | -1260.74 | -1120.04 |
| 공통 person_time+collision | -1143.51 | -1120.04 |
| remaining_path | -112.35 | 0 |
| reentry | -4.88 | 0 |

총보상 차이는 +140.70이며, 이 중 이전의 추가 비용은 117.23이다. 공통 항목 차이 +23.47은 약 2.1% 규모다. 난도·인원·행동이 다른 학습 궤적 평균이므로 이 2.1%도 P150의 인과적 효과는 아니다.

같은 구간 별도 shaping 로그는 최신 progress +145.38 / approach -67.86(로봇 전체 합), A20의 approach -67.09다. **부호·합계만으로 shaping의 품질을 평가하면 안 된다.** progress는 팀 항목으로 각 로봇의 학습 보상에 들어가고, approach는 해당 로봇 몫만 들어가므로 계수 20:150이나 로그 합계로 신호 비중을 단순 비교할 수도 없다.

## 5. shaping 함수 자체에서 확인한 학습 신호

형태는 둘 다 potential-based shaping이다.

`F = gamma_step * Phi(next_state) - Phi(state)`

Progress의 potential은 위험 구역 안 사람들의 안전 지점까지 남은 거리를 정규화한 음수다. Approach의 potential은 위험 구역 안에서 추종 중이 아니고, 로봇에게 정보를 받지 않았으며 위험 위치 기억도 없는 사람을 대상으로 한다.

`Phi_approach = 1 - exp(-S / 10)`

`S = sum(max(0, 1 - robot_person_distance / 50))`

### 접촉 직후의 불연속

실제 `TaskReward._phi_approach`를 호출한 1명·1로봇의 단순 기하 계산이다. 사람은 위험 구역 안에 있다고 가정한다. 전체 시뮬레이션이나 실제 정책 궤적 재현은 아니다.

| 상태 | potential |
| --- | ---: |
| 대상과 거리 10 m | 0.07688 |
| 대상과 거리 5 m | 0.08607 |
| 같은 사람이 추종 수용(type=0) | 0 |

weight=20, gamma=0.99/2초 조건에서:

- 4초 전후로 10→5 m 접근한 두 상태 사이 shaping은 **+0.149**.
- 다음 한 스텝에 추종 수용으로 대상에서 빠질 때 shaping은 **-1.721**.

**[Certain]** 성공적인 모집 순간에 Approach 항목만 보면 음수 신호가 나올 수 있다. **[Likely]** critic이 긴 효과를 충분히 배우지 못하면 이 신호가 접근–모집을 직관적으로 가르치기 어려울 수 있다. 다만 이것이 실제 run 병목인지, 얼마나 자주 일어나는지는 history에 해당 전이 지표가 없어 아직 모른다.

이를 곧바로 reward bug나 ‘추종할수록 최종 return이 나빠진다’고 해석하면 안 된다. 올바른 할인·종료 처리와 적절한 상태 표현 아래에서는 potential 차이가 할인 합에서 상쇄되는 것이 PBRS의 원리다. 최적 정책 보존과 함수 근사 기반 학습의 속도·안정성은 별개다. [Ng, Harada & Russell, ICML 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

또한 Progress는 음수 potential을 사용하므로 상태가 그대로여도 `(gamma_step-1)*Phi > 0`일 수 있다. 따라서 episode/shape_progress가 +150 근처라는 사실은 그만큼 군중을 대피시켰다는 뜻이 아니다. 적분 형태의 로그 합계 대신 **같은 상태에서 대안 행동 사이 shaping 차이**를 봐야 한다.

Approach는 `S`가 커지면 포화되어 거리 변화의 미분이 작아진다. 현재 정책에서 얼마나 자주 포화되는지는 이번 분석에서 재측정하지 않았다.

## 6. 효과를 분리하기 위한 다음 reward 실험

현재 alpha 실험과 reward 비교는 별도로 한다. reward를 비교할 때는 alpha·encoder·clock·UPT·환경·기본 보상을 모두 고정하고 fresh run으로 시작한다.

| 조건 | Approach | Progress | 기본 path/reentry |
| --- | ---: | ---: | --- |
| 무shaping 대조 | 0 | 0 | 0 / 0 |
| 접근만 | 20 | 0 | 0 / 0 |
| 진행만 | 0 | 150 | 0 / 0 |
| 조합 | 20 | 150 | 0 / 0 |

기존 최신 run은 마지막 칸의 참고점이지만, 학습 seed 반복·코드 버전을 맞춘 확인이 필요하다. **추가 run 하나만 할 수 있다면 `A20/P0 + path0/reentry0`를 우선**해 기존 최신 설정에서 Progress150을 뺀 비교를 권한다. ‘기존 A20 run’은 path1/reentry2라 이 대조를 대신할 수 없다.

추가 로그 후보:

- 로봇별 `S`, potential, approach shaping의 양/음 성분 및 포화 비율.
- 추종 수용 전후 potential 변화와 그 전이에서의 shaping 부호.
- 기본 보상·진행·접근 각각의 결정당 할인 합과 행동 후보 간 차이.
- 모집한 사람의 실제 위험 구역 이탈 시간, 재진입 및 로봇별 유도 완료량.

Approach 잠재함수의 대상 유지/종료 방식 변경은 위 진단 후 별도의 설계 실험으로 다룬다. 신호가 음수라고 즉시 접촉 보너스를 덧붙이면 PBRS 형태가 깨져 목적 자체가 바뀔 수 있다. 원문에서 보장하는 조건을 지키며 비교해야 한다.

## 7. 재현

- [분석 스크립트](analyze_shaping.py), [수치 결과](shaping_analysis.json), [비교 그림](shaping_comparison.png).
- 실행: `MPLCONFIGDIR=/tmp/adds_review_mpl_20261019 PYTHONDONTWRITEBYTECODE=1 python3 SAC_UED6_easy/docs/review_20261019/analyze_shaping.py`.
- 저장된 스냅샷과 현재 production potential 함수를 사용한다. 새 W&B 조회·정책 학습·장시간 rollout은 실행하지 않는다.
