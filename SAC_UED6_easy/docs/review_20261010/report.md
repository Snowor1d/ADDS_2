# SAC_UED6_easy 학습 검토 — 2026-10-10

**[Certain] 학습 실패 전체가 아니라 초기 개선 후 정체다. [Likely] 평가 공백, entropy 하한, 여전히 복잡한 행동·협동·군중 반응이 우선 검토 대상이다.** 현재 데이터로 어느 하나를 정체의 단일 원인이라고 확정할 수 없다.

## 근거와 범위

- W&B run: https://forge.coreweave.com/wandb/437snowcap-sungkyunkwan-university/adds-sac-ued6/runs/fcfadzfh
- 공식 `api.wandb.ai` API로 읽기 성공. 5,342 history rows, 3,462 episode rows, 1,863 learner rows를 내려받았다. 실행 중인 run의 한 시점 snapshot이다.
- 저장: `run_config.json`(비밀값 이름 제외), `run_metrics.jsonl`(스칼라만), `analysis.json`, `learning_curves.png`. `python3 docs/review_20261010/analyze.py`로 오프라인 분석 재현 가능.
- 현재 코드의 SAC, 네트워크, replay/rollout, 관측, 보상, 군중 반응, 평가, 설정을 확인했다. 원격 실행의 코드 revision과 현재 소스가 완전히 같다는 보장은 없지만 주요 설정은 일치한다.
- 검증: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests/test_madrl.py -q --disable-warnings` → **67 passed, 1 warning, 261.47 s**. 전체 저장소 테스트나 장기 재학습은 실행하지 않았다.
- 훈련 코드와 설정은 변경하지 않았다.

## 실제 개선과 정체

현재 total reward는 벌점 합계다. **-1,000 → -700이 개선**이며, 수치가 더 작아지는 -700 → -1,000은 악화다. 누적 위험 노출은 낮을수록 좋다.

| 구간 | 평균 total reward | hazard person-steps | 30초 연속 빈 구역 경험 비율 | 평균 실제 속도 | guide 비율 |
|---|---:|---:|---:|---:|---:|
| 1–500, random warmup | -1,035.6 | 19,316 | 27.2% | 0.943 m/s | 50.0% |
| 501–1,000 | -966.5 | 16,579 | 28.2% | 0.750 m/s | 65.2% |
| 1,001–1,500 | -782.3 | 14,021 | 38.2% | 0.675 m/s | 86.3% |
| 2,001–2,500 | -722.0 | 12,679 | 43.8% | 0.765 m/s | 88.0% |
| 2,501–3,000 | -713.6 | 13,383 | 40.8% | 0.783 m/s | 90.2% |
| 3,001–3,462 | -729.6 | 13,296 | 41.8% | 0.764 m/s | 89.2% |

501–1,000 대비 최근 구간: 보상 벌점 크기 24.5%, 위험 노출 19.8% 감소. 무작위 warmup 대비 위험 노출은 약 31.2% 감소한다. 서로 다른 에피소드 분포의 관측 평균 비교이므로 paired policy/off 효과로 해석하지 않는다.

지도별 post-warmup 초기/최근 위험 노출은 dotonbori 12,787→10,004, covent_garden 16,808→12,935, hongdae 21,116→17,269로 모두 개선했다. 지도 구성 변화만으로 전체 개선을 설명하기 어렵다.

후기 개별 episode reward SD는 약 440으로 초기/후기 평균 차이 237보다 크다. 위치·군중·인지 변동이 크다. 지도, 초기 인원, 위험 면적, perceptibility, 인구를 보정한 descriptive OLS에서는 1,000 episode당 reward +90.5, HC1 SE 10.6이나 R²는 0.111이다. 시계열 상관과 비관측 조건을 통제하지 않으므로 인과 효과나 정체 원인의 증거로 사용하지 않는다.

## 우선순위 1: 평가 계약

**[Certain] `PERIODIC_VALIDATION=False`이고 고정 개발 시나리오에서 checkpoint별 policy/off/shuttle 비교 기록이 없다.** 훈련 return은 stochastic policy, seed·hazard·지도 분포까지 섞인 값이다. 현재 정책의 무로봇 대비 순효과와 deterministic 성능을 알 수 없다.

권고: 기존 checkpoint를 고정된 소규모 개발 세트에서 off, waypoint에 맞춘 shuttle, stochastic policy, deterministic policy로 비교한다. 세 지도×10–20 시나리오를 시작점으로 하고 분산에 따라 확대한다. 최종 명동 테스트는 튜닝에 쓰지 않는다. 누적 person-seconds, 최초 empty, 최초 30초 held-clear, 이후 침입·재점유, 최종 인원, 충돌을 함께 측정한다.

평가 지표 의미도 주의:

- `held_clear_success`는 한 번이라도 60 steps=30초 연속 비웠다는 뜻이다. 끝까지 안전을 유지했다는 뜻이 아니다.
- `reentries_after_clear`는 각 사람이 안전 여유 거리 밖에 있다가 들어온 사건이다. 구역 전체의 최초 empty 이후만을 세는 변수가 아니다.
- 실패 시 `evac_time_100`은 1,000 steps로 채워지므로 censored time이다. 성공률과 함께 해석한다.
- 같은 seed pairing은 초기 조건을 맞추지만 행동이 전역 RNG 소비 순서를 바꿀 수 있다. 외생 과정까지 완전 동일한 counterfactual은 아니다.

## 우선순위 2: entropy 제약

**[Certain] alpha는 episode 548/update 5,200에 0.05에 도달했고 learner 로그의 98.66%에서 하한에 붙어 있다.** 최근 200 learner rows의 평균 entropy는 +2.143, target은 -3이다. 자동 조절은 alpha를 더 낮추려 하지만 clamp가 막는다. Alpha floor가 reward 규모와 정책 개선에 적합하다는 실험 근거는 현재 없다.

**[Likely] 정밀한 경로·역할 분화가 필요한 후기에도 불필요한 무작위성이 유지될 가능성**이 있다. 그러나 alpha 자체만으로 Q action 차이 대비 entropy가 지배한다고 확정할 수 없다. 총 entropy만으로 이동과 속도의 포화/분산을 판별할 수도 없다. guide 약 89% 역시 자동으로 실패를 뜻하지 않는다.

권고: 우선 동일 checkpoint의 deterministic/stochastic paired 평가. 다음은 새 run에서 ALPHA_MIN 0.05 vs 0.01 비교, 필요 시 무하한 비교. 연속 entropy, mode entropy, mean/log_std, action saturation, 모드별 Q 차이를 계측한다. 온도 하한을 바로 제거하면 과거 탐색 고착이 재발할 수 있으므로 안정 Jacobian을 유지하고 진단 지표를 같이 기록한다.

## 우선순위 3: easy의 실제 난이도

**[Certain] 쉬워진 것은 군중 위치 가시성·위험 면적·결정 시간이다.** 3대 협동, 속도 학습, off/guide, 위험구역 외부 5–15 m 시작, 무경고 유입, 낮은 위험 인지, 접촉 누적은 여전히 남아 있다.

- 기본 학습 500회 동안 업데이트가 없다. 1회 125 joint transitions이므로 약 62,500 random transitions을 먼저 모은다. 후기 replay 435k 안에도 초기 데이터가 남는다. 이는 버그가 아니라 탐색/적응 속도의 절충이다.
- 정보 전달은 시야 내 3.5m에서 누적 10초 노출, 로봇당 1회 성공확률 0.5다. 4초 action 변경이 접촉 누적을 직접 초기화하지는 않지만, 접촉 유지가 필요하다. 따라가기와 hazard 정보 전달은 별도 경로라 모든 대피에 10초가 필수인 것은 아니다.
- 4초에 최대 8m만 이동하지만 waypoint는 축당 최대 20m다. 재선택되는 긴 목표와 건물 projection 때문에 서로 다른 행동의 실제 결과가 비슷해질 수 있다. 목표 축소/후보 선택은 비교 실험 대상이다.
- perceptibility U(0.05,0.30), sensory floor 0.25이므로 약 80% 시나리오는 직접 감지 채널이 꺼진다. 작은 위험구역이 반드시 쉬운 로봇 제어 과제를 만들지는 않는다.

권고: 지도·hazard 고정 → 속도 고정 → guide 고정으로 대피 이동만 학습하는 통제 실험. 로봇 효과가 충분한 scenario를 먼저 scripted intervention으로 확인한다. 이후 모드, 속도, 유입 방어, hazard/지도 변동을 하나씩 복원한다. 기존 한 로봇 baseline의 개선폭이 작았으므로 로봇 수만 1대로 줄이면 오히려 학습 신호가 사라질 수 있다. 1대 검증은 한 대로 실제 효과가 나는 전용 배치를 구성한 경우에만 권한다.

## 우선순위 4: 혼합 SAC와 협동

**[Certain] mode는 hard straight-through Gumbel-Softmax다.** actor gradient는 critic의 off/guide 사이 보간에 의존하는 편향된 relaxation이다. 그 자체가 학습 실패를 입증하지는 않는다.

권고: 로봇별 actor update에서 같은 연속 행동에 off/guide Q를 모두 평가하고 확률 가중 기대값을 계산한다. 두 모드라 정확한 열거가 작다. 연속/mode entropy 계수를 분리하고 Bellman target의 entropy 정의도 일치시킨다. 근거: [Hybrid SAC](https://arxiv.org/abs/1912.11077), [ALF 공식 유도](https://alf.readthedocs.io/en/latest/notes/sac_with_hybrid_action_types.html). 이 자료는 이 환경에서의 우월성을 보장하지 않는다.

`ACTOR_TEAMMATE_ACTIONS="stored"`는 과거 teammate action에 대한 best response다. 버그로 취급하지 않는다. 협동 정체 가능성이 있으므로 actor의 `current` 옵션을 별도 비교하되, policy lag와 실제 역할/중복 커버리지를 함께 측정한다. worker policy는 episode 시작에만 갱신되고 broadcast는 전체 10 episodes마다라 실제 update lag를 기록할 필요가 있다.

## 우선순위 5: 관측과 장기 목표

**[Certain] fullinfo는 전체 군중의 위치/밀도 정답이지 전체 행동 내부 상태 정답이 아니다.** hazard 인지·반응 지연·로봇당 정보 접촉 누적·follow 상태 등이 입력에 직접 없다. 같은 밀도 배치에서도 안내 반응은 다를 수 있다. critic도 군중 지도와 N_ref/D_ref/perceptibility만 추가로 받는다.

권고: 디버깅용 oracle critic에 위험구역 내 인지/미인지 인원, 반응 대기 상태, 로봇별 follower·접촉 상태를 넣어 대조한다. 실제 배포 actor에는 직접 관측 가능한 follower/속도·흐름 요약 또는 recurrent memory를 검토한다. sim 내부 상태를 실측 actor 입력인 것처럼 취급하지 않는다.

gamma 0.99/2초의 할인 반감기는 약 138초, 300초 뒤 가중치는 0.221이다. 500초 과제에서 후기 방어를 상대적으로 약하게 본다. 이것도 즉시 버그가 아니다. 대피와 방어를 분리한 뒤 gamma 0.995 대조를 고려하되 더 긴 bootstrap horizon의 variance/credit 비용을 감수한다. person-time 목표를 처음부터 sparse terminal reward로 바꾸는 것은 권하지 않는다.

## 이미 수정되어 원인으로 재사용하지 않은 항목

현재 source에는 stable log-sigmoid Jacobian, critic의 N_ref/D_ref/perceptibility scalar, reward/dynamics fingerprint, 적용되지 않는 simulator override 거부가 있다. 예전 10월 7일 리뷰의 해당 결함을 그대로 현재 결함으로 반복하지 않았다. 기존 shuttle baseline은 여전히 velocity식 heading×speed를 waypoint offset으로 쓰고 speed를 명시하지 않아, 명시된 1.2m/s/10초 대기와 실제 waypoint 실행이 다르다. baseline 정비가 먼저다.

## 권장 순서

1. 현재 checkpoint의 고정 개발 세트 paired 평가 및 필요한 진단 로깅.
2. alpha floor 대조와 deterministic/stochastic 성능 차이 확인.
3. 충분한 로봇 개입 효과가 검증된 고정 배치에서 guide·속도 고정 제어 학습.
4. mode 정확 열거, entropy 분리; 이후 teammate current 대조.
5. 필요할 때 내부 상태 oracle critic 및 관측 메모리/흐름 정보.

각 변경은 독립적으로 비교하고 최종적으로 최소 3개 training seed로 확인한다. 네트워크 확대, LR 확대, VLM 도입, 보상 전면 개편을 동시에 수행하면 현재 정체를 진단하기 어렵다.
