# SAC_UED5 환경·학습기·UED 검토 — 2026-10-07

**[Likely] 다음 단계는 네트워크 확대나 최신 알고리즘 교체보다, 비교 평가 복구 → 실행 일관성 수정 → 혼합 행동 SAC 및 시간 표현 개선이다.** 현재 코드에는 재현되는 결함이 있지만, 그것이 최신 학습 실패에 얼마나 기여했는지는 최신 학습 기록과 대조 실험 없이 확정할 수 없다.

학습 코드와 기본 설정은 변경하지 않았다. 이 문서, 작은 읽기 전용 재현 스크립트 `probe.py`, 그 실행 결과 `probe_results.json`만 추가했다.

## 1. 검토 범위와 증거의 한계

- 검토 기준 커밋: `283669109f5279653bb6f8a42898d08a97217897`.
- 주요 경로: `configs/`, `learn/{ADDS_AS_reinforcement,sac,networks,replay,rollout,training_maps,zero_shot,metrics_logger}.py`, `sim/{model,agent,observation,rewards,robot_action,core,space,danger}.py`, `ued/{runner,population,level,mutate}.py`, 도시 검증기와 행동·shuttle 검증 경로. 관련 설계·검증 문서와 저장된 분석도 확인했다. 모든 보조 모듈의 모든 줄을 검증했다는 뜻은 아니다.
- **[Certain] 현재 기본 설정:** 실제 OSM 지도 `dotonbori`, `covent_garden`, `hongdae`, 모두 200 m, 로봇 3대, waypoint + speed + off/guide, `rew-v5-projection`, replay 100만 record, UPT 1.0. `TRAIN_MAP_SOURCE="dataset"`이므로 ACCEL 선택·변이는 현재 학습에 적용되지 않는다.
- **[Certain] 현재 로그 경로 `~/Log_SAC_UED5_madrl/events.jsonl`에는 run_start 한 건만 있고 체크포인트·replay snapshot은 없다.** 저장된 run_start는 velocity/이전 행동 모델이며 현재 소스와 다르다. 이 파일로 최신 waypoint 정책의 학습 추세를 판단할 수 없다.
- 10월 5일 저장 분석은 velocity, 지도 20개 실행의 기록이다. 그 분석에서 충돌 악화와 이동 고착이 보고됐지만, 현재는 action·행동 모델·지도 분포가 달라 직접 원인으로 재사용하지 않았다.
- 전체 테스트: **324 passed, 1 skipped, 1 failed** / 458.32 s. 실패는 `LoggerTest.test_real_wandb_disabled_and_offline`이며, W&B core가 이 실행 환경의 읽기 전용 캐시와 금지된 소켓에 접근하면서 발생했다. SAC 수치 테스트 실패로 해석하면 안 된다. 다른 검사가 통과해도 아래의 학습 목적·설정 전달 문제가 자동으로 배제되지는 않는다.

재현:

```bash
cd SAC_UED5
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 docs/code_review_20261007/probe.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest tests/ -q --disable-warnings
```

## 2. 수정 우선순위

| 순서 | 대상 | 판단 | 현재 기본 실행에 해당하는가 |
|---|---|---|---|
| 1 | 고정 시드 policy/off/random/shuttle 비교와 유효한 baseline | 평가 공백 및 baseline 의미 불일치 확인 | 예 |
| 2 | cfg 전달과 reward/dynamics resume 계약 | 재현되는 실행 일관성 문제 | override·재개 시 |
| 3 | 안정적인 squash Jacobian | 큰 mean에서 기울기 소실 재현 | 해당 수치 영역에 진입하면 |
| 4 | Gumbel 대신 모드 기대값 열거, entropy 분리 | 학습 구조 개선 권고 | 예 |
| 5 | 가변 결정 길이, 관측 시간·자기 이동 정보 | 목적·관측의 모호성 확인, 영향은 실험 필요 | 예 |
| 6 | critic 보상 기준·행동 상태 context | 입력 누락 확인, 영향은 실험 필요 | 예 |
| 7 | 단일 에이전트 ID allocator | 에이전트 덮어쓰기 재현 | 큰 지도·누적 inflow |
| 8 | UED 점수·메시지 순서·혼합 점수 단위 | 조건부 결함 및 목적 불일치 확인 | UED 사용 시 |

### 2.1 비교 평가를 먼저 복구하고 shuttle을 waypoint에 맞춘다

위치: `configs/training/common.py:250` 부근, `learn/zero_shot.py:315`, `validation/shuttle_baseline.py:130`, `validation/behaviour.py:74`.

**[Certain] `PERIODIC_VALIDATION=False`다.** 학습 rollout 보상은 서로 다른 hazard·군중 시나리오의 stochastic 정책 성능이므로 정책 개선의 직접 증거가 되기 어렵다. 기본 검증을 켜더라도 5,000 episode 간격과 100/200/400 m × 난이도 × 1/2/3대 조합은 디버깅 피드백이 느리다.

**[Certain] 현재 shuttle은 velocity 의미로 action을 만든다.** 예를 들어 guide에서 `(0.6, 0)`을 `encode`하면 waypoint는 x 방향 6 m 목표, 속도는 기본값 1.0 즉 2 m/s다. 의도한 1.2 m/s가 아니다. `EXIT_WAIT_DECISIONS=5`도 가변 결정에서는 10초를 뜻하지 않는다. 행동 검증의 `_scripted_act` 역시 단위 방향을 waypoint offset으로 사용한다. waypoint에서 실행할 때 경로를 여러 번 다시 정하는 동작은 가능하지만, 문서에 적힌 속도·대기시간과 동일한 baseline은 아니다.

권고:

1. 훈련용 지도 1개에서 hazard·crowd seed를 고정한 개발 세트를 만든다. 최종 명동 제로샷은 튜닝에 쓰지 않는다.
2. off, random, waypoint 전용 shuttle, stochastic policy, deterministic policy를 같은 시나리오에서 비교한다.
3. waypoint offset은 `2 * (target - robot_position) / ROBOT_WAYPOINT_RANGE_M`, guide 속도는 `encode(..., speed=0.6)`처럼 명시하고 대기시간은 simulation seconds로 센다. 범위 밖 목표는 정해진 방식으로 자른다.
4. person-steps의 paired 절대 차이·상대 차이, 충돌, 최종 위험구역 인원, held-clear, 안내 중 실제 속도와 영향을 받은 인원을 함께 기록한다. off person-steps가 거의 0인 pair는 상대 감소율이 불안정하므로 별도로 보고한다.
5. 작은 개발 평가를 일정한 새 transition 수마다 실행하고, 전체 일반화 검증은 더 낮은 빈도로 수행한다.

같은 seed를 다시 넣어도 행동에 따른 전역 `random` 호출 수가 달라지면 뒤의 군중 확률 시행은 달라진다. 현재 pairing은 초기 조건을 맞추는 의미가 크다. 더 정밀한 counterfactual 비교에는 사람별/사건별 RNG 또는 사전 생성 외생 이벤트가 필요하다.

### 2.2 ResolvedConfig가 시뮬레이터와 UED까지 전달되지 않는다

위치: `configs/__init__.py:240`, `learn/ADDS_AS_reinforcement.py:489`, `ued/runner.py:82`, `sim/model.py:38`, `sim/agent.py`의 module-level config import.

**[Certain] 다음 불일치를 재현했다.**

| resolve한 설정 | 실제 하위 모듈 동작 |
|---|---|
| `TRAIN_MAP_SOURCE="ued"` | `UEDRunner().enabled == False` |
| `ROBOT_INFORM_TIME_S=5.0` | `sim.agent.ROBOT_INFORM_TIME_S == 10.0` |
| `ROBOT_START_RING_M=(1,2)` | `sim.model.ROBOT_START_RING_M == (5,15)` |

기본 파일을 수정한 뒤 프로세스를 완전히 재시작하면 위 사례는 파일값을 읽는다. 문제는 명시적으로 지원하는 `main(overrides=...)`와 테스트·실험 경로다. 기록한 설정과 실제 환경이 달라져 ablation 결과를 잘못 해석할 수 있다. `base_fingerprint` 검사는 파일이 바뀌었는지를 검사할 뿐 이 문제를 해결하지 않는다.

권고: `FightingModel(..., cfg=cfg)`, agent의 `model.cfg`, `UEDRunner(cfg=cfg)`, population·generator의 cfg 전달로 통일한다. 당장 전부 전환하기 어렵다면 아직 전달되지 않는 override를 시작 시 거부한다. module 전역을 한 번 patch하는 방식은 spawn·다중 실험에서 다시 어긋나기 쉽다.

### 2.3 Resume 검사는 reward와 dynamics의 실제 의미를 충분히 검사하지 않는다

위치: `learn/sac.py:339,368`, `learn/replay.py:405,414`, `configs/__init__.py:48`.

**[Certain] checkpoint에 config fingerprint를 저장하지만 load는 그 전체값을 비교하지 않는다.** 버전 문자열과 observation schema 중심이다. `REWARD_W_COLLISION`을 0.05에서 0.1로 바꿔도 두 검사는 그대로 같다. Replay에는 이미 가중치가 곱해진 보상이 저장되므로 오래된 reward와 새 reward가 섞일 수 있다. `ROBOT_SPEED_MAX`, 정보 전달 시간 등 dynamics의 여러 설정도 observation schema에 없다.

권고: 전체 config의 완전 동일성을 강제하기보다 reward contract, transition/dynamics contract, observation/action contract의 fingerprint를 나눈다. LR·로깅 설정 변경과 환경 의미 변경을 구분하고, 후자는 기본 resume를 거부한다. 명시적인 weight transfer는 새 buffer와 별도 실험으로 처리한다. 현재 `RESUME_MODE="fresh"`는 같은 폴더의 옛 checkpoint를 제거하지 않으므로 새 LOG_DIR도 필요하다.

### 2.4 Squash log-probability를 안정적인 수식으로 바꾼다

위치: `learn/networks.py:250`.

현재 변환은 `a = 4 * sigmoid(u) - 2`다. 이 변환 자체는 유효하다. 문제는 `log(4 * sig * (1-sig) + 1e-8)`로 Jacobian을 계산하는 수치 구현이다.

**[Certain] float32, mean=20 또는 40의 고정 reparameterization probe에서 현재 log-prob의 mean 방향 기울기는 0이며 안정 수식은 1이다.** sigmoid가 정확히 1로 반올림되고 epsilon이 Jacobian을 상수로 만들기 때문이다. 현재 checkpoint가 이 영역에 실제 도달했다는 증거는 없다.

같은 변환을 유지하는 최소 수정:

```python
log_abs_det = math.log(4.0) + F.logsigmoid(u) + F.logsigmoid(-u)
logp_cont = (logp_u - log_abs_det).sum(-1)
```

mean 크기·포화율을 같이 기록한다. `LOG_STD_MIN=-5`나 alpha floor만으로 mean 포화 문제를 해결했다고 판단하지 않는다. `temperature != 1`을 API로 유지할 경우 실제 샘플의 표준편차와 log-prob 수식도 일치시켜야 한다. 현재 기본 경로는 temperature=1이다.

### 2.5 두 개 모드는 Gumbel 경로 대신 기대값을 정확히 계산하는 편이 낫다

위치: `learn/networks.py:265`, `learn/sac.py:266`.

**[Certain] off/guide는 hard straight-through Gumbel-Softmax로 미분한다.** 실제 환경은 이산 꼭짓점만 실행하지만 actor gradient는 critic의 그 사이 보간에 의존한다. 이는 편향된 relaxation이며 구현이 존재한다는 이유만으로 잘못된 알고리즘이라고 단정할 수는 없다. 다만 모드가 2개뿐인 현재 문제에서 그 편향을 감수할 이유가 약하다.

권고하는 한 로봇의 actor objective는 다음 형태다. 다른 로봇 action을 고정한 채 해당 로봇의 두 모드를 평가한다.

\[
 L_i = \mathbb E_{a_i^c}\left[\alpha_c\log\pi_c(a_i^c|o_i)
 +\sum_m p_i(m|o_i)\{\alpha_d\log p_i(m|o_i)
 -\min(Q_{1,i},Q_{2,i})(s,a_{-i},a_i^c,m)\}\right].
\]

모드에는 확률의 정확한 gradient가, 연속 이동·속도에는 reparameterization gradient가 흐른다. Target에도 같은 objective를 사용한다. 팀의 모든 mode joint expectation까지 정확히 구한다면 로봇 3대에서 조합은 8개다. actor 자신의 모드만 열거하고 teammate를 샘플링하는 계산량 절충도 가능하다.

연속 differential entropy와 이산 categorical entropy를 각각 기록하고, 온도·목표도 분리한다. 이산 목표는 예를 들어 `0.5*log(2)`와 `0.8*log(2)`를 실험값으로 비교할 수 있지만 최적값이라는 근거는 없다. 자동 온도의 현재 업데이트 부호는 뒤집혀 있지 않으며, 연속 entropy가 음수인 것도 오류가 아니다. 현재 waypoint의 연속 차원은 **3**, 자동 기본 목표는 **-3**이다.

이 방향의 수식은 [Hybrid SAC 논문](https://arxiv.org/abs/1912.11077)과 [ALF 공식 혼합 행동 SAC 설명](https://alf.readthedocs.io/en/latest/notes/sac_with_hybrid_action_types.html)에 근거한다.

### 2.6 가변 길이 결정의 목적과 관측 시간을 명확히 한다

위치: `learn/rollout.py:79`, `learn/sac.py:142`, `sim/agent.py:2665`, `sim/observation.py:47,285,650`.

**[Certain] 환경 보상 할인은 실제 hold k에 맞춰 `sum(g^j*r_j)` 및 `g^k`를 사용한다. 이 부분을 고정 gamma로 바꾸면 오히려 틀린다.** 그러나 entropy는 결정 한 번당 부과되고 결정 길이는 1–60 simulation step, 즉 0.5–30초다. 한 로봇의 도착·벽 막힘이 팀 전체 action을 다시 정하게 한다.

따라서 현재 목적은 물리 시간당 entropy가 아닌 결정당 entropy다. 이런 semi-MDP objective는 정의할 수 있지만, 빈번한 재결정·대기시간과 탐색 보상이 결합한다. 선호 방향은 entropy 부호와 정책에 따라 달라진다. 무조건 짧은 action을 선호한다고 단정하면 안 된다.

**[Certain] DecisionRecord에는 실제 simulation timestamp가 없고, observation age도 결정 개수 기준이다.** history의 한 칸이 0.5초인지 30초인지 actor/critic이 직접 알 수 없다. ego의 과거 crowd frame도 그때의 robot anchor 그대로 쌓고 과거 자기 위치·속도를 입력하지 않는다. 같은 픽셀 변화가 사람 이동인지 자기 이동인지, 같은 메시지 age가 몇 초인지 모호하다.

권고:

- 먼저 waypoint에서 고정 2–5초 결정 간격, events off 대조를 실행한다. 수치는 실험 시작값이다. 속도 action을 잠시 0.6–0.8로 고정하는 별도 ablation도 문제를 단순화한다.
- 가변 결정을 유지하면 실제 timestamp와 frame 간 delta-time, 자기 변위/속도를 저장·입력한다. ego history를 현재 좌표계로 옮기거나 각 frame의 pose를 제공한다.
- 물리 시간당 entropy를 원하면 semi-MDP 목적부터 정하고 actor와 target을 함께 유도한다. replay에 저장된 hold를 현재 actor의 새 action에 기계적으로 곱하면 duration이 action에 의존한다는 문제를 남긴다.
- replay의 hold 분포, 1-step 비율, 종료 event 원인, 로봇별 도착이 다른 로봇 경로를 얼마나 자르는지 기록한다.

현재 gamma=0.99는 2초 기준이다. 할인 반감기는 약 **137.94초**, 300초 뒤 가중치는 **0.221**다. 장기 방어가 약하다면 이 시간 스케일도 비교하되 먼저 구현 문제와 분리한다. MAX_STEPS를 truncation으로 bootstrap하는 것은 continuing-task 해석에서는 맞는다. 정말로 500초까지만의 finite-horizon 최적화를 원한다면 시간 입력과 terminal 정의를 그 목적에 맞춰야 한다.

### 2.7 Critic은 중앙집중이지만 완전한 상태를 보는 것은 아니다

위치: `sim/rewards.py:39`, `sim/observation.py:787`, `learn/networks.py:326`.

**[Certain] critic의 privileged 입력은 군중 density 1채널이다.** reward의 초기 `N_ref`, `D_ref`, 시나리오 perceptibility, 사람의 awareness·milling·hazard memory·following/inform exposure는 직접 들어오지 않는다. 같은 현재 사람 수·위치도 초기 N_ref에 따라 보상이 다르고, 인지 상태에 따라 미래 반응이 다르다.

**[Likely] 이 aliasing이 Q 회귀를 어렵게 한다.** 추가 입력의 실제 효과는 ablation이 필요하다.

권고: 우선 critic 전용 scalar로 N_ref, D_ref, perceptibility를 제공한다. 이후 aware/acting 수, 로봇별 follower 요약이나 채널을 비교한다. actor에는 실제 센서에서 얻을 수 있는 crowd motion과 자기 속도를 제공하는 방향이 적절하다. simulator의 심리 상태를 현실적인 actor 관측으로 슬쩍 넣으면 다른 실험이 된다. Recurrent actor/critic은 이 입력 정리 이후의 선택지다.

### 2.8 에이전트 ID 충돌은 큰 지도 사용 전에 반드시 수정한다

위치: `sim/model.py:523,1079,2053,2219`, `sim/core.py:39`, `sim/space.py:88`.

**[Certain] 초기 군중 ID는 0부터, 로봇 ID는 1000부터 시작한다.** 실제 FightingModel에 1,001명과 로봇 3대를 생성하면 agent list는 1,004개지만 schedule은 1,003개이고 ID 1000의 군중은 schedule과 space에서 사라진다. 군중 리스트에는 남는다. 이후 inflow의 군중 ID도 로봇 ID에 도달하면 로봇을 덮어쓸 수 있다.

현재 세 학습 지도의 walkable area는 약 13,325–14,763 m²이고 초기 군중은 기본 밀도에서 약 533–738명이다. 1,000 step inflow는 약 40명이라 이 결함을 현재 기본 학습 실패의 원인으로 지목할 수는 없다. 하지만 200/400 m 생성 지도와 큰 군중 실험에는 해당한다.

권고: model의 단일 monotonic allocator로 crowd와 robot의 ID를 모두 발급한다. crowd 인원 카운터와 ID 카운터를 분리하고 duplicate insertion은 schedule/space에서 조용히 덮어쓰지 않도록 검사한다. 생성 시뿐 아니라 inflow로 경계를 넘는 회귀 검사도 필요하다.

### 2.9 UED를 다시 켜기 전에 수정할 것

현재 dataset에서는 아래가 비활성이다. 현 학습 부진의 직접 원인과 구분해야 한다.

**A. MaxMC의 서로 다른 return 정의** — `ued/runner.py:250`, `learn/sac.py:472`.

trace는 undiscounted 전체 episode team reward를 누적한다. 비교 대상은 discounted·entropy-regularized V이며 rew-v5에서는 task + 평균 robot penalty다. 시작 state return과 임의의 중간 state-to-go value도 섞인다. global best를 모든 레벨의 공통 상한처럼 쓰므로 원래 MaxMC와도 다르다. 완전히 같은 trace/value라도 다른 쉬운 레벨이 global best를 -100에서 0으로 올리면 probe score가 0에서 100으로 바뀐다.

권고: 초기에 SFL 방식의 repeated-trial learnability를 단독으로 쓴다. value proxy를 유지하려면 같은 레벨·같은 시점·같은 할인·같은 reward/entropy objective의 return-to-go를 비교하고 검열된 마지막 state를 처리한다. 레벨 간 좋은 최대 return은 그 레벨의 최적 return 추정치가 아니다.

**B. Hybrid의 단위 불일치는 global ranking만으로 없어지지 않는다** — `ued/population.py:195,212`.

learnability는 최대 0.25, MaxMC는 return 단위다. 두 값을 섞어 정렬한 다음 rank로 바꿔도 서로 다른 척도로 만든 초기 순서가 남는다. probe의 mature level 0.122와 cold level MaxMC 10은 여전히 후자가 높은 priority를 얻는다. `should_breed`는 충분히 평가된 레벨끼리 비교하므로 이 지적은 주로 sampling/eviction에 관한 것이다.

권고: warm/cold group을 분리해서 각각 rank·quantile을 구하고 명시적인 혼합 비율로 샘플링하거나 cold-start trial 예산을 따로 배정한다. 회귀된 `maxmc`는 오래된 정책 값이 남는 점도 함께 처리한다.

**C. Episode summary와 transition이 서로 다른 Queue에 있다** — `learn/ADDS_AS_reinforcement.py:242,286,565,596`.

main은 transition 하나를 처리한 뒤 stats queue를 전부 비운다. 두 queue 사이 소비 순서 보장이 없으므로 summary를 incomplete trace에 적용할 수 있다. summary 뒤 늦은 transition이 도착하면 이미 소비한 episode의 trace가 다시 생성된다. probe는 이 허용된 순서로 trace 재생성을 확인했다. 실제 실행에서 발생 빈도는 계측하지 않았다.

권고: episode 종료 메시지를 transition과 같은 FIFO로 보내거나, expected decision count/last step을 받아 모든 transition이 들어온 뒤 scoring한다. 추적 키는 worker/episode index보다 episode_uid가 안전하다.

**D. population은 curriculum 분포이며 replay는 역사적 분포다.** 현재 queue 선택을 바꿔도 uniform million-record replay의 update 분포는 즉시 바뀌지 않는다. transition에 level/task id와 collection policy version을 두고 실제 학습 batch 분포를 계측한다. UED 효과가 확인된 뒤 level-stratified sampling이나 최근/전체 replay 혼합을 비교한다.

### 2.10 다른 환경·운영 결함과 비용

- **[Certain] 회전 사각형 검증 불일치:** `citygen.validate._zone_masks`의 rect 경로는 angle을 쓰지 않는다. 90도 회전한 24×4 m 사각형에서 실제 DangerZone과 640개 0.5 m cell이 다르다. 현재 학습은 circle이지만 rect/street 확장 전에 고쳐야 한다.
- **[Certain] free-flow reference:** `sim/model.py:1336`은 uniform spawn에서도 `total_agents * danger_inside_fraction`을 초기 내부 인원으로 쓴다. uniform에서는 실제 `agents_in_danger()`와 다르다. UED의 success 기준에 영향을 주므로 생성 직후 실제 인원을 사용한다. perimeter 전체를 유효 통과 폭으로 보는 근사도 실외 가로망에서는 과도할 수 있다.
- `main_walkable_component`는 면적이 아닌 triangle 개수로 component를 선택한다. 밀도 headcount는 전체 walkable area로 계산하고 spawn은 선택 component에 한정하므로, disconnected crop에서 실제 군중 밀도가 더 높아질 수 있다. component 면적·사용 면적 기준을 통일하는 것이 맞다.
- `nearest_main_ground` fallback은 robot body가 들어갈 수 있는 centroid인지 다시 검사하지 않는다. navmesh walkability는 pedestrian radius 기준이며 로봇은 더 크다. 거리상 가장 가까운 centroid가 항상 robot-reachable이라는 보장은 없다. body-inflated navigation graph와 reachable 후보 검사가 더 견고하다.
- **[Certain] worker는 새 PolicyNetwork를 만들고 episode 시작에만 parameter queue를 읽는다.** 시작·재시작 시 즉시 최신 policy를 넣는 경로가 없고 broadcast는 episode 주기다. resume/worker 복구 후 첫 episode를 임의 초기 policy로 실행할 수 있다. 초기 snapshot handshake와 update/version 기준 broadcast, decision 경계 갱신 및 policy lag 로그를 권한다. 에피소드 중 업데이트 허용 여부는 실험 목적에 맞춰 명시한다.
- main/worker의 torch seed가 명시적으로 고정되지 않고 generation에 wall-clock이 들어간다. 반복 가능한 실험 seed와 checkpoint RNG 상태를 분리해서 저장한다. 여러 async worker의 완전 비트 재현과 통계적 재현은 구분한다.
- **[Certain] 현재 replay 동적 배열은 record당 7,013 B, 100만 개에서 6.53 GiB다.** worker·정적 지도·batch·optimizer 메모리는 별도다. 매 100 episode에 전체 buffer를 저장하며 정적 hazard key 파일을 계속 남긴다. 실제 저장 시간·queue 대기·RAM을 측정하고 checkpoint snapshot과 replay 주기, 정적 캐시 정리 정책을 분리한다.
- old docs의 UPT 0.25·buffer 10만·2,000 step 성능은 현재 all-robot actor update와 waypoint/100만 buffer의 비용 보장이 아니다. UPT 1.0 자체가 과도하다고 단정할 수 없다. 새로 들어온 **valid action transition** 수, update 수, simulation seconds, GPU/queue 시간을 함께 보고 0.25와 1.0을 비교한다. 현재는 final state record도 update 예산을 늘린다.
- raw NaN/Inf를 logger에서 제외하므로, 학습 loss/Q/gradient의 finite 검사와 별도 실패 이벤트가 필요하다. gradient clipping은 norm을 먼저 계측한 뒤 적용한다. MSE를 Huber로 바꾸는 것은 outlier 분포 확인 이후의 선택이다.
- off control cache key에는 geometry/dynamics fingerprint가 없고 `best_score`는 재개 시 None이다. 검증을 활성화한다면 cache 무효화와 best score 복원을 함께 고친다.

## 3. 환경 설계에서 먼저 검증할 가정

**[Certain] 현재 hazard perceptibility는 U(0.05,0.30), 직접 감지 floor는 0.25다.** 따라서 약 80%의 episode에서 사람의 직접 위험 감지 채널이 닫힌다. prior warning은 0이다. 과거 0.5 perceptibility 실행과 다른 문제이며, 자연 대피가 지배한다는 과거 해석을 그대로 적용할 수 없다. 일반 통행과 outflow로 위험구역이 자연스럽게 비는 효과는 여전히 있다.

낮은 perceptibility를 쓰는 것 자체는 숨은 위험 과제로 타당할 수 있다. 하지만 처음부터 경고·접근·유도·해제·재진입 방어를 동시에 발견해야 한다. 먼저 사람이 충분히 만날 수 있는 작은 과제에서 로봇 개입의 upper bound를 확인하는 편이 낫다. 쉬운 과제와 어려운 과제를 구분하는 기준은 단순 무로봇 성공률만이 아니라 **유효한 scripted intervention의 개선 가능성**이어야 한다.

현재 `ROBOT_INFORM_RADIUS_M=3.5`, 정보 전달 10초, 안내 standoff 3m다. 2m/s로 사람을 스쳐 지나가는 정책은 충분한 접촉 시간을 만들기 어렵고, 로봇보다 느린 군중을 놓칠 수 있다. 안내 속도별 follower 유지·메시지 전달률을 먼저 측정한다. 이것이 waypoint speed action을 학습해야 할 이유이지만, bootstrap 디버깅에서는 고정 속도 대조가 유용하다.

person-time 보상은 과제와 직접 연결되고 초기 reference 고정도 누적 유입으로 벌점이 희석되는 문제를 피한다. 이를 바로 sparse terminal reward로 바꿀 이유는 없다. 위치·초기 인원·인지 차이가 로봇 효과보다 크면 critic context와 paired 평가를 보완한다. projection penalty는 작은 regularizer지만 비잠재형 shaping이며 정지/짧은 목표를 선호하게 만들 수 있어 no-projection 대조를 남긴다. 계속 invalid 목표가 많으면 reachable waypoint 후보와 action mask가 더 직접적인 대안이다.

군중 모델의 기본도·병목 검증은 이동 물리의 근거이며 안내 신뢰도·인지 전파의 실외 타당성을 자동으로 검증하지 않는다. 현재 문서가 실내 로봇 실험과 실외 VR의 범위를 구분하는 것은 적절하다. [실외 침수 VR 연구](https://nhess.copernicus.org/articles/26/981/2026/)도 모든 도심 재난의 로봇 순응률을 제공하지 않는다. 따라서 inform probability·response delay·nonresponse 등에 대한 sensitivity 결과를 별도로 보고해야 한다.

## 4. 논문·프레임워크 비교와 적용 판단

아래는 2026-10-07에 원 논문·공식 문서를 확인한 관련 작업이다. 이 표는 모든 최신 연구를 포괄하거나 이 환경에서의 성능 순위를 입증하지 않는다.

| 작업 | 확인한 내용 | SAC_UED5에 대한 권고 |
|---|---|---|
| [Hybrid SAC, 2019](https://arxiv.org/abs/1912.11077), [ALF 공식 구현 설명](https://alf.readthedocs.io/en/latest/notes/sac_with_hybrid_action_types.html) | 혼합 action objective, 계산 가능한 이산 기대값과 entropy 계수 분리 | 지금 가장 직접적. off/guide 열거를 우선 적용 |
| [ACCEL, ICML 2022](https://arxiv.org/abs/2203.01302) | regret 기반 level replay와 level editing | 현재 코드는 ACCEL-style 변형이며 원 논문과 동일한 보장·점수가 아님. dataset에서 켜져 있다고 주장하지 않기 |
| [No Regrets / SFL, NeurIPS 2024](https://arxiv.org/abs/2408.15099) | regret proxy를 검토하고 high-learnability 시나리오 샘플링 제시 | 잘 정의된 반복 성공률을 먼저 사용. 전체 실패인 초기 단계에서는 p(1-p)만으로 충분하지 않음 |
| [DRED, 2024](https://arxiv.org/abs/2402.03479) | 데이터 분포로 환경 설계를 regularize하여 zero-shot transfer 개선 | OSM 배포가 목적이면 생성 도시 curriculum이 실제 도로 통계에서 멀어지지 않도록 하기 |
| [CrossQ, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f381114cf5aba4e45552869863deaaa7-Paper-Conference.pdf) | critic BatchNorm의 사용 방식과 target network 제거, UTD=1의 sample efficiency | 현재 GroupNorm을 BatchNorm으로 치환하는 것만으로 CrossQ가 되지 않음. 혼합/MARL 경로를 고친 뒤 별도 baseline |
| [CrossQ + weight normalization, 2025](https://arxiv.org/abs/2502.07523) | 높은 UTD에서의 학습 안정성과 plasticity를 다룬 방법 | CPU simulation이 병목인 경우 후보. 먼저 policy lag/critic gradient/실제 UTD를 계측 |
| [BRO, 2024](https://arxiv.org/abs/2405.16158), [BRC, 2025](https://arxiv.org/abs/2505.23150) | regularized 큰 critic, BRC는 categorical value와 task conditioning으로 multi-task interference 대응 | critic 강화 후보. 단순 actor 확대나 scalar regression에 이름만 붙이는 방식은 피함. 단일 과제 성공 이후 비교 |
| [TRACED, ICLR 2026](https://arxiv.org/abs/2506.19997) | transition-prediction error와 task 간 Co-Learnability로 curriculum 점수 보완 | 최신 관련 연구. 논문도 stochastic transition noise와 PPO 외 알고리즘 확장을 한계로 명시. 확률 군중 + SAC에 즉시 효과를 가정하지 않기 |
| [DEGen + MNA, 2026 preprint](https://arxiv.org/abs/2601.14957) | 동적 환경 생성의 teacher credit assignment와 regret approximation 개선 | 정적 hazard/도시 편집 중심인 현재 과제의 즉시 수정 대상은 아님. 동적 hazard 생성으로 확장할 때 연구 후보 |

프레임워크는 **당장 전면 교체보다 작은 독립 기준 구현**을 권한다.

- [BenchMARL 공식 알고리즘 목록](https://benchmarl.readthedocs.io/en/latest/modules/algorithms.html)은 MASAC, MAPPO, ISAC 등을 제공한다. 환경의 reset/step, agent별 관측, 중앙 state, termination/truncation, discount 계약을 정리하고 MASAC 또는 MAPPO 대조를 만들기에 적합하다. 현재 custom hybrid action과 variable-duration 할인은 별도 통합이 필요하다.
- [SB3 SAC 공식 지원표](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)는 Box action을 지원하며 Discrete/Dict action과 recurrent policy는 지원하지 않는다. 현재 action을 그대로 넣는 대체 구현은 아니다. 한 로봇·고정 guide·고정 주기의 연속 waypoint 과제에 대한 기준 실험은 가능하다.
- MAPPO는 on-policy이므로 오래된 teammate/replay 정책 문제를 피하는 비교축이지만 sample efficiency 비용이 있다. 확률 군중 시뮬레이션이 비싸므로 무조건 더 낫다고 권하지 않는다.
- Dreamer 등 model-based 전환은 world model과 부분 관측·다중 로봇·혼합 action까지 새 검증 대상이 늘어난다. 현재의 실행 일관성 문제를 해결하는 최소 수정은 아니다.

## 5. 권장 실험 순서와 통과 기준

1. **실행·평가 계약 수정:** cfg 전달, ID allocator, 안정 Jacobian, waypoint baseline, resume/cache 계약. reward 튜닝과 동시에 하지 않는다.
2. **고정 작은 과제:** 지도 1개·로봇 1대·고정 hazard·고정 2–5초 결정·고정 안내 속도. 먼저 off보다 scripted가 나은지 확인하고, 그다음 SAC가 random보다 나은지 본다. 소수 고정 seed로 학습 가능성만 디버깅하고 최종 일반화 증거로 쓰지 않는다.
3. **혼합 SAC 비교:** 기존 Gumbel 대 정확한 2-mode expectation, 나머지 동일. critic context 추가는 별도 실험. continuous/mode entropy, mean/std, saturation과 TD 오차를 남긴다.
4. **행동·시간 복잡성 복원:** 속도 학습, 가변 결정, 3대 순으로 복원한다. 고정 sample 수와 실제 wall-clock 비용을 함께 비교한다. `stored` teammate action은 MADDPG 계열에서 쓰는 선택이며 그 자체가 버그는 아니다. `current` 비교는 policy lag와 coordination을 계측한 뒤 한다.
5. **일반화와 curriculum:** 세 지도 → augmentation → 지도 다양성. 마지막에 DR/SFL/수정 ACCEL을 동일 compute·seed 조건으로 비교한다. 최종 명동은 모델·설정 선택이 끝난 뒤 사용한다.

한 seed에서만 좋아지는 결과로 결론내리지 않는다. 작은 실험 단계에서는 최소 3개 독립 학습 seed를 권하고, 개발 평가에서는 같은 시나리오의 paired 개선량과 불확실성을 보고한다. 필요한 평가 seed 수는 관측 분산과 실제 비용에 맞춰 늘린다.

추가할 최소 진단 항목은 continuous/mode entropy, alpha 각각, mean/log_std 분포, action 포화, hold/event 분포, robot guide speed, contact/inform/following 수, TD-error 분포와 Q disagreement, policy version lag, replay의 지도·level·나이 분포, learner/queue/checkpoint 시간이다. 이 항목들이 있어야 환경 효과 부족, 탐색 고착, critic 오류, 시스템 병목을 구분할 수 있다.
