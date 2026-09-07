# PPO_FE_RV

`SAC_FE_RV3`의 환경, reward, observation, actor architecture, action repeat,
map augmentation 및 zero-shot 평가 조건을 유지하고 학습 알고리즘만 PPO로
교체한 baseline이다.

## SAC_FE_RV3와 동일한 요소

- 4-frame ego/global observation과 3차원 robot state
- ego/global `CNNEncoder`
- actor MLP `512 -> 256 -> 64`
- state-dependent mean 및 log-standard-deviation head
- sigmoid action transform `4 * sigmoid(u) - 2`
- `ACTION_SCALE`, reward components, map sampling/augmentation 및 평가 설정
- `EGO_USE`, `FiLM_USE`, `ROBOT_STATE_EMBEDDING` ablation flag

PPO의 critic은 정의상 action을 입력받지 않는 `V(s)`이다. SAC Q-network의
action input만 제거하고 CNN, robot embedding, FiLM과 hidden widths를
유지했다. `FiLM_USE=True`일 때 PPO value critic은 robot embedding으로 image
features를 condition한다.

## 표준 PPO 요소

- 한 policy version 단위의 synchronous rollout
- clipped policy objective
- generalized advantage estimation (GAE)
- clipped value loss
- advantage normalization
- entropy bonus와 gradient clipping
- approximate KL, clip fraction 및 target-KL early stopping
- true termination과 time-limit truncation의 분리

epsilon-greedy, replay buffer, SAC alpha, twin Q/target Q 및 random warm-up은
사용하지 않는다.

## 실행

```bash
cd /home/amrl_sunny/ADDS_2/PPO_FE_RV
python3 Start_training.py
```

watchdog 없이 직접 실행하려면 다음을 사용한다.

```bash
python3 ADDS_AS_reinforcement.py
```

기본 learner device는 `config.py`의 `DEVICE="cuda"`이고 rollout worker는
CPU에서 실행된다. 로그와 checkpoint는 `~/Log_PPO_FE_RV_aug`에 저장된다.

## 주요 설정

```python
N_ENVS = 4
PPO_ROLLOUT_STEPS_PER_ENV = 512
PPO_EPOCHS = 10
PPO_MINIBATCH_SIZE = 256
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_VALUE_CLIP_EPS = 0.2
PPO_ENTROPY_COEF = 0.01
PPO_LOGPROB_WARN_TOL = 1e-3
PPO_LOGPROB_FAIL_TOL = 1e-2
PPO_CHECKPOINT_INTERVAL_UPDATES = 5

EGO_USE = True
FiLM_USE = True
ROBOT_STATE_EMBEDDING = True
```

하나의 update batch 크기는
`N_ENVS * PPO_ROLLOUT_STEPS_PER_ENV`이다. 한 policy transition은
`ACTION_SCALE`개의 simulator step에 해당한다.

`FiLM_USE=True`와 `ROBOT_STATE_EMBEDDING=False`의 조합은 value critic의
conditioning input이 없어 의미가 없으므로 config validation에서 거부한다.

## 검증

```bash
pytest -q tests/test_ppo.py tests/test_map_augmentation.py
python3 tests/smoke_multiprocess.py
```

테스트는 GAE terminal/truncation 처리, action bound, old/new log-prob 일치,
FiLM/EGO flag, mixed policy-version 거부, 실제 clipped PPO update 및 실제
환경을 사용하는 multiprocessing rollout을 확인한다.
