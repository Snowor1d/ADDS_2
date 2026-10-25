# Critic 진단: 반사실 probe와 오프라인 testbed (2026-10-22)

대상: ul6kds66 (A0/P150, shared head, γ=0.995) ep3800 checkpoint와 그 replay (954k records).

## 1. 반사실 probe (`true_adv_v2.py`)

검증 시나리오 9개 × 분기 시점 2개(40 s, 160 s) = 18 상태. 로봇 0이 후보 행동 8개 중 하나로 한 결정을 하고, 이후에는 모든 로봇이 확률적 정책을 따라 에피소드 끝까지 진행한다. 반복 3회(효과가 커 보인 6개 상태는 10회), 반복 안에서는 공통 난수.
결정 시점·보상·할인은 `run_episode`와 같다(`check_engine.py`: 240 step 보상 차이 0.0).

| 항목 | 값 |
|---|---:|
| 행동 효과, 상태별 검정 Fisher 결합 | p = 0.001 |
| 효과가 분명한 상태 (p<0.05) | 3 / 18 (s3_t320, s2_t320, s7_t320) |
| 잡음 보정 행동 효과 sd (평균) | 6.5 |
| 롤아웃 1회 return 잡음 sd (평균) | 39 |
| checkpoint critic: 행동 간 Q sd / 실제 sd | 0.27 |
| checkpoint critic: 실제 순위와 상관 (효과가 분명한 3개 상태) | −0.40, −0.01, −0.24 |

반복 3회일 때 커 보였던 효과는 반복 10회에서 줄었다(s3_t80 47→17, s6_t80 53→22): 잡음으로 커 보인 상태를 골랐기 때문이다.

## 2. 오프라인 testbed (`testbed_multi.py`)

actor·α·replay를 고정하고 critic만 정책 평가로 학습한다. 다섯 변형이 같은 배치를 쓴다. encoder는 checkpoint에서 시작하고, 행동을 보는 부분은 모든 변형에서 새로 초기화한다. 20,000 업데이트, 변형당 학습 1회.

반복 10회 ground truth로 채점 (`eval_saved.py`). corr_sig3 = 효과가 분명한 3개 상태의 순위 상관 평균.

| 변형 | 5k | 10k | 15k | 20k | 20k 행동 sd 비율 | 20k q1·q2 일치도 | 20k TD MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| checkpoint (450k 업데이트) | −0.22 | | | | 0.27 | 0.22 | |
| mse:1 | −0.34 | −0.13 | −0.08 | −0.18 | 0.24 | 0.45 | 54 |
| mse:3 (n-step 3) | −0.05 | 0.01 | 0.00 | 0.21 | 0.23 | 0.38 | 68 |
| dueling:1 | 0.10 | −0.20 | 0.06 | 0.18 | 0.09 | 0.60 | 57 |
| hlgauss:1 | 0.20 | −0.05 | −0.06 | 0.09 | 1.16 | 0.52 | 161 |
| hlgauss_dueling:1 | 0.35 | 0.01 | 0.05 | 0.07 | 0.89 | 0.30 | 1822 |

- 어느 변형도 순위를 안정적으로 맞히지 못한다. 같은 변형의 checkpoint 사이 변동이 변형 간 차이만큼 크다.
- HL-Gauss는 행동 간 Q 폭을 실제 수준까지 키우지만 순위는 맞지 않는다. hlgauss_dueling은 TD 오차가 발산했다.
- 모든 변형에서 학습이 진행될수록 q1·q2 일치도가 내려간다(1k에서 0.6–0.78).

한계: 변형당 1회 학습, 20k 업데이트, 한 checkpoint의 정책과 replay, 후보 행동 일부(in/out/still)는 replay 분포 밖일 수 있다. 정답이 믿을 만한 상태는 3–6개다.

## 재현

```bash
CK=<checkpoint>; ./run_batch.sh                       # probe 18 상태 (CK 환경변수)
python3 true_adv_v2.py S T0 $CK results/sS_tT0_more.json --reps 7 --rep-start 3
python3 merge_reps.py results; python3 analyze_probe.py results
python3 dump_states.py $CK results
python3 testbed_multi.py --ckpt $CK --replay <replay_buffer.npz> --static-dir <replay_static> \
    --probes $PWD/results --out $PWD/results/testbed.jsonl --save-dir <dir>
python3 eval_saved.py <dir> $CK $PWD/results/eval_saved_final.jsonl $PWD/results
```

출력 경로는 절대경로로 준다(스크립트가 프로젝트 루트로 chdir한다).

## 3. 잡음은 어디서 오나 (`true_adv_persons.py`, `analyze_persons.py`, `exo_rng.py`)

같은 5개 상태(s2_t320, s3_t320, s7_t320, s3_t80, s6_t80), 반복 10회. 로봇 0의 return을 사람별로 분해했고(매 step 보상과의 오차 < 3e-14), 기존 probe를 같은 seed로 정확히 재현했다(차이 0.0).

**사람별 분해.** 행동 효과도 잡음도 집단적이다. 사람별 효과 분산의 합은 전체 효과 분산의 4–22%, 사람별 잡음 분산의 합은 전체 잡음 분산의 16–58%다. 사람별 신호 대 잡음비 중앙값(0.15–0.25)은 팀 전체(0.22–0.73)보다 낮아, 대피자 단위로 critic을 분해하는 안의 전제가 성립하지 않는다.

**로봇의 무작위성.** 분기 이후 로봇을 결정적 정책으로 바꿔도 잡음은 그대로다(전체 sd 비 1.03). 잡음은 군중에서 나온다.

**군중의 외생 무작위성.** probe 안에서만 군중 난수를 (사람, step, 호출 위치)별 흐름으로 바꿔, 한 반복 안의 모든 후보가 같은 외생 난수를 공유하게 했다(`exo_rng.py`). 반복 효과(모든 행동에 공통인 잡음)의 몫이 외생 정보를 아는 critic이 지울 수 있는 상한이다.

| 조건 | 상호작용 잡음 sd | 반복 효과 sd | 반복 효과 몫 | 행동 효과 sd |
|---|---:|---:|---:|---:|
| 확률적 정책 | 60.0 | 7.6 | 0.02 | 21.6 |
| 결정적 정책 | 61.7 | 14.7 | 0.05 | 17.6 |
| 결정적 + 사람 단위 공통 난수 | 52.6 | 35.9 | 0.32 | 19.9 |

반복 효과 몫은 상태별로 0.01–0.43이다. 외생 정보를 완벽히 쓰는 critic도 잡음 분산의 약 1/3만 지울 수 있다(신호 대 잡음비로 약 1.2배). 나머지는 같은 외생 난수라도 첫 행동에 따라 다르게 번지는 상호작용(행동 효과의 이질성)이다.
