# SAC_UED6_easy: "can it learn at all?" sanity run

A copy of SAC_UED6 (2026-10-09) with every suspected source of difficulty
removed at once. If the policy cannot beat "robots off" here, the problem is in
the learning algorithm or the reward, not in the environment's difficulty.
If it can, add the difficulties back one at a time (see below).

| Setting | SAC_UED6 | Here | File |
| --- | --- | --- | --- |
| `DATASET_DANGER_AREA_RANGE` | (0.1, 0.3) | (0.04, 0.1) | configs/training/dataset.py |
| `DATASET_DANGER_PERCEPTIBILITY` | (0.05, 0.3) | unchanged, already low | configs/training/dataset.py |
| `DATASET_ROBOT_RANGE` | (3, 3) | (3, 3), kept: see below | configs/training/dataset.py |
| `ACTOR_GLOBAL_CROWD_TRUTH` | False | True (full information) | configs/environment.py |
| `ROBOT_DECISION_ON_EVENTS_WAYPOINT` | True | False | configs/environment.py |
| `ROBOT_DECISION_MAX_S_WAYPOINT` | 30 s | 4 s (fixed-length decisions) | configs/environment.py |
| `REWARD_W_PROJECTION` | 0.2 | 0.0 | configs/training/common.py |
| `LOG_DIR` / `EXPERIMENT_ID` | ... | `Log_SAC_UED6_easy_fullinfo` / `outdoor-madrl-v2-easy-fullinfo` (required by the full-information check) | configs/training/common.py |

Judge it with paired evaluation on fixed seeds (learned policy vs. robots off
vs. the shuttle baseline, `validation/shuttle_baseline.py`), not with the
episode-reward curve, whose seed-to-seed spread is far larger than the effect.

If it learns, put the difficulties back in this order, one per run:
1. fixed 4 s decisions -> event-driven (`ROBOT_DECISION_ON_EVENTS_WAYPOINT`, `ROBOT_DECISION_MAX_S_WAYPOINT`)
2. full -> partial observation (`ACTOR_GLOBAL_CROWD_TRUTH`, and the LOG_DIR/EXPERIMENT_ID back)
3. small -> actual hazard size (`DATASET_DANGER_AREA_RANGE`)

The first step that stops learning is where to spend effort.

## Why three robots, not one (2026-10-09)

The shuttle baseline (`validation/shuttle_baseline.py`) against robots off, on
the three training crops x 3 seeds in this easy setting:

| Robots, perceptibility | Person-time change | Pairs better |
| --- | --- | --- |
| 1, 0.5 | -2.0% +- 9.5% | 5 / 9 |
| 3, 0.5 | -21.0% +- 8.0% | 9 / 9 |
| 3, 0.2 | -35.9% +- 16.6% | 8 / 9 |

One robot leaves almost nothing to learn; three give a large, consistent
effect. A sanity run needs the effect, so the team stays at three.
