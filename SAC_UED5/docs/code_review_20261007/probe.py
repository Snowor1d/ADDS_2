"""Read-only, small reproductions for the 2026-10-07 review.

From SAC_UED5: python3 docs/code_review_20261007/probe.py
No training, downloads, checkpoint writes, or external logging.
The ID check builds a real open-field simulator with 1,001 people.
"""

from __future__ import annotations

import collections
import json
import math
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/adds-review-mpl")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F


def config_probe():
    from configs import resolve_config
    import sim.agent as agent_module
    import sim.model as model_module
    from ued.runner import UEDRunner

    base = resolve_config(check_data=False)
    over = resolve_config({"TRAIN_MAP_SOURCE": "ued",
                           "ROBOT_INFORM_TIME_S": 5.0,
                           "ROBOT_START_RING_M": (1.0, 2.0)},
                          check_data=False)
    changed_reward = resolve_config({"REWARD_W_COLLISION": 0.1},
                                    check_data=False)
    return {
        "default": {k: base[k] for k in (
            "TRAIN_MAP_SOURCE", "DATASET_SITES", "DATASET_SIZES_M",
            "DATASET_DANGER_PERCEPTIBILITY", "DATASET_PRIOR_INFORMED_FRACTION",
            "ROBOT_ACTION_MODE", "PERIODIC_VALIDATION", "BUFFER_SIZE",
            "UPDATES_PER_TRANSITION", "REWARD_VERSION")},
        "override": {
            "requested_source": over.TRAIN_MAP_SOURCE,
            "actual_ued_enabled": UEDRunner().enabled,
            "requested_inform_seconds": over.ROBOT_INFORM_TIME_S,
            "actual_inform_seconds": agent_module.ROBOT_INFORM_TIME_S,
            "requested_start_ring": over.ROBOT_START_RING_M,
            "actual_start_ring": model_module.ROBOT_START_RING_M,
        },
        "changed_reward_not_distinguished_by_load_checks": {
            "versions_equal": base.schema_versions() == changed_reward.schema_versions(),
            "observation_schemas_equal": (
                base.observation_schema() == changed_reward.observation_schema()),
            "fingerprints_equal": base.fingerprint == changed_reward.fingerprint,
        },
    }


def id_probe():
    from ued.level import Level
    from sim.danger import DangerZone
    from sim.model import FightingModel

    random.seed(7)
    np.random.seed(7)
    level = Level([], [], 1001, width=200, height=200,
                  danger=DangerZone("circle", 100, 100, radius=30), robot_num=3)
    model = FightingModel(1001, 200, 200, robot="Q", level=level)
    counts = collections.Counter(a.unique_id for a in model.agents)
    person = next(a for a in model.crowds if a.unique_id == 1000)
    return {
        "people": len(model.crowds), "robots": len(model.robots),
        "all_agents": len(model.agents), "scheduled_agents": len(model.schedule.agents),
        "duplicate_ids": [k for k, v in counts.items() if v > 1],
        "person_1000_scheduled": person in model.schedule.agents,
        "person_1000_in_space": model.space.get_body_by_agent(person) is not None,
    }


def squash_probe():
    # Same continuous log-probability arithmetic as PolicyNetwork.sample_action.
    # Hold a reparameterized noise draw fixed to compare the two Jacobians.
    results = []
    for value in (5.0, 20.0, 40.0):
        mean = torch.tensor(value, requires_grad=True)
        u = mean + 0.2
        sig = torch.sigmoid(u)
        gaussian = -0.5 * (u - mean).square()
        old = gaussian - torch.log(4 * sig * (1 - sig) + 1e-8)
        stable = gaussian - (
            math.log(4) + F.logsigmoid(u) + F.logsigmoid(-u))
        old_grad = torch.autograd.grad(old, mean, retain_graph=True)[0]
        stable_grad = torch.autograd.grad(stable, mean)[0]
        results.append({"mean": value, "current_logp_without_normal_constant": float(old),
                        "stable_logp_without_normal_constant": float(stable),
                        "current_mean_gradient": float(old_grad),
                        "stable_mean_gradient": float(stable_grad)})
    return results


def geometry_probe():
    from citygen.validate import _zone_masks
    from sim.danger import DangerZone

    zone = DangerZone("rect", 20, 20, half_w=12, half_h=2, angle=math.pi / 2)
    mask, _ = _zone_masks(SimpleNamespace(width=40, height=40), zone, (80, 80))
    rows, cols = np.mgrid[0:80, 0:80]
    xs, ys = (cols + 0.5) * 0.5, (rows + 0.5) * 0.5
    actual = np.array([[zone.contains(float(x), float(y)) for x, y in zip(xx, yy)]
                       for xx, yy in zip(xs, ys)])
    return {"rotated_rectangle_disagreeing_cells": int(np.count_nonzero(mask != actual))}


def curriculum_probe():
    from ued.runner import UEDRunner, _EpisodeTrace
    from ued.population import LevelPopulation
    from ued.level import Level

    runner = UEDRunner(value_fn=lambda samples: np.full(len(samples), -100.0))
    trace = _EpisodeTrace()
    trace.level_id, trace.ret, trace.samples = 1, -100.0, [0]
    first_score = runner._maxmc(trace)
    runner._global_best_return = 0.0
    second_score = runner._maxmc(trace)

    pop = LevelPopulation(score_fn="hybrid")
    pop.add(Level([], [], 10, level_id=1))
    pop.add(Level([], [], 10, level_id=2))
    for _ in range(5):
        pop.update(1, evac_time=100, freeflow_steps=100)
    pop.update(2, evac_time=100, freeflow_steps=100, maxmc=10)
    records = list(pop._records.values())

    # Reproduce the supported arrival order across the two separate queues:
    # summary first, then a late transition. No multiprocessing timing needed.
    runner.enabled = True
    runner.epsilon = 0.0
    runner.on_episode(SimpleNamespace(worker_id=0, episode_idx=0,
                                     level_id=77, abnormal=0,
                                     evac_time_100=100, freeflow_steps=100))
    runner.on_transition(SimpleNamespace(episode_key=(0, 0), level_id=77,
                                        reward=-1.0, value_sample=0))
    return {
        "same_trace_score_before_other_easy_level": first_score,
        "same_trace_score_after_other_easy_level": second_score,
        "hybrid_raw_scores": [pop._raw_score(r) for r in records],
        "hybrid_rank_weights": pop._ranked_scores(records),
        "trace_recreated_after_summary": (0, 0) in runner._traces,
    }


def budget_probe():
    from configs import resolve_config
    from learn.replay import ReplayBuffer, StaticStore
    from sim.observation import obs_shapes

    cfg = resolve_config(check_data=False)
    small_buffer = ReplayBuffer(cfg, 1, StaticStore(None))
    gamma_step = cfg.gamma_per_step()
    return {"replay_bytes_per_record": small_buffer.nbytes(),
            "configured_dynamic_replay_gib": small_buffer.nbytes() * cfg.BUFFER_SIZE / 2**30,
            "gamma_per_step": gamma_step,
            "discount_half_life_seconds": math.log(0.5) / math.log(gamma_step) * cfg.AGENT_TIME_STEP,
            "discount_weight_at_300_seconds": gamma_step ** (300 / cfg.AGENT_TIME_STEP),
            "observation_shapes": obs_shapes(cfg)}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps({"configuration": config_probe(), "agent_ids": id_probe(),
                      "squash": squash_probe(), "geometry": geometry_probe(),
                      "curriculum": curriculum_probe(), "budget": budget_probe()},
                     indent=2, ensure_ascii=False))
