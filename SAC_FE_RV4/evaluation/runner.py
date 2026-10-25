"""Run suite scenarios under evaluation conditions in parallel processes.

Conditions:
    no_robot, exit_robot, shuttle_random, shuttle_nearest   evaluation/baselines.py
    policy        the checkpoint, EVAL_POLICY_MODE
    policy_det    the checkpoint, mean action
    policy_stoch  the checkpoint, sampled actions (RV3's experiments used this)

Every condition builds the scenario from its placement seed, then reseeds the
dynamics' random stream from the same seed, so conditions start from the same
crowd and robot and share the crowd's random draws as far as they go.

Baseline results do not depend on any policy and are cached under
~/<EVAL_RESULTS_DIR>/cache, keyed by the suite digest, the condition and its
parameters, and a hash of the simulator sources and physics settings; a code
or setting change that could alter them starts a new cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICY_CONDITIONS = ("policy", "policy_det", "policy_stoch")
BASELINES = ("no_robot", "exit_robot", "shuttle_random", "shuttle_nearest")
SIM_FILES = ("model.py", "agent.py", "space.py", "core.py", "visibility_atlas.py",
             "sim_accel.py", "map_augmentation.py", "evaluation/baselines.py", "learn/rollout.py")
SIM_SETTINGS = ("ACTION_SCALE", "ROBOT_SPEED_NORM_LIMIT", "ROBOT_EDGE_TRAP_FIX", "ROBOT_BODY_RADIUS", "AGENT_BODY_RADIUS", "ROBOT_VISION",
                "AGENT_VISION", "EXIT_CONFIRM_RADIUS", "AGENT_SPEED_MEAN", "ROBOT_SPEED_MAX",
                "AGENT_TIME_STEP", "ROBOT_TIME_STEP", "K1", "K2", "K3", "MAP_SIZE_FROM_FILE",
                "HARDCODED_MAP_SIZES", "EVAL_MAX_STEPS", "EVAL_MAX_STEPS_SCALE",
                "BASELINE_DECISION_INTERVAL", "SHUTTLE_PICKUP_RADIUS_M", "SHUTTLE_EXIT_REACH_M",
                "SHUTTLE_EXIT_WAIT_STEPS", "SHUTTLE_APPROACH_TIMEOUT_STEPS",
                "SHUTTLE_TARGET_FILTER", "SHUTTLE_DISTANCE", "SHUTTLE_IGNORE_NEAR_EXIT_M")


def sim_version(cfg) -> str:
    h = hashlib.sha256()
    for rel in SIM_FILES:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            h.update(fh.read())
    h.update(json.dumps({k: cfg.get(k) for k in SIM_SETTINGS}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def results_root(cfg) -> str:
    p = cfg.EVAL_RESULTS_DIR
    return p if os.path.isabs(p) else os.path.join(os.path.expanduser("~"), p)


# ----------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------

_POLICY = None


def _init(policy_path: Optional[str]):
    global _POLICY
    import torch
    torch.set_num_threads(1)
    if policy_path:
        from learn.policy_io import load_policy
        _POLICY = load_policy(policy_path)


def run_job(sc: dict, condition: str, params: Optional[dict] = None) -> dict:
    from configs import cfg
    from evaluation.baselines import make_controller
    from evaluation.suites import scenario_spec
    from learn.rollout import PolicyController, build_env, eval_step_cap, freeflow_steps, run_episode

    params = dict(params or {})
    spec = scenario_spec(sc)
    t0 = time.time()
    if condition in POLICY_CONDITIONS:
        actor, pcfg, meta = _POLICY
        det = condition == "policy_det" or (condition == "policy" and
                                             cfg.EVAL_POLICY_MODE == "deterministic")
        ctrl = PolicyController(actor, pcfg, deterministic=det, seed=sc["seed"])
        env = build_env(spec, cfg, robot="Q")
        obs_cfg = pcfg
        extra = {}
    else:
        ctrl, rv = make_controller(condition, cfg, seed=sc["seed"], **params)
        env = build_env(spec, cfg, robot=rv)
        obs_cfg = cfg
        extra = {"speed_source": getattr(ctrl, "speed_source", None),
                 "approach": getattr(ctrl, "approach", None), "escort": getattr(ctrl, "escort", None)}
    ff = freeflow_steps(env, cfg)
    random.seed(int(sc["seed"]) + 1)
    np.random.seed((int(sc["seed"]) + 1) % (2 ** 32))
    t_build = time.time() - t0
    res = run_episode(env, ctrl, obs_cfg, max_steps=eval_step_cap(cfg, env))
    res.update(id=sc["id"], map_key=sc["map_key"], group=sc["group"], condition=condition,
               freeflow_steps=ff, build_s=t_build, duration_s=time.time() - t0,
               trips=getattr(ctrl, "trips", None), **extra)
    return res


def _job(args):
    sc, cond, params = args
    try:
        return run_job(sc, cond, params)
    except Exception as e:
        import traceback
        return {"id": sc["id"], "map_key": sc["map_key"], "group": sc["group"],
                "condition": cond, "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-2000:]}


# ----------------------------------------------------------------------------
# main side
# ----------------------------------------------------------------------------

def _cache_file(cfg, suite: dict, condition: str, params: dict) -> str:
    key = json.dumps({"c": condition, "p": params, "speeds": _speed_key(cfg, condition, params)},
                     sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:10]
    d = os.path.join(results_root(cfg), "cache", f"{suite['name']}-{suite['digest']}",
                     f"sim-{sim_version(cfg)}")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{condition}-{h}.jsonl")


def _speed_key(cfg, condition, params):
    if not condition.startswith("shuttle") or params:
        return None
    from evaluation.baselines import load_tuned
    return load_tuned(cfg, condition.split("_")[1])


def _read_cache(path: str) -> Dict[str, dict]:
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "error" not in r:
                    out[r["id"]] = r
    return out


def run_many(cfg, suite: dict, specs: Sequence[tuple], scenarios: Sequence[dict],
             policy_path: Optional[str] = None, workers: Optional[int] = None,
             use_cache: bool = True, log=print, tag: str = "") -> Dict[tuple, List[dict]]:
    """Run every (condition, params) in `specs` on `scenarios` in one pool.
    Returns {(condition, params_json): [results]}; baseline results are read
    from and appended to their cache."""
    out: Dict[tuple, List[dict]] = {}
    jobs = []
    caches = {}
    for cond, p in specs:
        p = dict(p or {})
        key = (cond, json.dumps(p, sort_keys=True))
        out[key] = []
        if cond in BASELINES and use_cache:
            path = _cache_file(cfg, suite, cond, p)
            caches[key] = path
            have = _read_cache(path)
            for sc in scenarios:
                if sc["id"] in have:
                    out[key].append(have[sc["id"]])
                else:
                    jobs.append((sc, cond, p))
        else:
            jobs.extend((sc, cond, p) for sc in scenarios)
    if any(j[1] in POLICY_CONDITIONS for j in jobs) and not policy_path:
        raise ValueError("a policy condition needs policy_path")
    if not jobs:
        return out
    order = {"no_robot": 0, "exit_robot": 1}
    jobs.sort(key=lambda j: order.get(j[1], 2))
    n = int(workers or cfg.EVAL_WORKERS)
    cached = sum(len(v) for v in out.values())
    log(f"[eval] {suite['name']}{tag}: {len(jobs)} episodes on {n} processes ({cached} cached)")
    t0 = time.time()
    import multiprocessing as mp
    done = 0
    with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"),
                             initializer=_init, initargs=(policy_path,)) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for f in as_completed(futs):
            sc, cond, p = futs[f]
            r = f.result()
            r["params"] = p
            key = (cond, json.dumps(p, sort_keys=True))
            out[key].append(r)
            done += 1
            if "error" in r:
                log(f"[eval] ERROR {cond} {r['id']}: {r['error']}")
            elif key in caches:
                with open(caches[key], "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r) + "\n")
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                el = time.time() - t0
                log(f"[eval] {suite['name']}{tag}: {done}/{len(jobs)} "
                    f"({el:.0f}s, ~{el / done * (len(jobs) - done):.0f}s left)")
    return out


def run_suite(cfg, suite: dict, conditions: Sequence[str], policy_path: Optional[str] = None,
              workers: Optional[int] = None, params: Optional[Dict[str, dict]] = None,
              limit: Optional[int] = None, use_cache: bool = True, log=print,
              scenario_filter=None) -> List[dict]:
    """All scenarios of `suite` (optionally filtered / truncated) under each
    condition; returns one flat list of per-episode results."""
    params = params or {}
    scen = suite["scenarios"]
    if scenario_filter is not None:
        scen = [s for s in scen if scenario_filter(s)]
    if limit:
        scen = scen[: int(limit)]
    res = run_many(cfg, suite, [(c, params.get(c, {})) for c in conditions], scen,
                   policy_path=policy_path, workers=workers, use_cache=use_cache, log=log)
    return [r for rs in res.values() for r in rs]
