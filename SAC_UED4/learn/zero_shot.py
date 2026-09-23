"""Evaluation for outdoor hazard-zone evacuation: validation, auxiliary OSM
report and the final zero-shot, all through one episode function.

docs/outdoor_madrl_redesign.md section 7 and docs/zero_shot_evaluation.md.

  validation   generated levels at fixed seeds (VALIDATION_*), 100/200/400 m.
               The only thing models are selected on.
  auxiliary    the four OSM crops in ZSG_REAL_SITES, a report only.
  final        one pre-registered OSM crop (FINAL_ZERO_SHOT_*), run once on a
               model already fixed, at several pre-registered hazard and crowd
               seeds; see cli/final_zero_shot.py.

Every condition is paired: the trained policy and the zero-command,
signal-off control run on the same level, hazard and seed. The control's
robot bodies stay in the world, so it is not a no-robot condition.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EpisodeMetrics:
    first_empty_step: Optional[int] = None
    held_clear_step: Optional[int] = None
    reentries: int = 0
    occupancy_step_sum: float = 0.0
    outflows: int = 0
    informed_departures: int = 0
    evacuation_departures: int = 0
    informed_trip_outflows: int = 0
    background_departures: int = 0
    stuck_releases: int = 0
    inflows: int = 0
    initial_population: int = 0
    ever_acted: int = 0
    hazard_person_steps: int = 0
    initial_inside: int = 0
    # Entries by persons who had been clear by DANGER_SAFE_MARGIN_M, the
    # definition the rew-v2 reward uses. `reentries` counts every crossing of
    # the line, boundary jitter included.
    reentries_after_clear: int = 0
    active_occupancy_step_sum: float = 0.0
    final_occupancy: float = 0.0
    steps: int = 0

    def observe(self, model, outside_before: set) -> set:
        """Record one post-step state; outflow is not a re-entry."""
        zone = model.danger_zone
        active = {
            person.unique_id: person for person in model.crowds
            if person.type in (0, 1, 2) and not person.dead
        }
        active_ids = set(active)
        departed = getattr(self, "_active_ids", active_ids) - active_ids
        self.outflows += len(departed)
        for person in model.crowds:
            if person.unique_id not in departed:
                continue
            reason = getattr(person, "outflow_reason", None)
            if reason == "evacuation_departure":
                self.evacuation_departures += 1
                self.informed_departures += 1
            elif reason == "informed_trip":
                self.informed_trip_outflows += 1
                self.informed_departures += 1
            elif reason == "background_trip":
                self.background_departures += 1
            elif reason == "stuck_release":
                self.stuck_releases += 1
        self._active_ids = active_ids
        self.inflows = max(0, int(model.total_agents) - self.initial_population)
        self.ever_acted = sum(bool(getattr(person, "ever_acted", False))
                              for person in model.crowds)
        outside = {
            uid for uid, person in active.items()
            if not zone.contains(float(person.xy[0]), float(person.xy[1]))
        }
        self.reentries += len((outside_before & active.keys()) - outside)
        inside = model.agents_in_danger()
        self.final_occupancy = inside / max(1, model.total_agents)
        self.occupancy_step_sum += self.final_occupancy
        self.active_occupancy_step_sum += inside / max(1, len(active_ids))
        self.hazard_person_steps += inside
        self.steps += 1
        if inside == 0 and self.first_empty_step is None:
            self.first_empty_step = int(model.step_count)
        if model.is_cleared_and_held() and self.held_clear_step is None:
            self.held_clear_step = int(model.cleared_at() or model.step_count)
        return outside

    # Step interface for learn.rollout.run_episode.
    def start(self, model) -> "EpisodeMetrics":
        self.initial_population = int(model.total_agents)
        self._active_ids = {person.unique_id for person in model.crowds
                            if person.type in (0, 1, 2) and not person.dead}
        self._outside = _outside_ids(model)
        self._clear = {p.unique_id for p in model.crowds
                       if p.type in (0, 1, 2) and not p.dead
                       and model.is_safe(p.xy)}
        self.initial_inside = int(model.agents_in_danger())
        return self

    def step(self, model) -> None:
        self._outside = self.observe(model, self._outside)
        zone = model.danger_zone
        alive = {p.unique_id: p for p in model.crowds
                 if p.type in (0, 1, 2) and not p.dead}
        inside = {uid for uid, p in alive.items()
                  if zone.contains(float(p.xy[0]), float(p.xy[1]))}
        clear = getattr(self, "_clear", set())
        self.reentries_after_clear += len(clear & inside)
        self._clear = ((clear - inside) | {
            uid for uid, p in alive.items() if model.is_safe(p.xy)}) & set(alive)

    def summary(self) -> dict:
        result = {k: v for k, v in asdict(self).items()}
        result["mean_occupancy"] = self.occupancy_step_sum / max(1, self.steps)
        result["mean_active_occupancy"] = (
            self.active_occupancy_step_sum / max(1, self.steps))
        result["informed_departure_fraction"] = (
            self.informed_departures / max(1, self.ever_acted))
        result["held_clear_success"] = int(self.held_clear_step is not None)
        return result


def _outside_ids(model) -> set:
    zone = model.danger_zone
    return {
        person.unique_id for person in model.crowds
        if person.type in (0, 1, 2) and not person.dead
        and not zone.contains(float(person.xy[0]), float(person.xy[1]))
    }


# ------------------------------------------------------------- one episode

CONDITIONS = ("policy", "off_zero_command")


def build_eval_model(level, seed: int):
    """The level as generated (no augmentation), with seeded randomness."""
    import sim.model as model_module

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    lv = copy.deepcopy(level)
    lv.augmentation = "identity"
    return model_module.FightingModel(
        int(lv.crowd_size), int(lv.width), int(lv.height),
        robot="Q", level=lv)


def evaluate_level(agent, level, seed: int, condition: str, cfg,
                   max_steps: Optional[int] = None) -> dict:
    """One fixed level for a fixed duration.

    Never stopped early, so a re-entry after the first clearing is still
    observed, whatever DANGER_TERMINATION says for training.
    """
    from learn.rollout import run_episode

    if condition not in CONDITIONS:
        raise ValueError(condition)
    if level.danger is None:
        raise ValueError("evaluation level has no hazard zone")
    if not 1 <= int(level.robot_num) <= int(cfg.MAX_ROBOTS):
        raise ValueError("invalid robot count")
    model = build_eval_model(level, seed)
    act_fn = None
    if condition == "policy":
        act_fn = lambda obs, rec: agent.act(obs, deterministic=True)
    tm = EpisodeMetrics().start(model)
    gamma = agent.gamma if agent is not None else float(cfg.GAMMA_START)
    model.should_finish = lambda: False
    res = run_episode(model, cfg, act_fn, gamma=gamma, max_steps=max_steps,
                      seed=seed, task_metrics=tm,
                      needs_obs=act_fn is not None)
    walk = _walkable_m2(model)
    out = dict(res.task)
    out.update({
        "total_reward": res.total_reward,
        **{f"reward_{k}": v for k, v in res.components.items()},
        "command_norm_mean": res.command_norm_mean,
        "robot_speed_mean": res.robot_speed_mean,
        "mode_switch_rate": res.mode_switch_rate,
        **{f"mode_share_{k}": v for k, v in res.mode_share.items()},
        "inference_ms_mean": res.inference_ms_mean,
        "sim_ms_per_step": res.sim_ms_per_step,
        "actual_density": (out.get("initial_population", 0) / walk
                           if walk > 0 else None),
        "steps": res.steps,
    })
    held = out.get("held_clear_step")
    out["clear_step_censored"] = (held if held is not None
                                  else int(max_steps or cfg.MAX_STEPS))
    return out


def _walkable_m2(model) -> float:
    from sim.crowd_density import walkable_area
    return float(walkable_area(model.width, model.height, model.obstacles))


# ----------------------------------------------------------- validation set

VALIDATION_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ued_validation_levels.json")
VALIDATION_CACHE_VERSION = 1


def _validation_spec(cfg) -> dict:
    from configs import validation_seeds
    return {"version": VALIDATION_CACHE_VERSION,
            "seeds": {f"{s}_{d}_{k}": v
                      for (s, d, k), v in validation_seeds(cfg).items()},
            "density": [list(cfg.CROWD_DENSITY_RANGE),
                        {str(k): v for k, v in
                         cfg.UED_DENSITY_BY_SIZE.items()}],
            "crowd_cap": list(cfg.CROWD_SIZE_LIMIT),
            # The generator draws each validation level's task from the UED
            # parameters, so a change there is a different validation set.
            "task": {k: _jsonable(cfg[k]) for k in (
                "UED_DANGER_AREA_RANGE", "UED_DANGER_SHAPES",
                "UED_DANGER_INSIDE_FRACTION", "UED_DANGER_PERCEPTIBILITY",
                "UED_PRIOR_INFORMED_FRACTION", "UED_ROBOT_RANGE",
                "UED_DIFFICULTY_RANGE", "UED_MORPHOLOGY_WEIGHTS")}}


def _jsonable(v):
    return json.loads(json.dumps(v, default=str))


def validation_levels(cfg, rebuild: bool = False) -> List[Tuple[str, object]]:
    """The fixed generated validation levels, built once and cached.

    A cache built under a different seed list or density band is rebuilt, not
    reused: the metric names would otherwise refer to a different set.
    """
    from configs import validation_seeds
    from ued.holdout import _deserialise, _serialise
    from ued.level import generate_city_level

    spec = _validation_spec(cfg)
    if os.path.exists(VALIDATION_CACHE) and not rebuild:
        try:
            with open(VALIDATION_CACHE) as fh:
                payload = json.load(fh)
            if payload.get("spec") == spec:
                levels = _deserialise(payload["levels"])
                return list(zip(payload["names"], levels))
        except Exception as exc:
            print(f"[validation] cache unreadable ({exc}); rebuilding")
    names, levels = [], []
    for (size, difficulty, k), seed in sorted(validation_seeds(cfg).items()):
        lv = generate_city_level(rng=random.Random(seed),
                                 difficulty=int(difficulty), crowd_size=None,
                                 width=int(size), height=int(size), seed=seed)
        lv.augmentation = "identity"
        names.append(f"val_{size}m_d{difficulty}_{k}")
        levels.append(lv)
    with open(VALIDATION_CACHE + ".tmp", "w") as fh:
        json.dump({"spec": spec, "names": names,
                   "levels": _serialise(levels)}, fh)
    os.replace(VALIDATION_CACHE + ".tmp", VALIDATION_CACHE)
    return list(zip(names, levels))


def validation_seed(name: str) -> int:
    """The crowd seed a validation scenario always runs with."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 5000 + 955_000


# --------------------------------------------------------------- summaries

SUMMARY_KEYS = ("hazard_person_steps", "reentries", "reentries_after_clear",
                "held_clear_success", "clear_step_censored", "mean_occupancy",
                "mean_active_occupancy", "final_occupancy", "outflows",
                "inflows", "evacuation_departures", "informed_trip_outflows",
                "background_departures", "stuck_releases", "actual_density",
                "inference_ms_mean", "total_reward", "command_norm_mean",
                "robot_speed_mean", "mode_switch_rate")


def paired_records(agent, scenarios, cfg, episode: int, seeds_for,
                   robot_counts, off_cache: Optional[dict] = None,
                   max_steps: Optional[int] = None) -> List[dict]:
    """Run policy and control for each scenario, robot count and seed.

    The control does not depend on the policy, so a cache keyed by level,
    robot count and seed lets periodic validation compute it once.
    """
    records = []
    for name, original in scenarios:
        for robots in robot_counts:
            for seed in seeds_for(name):
                for condition in CONDITIONS:
                    key = (name, int(robots), int(seed), int(max_steps or 0))
                    if (condition == "off_zero_command"
                            and off_cache is not None and key in off_cache):
                        result = off_cache[key]
                    else:
                        level = copy.deepcopy(original)
                        level.robot_num = int(robots)
                        result = evaluate_level(agent, level, seed, condition,
                                                cfg, max_steps=max_steps)
                        if (condition == "off_zero_command"
                                and off_cache is not None):
                            off_cache[key] = result
                    records.append({
                        "episode": int(episode), "scenario": name,
                        "size_m": int(original.width),
                        "difficulty": getattr(original, "difficulty", None),
                        "morphology": getattr(original, "morphology", None),
                        "robot_num": int(robots), "seed": int(seed),
                        "condition": condition,
                        "hazard_shape": (original.danger.shape
                                         if original.danger else None),
                        **result,
                    })
    return records


def summarise(records: List[dict], prefix: str) -> Dict[str, float]:
    """Means per size, robot count and condition, plus paired differences
    against the control.

    Failures stay in every mean: a failed clearance is censored at the
    horizon, never dropped, and success rates are reported beside times.
    """
    out: Dict[str, float] = {}
    groups: Dict[Tuple, List[dict]] = {}
    for r in records:
        groups.setdefault((r["size_m"], r["robot_num"], r["condition"]),
                          []).append(r)
    for (size, robots, cond), rows in sorted(groups.items()):
        base = f"{prefix}/{size}m/robots_{robots}/{cond}"
        for k in SUMMARY_KEYS:
            vals = [float(r[k]) for r in rows if r.get(k) is not None]
            if vals:
                out[f"{base}/{k}"] = float(np.mean(vals))
        out[f"{base}/n"] = float(len(rows))
    by_key: Dict[Tuple, Dict[str, dict]] = {}
    for r in records:
        by_key.setdefault((r["scenario"], r["robot_num"], r["seed"]),
                          {})[r["condition"]] = r
    reductions, reentry_diff = [], []
    per_size: Dict[int, List[float]] = {}
    for pair in by_key.values():
        if set(pair) != set(CONDITIONS):
            continue
        p, o = pair["policy"], pair["off_zero_command"]
        red = 1.0 - (float(p["hazard_person_steps"])
                     / max(1.0, float(o["hazard_person_steps"])))
        reductions.append(red)
        per_size.setdefault(int(p["size_m"]), []).append(red)
        reentry_diff.append(float(p["reentries_after_clear"])
                            - float(o["reentries_after_clear"]))
    if reductions:
        out[f"{prefix}/paired/person_steps_reduction_vs_off"] = float(
            np.mean(reductions))
        out[f"{prefix}/paired/reentries_after_clear_minus_off"] = float(
            np.mean(reentry_diff))
        for size, vals in sorted(per_size.items()):
            out[f"{prefix}/paired/{size}m/person_steps_reduction_vs_off"] = \
                float(np.mean(vals))
    return out


def selection_score(summary: Dict[str, float], prefix: str) -> Optional[float]:
    """The single number models are selected on: the mean paired reduction
    in hazard person-steps against the control, over all validation pairs."""
    return summary.get(f"{prefix}/paired/person_steps_reduction_vs_off")


def run_validation(agent, cfg, episode: int, off_cache: Optional[dict] = None,
                   max_steps: Optional[int] = None,
                   output_dir: Optional[str] = None) -> Dict[str, float]:
    scenarios = validation_levels(cfg)
    records = paired_records(agent, scenarios, cfg, episode,
                             lambda name: (validation_seed(name),),
                             cfg.VALIDATION_ROBOT_COUNTS, off_cache=off_cache,
                             max_steps=max_steps)
    if output_dir:
        append_jsonl(os.path.join(output_dir, "validation_metrics.jsonl"),
                     records)
    return summarise(records, "eval/validation")


def append_jsonl(path: str, records: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------- auxiliary OSM

def auxiliary_levels(cfg):
    from osm_corpus.export import load_levels
    from sim.scenario import attach_hazard

    crops = {(lv.site_key, int(lv.width)): lv for lv in load_levels()}
    for index, site in enumerate(cfg.ZSG_REAL_SITES):
        key = (site, int(cfg.ZSG_REAL_SIZE))
        if key not in crops:
            raise ValueError(f"missing OSM crop {key}")
        level = attach_hazard(copy.deepcopy(crops[key]),
                              rng=random.Random(902_000 + index))
        yield f"osm_{site}_{cfg.ZSG_REAL_SIZE}m", level


def run_auxiliary(agent, cfg, episode: int, output_dir: str,
                  max_steps: Optional[int] = None) -> Dict[str, float]:
    scenarios = list(auxiliary_levels(cfg))
    seeds_for = lambda name: tuple(
        episode * 1_000_003 + it * 1009 + sum(map(ord, name)) * 37
        for it in range(int(cfg.ZSG_ITERATION)))
    records = paired_records(agent, scenarios, cfg, episode, seeds_for,
                             cfg.ZSG_ROBOT_NUM, max_steps=max_steps)
    append_jsonl(os.path.join(output_dir, "auxiliary_osm_metrics.jsonl"),
                 records)
    return summarise(records, "eval/auxiliary_osm")


# ------------------------------------------------------ final zero-shot

def final_base_level(cfg):
    """The pre-registered crop with its headcount at the target density."""
    from osm_corpus.export import load_levels
    from sim.crowd_density import crowd_size_for, walkable_area

    for lv in load_levels():
        if (lv.site_key == cfg.FINAL_ZERO_SHOT_SITE
                and int(lv.width) == int(cfg.FINAL_ZERO_SHOT_SIZE_M)):
            level = copy.deepcopy(lv)
            break
    else:
        raise ValueError("final zero-shot crop not in the corpus")
    walk = walkable_area(level.width, level.height, level.obstacles)
    stored = {"crowd_size": int(level.crowd_size),
              "density": float(level.crowd_size) / max(1.0, walk)}
    if cfg.FINAL_ZERO_SHOT_DENSITY is not None:
        level.crowd_size = crowd_size_for(
            walk, density=float(cfg.FINAL_ZERO_SHOT_DENSITY))
    level.crowd_density = float(level.crowd_size) / max(1.0, walk)
    level.exits = []
    level.augmentation = "identity"
    return level, {"walkable_m2": walk, "stored": stored,
                   "used_crowd_size": int(level.crowd_size),
                   "used_density": level.crowd_density,
                   "low_density": cfg.FINAL_ZERO_SHOT_DENSITY is None}


def draw_final_hazard(level, seed: int, cfg):
    """The hazard for one pre-registered seed and the pre-evaluation checks.

    Returns (level with hazard, or None if rejected, check record). The
    shape recorded is the shape placed: a rectangle is reported as one.
    """
    from sim.danger import sample_zone

    rng = random.Random(int(seed))
    lo, hi = cfg.FINAL_ZERO_SHOT_HAZARD_AREA_RANGE
    area = rng.uniform(float(lo), float(hi))
    shapes = tuple(cfg.FINAL_ZERO_SHOT_HAZARD_SHAPES)
    shape = shapes[rng.randrange(len(shapes))]
    zone = sample_zone(rng, float(level.width), float(level.height), area,
                       shape)
    lv = copy.deepcopy(level)
    lv.danger = zone
    lv.robot_num = 1
    check = {"seed": int(seed), "shape": zone.shape,
             "area_fraction": area, "zone": zone.to_dict()}
    model = build_eval_model(lv, int(seed))
    walk_inside = _walkable_inside(model)
    unreachable = 0
    for mesh, d in model.mesh_danger.items():
        cx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
        cy = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
        if d >= 1e5 and zone.contains(cx, cy):
            unreachable += 1
    walk = _walkable_m2(model)
    density = float(model.total_agents) / max(1.0, walk)
    check.update({"walkable_inside_m2": walk_inside,
                  "unreachable_inside_triangles": int(unreachable),
                  "effective_density": density,
                  "initial_inside": int(model.agents_in_danger())})
    reasons = []
    if walk_inside < float(cfg.FINAL_ZERO_SHOT_MIN_WALKABLE_M2):
        reasons.append("too_little_walkable_ground_inside")
    if unreachable > 0:
        reasons.append("walkable_ground_inside_has_no_route_to_safety")
    dlo, dhi = cfg.FINAL_ZERO_SHOT_DENSITY_BAND
    if not (cfg.FINAL_ZERO_SHOT_DENSITY is None or dlo <= density <= dhi):
        reasons.append("effective_density_outside_band")
    check["rejected"] = reasons
    return (None if reasons else lv), check


def _walkable_inside(model) -> float:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    zone = model.danger_zone.polygon()
    polys = [Polygon(p).buffer(0) for p in model.obstacles if len(p) >= 3]
    free = zone if not polys else zone.difference(unary_union(polys))
    return float(free.area)
