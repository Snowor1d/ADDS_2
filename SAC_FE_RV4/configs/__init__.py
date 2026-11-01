"""Merge, override and check the configuration files.

    configs/environment.py        simulator physics and observation
    configs/training/common.py    shared training settings, TRAIN_MAP_SOURCE
    configs/training/dataset.py   dataset mode
    configs/training/ued.py       curriculum mode
    configs/evaluation.py         test suites, baselines, tuning

Every setting is defined in exactly one file. Importing this package merges
them, applies overrides and validates the result; a contradiction stops the
program before any worker starts.

Overrides come from the environment so that spawned worker processes see the
same values as the process that started them:

    RV4_PRESET=presets/x.json      a JSON object of settings
    RV4_OVERRIDES='{"LR": 3e-4}'   applied after the preset

`apply_cli_overrides(argv)` turns ``--preset FILE`` and ``--set KEY=VALUE``
into those variables; entry points call it before importing anything that
reads the configuration. Unknown names are rejected, so a typo cannot silently
do nothing.

`from configs import *` (and the `config.py` shim, `from config import *`)
exports every setting as a module constant; new code reads `cfg`.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECTIONS = (
    ("environment", "configs.environment"),
    ("training.common", "configs.training.common"),
    ("training.dataset", "configs.training.dataset"),
    ("training.ued", "configs.training.ued"),
    ("evaluation", "configs.evaluation"),
)

# Settings that change how a run is labelled or logged, not what it learns.
_NOT_FINGERPRINTED = {
    "WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_GROUP",
    "WANDB_RUN_NAME", "WANDB_TAGS", "WANDB_UPLOAD_CHECKPOINTS",
    "LOG_TRAIN_EVERY_UPDATES", "LOG_TXT_COMPAT", "TENSORBOARD", "PORT_NUM",
    "ENABLE_TIMER", "EVAL_WORKERS", "VALIDATION_WORKERS", "VALIDATION_VIDEO",
    "N_ENVS", "N_ENVS_WARMUP", "DEVICE", "EXPERIMENT_ID", "LOG_DIR",
    "RESUME_MODE", "KEEP_LAST_CHECKPOINTS", "CHECKPOINT_EVERY_EPISODES",
    "SAVE_REPLAY", "SHOW_CONTROLLED_CROWD",
}

# Settings added after runs had started, with the value that reproduces the
# earlier behaviour. At that value they are left out of the fingerprint, so
# checkpoints written before the setting existed still resume.
_FINGERPRINT_DEFAULTS = {
    "DANGER_DISTANCE": "euclidean",
    "DANGER_GEO_RES": 0.5,
    "REWARD_B_2": 0,
    "ROBOT_ACTION_MODE": "velocity",
    "REPLAY_FRAMES_PER_TRANSITION": None,
}
# Settings that only mean something in ROBOT_ACTION_MODE = "waypoint": left
# out of the fingerprint in "velocity" mode whatever their value, so velocity
# runs keep their fingerprints.
_WAYPOINT_ONLY = ("ROBOT_WAYPOINT_RANGE_M", "ROBOT_WAYPOINT_SPEED_MAX", "ROBOT_WAYPOINT_ARRIVAL_M",
                  "ROBOT_DECISION_MAX_S_WAYPOINT", "ROBOT_DECISION_ON_EVENTS_WAYPOINT")

# Settings renamed after runs had started: the fingerprint hashes them under
# the old name, and overrides (presets, --set, RV4_OVERRIDES) may still use it.
_RENAMED = {
    "REWARD_B_1": "REWARD_B",
}

_VALID = {
    "TRAIN_MAP_SOURCE": ("dataset", "ued", "mixed"),
    "OBS_EGO_MODE": ("normalized", "metric"),
    "NET_ENCODER": ("legacy_rv3", "cnn", "impala"),
    "NET_SIZE": ("s", "m", "l", "xl"),
    "NET_NORM": ("batch", "group", "layer", "none"),
    "NET_SPATIAL": ("flatten", "pool"),
    "ENCODER_SHARING": ("separate", "critic"),
    "RESET_SCOPE": ("heads", "all"),
    "REWARD_COLLISION_MODE": ("last_step", "sum"),
    "DANGER_DISTANCE": ("euclidean", "geodesic"),
    "ROBOT_ACTION_MODE": ("velocity", "waypoint"),
    "RESUME_MODE": ("latest", "fresh"),
    "WANDB_MODE": ("online", "offline", "disabled"),
    "WANDB_UPLOAD_CHECKPOINTS": ("none", "selected"),
    "VALIDATION_POLICY_MODE": ("deterministic", "stochastic"),
    "EVAL_POLICY_MODE": ("deterministic", "stochastic"),
    "UED_METHOD": ("dr", "plr", "accel"),
    "UED_SCORE": ("learnability", "maxmc", "pvl", "hybrid", "shuttle_regret"),
    "UED_SUCCESS_REFERENCE": ("freeflow", "shuttle"),
    "SHUTTLE_TARGET_FILTER": ("alive", "uninformed"),
    "SHUTTLE_DISTANCE": ("geodesic", "euclidean"),
    "SCHEDULER_TYPE": ("l", "e"),
}

_TRANSFORMS = ("identity", "rotate_90", "rotate_180", "rotate_270",
               "reflect", "reflect_rotate_90", "reflect_rotate_180",
               "reflect_rotate_270")


class ConfigError(ValueError):
    pass


def _module_settings(modname: str) -> Dict[str, Any]:
    mod = importlib.import_module(modname)
    return {k: copy.deepcopy(getattr(mod, k)) for k in dir(mod)
            if k.isupper() or k in ("FiLM_USE",)}


def _load_files() -> Tuple[Dict[str, Any], Dict[str, str]]:
    values: Dict[str, Any] = {}
    owner: Dict[str, str] = {}
    for section, modname in SECTIONS:
        for k, v in _module_settings(modname).items():
            if k in values:
                raise ConfigError(f"{k} is defined in both {owner[k]} and {section}")
            values[k] = v
            owner[k] = section
    return values, owner


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.isabs(path):
        cand = os.path.join(_ROOT, path)
        path = cand if os.path.exists(cand) else path
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"preset {path} must hold a JSON object")
    data.pop("_comment", None)
    return data


def env_overrides() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    preset = os.environ.get("RV4_PRESET")
    if preset:
        for p in preset.split(os.pathsep):
            if p.strip():
                out.update(_read_json(p.strip()))
    raw = os.environ.get("RV4_OVERRIDES")
    if raw:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ConfigError("RV4_OVERRIDES must be a JSON object")
        out.update(data)
    return out


def _coerce(name: str, value: Any, default: Any) -> Any:
    """JSON has no tuples; keep the type of the file's default."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(tuple(v) if isinstance(v, list) and default and isinstance(default[0], tuple) else v
                     for v in value)
    if isinstance(default, dict) and isinstance(value, dict):
        # JSON object keys are strings; numbered-map dictionaries use ints.
        if default and all(isinstance(k, int) for k in default):
            return {int(k): v for k, v in value.items()}
        return value
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def parse_cli_value(text: str) -> Any:
    """--set KEY=VALUE: a Python/JSON literal, else the plain string."""
    t = text.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(t)
        except Exception:
            pass
    if t in ("None", "null"):
        return None
    return t


def apply_cli_overrides(argv: List[str]) -> List[str]:
    """Consume --preset/--set from argv into the environment; return the
    remaining arguments."""
    rest: List[str] = []
    sets: Dict[str, Any] = json.loads(os.environ.get("RV4_OVERRIDES") or "{}")
    presets: List[str] = [p for p in os.environ.get("RV4_PRESET", "").split(os.pathsep) if p]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--preset", "--set") and i + 1 < len(argv):
            val = argv[i + 1]
            i += 2
        elif a.startswith("--preset=") or a.startswith("--set="):
            a, val = a.split("=", 1)
            i += 1
        else:
            rest.append(a)
            i += 1
            continue
        if a == "--preset":
            presets.append(os.path.abspath(val) if os.path.exists(val) else val)
        else:
            if "=" not in val:
                raise ConfigError(f"--set expects KEY=VALUE, got {val!r}")
            k, v = val.split("=", 1)
            sets[k.strip()] = parse_cli_value(v)
    if presets:
        os.environ["RV4_PRESET"] = os.pathsep.join(presets)
    if sets:
        os.environ["RV4_OVERRIDES"] = json.dumps(sets)
    if presets or sets:
        refresh()
    return rest


def refresh() -> "Config":
    """Re-resolve from the environment and update the exported constants
    here and in the `config` shim. Entry points call it (through
    apply_cli_overrides) before importing modules that read settings at
    import time; a module that already did `from config import *` keeps the
    values it copied."""
    import sys
    global cfg
    new = resolve_config()
    cfg = new
    globals().update(new.as_dict())
    shim = sys.modules.get("config")
    if shim is not None:
        shim.__dict__.update(new.as_dict())
        shim.cfg = new
    return new


class Config:
    """Read-only merged settings. Attribute access, plus as_dict()."""

    def __init__(self, values: Mapping[str, Any], owner: Mapping[str, str],
                 overridden: Iterable[str]):
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))
        object.__setattr__(self, "_owner", MappingProxyType(dict(owner)))
        object.__setattr__(self, "_overridden", tuple(sorted(overridden)))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        raise AttributeError("Config is read-only; use resolve_config(overrides=...)")

    def __contains__(self, name):
        return name in self._values

    def get(self, name, default=None):
        return self._values.get(name, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._values)

    def by_section(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in self._values.items():
            out.setdefault(self._owner.get(k, "?"), {})[k] = v
        return out

    @property
    def overridden(self) -> Tuple[str, ...]:
        return self._overridden

    def fingerprint(self) -> str:
        velocity = self._values.get("ROBOT_ACTION_MODE", "velocity") == "velocity"
        data = {_RENAMED.get(k, k): v for k, v in self._values.items() if k not in _NOT_FINGERPRINTED
                and not (k in _FINGERPRINT_DEFAULTS and v == _FINGERPRINT_DEFAULTS[k])
                and not (velocity and k in _WAYPOINT_ONLY)}
        blob = json.dumps(data, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    # -- derived values used in several places ---------------------------
    def lr_actor(self) -> float:
        return float(self.LR_ACTOR if self.LR_ACTOR is not None else self.LR)

    def lr_critic(self) -> float:
        return float(self.LR_CRITIC if self.LR_CRITIC is not None else self.LR)

    def action_dim(self) -> int:
        """2 (velocity) or 3 (waypoint: target offset x, y and speed)."""
        return 3 if self.ROBOT_ACTION_MODE == "waypoint" else 2

    def decision_max_ticks(self) -> int:
        """Longest decision in ACTION_SCALE windows (1 in velocity mode)."""
        if self.ROBOT_ACTION_MODE != "waypoint":
            return 1
        tick_s = int(self.ACTION_SCALE) * float(self.AGENT_TIME_STEP)
        return max(1, int(round(float(self.ROBOT_DECISION_MAX_S_WAYPOINT) / tick_s)))

    def obs_channels(self) -> int:
        return int(self.FRAME_STACK) + (1 if self.OBS_STATIC_DISTANCE_CHANNEL else 0)

    def actor_state_dim(self) -> int:
        return 3 + (2 if self.MAP_SCALE_TO_ACTOR else 0)

    def critic_state_dim(self) -> int:
        return 3 + (2 if self.MAP_SCALE_TO_CRITIC else 0)


def _validate(v: Dict[str, Any]) -> None:
    errs: List[str] = []
    for k, allowed in _VALID.items():
        if k in v and v[k] not in allowed:
            errs.append(f"{k}={v[k]!r} must be one of {allowed}")
    for k in ("DATASET_AUGMENTATION_TRANSFORMS", "UED_AUGMENTATION_TRANSFORMS",
              "MAP_AUGMENTATION_TRANSFORMS"):
        t = v.get(k)
        if not isinstance(t, tuple) or not t or any(x not in _TRANSFORMS for x in t):
            errs.append(f"{k} must be a non-empty tuple of {_TRANSFORMS}")
    for k in ("DATASET_CROWD_RANGE", "UED_CROWD_RANGE", "UED_CROWD_LIMITS",
              "UED_DIFFICULTY_RANGE", "UED_MUTATIONS_PER_CHILD"):
        r = v.get(k)
        if not (isinstance(r, tuple) and len(r) == 2 and r[0] <= r[1]):
            errs.append(f"{k} must be a (low, high) tuple with low <= high")
    if not isinstance(v.get("UED_MAP_SIZES_M"), tuple) or not v["UED_MAP_SIZES_M"]:
        errs.append("UED_MAP_SIZES_M must be a non-empty tuple (a one-element tuple needs its comma)")
    w = v.get("UED_MAP_SIZE_WEIGHTS")
    if w is not None and len(w) != len(v["UED_MAP_SIZES_M"]):
        errs.append("UED_MAP_SIZE_WEIGHTS must match UED_MAP_SIZES_M")
    if not v.get("DATASET_MAPS"):
        errs.append("DATASET_MAPS must not be empty")
    dw = v.get("DATASET_MAP_WEIGHTS")
    if dw is not None and len(dw) != len(v["DATASET_MAPS"]):
        errs.append("DATASET_MAP_WEIGHTS must match DATASET_MAPS")
    if v["CRITIC_TARGET_SUBSET"] > v["CRITIC_ENSEMBLE"] or v["CRITIC_TARGET_SUBSET"] < 1:
        errs.append("CRITIC_TARGET_SUBSET must be in [1, CRITIC_ENSEMBLE]")
    if v["CRITIC_ENSEMBLE"] < 1:
        errs.append("CRITIC_ENSEMBLE must be >= 1")
    if v["N_STEP"] < 1:
        errs.append("N_STEP must be >= 1")
    if v["NET_ENCODER"] == "legacy_rv3":
        bad = []
        if v["ENCODER_SHARING"] != "separate":
            bad.append("ENCODER_SHARING='separate'")
        if v["MAP_SCALE_TO_ACTOR"] or v["MAP_SCALE_TO_CRITIC"]:
            bad.append("MAP_SCALE_TO_*=False")
        if v["OBS_STATIC_DISTANCE_CHANNEL"]:
            bad.append("OBS_STATIC_DISTANCE_CHANNEL=False")
        if v["CRITIC_ENSEMBLE"] != 2:
            bad.append("CRITIC_ENSEMBLE=2")
        if v["CRITIC_LAYERNORM"] or v["CRITIC_DROPOUT"]:
            bad.append("CRITIC_LAYERNORM=False, CRITIC_DROPOUT=0")
        if bad:
            errs.append("NET_ENCODER='legacy_rv3' reproduces the RV3 network and needs "
                        + ", ".join(bad))
    if v["AUG_DRQ_K"] < 1 or v["AUG_DRQ_M"] < 1:
        errs.append("AUG_DRQ_K and AUG_DRQ_M must be >= 1")
    if (v["AUG_DRQ_K"] > 1 or v["AUG_DRQ_M"] > 1) and not (v["AUG_RANDOM_SHIFT"] or v["AUG_REPLAY_D4"]):
        errs.append("AUG_DRQ_K/M > 1 averages over augmentations; enable AUG_RANDOM_SHIFT or AUG_REPLAY_D4")
    if not 0.0 <= v["SHRINK_PERTURB_ALPHA"] < 1.0:
        errs.append("SHRINK_PERTURB_ALPHA must be in [0, 1)")
    if not 0.0 <= v["MIXED_DATASET_PROB"] <= 1.0:
        errs.append("MIXED_DATASET_PROB must be in [0, 1]")
    if v["OBS_FOLLOWER_PIXEL"] is not None and not (0 < int(v["OBS_FOLLOWER_PIXEL"]) < 256):
        errs.append("OBS_FOLLOWER_PIXEL must be None or an int in 1..255")
    lim = v.get("ROBOT_SPEED_NORM_LIMIT")
    if lim is not None and not (0 < float(lim) <= 2.0 * 2 ** 0.5):
        errs.append("ROBOT_SPEED_NORM_LIMIT must be None or in (0, 2.83]")
    if v.get("ROBOT_ACTION_MODE") == "waypoint":
        tick_s = int(v["ACTION_SCALE"]) * float(v["AGENT_TIME_STEP"])
        ratio = float(v["ROBOT_DECISION_MAX_S_WAYPOINT"]) / tick_s
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
            errs.append(f"ROBOT_DECISION_MAX_S_WAYPOINT must be a positive multiple of {tick_s:g} s "
                        "(decisions end on the ACTION_SCALE clock)")
        if v["NET_ENCODER"] == "legacy_rv3":
            errs.append("ROBOT_ACTION_MODE='waypoint' needs a 3-number action; legacy_rv3 has 2")
        if float(v["GAMMA_START"]) != float(v["GAMMA_END"]):
            errs.append("ROBOT_ACTION_MODE='waypoint' needs a constant discount (GAMMA_START == GAMMA_END): "
                        "workers discount the 2 s rewards inside a decision with it")
        for k in ("ROBOT_WAYPOINT_RANGE_M", "ROBOT_WAYPOINT_SPEED_MAX", "ROBOT_WAYPOINT_ARRIVAL_M"):
            if not float(v[k]) > 0:
                errs.append(f"{k} must be > 0")
    b2 = v.get("REWARD_B_2")
    if not (b2 == "auto" or (isinstance(b2, (int, float)) and not isinstance(b2, bool) and b2 >= 0)):
        errs.append("REWARD_B_2 must be 0 (off), a K > 0 or 'auto'")
    elif b2 and float(v["GAMMA_START"]) != float(v["GAMMA_END"]):
        errs.append("REWARD_B_2 (potential shaping) needs a constant discount (GAMMA_START == GAMMA_END): "
                    "the shaping term uses the learner's gamma and is computed in the workers")
    res = v.get("DANGER_GEO_RES")
    if res is not None and not (isinstance(res, (int, float)) and 0.1 <= res <= 2.0):
        errs.append("DANGER_GEO_RES must be in [0.1, 2.0] m")
    if v["EGO_MAP_SIZE"] % 2 != 1:
        errs.append("EGO_MAP_SIZE must be odd so the robot sits on the centre pixel")
    if v["FRAME_STACK"] < 1:
        errs.append("FRAME_STACK must be >= 1")
    if v["UPDATES_PER_TRANSITION"] < 0:
        errs.append("UPDATES_PER_TRANSITION must be >= 0")
    if set(v["UED_MUTATION_WEIGHTS"]) - {"move_obstacle", "scale_obstacle", "add_obstacle",
                                         "remove_obstacle", "move_exit", "resize_exit", "crowd"}:
        errs.append("UED_MUTATION_WEIGHTS has an unknown edit")
    specs = v["SUITE_SPECS"]
    for key in ("VALIDATION_SUITE", "TUNE_SUITE"):
        if v[key] not in specs:
            errs.append(f"{key}={v[key]!r} is not a suite in SUITE_SPECS")
    for s in v["EVAL_HEADLINE_SUITES"]:
        if s not in specs:
            errs.append(f"EVAL_HEADLINE_SUITES: {s!r} is not a suite in SUITE_SPECS")
    # Maps chosen for validation or tuning may not be test maps.
    def _numbered(name):
        sp = specs.get(name, {})
        m = sp.get("maps")
        return set(m) if sp.get("kind") == "numbered" and m != "auto" else set()
    tests = [n for n in specs if n.startswith("test")] + list(v["EVAL_HEADLINE_SUITES"])
    test_maps = set().union(*[_numbered(n) for n in tests]) if tests else set()
    for key in ("VALIDATION_SUITE", "TUNE_SUITE"):
        both = _numbered(v[key]) & test_maps
        if both:
            errs.append(f"{key} ({v[key]}) shares maps with the test suites: {sorted(both)}")
    gen = sorted((int(sp["seed_base"]), n) for n, sp in specs.items() if sp.get("kind") == "generated")
    for (a, na), (b, nb) in zip(gen, gen[1:]):
        if b - a < 100_000:
            errs.append(f"suites {na} and {nb}: seed_base values {a} and {b} are less than "
                        "100000 apart, so their generator seeds can overlap")
    lo_hi = [r for r in v["UED_RESERVED_SEED_RANGES"]]
    for name, spec in v["SUITE_SPECS"].items():
        if spec.get("kind") == "generated":
            base = int(spec["seed_base"])
            if not any(lo <= base < hi for lo, hi in lo_hi):
                errs.append(f"suite {name}: seed_base {base} is outside UED_RESERVED_SEED_RANGES, "
                            "so training could draw the same maps")
    for k, val in v.items():
        if isinstance(val, str) and any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            errs.append(f"{k} looks like a credential; keep secrets out of the config")
    if errs:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errs))


def resolve_config(overrides: Optional[Mapping[str, Any]] = None,
                   use_env: bool = True) -> Config:
    values, owner = _load_files()
    over: Dict[str, Any] = {}
    if use_env:
        over.update(env_overrides())
    if overrides:
        over.update(overrides)
    old_names = {o: n for n, o in _RENAMED.items()}
    over = {old_names.get(k, k): v for k, v in over.items()}
    unknown = sorted(set(over) - set(values))
    if unknown:
        raise ConfigError(f"unknown setting(s) in overrides: {unknown}")
    for k, val in over.items():
        values[k] = _coerce(k, val, values[k])
    _validate(values)
    return Config(values, owner, over.keys())


cfg = resolve_config()

# Module constants for `from configs import *` / `from config import *`.
globals().update(cfg.as_dict())
__all__ = sorted(cfg.as_dict().keys()) + ["cfg", "resolve_config", "Config",
                                          "ConfigError", "apply_cli_overrides"]
