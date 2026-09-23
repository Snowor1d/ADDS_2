"""The three configuration files and the one object built from them.

    configs/simulation_run.py   one viewer or manual run: which map to open,
                                robot count, checkpoint, rendering speed
    configs/environment.py      simulator physics, measurement and the shared
                                observation contract
    configs/training/common.py  SAC and critic, exploration, reward, replay,
                                map splits and seeds, logging, and the
                                training-map mode (TRAIN_MAP_SOURCE)
    configs/training/ued.py     curriculum levels and their task parameters
    configs/training/dataset.py OSM crops to train on and their task
                                parameters

Every setting is owned by exactly one file. `resolve_config()` merges the three
once, at an entry point, into an immutable `ResolvedConfig`, validates it, and
that object is what gets passed to every worker process. `config.py` survives
as a re-export so the many `from config import *` callers keep working while
they are moved to the explicit object.

Nothing in this package has import-time side effects.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from types import ModuleType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from configs import environment as _environment
from configs import simulation_run as _simulation_run
from configs.training import common as _training_common
from configs.training import dataset as _training_dataset
from configs.training import ued as _training_ued

SECTIONS: Tuple[Tuple[str, ModuleType], ...] = (
    ("environment", _environment),
    ("training/common", _training_common),
    ("training/ued", _training_ued),
    ("training/dataset", _training_dataset),
    ("simulation_run", _simulation_run),
)

# Settings that change what a policy observes or how its output is applied.
# They are recorded with a checkpoint as its observation schema: a checkpoint
# is only loaded under the same values.
OBSERVATION_SCHEMA_KEYS = (
    "OBSERVATION_SCHEMA_VERSION", "ACTION_SCHEMA_VERSION", "MAX_ROBOTS",
    "EGO_MAP_SIZE", "OBS_EGO_RES_M", "OBS_MID_SIZE", "OBS_MID_RES_M",
    "OBS_GLOBAL_SIZE", "OBS_HISTORY_DECISIONS", "OBS_DENSITY_SATURATION",
    "OBS_PATH_SCALE_M", "OBS_TEAM_DISTANCE_SCALE_M", "ROBOT_VISION",
    "TEAM_SHARE_OBSERVATIONS", "ACTOR_GLOBAL_CROWD_TRUTH", "ROBOT_MODES",
    "ACTION_SCALE", "MAP_SIZE_REFERENCE",
)

_SECRET_NAME = re.compile(r"(API_?KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)",
                          re.IGNORECASE)
# A W&B key is 40 hex characters (86 for some self-hosted servers); provider
# keys commonly start with sk- or similar.
_SECRET_VALUE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{86}|sk-[A-Za-z0-9_\-]{16,}"
                           r"|local-[0-9a-f]{40})$")


class ConfigError(ValueError):
    """The configuration contradicts itself or the data it names."""


def _public_settings(module: ModuleType) -> Dict[str, Any]:
    out = {}
    for name in dir(module):
        if name.startswith("_") or not (name[:1].isupper()):
            continue
        value = getattr(module, name)
        if isinstance(value, ModuleType) or callable(value):
            continue
        out[name] = value
    return out


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _fingerprint(values: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(dict(values)), sort_keys=True,
                         default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class ResolvedConfig:
    """The merged, validated settings of one run. Read-only.

    Attribute access (`cfg.MAX_ROBOTS`) and item access both work. `sections`
    says which file owned each setting, `fingerprint` identifies the exact
    values, and `observation_schema()` is what a checkpoint is stamped with.
    """

    __slots__ = ("_values", "_owner", "_fingerprint", "_base_fingerprint")

    def __init__(self, values: Mapping[str, Any], owner: Mapping[str, str],
                 base_fingerprint: Optional[str] = None):
        object.__setattr__(self, "_values", dict(values))
        object.__setattr__(self, "_owner", dict(owner))
        object.__setattr__(self, "_fingerprint", _fingerprint(self._values))
        # The fingerprint of the files before any override, so a worker can
        # tell an edited file from a deliberate override.
        object.__setattr__(self, "_base_fingerprint",
                           base_fingerprint or self._fingerprint)

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        try:
            return values[name]
        except KeyError:
            raise AttributeError(name) from None

    def __getitem__(self, name):
        return self._values[name]

    def __contains__(self, name):
        return name in self._values

    def get(self, name, default=None):
        return self._values.get(name, default)

    def __setattr__(self, name, value):
        raise AttributeError("ResolvedConfig is immutable")

    def __delattr__(self, name):
        raise AttributeError("ResolvedConfig is immutable")

    def __reduce__(self):
        return (ResolvedConfig, (self._values, self._owner,
                                 self._base_fingerprint))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def base_fingerprint(self) -> str:
        return self._base_fingerprint

    def owner(self, name: str) -> str:
        return self._owner[name]

    def section(self, section: str) -> Dict[str, Any]:
        return {k: v for k, v in self._values.items()
                if self._owner[k] == section}

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Settings grouped by the file that owns them, JSON-ready."""
        return {name: _jsonable(self.section(name)) for name, _ in SECTIONS}

    def observation_schema(self) -> Dict[str, Any]:
        return {k: _jsonable(self._values[k]) for k in OBSERVATION_SCHEMA_KEYS
                if k in self._values}

    def schema_versions(self) -> Dict[str, str]:
        return {
            "observation_schema_version": self.OBSERVATION_SCHEMA_VERSION,
            "reward_version": self.REWARD_VERSION,
            "action_schema_version": self.ACTION_SCHEMA_VERSION,
        }

    def gamma_per_step(self, gamma_per_decision: Optional[float] = None) -> float:
        g = float(self.GAMMA_START if gamma_per_decision is None
                  else gamma_per_decision)
        return g ** (1.0 / max(1, int(self.ACTION_SCALE)))


def collect() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """All settings and their owning file. Raises on a name owned twice."""
    values: Dict[str, Any] = {}
    owner: Dict[str, str] = {}
    for section, module in SECTIONS:
        for name, value in _public_settings(module).items():
            if name in owner:
                raise ConfigError(
                    f"{name} is defined in both configs/{owner[name]}.py and "
                    f"configs/{section}.py; each setting has one owner")
            values[name] = value
            owner[name] = section
    return values, owner


def resolve_config(overrides: Optional[Mapping[str, Any]] = None,
                   validate: bool = True,
                   check_data: bool = True) -> ResolvedConfig:
    """Merge the three files, apply explicit overrides, validate.

    `overrides` is for tests and one-off command-line experiments; an override
    must name an existing setting, so a typo cannot silently add a new one.
    `check_data` also checks settings against the exported OSM corpus.
    """
    values, owner = collect()
    base = _fingerprint(values)
    for name, value in (overrides or {}).items():
        if name not in values:
            raise ConfigError(f"override names unknown setting {name}")
        values[name] = value
    cfg = ResolvedConfig(values, owner, base_fingerprint=base)
    if validate:
        validate_config(cfg, check_data=check_data)
    return cfg


# ---------------------------------------------------------------- validation

def _check(cond: bool, message: str, problems: list) -> None:
    if not cond:
        problems.append(message)


def _reserved_seed(seed: int, ranges: Iterable[Tuple[int, int]]) -> bool:
    return any(lo <= int(seed) < hi for lo, hi in ranges)


def validation_seeds(cfg) -> Dict[Tuple[int, int, int], int]:
    """(size, difficulty, k) -> generation seed for the validation levels."""
    out = {}
    for si, size in enumerate(cfg.VALIDATION_SIZES_M):
        for difficulty in cfg.VALIDATION_DIFFICULTIES:
            for k in range(int(cfg.VALIDATION_SEEDS_PER_CELL)):
                out[(int(size), int(difficulty), k)] = (
                    int(cfg.VALIDATION_SEED_BASE) + si * 1000
                    + int(difficulty) * 10 + k)
    return out


def _crop_bounds(lat: float, lon: float, size_m: float):
    """Approximate lat/lon box of a square crop centred on (lat, lon)."""
    half = 0.5 * float(size_m)
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * max(1e-6, math.cos(math.radians(lat))))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def _boxes_overlap(a, b) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0] or a[3] <= b[2] or b[3] <= a[2])


def _scan_secrets(cfg: ResolvedConfig, problems: list) -> None:
    for name, value in cfg._values.items():
        texts = []
        if isinstance(value, str):
            texts = [value]
        elif isinstance(value, (tuple, list)):
            texts = [v for v in value if isinstance(v, str)]
        if _SECRET_NAME.search(name) and any(t.strip() for t in texts):
            problems.append(f"{name} looks like a credential; credentials are "
                            "read from the environment or ~/.netrc, never "
                            "from configs/")
        for t in texts:
            if _SECRET_VALUE.match(t.strip()):
                problems.append(f"{name} holds a value shaped like an API key")


def _check_task_params(cfg, mode: str, max_robots: int, p: list) -> None:
    """The per-episode task parameters one training mode draws from."""
    def get(name):
        return cfg[f"{mode}_{name}"]

    shapes = get("DANGER_SHAPES")
    _check(isinstance(shapes, (tuple, list)) and len(shapes) > 0
           and all(sh in ("circle", "rect", "street") for sh in shapes),
           f"{mode}_DANGER_SHAPES={shapes!r} must be a non-empty tuple of "
           "'circle', 'rect', 'street' (a one-element tuple needs a comma)", p)
    for name, lo_ok, hi_ok in (("DANGER_AREA_RANGE", 0.0, 0.45),
                               ("DANGER_INSIDE_FRACTION", 0.0, 1.0),
                               ("DANGER_PERCEPTIBILITY", 0.0, 1.0),
                               ("PRIOR_INFORMED_FRACTION", 0.0, 1.0)):
        rng = get(name)
        ok = (isinstance(rng, (tuple, list)) and len(rng) == 2
              and lo_ok <= float(rng[0]) <= float(rng[1]) <= hi_ok)
        _check(ok, f"{mode}_{name}={rng!r} must be (low, high) within "
               f"[{lo_ok}, {hi_ok}]", p)
    lo, hi = get("ROBOT_RANGE")
    _check(1 <= int(lo) <= int(hi) <= max_robots,
           f"{mode}_ROBOT_RANGE={get('ROBOT_RANGE')} outside 1..{max_robots}", p)
    _check(isinstance(get("AUGMENTATION"), bool),
           f"{mode}_AUGMENTATION={get('AUGMENTATION')!r} must be True or False", p)
    try:
        from sim.map_augmentation import validate_transforms
        validate_transforms(get("AUGMENTATION_TRANSFORMS"))
    except Exception as exc:
        p.append(f"{mode}_AUGMENTATION_TRANSFORMS: {exc}")


def validate_config(cfg: ResolvedConfig, check_data: bool = True) -> None:
    """Refuse configurations that contradict themselves. Raises ConfigError
    listing every problem, not just the first."""
    p: list = []

    # Settings that must be sequences. A one-element tuple written without its
    # comma is a scalar or a string, and fails in confusing ways later.
    for name in ("UED_MAP_SIZES_M", "UED_MAP_SIZE_WEIGHTS", "UED_DANGER_SHAPES",
                 "UED_AUGMENTATION_TRANSFORMS", "DATASET_SITES",
                 "DATASET_SIZES_M", "DATASET_DANGER_SHAPES",
                 "DATASET_AUGMENTATION_TRANSFORMS",
                 "VALIDATION_SIZES_M", "VALIDATION_DIFFICULTIES",
                 "ZSG_REAL_SITES", "FINAL_ZERO_SHOT_HAZARD_SEEDS",
                 "FINAL_ZERO_SHOT_CROWD_SEEDS", "FINAL_ZERO_SHOT_HAZARD_SHAPES",
                 "SUPPORTED_CROP_SIZES_M"):
        if name in cfg and not isinstance(cfg[name], (tuple, list)):
            p.append(f"{name}={cfg[name]!r} must be a tuple (a one-element "
                     "tuple needs a comma)")
    if p:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(p))

    # Crop sizes the corpus exports.
    sizes = tuple(int(s) for s in cfg.SUPPORTED_CROP_SIZES_M)
    _check(int(cfg.SIM_REAL_SIZE) in sizes,
           f"SIM_REAL_SIZE={cfg.SIM_REAL_SIZE} is not one of {sizes}", p)
    _check(int(cfg.ZSG_REAL_SIZE) in sizes,
           f"ZSG_REAL_SIZE={cfg.ZSG_REAL_SIZE} is not one of {sizes}", p)
    _check(int(cfg.FINAL_ZERO_SHOT_SIZE_M) in sizes,
           f"FINAL_ZERO_SHOT_SIZE_M={cfg.FINAL_ZERO_SHOT_SIZE_M} is not one "
           f"of {sizes}", p)
    for s in cfg.VALIDATION_SIZES_M:
        _check(int(s) in sizes, f"VALIDATION_SIZES_M has {s}, not in {sizes}", p)

    # Training mode. Each mode's own parameters are checked whichever is
    # active, so switching modes never uncovers a broken file.
    _check(cfg.TRAIN_MAP_SOURCE in ("dataset", "ued"),
           f"TRAIN_MAP_SOURCE={cfg.TRAIN_MAP_SOURCE!r}; expected 'dataset' "
           "or 'ued'", p)
    mr = int(cfg.MAX_ROBOTS)
    _check(mr >= 1, "MAX_ROBOTS must be at least 1", p)
    for mode in ("UED", "DATASET"):
        _check_task_params(cfg, mode, mr, p)

    # UED: sizes and their weights.
    ws = cfg.UED_MAP_SIZE_WEIGHTS
    _check(len(ws) == len(cfg.UED_MAP_SIZES_M)
           and all(float(w) >= 0 for w in ws)
           and (not ws or sum(float(w) for w in ws) > 0),
           "UED_MAP_SIZE_WEIGHTS must be one non-negative weight per "
           "UED_MAP_SIZES_M entry, not all zero", p)
    for s in cfg.UED_MAP_SIZES_M:
        _check(int(s) in cfg.UED_DENSITY_BY_SIZE,
               f"UED_DENSITY_BY_SIZE has no entry for {s} m", p)

    # Dataset: sites and sizes.
    sites = tuple(cfg.DATASET_SITES)
    _check(len(sites) > 0 or cfg.TRAIN_MAP_SOURCE != "dataset",
           "DATASET_SITES is empty", p)
    _check(len(set(sites)) == len(sites), "DATASET_SITES repeats a site", p)
    _check(len(cfg.DATASET_SIZES_M) > 0, "DATASET_SIZES_M is empty", p)
    for s in cfg.DATASET_SIZES_M:
        _check(int(s) in sizes, f"DATASET_SIZES_M has {s}, not in {sizes}", p)
        _check(int(s) in cfg.DATASET_DENSITY_BY_SIZE,
               f"DATASET_DENSITY_BY_SIZE has no entry for {s} m", p)
    _check(str(cfg.FINAL_ZERO_SHOT_SITE) not in sites,
           f"the final zero-shot site {cfg.FINAL_ZERO_SHOT_SITE} is in "
           "DATASET_SITES", p)
    both = sorted(set(sites) & set(cfg.ZSG_REAL_SITES))
    _check(not both, f"{both} are both training (DATASET_SITES) and auxiliary "
           "evaluation sites", p)
    _check("street" not in tuple(cfg.DATASET_DANGER_SHAPES),
           "a 'street' hazard on an OSM crop is a rectangle that ignores the "
           "street network; use 'circle' and 'rect' in DATASET_DANGER_SHAPES", p)

    # Robot counts.
    _check(1 <= int(cfg.SIM_ROBOTS) <= mr,
           f"SIM_ROBOTS={cfg.SIM_ROBOTS} outside 1..{mr}", p)
    for name in ("ZSG_ROBOT_NUM", "FINAL_ZERO_SHOT_ROBOT_COUNTS",
                 "VALIDATION_ROBOT_COUNTS"):
        if not isinstance(cfg[name], (tuple, list)):
            p.append(f"{name}={cfg[name]!r} must be a tuple (a one-element "
                     "tuple needs a comma, e.g. (3,))")
            continue
        for n in cfg[name]:
            _check(1 <= int(n) <= mr, f"{name} has {n}, outside 1..{mr}", p)

    # Observation contract.
    ego_half = 0.5 * int(cfg.EGO_MAP_SIZE) * float(cfg.OBS_EGO_RES_M)
    _check(ego_half >= float(cfg.ROBOT_VISION),
           f"the local crop reaches {ego_half} m but ROBOT_VISION is "
           f"{cfg.ROBOT_VISION} m; measured people would fall outside it", p)
    mid_half = 0.5 * int(cfg.OBS_MID_SIZE) * float(cfg.OBS_MID_RES_M)
    _check(mid_half > ego_half, "the middle branch must reach further than "
           "the local one", p)
    _check(int(cfg.OBS_HISTORY_DECISIONS) >= 1,
           "OBS_HISTORY_DECISIONS must be at least 1", p)
    _check(int(cfg.OBS_GLOBAL_SIZE) >= 8, "OBS_GLOBAL_SIZE too small", p)
    _check(0.0 <= float(cfg.COMM_DROP_PROB) < 1.0,
           "COMM_DROP_PROB must be in [0, 1)", p)
    _check(0 <= int(cfg.COMM_DELAY_DECISIONS) < int(cfg.OBS_HISTORY_DECISIONS),
           "COMM_DELAY_DECISIONS must be shorter than the observation window, "
           "or no teammate message would ever be shown", p)
    for name in ("OBSERVATION_SCHEMA_VERSION", "ACTION_SCHEMA_VERSION",
                 "REWARD_VERSION"):
        _check(isinstance(cfg[name], str) and cfg[name].strip() != "",
               f"{name} must be a non-empty string", p)
    if cfg.ACTOR_GLOBAL_CROWD_TRUTH:
        _check(str(cfg.EXPERIMENT_ID).endswith("-fullinfo"),
               "ACTOR_GLOBAL_CROWD_TRUTH=True is the full-information upper "
               "bound; give it its own EXPERIMENT_ID ending in '-fullinfo'", p)
        _check("fullinfo" in str(cfg.LOG_DIR),
               "ACTOR_GLOBAL_CROWD_TRUTH=True needs its own LOG_DIR "
               "(containing 'fullinfo') so its checkpoints are never "
               "resumed as the realistic policy", p)
    _check(cfg.RESUME_MODE in ("latest_compatible", "fresh"),
           f"RESUME_MODE={cfg.RESUME_MODE!r}", p)

    # Episode videos.
    _check(int(cfg.VIDEO_EVERY_EPISODES) >= 0,
           "VIDEO_EVERY_EPISODES must be 0 (off) or positive", p)
    steps_per_frame = float(cfg.VIDEO_SPEEDUP) / (
        max(1e-9, float(cfg.VIDEO_FPS)) * float(cfg.AGENT_TIME_STEP))
    _check(int(cfg.VIDEO_FPS) >= 1 and steps_per_frame >= 1.0,
           f"VIDEO_SPEEDUP={cfg.VIDEO_SPEEDUP} at VIDEO_FPS={cfg.VIDEO_FPS} "
           "would need more than one frame per simulation step; lower the "
           "fps or raise the speed-up", p)

    # Logging.
    _check(cfg.WANDB_MODE in ("online", "offline", "disabled"),
           f"WANDB_MODE={cfg.WANDB_MODE!r}", p)
    _check(cfg.WANDB_UPLOAD_CHECKPOINTS in ("none", "selected"),
           f"WANDB_UPLOAD_CHECKPOINTS={cfg.WANDB_UPLOAD_CHECKPOINTS!r}", p)
    _check(cfg.WANDB_RUN_NAME is None
           or (isinstance(cfg.WANDB_RUN_NAME, str)
               and cfg.WANDB_RUN_NAME.strip() != ""),
           "WANDB_RUN_NAME must be None or a non-empty string", p)
    _check(int(cfg.LOG_TRAIN_EVERY_UPDATES) >= 1,
           "LOG_TRAIN_EVERY_UPDATES must be positive", p)
    _scan_secrets(cfg, p)

    # Map splits: generation and hazard seeds.
    reserved = tuple(tuple(r) for r in cfg.TRAIN_RESERVED_SEED_RANGES)
    vseeds = validation_seeds(cfg)
    for key, seed in vseeds.items():
        _check(_reserved_seed(seed, reserved),
               f"validation seed {seed} for {key} is not in "
               "TRAIN_RESERVED_SEED_RANGES, so training could draw it", p)
    _check(len(set(vseeds.values())) == len(vseeds),
           "validation seeds collide", p)
    final_seeds = (tuple(cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS)
                   + tuple(cfg.FINAL_ZERO_SHOT_CROWD_SEEDS))
    _check(len(set(final_seeds)) == len(final_seeds),
           "final zero-shot seeds must be distinct", p)
    for seed in final_seeds:
        _check(_reserved_seed(seed, reserved),
               f"final zero-shot seed {seed} is not reserved from training", p)
        _check(seed not in set(vseeds.values()),
               f"final zero-shot seed {seed} is also a validation seed", p)
    _check(len(cfg.FINAL_ZERO_SHOT_HAZARD_SEEDS) >= 1,
           "pre-register at least one final hazard seed", p)
    _check("street" not in tuple(cfg.FINAL_ZERO_SHOT_HAZARD_SHAPES),
           "an OSM crop has no street network to align a 'street' hazard to; "
           "it would be a rectangle reported as a street", p)
    _check(cfg.FINAL_ZERO_SHOT_CONTROL == "off_zero_command",
           "FINAL_ZERO_SHOT_CONTROL must be 'off_zero_command'", p)

    # Map splits: places.
    final_site = str(cfg.FINAL_ZERO_SHOT_SITE)
    _check(final_site not in tuple(cfg.ZSG_REAL_SITES),
           f"the final zero-shot site {final_site} is also an auxiliary OSM "
           "evaluation site", p)
    if check_data:
        _check_places(cfg, p)

    if p:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(p))


def _check_places(cfg: ResolvedConfig, p: list) -> None:
    """Geographic and spatial leakage checks against the exported corpus."""
    try:
        from osm_corpus.export import load_levels
        levels = load_levels()
    except Exception as exc:        # corpus not exported on this machine
        p.append(f"cannot read the OSM corpus to check map splits: {exc}")
        return
    by_key = {(lv.site_key, int(lv.width)): lv for lv in levels}
    final_key = (str(cfg.FINAL_ZERO_SHOT_SITE), int(cfg.FINAL_ZERO_SHOT_SIZE_M))
    if final_key not in by_key:
        p.append(f"final zero-shot crop {final_key} is not in the corpus")
        return
    final = by_key[final_key]
    if final.lat is None or final.lon is None:
        p.append("the final zero-shot crop has no coordinates to check "
                 "overlap against")
        return
    fbox = _crop_bounds(float(final.lat), float(final.lon), final.width)
    # Every other crop the run may use: auxiliary OSM sites (any evaluated
    # size) and the viewer's crop when it is used for anything but viewing.
    used = [(s, int(cfg.ZSG_REAL_SIZE), "ZSG_REAL_SITES")
            for s in cfg.ZSG_REAL_SITES]
    used += [(s, int(z), "DATASET_SITES") for s in cfg.DATASET_SITES
             for z in cfg.DATASET_SIZES_M]
    for site, size, owner in used:
        key = (site, size)
        lv = by_key.get(key)
        if lv is None:
            p.append(f"OSM crop {key} named in {owner} is not exported")
            continue
        if lv.lat is None or lv.lon is None:
            continue
        box = _crop_bounds(float(lv.lat), float(lv.lon), lv.width)
        if _boxes_overlap(fbox, box):
            p.append(f"OSM crop {key} overlaps the final zero-shot crop "
                     f"{final_key} on the ground")


def config_fingerprint_of_modules() -> str:
    """Fingerprint of the settings as the modules currently define them.

    A worker compares this with the ResolvedConfig it was handed, which
    catches a configuration file edited on disk while the run was going."""
    values, _owner = collect()
    return _fingerprint(values)
