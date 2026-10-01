"""One structured metric event, four outputs.

The training loop used to write one TXT file per metric and start a watcher
thread per file that re-read it and wrote TensorBoard; the axis was whatever
line number the watcher had reached. Here the main process emits one event,
and the same values on the same axes go to:

    events.jsonl   the source of record, one JSON object per event
    *.txt          the legacy per-metric files, for older analysis scripts
    TensorBoard    one scalar per metric
    W&B            one run.log() per event, with global_episode and
                   global_update as separate chart axes

Workers never create W&B runs; they send their numbers to the main process.
W&B is written with run.log() directly, never through sync_tensorboard, so
nothing is recorded twice. A W&B failure is reported once and logging carries
on locally.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from typing import Any, Dict, Iterable, Mapping, Optional

# Episode metrics that also go to the legacy TXT files, and the file each goes
# to. Everything else is JSONL/TensorBoard/W&B only.
LEGACY_TXT = {
    "episode/total_reward": "total_reward.txt",
    "episode/evac_time_80": "evacuation_80.txt",
    "episode/evac_time_100": "evacuation_100.txt",
    "episode/total_lifetime": "total_lifetime.txt",
}
LEGACY_COMPONENT_PREFIX = "episode/reward/"

RUN_STATE_FILE = "wandb_run_state.json"


def code_version(root: Optional[str] = None) -> Dict[str, str]:
    """Which code produced a run: the git commit when the tree is tracked,
    and always a digest of the project's Python sources, because this
    project folder is not necessarily committed."""
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out: Dict[str, str] = {}
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        tracked = subprocess.run(
            ["git", "-C", root, "ls-files", "--error-unmatch", "config.py"],
            capture_output=True, text=True, timeout=5)
        if rev.returncode == 0 and tracked.returncode == 0:
            out["git_commit"] = rev.stdout.strip()
    except Exception:
        pass
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith((".", "__pycache__")))
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                digest.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as fh:
                    digest.update(fh.read())
    out["source_sha256"] = digest.hexdigest()[:16]
    return out


def config_source_files():
    """(path, name inside the artifact) for every configuration file."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for rel in ("config.py",):
        out.append((os.path.join(root, rel), rel))
    cdir = os.path.join(root, "configs")
    for dirpath, dirnames, filenames in os.walk(cdir):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                out.append((path, os.path.relpath(path, root)))
    return out


def run_name(cfg, start_episode: int = 0,
             now: Optional[float] = None) -> str:
    """The W&B run name: WANDB_RUN_NAME, or one built from the experiment,
    training mode, map sizes and start time. A new run started from a
    checkpoint says so."""
    if cfg.WANDB_RUN_NAME:
        name = str(cfg.WANDB_RUN_NAME).strip()
    else:
        if cfg.TRAIN_MAP_SOURCE == "dataset":
            sizes = "-".join(str(int(s)) for s in cfg.DATASET_SIZES_M)
            mode = f"dataset-{sizes}m"
        else:
            sizes = "-".join(str(int(s)) for s in cfg.UED_MAP_SIZES_M)
            mode = f"ued-{cfg.UED_METHOD}-{sizes}m"
        stamp = time.strftime("%m%d-%H%M", time.localtime(now))
        name = f"{cfg.EXPERIMENT_ID}-{mode}-{stamp}"
    if int(start_episode) > 0:
        name += f"-resume{int(start_episode)}"
    return name


def wandb_config(run_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Run metadata shaped for the W&B config panel.

    Every setting becomes a top-level key (LR, DATASET_SITES, ...), so runs
    can be filtered and compared by it; `setting_file` says which file owns
    each. The rest of the metadata (schema versions, map split, code version)
    sits beside them under its own keys.
    """
    out: Dict[str, Any] = {}
    owners: Dict[str, str] = {}
    for section, values in (run_metadata.get("config") or {}).items():
        for name, value in values.items():
            out[name] = value
            owners[name] = section
    out["setting_file"] = owners
    for key, value in run_metadata.items():
        if key != "config":
            out[f"run/{key}"] = value
    return out


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class MetricsLogger:
    """Fan one event out to JSONL, TXT, TensorBoard and W&B.

    `wandb_module` and `writer` exist so tests can substitute recorders; in
    a run they are the real `wandb` and a `SummaryWriter`.
    """

    def __init__(self, cfg, log_dir: str, *, run_metadata: Mapping[str, Any],
                 start_episode: int = 0, start_update: int = 0,
                 writer=None, wandb_module=None, enable_tensorboard: bool = True):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.events_path = os.path.join(log_dir, "events.jsonl")
        self._events = open(self.events_path, "a", encoding="utf-8")
        self._txt_enabled = bool(cfg.LOG_TXT_COMPAT)
        self._writer = writer
        if self._writer is None and enable_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(
                log_dir=os.path.join(log_dir, "tensorboard_logs"))
        self._wandb = None
        self._wandb_module = wandb_module
        self._wandb_failed = False
        self.run_metadata = dict(run_metadata)
        self._start_wandb(int(start_episode), int(start_update))
        self._emit({"kind": "run_start", **self.run_metadata,
                    "wandb_run_id": getattr(self._wandb, "id", None),
                    "global_episode": int(start_episode),
                    "global_update": int(start_update)})

    # ---------------------------------------------------------------- W&B

    def _start_wandb(self, start_episode: int, start_update: int) -> None:
        mode = str(self.cfg.WANDB_MODE)
        if mode == "disabled":
            return
        try:
            wandb = self._wandb_module
            if wandb is None:
                import wandb  # noqa: F811
            state = self._read_run_state()
            group = self.cfg.WANDB_GROUP or self.cfg.EXPERIMENT_ID
            resume_id = None
            # Continue the previous run only when the counters carry on from
            # where it stopped. A restart from an older checkpoint rolls the
            # counters back, and writing those steps into the same run would
            # draw two different histories over one axis; it gets a new run in
            # the same group instead.
            if (state and state.get("experiment_id") == self.cfg.EXPERIMENT_ID
                    and state.get("config_fingerprint")
                    == self.run_metadata.get("config_fingerprint")
                    and start_episode >= int(state.get("global_episode", 0))
                    and start_update >= int(state.get("global_update", 0))):
                resume_id = state.get("run_id")
            kwargs = dict(
                project=self.cfg.WANDB_PROJECT,
                entity=self.cfg.WANDB_ENTITY,
                group=group,
                job_type="train",
                mode=mode,
                dir=self.log_dir,
                config=wandb_config(self.run_metadata),
                tags=[self.cfg.OBSERVATION_SCHEMA_VERSION,
                      self.cfg.REWARD_VERSION,
                      self.cfg.ACTION_SCHEMA_VERSION],
            )
            if resume_id:
                # Carrying on the same run keeps the name it already has.
                kwargs.update(id=resume_id, resume="allow")
            else:
                kwargs.update(name=run_name(self.cfg, start_episode))
            run = wandb.init(**kwargs)
            run.define_metric("global_episode")
            run.define_metric("global_update")
            run.define_metric("episode/*", step_metric="global_episode")
            run.define_metric("eval/*", step_metric="global_episode")
            run.define_metric("train/*", step_metric="global_update")
            self._wandb = run
            self._write_run_state(start_episode, start_update)
            self._upload_config_files(wandb)
        except Exception as exc:
            self._wandb_error("init", exc)

    def _upload_config_files(self, wandb) -> None:
        """The configuration files as written, comments included, and the
        merged values the run actually used, as one `config` artifact.

        The run config holds the values; this keeps the files themselves, so
        a run can be reproduced from W&B alone. The artifact name carries the
        config fingerprint, so identical configurations share one version.
        """
        try:
            fp = str(self.run_metadata.get("config_fingerprint", "unknown"))
            art = wandb.Artifact(name=f"config-{fp}", type="config",
                                 metadata={"config_fingerprint": fp,
                                           **self.run_metadata.get(
                                               "schema_versions", {})})
            for path, name in config_source_files():
                art.add_file(path, name=name)
            resolved = os.path.join(self.log_dir, "config_resolved.json")
            with open(resolved, "w", encoding="utf-8") as fh:
                json.dump({k: self.run_metadata.get(k) for k in (
                    "config", "config_fingerprint", "schema_versions",
                    "observation_schema", "map_split", "code_version",
                    "experiment_id")}, fh, indent=2, ensure_ascii=False,
                    default=str)
            art.add_file(resolved, name="config_resolved.json")
            self._wandb.log_artifact(art)
        except Exception as exc:
            self._wandb_error("config upload", exc)

    def _wandb_error(self, where: str, exc: BaseException) -> None:
        if not self._wandb_failed:
            msg = (f"[W&B] {where} failed: {type(exc).__name__}: {exc}. "
                   f"Training continues; metrics are still written to "
                   f"{self.events_path} and TensorBoard.")
            print("\n" + "!" * 78 + "\n" + msg + "\n" + "!" * 78 + "\n",
                  flush=True)
        self._wandb_failed = True
        self._wandb = None
        try:
            self._emit({"kind": "wandb_error", "where": where,
                        "error": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass

    def _run_state_path(self) -> str:
        return os.path.join(self.log_dir, RUN_STATE_FILE)

    def _read_run_state(self) -> Optional[dict]:
        try:
            with open(self._run_state_path(), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def _write_run_state(self, episode: int, update: int) -> None:
        if self._wandb is None:
            return
        state = {
            "run_id": getattr(self._wandb, "id", None),
            "experiment_id": self.cfg.EXPERIMENT_ID,
            "config_fingerprint": self.run_metadata.get("config_fingerprint"),
            "global_episode": int(episode),
            "global_update": int(update),
        }
        tmp = self._run_state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, self._run_state_path())

    # ------------------------------------------------------------- emit

    def _emit(self, event: Dict[str, Any]) -> None:
        event = {"time": time.time(), **event}
        self._events.write(json.dumps(event, ensure_ascii=False,
                                      default=str) + "\n")
        self._events.flush()

    def _log(self, kind: str, axis_value: int, global_episode: int,
             global_update: int, metrics: Mapping[str, Any]) -> Dict[str, float]:
        clean = {}
        for key, value in metrics.items():
            v = _finite(value)
            if v is not None:
                clean[key] = v
        self._emit({"kind": kind, "global_episode": int(global_episode),
                    "global_update": int(global_update), "metrics": clean})
        if self._writer is not None:
            for key, value in clean.items():
                self._writer.add_scalar(key, value, int(axis_value))
        if self._wandb is not None:
            try:
                self._wandb.log({**clean,
                                 "global_episode": int(global_episode),
                                 "global_update": int(global_update)})
            except Exception as exc:
                self._wandb_error("log", exc)
        return clean

    def log_episode(self, global_episode: int, global_update: int,
                    metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Per-episode metrics; every key should start with "episode/"."""
        clean = self._log("episode", global_episode, global_episode,
                          global_update, metrics)
        if self._txt_enabled:
            self._write_txt(metrics)
        if self._wandb is not None and global_episode % 50 == 0:
            self._write_run_state(global_episode, global_update)
        return clean

    def log_train(self, global_update: int, global_episode: int,
                  metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Learner metrics; keys start with "train/"."""
        return self._log("train", global_update, global_episode,
                         global_update, metrics)

    def log_eval(self, global_episode: int, global_update: int,
                 metrics: Mapping[str, Any]) -> Dict[str, float]:
        """Evaluation summaries; keys start with "eval/"."""
        return self._log("eval", global_episode, global_episode,
                         global_update, metrics)

    def log_video(self, global_episode: int, global_update: int, path: str,
                  fps: int, caption: str = "") -> None:
        """A recorded episode: path in JSONL, the clip itself to W&B."""
        self._emit({"kind": "video", "global_episode": int(global_episode),
                    "global_update": int(global_update), "path": path,
                    "caption": caption})
        if self._wandb is None:
            return
        try:
            wandb = self._wandb_module
            if wandb is None:
                import wandb  # noqa: F811
            self._wandb.log({"episode/video": wandb.Video(
                path, fps=int(fps), format="mp4", caption=caption),
                "global_episode": int(global_episode),
                "global_update": int(global_update)})
        except Exception as exc:
            self._wandb_error("video", exc)

    def note(self, kind: str, **payload) -> None:
        """A non-scalar record, JSONL only (checkpoint saved, map rejected)."""
        self._emit({"kind": kind, **payload})

    def _write_txt(self, metrics: Mapping[str, Any]) -> None:
        # The legacy files write the raw value, NaN included, one line per
        # episode, so a line number stays an episode index.
        for key, fname in LEGACY_TXT.items():
            if key in metrics:
                with open(os.path.join(self.log_dir, fname), "a") as fh:
                    fh.write(f"{metrics[key]}\n")
        for key, value in metrics.items():
            if key.startswith(LEGACY_COMPONENT_PREFIX):
                name = key[len(LEGACY_COMPONENT_PREFIX):]
                with open(os.path.join(self.log_dir, f"reward_{name}.txt"),
                          "a") as fh:
                    fh.write(f"{value}\n")

    # ---------------------------------------------------------- artifacts

    def upload_selected_model(self, path: str, name: str,
                              aliases: Iterable[str] = (),
                              metadata: Optional[Mapping[str, Any]] = None):
        """Record a model chosen on validation as a W&B artifact.

        Only under WANDB_UPLOAD_CHECKPOINTS = "selected". The replay buffer
        and routine periodic checkpoints are never uploaded."""
        self.note("model_selected", path=path, name=name,
                  aliases=list(aliases), metadata=dict(metadata or {}))
        if self._wandb is None or self.cfg.WANDB_UPLOAD_CHECKPOINTS != "selected":
            return
        try:
            wandb = self._wandb_module
            if wandb is None:
                import wandb  # noqa: F811
            art = wandb.Artifact(name=name, type="model",
                                 metadata=dict(metadata or {}))
            art.add_file(path)
            self._wandb.log_artifact(art, aliases=list(aliases))
        except Exception as exc:
            self._wandb_error("artifact", exc)

    def close(self) -> None:
        try:
            if self._writer is not None:
                self._writer.flush()
                self._writer.close()
        finally:
            if self._wandb is not None:
                try:
                    self._wandb.finish()
                except Exception as exc:
                    self._wandb_error("finish", exc)
            self._events.close()


class TrainStatsAccumulator:
    """Averages learner scalars over LOG_TRAIN_EVERY_UPDATES updates, so W&B
    and TensorBoard get one batched point per interval rather than one per
    gradient step."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            v = _finite(value)
            if v is None:
                continue
            self._sums[key] = self._sums.get(key, 0.0) + v
            self._counts[key] = self._counts.get(key, 0) + 1

    def pop_means(self) -> Dict[str, float]:
        out = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums.clear()
        self._counts.clear()
        return out
