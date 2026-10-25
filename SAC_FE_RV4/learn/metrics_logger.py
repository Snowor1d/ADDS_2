"""One structured event, four outputs (adapted from SAC_UED6_easy).

RV3 wrote one TXT file per metric and started a watcher thread per file that
re-read it into TensorBoard, with the line number as the axis. Here the main
process emits one event and the same values on the same axes go to

    events.jsonl   the source of record, one JSON object per event
    *.txt          RV3's per-metric files (LOG_TXT_COMPAT), for old scripts
    TensorBoard    one scalar per metric
    W&B            one run.log() per event; global_episode and global_update
                   are separate chart axes

Workers never create W&B runs. A W&B failure is reported once and logging
continues locally. A restart from the latest checkpoint continues the same
W&B run when the configuration fingerprint matches and the counters carry on;
otherwise it is a new run in the same group.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from typing import Any, Dict, Iterable, Mapping, Optional

LEGACY_TXT = {
    "episode/total_reward": "total_reward.txt",
    "episode/evac_time_80": "evacuation_80.txt",
    "episode/evac_time_100": "evacuation_100.txt",
    "episode/total_lifetime": "total_lifetime.txt",
}
LEGACY_COMPONENT_PREFIX = "episode/reward/"
RUN_STATE_FILE = "wandb_run_state.json"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_version(root: str = ROOT) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        rev = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        st = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "."],
                            capture_output=True, text=True, timeout=5)
        if rev.returncode == 0:
            out["git_commit"] = rev.stdout.strip()
            out["git_dirty"] = str(bool(st.stdout.strip()))
    except Exception:
        pass
    digest = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith((".", "__pycache__", "data")))
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                digest.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as fh:
                    digest.update(fh.read())
    out["source_sha256"] = digest.hexdigest()[:16]
    return out


def config_source_files():
    out = [(os.path.join(ROOT, "config.py"), "config.py")]
    cdir = os.path.join(ROOT, "configs")
    for dirpath, dirnames, filenames in os.walk(cdir):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                out.append((path, os.path.relpath(path, ROOT)))
    return out


def run_name(cfg, start_episode: int = 0, now: Optional[float] = None) -> str:
    if cfg.WANDB_RUN_NAME:
        name = str(cfg.WANDB_RUN_NAME).strip()
    else:
        if cfg.TRAIN_MAP_SOURCE == "dataset":
            mode = f"dataset{len(cfg.DATASET_MAPS)}"
        elif cfg.TRAIN_MAP_SOURCE == "ued":
            mode = f"ued-{cfg.UED_METHOD}-{cfg.UED_SCORE}"
        else:
            mode = f"mixed{cfg.MIXED_DATASET_PROB}-{cfg.UED_METHOD}"
        net = cfg.NET_ENCODER if cfg.NET_ENCODER == "legacy_rv3" else f"{cfg.NET_ENCODER}-{cfg.NET_SIZE}"
        stamp = time.strftime("%m%d-%H%M", time.localtime(now))
        name = f"{cfg.EXPERIMENT_ID}-{mode}-{net}-{stamp}"
    if int(start_episode) > 0:
        name += f"-from{int(start_episode)}"
    return name


def wandb_config(cfg, run_metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """Every setting as a top-level key so runs can be filtered by it."""
    out: Dict[str, Any] = {}
    owners: Dict[str, str] = {}
    for section, values in cfg.by_section().items():
        for name, value in values.items():
            out[name] = value if isinstance(value, (int, float, str, bool, type(None))) else str(value)
            owners[name] = section
    out["setting_file"] = owners
    out["overridden"] = list(cfg.overridden)
    for key, value in run_metadata.items():
        out[f"run/{key}"] = value
    return out


def _finite(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class MetricsLogger:
    def __init__(self, cfg, log_dir: str, *, run_metadata: Mapping[str, Any],
                 start_episode: int = 0, start_update: int = 0, job_type: str = "train",
                 writer=None, wandb_module=None):
        self.cfg = cfg
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.events_path = os.path.join(log_dir, "events.jsonl")
        self._events = open(self.events_path, "a", encoding="utf-8")
        self._txt = bool(cfg.LOG_TXT_COMPAT) and job_type == "train"
        self._writer = writer
        if self._writer is None and cfg.TENSORBOARD:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard_logs"))
            except Exception as e:
                print(f"[log] TensorBoard unavailable: {e}")
        self._wandb = None
        self._wandb_module = wandb_module
        self._wandb_failed = False
        self.job_type = job_type
        self.run_metadata = dict(run_metadata)
        self._start_wandb(int(start_episode), int(start_update))
        self._emit({"kind": "run_start", **self.run_metadata,
                    "wandb_run_id": getattr(self._wandb, "id", None),
                    "global_episode": int(start_episode), "global_update": int(start_update)})

    # ----------------------------------------------------------------- W&B

    def _wb(self):
        if self._wandb_module is not None:
            return self._wandb_module
        import wandb
        return wandb

    def _start_wandb(self, start_episode: int, start_update: int) -> None:
        c = self.cfg
        if c.WANDB_MODE == "disabled":
            return
        try:
            wandb = self._wb()
            state = self._read_run_state() if self.job_type == "train" else None
            group = c.WANDB_GROUP or c.EXPERIMENT_ID
            resume_id = None
            if (state and state.get("experiment_id") == c.EXPERIMENT_ID
                    and state.get("config_fingerprint") == self.run_metadata.get("config_fingerprint")
                    and start_episode >= int(state.get("global_episode", 0))
                    and start_update >= int(state.get("global_update", 0))):
                resume_id = state.get("run_id")
            tags = list(c.WANDB_TAGS) + [c.TRAIN_MAP_SOURCE, c.NET_ENCODER]
            if c.TRAIN_MAP_SOURCE != "dataset":
                tags.append(f"ued-{c.UED_METHOD}")
            kwargs = dict(project=c.WANDB_PROJECT, entity=c.WANDB_ENTITY, group=group,
                          job_type=self.job_type, mode=c.WANDB_MODE, dir=self.log_dir,
                          config=wandb_config(c, self.run_metadata), tags=tags)
            if resume_id:
                kwargs.update(id=resume_id, resume="allow")
            else:
                name = self.run_metadata.get("run_name") or run_name(c, start_episode)
                kwargs.update(name=name)
            run = wandb.init(**kwargs)
            run.define_metric("global_episode")
            run.define_metric("global_update")
            for p in ("episode/*", "eval/*", "key/*", "ued/*", "perf/*"):
                run.define_metric(p, step_metric="global_episode")
            run.define_metric("train/*", step_metric="global_update")
            self._wandb = run
            if self.job_type == "train":
                self._write_run_state(start_episode, start_update)
            self._upload_config_files(wandb)
        except Exception as exc:
            self._wandb_error("init", exc)

    def _upload_config_files(self, wandb) -> None:
        try:
            fp = str(self.run_metadata.get("config_fingerprint", "unknown"))
            art = wandb.Artifact(name=f"config-{fp}", type="config",
                                 metadata={"config_fingerprint": fp})
            for path, name in config_source_files():
                art.add_file(path, name=name)
            resolved = os.path.join(self.log_dir, "config_resolved.json")
            with open(resolved, "w", encoding="utf-8") as fh:
                json.dump({"config": self.cfg.by_section(), "overridden": list(self.cfg.overridden),
                           **{k: v for k, v in self.run_metadata.items()}},
                          fh, indent=2, ensure_ascii=False, default=str)
            art.add_file(resolved, name="config_resolved.json")
            self._wandb.log_artifact(art)
        except Exception as exc:
            self._wandb_error("config upload", exc)

    def _wandb_error(self, where: str, exc: BaseException) -> None:
        if not self._wandb_failed:
            msg = (f"[W&B] {where} failed: {type(exc).__name__}: {exc}. "
                   f"Training continues; metrics are still written to {self.events_path}.")
            print("\n" + "!" * 78 + "\n" + msg + "\n" + "!" * 78 + "\n", flush=True)
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
        state = {"run_id": getattr(self._wandb, "id", None),
                 "experiment_id": self.cfg.EXPERIMENT_ID,
                 "config_fingerprint": self.run_metadata.get("config_fingerprint"),
                 "global_episode": int(episode), "global_update": int(update)}
        tmp = self._run_state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, self._run_state_path())

    # ---------------------------------------------------------------- emit

    def _emit(self, event: Dict[str, Any]) -> None:
        event = {"time": time.time(), **event}
        self._events.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def _log(self, kind: str, axis: int, episode: int, update: int,
             metrics: Mapping[str, Any]) -> Dict[str, float]:
        clean = {k: v for k, v in ((k, _finite(v)) for k, v in metrics.items()) if v is not None}
        self._emit({"kind": kind, "global_episode": int(episode), "global_update": int(update),
                    "metrics": clean})
        if self._writer is not None:
            for k, v in clean.items():
                self._writer.add_scalar(k, v, int(axis))
        if self._wandb is not None:
            try:
                self._wandb.log({**clean, "global_episode": int(episode),
                                 "global_update": int(update)})
            except Exception as exc:
                self._wandb_error("log", exc)
        return clean

    def log_episode(self, episode: int, update: int, metrics: Mapping[str, Any]):
        clean = self._log("episode", episode, episode, update, metrics)
        if self._txt:
            self._write_txt(metrics)
        if self._wandb is not None and episode % 50 == 0:
            self._write_run_state(episode, update)
        return clean

    def log_train(self, update: int, episode: int, metrics: Mapping[str, Any]):
        return self._log("train", update, episode, update, metrics)

    def log_eval(self, episode: int, update: int, metrics: Mapping[str, Any]):
        return self._log("eval", episode, episode, update, metrics)

    def log_media(self, episode: int, update: int, items: Mapping[str, Any]) -> None:
        """Images / videos / tables: {key: ("image", path) | ("video", path, fps) |
        ("table", columns, rows)}."""
        self._emit({"kind": "media", "global_episode": int(episode),
                    "keys": list(items.keys())})
        if self._wandb is None:
            return
        try:
            wandb = self._wb()
            payload = {}
            for k, v in items.items():
                if v[0] == "image":
                    payload[k] = wandb.Image(v[1])
                elif v[0] == "video":
                    payload[k] = wandb.Video(v[1], fps=int(v[2]), format="mp4")
                elif v[0] == "table":
                    payload[k] = wandb.Table(columns=list(v[1]), data=[list(r) for r in v[2]])
            payload.update({"global_episode": int(episode), "global_update": int(update)})
            self._wandb.log(payload)
        except Exception as exc:
            self._wandb_error("media", exc)

    def note(self, kind: str, **payload) -> None:
        self._emit({"kind": kind, **payload})

    def summary(self, values: Mapping[str, Any]) -> None:
        if self._wandb is None:
            return
        try:
            for k, v in values.items():
                self._wandb.summary[k] = v
        except Exception as exc:
            self._wandb_error("summary", exc)

    def _write_txt(self, metrics: Mapping[str, Any]) -> None:
        for key, fname in LEGACY_TXT.items():
            if key in metrics:
                with open(os.path.join(self.log_dir, fname), "a") as fh:
                    fh.write(f"{metrics[key]}\n")
        for key, value in metrics.items():
            if key.startswith(LEGACY_COMPONENT_PREFIX):
                name = key[len(LEGACY_COMPONENT_PREFIX):]
                with open(os.path.join(self.log_dir, f"{name}.txt"), "a") as fh:
                    fh.write(f"{value}\n")

    def write_by_map(self, label: str, reward: float, evac100: float) -> None:
        if not self._txt:
            return
        d = os.path.join(self.log_dir, "by_map")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"reward_{label}.txt"), "a") as fh:
            fh.write(f"{reward}\n")
        with open(os.path.join(d, f"evacuation_100_{label}.txt"), "a") as fh:
            fh.write(f"{evac100}\n")

    # ------------------------------------------------------------ artifacts

    def upload_model(self, path: str, name: str, aliases: Iterable[str] = (),
                     metadata: Optional[Mapping[str, Any]] = None) -> None:
        self.note("model_selected", path=path, name=name, aliases=list(aliases),
                  metadata=dict(metadata or {}))
        if self._wandb is None or self.cfg.WANDB_UPLOAD_CHECKPOINTS != "selected":
            return
        try:
            wandb = self._wb()
            art = wandb.Artifact(name=name, type="model", metadata=dict(metadata or {}))
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
    """Averages learner scalars over LOG_TRAIN_EVERY_UPDATES updates."""

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def add(self, values: Mapping[str, Any]) -> None:
        for k, v in values.items():
            f = _finite(v)
            if f is None:
                continue
            self._sums[k] = self._sums.get(k, 0.0) + f
            self._counts[k] = self._counts.get(k, 0) + 1

    def pop_means(self) -> Dict[str, float]:
        out = {k: self._sums[k] / self._counts[k] for k in self._sums}
        self._sums.clear()
        self._counts.clear()
        return out
