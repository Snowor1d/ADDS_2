"""Loading policies for evaluation, viewing and the in-model robot.

A checkpoint carries the settings that define its policy (network and
observation contract, POLICY_KEYS); the policy is rebuilt with those, not
with whatever configs/ says today, so evaluating an old checkpoint after
changing OBS_* or NET_* still feeds it the inputs it was trained on.

SAC_FE_RV3 checkpoints (keys policy, q1, q2, ...) are recognised and loaded
into NET_ENCODER = "legacy_rv3" with RV3's observation settings.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

POLICY_KEYS = (
    "NET_ENCODER", "NET_SIZE", "NET_NORM", "NET_SPATIAL", "NET_COORD_CHANNELS",
    "ENCODER_SHARING", "FiLM_USE", "EGO_USE", "MAP_SCALE_TO_ACTOR", "MAP_SCALE_TO_CRITIC",
    "LOG_STD_MIN", "LOG_STD_MAX", "SAC_STABLE_LOGPROB", "CRITIC_ENSEMBLE",
    "CRITIC_TARGET_SUBSET", "CRITIC_LAYERNORM", "CRITIC_DROPOUT",
    "OBS_EGO_MODE", "OBS_EGO_METERS_PER_PX", "OBS_FOLLOWER_PIXEL",
    "OBS_STATIC_DISTANCE_CHANNEL", "MAP_SCALE_UNIT_M", "MAP_SCALE_N", "FRAME_STACK",
    "EGO_MAP_SIZE", "DOWNSAMPLE_MAP_SIZE", "MAP_H", "MAP_W", "ACTION_SCALE",
    "ROBOT_ACTION_MODE", "ROBOT_WAYPOINT_RANGE_M", "ROBOT_WAYPOINT_SPEED_MAX",
    "ROBOT_WAYPOINT_ARRIVAL_M", "ROBOT_DECISION_MAX_S_WAYPOINT", "ROBOT_DECISION_ON_EVENTS_WAYPOINT",
)

RV3_POLICY = {
    "NET_ENCODER": "legacy_rv3", "ENCODER_SHARING": "separate", "FiLM_USE": True,
    "EGO_USE": True, "MAP_SCALE_TO_ACTOR": False, "MAP_SCALE_TO_CRITIC": False,
    "LOG_STD_MIN": -20, "LOG_STD_MAX": 0.5, "SAC_STABLE_LOGPROB": False,
    "CRITIC_ENSEMBLE": 2, "CRITIC_TARGET_SUBSET": 2, "CRITIC_LAYERNORM": False,
    "CRITIC_DROPOUT": 0.0, "OBS_EGO_MODE": "normalized", "OBS_FOLLOWER_PIXEL": None,
    "OBS_STATIC_DISTANCE_CHANNEL": False, "FRAME_STACK": 4, "EGO_MAP_SIZE": 25,
    "DOWNSAMPLE_MAP_SIZE": 50, "MAP_H": 100, "MAP_W": 100, "ACTION_SCALE": 4,
}


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    from configs import cfg
    cand = os.path.join(os.path.expanduser("~"), cfg.LOG_DIR, path)
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu", weights_only=False)


def is_rv3(ckpt: Dict[str, Any]) -> bool:
    return "policy" in ckpt and "q1" in ckpt and "format" not in ckpt


def policy_settings(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    if is_rv3(ckpt):
        return dict(RV3_POLICY)
    saved = ckpt.get("config", {})
    out = {k: saved[k] for k in POLICY_KEYS if k in saved}
    # checkpoints from before ROBOT_ACTION_MODE existed are velocity policies,
    # whatever configs/ says today
    out.setdefault("ROBOT_ACTION_MODE", "velocity")
    return out


def policy_config(ckpt: Dict[str, Any]):
    from configs import resolve_config
    return resolve_config(overrides=policy_settings(ckpt))


def load_policy(path: str) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """(actor on CPU in eval mode, the policy's configuration, metadata)."""
    from learn.networks import build_policy
    ckpt = load_checkpoint(path)
    pcfg = policy_config(ckpt)
    actor = build_policy(pcfg)
    if is_rv3(ckpt):
        actor.net.load_state_dict(ckpt["policy"])
        meta = {"format": "rv3", "path": os.path.abspath(resolve_path(path))}
    else:
        actor.load_state_dict(ckpt["actor"])
        meta = {"format": ckpt.get("format", "rv4"), "path": os.path.abspath(resolve_path(path)),
                "global_episode": ckpt.get("global_episode"),
                "config_fingerprint": ckpt.get("config_fingerprint"),
                "train_maps": ckpt.get("train_maps")}
    actor.eval()
    return actor, pcfg, meta


class _InModelPolicy:
    """select_action(ego, glob, robot_state, deterministic) for model.use_model."""

    def __init__(self, actor, pcfg):
        if pcfg.OBS_STATIC_DISTANCE_CHANNEL or pcfg.MAP_SCALE_TO_ACTOR or pcfg.OBS_EGO_MODE != "normalized" \
                or pcfg.OBS_FOLLOWER_PIXEL is not None:
            raise ValueError("this policy needs observation features the in-model path does not "
                             "build; evaluate it with python3 -m cli.evaluate")
        self.actor = actor
        self.policy = actor

    @torch.no_grad()
    def select_action(self, ego_state_np, global_state_np, robot_state_np=None,
                      deterministic=False, log_action=False):
        e = torch.as_tensor(np.asarray(ego_state_np, np.float32))[None]
        g = torch.as_tensor(np.asarray(global_state_np, np.float32))[None]
        r = torch.as_tensor(np.asarray(robot_state_np, np.float32)[:3])[None]
        if deterministic:
            a = self.actor.deterministic(e, g, r)
        else:
            a, _ = self.actor.sample(e, g, r)
        return a[0].numpy(), False


def load_policy_for_model(path: str) -> _InModelPolicy:
    actor, pcfg, _ = load_policy(path)
    return _InModelPolicy(actor, pcfg)
