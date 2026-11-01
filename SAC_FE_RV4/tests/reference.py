"""The configuration the tests are written against.

configs/ holds the settings of whichever run is being prepared, and they are
edited between runs (an observation channel, waypoint actions, auto alpha,
LayerNorm...). Tests that check a behaviour start from this fixed reference
instead, so their outcome does not depend on the run currently configured.
Only the settings that runs commonly change are pinned; everything else comes
from configs/. External RV4_OVERRIDES / RV4_PRESET are ignored.
"""

from configs import resolve_config

REFERENCE = {
    # actions and observation
    "ROBOT_ACTION_MODE": "velocity",
    "OBS_STATIC_DISTANCE_CHANNEL": False,
    "OBS_FOLLOWER_PIXEL": None,
    "OBS_EGO_MODE": "normalized",
    "MAP_SCALE_TO_ACTOR": False,
    "MAP_SCALE_TO_CRITIC": False,
    # networks and learning
    "NET_ENCODER": "cnn",
    "NET_SIZE": "m",
    "ENCODER_SHARING": "separate",
    "CRITIC_LAYERNORM": False,
    "CRITIC_DROPOUT": 0.0,
    "CRITIC_ENSEMBLE": 2,
    "CRITIC_TARGET_SUBSET": 2,
    "ALPHA_AUTO": False,
    "UPDATES_PER_TRANSITION": 1,
    "N_STEP": 1,
    "GAMMA_START": 0.99,
    "GAMMA_END": 0.99,
    "AUG_RANDOM_SHIFT": False,
    "AUG_REPLAY_D4": False,
    # task and reward
    "TRAIN_MAP_SOURCE": "dataset",
    "DANGER_DISTANCE": "euclidean",
    "REWARD_B_1": 0.003,
    "REWARD_B_2": 0,
    "REWARD_D": 2,
}


def reference_config(overrides=None):
    """resolve_config on top of REFERENCE (then `overrides`), ignoring the
    environment."""
    return resolve_config(overrides={**REFERENCE, **(overrides or {})}, use_env=False)
