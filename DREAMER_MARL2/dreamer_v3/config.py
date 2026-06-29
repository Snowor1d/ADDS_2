from dataclasses import dataclass
from typing import Tuple


@dataclass
class DreamerConfig:
    max_robots: int
    ego_shape: Tuple[int, int, int]
    global_shape: Tuple[int, int, int]
    robot_dim: int
    action_dim: int
    device: str = "cpu"

    replay_capacity: int = 200_000
    batch_size: int = 16
    sequence_length: int = 32

    deter_size: int = 256
    stoch_size: int = 32
    discrete_size: int = 32
    hidden_size: int = 256
    embed_size: int = 512
    action_embed_size: int = 128
    twohot_bins: int = 255
    twohot_min: float = -20.0
    twohot_max: float = 20.0
    robot_state_clip: float = 5.0
    robot_map_scale_norm: float = 4.0

    horizon: int = 20
    discount: float = 0.997
    lambda_: float = 0.95
    free_nats: float = 1.0
    unimix_ratio: float = 0.01
    kl_scale: float = 0.5
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    entropy_scale: float = 1e-4
    actor_std_min: float = 0.05
    actor_std_max: float = 1.0
    actor_entropy_clip: float = 20.0

    model_lr: float = 1e-4
    actor_lr: float = 8e-5
    value_lr: float = 3e-5
    grad_clip: float = 100.0
    actor_grad_clip: float = 10.0
    value_grad_clip: float = 10.0
    target_value_tau: float = 0.01
    return_norm_decay: float = 0.99
    return_norm_low: float = 0.05
    return_norm_high: float = 0.95
    return_norm_min_scale: float = 1.0

    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    recon_loss_scale: float = 1.0

    spd_min: float = -2.0
    spd_max: float = 2.0

    train_ratio: float = 1.0
    min_replay_sequences: int = 8
