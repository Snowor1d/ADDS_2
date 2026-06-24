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

    horizon: int = 15
    discount: float = 0.997
    lambda_: float = 0.95
    free_nats: float = 1.0
    kl_scale: float = 0.5
    dyn_scale: float = 0.5
    rep_scale: float = 0.1
    entropy_scale: float = 1e-4

    model_lr: float = 1e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    grad_clip: float = 100.0

    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    recon_loss_scale: float = 1.0

    r_min: float = 2.0
    r_max: float = 8.0
    spd_min: float = 1.0
    spd_max: float = 2.0

    train_ratio: float = 1.0
    min_replay_sequences: int = 8
