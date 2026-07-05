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
    batch_size: int = 8
    sequence_length: int = 32
    replay_context: int = 4

    deter_size: int = 128
    stoch_size: int = 16
    discrete_size: int = 16
    hidden_size: int = 128
    embed_size: int = 256
    action_embed_size: int = 64
    rssm_blocks: int = 8
    twohot_bins: int = 127
    twohot_min: float = -20.0
    twohot_max: float = 20.0

    horizon: int = 8
    discount: float = 1.0 - 1.0 / 333.0
    lambda_: float = 0.95
    free_nats: float = 1.0
    unimix_ratio: float = 0.01
    prediction_loss_scale: float = 1.0
    dynamics_loss_scale: float = 1.0
    representation_loss_scale: float = 0.1
    entropy_scale: float = 3e-4
    actor_std_min: float = 0.05
    actor_std_max: float = 1.0
    actor_entropy_clip: float = 20.0

    model_lr: float = 4e-5
    actor_lr: float = 4e-5
    value_lr: float = 4e-5
    lr_warmup_steps: int = 1000
    agc_clip: float = 0.3
    agc_eps: float = 1e-3
    grad_clip: float = 100.0
    actor_grad_clip: float = 10.0
    value_grad_clip: float = 10.0
    target_value_tau: float = 0.02
    return_norm_decay: float = 0.99
    return_norm_low: float = 0.05
    return_norm_high: float = 0.95
    return_norm_min_scale: float = 1.0
    value_loss_scale: float = 1.0
    replay_value_loss_scale: float = 0.3
    value_slowreg_scale: float = 1.0

    reward_loss_scale: float = 1.0
    continue_loss_scale: float = 1.0
    recon_loss_scale: float = 1.0

    spd_min: float = -2.0
    spd_max: float = 2.0

    train_ratio: float = 1.0
    min_replay_sequences: int = 8

    use_amp: bool = True
    decoder_chunk_size: int = 128
