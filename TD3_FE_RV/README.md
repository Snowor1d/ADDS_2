# TD3_FE_RV

TD3 baseline derived from `SAC_FE_RV3`.  It keeps the same environment,
observations, rewards, map augmentation, replay buffer, CNN/robot feature
fusion, FiLM-conditioned twin Q-networks, warm-up, and asynchronous data
collection.

The algorithm-specific differences are:

- deterministic actor;
- Gaussian behavior-policy exploration noise;
- actor target network;
- clipped target-policy smoothing noise;
- delayed actor and target-network updates;
- minimum of the two target critics;
- no SAC entropy or temperature term.

The main TD3 settings are in `config.py`.  Start training from this directory:

```bash
python3 ADDS_AS_reinforcement.py
```

Checkpoints and TensorBoard data are written under `~/Log_TD3_FE_RV`.
