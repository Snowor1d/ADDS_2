"""Small real-environment smoke test for the synchronous PPO worker protocol."""

import multiprocessing as mp
import pathlib
import sys


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import ADDS_AS_reinforcement as ppo


def main():
    ppo.PPO_ROLLOUT_STEPS_PER_ENV = 1
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=2)
    workers, command_queues = ppo.start_workers(context, 1, result_queue)
    try:
        agent = ppo.PPOAgent(device="cpu", ppo_epochs=1, mini_batch_size=2)
        rollouts = ppo.collect_synchronous_rollouts(
            agent, 0, workers, command_queues, result_queue
        )
        batch = rollouts[0]
        assert batch.policy_version == 0
        assert len(batch) == 1
        assert batch.global_states.shape == (
            1, 4, ppo.DOWNSAMPLE_MAP_SIZE, ppo.DOWNSAMPLE_MAP_SIZE
        )
        print(
            "synchronous PPO smoke passed:",
            f"transitions={len(batch)}",
            f"simulator_steps={batch.simulator_steps}",
            f"sampled_actions={batch.sampled_actions}",
        )
    finally:
        ppo.stop_workers(workers, command_queues)


if __name__ == "__main__":
    main()
