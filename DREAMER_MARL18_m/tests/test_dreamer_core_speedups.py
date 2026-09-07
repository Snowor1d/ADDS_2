import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dreamer_v3.agent import DreamerAgent
from dreamer_v3.config import DreamerConfig
from dreamer_v3.networks import ConvEncoder, RSSMState, WorldModel
from dreamer_v3.replay import DreamerSequenceReplay, DreamerStep


def tiny_config(**overrides):
    values = dict(
        max_robots=1,
        ego_shape=(1, 8, 8),
        global_shape=(1, 8, 8),
        robot_dim=2,
        action_dim=2,
        deter_size=8,
        stoch_size=2,
        discrete_size=3,
        hidden_size=8,
        conv_depth=2,
        rssm_blocks=2,
        decoder_bspace=2,
        twohot_bins=7,
        batch_size=1,
        sequence_length=2,
        replay_context=1,
        horizon=2,
        use_amp=False,
        decoder_chunk_size=100,
    )
    values.update(overrides)
    return DreamerConfig(**values)


def replay_step(index=0, terminal=False):
    return DreamerStep(
        joint_ego=np.full((1, 1, 8, 8), index, dtype=np.uint8),
        global_state=np.full((1, 8, 8), index, dtype=np.uint8),
        joint_robot=np.zeros((1, 2), dtype=np.float32),
        action=np.zeros((1, 2), dtype=np.float32),
        reward=float(index),
        is_first=index == 0,
        is_terminal=terminal,
        joint_mask=np.ones((1,), dtype=np.float32),
        delta_t=1.0,
    )


class DreamerCoreSpeedupsTest(unittest.TestCase):
    def test_imagination_starts_are_valid_unique_and_capped_per_row(self):
        agent = DreamerAgent.__new__(DreamerAgent)
        agent.cfg = SimpleNamespace(imag_starts_per_sequence=3)
        posts = [
            RSSMState(
                deter=torch.randn(3, 4),
                stoch=torch.randn(3, 2, 2),
                logits=torch.randn(3, 2, 2),
            )
            for _ in range(6)
        ]
        loss_mask = torch.tensor(
            [
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )

        _, indices, batch_indices = agent._sample_start_state(
            posts,
            {"loss_mask": loss_mask},
        )

        pairs = list(zip(batch_indices.tolist(), indices.tolist()))
        self.assertEqual(len(pairs), 5)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(sum(batch_id == 0 for batch_id, _ in pairs), 3)
        self.assertEqual(sum(batch_id == 1 for batch_id, _ in pairs), 2)
        self.assertEqual(sum(batch_id == 2 for batch_id, _ in pairs), 0)
        self.assertTrue(all(loss_mask[batch_id, index] > 0 for batch_id, index in pairs))

    def test_fixed_context_imagination_fast_path_returns_k_per_sequence(self):
        agent = DreamerAgent.__new__(DreamerAgent)
        agent.cfg = SimpleNamespace(
            imag_starts_per_sequence=3,
            replay_context=2,
        )
        posts = [
            RSSMState(
                deter=torch.randn(2, 4),
                stoch=torch.randn(2, 2, 2),
                logits=torch.randn(2, 2, 2),
            )
            for _ in range(7)
        ]
        loss_mask = torch.tensor(
            [[0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        )

        _, indices, batch_indices = agent._sample_start_state(
            posts,
            {"loss_mask": loss_mask},
        )

        pairs = list(zip(batch_indices.tolist(), indices.tolist()))
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertTrue(all(index >= 2 for _, index in pairs))
        self.assertEqual(sum(batch_id == 0 for batch_id, _ in pairs), 3)
        self.assertEqual(sum(batch_id == 1 for batch_id, _ in pairs), 3)

    def test_uint8_encoder_matches_normalized_float_encoder(self):
        torch.manual_seed(1)
        encoder = ConvEncoder(tiny_config(), 1, (8, 8))
        image = torch.randint(0, 256, (3, 1, 8, 8), dtype=torch.uint8)
        uint8_output = encoder(image)
        float_output = encoder(image.float() / 255.0)
        torch.testing.assert_close(uint8_output, float_output)

    def test_context_predictions_are_skipped_without_changing_loss(self):
        full_model = WorldModel(tiny_config(skip_context_prediction=False))
        skip_model = WorldModel(tiny_config(skip_context_prediction=True))
        skip_model.load_state_dict(full_model.state_dict())
        batch_size, time_steps = 2, 3
        batch = {
            "joint_ego": torch.randint(
                0, 256, (batch_size, time_steps, 1, 1, 8, 8), dtype=torch.uint8
            ),
            "global_state": torch.randint(
                0, 256, (batch_size, time_steps, 1, 8, 8), dtype=torch.uint8
            ),
            "joint_robot": torch.randn(batch_size, time_steps, 1, 2),
            "action": torch.randn(batch_size, time_steps, 1, 2),
            "reward": torch.randn(batch_size, time_steps),
            "is_first": torch.zeros(batch_size, time_steps),
            "is_terminal": torch.zeros(batch_size, time_steps),
            "continue": torch.ones(batch_size, time_steps),
            "joint_mask": torch.ones(batch_size, time_steps, 1),
            "delta_t": torch.ones(batch_size, time_steps),
            "loss_mask": torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.float32),
        }
        decoder_batch_sizes = []
        full_hook = full_model.ego_decoder.register_forward_pre_hook(
            lambda _module, args: decoder_batch_sizes.append(("full", args[0].shape[0]))
        )
        skip_hook = skip_model.ego_decoder.register_forward_pre_hook(
            lambda _module, args: decoder_batch_sizes.append(("skip", args[0].shape[0]))
        )
        try:
            torch.manual_seed(7)
            full_loss, _, _ = full_model.loss(batch)
            torch.manual_seed(7)
            skip_loss, skip_metrics, _ = skip_model.loss(batch)
        finally:
            full_hook.remove()
            skip_hook.remove()

        torch.testing.assert_close(full_loss, skip_loss)
        self.assertEqual(decoder_batch_sizes, [("full", 6), ("skip", 4)])
        self.assertTrue(all(torch.is_tensor(value) for value in skip_metrics.values()))

    def test_replay_keeps_images_uint8_and_snapshot_is_stable(self):
        replay = DreamerSequenceReplay(100, 2, "cpu", context_length=1)
        first_episode = [replay_step(i, terminal=i == 3) for i in range(4)]
        replay.add_episode(first_episode)
        batch = replay.sample(1)
        self.assertEqual(batch["joint_ego"].dtype, torch.uint8)
        self.assertEqual(batch["global_state"].dtype, torch.uint8)

        snapshot = replay.snapshot()
        replay.add_episode([replay_step(i, terminal=i == 3) for i in range(4)])
        self.assertEqual(snapshot.num_steps, 4)
        self.assertEqual(len(snapshot.episodes), 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "replay.npz")
            replay.save_snapshot(snapshot, path)
            loaded = DreamerSequenceReplay(100, 2, "cpu", context_length=1)
            self.assertTrue(loaded.load(path))
            self.assertEqual(loaded.num_steps, 4)
            self.assertEqual(loaded.num_episodes, 1)

    def test_worker_payload_is_minimal_and_legacy_compatible(self):
        source = DreamerAgent(tiny_config())
        target = DreamerAgent(tiny_config())
        try:
            worker_state = source.get_worker_state()
            self.assertEqual(set(worker_state), {"encoder", "rssm", "actor"})
            target.load_worker_state(worker_state)

            legacy_state = {
                "world_model": source._state_dict_to_numpy(source.world_model.state_dict()),
                "actor": worker_state["actor"],
            }
            target.load_worker_state(legacy_state)
        finally:
            source.close()
            target.close()

    def test_full_update_accepts_uint8_batch_and_returns_host_metrics(self):
        agent = DreamerAgent(tiny_config(imag_starts_per_sequence=2))
        try:
            agent.replay.add_episode(
                [replay_step(i, terminal=i == 3) for i in range(4)]
            )
            metrics = agent.update()
            self.assertIsNotNone(metrics)
            self.assertTrue(all(type(value) is float for value in metrics.values()))
            self.assertEqual(metrics["imag_start_count"], 2.0)
        finally:
            agent.close()

    def test_checkpoint_payload_is_cpu_cloned_and_atomically_writable(self):
        agent = DreamerAgent(tiny_config())
        try:
            payload = agent.checkpoint_payload_cpu()
            key = next(iter(payload["world_model"]))
            saved_value = payload["world_model"][key].clone()
            with torch.no_grad():
                agent.world_model.state_dict()[key].add_(1.0)
            torch.testing.assert_close(payload["world_model"][key], saved_value)
            self.assertEqual(payload["world_model"][key].device.type, "cpu")

            with tempfile.TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "model.pt")
                agent.save_checkpoint_payload(payload, path)
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(set(loaded), set(payload))
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
