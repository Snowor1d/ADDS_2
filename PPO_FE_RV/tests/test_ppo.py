import pathlib
import sys

import numpy as np
import pytest
import torch


PROJECT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import ADDS_AS_reinforcement as ppo


def test_gae_true_terminal_has_no_bootstrap():
    advantages, returns = ppo.compute_gae(
        rewards=np.array([1.0], dtype=np.float32),
        values=np.array([0.25], dtype=np.float32),
        next_values=np.array([100.0], dtype=np.float32),
        terminated=np.array([1.0], dtype=np.float32),
        truncated=np.array([0.0], dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )
    np.testing.assert_allclose(advantages, [0.75])
    np.testing.assert_allclose(returns, [1.0])


def test_gae_time_limit_bootstraps_but_stops_recursion():
    advantages, returns = ppo.compute_gae(
        rewards=np.array([1.0, 7.0], dtype=np.float32),
        values=np.array([0.25, 3.0], dtype=np.float32),
        next_values=np.array([2.0, 11.0], dtype=np.float32),
        terminated=np.array([0.0, 0.0], dtype=np.float32),
        truncated=np.array([1.0, 0.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )
    # t=0 is truncated, so its delta bootstraps from 2.0 but cannot include t=1.
    np.testing.assert_allclose(advantages[0], 1.0 + 0.9 * 2.0 - 0.25)
    np.testing.assert_allclose(returns[0], 1.0 + 0.9 * 2.0)


def test_policy_action_bounds_and_log_prob_recomputation():
    torch.manual_seed(7)
    policy = ppo.PolicyNetwork(use_robot=True).eval()
    ego = torch.rand(3, 4, ppo.EGO_MAP_SIZE, ppo.EGO_MAP_SIZE)
    glob = torch.rand(3, 4, ppo.DOWNSAMPLE_MAP_SIZE, ppo.DOWNSAMPLE_MAP_SIZE)
    robot = torch.rand(3, ppo.ROBOT_STATE_DIM)
    with torch.no_grad():
        actions, old_log_probs, raw_actions = policy.sample_action(ego, glob, robot)
        new_log_probs, entropy = policy.evaluate_raw_actions(
            ego, glob, robot, raw_actions
        )
    assert torch.all(actions >= -2.0)
    assert torch.all(actions <= 2.0)
    torch.testing.assert_close(old_log_probs, new_log_probs)
    assert torch.isfinite(entropy).all()


def test_default_value_network_keeps_robot_conditioned_film():
    value = ppo.ValueNetwork(use_robot=True)
    assert value.film is not None
    assert value.film[0].in_features == 32
    assert value.fc1.out_features == 512
    assert value.fc2.out_features == 256


@pytest.mark.parametrize(
    ("ego_use", "film_use"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_ego_and_film_flag_combinations(monkeypatch, ego_use, film_use):
    monkeypatch.setattr(ppo, "EGO_USE", ego_use)
    monkeypatch.setattr(ppo, "FiLM_USE", film_use)
    policy = ppo.PolicyNetwork(use_robot=True).eval()
    value = ppo.ValueNetwork(use_robot=True).eval()
    assert (policy.ego_enc is not None) is ego_use
    assert (value.ego_enc is not None) is ego_use
    assert (value.film is not None) is film_use
    ego = (
        torch.rand(2, 4, ppo.EGO_MAP_SIZE, ppo.EGO_MAP_SIZE)
        if ego_use else None
    )
    glob = torch.rand(2, 4, ppo.DOWNSAMPLE_MAP_SIZE, ppo.DOWNSAMPLE_MAP_SIZE)
    robot = torch.rand(2, ppo.ROBOT_STATE_DIM)
    with torch.no_grad():
        action = policy.deterministic_action(ego, glob, robot)
        prediction = value(ego, glob, robot)
    assert action.shape == (2, 2)
    assert prediction.shape == (2,)


def test_mixed_policy_versions_are_rejected():
    agent = ppo.PPOAgent(device="cpu", ppo_epochs=1, mini_batch_size=2)
    empty_shape = (2, 4, ppo.EGO_MAP_SIZE, ppo.EGO_MAP_SIZE)

    def batch(version):
        return ppo.RolloutBatch(
            worker_id=version,
            policy_version=version,
            ego_states=np.zeros(empty_shape, dtype=np.uint8),
            global_states=np.zeros(
                (2, 4, ppo.DOWNSAMPLE_MAP_SIZE, ppo.DOWNSAMPLE_MAP_SIZE),
                dtype=np.uint8,
            ),
            robot_states=np.zeros((2, ppo.ROBOT_STATE_DIM), dtype=np.float32),
            raw_actions=np.zeros((2, 2), dtype=np.float32),
            old_log_probs=np.zeros(2, dtype=np.float32),
            values=np.zeros(2, dtype=np.float32),
            next_values=np.zeros(2, dtype=np.float32),
            rewards=np.zeros(2, dtype=np.float32),
            terminated=np.zeros(2, dtype=np.float32),
            truncated=np.zeros(2, dtype=np.float32),
            episode_stats=[],
            simulator_steps=8,
            sampled_actions=2,
        )

    with pytest.raises(ValueError, match="Mixed on-policy versions"):
        agent.update([batch(0), batch(1)])


def test_one_ppo_update_uses_eval_mode_batchnorm_and_unclipped_value_mse():
    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    count = 4
    agent = ppo.PPOAgent(
        device="cpu", ppo_epochs=1, mini_batch_size=count, target_kl=None
    )
    ego_u8 = rng.integers(
        0, 256, (count, 4, ppo.EGO_MAP_SIZE, ppo.EGO_MAP_SIZE), dtype=np.uint8
    )
    glob_u8 = rng.integers(
        0, 256,
        (count, 4, ppo.DOWNSAMPLE_MAP_SIZE, ppo.DOWNSAMPLE_MAP_SIZE),
        dtype=np.uint8,
    )
    robot_np = rng.normal(size=(count, ppo.ROBOT_STATE_DIM)).astype(np.float32)
    ego, glob, robot = agent._states_to_tensors(
        ego_u8, glob_u8, robot_np, agent.device
    )
    agent.actor.eval()
    agent.value.eval()
    with torch.no_grad():
        _, old_log_probs, raw_actions = agent.actor.sample_action(ego, glob, robot)
        values = agent.value(ego, glob, robot)
    _, expected_returns = ppo.compute_gae(
        rewards=np.ones(count, dtype=np.float32),
        values=values.numpy(),
        next_values=values.numpy(),
        terminated=np.array([0, 0, 0, 1], dtype=np.float32),
        truncated=np.zeros(count, dtype=np.float32),
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
    )
    expected_value_loss = np.mean((values.numpy() - expected_returns) ** 2)
    ego_bn_mean = agent.value.ego_enc.bn1.running_mean.detach().clone()
    global_bn_mean = agent.value.glob_enc.bn1.running_mean.detach().clone()
    rollout = ppo.RolloutBatch(
        worker_id=0,
        policy_version=5,
        ego_states=ego_u8,
        global_states=glob_u8,
        robot_states=robot_np,
        raw_actions=raw_actions.numpy(),
        old_log_probs=old_log_probs.numpy(),
        values=values.numpy(),
        next_values=values.numpy(),
        rewards=np.ones(count, dtype=np.float32),
        terminated=np.array([0, 0, 0, 1], dtype=np.float32),
        truncated=np.zeros(count, dtype=np.float32),
        episode_stats=[],
        simulator_steps=count * ppo.ACTION_SCALE,
        sampled_actions=count,
    )
    metrics = agent.update([rollout])
    assert metrics["samples"] == count
    assert metrics["epochs_completed"] == 1
    assert metrics["max_preupdate_logratio"] < 1e-5
    assert np.isfinite(metrics["policy_loss"])
    assert metrics["value_loss"] == pytest.approx(expected_value_loss, rel=1e-5)
    assert agent.value.training is False
    torch.testing.assert_close(agent.value.ego_enc.bn1.running_mean, ego_bn_mean)
    torch.testing.assert_close(agent.value.glob_enc.bn1.running_mean, global_bn_mean)
