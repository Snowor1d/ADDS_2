import numpy as np
import torch
import torch.nn as nn

import ADDS_AS_reinforcement as td3


class TinyReplayBuffer:
    def __init__(self, capacity, device, **_kwargs):
        self.capacity = capacity
        self.device = torch.device(device)
        self.size = 32

    def __len__(self):
        return self.size

    def sample(self, batch_size):
        glob = torch.linspace(-1.0, 1.0, batch_size, device=self.device).unsqueeze(1)
        ego = glob.clone()
        robot = torch.zeros(batch_size, 3, device=self.device)
        action = torch.zeros(batch_size, 2, device=self.device)
        reward = torch.ones(batch_size, device=self.device)
        next_glob = glob + 0.1
        done = torch.zeros(batch_size, device=self.device)
        return ego, glob, robot, action, reward, ego, next_glob, robot, done


class TinyPolicy(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def action(self, ego, glob, robot=None):
        return 2.0 * torch.tanh(self.linear(glob[:, :1]))


class TinyQ(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.last_action = None

    def forward(self, ego, glob, action, robot=None):
        self.last_action = action.detach().clone()
        return self.linear(torch.cat((glob[:, :1], action), dim=1))


def make_tiny_agent(monkeypatch, **kwargs):
    monkeypatch.setattr(td3, "ReplayBuffer", TinyReplayBuffer)
    monkeypatch.setattr(td3, "PolicyNetwork", TinyPolicy)
    monkeypatch.setattr(td3, "QNetwork", TinyQ)
    return td3.TD3Agent(
        device="cpu",
        batch_size=4,
        replay_size=32,
        policy_delay=2,
        target_policy_noise=10.0,
        target_noise_clip=0.5,
        **kwargs,
    )


def parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def changed(before, module):
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, module.parameters())
    )


def test_td3_delays_actor_and_bounds_smoothed_target_action(monkeypatch):
    torch.manual_seed(7)
    agent = make_tiny_agent(monkeypatch)
    initial_policy = parameters(agent.policy)
    initial_policy_target = parameters(agent.policy_target)

    first = agent.update()
    assert first["policy_updated"] == 0.0
    assert not changed(initial_policy, agent.policy)
    assert not changed(initial_policy_target, agent.policy_target)
    assert torch.all(agent.q1_target.last_action <= td3.TD3_ACTION_HIGH)
    assert torch.all(agent.q1_target.last_action >= td3.TD3_ACTION_LOW)

    second = agent.update()
    assert second["policy_updated"] == 1.0
    assert changed(initial_policy, agent.policy)
    assert changed(initial_policy_target, agent.policy_target)
    assert all(parameter.requires_grad for parameter in agent.q1.parameters())
    assert agent.q1.training


def test_td3_checkpoint_restores_actor_target_and_update_counter(
    monkeypatch, tmp_path
):
    torch.manual_seed(11)
    agent = make_tiny_agent(monkeypatch)
    agent.update()
    agent.update()
    checkpoint = tmp_path / "td3_checkpoint.pth"
    agent.save_model(checkpoint)

    restored = make_tiny_agent(monkeypatch)
    restored.load_model(str(checkpoint))

    assert restored.total_updates == 2
    for expected, actual in zip(
        agent.policy_target.parameters(), restored.policy_target.parameters()
    ):
        assert torch.equal(expected, actual)


def test_policy_action_is_bounded():
    torch.manual_seed(3)
    policy = td3.PolicyNetwork(
        ego_shape=(25, 25),
        global_shape=(50, 50),
        robot_dim=3,
        use_robot=True,
    )
    policy.eval()
    ego = torch.rand(2, 4, 25, 25)
    glob = torch.rand(2, 4, 50, 50)
    robot = torch.rand(2, 3)
    with torch.no_grad():
        action = policy.action(ego, glob, robot)

    assert action.shape == (2, 2)
    assert np.all(action.numpy() >= td3.TD3_ACTION_LOW)
    assert np.all(action.numpy() <= td3.TD3_ACTION_HIGH)
