"""Analytic gradient and masking checks for the categorical expectation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs import ConfigError, resolve_config
from learn.sac import SACAgent


def test_actor_matches_analytic_loss_and_mode_gradient():
    agent = object.__new__(SACAgent)
    agent.cfg = SimpleNamespace(ACTOR_TEAMMATE_ACTIONS="stored", ACTOR_UPDATE_ROBOTS="all")
    agent.own_collision = True
    agent.alpha = torch.tensor(0.2)
    logits = torch.tensor([[[np.log(.25), np.log(.75)],
                            [np.log(.6), np.log(.4)]]], dtype=torch.float32, requires_grad=True)
    cont = torch.tensor([[[.2], [.4]]], requires_grad=True)
    p = logits.softmax(-1)
    lp = logits.log_softmax(-1)
    agent._per_robot_hybrid = lambda o, m: (cont, torch.zeros(1,2), p, lp)
    agent._robot_q_min = lambda o,a,m,j: 2*a[:,j,0] + 3*a[:,j,-1] + 5*a[:,1-j,-1]
    base = torch.tensor([[[0.,1.,0.], [0.,1.,0.]]])
    loss, logp = agent._expected_actor_loss({}, base, torch.ones(1,2), torch.tensor([0]))
    expected = (.2 * (p * lp).sum(-1) - 2*cont[...,0] - 3*p[...,1]).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logp, (p*lp).sum(-1).flatten())
    loss.backward()
    torch.testing.assert_close(cont.grad, torch.full_like(cont, -1.))
    expected_guide_grad = p[...,0]*p[...,1]*(.2*(lp[...,1]-lp[...,0])-3)/2
    torch.testing.assert_close(logits.grad[...,1], expected_guide_grad)
    torch.testing.assert_close(logits.grad[...,0], -expected_guide_grad)


def test_target_expectation_handles_padding_and_clipped_team_order():
    agent = object.__new__(SACAgent)
    agent.alpha = torch.tensor(0.)
    p = torch.tensor([[[.25,.75],[.6,.4],[.9,.1]],
                      [[.75,.25],[.3,.7],[.2,.8]]])
    agent._per_robot_hybrid = lambda o,m: (torch.zeros(2,3,1), torch.zeros(2,3),p,p.log())
    agent.q1_target = lambda o,a,m: (torch.tensor([[0.,10.,99.],[0.,10.,99.]])+a[:,0,-1,None])*m
    agent.q2_target = lambda o,a,m: (torch.tensor([[10.,0.,99.],[10.,0.,99.]])+a[:,0,-1,None])*m
    mask = torch.tensor([[1.,1.,0.],[1.,0.,0.]])
    team = agent._expected_next_value({}, mask, per_robot=False)
    own = agent._expected_next_value({}, mask, per_robot=True)
    torch.testing.assert_close(team, torch.tensor([5.75,.25]))
    torch.testing.assert_close(own, torch.tensor([[.75,.75,0.],[.25,0.,0.]]))


def test_expected_actor_one_scores_only_selected_robots():
    agent = object.__new__(SACAgent)
    agent.cfg = SimpleNamespace(ACTOR_TEAMMATE_ACTIONS="stored", ACTOR_UPDATE_ROBOTS="one")
    agent.own_collision = True
    agent.alpha = torch.tensor(0.)
    cont = torch.zeros(2,3,1, requires_grad=True)
    p = torch.full((2,3,2),.5)
    agent._per_robot_hybrid = lambda o,m: (cont,torch.zeros(2,3),p,p.log())
    agent._robot_q_min = lambda o,a,m,j: a[:,j,0]
    loss,logp = agent._expected_actor_loss({},torch.zeros(2,3,3),
                                         torch.tensor([[1.,1.,0.],[1.,1.,1.]]),
                                         torch.tensor([1,2]))
    loss.backward()
    torch.testing.assert_close(cont.grad[...,0],torch.tensor([[0.,-.5,0.],[0.,0.,-.5]]))
    assert logp.shape == (2,)


@pytest.mark.parametrize('bad', ['softmax','',None])
def test_invalid_mode_estimator_is_rejected(bad):
    with pytest.raises(ConfigError):
        resolve_config({'SAC_MODE_ESTIMATOR':bad},check_data=False)


def test_reused_critic_features_preserve_values_and_action_gradients():
    from learn.networks import CentralizedCritic
    from sim.observation import obs_shapes
    from sim.robot_action import ACTION_DIM
    cfg=resolve_config(check_data=False)
    q=CentralizedCritic(cfg,ACTION_DIM)
    for parameter in q.parameters():
        parameter.requires_grad_(False)
    shapes=obs_shapes(cfg)
    obs={k:torch.zeros(1,3,*shape) for k,shape in shapes.items() if k!='priv'}
    obs['priv']=torch.zeros(1,*shapes['priv'])
    obs['priv_scalars']=torch.zeros(1,3)
    mask=torch.tensor([[1.,1.,0.]])
    action=torch.randn(1,3,ACTION_DIM,requires_grad=True)
    direct=q(obs,action,mask)
    reused=q.values_from_features(q.encode_observation(obs),action,mask)
    torch.testing.assert_close(direct,reused)
    grad1=torch.autograd.grad(direct.sum(),action)[0]
    grad2=torch.autograd.grad(reused.sum(),action)[0]
    torch.testing.assert_close(grad1,grad2)
    assert grad2[:,:2].abs().sum()>0


@pytest.mark.parametrize('estimator',['expectation','gumbel'])
def test_both_estimators_update_real_replay_and_mode_parameters(tmp_path,estimator):
    from learn.replay import ReplayBuffer,StaticStore
    from tests.test_madrl import _fill
    cfg=resolve_config({'SAC_MODE_ESTIMATOR':estimator,'BATCH_SIZE':4},check_data=False)
    store=StaticStore(str(tmp_path))
    replay=ReplayBuffer(cfg,100,store)
    agent=SACAgent(cfg)
    _fill(cfg,replay,store,agent,episodes=((1,16),(3,16)))
    before=agent.policy.mode_head.weight.detach().clone()
    values=agent.update(replay.sample(4,np.random.default_rng(5)))
    assert all(np.isfinite(v) for v in values.values())
    assert not torch.equal(before,agent.policy.mode_head.weight)
    assert all(p.requires_grad for p in agent.q1.parameters())
    assert not agent._actor_critic_features
