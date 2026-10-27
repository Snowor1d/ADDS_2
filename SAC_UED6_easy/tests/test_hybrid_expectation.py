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
    # Shared heads: one move draw paired with both modes.
    agent._per_robot_hybrid = lambda o, m: (cont.unsqueeze(2).expand(1,2,2,1),
                                            torch.zeros(1,2,2), p, lp)
    agent._robot_q_min = lambda o,a,m,j: 2*a[:,j,0] + 3*a[:,j,-1] + 5*a[:,1-j,-1]
    base = torch.tensor([[[0.,1.,0.], [0.,1.,0.]]])
    chosen = torch.ones(1, 2, dtype=torch.bool)       # both robots decided
    loss, logp = agent._expected_actor_loss({}, base, torch.ones(1,2), chosen)
    expected = (.2 * (p * lp).sum(-1) - 2*cont[...,0] - 3*p[...,1]).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(logp, (p*lp).sum(-1).flatten())
    loss.backward()
    torch.testing.assert_close(cont.grad, torch.full_like(cont, -1.))
    expected_guide_grad = p[...,0]*p[...,1]*(.2*(lp[...,1]-lp[...,0])-3)/2
    torch.testing.assert_close(logits.grad[...,1], expected_guide_grad)
    torch.testing.assert_close(logits.grad[...,0], -expected_guide_grad)


def test_target_redraws_only_the_deciding_robots():
    """At robot j's next decision frame the robots that decide there are
    re-drawn (modes enumerated), the others keep their held actions, padding
    is ignored, the clipped minimum is inside the expectation, and the
    entropy bonus is j's own."""
    agent = object.__new__(SACAgent)
    agent.alpha = torch.tensor(1.)
    p = torch.tensor([[[.25,.75],[.6,.4],[.9,.1]],
                      [[.75,.25],[.3,.7],[.2,.8]]])
    agent._per_robot_hybrid = lambda o,m: (torch.zeros(2,3,2,1), torch.zeros(2,3,2),p,p.log())
    # Per robot output; robot 0's critic 1 is lower, robot 1's critic 2.
    base1 = torch.tensor([[0.,5.,99.],[0.,5.,99.]])
    base2 = torch.tensor([[3.,0.,99.],[3.,0.,99.]])
    effect = lambda a: (a[:,0,-1] + 10*a[:,1,-1])[:, None]   # guide indicators
    agent.q1_target = lambda o,a,m: (base1 + effect(a))*m
    agent.q2_target = lambda o,a,m: (base2 + effect(a))*m
    mask = torch.tensor([[1.,1.,0.],[1.,1.,0.]])
    held = torch.zeros(2,3,3)
    held[1,1,2] = 1.0                       # robot 1 of sample 1 keeps guiding
    deciding = torch.tensor([[True,True,False],[True,False,False]])
    v = agent._expected_next_value({}, mask, held, deciding, torch.tensor([0,0]))
    h = lambda q: -(q*q.log()).sum()
    expected = torch.tensor([.75 + 10*.4 + h(p[0,0]),        # both re-drawn
                             .25 + 10*1. + h(p[1,0])])       # robot 1 held
    torch.testing.assert_close(v, expected)
    # Robot 1's own value: its own entropy, and critic 2's lower baseline.
    v1 = agent._expected_next_value({}, mask, held, deciding[:1].expand(2,3), torch.tensor([1,1]))
    torch.testing.assert_close(v1[0], .75 + 10*.4 + h(p[0,1]))


def test_expected_actor_one_scores_only_selected_robots():
    agent = object.__new__(SACAgent)
    agent.cfg = SimpleNamespace(ACTOR_TEAMMATE_ACTIONS="stored", ACTOR_UPDATE_ROBOTS="one")
    agent.own_collision = True
    agent.alpha = torch.tensor(0.)
    cont = torch.zeros(2,3,1, requires_grad=True)
    p = torch.full((2,3,2),.5)
    agent._per_robot_hybrid = lambda o,m: (cont.unsqueeze(2).expand(2,3,2,1),
                                           torch.zeros(2,3,2),p,p.log())
    agent._robot_q_min = lambda o,a,m,j: a[:,j,0]
    chosen = torch.tensor([[False,True,False],[False,False,True]])
    loss,logp = agent._expected_actor_loss({},torch.zeros(2,3,3),
                                         torch.tensor([[1.,1.,0.],[1.,1.,1.]]),
                                         chosen)
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


def test_per_mode_moves_take_their_own_mode_gradient_and_target():
    """Under per-mode heads the off term reads and trains the off move only,
    the guide term the guide move only."""
    agent = object.__new__(SACAgent)
    agent.cfg = SimpleNamespace(ACTOR_TEAMMATE_ACTIONS="stored", ACTOR_UPDATE_ROBOTS="all")
    agent.own_collision = True
    agent.alpha = torch.tensor(0.)
    cont = torch.tensor([[[[.5], [-1.]]]], requires_grad=True)   # (B, N, K, C)
    p = torch.tensor([[[.3, .7]]])
    agent._per_robot_hybrid = lambda o, m: (cont, torch.zeros(1,1,2), p, p.log())
    # Q rewards moving with +2 in off and -3 in guide.
    q = lambda a: a[:,0,0] * (2*a[:,0,1] - 3*a[:,0,2])
    agent._robot_q_min = lambda o,a,m,j: q(a)
    loss, _ = agent._expected_actor_loss({}, torch.zeros(1,1,3), torch.ones(1,1),
                                         torch.ones(1,1,dtype=torch.bool))
    loss.backward()
    torch.testing.assert_close(cont.grad, torch.tensor([[[[-2*.3], [3*.7]]]]))

    agent.q1_target = agent.q2_target = lambda o,a,m: q(a)[:,None]*m
    v = agent._expected_next_value({}, torch.ones(1,1), torch.zeros(1,1,3),
                                   torch.ones(1,1,dtype=torch.bool), torch.tensor([0]))
    torch.testing.assert_close(v, torch.tensor([.3*2*.5 + .7*(-3)*(-1.)]))


def _policy(heads):
    from learn.networks import PolicyNetwork
    from sim.observation import obs_shapes
    cfg = resolve_config({'ACTOR_CONT_HEADS': heads}, check_data=False)
    torch.manual_seed(0)
    pol = PolicyNetwork(cfg)
    sh = obs_shapes(cfg)
    x = [torch.randn(5, *sh[k]) for k in ('ego', 'mid', 'glob', 'state')]
    return pol, x


@pytest.mark.parametrize('estimator', ['expectation', 'gumbel'])
def test_per_mode_policy_executes_the_chosen_modes_move(estimator):
    from sim import robot_action as ra
    pol, x = _policy('per_mode')
    pol.mode_estimator = estimator
    K = ra.ACTION_DIM - ra.CONT_DIM
    assert pol.mean_head.out_features == K * ra.CONT_DIM
    torch.manual_seed(1)
    cont, logpc, probs, logpm = pol.sample_hybrid(*x)
    assert cont.shape == (5, K, ra.CONT_DIM) and logpc.shape == (5, K)
    assert not torch.allclose(cont[:, 0], cont[:, 1])
    torch.manual_seed(1)
    a, logp = pol.sample_action(*x)
    idx = a[:, ra.MODE].argmax(-1)
    rows = torch.arange(5)
    torch.testing.assert_close(a[:, :ra.CONT_DIM], cont[rows, idx])
    torch.testing.assert_close(logp, logpc[rows, idx] + logpm[rows, idx])

    det = pol.deterministic_action(*x)
    mean, _, logits = pol(*x)
    idx = logits.argmax(-1)
    torch.testing.assert_close(det[:, :ra.CONT_DIM], 4*torch.sigmoid(mean[rows, idx]) - 2)
    assert torch.equal(det[:, ra.MODE].argmax(-1), idx)


def test_shared_heads_pair_one_draw_with_every_mode_and_keep_old_shapes():
    from sim import robot_action as ra
    pol, x = _policy('shared')
    assert tuple(pol.mean_head.weight.shape)[0] == ra.CONT_DIM
    cont, logpc, _, _ = pol.sample_hybrid(*x)
    assert torch.equal(cont[:, 0], cont[:, 1])
    assert torch.equal(logpc[:, 0], logpc[:, 1])


def test_checkpoint_of_another_head_layout_is_refused(tmp_path):
    from learn.replay import SchemaMismatch
    cfg = lambda h: resolve_config({'ACTOR_CONT_HEADS': h}, check_data=False)
    path = str(tmp_path / 'ck.pth')
    SACAgent(cfg('shared')).save(path)
    with pytest.raises(SchemaMismatch):
        SACAgent(cfg('per_mode')).load(path)
    # A checkpoint from before the setting existed is a shared-head one.
    ck = torch.load(path, weights_only=False)
    del ck['network']['actor_cont_heads']
    torch.save(ck, path)
    SACAgent(cfg('shared')).load(path)
    with pytest.raises(SchemaMismatch):
        SACAgent(cfg('per_mode')).load(path)


@pytest.mark.parametrize('bad', ['per-mode', '', None])
def test_invalid_actor_cont_heads_is_rejected(bad):
    with pytest.raises(ConfigError):
        resolve_config({'ACTOR_CONT_HEADS': bad}, check_data=False)


def test_per_mode_update_trains_every_modes_head(tmp_path):
    from learn.replay import ReplayBuffer, StaticStore
    from sim import robot_action as ra
    from tests.test_madrl import _fill
    cfg = resolve_config({'ACTOR_CONT_HEADS': 'per_mode', 'BATCH_SIZE': 4},
                         check_data=False)
    store = StaticStore(str(tmp_path))
    replay = ReplayBuffer(cfg, 100, store)
    agent = SACAgent(cfg)
    _fill(cfg, replay, store, agent, episodes=((1,16),(3,16)))
    before = agent.policy.mean_head.weight.detach().clone()
    values = agent.update(replay.sample(4, np.random.default_rng(5)))
    assert all(np.isfinite(v) for v in values.values())
    after = agent.policy.mean_head.weight.detach()
    for k in range(ra.ACTION_DIM - ra.CONT_DIM):
        rows = slice(k * ra.CONT_DIM, (k + 1) * ra.CONT_DIM)
        assert not torch.equal(before[rows], after[rows]), k


def test_actor_trains_only_the_robots_that_decided():
    agent = object.__new__(SACAgent)
    mask = torch.tensor([[1.,1.,0.],[1.,1.,1.]])
    decided = torch.tensor([[True,False,False],[False,True,True]])
    agent.cfg = SimpleNamespace(ACTOR_UPDATE_ROBOTS="all")
    assert agent._actor_chosen(mask, decided, torch.tensor([0,1])).tolist() == \
        [[True,False,False],[False,True,True]]
    agent.cfg = SimpleNamespace(ACTOR_UPDATE_ROBOTS="one")
    assert agent._actor_chosen(mask, decided, torch.tensor([0,2])).tolist() == \
        [[True,False,False],[False,False,True]]
