"""Regression checks for the outdoor task's zero-shot score."""

from types import SimpleNamespace

from learn.zero_shot import EpisodeMetrics, validation_levels


class Zone:
    def contains(self, x, y):
        return x < 0


class Model:
    def __init__(self):
        self.danger_zone = Zone()
        self.crowds = [SimpleNamespace(unique_id=1, xy=(-1.0, 0.0),
                                       type=1, dead=False)]
        self.total_agents = 1
        self.step_count = 0
        self._empty_since = None

    def agents_in_danger(self):
        return sum(int(not person.dead and self.danger_zone.contains(*person.xy))
                   for person in self.crowds)

    def is_cleared_and_held(self):
        if self.agents_in_danger():
            self._empty_since = None
            return False
        if self._empty_since is None:
            self._empty_since = self.step_count
        return self.step_count - self._empty_since >= 2

    def cleared_at(self):
        return self._empty_since


def test_reentry_and_sustained_clearance_are_distinct():
    model = Model()
    metrics = EpisodeMetrics()
    outside = set()
    for step, x in enumerate((1.0, -1.0, 1.0, 1.0, 1.0), start=1):
        model.step_count = step
        model.crowds[0].xy = (x, 0.0)
        outside = metrics.observe(model, outside)
    result = metrics.summary()
    assert result["first_empty_step"] == 1
    assert result["held_clear_step"] == 3
    assert result["reentries"] == 1
    assert result["held_clear_success"] == 1
    assert result["mean_occupancy"] == 0.2
    assert result["final_occupancy"] == 0.0


def test_outflow_is_reported_without_false_reentry():
    model = Model()
    metrics = EpisodeMetrics()
    metrics._active_ids = {1}
    model.step_count = 1
    model.crowds[0].xy = (1.0, 0.0)
    outside = metrics.observe(model, set())
    model.step_count = 2
    model.crowds[0].dead = True
    metrics.observe(model, outside)
    result = metrics.summary()
    assert result["outflows"] == 1
    assert result["reentries"] == 0


def test_open_flow_exposure_and_departure_reasons_are_separate():
    model = Model()
    metrics = EpisodeMetrics(initial_population=1)
    metrics._active_ids = {1}
    model.crowds[0].ever_acted = True
    model.crowds[0].outflow_reason = None
    model.step_count = 1
    outside = metrics.observe(model, set())
    model.crowds.append(SimpleNamespace(
        unique_id=2, xy=(1.0, 0.0), type=1, dead=False,
        ever_acted=False, outflow_reason=None))
    model.total_agents = 2
    model.crowds[0].dead = True
    model.crowds[0].outflow_reason = "evacuation_departure"
    model.step_count = 2
    metrics.observe(model, outside)
    result = metrics.summary()
    assert result["inflows"] == 1
    assert result["outflows"] == 1
    assert result["informed_departures"] == 1
    assert result["evacuation_departures"] == 1
    assert result["informed_trip_outflows"] == 0
    assert result["background_departures"] == 0
    assert result["informed_departure_fraction"] == 1.0
    assert result["hazard_person_steps"] == 1
    assert result["mean_active_occupancy"] == 0.5


def test_validation_set_spans_the_trained_sizes():
    """Models are selected on generated levels at 100, 200 and 400 m, drawn
    from seeds that training may never draw."""
    from configs import resolve_config, _reserved_seed
    cfg = resolve_config(check_data=False)
    levels = [level for _, level in validation_levels(cfg)]
    assert {level.width for level in levels} == set(cfg.VALIDATION_SIZES_M)
    assert {level.difficulty for level in levels} == set(cfg.VALIDATION_DIFFICULTIES)
    assert all(level.danger is not None for level in levels)
    assert all(_reserved_seed(level.source_seed, cfg.TRAIN_RESERVED_SEED_RANGES)
               for level in levels)


def test_four_condition_summary_and_selection_use_deterministic_policy():
    from learn.zero_shot import summarise, selection_score
    rows = [dict(scenario="fixed",size_m=200,robot_num=3,seed=9,condition=c,
                 hazard_person_steps=n,reentries_after_clear=2)
            for c,n in [('off_zero_command',100),('shuttle',60),
                        ('policy_stochastic',80),('policy_deterministic',40)]]
    out = summarise(rows,'eval/validation')
    assert abs(selection_score(out,'eval/validation')-.6)<1e-6
    assert abs(out['eval/validation/paired/policy_stochastic/person_steps_reduction_vs_off']-.2)<1e-6
    assert abs(out['eval/validation/paired/shuttle/person_steps_reduction_vs_off']-.4)<1e-6
    assert abs(out['eval/validation/paired/policy_deterministic/person_steps_reduction_vs_shuttle']-1/3)<1e-6


def test_zero_exposure_control_does_not_create_relative_score():
    from learn.zero_shot import summarise, selection_score
    rows = [dict(scenario="empty",size_m=200,robot_num=3,seed=1,condition=c,
                 hazard_person_steps=n,reentries_after_clear=0)
            for c,n in [('off_zero_command',0),('policy_deterministic',10)]]
    out=summarise(rows,'eval/validation')
    assert selection_score(out,'eval/validation') is None
    assert out['eval/validation/paired/policy_deterministic/person_steps_difference_vs_off']==10


def test_four_conditions_dispatch_cache_and_reproducible_stochastic_policy(monkeypatch):
    import numpy as np
    import torch
    from configs import resolve_config
    import learn.zero_shot as z
    from ued.level import Level
    from sim.danger import DangerZone
    cfg=resolve_config(check_data=False)
    level=Level([],[],20,width=80,height=80,
                danger=DangerZone('circle',40,40,radius=12),robot_num=3)
    calls=[]
    class Agent:
        device=torch.device('cpu')
        gamma=cfg.gamma()
        def act(self,obs,deterministic=False):
            from sim.robot_action import encode
            calls.append(deterministic)
            move=(0.,0.) if deterministic else tuple(torch.rand(2).tolist())
            return np.stack([encode(move,'guide',speed=.6)]*obs['state'].shape[0])
    agent=Agent()
    state=torch.random.get_rng_state().clone()
    first=z.evaluate_level(agent,level,7,'policy_stochastic',cfg,max_steps=8)
    assert torch.equal(state,torch.random.get_rng_state())
    second=z.evaluate_level(agent,level,7,'policy_stochastic',cfg,max_steps=8)
    for key in ['total_reward','hazard_person_steps','command_norm_mean','robot_speed_mean']:
        assert first[key]==second[key]
    assert calls and not any(calls)
    cache={}
    records=z.paired_records(agent,[('fixed',level)],cfg,500,lambda name:(7,),
                            (3,),off_cache=cache,max_steps=8,
                            conditions=cfg.VALIDATION_CONDITIONS)
    assert {r['condition'] for r in records}==set(z.COMPARISON_CONDITIONS)
    assert len(cache)==2
    assert any(calls)
    original=z.evaluate_level
    cached_calls=[]
    def counted(*args,**kwargs):
        cached_calls.append(args[3])
        return original(*args,**kwargs)
    monkeypatch.setattr(z,'evaluate_level',counted)
    z.paired_records(agent,[('fixed',level)],cfg,1000,lambda name:(7,),
                     (3,),off_cache=cache,max_steps=8,conditions=cfg.VALIDATION_CONDITIONS)
    assert set(cached_calls)=={'policy_stochastic','policy_deterministic'}


def test_shuttle_encodes_waypoint_speed_and_waits_in_seconds():
    import numpy as np
    from configs import resolve_config
    from sim import robot_action as ra
    from validation.shuttle_baseline import ShuttlePolicy
    cfg=resolve_config(check_data=False)
    policy=object.__new__(ShuttlePolicy)
    policy.cfg=cfg
    policy.ra=ra
    robot=SimpleNamespace(xy=[0.,0.])
    triangle=((19.,-1.),(21.,-1.),(20.,2.))  # centre=(20,0)
    action=policy._command(robot,triangle,'guide')
    np.testing.assert_allclose(action[:2],[2.,0.])
    assert abs(ra.speed_fraction(action)-.6)<1e-6
    policy.m=SimpleNamespace(step_count=16,find_mesh=lambda xy:None)
    policy.entries=[triangle]
    policy.state=[{'phase':'wait','wait_until':10.,'target':None,
                   'best':float('inf'),'last_progress':None}]
    assert ra.decode(policy._act_one(0,robot))[1]=='guide'  # 8 seconds
    policy.m.step_count=24
    move,mode,_=ra.decode(policy._act_one(0,robot))
    assert mode=='off' and move[0]>0  # depart at the next boundary, 12 seconds
