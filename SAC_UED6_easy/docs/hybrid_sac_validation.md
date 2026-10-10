# Mode expectation and four-condition validation

Settings are in `configs/training/common.py`:

```python
PERIODIC_VALIDATION = True
VALIDATION_CYCLE_EPISODE = 500
VALIDATION_CONDITIONS = (
    "off_zero_command", "shuttle", "policy_stochastic", "policy_deterministic",
)
SAC_MODE_ESTIMATOR = "expectation"  # alternative: "gumbel"
```

Restart the trainer to load these settings. Existing compatible checkpoints
and replay can resume: the action, reward, network shapes and simulator
dynamics have not changed. Checkpoints record the estimator used. A resume
with a different estimator is an intentional algorithm change, not a fresh
training-seed comparison. Use a separate log directory for a fresh ablation.

## Learning

The default samples a reparameterized continuous move/speed, then enumerates
off/guide for the differentiated robot. Its actor objective is

`sum_m p(m|o) * [alpha * (logp_cont + log p(m|o)) - min(Q1,Q2)]`.

Teammate actions follow `ACTOR_TEAMMATE_ACTIONS`, and robot selection follows
`ACTOR_UPDATE_ROBOTS`. Current teammate actions are sampled without gradient.
Alpha receives the exact expected categorical log probability. The existing
single alpha and its floor are retained.

Bellman targets enumerate the entire team's categorical joint distribution
conditional on one continuous action sample. Three robots with two modes
require eight joint combinations. The double-Q minimum stays inside the
expectation; team-reward variants average robot Qs before taking the minimum.
Padded robots contribute no probability multiplicity. Discrete enumeration is
exact; continuous actions still use Monte Carlo sampling.
The critic's observation encodings are computed once and reused across mode
candidates; action-dependent layers still evaluate every candidate.

`gumbel` restores the original sampled targets and straight-through actor
estimator. Both methods execute one categorical mode during rollouts. Neither
changes deterministic inference or network parameter shapes. Expectation
requires more critic evaluations; this is a computational tradeoff.

## Validation

Each frozen checkpoint is evaluated on the same reserved generated levels,
hazard, robot count and crowd seed under all four conditions. The existing
100/200/400 m, difficulty 2/4/6, and 1/2/3 robot grid is retained: 27 pairs,
108 rollouts for the first cycle. Off and shuttle results are cached, so later
cycles need 54 policy rollouts. A running validation process causes scheduled
overlapping cycles to be skipped rather than launching another process.

Validation starts at eligible episode multiples after warmup. The trainer
writes records to `LOG_DIR/validation/validation_metrics.jsonl`; summaries go
to the local event log, TensorBoard, and W&B. The final zero-shot protocol
continues to use its existing deterministic policy/off conditions.

Example W&B keys:

- `eval/validation/200m/robots_3/policy_stochastic/hazard_person_steps`
- `eval/validation/paired/policy_deterministic/person_steps_reduction_vs_off`
- `eval/validation/paired/policy_stochastic/person_steps_reduction_vs_off`
- `eval/validation/paired/shuttle/person_steps_reduction_vs_off`
- `eval/validation/paired/policy_deterministic/person_steps_reduction_vs_shuttle`

Best-model selection retains the legacy key
`eval/validation/paired/person_steps_reduction_vs_off`, now explicitly backed
by deterministic policy/off pairs. Relative reductions exclude zero-exposure
controls; absolute differences and the number of relative pairs are logged.

The off control keeps robot bodies stationary with signals off. Shuttle moves
to hazard entry points with signals off, then guides at 0.6 of maximum speed
(1.2 m/s under current settings). Its waypoint offsets and speed are encoded
separately. Exit waiting and stall detection use simulation seconds, not
decision counts; durations are rounded up to the next decision boundary.

Stochastic policy evaluation seeds and restores the PyTorch RNG. Python and
NumPy crowd seeds match across conditions. Different actions can consume a
different number of crowd RNG draws, so this pairing aligns initial conditions
and seeds, not every subsequent external event.

Baseline cache keys include the level geometry/task, complete configuration,
robot count, seed, horizon, condition, and shuttle implementation version.
Old off-only cache entries do not match the new keys.
