"""Training and evaluation settings shared by both training modes.

configs/training/ holds three files:

    common.py    this one: SAC and critic, exploration, reward, replay,
                 validation and final zero-shot maps and seeds, logging and
                 W&B, and TRAIN_MAP_SOURCE, which picks the training mode
    ued.py       the curriculum and its task parameters (UED_*)
    dataset.py   the OSM crops to train on and their task parameters
                 (DATASET_*)

See configs/__init__.py for how the files are merged and checked.

Constants only; importing this module has no side effects. Credentials never
go in this file: W&B reads its key from the environment or ~/.netrc, and the
resolved configuration refuses to start if a setting looks like a secret.
"""

# ------------- BASIC PARAMETERS ------------------
LR = 1e-4
# Joint transitions. A starting point for measurement, not a tuned value:
# docs/outdoor_madrl_redesign.md section 6 and docs/replay_memory_budget.md
# record what one transition costs and the peak memory actually measured.
BUFFER_SIZE = 1_000_000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1              # intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500
DEVICE = "cuda"
# Discount per ACTION_SCALE simulation steps (2 s), one per
# ROBOT_ACTION_MODE; `cfg.gamma()` picks the one in use. Per 2 s rather than
# per decision, because a decision's length varies (ROBOT_DECISION_MAX_S_<MODE>):
# the learner converts it to a per-step factor gamma ** (1 / ACTION_SCALE),
# sums a held action's rewards with that factor and bootstraps with its k-th
# power, so every transition is discounted for the steps it actually lasted.
GAMMA_VELOCITY = 0.99
GAMMA_WAYPOINT = 0.99
# Not read anywhere.
WD_Q = 3e-4                      # weight decay
WD_PI = 0.0
SCALE_CHECK = 0                  # want to check reward scale?

# --------------- EPSILON-EXPLORATION ------------------
# Uniform random actions only while the buffer warms up, before the first
# update; after that SAC explores through its own stochastic policy and
# entropy bonus. A long linear decay (3000 episodes) kept 74% of actions
# random at episode 1346, so the logged returns measured random robots and
# the policy's own effect could not show (2026-10-04).
EPSILON_MIN = 0
START_EPSILON = 1
SCHEDULER_TYPE = "l"
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 1
START_DECAY_STEP = START_UPDATE_EPISODE
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0
DECAY_MODE = 'episode'

# -------------- PATH -------------------
# Distinct from the other copies so runs can share a machine without
# clobbering each other's logs, checkpoints or tensorboard port.
# It was "Log_SAC_UED3", the same folder a SAC_UED3 run on this machine
# writes to, so a SAC_UED4 run would have resumed from and overwritten it.
LOG_DIR = "Log_SAC_UED6_easy_fullinfo_clock_pbrs"
PORT_NUM = 9001

# --------------- NETWORK ---------------
# Encoder family and size for the actor and the critic (learn/networks.py).
#   "cnn"     three stride-2 convolutions, flattened into a linear layer; most
#             parameters sit in that position-specific layer. "m" is the
#             original network.
#   "impala"  three residual stacks with coordinate channels, pooled to 4x4;
#             most parameters sit in convolutions, which is what generalised
#             to unseen levels in Procgen (Cobbe et al. 2020).
# Sizes are matched across the two families; parameters (actor / one critic):
#   m  ~5.6 M / ~8.6 M    l  ~12 M / ~18 M    xl ~21 M / ~31 M
# docs/network_sizes.md has the exact counts and the update cost of each.
# A checkpoint only loads into the family and size it was trained with.
NET_ENCODER = "cnn"                 # "cnn" | "impala"
NET_SIZE = "m"                      # "m" | "l" | "xl"

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
# -5 rather than -20: a std of e^-20 is a deterministic actor whose squashed
# output can sit on the edge of its range with no gradient back (2026-10-05).
LOG_STD_MIN = -5
# Entropy temperature. With ALPHA_AUTO the temperature is learned so that
# the policy's entropy tracks ALPHA_TARGET_ENTROPY (Haarnoja et al. 2018,
# "Soft Actor-Critic Algorithms and Applications"); ALPHA_START is then only
# its initial value. Without it alpha stays at ALPHA_START, and changing any
# reward weight also changes how much entropy is worth relative to reward.
# The entropy is that of the whole action: the squashed move (and signal)
# plus the mode choice. None sets the usual -(continuous action dimensions),
# -2 without "direct" and -4 with it.
ALPHA_AUTO = True
ALPHA_START = 0.2
ALPHA_LR = 3e-4
ALPHA_TARGET_ENTROPY = None
# Floor on the learned temperature. A design value relative to a reward of a
# few units per decision; without a floor alpha collapsed to 0.001.
ALPHA_MIN = 0.05

# --------------- ACTOR UPDATE (multi-robot) ---------------
# A replay sample is one robot's decision (docs/async_decisions.md). Whose
# action the actor loss re-draws from the current policy and differentiates
# at the sampled frame:
#   "all"  every robot that decided at that frame (under
#          ROBOT_DECISION_SCOPE = "team", every real robot)
#   "one"  only the robot whose decision was sampled
# What the other robots are taken to do meanwhile:
#   "stored"   the actions they had at that frame, as recorded in the replay
#              (MADDPG): the best reply to what teammates were doing.
#   "current"  for teammates that decided at that frame too, what the
#              current policy would do now (without gradient through them);
#              teammates carrying out an earlier decision keep it.
ACTOR_UPDATE_ROBOTS = "all"         # "all" | "one"
ACTOR_TEAMMATE_ACTIONS = "stored"   # "stored" | "current"
# "expectation": enumerate each actor's modes and the target's joint modes.
# "gumbel": original hard straight-through Gumbel-Softmax estimator.
# Rollouts always execute one categorical mode; expectation is for learning.
SAC_MODE_ESTIMATOR = "expectation"   # "expectation" | "gumbel"
# The actor's continuous part (move, speed, heading) given the mode:
#   "shared"    pi(m, a | o) = pi(m | o) pi(a | o): one move distribution,
#               drawn once and paired with every mode.
#   "per_mode"  pi(m, a | o) = pi(m | o) pi(a | o, m): the encoder and MLP are
#               shared, the mean and log-std heads are one per mode, so "off"
#               can run into the hazard while "guide" walks at a pace the
#               crowd can follow, from the same observation.
# The mode-weighted Q (SAC_MODE_ESTIMATOR) is unchanged; each mode's term
# reads its own move. A checkpoint only loads under the layout it was
# trained with.
ACTOR_CONT_HEADS = "shared"         # "shared" | "per_mode"

ENABLE_TIMER = True
N_ENVS = 12
# Gradient updates per stored frame at which some robot decided (and per
# episode's final frame). Measured on this machine
# (docs/replay_memory_budget.md): one update at batch 128 with three robots
# costs about 0.18 s to rebuild (overlapped with the previous update) plus
# 0.18 s on the GPU, while 12 workers on the generated 0.6/0.3/0.1 size mix produce
# roughly 11 decision instants a second. 1.0 made the learner the bottleneck
# and stalled the workers on a full queue.
# Per decision. Event-driven decisions last ~10 s, so an episode holds ~45 of
# them instead of 125 at fixed 4 s; 2.0 keeps the updates per episode near
# the earlier runs (2026-10-15).
UPDATES_PER_TRANSITION = 4.0
POLICY_BROADCAST_INTERVAL = 10
FiLM_USE = True
EGO_USE = True

# --------------- AUXILIARY OSM EVALUATION ---------------
# Four real crops evaluated from the command line (python3 -m cli.evaluate
# auxiliary). A report, never a model-selection signal: periodic evaluation,
# which models are selected on, uses the generated validation levels under
# MAP SPLITS below, and the final zero-shot site is kept apart from all of
# these.
ZSG_REAL_SITES = ()
ZSG_REAL_SIZE = 100
ZSG_ITERATION = 1
ZSG_ROBOT_NUM = (3,)             # a tuple: (3) without the comma is the int 3

# --------------- RESUMING AND TRANSFER ---------------
# "latest_compatible" resumes the newest checkpoint in LOG_DIR whose schema
# versions match this configuration, and refuses to start if the newest one
# does not match (it never silently starts over on top of an old run).
# "fresh" ignores what is there. Loading weights trained under different
# observation, reward or action schemas is only allowed as an explicit
# experiment: name the checkpoint here, and the run records it.
RESUME_MODE = "latest_compatible"   # "latest_compatible" | "fresh"
ALLOW_WEIGHT_TRANSFER_FROM = None

# --------------- CENTRALISED CRITIC ---------------
# The critic may see a coarse map of where everyone actually is, during
# training only. It is never an actor input: centralised training,
# decentralised execution.
CRITIC_PRIVILEGED_CROWD = True

# --------------- TASK REWARD (rew-v2 terms; rew-v3 credit) ---------------
# Per simulation step, summed over the steps an action is held:
#
#   person_time      - persons inside the hazard x AGENT_TIME_STEP / N_ref
#   remaining_path   - sum over those persons of (geodesic metres to safety
#                      / D_ref) x AGENT_TIME_STEP / N_ref
#   reentry          - entries into the hazard by persons still in the crop
#                      who had been clear of it by DANGER_SAFE_MARGIN_M, / N_ref.
#                      The margin keeps boundary jitter from counting as
#                      repeated entries. Leaving the crop is not an entry.
#   collision        - robots that hit a wall this step, summed over the team
#
# N_ref and D_ref are fixed when the episode starts: the persons then inside
# the hazard (at least REWARD_MIN_REFERENCE_POPULATION) and the worst geodesic
# escape distance from inside it (at least REWARD_MIN_REFERENCE_DISTANCE_M). A
# cumulative headcount that grows with inflow is never a denominator: it would
# make the same crowd inside the hazard cost less as the episode goes on.
#
# Leaving the crop is never rewarded as such. A person walking out of the hazard
# and then out of the crop stops counting toward person_time, which is the
# only credit it earns; a person outside the hazard leaving on its own trip
# changes nothing.
#
# The weights are design values set before any policy was trained on them, not
# measured quantities.
# rew-v3: the same terms, but each robot is charged only its own collisions
# (team task reward + own collision); see learn/sac.py. rew-v2 charges the
# team sum to every robot alike. rew-v6 is rew-v5 plus learning-only
# potential-based shaping (REWARD_W_SHAPE_* below).
REWARD_VERSION = "rew-v6-pbrs"
REWARD_W_PERSON_TIME = 5.0
REWARD_W_REMAINING_PATH = 0
REWARD_W_REENTRY = 0
REWARD_W_COLLISION = 0.05
# rew-v4 only. Both are per N_ref like the task terms, so their meaning holds
# across crop sizes; read them in person-steps inside the hazard (a person in
# it costs REWARD_W_PERSON_TIME * AGENT_TIME_STEP = 2.5 per step / N_ref).
#   guide_use   per robot-step in guide mode. Signalling has a cost: it pulls
#               people off their way and spends their trust.
#   bystander   per person drawn into following who was at no risk then:
#               outside the hazard, its own way on not passing through it.
# Calibrated against the shuttle baseline (docs/outdoor_madrl_redesign.md).
REWARD_W_GUIDE_USE = 0
REWARD_W_BYSTANDER = 0
# rew-v5 only, waypoint mode. Per decision, the distance the requested target
# was moved to reach walkable, reachable ground, over ROBOT_WAYPOINT_RANGE_M:
# at most this much. 79% of random 20 m targets on dotonbori were moved
# (median 9.8 m), and every target in one building moves to the same point, a
# flat stretch the actor gets no gradient from. This gives it one, so it
# learns to ask for ground it can reach; the projection still keeps every
# target reachable. Kept small: short targets are moved less often, and a
# large weight would bias the robot towards timid steps. A design value.
REWARD_W_PROJECTION = 0.0    # easy sanity run: shaping off
# rew-v6 only: potential-based shaping, F = g * Phi(s') - Phi(s) every step
# (sim/rewards.py). It moves credit in time without changing which policy is
# best (Ng, Harada and Russell 1999), and is left out of the task reward,
# `episode/total_reward` and every evaluation metric; it is logged as
# episode/reward/shape_progress and shape_approach.
#   SHAPE_APPROACH  per robot. Phi = 1 - exp(-S / REWARD_SHAPE_APPROACH_PEOPLE),
#                   S = sum over persons in the hazard nobody has reached yet
#                   of max(0, 1 - metres to the robot / REWARD_SHAPE_APPROACH_CAP_M):
#                   S counts the unreached people near the robot, nearer ones
#                   more, and Phi saturates once S passes a few PEOPLE. Phi is
#                   in [0, 1), so its discounted sum over an episode is at
#                   most the weight. Straight-line metres and a smooth sum:
#                   a walking-distance nearest-person version was 98% jumps
#                   (sim/rewards.py).
#   SHAPE_PROGRESS  team. Phi = -sum over persons inside of min(1, metres to
#                   safety / D_ref) / N_ref: followers walked out pay at once,
#                   followers walked back in cost at once.
# Weights from a one-decision probe on the validation levels
# (docs/review_20261013 section 5.3) put a decisive move at roughly 1 per
# 4 s decision against a task reward of about -11. PEOPLE from
# docs/review_20261013/shaping_kernel.py (3 levels x shuttle / random /
# still robots, 800 steps): S has median 5-14 and p90 up to 68, so 5 left
# Phi above 0.9 at 33-44% of steps; at 10, 12-25%, a robot's approach term
# per 4 s has sd 1.7-1.8 when it moves against 0.6 when it stands still.
# Design values, not tuned on returns.
REWARD_W_SHAPE_APPROACH = 0 #로봇별로 붙음
REWARD_W_SHAPE_PROGRESS = 150 #팀 전체에 수여
REWARD_SHAPE_APPROACH_CAP_M = 50.0
REWARD_SHAPE_APPROACH_PEOPLE = 10.0
REWARD_MIN_REFERENCE_POPULATION = 5
REWARD_MIN_REFERENCE_DISTANCE_M = 5.0
# What the terms are divided by.
#   "episode"  N_ref and D_ref of this episode, as above: every episode weighs
#              the same however many people its hazard held, so one person's
#              second counts more in a small incident than in a large one.
#   "fixed"    the constants below for every episode: the objective is then
#              plain person-seconds (and metres) inside the hazard, scaled.
# The defaults keep the reward near its "episode" size on the current
# training crops (200 m, hazard 0.2-0.4 of the crop: about 150-250 people
# inside, tens of metres to walk out).
REWARD_REFERENCE = "episode"       # "episode" | "fixed"
REWARD_FIXED_N_REF = 200.0
REWARD_FIXED_D_REF = 50.0

# --------------- REPLAY STORAGE ---------------
# Static layers (buildings, hazard, path fields) are stored once per map
# geometry and hazard placement, keyed by both, in RAM with a disk copy under
# LOG_DIR so an evicted entry can be rebuilt. Dynamic data is stored once per
# decision instant per robot and the stacks are rebuilt from indices.
REPLAY_STATIC_CACHE_ENTRIES = 256
REPLAY_STATIC_DIR = "replay_static"

# --------------- MAP SPLITS ---------------
# Every map a run touches is declared here, and the resolved configuration
# checks the three sets against each other at start-up: geographic place,
# spatial overlap of crops, generation seeds and hazard seeds.
#
# Training maps: where each episode's level comes from.
#
#   "dataset"  real OSM crops, site x size uniformly, with the task
#              parameters in configs/training/dataset.py
#   "ued"      generated citygen levels chosen by the curriculum
#              (UED_METHOD "accel") or at random ("dr"), with the task
#              parameters in configs/training/ued.py
TRAIN_MAP_SOURCE = "dataset"        # "dataset" | "ued"

# Validation: OSM crops VALIDATION_SITES x VALIDATION_SIZES_M, each turned
# into VALIDATION_SEEDS_PER_SITE tasks by the training task sampler
# (learn/training_maps.py, DATASET_* task ranges) at fixed seeds. None means
# the training value (DATASET_SITES, DATASET_SIZES_M): same buildings as
# training, fresh hazards and crowds, so it measures fit, not generalisation.
# Naming sites outside DATASET_SITES makes it a held-out validation instead.
# Either way the final zero-shot crop may not be used.
# A frozen policy is evaluated in a separate process under the conditions
# below; off/shuttle baselines are cached. Best-model selection uses the
# deterministic policy's paired person-time reduction against off.
VALIDATION_SITES = None            # None = DATASET_SITES  #VALIDATION_SITES = ("gangnam_superblock", "kreuzberg")
VALIDATION_SIZES_M = None          # None = DATASET_SIZES_M
VALIDATION_SEEDS_PER_SITE = 3
VALIDATION_SEED_BASE = 950_000
VALIDATION_ROBOT_COUNTS = (3,)
VALIDATION_CYCLE_EPISODE = 500
VALIDATION_CONDITIONS = ("off_zero_command", "shuttle",
                         "policy_deterministic")
# False skips evaluation and best_validation.pth selection. Final zero-shot
# remains a separate CLI evaluation after model selection.
PERIODIC_VALIDATION = True

# Generation and hazard seeds training may never draw: the validation block
# above and the final zero-shot block below.
TRAIN_RESERVED_SEED_RANGES = ((950_000, 960_000), (970_000, 980_000))

# Final zero-shot: one real crop, fixed before any policy is evaluated on it and
# never used to select a model. Pre-registered 2026-09-23.
#
# Myeongdong is not among the auxiliary OSM sites in ZSG_REAL_SITES or the
# training DATASET_SITES, and no other crop of it may appear in training or
# validation (its 100 and 200 m
# crops share its centre). The stored 400 m level holds 800 people, 0.0203
# persons per square metre of walkable ground, below the 0.04-0.05 target;
# FINAL_ZERO_SHOT_DENSITY re-derives the headcount from the walkable area.
# Set it to None to run at the stored density and report it as low-density.
FINAL_ZERO_SHOT_SITE = "myeongdong"
FINAL_ZERO_SHOT_SIZE_M = 200
FINAL_ZERO_SHOT_DENSITY = 0.045
FINAL_ZERO_SHOT_HAZARD_SEEDS = (970_001, 970_002, 970_003, 970_004,
                                970_005, 970_006)
FINAL_ZERO_SHOT_CROWD_SEEDS = (971_001, 971_002)
FINAL_ZERO_SHOT_ROBOT_COUNTS = (1, 2, 3)
FINAL_ZERO_SHOT_CONTROL = "off_zero_command"
# Hazards are drawn per seed with these shapes and areas. "street" is not
# offered: on an OSM crop it falls back to an axis-free rectangle rather than
# following the street network, and would be mislabelled.
FINAL_ZERO_SHOT_HAZARD_SHAPES = ("circle", "rect")
FINAL_ZERO_SHOT_HAZARD_AREA_RANGE = (0.04, 0.20)
# A drawn hazard is rejected, and the rejection recorded, when it holds less
# walkable ground than this, when part of its walkable ground has no walking
# route to safety, or when the density it would run at is outside the band.
FINAL_ZERO_SHOT_MIN_WALKABLE_M2 = 400.0
FINAL_ZERO_SHOT_DENSITY_BAND = (0.035, 0.055)

# --------------- LOGGING ---------------
# One structured event per episode and per logging interval of updates, from
# the main process only. JSONL is the source of record; the TXT files are kept
# for older analysis scripts; TensorBoard nd W&B receive the same values on the
# same axes (global_episode for episode metrics, global_update for learner
# metrics).
EXPERIMENT_ID = "outdoor-madrl-v2-easy-fullinfo"
LOG_TRAIN_EVERY_UPDATES = 200
LOG_TXT_COMPAT = True

# Episode videos. Every VIDEO_EVERY_EPISODES episodes one worker records its
# next episode as a time-lapse mp4 (LOG_DIR/videos) and the main process logs
# it to W&B as episode/video. VIDEO_SPEEDUP is how many times faster than
# simulated time the clip plays: a frame is drawn every
# VIDEO_SPEEDUP / (VIDEO_FPS * AGENT_TIME_STEP) steps, so 40x at 20 fps is one
# frame every 4 steps and a 2,000-step episode becomes a 25 s clip. Recording
# costs that worker a few tens of seconds per clip. 0 turns it off.
VIDEO_EVERY_EPISODES = 200
VIDEO_SPEEDUP = 40
VIDEO_FPS = 20
VIDEO_DPI = 100

# Weights & Biases. "online" syncs to wandb.ai, "offline" writes locally for a
# later `wandb sync`, "disabled" turns it off. A W&B failure never stops
# training: local logs continue and the failure is reported once, loudly.
WANDB_MODE = "online"               # "online" | "offline" | "disabled"
WANDB_PROJECT = "adds-sac-ued6-261011ENV"
WANDB_ENTITY = None                 # None = the account's default entity
WANDB_GROUP = None                  # None = EXPERIMENT_ID
# The run's name in the W&B run list. None builds one from the experiment,
# the training mode and map sizes, and the start time, e.g.
# "outdoor-madrl-v2-dataset-100m-0924-1530". A run started from a checkpoint
# as a new run (see the resume rule in learn/metrics_logger.py) gets
# "-resume<episode>" appended either way; a run that carries on keeps its name.
WANDB_RUN_NAME = "Reward_Progress150_UPT2_asy"  # None = auto
# Only models chosen on validation are uploaded, never the replay buffer or
# every periodic checkpoint.
WANDB_UPLOAD_CHECKPOINTS = "selected"   # "none" | "selected"
