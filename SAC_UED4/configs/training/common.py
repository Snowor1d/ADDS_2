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
# Discount per decision interval of ACTION_SCALE simulation steps. The
# learner converts it to a per-step factor gamma ** (1 / ACTION_SCALE), sums a
# held action's rewards with that factor and bootstraps with its k-th power,
# so a shortened last interval is discounted for the steps it actually lasted.
GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000
WD_Q = 3e-4                      # weight decay
WD_PI = 0.0
SCALE_CHECK = 0                  # want to check reward scale?

# --------------- EPSILON-EXPLORATION ------------------
EPSILON_MIN = 0
START_EPSILON = 1
SCHEDULER_TYPE = "l"
DECAY_VALUE = 0
LINEARLY_DECAY_STEP = 3000
START_DECAY_STEP = 500
EXPLORATION_TYPE = 0
LONG_EPSILON_MIN = 0
START_LONG_EPSILON = 0
DECAY_MODE = 'episode'

# -------------- PATH -------------------
# Distinct from the other copies so runs can share a machine without
# clobbering each other's logs, checkpoints or tensorboard port.
# It was "Log_SAC_UED3", the same folder a SAC_UED3 run on this machine
# writes to, so a SAC_UED4 run would have resumed from and overwritten it.
LOG_DIR = "Log_SAC_UED4_madrl"
PORT_NUM = 9000

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
LOG_STD_MIN = -20
ALPHA_START = 0.2
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

ENABLE_TIMER = True
N_ENVS = 12
# Gradient updates per stored decision instant. Measured on this machine
# (docs/replay_memory_budget.md): one update at batch 128 with three robots
# costs about 0.18 s to rebuild (overlapped with the previous update) plus
# 0.18 s on the GPU, while 12 workers on the generated 0.6/0.3/0.1 size mix produce
# roughly 11 decision instants a second. 1.0 made the learner the bottleneck
# and stalled the workers on a full queue.
UPDATES_PER_TRANSITION = 0.25
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

# --------------- TASK REWARD (rew-v2) ---------------
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
REWARD_VERSION = "rew-v2-person-time"
REWARD_W_PERSON_TIME = 1.0
REWARD_W_REMAINING_PATH = 1.0
REWARD_W_REENTRY = 2.0
REWARD_W_COLLISION = 0.05
REWARD_MIN_REFERENCE_POPULATION = 5
REWARD_MIN_REFERENCE_DISTANCE_M = 5.0

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

# Validation: generated levels at fixed seeds, per size and difficulty. Models
# are selected on these and nothing else. Evaluated in a separate process on a
# frozen copy of the policy, paired with the signal-off control on the same
# seeds; the control does not depend on the policy and is computed once.
VALIDATION_SIZES_M = (100, 200, 400)
VALIDATION_DIFFICULTIES = (2, 4, 6)
VALIDATION_SEEDS_PER_CELL = 1
VALIDATION_SEED_BASE = 950_000
VALIDATION_ROBOT_COUNTS = (1, 2, 3)
VALIDATION_CYCLE_EPISODE = 5000
# Whether training evaluates at all. False skips periodic validation entirely:
# no evaluation process is started, no validation levels are built, and no
# best_validation.pth is written (the routine checkpoints every 100 episodes
# are still saved, so a model can be validated afterwards). The final zero-shot
# never runs during training in either case; it is run by hand with
# cli/final_zero_shot.py.
PERIODIC_VALIDATION = False

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
# for older analysis scripts; TensorBoard and W&B receive the same values on the
# same axes (global_episode for episode metrics, global_update for learner
# metrics).
EXPERIMENT_ID = "outdoor-madrl-v2"
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
WANDB_PROJECT = "adds-sac-ued4"
WANDB_ENTITY = None                 # None = the account's default entity
WANDB_GROUP = None                  # None = EXPERIMENT_ID
# The run's name in the W&B run list. None builds one from the experiment,
# the training mode and map sizes, and the start time, e.g.
# "outdoor-madrl-v2-dataset-100m-0924-1530". A run started from a checkpoint
# as a new run (see the resume rule in learn/metrics_logger.py) gets
# "-resume<episode>" appended either way; a run that carries on keeps its name.
WANDB_RUN_NAME = "madrl-NotDirect"
# Only models chosen on validation are uploaded, never the replay buffer or
# every periodic checkpoint.
WANDB_UPLOAD_CHECKPOINTS = "selected"   # "none" | "selected"
