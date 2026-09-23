import os

# ------------- BASIC PARAMETERS ------------------
LR = 1e-4
BUFFER_SIZE = 1000000
BATCH_SIZE = 128
INTRINSIC_ETA = 0.1              # intrinsic reward
START_BATCH_TIMES = 1
START_UPDATE_EPISODE = 500
DEVICE = "cuda"

GAMMA_START = 0.99
GAMMA_END = 0.99
GAMMA_SCHEDULE_STEP = 1000
MAP_H = 100                      # fallback world size; levels carry their own
MAP_W = 100

WD_Q = 3e-4                      # weight decay
WD_PI = 0.0

# ---------------- SIMULATION ENVIRONMENT ---------------------
# Crowd size comes from CROWD_DENSITY_RANGE, not from these. They survive for
# the numbered maps and ADDS_AS_HumanPlay, which build the simulator before
# the map is loaded and so have no walkable area to scale from.
CROWD_NUMBER_MIN = 30
CROWD_NUMBER_MAX = 30

MAP_NUM = -1                     # -2 unused, -1 random from MAP_NUM_RANDOM
MAP_NUM_RANDOM = list(range(1000, 1300))
# Smaller training subsets used in earlier runs, kept to reproduce them:
# MAP_NUM_RANDOM = list(range(1001, 1299, 3))  # 100 maps
# MAP_NUM_RANDOM = [1006, 1018, 1031, 1047, 1059, 1072, 1085, 1093, 1108, 1124, 1136, 1149, 1162, 1175, 1188, 1201, 1213, 1227, 1239, 1254, 1266, 1273, 1281, 1287, 1289, 1293, 1296, 1298, 1299, 1300]  # 30 maps
# MAP_NUM_RANDOM = [1014, 1042, 1068, 1095, 1127, 1153, 1186, 1219, 1264, 1298]  # 10 maps

# One random geometric symmetry per created map. Off for evaluation, which
# needs the original orientation.
MAP_DATA_AUGMENTATION = True
MAP_AUGMENTATION_TRANSFORMS = (
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "reflect",
    "reflect_rotate_90",
    "reflect_rotate_180",
    "reflect_rotate_270",
)

SCALE_CHECK = 0                  # want to check reward scale?
ACTION_SCALE = 4
# Steps per episode. 2000, down from 6000: clearing takes tens to hundreds of
# steps, so the rest was the robots holding an already-empty zone.
MAX_STEPS = 2000

ROBOT_BODY_RADIUS = 1            # m
# Pedestrian body radius. 0.25 m is a shoulder width of about half a metre,
# which is what the measurements the fundamental diagram and the bottleneck
# tests are held against were taken on. It was 0.5, and that was the single
# most damaging number in the model: a disc a metre across cannot pass a
# 0.8 m door at all and packs at 1.15 per square metre, so the whole range
# where crowd dynamics happen was outside the model's geometry. See
# docs/crowd_validation.md.
AGENT_BODY_RADIUS = 0.25         # m
ROBOT_VISION = 10                # m
AGENT_VISION = 10                # m
EXIT_CONFIRM_RADIUS = 10
EXIT_CONFIRM_RADIUS_BONUS = 0
AGENT_SPEED_MEAN = 1.5           # m/s
ROBOT_SPEED_MAX = 2              # m/s
AGENT_TIME_STEP = 0.5
ROBOT_TIME_STEP = 0.5

# --------------- SOCIAL FORCE CONSTANTS ---------------
# The pedestrian force model, calibrated against the standard test cases in
# validation/. Every one of these used to be a literal inside agent_modeling,
# where it could not be swept or even found; docs/crowd_validation.md records
# what each was before and what it is held to now.

# Exponential repulsion between pedestrians: SF_K_AGENT * exp((r_sum - d) /
# SF_LAMBDA_A), chosen by sweeping both against Weidmann's speed-density
# curve; see docs/crowd_validation.md for the table. The strength came down
# from 200 because the body radius halved, and the range stayed at contact
# scale: what had removed the speed-density coupling was not the range but a
# visibility bug that stopped a fifth of a dense crowd from seeing anyone.
SF_K_AGENT = 40.0
SF_LAMBDA_A = 0.3

# Front weighting. A pedestrian responds to what is in front of it far more
# than to what is behind, and without that a uniform queue is symmetric: the
# person ahead pushes back exactly as hard as the person behind pushes
# forward, the net force is zero, and walking speed is independent of density
# whatever the other constants are. This is why the model had no fundamental
# diagram. The weight runs from SF_ANISOTROPY directly behind to 1.0 directly
# ahead, following the anisotropic form in Helbing and Johansson's social
# force papers.
SF_ANISOTROPY = 0.2

# Contact, once two bodies actually overlap: normal spring, normal damping,
# tangential friction.
SF_KN = 0.8e5
SF_CN = 1000.0
SF_MU_T = 2.5e5

# How close a pedestrian will walk to a wall before being pushed off it, over
# and above its own body radius. It was a flat 1 m, which meant a standoff of
# 1.5 m from every facade: a three metre doorway left half a metre of usable
# width down the middle and everyone went through in single file. People walk
# within a hand's breadth of a wall.
SF_WALL_MARGIN_M = 0.15

# Wall contact, the same three terms against a facade. These apply only where
# the body actually overlaps the wall, not across the whole standoff: the
# tangential term used to brake anyone walking within 1.5 m of a facade at the
# acceleration limit, every step, which is why nobody could get through a
# doorway. A person walking down a pavement is not being braked by the wall
# beside them.
SF_WALL_KN = 1.8e5
SF_WALL_CN = 1000.0
SF_WALL_MU_T = 2.5e5

# The soft push that keeps a body off a wall it is not touching, in newtons
# per metre of encroachment into SF_WALL_MARGIN_M. Small next to the drive
# force, which is about 100 N: enough to keep pedestrians off facades, not
# enough to stop them using a doorway.
SF_WALL_SOFT_K = 300.0

# How far ahead the wall-aware heading looks when deciding to slide along a
# facade rather than walk into it.
SF_WALL_LOOKAHEAD_M = 1.5

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
RANDOM_EXIT = False

# --------------- REAL-MAP EXTRACTION (osm_corpus) ---------------
# How a real crop's traversable space is decided. "roads" builds it from the
# road and footway network widened to its carriageway, plus mapped pedestrian
# areas, which is what a robot can drive on. "buildings" treats only footprints
# as obstacles, which overstates it: the gaps are largely private plots, yards,
# car parks, planting, water and rail.
OSM_TRAVERSABILITY = "roads"     # "roads" | "buildings"

# Where downloaded OpenStreetMap data is cached. Outside the project because
# it is a cache: the regional extracts run to about 12.5 GiB, against under
# 20 MiB for the project itself. Nothing downstream needs it once the corpus
# index and the exported levels are built; it is needed again only to re-cut
# crops. Set ADDS_OSM_CACHE to move it.
OSM_CACHE_DIR = os.environ.get(
    "ADDS_OSM_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "adds-osm"))

# How far block outlines may move when simplified, in metres. Buffered road
# edges arrive as dense arcs and navmesh cost follows vertex count, so this is
# the main lever on simulation speed. Bounded by what a robot needs: move a
# wall much further than its own width and a passable corridor closes.
OSM_SIMPLIFY_M = 1.5

# Whether unbuilt ground beside the streets counts as walkable. Off: measured
# and rejected, because it hollows out real footprints where the survey is
# poor, which is where it was aimed. See reachable_open_ground in
# osm_corpus/roads.py; sparsely mapped crops are dropped instead.
OSM_OPEN_GROUND = False

# Reject a crop when this much of what the road network calls blocked holds no
# building, which measures how much of the place OpenStreetMap does not record
# rather than how the place is built. Chosen from the corpus distribution; see
# the note beside the value in osm_corpus/extract.py.
OSM_MAX_UNBUILT_BLOCKED = 0.70

# --------------- NAVMESH RESOLUTION ---------------
# Intermediate points are inserted along long obstacle and boundary edges every
# this many metres before triangulation, so routing has triangles to work with.
# Raising it coarsens paths and cuts construction cost.
NAVMESH_SEGMENT_STEP_M = 20.0

# --------------- SIMULATION VIEWER (run_sim.py) ---------------
# Which map the viewer shows. The command line overrides these.
#   "real"        a real downtown crop with a hazard placed on it, and the
#                 OpenStreetMap tiles beside it. Default, because whether the
#                 extraction is a fair reading of that map is only visible
#                 with the two side by side.
#   "curriculum"  a generated level with a hazard: exactly what training runs.
#   "numbered"    the hand-made maps in map_infos/, which still carry exits
#                 and no hazard, so they show the task that was replaced.
SIM_SOURCE = "real"              # "real" | "curriculum" | "numbered"

# For "curriculum". Difficulty drives both how developed the fabric is and how
# much of the crop the hazard covers.
SIM_DIFFICULTY = 5
SIM_SIZE = 140
SIM_MORPHOLOGY = None            # None = drawn at random from the seven
SIM_ROBOTS = 2
SIM_SEED = None                  # None = a new level every run

# For "real". `python3 ADDS_AS_osm_pipeline.py status` lists what is exported.
SIM_REAL_SITE = "shibuya"
SIM_REAL_SIZE = 400              # 100, 200 or 400 m
#
# The 26 usable sites and the morphology tag each carries. The tag is what the
# generator's per-morphology tables were fitted to, so it is also how a
# generated level should be held against a real one. Why each downtown was
# chosen is in osm_corpus/sites.py.
#
#   Asia            shibuya Tokyo lowrise_dense, dotonbori Osaka lowrise_dense,
#                   myeongdong Seoul lowrise_dense, hongdae Seoul lowrise_dense,
#                   gangnam_superblock Seoul superblock,
#                   nanjing_road Shanghai grid, khao_san Bangkok lowrise_dense
#   Europe          covent_garden London organic, kreuzberg Berlin grid,
#                   mitte Berlin grid, pigalle Paris boulevard,
#                   eixample Barcelona grid, gracia Barcelona organic,
#                   la_latina Madrid organic, trastevere Rome organic,
#                   grachtengordel Amsterdam lowrise_dense,
#                   sultanahmet Istanbul organic
#   North America   times_square New York grid, soho_nyc New York grid,
#                   french_quarter New Orleans colonial_grid,
#                   gastown Vancouver grid,
#                   centro_historico_cdmx Mexico City colonial_grid
#   Africa          marrakesh_medina Marrakesh medina,
#                   maboneng Johannesburg grid
#   South America   vila_madalena Sao Paulo lowrise_dense
#
# Three more sites (chandni_chowk, khan_el_khalili, palermo_soho) are listed in
# sites.py but DROPPED: OpenStreetMap does not record enough of them for a road
# network to describe the place, so every crop is rejected. Putting one here
# will fail. Every site has 100, 200 and 400 m crops except surry_hills, whose
# 400 m crop hit a shapely topology error.
#
# Morphology counts: grid 8, lowrise_dense 8, organic 5, colonial_grid 2,
# medina 1, boulevard 1, superblock 1. Medina now rests on Marrakesh alone, so
# its generator table, like boulevard's and superblock's, is plausible rather
# than measured.

# Space left under the basemap thumbnail in the side panel, in pixels.
BASEMAP_PANEL_BOTTOM_MARGIN = 120

# --------------- HUMAN PLAY (ADDS_AS_HumanPlay.py) ---------------
# The human-play script had its own copy of all of this, including its own
# step limit of 3000 against MAX_STEPS and its own hardcoded map id, so it
# ran a different scenario from the viewer and from training: a numbered map
# with boundary exits and no hazard, which is the task this project replaced.
# It now takes the level from SIM_SOURCE like everything else, and what is
# left here is the part that is genuinely about playing rather than about the
# task.
HUMANPLAY_EPISODES = 3
HUMANPLAY_SCREEN = (1000, 1000)
HUMANPLAY_PANEL_WIDTH = 200
HUMANPLAY_SIM_FPS = 10
HUMANPLAY_RENDER_FPS = 30
HUMANPLAY_EXP_NAME = "humanplay"

# How the episode is recorded. "live" draws only; the others also collect
# frames and write them out at the end.
HUMANPLAY_VIS_MODE = "mp4"          # "live" | "mp4" | "png_every" | "png_last"
HUMANPLAY_SAVE_EVERY = 1
HUMANPLAY_MP4_FPS = 20
HUMANPLAY_DPI = 200
HUMANPLAY_BITRATE = 8000

# -------------- PATH -------------------
# Distinct from the other copies so runs can share a machine without
# clobbering each other's logs, checkpoints or tensorboard port.
LOG_DIR = "Log_SAC_UED3"
PORT_NUM = 7757

# --------------- SAC ALGORITHM PARAMETER ---------------
LOG_STD_MAX = 0.5
LOG_STD_MIN = -20
ALPHA_START = 0.2
ALPHA_END = 0.2
ALPHA_DECAY_STEPS = 3000

# --------------- REWARD SHAPING -----------------
REWARD_A = 2                     # reward_based_alived
REWARD_B = 0.003                 # reward_based_all_agents_danger
REWARD_D = 2                     # reward_based_penalty
REWARD_K = 6                     # reward_penalty_collsion
REWARD_J = 0                     # reward_based_all_agents_danger_root
REWARD_L = 1                     # reward_based_farthest_agent_distance
REWARD_FIXED = -0.5

REWARD_I = 0                     # reward_based_alived_root
REWARD_C = 0                     # reward_based_gain
REWARD_E = 0                     # reward_based_evacuated_with_robot
REWARD_F = 0                     # reward_based_distance_from_near_agents
REWARD_G = 0                     # reward_based_distance_from_near_agent_gain
REWARD_H = 0                     # reward_based_gain_with_time_bonus
# Re-entry penalty. Off by default, so the baseline is the occupancy term
# alone and this can be measured as an ablation.
REWARD_N = 0                     # reward_based_reentry
FINISHED_BONUS = 0

USING_TRAINED_MODEL = True
SHOW_CONTROLLED_CROWD = False
ENABLE_TIMER = True

N_ENVS = 4
UPDATES_PER_TRANSITION = 1
POLICY_BROADCAST_INTERVAL = 10

EGO_MAP_SIZE = 25
DOWNSAMPLE_MAP_SIZE = 50
FiLM_USE = True
EGO_USE = True

# --------------- ZERO-SHOT EVALUATION ---------------
# Held-out JSON maps, outside the 1000-1299 range the DR baseline trains on.
# Never add one of these to MAP_NUM_RANDOM.
ZSG_MAP = [1500, 1501, 1502, 1503]
ZSG_CYCLE_EPISODE = 500
ZSG_ITERATION = 10
ZSG_ROBOT_NUM = [1]

# The procedural half of the held-out set, from fixed seeds in ued/holdout.py:
# four size bands x six difficulties x three seeds, so each pass is that many
# levels times ZSG_HOLDOUT_ITERATION episodes.
ZSG_HOLDOUT_LEVELS = True
ZSG_HOLDOUT_ITERATION = 2

# --------------- UNSUPERVISED ENVIRONMENT DESIGN -----------------
# ACCEL-style evolutionary curriculum: keep a population of levels, score them,
# and breed from the ones the agent is on the edge of solving. The plan
# document says why the minimax-regret framing is not adopted.
UED_ENABLED = True
UED_METHOD = "accel"             # "dr" (uniform over generated levels) | "accel"

# "learnability" is p*(1-p) over a level's success history and needs several
# trials; "maxmc" scores a single episode from the SAC critic; "hybrid" uses
# maxmc until UED_MIN_TRIALS trials exist, covering a fresh child's cold start.
UED_SCORE = "hybrid"             # "learnability" | "maxmc" | "hybrid"
UED_MIN_TRIALS = 5               # fewer cannot separate a frontier level from noise
UED_TRIAL_HISTORY = 10           # sliding window used for the success rate

# Robust-PLR's replay-only rule cannot hold exactly against a shared off-policy
# buffer. An experiment axis rather than a silent choice: when True,
# transitions from fresh levels are scored but not stored.
UED_REPLAY_ONLY_UPDATES = False

# Sized by how many episodes a level needs before its score means anything.
# Population size, UED_TRIAL_HISTORY and UED_TRIAL_MAX_AGE have to stay
# consistent or no level ever reaches the breeding bar; CurriculumTimingTest
# holds that invariant.
UED_POP_SIZE = 200

# Children start on their parent's score, so without a cap one lucky ancestor
# crowds the population with its own lineage.
UED_MAX_CHILDREN_PER_LEVEL = 3
UED_P_NEW_START = 0.5            # probability of a fresh level early on
UED_P_NEW_END = 0.1              # ... once the population has filled out
UED_P_NEW_DECAY_EPISODES = 5000
UED_STALENESS_COEF = 0.2         # rho: weight on the staleness distribution
UED_SCORE_TEMPERATURE = 1.0      # rank-based sampling temperature

# Breed only from the top quarter by score percentile. Half turned the
# population over faster than levels could accumulate trials.
UED_MUTATE_THRESHOLD = 0.75
UED_MUTATIONS_PER_CHILD = (1, 3)
UED_MUTATE_MAX_TRIES = 40

# A level counts as solved when everyone is out within this multiple of its own
# free-flow estimate. Normalised per level, because a hand-picked constant
# makes p saturate and takes learnability with it. Calibrate from the logged
# UED/mean_evac_ratio; raw times are stored per level, so a new K can be
# applied to the existing history.
UED_SUCCESS_K = 5.0

# Reference world size for the scale scalars in the robot state, as
# log(width / reference): zero at the reference, symmetric, and bounded.
MAP_SIZE_REFERENCE = 100.0

# Specific flow through an exit, persons per metre of width per second.
EVAC_SPECIFIC_FLOW = 1.2

UED_DIFFICULTY_RANGE = (0, 6)    # 0 is an empty field, 6 real downtown density

# ---------------- CROWD DENSITY ----------------
# Crowd size is a density in persons per square metre of walkable ground, and
# every path that builds a level derives its headcount from it. 0.05 is the
# reference, one person per 20 m2, well inside free flow; the band is 0.8 to
# 1.0 of it. CROWD_SIZE_LIMIT is a simulation-cost clamp, not a modelling
# claim: a clamped level records the lower density it actually ran at. See
# docs/crowd_density.md and crowd_density.py.
CROWD_DENSITY_RANGE = (0.04, 0.05)
CROWD_SIZE_LIMIT = (10, 5000)
# Measured episode cost at the 200 m Soho crop, 2000 steps, no rendering:
# 800 pedestrians is 263 ms per step and 8.8 minutes. Cost grows faster than
# linearly, so the current ceiling of 5000 is hours per episode rather than
# minutes and is a ceiling for evaluation, not for training.

# --------------- DANGER ZONE (the task the robots are given) ---------------
# The crowd must leave a hazard region and be kept from wandering back in.
# This replaced the boundary exits, and it is a different problem: safety is
# everywhere except one region, so the crowd disperses instead of converging.
# The zone is static within an episode; a spreading hazard has no stable
# free-flow reference to normalise the success criterion by.
#
# Size as a share of crop area rather than a radius, because that is what
# compares across UED_MAP_SIZE_RANGE.
UED_DANGER_AREA_RANGE = (0.04, 0.30)

# A circle escapes uniformly; a rectangle is a stretch of street, where
# leaving sideways is quick and lengthwise is not. "street" lies along one of
# the plan's own streets, at its width and angle, which is the shape most real
# hazards in a city take.
UED_DANGER_SHAPES = ("circle", "rect", "street")

# How far past the boundary counts as safe. Without a margin one social-force
# jostle puts a pedestrian back inside and the episode never finishes.
DANGER_SAFE_MARGIN_M = 2.0

# Share of pedestrians starting inside the zone. The rest are what the robots
# have to block; an all-inside crowd would never exercise the inflow half.
UED_DANGER_INSIDE_FRACTION = (0.5, 1.0)

# --------------- CROWD HAZARD AWARENESS ---------------
# Design and justification: docs/crowd_awareness_design.md and
# docs/crowd_awareness_implementation.md.
#
# How detectable the hazard is, 0 to 1: a property of the hazard, so a
# curriculum variable. Near 1 is a fire, near 0 a gas leak, where the only
# channels are word of mouth and the robots. It sets both how fast direct
# perception fires and whether perceiving gives a global picture or only the
# spot touched. Watch for degeneration: the curriculum can raise difficulty by
# starving the crowd of information instead of by geometry, so if the logged
# distribution piles up at the floor, narrow this or close the axis.
UED_DANGER_PERCEPTIBILITY = (0.05, 0.95)

# Share of the crowd already cued at the start: an alarm, the news, someone
# off-map. Zero is a hazard nobody has announced.
UED_PRIOR_INFORMED_FRACTION = (0.0, 0.3)

# Above this, perceiving also conveys a coarse sense of the hazard as a whole,
# the way a smoke plume does. Below it, only the spots personally sensed.
PERCEPTIBILITY_GLOBAL_CUE = 0.7

# Below this the hazard emits no sensory cue at all. A threshold rather than a
# small probability, because an unodorised leak gives no chance of being
# smelled, and a low per-step rate becomes near-certain over 2000 steps, which
# collapsed the whole low end of the axis.
PERCEPTIBILITY_SENSORY_FLOOR = 0.25

# Per-step probability of noticing by direct perception at perceptibility 1.0,
# scaled by how far perceptibility sits above the floor. Standing in smoke is
# not the same as seeing it across a street.
AWARENESS_P_INSIDE = 0.35
AWARENESS_P_VISIBLE = 0.06

# Per-step probability of being cued by one neighbour who is already acting.
# Saturates with more: ten people running past is not ten times one.
AWARENESS_P_SOCIAL = 0.12
AWARENESS_SOCIAL_SATURATION = 4.0

# Pre-movement delay in steps, lognormal, the shape fire-safety engineering
# reports for recognition and response. The tail matters more than the median,
# because the last occupants set the evacuation time.
PREMOVEMENT_MEDIAN_STEPS = 40.0
PREMOVEMENT_SIGMA = 0.7

# Milling shortens when others are visibly acting and when a robot is present:
# both reduce the need to seek confirmation.
MILLING_SOCIAL_SPEEDUP = 0.5
MILLING_ROBOT_SPEEDUP = 0.25

# How far a remembered sighting pushes a pedestrian away. Avoidance is from
# memory, not from the true zone, so a pedestrian can round a block and walk
# into a face of the same hazard it has never seen.
HAZARD_MEMORY_RADIUS_M = 12.0
HAZARD_MEMORY_MAX_POINTS = 12

# How long the zone must stay empty to count as cleared. Nobody is removed
# here, so without a hold a straggler stepping over the line for one step
# would record a clearing time that describes a jostle. It also makes the
# inflow half count: a policy that sweeps and stops would never be scored.
DANGER_CLEAR_HOLD_STEPS = 60

# When an episode ends. The clearing time is recorded the same way under all
# three, at the step the zone first emptied; only `done` differs.
#   "none"     never early. The robots go on holding the zone against
#              re-entry, which is the half an early finish never exercises.
#   "cleared"  ends once the zone has been empty for DANGER_CLEAR_HOLD_STEPS.
#              Cheaper, and isolates the dispersal half.
#   "defend"   ends DANGER_DEFEND_STEPS after the zone first emptied. Both
#              halves are trained and the tail is bounded.
DANGER_TERMINATION = "none"      # "none" | "cleared" | "defend"

# The defence window. Measured from the first clearing and never restarted, so
# a policy cannot extend its own episode by letting people back in.
DANGER_DEFEND_STEPS = 400

# --------------- CROWD FLOW ACROSS THE MAP EDGE ---------------
# Whether the crop is a closed box or a piece of a larger city. Inflow brings
# newcomers who know nothing about the hazard. Outflow lets a pedestrian that
# reaches the edge leave; without it the boundary is a wall the crowd piles up
# against, which is an artefact of the crop.
CROWD_ALLOW_INFLOW = False
CROWD_ALLOW_OUTFLOW = True

# Pedestrians entering per 100 steps when inflow is on. A rate, so it means
# the same thing at any MAX_STEPS.
CROWD_INFLOW_PER_100_STEPS = 4.0

# --------------- WHERE PEOPLE ARE GOING BEFORE ANYTHING HAPPENS ---------------
# A pedestrian who does not know about the hazard is not wandering: it is on
# its way somewhere. Trips run between street mouths at the crop edge, because
# a downtown crop is a piece of a larger city and most people crossing it are
# passing through. The rest are local errands to a point inside.
#
# This replaced a uniformly random walkable triangle as the destination, which
# had no basis and mattered more than it looks: where people are when the
# hazard starts is what sets the difficulty, and that distribution was an
# artefact of the navmesh. See docs/crowd_od.md.
CROWD_THROUGH_TRIP_SHARE = 0.7

# Street mouths are found by clustering the walkable triangles along the crop
# edge; a gap wider than this starts a new one. Roughly a street width, so two
# separate streets do not merge into one mouth.
CROWD_GATE_GAP_M = 6.0

# A destination nearer than this is not a trip. Without it a pedestrian
# standing beside a street mouth picks that mouth and arrives immediately.
CROWD_MIN_TRIP_M = 25.0

# How long a pedestrian stays at a destination before setting off again, in
# steps. Drawn uniformly. A dwelling pedestrian holds its position and is
# exempt from the wedge release, which exists for pedestrians that cannot
# move rather than for ones that are not trying to.
CROWD_DWELL_STEPS = (10, 60)

# A through trip ends at the crop edge, but the pedestrian only leaves when
# the crop is an open system, which needs both flow flags. With outflow alone
# the crowd would drain out of a closed box over the episode and the task
# would solve itself.

# --------------- WHAT A ROBOT SIGNALS ---------------
# The robot's action has two parts: where it moves, and what it signals. The
# signal is what the crowd responds to, and it has three settings.
#
#   "off"     signals nothing. The crowd feels the robot only as a body in
#             the way. This is the control condition: without it there is no
#             way to separate what the guidance channel contributes from what
#             a moving obstacle contributes, and it is also the honest
#             description of a robot that is repositioning rather than
#             working.
#   "guide"   "follow me". The crowd heads for the robot's own position, so
#             the robot leads and its path is the instruction. This is the
#             mechanism the robot-guided evacuation studies use, and the one
#             human-robot trust experiments measured compliance for.
#   "direct"  "go that way". The robot signals a heading and the crowd takes
#             that heading rather than converging on the robot. Closer to
#             dynamic signage than to a leader, and it is what lets one robot
#             turn a flow without standing in it.
#
# A mode that commanded people to reverse was considered and dropped. Turning
# a crowd back is done in practice by presence and by barriers, not by an
# instruction, and an instruction that contradicts what somebody can see is
# exactly the kind the protective-action literature says people stop to
# question. "direct" covers the same ground without that claim: a heading
# away from the hazard is a redirection, not a reversal.
ROBOT_MODES = ("off", "guide", "direct")

# How far the signal carries, in metres.
#
# Not arbitrary: emergency signage standards express legibility as a multiple
# of the sign's height, a factor of about 100 for an externally illuminated
# sign and 200 for an internally illuminated one. Ten metres is therefore a
# 0.1 m illuminated pictogram, which is what fits on a robot of this size. A
# loudspeaker would carry further and would not be blocked by buildings, so
# the channel has to be stated before the number means anything.
ROBOT_SIGNAL_RADIUS_M = 10.0

# Whether the signal needs line of sight. True for a visual signal, and the
# visibility atlas already answers it. This was inconsistent before: the
# compliance path went through the pedestrian's field of view while the
# hazard cue from a robot was a plain distance test, so a robot behind a block
# told people about a hazard it could not be seen to be near.
ROBOT_SIGNAL_REQUIRES_SIGHT = True

# How far ahead of itself a directed pedestrian puts its goal, in metres.
# Far enough that the drive force points along the signalled heading rather
# than at a spot it reaches immediately.
ROBOT_DIRECT_GOAL_M = 12.0

# --------------- HOW READILY PEOPLE FOLLOW A ROBOT ---------------
# Per-pedestrian compliance, drawn once at spawn like mass and desired speed,
# and applied each time the pedestrian decides whether to follow the nearest
# robot it can see.
#
# It was 1.0, which made every pedestrian that saw a robot follow it. That is
# the single assumption this project's conclusion rests on most heavily, and
# nothing supports it: studies of guidance compliance report high but not
# unit rates, in both directions. So it is a range, and a pedestrian
# parameter rather than a curriculum variable, which keeps it in the fixed
# deployment distribution the CICS argument protects.
ROBOT_COMPLIANCE_RANGE = (0.5, 1.0)

# Per-mode multipliers on that trait. Following a leader and reading a sign
# are not the same act: the trust experiments that report near-unit compliance
# measured people following a robot, while signage studies report that a
# sign changes exit choice for a substantial minority rather than for
# everybody. So "direct" is discounted.
#
# The size of the discount has no measurement behind it. It belongs in the
# sensitivity analysis, not in a claim.
ROBOT_GUIDE_COMPLIANCE = 1.0
ROBOT_DIRECT_COMPLIANCE = 0.8

# The band along the edge where leaving is possible. Wider than it looks it
# should be: wall repulsion pushes a pedestrian off the edge, so a one-metre
# band almost never fires.
CROWD_OUTFLOW_MARGIN_M = 3.0

# Stuck in that band for this long means nowhere left to go, and the
# pedestrian leaves. This is the case outflow exists for: somebody pressed
# toward the edge with a wall behind and a robot in front, who would otherwise
# stand there for the rest of the episode and, if that spot is inside the
# hazard, could never be cleared.
CROWD_OUTFLOW_STUCK_STEPS = 40
CROWD_OUTFLOW_STUCK_MOVE_M = 0.5

# How the hazard is drawn in the viewer and the curriculum snapshots.
# Translucent, because the question asked of these pictures is usually who is
# still inside.
DANGER_FILL_COLOR = "#d62728"
DANGER_FILL_ALPHA = 0.22
DANGER_EDGE_COLOR = "#b00020"
DANGER_ZORDER = 1.5

# Grey level in the observation raster: between the crowd at 150 and a robot
# at 255, clear of the 50 shared by walls and ego-crop padding.
DANGER_PIXEL_VALUE = 200

# --------------- MULTI-ROBOT ---------------
# Clearing a region and holding it are two jobs at once, which is why the task
# moved to a team: with one robot, doing either leaves the other undone.
#
# MAX_ROBOTS fixes tensor width everywhere (joint observation, joint action,
# centralised critic), and a shorter team is carried by a mask. Raising it
# invalidates saved checkpoints.
MAX_ROBOTS = 3

# How many robots a fresh level gets, inclusive, capped by MAX_ROBOTS. A
# curriculum axis, so the mask is exercised rather than a formality.
UED_ROBOT_RANGE = (1, 3)

# Where the team starts. "outside" is the realistic posture: responders arrive
# from outside a danger area. "anywhere" exists so the choice can be measured.
ROBOT_START = "outside"          # "outside" | "anywhere"

# World size in metres per fresh level. The held-out set deliberately reaches
# outside this range so interpolation and extrapolation can be told apart.
UED_MAP_SIZE_RANGE = (70, 140)
# Square only for now: aspect ratio interacts with the D4 augmentation, which
# swaps width and height on the odd rotations.
UED_MAP_SQUARE = True

# The one generator family: citygen lays out a street network for one of the
# seven morphologies, then decides which blocks are built up. It replaced
# random_map and street_map, whose obstacle polygons the curriculum edited
# directly; those edits were morphology-blind, so a generated downtown stopped
# looking like one within a few generations. Both modules remain for the
# numbered maps and the GUI editors.
UED_GENERATOR = "citygen"

# Uniform, so the curriculum's own score decides where to spend episodes.
# Weight a morphology up only to study it deliberately.
UED_MORPHOLOGY_WEIGHTS = None    # None = uniform over citygen.MORPHOLOGIES

UED_WARMUP_EPISODES = 500        # pure DR until the population has some mass

# Level scores are recorded only once exploration has decayed below this. A
# success rate measured while actions are mostly random describes what a random
# walker solves, not where the policy's ability ends. Episodes still run and
# still train; only their outcomes are withheld from the curriculum.
UED_SCORE_MAX_EPSILON = 0.25

# Outcomes older than this many episodes are dropped from a level's history, so
# a score stays attached to a recognisable policy. A fixed trial count spanned
# more episodes than the whole epsilon decay.
UED_TRIAL_MAX_AGE = 3000

UED_LEVEL_QUEUE_SIZE = 4         # per-worker prefetch depth
UED_PRODUCER_TARGET = 64         # fresh levels kept ready on the producer thread
UED_SNAPSHOT_EPISODES = 200      # how often population pictures go to TensorBoard

# --------------- crowd evacuation parameter -----------------
K1 = 1                           # distance weight
K2 = 0.000001                    # density weight (unused: one exit only)
K3 = 1                           # width weight
