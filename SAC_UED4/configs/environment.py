"""Simulator physics, measurement and the shared observation contract.

One of three configuration files; see configs/__init__.py for which setting
belongs where. Everything here describes the world the robots act in and what
they can measure of it, so anything that changes an observation is part of a
checkpoint's observation schema (OBSERVATION_SCHEMA_VERSION below).

Constants only. Importing this module must not start processes, create files
or contact any service.
"""

import os

# ---------------- WORLD ---------------------
# World size used when a caller builds a simulator without naming one; every
# level carries its own.
MAP_H = 100
MAP_W = 100

ACTION_SCALE = 4
# Steps per episode. 2000, down from 6000: clearing takes tens to hundreds of
# steps, so the rest was the robots holding an already-empty zone.
MAX_STEPS = 2000
ROBOT_BODY_RADIUS = 0.5          # m; 1 m diameter
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
OSM_SIMPLIFY_M = 1.0
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

# --------------- OBSERVATION RASTERS ---------------
# Side of the local (ego) crop in cells. With OBS_EGO_RES_M = 1 it covers
# +-12.5 m, which contains the whole ROBOT_VISION disc.
EGO_MAP_SIZE = 25
# The legacy single-image global view (max-pooled grey raster). Only the viewer
# and the human-play script still draw it; the policy no longer reads it.
DOWNSAMPLE_MAP_SIZE = 50
# Reference world size for the scale scalars in the robot state, as
# log(width / reference): zero at the reference, symmetric, and bounded.
MAP_SIZE_REFERENCE = 100.0
# Specific flow through an exit, persons per metre of width per second.
EVAC_SPECIFIC_FLOW = 1.2

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
# How far past the boundary counts as safe. Without a margin one social-force
# jostle puts a pedestrian back inside and the episode never finishes.
DANGER_SAFE_MARGIN_M = 2.0
# Where the crowd starts.
#
# "uniform" spreads it over the walkable ground by area, so the share that
# begins inside the hazard is the share of the ground the hazard covers. That
# is the only placement consistent with CROWD_DENSITY_RANGE: the density is
# defined per square metre of walkable ground, and it has to hold locally as
# well as on average.
#
# "inside_fraction" is the older behaviour, kept so the difference can be
# measured rather than assumed. It puts UED_DANGER_INSIDE_FRACTION of the
# crowd inside the zone regardless of how big the zone is. Measured on five
# 120 m levels that produced 0.11 to 0.41 pedestrians per square metre inside
# the hazard against a target of 0.05, and about half the target outside it,
# so the crowd was in a crush loading in the one place the task is about.
#
# The reason it was written that way was that a small hazard holds few people
# and the dispersal half of the episode is then over quickly. That is a real
# effect, and the answer to it is the hazard's size, which the curriculum
# already varies through UED_DANGER_AREA_RANGE, rather than an initial
# condition no city ever has.
CROWD_SPAWN = "uniform"        # "uniform" | "inside_fraction"
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
# Outdoor road-flood VR experiment: safe-crowd mean 7.5 s and risky-crowd
# mean 13.8 s (NHESS 2026, doi:10.5194/nhess-26-981-2026).
# With existing lognormal sigma 0.7, 16.7 steps gives a draw mean of 10.67 s,
# the midpoint of those two conditions. This is a provisional VR anchor:
# model cue latency adds to the draw, and it is not a real-street estimate.
# See docs/outdoor_human_calibration.md for scope and sensitivity bounds.
PREMOVEMENT_MEDIAN_STEPS = 16.7
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
# How long the zone must stay empty to count as cleared. Some people can
# leave the crop and others can re-enter the danger zone, so a one-step empty
# observation does not establish sustained safety. Inflow also keeps the
# defence half of the task meaningful.
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
# The downtown crop is open. Entry and exit are independent physical flows;
# neither is a prerequisite for the other. Both default on so ordinary travel
# cannot be mistaken for a wall or a permanently shrinking closed population.
CROWD_ALLOW_INFLOW = True
CROWD_ALLOW_OUTFLOW = True
# Pedestrians entering per 100 steps when inflow is on. A rate, so it means
# the same thing at any MAX_STEPS.
CROWD_INFLOW_PER_100_STEPS = 4.0
# An informed newcomer has already heard a warning outside the crop. This is
# a scenario assumption, not a measured city-wide warning penetration rate.
CROWD_INFLOW_WARNED_FRACTION = 0.0
# After clearing the local hazard, people may leave the crop, pause at a safe
# location, or continue a trip around the remembered hazard. The weights are
# deliberately exposed for sensitivity sweeps: no cited field study identifies
# these probabilities for a 50-180 m pedestrian crop.
CROWD_POSTSAFE_INTENT_WEIGHTS = (0.50, 0.20, 0.30)  # depart, pause, continue
CROWD_SAFE_PAUSE_STEPS = (60, 240)
# A gate candidate whose straight-line approach passes close to a remembered
# danger sighting receives a penalty. This uses pedestrian memory, not the
# simulator's omniscient hazard polygon.
CROWD_GATE_MEMORY_AVOID_M = 12.0

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
# A through trip ends at a walkable crop-edge street mouth. Outflow is
# independent of inflow; a finite-cohort experiment with no inflow can drain
# naturally, so policy scores must be contrasted with the paired off control.

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
# Probability for one continuous encounter with one signalling robot, not
# a coin flip every step. Person, mode, scenario and robot form are separate.
# Model/robot instances may override the latter two for controlled comparisons.
# Use equal mode baselines for the zero-shot comparison: the old 0.8
# direct/guide ratio had no human-data calibration. The common 0.75 and
# person-factor range are also assumptions, not outdoor-calibrated estimates.
ROBOT_COMPLIANCE_PERSON_FACTOR_RANGE = (2.0 / 3.0, 4.0 / 3.0)
ROBOT_GUIDE_BASE_COMPLIANCE = 0.75
ROBOT_DIRECT_BASE_COMPLIANCE = 0.75
ROBOT_COMPLIANCE_SCENARIO_FACTOR = 1.0
ROBOT_COMPLIANCE_FORM_FACTOR = 1.0
# A momentary occlusion should not turn one instruction into many independent
# trials. This 5 s gap is a robustness setting, not measured human memory.
ROBOT_SIGNAL_REENCOUNTER_GAP_STEPS = 10
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
# Where the team starts. "outside" is the realistic posture: responders arrive
# from outside a danger area. "anywhere" exists so the choice can be measured.
ROBOT_START = "outside"          # "outside" | "anywhere"

# --------------- crowd evacuation parameter -----------------
K1 = 1                           # distance weight
K2 = 0.000001                    # density weight (unused: one exit only)
K3 = 1                           # width weight


# --------------- SCHEMA VERSIONS ---------------
# Bumped whenever the meaning of an observation or an action changes. A
# checkpoint, a replay file and a curriculum history all carry the versions
# they were produced under, and nothing is loaded across a mismatch: an old
# buffer read under a new meaning trains on the wrong thing without any error.
OBSERVATION_SCHEMA_VERSION = "obs-v2-multires-partial"
ACTION_SCHEMA_VERSION = "act-v1-move2-signal2-mode3"

# Crop sizes the real-map corpus exports and the observation contract is
# validated for. A viewer or evaluation setting outside this set is rejected at
# start-up rather than failing when the corpus lookup misses.
SUPPORTED_CROP_SIZES_M = (100, 200, 400)

# --------------- ROBOT MEASUREMENT AND TEAM COMMUNICATION ---------------
# A robot measures the crowd within ROBOT_VISION metres and in line of sight
# (the visibility atlas decides the second). What it has not measured is
# unobserved, never "zero people": every crowd channel travels with a mask.
#
# Team members exchange (position and mode, time, observed area, measured
# crowd). The first experiments assume a lossless, delay-free link; delay and
# loss are separate experiment axes, counted in decision intervals.
TEAM_SHARE_OBSERVATIONS = True
COMM_DELAY_DECISIONS = 0
COMM_DROP_PROB = 0.0

# Actor access to the simulator's true crowd map. False is the realistic
# default. True is a full-information upper bound only, and must run under its
# own checkpoint and experiment tag (the resolved config refuses to share a
# log directory between the two).
ACTOR_GLOBAL_CROWD_TRUTH = False

# --------------- MULTI-RESOLUTION OBSERVATION ---------------
# Three branches. The local and middle branches cover the same physical
# distance on every map; the global branch covers the whole map, so its cell
# size grows with the map and the map size also reaches the policy as a scalar.
#
#   local   EGO_MAP_SIZE^2 at OBS_EGO_RES_M: obstacles, hazard, off-map mask,
#           and the robot's own crowd count and observed mask for the last
#           OBS_HISTORY_DECISIONS decision instants.
#   middle  OBS_MID_SIZE^2 at OBS_MID_RES_M: walkable fraction, hazard
#           fraction and boundary, path distance to the hazard boundary,
#           off-map mask, and the team's recent crowd counts, observed mask
#           and observation age.
#   global  OBS_GLOBAL_SIZE^2 over the whole map: obstacle and hazard area
#           fractions, path distance to the walkable hazard boundary, and a
#           sparse summary of the team's observations. Never the true crowd.
#
# Cells are aggregated by what the channel means: area fractions for static
# layers and coverage, counts for the crowd. Max-pooling is not used; it mixed
# walls, people and hazard into one value.
OBS_EGO_RES_M = 1.0
OBS_MID_SIZE = 64
OBS_MID_RES_M = 2.0
OBS_GLOBAL_SIZE = 64
OBS_HISTORY_DECISIONS = 4

# Crowd channels are densities, clipped at this many persons per square metre.
OBS_DENSITY_SATURATION = 2.0

# Path-distance channels are metres divided by this, clipped to [0, 1].
OBS_PATH_SCALE_M = 100.0

# Teammate offsets in the scalar state are metres divided by this.
OBS_TEAM_DISTANCE_SCALE_M = 100.0
