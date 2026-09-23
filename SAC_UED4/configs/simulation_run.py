"""One viewer or manual run: what to open and how to show it.

One of three configuration files; see configs/__init__.py. Opening a site in
the viewer is a run choice and lives here; excluding a site from training and
keeping it for the final zero-shot evaluation is a training choice and lives
in configs/training/. Nothing here may change what a policy observes.

Constants only; importing this module has no side effects.
"""

# --------------- SIMULATION VIEWER (run_sim.py) ---------------
# Which map the viewer shows. The command line overrides these.
#   "real"        a real downtown crop with a hazard placed on it, and the
#                 OpenStreetMap tiles beside it. Default, because whether the
#                 extraction is a fair reading of that map is only visible
#                 with the two side by side.
#   "curriculum"  a generated city level with a hazard: exactly what training
#                 runs.
SIM_SOURCE = "real"              # "real" | "curriculum"
# For "curriculum". Difficulty drives both how developed the fabric is and how
# much of the crop the hazard covers.
SIM_DIFFICULTY = 5
SIM_SIZE = 200
SIM_MORPHOLOGY = None            # None = drawn at random from the seven
SIM_ROBOTS = 2
SIM_SEED = None                  # None = a new level every run
# For "real". `python3 -m cli.ADDS_AS_osm_pipeline status` lists what is exported.
SIM_REAL_SITE = "shibuya"
# Must be one of SUPPORTED_CROP_SIZES_M. It was 150, which the corpus has
# never exported; the lookup then failed only when the viewer opened.
SIM_REAL_SIZE = 200              # 100, 200 or 400 m
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
# How much bigger than life bodies are drawn. 1.0 draws them at
# AGENT_BODY_RADIUS and ROBOT_BODY_RADIUS.
#
# The renderer used to carry its own radii, 0.35 for a pedestrian and 0.6 for
# a robot, with nothing tying them to the simulation. A pedestrian is 0.25 m
# and a robot 1.0 m, so the picture showed pedestrians half again too wide
# and the robot at 60 per cent of its size, which is exactly the thing a
# viewer is used to judge: whether a robot fills a street.
#
# Raise this only to make a small crop legible, and say so in any figure it
# appears in.
RENDER_BODY_SCALE = 1.0

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
USING_TRAINED_MODEL = True
SHOW_CONTROLLED_CROWD = False
# How the hazard is drawn in the viewer and the curriculum snapshots.
# Translucent, because the question asked of these pictures is usually who is
# still inside.
DANGER_FILL_COLOR = "#d62728"
DANGER_FILL_ALPHA = 0.22
DANGER_EDGE_COLOR = "#b00020"
DANGER_ZORDER = 1.5

# --------------- VIEWER RUN (viz/run_sim.py) ---------------
# How the robots are driven in the viewer: "RL" runs SIM_CHECKPOINT on every
# robot, "Human" leaves them to the keyboard script.
SIM_ROBOT_CONTROL = "RL"            # "RL" | "Human"
# Checkpoint the viewer loads, relative to the training LOG_DIR unless
# absolute. None starts an untrained policy, which is only useful to check the
# pipeline runs. A checkpoint whose schema versions differ from the current
# configuration is refused rather than run with the wrong inputs.
SIM_CHECKPOINT = None
# Seconds of simulated time the viewer treats as one step for its clock, and
# its frame budget. Display only.
SIM_VIEW_TIMESTEP = 0.25
SIM_RENDER_FPS = 20
SIM_SPEED_RANGE = (0.25, 48.0)
