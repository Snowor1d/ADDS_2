"""The task reward, rew-v2 (docs/outdoor_madrl_redesign.md section 5.3).

Per simulation step:

    person_time     persons inside the hazard, in person-seconds
    remaining_path  those persons' remaining walk to safety, as a fraction of
                    the episode's distance scale, also in person-seconds
    reentry         entries into the hazard by persons still in the crop who
                    had been clear of it by DANGER_SAFE_MARGIN_M; leaving the
                    crop is never an entry
    collision       robots that hit a wall this step, summed over the team

The first three are divided by a reference population fixed when the episode
starts, and the path term by a distance scale fixed at the same time, so the
same weight means about the same thing on a 100 m and a 400 m map. A headcount
that grows with inflow is never a denominator.

Leaving the crop earns nothing by itself. Somebody who walks out of the
hazard stops adding person-time, which is the credit; somebody outside the
hazard who finishes a trip and leaves changes no term at all.

rew-v6-pbrs adds potential-based shaping for learning only (Ng, Harada and
Russell 1999): every step it adds F = g * Phi(s') - Phi(s), g the per-step
discount the learner uses. Over a held decision the discounted sum is
g^k Phi(s_k) - Phi(s_0), and over an episode it telescopes to the end points,
so it changes which actions look good now but not which policy is best. Two
potentials:

  shape_progress  team: Phi = -sum over persons inside the hazard of
                  min(1, geodesic metres to safety / D_ref), / N_ref, the
                  quantity remaining_path charges per step. People walking
                  out (followers led out) pay at once; led back in, they cost
                  at once.
  shape_approach  per robot: Phi_j = 1 - exp(-S_j / kappa), S_j the sum over
                  persons inside the hazard who know nothing of it yet of
                  max(0, 1 - metres from robot j / cap). The robot that walks
                  toward people still unreached, more so toward many, is paid
                  for it the same step; walking away gives it back. Charged
                  to that robot only, like its collisions.

The approach potential is a smooth sum on purpose. The first version, minus
the walking distance to the nearest such person, jumped at 19-27% of steps
and its jumps carried 98-99% of its variance (validation levels, shuttle and
random robots): the navmesh path through triangle centres jumps when either
end crosses a triangle edge, and a hard nearest jumps to the next person
whenever the nearest one steps out of the hazard, notices it or is reached.
Straight-line metres are continuous, and a person entering or leaving the sum
moves Phi_j by at most 1/kappa, less once S_j is large; buildings are ignored,
which the cap keeps local and which cannot change the best policy.

Neither is part of the task reward: rollouts report `total_reward` and every
evaluation metric without them, and log them as their own components.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

COMPONENTS = ("person_time", "remaining_path", "reentry", "collision",
              "guide_use", "bystander", "projection",
              "shape_progress", "shape_approach")
# The terms charged to one robot rather than the team, and the versions that
# do so. Under these, a robot's learning reward is the team's task terms plus
# its own share of these (learn/sac.py); the logged team total is unchanged.
ROBOT_TERMS = ("collision", "guide_use", "bystander", "projection",
               "shape_approach")
PER_ROBOT_VERSIONS = ("rew-v3-own-collision", "rew-v4-robot-costs",
                      "rew-v5-projection", "rew-v6-pbrs")
# Learning-only terms (rew-v6): in the learner's reward, never in the task
# reward a rollout reports or an evaluation compares.
SHAPING_TERMS = ("shape_progress", "shape_approach")
VERSIONS = ("rew-v2-person-time", "rew-v3-own-collision",
            "rew-v4-robot-costs", "rew-v5-projection", "rew-v6-pbrs")


def reference_scales(model, cfg):
    """(N_ref, D_ref) for an episode, from its state at the start, or the
    fixed constants under REWARD_REFERENCE = "fixed"."""
    if cfg.REWARD_REFERENCE == "fixed":
        return float(cfg.REWARD_FIXED_N_REF), float(cfg.REWARD_FIXED_D_REF)
    n_inside = int(model.agents_in_danger())
    n_ref = max(int(cfg.REWARD_MIN_REFERENCE_POPULATION), n_inside)
    zone = getattr(model, "danger_zone", None)
    worst = 0.0
    if zone is not None:
        margin = float(cfg.DANGER_SAFE_MARGIN_M)
        for mesh, d in getattr(model, "mesh_danger", {}).items():
            if d >= 1e5:
                continue
            cx = (mesh[0][0] + mesh[1][0] + mesh[2][0]) / 3.0
            cy = (mesh[0][1] + mesh[1][1] + mesh[2][1]) / 3.0
            if zone.signed_distance(cx, cy) < margin:
                worst = max(worst, float(d))
        if worst <= 0.0:
            worst = float(zone.max_escape_distance())
    d_ref = max(float(cfg.REWARD_MIN_REFERENCE_DISTANCE_M), worst)
    return float(n_ref), float(d_ref)


class TaskReward:
    """Per-step rew-v2 components for one episode.

    Construct it after the model is built and before the first step; call
    `step(model)` after every `model.step()`. Returns the weighted components
    and keeps raw counts in `last_raw` for logging.
    """

    def __init__(self, model, cfg):
        if cfg.REWARD_VERSION not in VERSIONS:
            raise ValueError(f"unsupported REWARD_VERSION {cfg.REWARD_VERSION}")
        self.cfg = cfg
        self.n_ref, self.d_ref = reference_scales(model, cfg)
        # For the critic (sim/observation.record_decision): the reward's own
        # scale for this episode, fixed at its start.
        model.reward_n_ref, model.reward_d_ref = self.n_ref, self.d_ref
        self.dt = float(cfg.AGENT_TIME_STEP)
        self.weights = {
            "person_time": float(cfg.REWARD_W_PERSON_TIME),
            "remaining_path": float(cfg.REWARD_W_REMAINING_PATH),
            "reentry": float(cfg.REWARD_W_REENTRY),
            "collision": float(cfg.REWARD_W_COLLISION),
        }
        # rew-v4 only: guiding costs a little per robot-step, and so does
        # each person drawn into following who was not at risk.
        v6 = cfg.REWARD_VERSION == "rew-v6-pbrs"
        v4 = cfg.REWARD_VERSION in ("rew-v4-robot-costs", "rew-v5-projection") or v6
        v5 = cfg.REWARD_VERSION == "rew-v5-projection" or v6
        self.weights["guide_use"] = float(cfg.REWARD_W_GUIDE_USE) if v4 else 0.0
        self.weights["bystander"] = float(cfg.REWARD_W_BYSTANDER) if v4 else 0.0
        # rew-v5: a waypoint asked for in a building or off the network is
        # moved; the distance moved, over ROBOT_WAYPOINT_RANGE_M, once per
        # decision.
        self.weights["projection"] = (float(cfg.REWARD_W_PROJECTION) if v5
                                      else 0.0)
        self.waypoint_range = float(cfg.ROBOT_WAYPOINT_RANGE_M)
        # rew-v6: potential-based shaping (module docstring).
        self.weights["shape_progress"] = (float(cfg.REWARD_W_SHAPE_PROGRESS)
                                          if v6 else 0.0)
        self.weights["shape_approach"] = (float(cfg.REWARD_W_SHAPE_APPROACH)
                                          if v6 else 0.0)
        self.approach_cap = (float(cfg.REWARD_SHAPE_APPROACH_CAP_M) if v6
                             else 1.0)
        self.approach_kappa = (float(cfg.REWARD_SHAPE_APPROACH_PEOPLE) if v6
                               else 1.0)
        # The learner's per-step discount (learn/sac.py): the same g makes
        # the shaping telescope inside its Bellman targets.
        gamma = (cfg.gamma() if callable(getattr(cfg, "gamma", None))
                 else float(cfg.GAMMA_WAYPOINT
                            if cfg.ROBOT_ACTION_MODE == "waypoint"
                            else cfg.GAMMA_VELOCITY))
        self.g_step = float(gamma) ** (1.0 / max(1, int(cfg.ACTION_SCALE)))
        self._shape_progress = self.weights["shape_progress"] != 0.0
        self._shape_approach = self.weights["shape_approach"] != 0.0
        self._phi_prog = 0.0
        self._phi_appr: List[float] = []
        if self._shape_progress or self._shape_approach:
            inside = self._inside(model)
            self._phi_prog = self._phi_progress(model, inside)
            self._phi_appr = self._phi_approach(
                model, list(getattr(model, "robots", []) or []), inside)
        self._clear = self._clear_ids(model)
        self.last_raw: Dict[str, float] = {}
        self.last_robot_penalties = []
        # Followers recruited outside any risk, by robot id; filled by
        # CrowdAgent._start_following and drained here every step.
        model.bystander_events = []

    @staticmethod
    def _alive(model):
        return [a for a in model.crowds
                if a.type in (0, 1, 2) and not a.dead]

    def _inside(self, model):
        zone = getattr(model, "danger_zone", None)
        if zone is None:
            return []
        return [a for a in self._alive(model)
                if zone.contains(float(a.xy[0]), float(a.xy[1]))]

    # ------------------------------------------------- rew-v6 potentials

    def _phi_progress(self, model, inside) -> float:
        """-sum over `inside` of min(1, escape metres / D_ref), / N_ref."""
        s = sum(min(1.0, float(model.escape_distance(a.xy)) / self.d_ref)
                for a in inside)
        return -s / self.n_ref

    @staticmethod
    def unreached(person) -> bool:
        """Inside the hazard (the caller's filter) and nobody has got to it:
        not following a robot, not told by one, not aware of the hazard by
        any route (sensing it, being told). Those who know will leave on
        their own; the approach potential points robots at the rest."""
        return (person.type != 0
                and not getattr(person, "informed_by_robot", False)
                and not getattr(person, "hazard_memory", None))

    def _phi_approach(self, model, robots, inside) -> List[float]:
        """Phi_j = 1 - exp(-S_j / kappa) per robot, S_j the sum over the
        unreached persons in `inside` of max(0, 1 - metres to robot j / cap);
        0 for every robot when nobody inside is unreached."""
        if not robots:
            return []
        pts = [(float(a.xy[0]), float(a.xy[1])) for a in inside
               if self.unreached(a)]
        if not pts:
            return [0.0] * len(robots)
        P = np.asarray(pts, dtype=np.float64)
        R = np.asarray([(float(rb.xy[0]), float(rb.xy[1])) for rb in robots],
                       dtype=np.float64)
        d = np.hypot(R[:, None, 0] - P[None, :, 0], R[:, None, 1] - P[None, :, 1])
        S = np.maximum(0.0, 1.0 - d / self.approach_cap).sum(axis=1)
        return [float(v) for v in -np.expm1(-S / self.approach_kappa)]

    def terminal_correction(self):
        """(team, per robot) amounts to add to the last step's learning
        reward when the task itself ended the episode: the potential of a
        terminal state is zero, so the last F is -Phi(s_{T-1}), not
        g Phi(s_T) - Phi(s_{T-1}). Weighted, like step()'s output."""
        team = -self.weights["shape_progress"] * self.g_step * self._phi_prog
        robots = [-self.weights["shape_approach"] * self.g_step * p
                  for p in self._phi_appr]
        return team, robots

    def _clear_ids(self, model):
        return {a.unique_id for a in self._alive(model)
                if model.is_safe(a.xy)}

    def step(self, model) -> Dict[str, float]:
        zone = getattr(model, "danger_zone", None)
        alive = self._alive(model)
        inside = []
        for a in alive:
            if zone is not None and zone.contains(float(a.xy[0]),
                                                  float(a.xy[1])):
                inside.append(a)
        # An entry is someone who had been clear of the hazard by the safety
        # margin, is still in the crop, and is inside now. The margin is a
        # hysteresis: a person standing on the boundary line and jostled back
        # and forth across it is not re-entering each time, and the robots can
        # do nothing about that jitter. Leaving the crop removes a person from
        # `alive`, so it is never an entry; newcomers from inflow start
        # counting once they are clear.
        inside_ids = {a.unique_id for a in inside}
        entries = len(self._clear & inside_ids)
        self._clear = (self._clear - inside_ids) | {
            a.unique_id for a in alive if model.is_safe(a.xy)}
        self._clear &= {a.unique_id for a in alive}

        n_inside = len(inside)
        path = 0.0
        for a in inside:
            path += min(1.0, float(model.escape_distance(a.xy)) / self.d_ref)
        # rew-v6 shaping: F = g Phi(s') - Phi(s) for the step just taken.
        d_prog = 0.0
        if self._shape_progress:
            phi = -path / self.n_ref           # the same sum remaining_path uses
            d_prog = self.g_step * phi - self._phi_prog
            self._phi_prog = phi
        # Per robot as well as summed: under PER_ROBOT_VERSIONS each robot's
        # learning reward carries only its own share (learn/sac.py).
        robots = list(getattr(model, "robots", []) or [])
        hit = [int(bool(getattr(rb, "collision_check", 0))) for rb in robots]
        guiding = [int(getattr(rb, "mode", "off") == "guide") for rb in robots]
        events = getattr(model, "bystander_events", None) or []
        drawn = [sum(1 for rid in events if rid == rb.unique_id)
                 for rb in robots]
        model.bystander_events = []
        moved = []
        for rb in robots:
            d = getattr(rb, "projection_pending", None)
            rb.projection_pending = None
            # Capped at 1: a target clamped to the crop and then moved to a
            # far triangle can be moved further than the range itself.
            moved.append(0.0 if d is None
                         else min(1.0, float(d) / self.waypoint_range))
        approach = [0.0] * len(robots)
        if self._shape_approach:
            phis = self._phi_approach(model, robots, inside)
            for j, phi in enumerate(phis):
                prev = self._phi_appr[j] if j < len(self._phi_appr) else phi
                approach[j] = self.g_step * phi - prev
            self._phi_appr = phis
        w = self.weights
        self.last_robot_penalties = [
            -w["collision"] * h - w["guide_use"] * g / self.n_ref
            - w["bystander"] * b / self.n_ref - w["projection"] * p
            + w["shape_approach"] * a
            for h, g, b, p, a in zip(hit, guiding, drawn, moved, approach)]
        collisions = sum(hit)
        raw = {
            "person_time": -n_inside * self.dt / self.n_ref,
            "remaining_path": -path * self.dt / self.n_ref,
            "reentry": -entries / self.n_ref,
            "collision": -float(collisions),
            "guide_use": -float(sum(guiding)) / self.n_ref,
            "bystander": -float(sum(drawn)) / self.n_ref,
            "projection": -float(sum(moved)),
            "shape_progress": d_prog,
            "shape_approach": float(sum(approach)),
        }
        self.last_raw = {"inside": n_inside, "entries": entries,
                         "collisions": collisions, "path_fraction_sum": path,
                         "guiding": sum(guiding), "bystanders": sum(drawn)}
        return {k: w[k] * raw[k] for k in COMPONENTS}
