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
"""

from __future__ import annotations

import math
from typing import Dict

COMPONENTS = ("person_time", "remaining_path", "reentry", "collision",
              "guide_use", "bystander")
# The terms charged to one robot rather than the team, and the versions that
# do so. Under these, a robot's learning reward is the team's task terms plus
# its own share of these (learn/sac.py); the logged team total is unchanged.
ROBOT_TERMS = ("collision", "guide_use", "bystander")
PER_ROBOT_VERSIONS = ("rew-v3-own-collision", "rew-v4-robot-costs")


def reference_scales(model, cfg):
    """(N_ref, D_ref) for an episode, from its state at the start."""
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
        if cfg.REWARD_VERSION not in ("rew-v2-person-time",
                                      "rew-v3-own-collision",
                                      "rew-v4-robot-costs"):
            raise ValueError(f"unsupported REWARD_VERSION {cfg.REWARD_VERSION}")
        self.cfg = cfg
        self.n_ref, self.d_ref = reference_scales(model, cfg)
        self.dt = float(cfg.AGENT_TIME_STEP)
        self.weights = {
            "person_time": float(cfg.REWARD_W_PERSON_TIME),
            "remaining_path": float(cfg.REWARD_W_REMAINING_PATH),
            "reentry": float(cfg.REWARD_W_REENTRY),
            "collision": float(cfg.REWARD_W_COLLISION),
        }
        # rew-v4 only: guiding costs a little per robot-step, and so does
        # each person drawn into following who was not at risk.
        v4 = cfg.REWARD_VERSION == "rew-v4-robot-costs"
        self.weights["guide_use"] = float(cfg.REWARD_W_GUIDE_USE) if v4 else 0.0
        self.weights["bystander"] = float(cfg.REWARD_W_BYSTANDER) if v4 else 0.0
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
        # Per robot as well as summed: under PER_ROBOT_VERSIONS each robot's
        # learning reward carries only its own share (learn/sac.py).
        robots = list(getattr(model, "robots", []) or [])
        hit = [int(bool(getattr(rb, "collision_check", 0))) for rb in robots]
        guiding = [int(getattr(rb, "mode", "off") == "guide") for rb in robots]
        events = getattr(model, "bystander_events", None) or []
        drawn = [sum(1 for rid in events if rid == rb.unique_id)
                 for rb in robots]
        model.bystander_events = []
        w = self.weights
        self.last_robot_penalties = [
            -w["collision"] * h - w["guide_use"] * g / self.n_ref
            - w["bystander"] * b / self.n_ref
            for h, g, b in zip(hit, guiding, drawn)]
        collisions = sum(hit)
        raw = {
            "person_time": -n_inside * self.dt / self.n_ref,
            "remaining_path": -path * self.dt / self.n_ref,
            "reentry": -entries / self.n_ref,
            "collision": -float(collisions),
            "guide_use": -float(sum(guiding)) / self.n_ref,
            "bystander": -float(sum(drawn)) / self.n_ref,
        }
        self.last_raw = {"inside": n_inside, "entries": entries,
                         "collisions": collisions, "path_fraction_sum": path,
                         "guiding": sum(guiding), "bystanders": sum(drawn)}
        return {k: w[k] * raw[k] for k in COMPONENTS}
