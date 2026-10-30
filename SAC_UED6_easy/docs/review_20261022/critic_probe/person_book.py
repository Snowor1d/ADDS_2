"""Each robot decision's return restricted to the people near the robot.

Used by collect_replay.py --person-returns. During learn/rollout.run_episode
(through its on_step hook) the team's learning reward is booked to persons,
as true_adv_persons.PersonEpisode does: per person inside the hazard,
person_time, remaining_path and reentry, and the progress shaping
w (g phi_p(s') - phi_p(s)) with phi_p = -min(1, escape_m / D_ref) / N_ref; the
terminal correction -w g phi_p(s_T) goes to the last step, where run_episode
puts it. Robot-charged terms are the robot's own and come from the emitted
transitions. Every step the persons' sum is checked against the episode's
team reward.

Only read-only reward helpers are used (TaskReward.step drains model events,
so a second instance must not step).

Targets per robot decision (frame k, robot j), discounted with the learner's
per-step g from the decision's first step:

  local20 / local40  persons within 20 / 40 m of robot j at frame k, over
                     HORIZON steps, plus robot j's own terms
  global             the whole team reward over HORIZON steps, plus robot j's
  global_full        the same to the end of the episode
"""
import numpy as np

from configs import frame_steps
from sim.rewards import TaskReward

HORIZON = 400                 # steps: 200 s at 0.5 s
RADII = (20.0, 40.0)
NAMES = ("local20", "local40", "global", "global_full")


class PersonBook:
    def __init__(self, model, cfg):
        # Constructing a TaskReward only sets values run_episode's own
        # constructor sets again; it is never stepped.
        tr = TaskReward(model, cfg)
        self.tr = tr
        self.F = frame_steps(cfg)
        self.R = int(cfg.MAX_ROBOTS)
        self.w = tr.weights
        self.g = tr.g_step
        self.clear = tr._clear_ids(model)
        self.phi = {}
        if self.w["shape_progress"]:
            for a in tr._inside(model):
                self.phi[a.unique_id] = -min(1.0, float(model.escape_distance(a.xy)) / tr.d_ref) / tr.n_ref
        self.steps = []               # per step: (pids, values)
        self.frames = {0: self._snapshot(model)}
        self.transitions = []         # (frame, decided or None, step_rewards, penalties, terminal)

    @staticmethod
    def _snapshot(model):
        people = [a for a in model.crowds if a.type in (0, 1, 2) and not a.dead]
        ids = np.asarray([a.unique_id for a in people], np.int64)
        xy = np.asarray([[float(a.xy[0]), float(a.xy[1])] for a in people], np.float64).reshape(-1, 2)
        robots = np.asarray([[float(r.xy[0]), float(r.xy[1])] for r in model.robots], np.float64)
        return ids, xy, robots

    def on_step(self, model, step):
        tr, w = self.tr, self.w
        inside = tr._inside(model)
        book = {}
        phi_new = {}
        for a in inside:
            pid = a.unique_id
            esc = min(1.0, float(model.escape_distance(a.xy)) / tr.d_ref)
            v = (-w["person_time"] * tr.dt / tr.n_ref
                 - w["remaining_path"] * esc * tr.dt / tr.n_ref)
            if pid in self.clear:
                v += -w["reentry"] / tr.n_ref
            book[pid] = book.get(pid, 0.0) + v
            phi_new[pid] = -esc / tr.n_ref
        if w["shape_progress"]:
            for pid in set(phi_new) | set(self.phi):
                book[pid] = book.get(pid, 0.0) + w["shape_progress"] * (
                    self.g * phi_new.get(pid, 0.0) - self.phi.get(pid, 0.0))
        self.phi = phi_new
        alive = tr._alive(model)
        inside_ids = {a.unique_id for a in inside}
        self.clear = (self.clear - inside_ids) | {a.unique_id for a in alive if model.is_safe(a.xy)}
        self.clear &= {a.unique_id for a in alive}
        self.steps.append((np.fromiter(book.keys(), np.int64, len(book)),
                           np.fromiter(book.values(), np.float64, len(book))))
        if step % self.F == 0:
            self.frames[step // self.F] = self._snapshot(model)

    def emit(self, tr):
        if tr.final:
            return
        decided = None if tr.action is None else np.asarray(tr.decided, bool).copy()
        self.transitions.append((int(tr.step), decided, list(tr.step_rewards),
                                 [list(p) for p in tr.robot_penalties], bool(tr.terminal)))

    def targets(self):
        """(frames (n,), targets (n, R, len(NAMES)), valid (n, R), max error)."""
        team = np.concatenate([np.asarray(t[2], np.float64) for t in self.transitions])
        pen = np.zeros((len(team), self.R))
        k = 0
        for t in self.transitions:
            for row in t[3]:
                pen[k, :len(row)] = row
                k += 1
        T = len(team)
        assert T == len(self.steps), (T, len(self.steps))
        if self.transitions[-1][4] and self.w["shape_progress"]:
            pids, vals = self.steps[-1]
            vals = vals.copy()
            for i, pid in enumerate(pids):
                vals[i] += -self.w["shape_progress"] * self.g * self.phi.get(int(pid), 0.0)
            # People whose last-step entry is only the correction.
            extra = [pid for pid in self.phi if pid not in set(pids.tolist())]
            if extra:
                pids = np.concatenate([pids, np.asarray(extra, np.int64)])
                vals = np.concatenate([vals, [-self.w["shape_progress"] * self.g * self.phi[p] for p in extra]])
            self.steps[-1] = (pids, vals)
        mine = np.asarray([v.sum() for _, v in self.steps])
        err = float(np.abs(mine - team).max()) if T else 0.0
        cols = {}
        for pids, _ in self.steps:
            for p in pids.tolist():
                cols.setdefault(p, len(cols))
        Mx = np.zeros((T, max(1, len(cols))))
        for t, (pids, vals) in enumerate(self.steps):
            if len(pids):
                Mx[t, [cols[p] for p in pids.tolist()]] = vals
        disc = self.g ** np.arange(T)
        frames, out, valid = [], [], []
        for frame, decided, *_ in self.transitions:
            if decided is None:
                continue
            t0 = frame * self.F
            ids, xy, robots = self.frames[frame]
            h = min(T, t0 + HORIZON) - t0
            dh = disc[:h]
            df = disc[:T - t0]
            row = np.zeros((self.R, len(NAMES)), np.float32)
            for j in range(min(self.R, len(robots))):
                if not decided[j]:
                    continue
                own_h = float(dh @ pen[t0:t0 + h, j])
                d = np.hypot(xy[:, 0] - robots[j, 0], xy[:, 1] - robots[j, 1])
                for r, rad in enumerate(RADII):
                    c = [cols[p] for p in ids[d < rad].tolist() if p in cols]
                    row[j, r] = (float(dh @ Mx[t0:t0 + h, c].sum(1)) if c else 0.0) + own_h
                row[j, len(RADII)] = float(dh @ team[t0:t0 + h]) + own_h
                row[j, len(RADII) + 1] = float(df @ (team[t0:] + pen[t0:, j]))
            frames.append(frame)
            out.append(row)
            valid.append(decided[:self.R].copy())
        return (np.asarray(frames, np.int64), np.asarray(out, np.float32).reshape(-1, self.R, len(NAMES)),
                np.asarray(valid, bool).reshape(-1, self.R), err)
