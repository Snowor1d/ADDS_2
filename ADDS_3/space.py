# ──────────────────────────────────────────────────────────────────────────────
if (predicate is None) or predicate(b):
out.append(b)
return out


def query_aabb(self, x0: float, y0: float, x1: float, y1: float, predicate: Callable[[Body], bool] | None = None) -> List[Body]:
out: List[Body] = []
c0 = self._cell_of((x0,y0)); c1 = self._cell_of((x1,y1))
for cx in range(min(c0[0],c1[0]), max(c0[0],c1[0])+1):
for cy in range(min(c0[1],c1[1]), max(c0[1],c1[1])+1):
for uid in self._cells.get((cx,cy), ()):
b = self._bodies[uid]
if (x0 <= b.pos[0] <= x1) and (y0 <= b.pos[1] <= y1):
if (predicate is None) or predicate(b):
out.append(b)
return out


def k_nearest(self, center: Vec2, k: int, r_max: float) -> List[Body]:
cand = self.query_radius(center, r_max)
cand.sort(key=lambda b: (b.pos[0]-center[0])**2 + (b.pos[1]-center[1])**2)
return cand[:k]


# ───── boundary clamp/torus ─────
def clamp(self, p: List[float]) -> None:
if self.torus:
p[0] = (p[0] % self.width)
p[1] = (p[1] % self.height)
else:
p[0] = max(0.0, min(self.width, p[0]))
p[1] = max(0.0, min(self.height, p[1]))
# ──────────────────────────────────────────────────────────────────────────────
# Project: Custom CrowdSim Engine (MESA replacement)
# Version: v0.2 (alpha) — Continuous Domain
# Author: you + ChatGPT
# Notes :
#   • v0.2 upgrades everything to a CONTINUOUS domain (no grid storage/scan):
#     - ContinuousSpace: float positions, spatial hashing index for O(1) avg insert/move
#     - Fast neighbor queries: query_radius / query_aabb / kNN without per-cell scans
#     - Obstacles as polygons (tri/square) with segment distance queries
#     - Optional NavMesh (triangulated) kept for macro pathing (centroid-to-centroid)
#   • API remains close to your current agents, but removes MultiGrid dependencies.
#   • Viewer now draws circles at float coords and supports zoom/scaling.
# ──────────────────────────────────────────────────────────────────────────────

# Directory layout (for reference)
# crowdlib/
#   __init__.py
#   core.py          # Agent, Model, Scheduler, DataCollector, RNG
#   space.py         # MultiGrid-like grid + ContinuousSpace placeholder
#   visualize.py     # Pygame-based viewer (play/pause/speed/step)
#   utils.py         # small helpers
# examples/
#   run_demo.py      # tiny demo wiring Model/Agent with Viewer
#
# Paste the following files to a folder named `crowdlib/` and an `examples/` dir.

# ──────────────────────────────────────────────────────────────────────────────
# crowdlib/__init__.py
# ──────────────────────────────────────────────────────────────────────────────
from .core import Agent, Model, RandomActivation, DataCollector
from .space import MultiGrid, ContinuousSpace
from .visualize import Viewer

__all__ = [
    "Agent", "Model", "RandomActivation", "DataCollector",
    "MultiGrid", "ContinuousSpace", "Viewer",
]

# ──────────────────────────────────────────────────────────────────────────────
# crowdlib/core.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import time
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Iterable

class Agent:
    """Lightweight base agent (MESA-like).
    - has: unique_id, model, pos (tuple[int,int] by convention), step()
    """
    def __init__(self, unique_id: int, model: "Model") -> None:
        self.unique_id = unique_id
        self.model = model
        self.pos = None  # set by grid.place_agent

    def step(self) -> None:
        pass

class RandomActivation:
    """Simple scheduler with add/remove/step (random order).
    API: add(agent), remove(agent), step() calls agent.step() in random order.
    """
    def __init__(self, model: "Model") -> None:
        self.model = model
        self._agents: Dict[int, Agent] = {}
        self.steps = 0

    @property
    def agents(self) -> List[Agent]:
        return list(self._agents.values())

    def add(self, agent: Agent) -> None:
        self._agents[agent.unique_id] = agent

    def remove(self, agent: Agent) -> None:
        self._agents.pop(agent.unique_id, None)

    def step(self) -> None:
        a = self.agents
        random.shuffle(a)
        for ag in a:
            ag.step()
        self.steps += 1

class DataCollector:
    """Ultra-minimal data collector.
    - model_reporters: {name: Callable[[Model], Any]}
    - agent_reporters: {name: str | Callable[[Agent], Any]}
    Usage: dc.collect(model); dc.get_model_df(); dc.get_agent_df(model)
    """
    def __init__(self, model_reporters: Dict[str, Callable[["Model"], Any]] | None = None,
                 agent_reporters: Dict[str, Any] | None = None) -> None:
        self.model_reporters = model_reporters or {}
        self.agent_reporters = agent_reporters or {}
        self._model_rows: List[Dict[str, Any]] = []
        self._agent_rows: List[Dict[str, Any]] = []

    def collect(self, model: "Model") -> None:
        mrow = {k: fn(model) for k, fn in self.model_reporters.items()}
        mrow["time"] = model.time
        self._model_rows.append(mrow)
        if self.agent_reporters:
            for ag in model.schedule.agents:
                row = {"id": ag.unique_id, "time": model.time}
                for k, rep in self.agent_reporters.items():
                    if callable(rep):
                        row[k] = rep(ag)
                    elif isinstance(rep, str):
                        row[k] = getattr(ag, rep)
                self._agent_rows.append(row)

    # Lazy: return plain lists; adapt to pandas later if needed
    def get_model_rows(self) -> List[Dict[str, Any]]:
        return self._model_rows

    def get_agent_rows(self) -> List[Dict[str, Any]]:
        return self._agent_rows

class Model:
    """Base model: holds schedule, time, running flag.
    You can subclass and add your own fields (grid, obstacles, etc.).
    """
    def __init__(self) -> None:
        self.random = random.Random()
        self.time: int = 0
        self.running: bool = True
        self.schedule = RandomActivation(self)

    def step(self) -> None:
        """Advance one tick. Override in subclass if needed.
        Default: scheduler.step(); time += 1
        """
        self.schedule.step()
        self.time += 1

# ──────────────────────────────────────────────────────────────────────────────
# crowdlib/space.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set, Callable, Optional
import math

Vec2 = Tuple[float, float]
Cell = Tuple[int, int]

@dataclass
class Body:
    id: int
    pos: List[float]
    vel: List[float]
    radius: float
    ref: object  # backref to Agent

class ContinuousSpace:
    """Continuous 2D world with spatial hashing.
    - Float positions, velocities
    - O(1) avg insert/move/remove
    - Fast neighbor queries without scanning a grid
    """
    def __init__(self, width: float, height: float, cell_size: float = 2.0, torus: bool = False):
        self.width  = float(width)
        self.height = float(height)
        self.cell   = float(cell_size)
        self.inv    = 1.0 / self.cell
        self.torus  = torus
        self._cells: Dict[Cell, Set[int]] = defaultdict(set)
        self._bodies: Dict[int, Body] = {}

    # ───── helpers ─────
    def _cell_of(self, p: Vec2) -> Cell:
        return (int(math.floor(p[0] * self.inv)), int(math.floor(p[1] * self.inv)))

    def _cell_neighbors(self, c: Cell, r: float) -> Iterable[Cell]:
        k = int(math.ceil(r * self.inv))
        for dx in range(-k, k+1):
            for dy in range(-k, k+1):
                yield (c[0]+dx, c[1]+dy)

    # ───── body mgmt ─────
    def add(self, uid: int, pos: Vec2, radius: float, ref: object, vel: Vec2=(0.0,0.0)) -> None:
        b = Body(uid, [float(pos[0]), float(pos[1])], [float(vel[0]), float(vel[1])], float(radius), ref)
        self._bodies[uid] = b
        self._cells[self._cell_of(b.pos)].add(uid)

    def remove(self, uid: int) -> None:
        b = self._bodies.pop(uid, None)
        if b is None: return
        self._cells[self._cell_of(b.pos)].discard(uid)

    def move(self, uid: int, new_pos: Vec2) -> None:
        b = self._bodies.get(uid)
        if b is None: return
        old_cell = self._cell_of(b.pos)
        b.pos[0] = float(new_pos[0]); b.pos[1] = float(new_pos[1])
        new_cell = self._cell_of(b.pos)
        if new_cell != old_cell:
            self._cells[old_cell].discard(uid)
            self._cells[new_cell].add(uid)

    # ───── queries ─────
    def get(self, uid: int) -> Optional[Body]:
        return self._bodies.get(uid)

    def query_radius(self, center: Vec2, r: float, predicate: Callable[[Body], bool] | None = None) -> List[Body]:
        out: List[Body] = []
        rc2 = r*r
        c = self._cell_of(center)
        for nc in self._cell_neighbors(c, r):
            for uid in self._cells.get(nc, ()):  
                b = self._bodies[uid]
                dx = b.pos[0]-center[0]; dy = b.pos[1]-center[1]
                if dx*dx + dy*dy <= rc2:
                    if (predicate is None) or predicate(b):
                        out.append(b)
        return out

    def query_aabb(self, x0: float, y0: float, x1: float, y1: float, predicate: Callable[[Body], bool] | None = None) -> List[Body]:
        out: List[Body] = []
        c0 = self._cell_of((x0,y0)); c1 = self._cell_of((x1,y1))
        for cx in range(min(c0[0],c1[0]), max(c0[0],c1[0])+1):
            for cy in range(min(c0[1],c1[1]), max(c0[1],c1[1])+1):
                for uid in self._cells.get((cx,cy), ()):  
                    b = self._bodies[uid]
                    if (x0 <= b.pos[0] <= x1) and (y0 <= b.pos[1] <= y1):
                        if (predicate is None) or predicate(b):
                            out.append(b)
        return out

    def k_nearest(self, center: Vec2, k: int, r_max: float) -> List[Body]:
        cand = self.query_radius(center, r_max)
        cand.sort(key=lambda b: (b.pos[0]-center[0])**2 + (b.pos[1]-center[1])**2)
        return cand[:k]

    # ───── boundary clamp/torus ─────
    def clamp(self, p: List[float]) -> None:
        if self.torus:
            p[0] = (p[0] % self.width)
            p[1] = (p[1] % self.height)
        else:
            p[0] = max(0.0, min(self.width,  p[0]))
            p[1] = max(0.0, min(self.height, p[1]))

    def get_ref(self, uid:int):
        b = self._bodies.get(uid)
        return None if b is None else b.ref
    
    def set_vel(self, uid: int, vel: Vec2):
        b = self.bodies.get(uid)
        if b is None: return
        b.vel[0], b.vel[1] = float(vel[0]), float(vel[1])

    def get_vel(self, uid: int) -> Tuple[float, float] | None:
        b = self._bodies.get(uid)
        if b is None: return None
        return (b.vel[0], b.vel[1])
