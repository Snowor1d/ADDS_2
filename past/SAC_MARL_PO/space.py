# ──────────────────────────────────────────────────────────────────────────────
# crowdlib/space.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Set, Callable, Optional, Union
from shapely.geometry import LineString
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

AgentKey = Union[int, object]  # uid(int) 또는 agent 객체

class ContinuousSpace:
    """Continuous 2D world with spatial hashing.
    - Float positions, velocities
    - O(1) avg insert/move/remove
    - Fast neighbor queries without scanning a grid
    """

    def __init__(self, width: float, height: float, cell_size: float = 10.0, torus: bool = False):
        self.width  = float(width)
        self.height = float(height)
        self.cell   = float(cell_size)
        self.inv    = 1.0 / self.cell
        self.torus  = torus
        self._cells: Dict[Cell, Set[int]] = defaultdict(set)
        self._bodies: Dict[int, Body] = {}
        # NEW: agent 객체 -> uid 역인덱스
        self._uid_of: Dict[object, int] = {}

    # ───── helpers ─────
    def _cell_of(self, p: Vec2) -> Cell:
        return (int(math.floor(p[0] * self.inv)), int(math.floor(p[1] * self.inv)))

    def _cell_neighbors(self, c: Cell, r: float) -> Iterable[Cell]:
        k = int(math.ceil(r * self.inv))
        for dx in range(-k, k + 1):
            for dy in range(-k, k + 1):
                yield (c[0] + dx, c[1] + dy)

    # ───── body mgmt ─────
    def add(self, uid: int, pos: Vec2, radius: float, ref: object, vel: Vec2 = (0.0, 0.0)) -> None:
        b = Body(uid, [float(pos[0]), float(pos[1])],
                      [float(vel[0]), float(vel[1])],
                      float(radius), ref)
        self._bodies[uid] = b
        self._cells[self._cell_of(b.pos)].add(uid)
        if ref is not None:
            self._uid_of[ref] = uid  # 역인덱스 등록

    def remove(self, uid: int) -> None:
        b = self._bodies.pop(uid, None)
        if b is None:
            return
        self._cells[self._cell_of(b.pos)].discard(uid)
        # 역인덱스 제거
        if b.ref in self._uid_of and self._uid_of[b.ref] == uid:
            self._uid_of.pop(b.ref, None)

    def move(self, uid: int, new_pos: Vec2) -> None:
        b = self._bodies.get(uid)
        if b is None:
            return
        old_cell = self._cell_of(b.pos)
        b.pos[0] = float(new_pos[0])
        b.pos[1] = float(new_pos[1])
        new_cell = self._cell_of(b.pos)
        if new_cell != old_cell:
            self._cells[old_cell].discard(uid)
            self._cells[new_cell].add(uid)

    # ───── agent-level API ─────
    def _resolve_uid(self, agent_or_uid: AgentKey) -> Optional[int]:
        if isinstance(agent_or_uid, int):
            return agent_or_uid if agent_or_uid in self._bodies else None
        return self._uid_of.get(agent_or_uid)

    def place_agent(self, agent: object, pos: Vec2, radius: float = 0.5, vel: Vec2 = (0.0, 0.0)) -> int:
        """agent 객체를 공간에 배치하고 uid 인덱싱까지 완료.
        agent.unique_id 또는 agent.id 를 uid로 사용.
        기존에 같은 uid가 있으면 제거 후 재등록."""
        if hasattr(agent, "unique_id"):
            uid = int(getattr(agent, "unique_id"))
        elif hasattr(agent, "id"):
            uid = int(getattr(agent, "id"))
        else:
            raise ValueError("place_agent: agent must have unique_id or id")

        if uid in self._bodies:
            self.remove(uid)

        self.add(uid, pos, float(radius), ref=agent, vel=vel)
        return uid

    def get_agent(self, uid: int) -> Optional[object]:
        b = self._bodies.get(uid)
        return b.ref if b else None

    def get_body(self, uid: int) -> Optional[Body]:
        return self._bodies.get(uid)

    def get_body_by_agent(self, agent: object) -> Optional[Body]:
        uid = self._uid_of.get(agent)
        return self._bodies.get(uid) if uid is not None else None

    def move_agent(self, agent_or_uid: AgentKey, new_pos: Vec2) -> None:
        uid = self._resolve_uid(agent_or_uid)
        if uid is None:
            return
        self.move(uid, new_pos)

    def remove_agent(self, agent_or_uid: AgentKey) -> None:
        uid = self._resolve_uid(agent_or_uid)
        if uid is None:
            return
        self.remove(uid)

    def position_of(self, agent_or_uid: AgentKey) -> Optional[Tuple[float, float]]:
        uid = self._resolve_uid(agent_or_uid)
        if uid is None:
            return None
        b = self._bodies.get(uid)
        return (b.pos[0], b.pos[1]) if b else None

    def update_radius(self, agent_or_uid: AgentKey, new_radius: float) -> None:
        uid = self._resolve_uid(agent_or_uid)
        if uid is None:
            return
        b = self._bodies.get(uid)
        if b:
            b.radius = float(new_radius)

    # ───── queries ─────
    def get(self, uid: int) -> Optional[Body]:
        return self._bodies.get(uid)

    def query_radius(self, center: Vec2, r: float,
                     predicate: Callable[[Body], bool] | None = None) -> List[Body]:
        out: List[Body] = []
        rc2 = r * r
        c = self._cell_of(center)
        for nc in self._cell_neighbors(c, r):
            for uid in self._cells.get(nc, ()):
                b = self._bodies[uid]
                dx = b.pos[0] - center[0]
                dy = b.pos[1] - center[1]
                if dx * dx + dy * dy <= rc2:
                    if (predicate is None) or predicate(b):
                        out.append(b)
        return out

    def query_aabb(self, x0: float, y0: float, x1: float, y1: float,
                   predicate: Callable[[Body], bool] | None = None) -> List[Body]:
        out: List[Body] = []
        c0 = self._cell_of((x0, y0))
        c1 = self._cell_of((x1, y1))
        for cx in range(min(c0[0], c1[0]), max(c0[0], c1[0]) + 1):
            for cy in range(min(c0[1], c1[1]), max(c0[1], c1[1]) + 1):
                for uid in self._cells.get((cx, cy), ()):
                    b = self._bodies[uid]
                    if (x0 <= b.pos[0] <= x1) and (y0 <= b.pos[1] <= y1):
                        if (predicate is None) or predicate(b):
                            out.append(b)
        return out

    def k_nearest(self, center: Vec2, k: int, r_max: float) -> List[Body]:
        cand = self.query_radius(center, r_max)
        cand.sort(key=lambda b: (b.pos[0] - center[0]) ** 2 + (b.pos[1] - center[1]) ** 2)
        return cand[:k]

    # ───── agent-level queries ─────
    def agents_in_radius(self, center: Vec2, r: float,
                         predicate: Optional[Callable[[object], bool]] = None) -> List[object]:
        """반경 r 안의 Agent 객체 목록 반환"""
        def _pred(b: Body) -> bool:
            return True if predicate is None else predicate(b.ref)
        bodies = self.query_radius(center, r, predicate=_pred)
        return [b.ref for b in bodies]

    def agent_at(self, pos: Vec2, r: float = 0.5,
                 predicate: Optional[Callable[[object], bool]] = None) -> Optional[object]:
        """pos 주변 r 안에서 가장 가까운 Agent 1명을 반환"""
        cand = self.query_radius(pos, r)
        best = None
        best_d2 = float("inf")
        for b in cand:
            if predicate and not predicate(b.ref):
                continue
            dx = b.pos[0] - pos[0]
            dy = b.pos[1] - pos[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = b.ref
        return best

    # ───── boundary clamp/torus ─────
    def clamp(self, p: List[float]) -> None:
        if self.torus:
            p[0] = (p[0] % self.width)
            p[1] = (p[1] % self.height)
        else:
            p[0] = max(0.0, min(self.width,  p[0]))
            p[1] = max(0.0, min(self.height, p[1]))

    def _segment_blocked(center, target, obstacle_polys) -> bool:
        if not obstacle_polys:
            return False
        seg = LineString([center, target])
        for poly in obstacle_polys:
            if seg.intersects(poly):  # touches 포함하려면 그대로, 엄격히면 crosses
                return True
        return False

    def query_radius_visible(self, center: Vec2, r: float,
                             obstacle_polys,
                             predicate: Callable[[Body], bool] | None = None) -> List[Body]:
        candidates: List[Body] = self.query_radius(center, r, predicate)
        out: List[Body] = []
        for b in candidates:
            if not self._segment_blocked(center, (b.pos[0], b.pos[1]), obstacle_polys):
                out.append(b)
        return out
    

