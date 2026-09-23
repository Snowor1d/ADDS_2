
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
        self.agents = []

    def step(self) -> None:
        """Advance one tick. Override in subclass if needed.
        Default: scheduler.step(); time += 1
        """
        self.schedule.step()
        self.time += 1
