"""Person-level common random numbers for the crowd simulator (probe only).

The simulator draws crowd randomness from the global `random` module, so a
different robot action shifts every later draw: the same person's response
to the same situation differs between two branches only because of the order
of calls. Installed, this module gives every draw its own stream keyed by

    (person id, simulation step, calling function)

or (model, step, calling function) for draws outside any person, plus a
per-person seed for the numpy draws a new person makes when it is created.
Within one repeat every branch then shares each person's exogenous draws;
repeats differ by `base`. Nothing under sim/ is edited: the module attribute
`random` of sim.agent, sim.model and sim.od is swapped for an ExoRandom.
"""
import random as _random
import sys

import numpy as np

import sim.agent
import sim.model
import sim.od


class ExoRandom:
    Random = _random.Random

    def __init__(self):
        self.base = 0
        self.t = 0
        self._cache = {}
        self._cache_t = None

    def reset(self, base: int) -> None:
        self.base = int(base)
        self._cache = {}
        self._cache_t = None

    def _key(self):
        f = sys._getframe(2)             # the code that called random.<fn>
        site = f.f_code.co_name
        owner = ("m",)
        g = f
        for _ in range(6):
            s = g.f_locals.get("self")
            if isinstance(s, sim.agent.CrowdAgent):
                owner = ("p", int(s.unique_id))
                break
            g = g.f_back
            if g is None:
                break
        return owner + (int(self.t), site)

    def __getattr__(self, name):
        if self._cache_t != self.t:
            self._cache = {}
            self._cache_t = self.t
        key = self._key()
        r = self._cache.get(key)
        if r is None:
            r = self._cache[key] = _random.Random(f"{self.base}:{key}")
        return getattr(r, name)


EXO = ExoRandom()
_orig = {}


def install():
    if _orig:
        return EXO
    _orig["agent"], _orig["model"], _orig["od"] = sim.agent.random, sim.model.random, sim.od.random
    sim.agent.random = sim.model.random = sim.od.random = EXO
    init = sim.agent.CrowdAgent.__init__
    _orig["init"] = init

    def seeded_init(self, unique_id, *a, **k):
        # A new person's numpy traits (mass, desired speed) from its own seed.
        np.random.seed((EXO.base * 1_000_003 + int(unique_id)) % 2 ** 32)
        init(self, unique_id, *a, **k)

    sim.agent.CrowdAgent.__init__ = seeded_init
    return EXO


def uninstall():
    if not _orig:
        return
    sim.agent.random, sim.model.random, sim.od.random = _orig["agent"], _orig["model"], _orig["od"]
    sim.agent.CrowdAgent.__init__ = _orig["init"]
    _orig.clear()
