"""Settlements: Mesa CellAgents whose numeric state lives in NumPy arrays.

Each settlement is a ``Settlement`` CellAgent on one cell (at most one per cell). Its
numbers (population, food store, lineage, parent, ...) are columns of a ``SettlementTable``
indexed by the settlement id, so the yearly demography runs as array operations over all
living settlements; Python loops only handle events (founding, moving, dissolving).
Ids are never reused, so parent ids and the lineage tree stay valid after a settlement
ends.
"""

from __future__ import annotations

import numpy as np
from mesa.discrete_space import CellAgent

ALIVE = -1  # value of ``ended`` while a settlement exists


class SettlementTable:
    """Struct of arrays over every settlement ever founded, grown by doubling."""

    FLOAT_COLUMNS = ("population", "store", "food_ratio")
    INT_COLUMNS = ("cell", "lineage", "parent", "founded", "ended", "next_split_year")

    def __init__(self, capacity: int = 256):
        self.n = 0  # settlements ever founded; ids 0..n-1
        for name in self.FLOAT_COLUMNS:
            setattr(self, name, np.zeros(capacity))
        for name in self.INT_COLUMNS:
            setattr(self, name, np.full(capacity, -1, dtype=np.int64))

    def _grow(self) -> None:
        for name in self.FLOAT_COLUMNS + self.INT_COLUMNS:
            old = getattr(self, name)
            new = np.full(2 * len(old), 0.0 if old.dtype.kind == "f" else -1, dtype=old.dtype)
            new[: len(old)] = old
            setattr(self, name, new)

    def add(self, cell: int, population: float, lineage: int, parent: int, year: int,
            store: float = 0.0) -> int:
        if self.n == len(self.population):
            self._grow()
        sid = self.n
        self.n += 1
        self.cell[sid] = cell
        self.population[sid] = population
        self.store[sid] = store
        self.food_ratio[sid] = 1.0
        self.lineage[sid] = lineage
        self.parent[sid] = parent
        self.founded[sid] = year
        self.ended[sid] = ALIVE
        self.next_split_year[sid] = year
        return sid

    def alive_ids(self) -> np.ndarray:
        return np.flatnonzero(self.ended[: self.n] == ALIVE)


class Settlement(CellAgent):
    """A settlement on one cell. Its state is read from the model's SettlementTable."""

    def __init__(self, model, sid: int, cell):
        super().__init__(model)
        self.sid = sid
        self.cell = cell

    @property
    def population(self) -> float:
        return float(self.model.settlements.population[self.sid])

    @property
    def lineage(self) -> int:
        return int(self.model.settlements.lineage[self.sid])

    @property
    def parent(self) -> int:
        return int(self.model.settlements.parent[self.sid])
