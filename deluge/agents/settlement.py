"""Settlements: Mesa CellAgents whose numeric state lives in NumPy arrays.

Each settlement is a ``Settlement`` CellAgent on one cell (at most one per cell). Its
numbers (population, food store, lineage, parent, subsistence shares, techniques, ...)
are rows of a ``SettlementTable`` indexed by the settlement id, so the yearly systems run
as array operations over all living settlements; Python loops only handle events
(founding, moving, dissolving). Ids are never reused, so parent ids and the lineage tree
stay valid after a settlement ends.
"""

from __future__ import annotations

import numpy as np
from mesa.discrete_space import CellAgent

ALIVE = -1  # value of ``ended`` while a settlement exists
MODES = ("forage", "herd", "farm")  # subsistence modes, the column order of ``shares``


class SettlementTable:
    """Struct of arrays over every settlement ever founded, grown by doubling.

    ``knows`` and ``strength`` have one column per technique: whether the settlement
    holds it, and how firmly (1 = fresh; decays when unused, forgotten at 0);
    ``practised``: it has used the technique while holding it. ``shares`` has one column
    per subsistence mode and sums to 1. ``works`` has one column per mineral: the cell of the
    deposit the settlement last worked, -1 for none. ``polity``: the polity it belongs to.

    ``appearance_mean`` / ``appearance_var`` (one column per appearance trait) and
    ``dialect`` are identity markers (spec "Ancestry, appearance and identity"). Only the
    perception-and-mixing system (``deluge/systems/identity.py``) reads or writes them; a
    new row gets zeros, and the caller sets them through that system.

    ``culture`` (one column per culture trait) is the settlement's own culture: its polity's,
    except in a conquered settlement, which keeps its own until assimilated (spec section 9).
    ``subject_of``: the polity that conquered it, while it is an unassimilated subject (-1 for
    none); ``grievance``: its resentment of conquest, decaying over generations."""

    FLOAT_COLUMNS = ("population", "store", "food_ratio", "peak_population", "grievance")
    INT_COLUMNS = ("cell", "lineage", "parent", "founded", "ended", "next_split_year", "famine_years", "polity",
                   "subject_of")
    MATRIX_COLUMNS = ("knows", "strength", "practised", "shares", "works", "appearance_mean", "appearance_var",
                      "dialect", "culture")

    def __init__(self, n_techniques: int = 0, capacity: int = 256, n_minerals: int = 0, n_appearance: int = 0,
                 n_dialect: int = 0, n_traits: int = 0):
        self.n = 0  # settlements ever founded; ids 0..n-1
        for name in self.FLOAT_COLUMNS:
            setattr(self, name, np.zeros(capacity))
        for name in self.INT_COLUMNS:
            setattr(self, name, np.full(capacity, -1, dtype=np.int64))
        self.knows = np.zeros((capacity, n_techniques), dtype=bool)
        self.strength = np.zeros((capacity, n_techniques))
        self.practised = np.zeros((capacity, n_techniques), dtype=bool)
        self.shares = np.zeros((capacity, len(MODES)))
        self.works = np.full((capacity, n_minerals), -1, dtype=np.int64)
        self.appearance_mean = np.zeros((capacity, n_appearance))
        self.appearance_var = np.zeros((capacity, n_appearance))
        self.dialect = np.zeros((capacity, n_dialect))
        self.culture = np.zeros((capacity, n_traits))

    def _grow(self) -> None:
        for name in self.FLOAT_COLUMNS + self.INT_COLUMNS + self.MATRIX_COLUMNS:
            old = getattr(self, name)
            fill = 0 if old.dtype.kind in "fb" else -1
            new = np.full((2 * old.shape[0],) + old.shape[1:], fill, dtype=old.dtype)
            new[: old.shape[0]] = old
            setattr(self, name, new)

    def add(self, cell: int, population: float, lineage: int, parent: int, year: int,
            store: float = 0.0, knows: np.ndarray | None = None, strength: np.ndarray | None = None,
            shares: np.ndarray | None = None, practised: np.ndarray | None = None, polity: int = -1) -> int:
        if self.n == len(self.population):
            self._grow()
        sid = self.n
        self.n += 1
        self.cell[sid] = cell
        self.polity[sid] = polity
        self.population[sid] = population
        self.peak_population[sid] = population
        self.store[sid] = store
        self.food_ratio[sid] = 1.0
        self.lineage[sid] = lineage
        self.parent[sid] = parent
        self.founded[sid] = year
        self.ended[sid] = ALIVE
        self.next_split_year[sid] = year
        self.famine_years[sid] = 0
        self.grievance[sid] = 0.0
        self.subject_of[sid] = -1
        self.knows[sid] = False if knows is None else knows
        self.strength[sid] = 0.0 if strength is None else strength
        self.practised[sid] = False if practised is None else practised
        self.works[sid] = -1
        if shares is None:
            self.shares[sid] = 0.0
            self.shares[sid, 0] = 1.0  # foraging
        else:
            self.shares[sid] = shares
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

    @property
    def polity(self) -> int:
        return int(self.model.settlements.polity[self.sid])

    @property
    def techniques(self) -> tuple[str, ...]:
        names = self.model.knowledge.names
        return tuple(names[i] for i in np.flatnonzero(self.model.settlements.knows[self.sid]))
