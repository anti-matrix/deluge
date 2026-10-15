"""Polities: Mesa Agents (no cell) whose numeric state lives in NumPy arrays.

A polity is the decision-making unit over a set of settlements (spec section 8). Each
polity is a ``Polity`` Agent; its numbers are rows of a ``PolityTable`` indexed by the
polity id, so the yearly cohesion and politics phases run as array operations. Members
are the living settlements whose ``polity`` column holds the id. Ids are never reused, so
parent ids keep the polity lineage tree valid after a polity ends.

Columns:

- ``lineage``: founding group the polity descends from; ``parent``: the polity it split
  from (-1 for the five founding polities); ``founded`` / ``ended`` year (-1 while alive);
  ``end_cause``: index into ``END_CAUSES``; ``merged_into``: the polity that absorbed it
  (merged) or destroyed it (conquered).
- ``capital``: settlement id; ``culture``: one column per culture trait
  (``founders.culture_traits``), 0..1; ``intermarriage``: the evolvable rate at which its
  settlements exchange people (appearance mixing, ``systems/identity.py``).
- ``cohesion`` (asabiya), ``elite_ratio``, ``pressure`` (structural-demographic),
  ``frontier`` (f), ``treasury`` (rations), ``monument_fund``, ``tribute`` and
  ``spending`` (this year), ``prestige`` (cost of its maintained monuments),
  ``war_recent`` (its battles, decaying: the war term of frontier pressure),
  ``wins_recent`` / ``losses_recent`` (battles won and lost, decaying: brain inputs),
  ``load`` (people over the capacity of its members' cells, last counted by the cohesion phase).
- ``marker``: (polities, ``MARKER_EFFECTS``) evolvable culture parameters weighing a member's
  perceived appearance and dialect distance from the capital (and, for secession, its
  culture distance) in cohesion, grievance, tribute and secession. Zero in the founding
  polities, mutated at splits, free to take either sign (spec "Ancestry, appearance and
  identity").
- ``deposit``: (polities, minerals) a deposit cell in its territory, -1 for none.
- Bookkeeping: ``population``, ``n_settlements`` (this year), their peaks.

The polity's brain lives in the model's ``BrainTable`` (``brains/network.py``), under the
same id.
"""

from __future__ import annotations

import mesa
import numpy as np

ALIVE = -1
END_CAUSES = ("", "extinct", "fragmented", "merged", "conquered")
MARKER_EFFECTS = ("cohesion_appearance", "cohesion_dialect", "grievance_appearance", "grievance_dialect",
                  "tribute_appearance", "tribute_dialect", "secession_appearance", "secession_dialect",
                  "secession_culture")


def marker_index(*names: str) -> list[int]:
    return [MARKER_EFFECTS.index(n) for n in names]


class PolityTable:
    FLOAT_COLUMNS = ("cohesion", "elite_ratio", "pressure", "frontier", "treasury", "monument_fund", "tribute",
                     "spending", "prestige", "intermarriage", "population", "peak_population", "war_recent",
                     "wins_recent", "losses_recent", "load")
    INT_COLUMNS = ("lineage", "parent", "founded", "ended", "end_cause", "merged_into", "capital", "n_settlements",
                   "peak_settlements")
    MATRIX_COLUMNS = ("culture", "deposit", "marker")

    def __init__(self, n_traits: int, n_minerals: int, capacity: int = 64):
        self.n = 0
        for name in self.FLOAT_COLUMNS:
            setattr(self, name, np.zeros(capacity))
        for name in self.INT_COLUMNS:
            setattr(self, name, np.full(capacity, -1, dtype=np.int64))
        self.culture = np.zeros((capacity, n_traits))
        self.deposit = np.full((capacity, n_minerals), -1, dtype=np.int64)
        self.marker = np.zeros((capacity, len(MARKER_EFFECTS)))

    def _grow(self) -> None:
        for name in self.FLOAT_COLUMNS + self.INT_COLUMNS + self.MATRIX_COLUMNS:
            old = getattr(self, name)
            fill = 0 if old.dtype.kind == "f" else -1
            new = np.full((2 * old.shape[0],) + old.shape[1:], fill, dtype=old.dtype)
            new[: old.shape[0]] = old
            setattr(self, name, new)

    def add(self, lineage: int, parent: int, year: int, capital: int, culture: np.ndarray, intermarriage: float,
            cohesion: float, elite_ratio: float, marker: np.ndarray | None = None) -> int:
        if self.n == len(self.cohesion):
            self._grow()
        pid = self.n
        self.n += 1
        for name in self.FLOAT_COLUMNS:
            getattr(self, name)[pid] = 0.0
        self.lineage[pid] = lineage
        self.parent[pid] = parent
        self.founded[pid] = year
        self.ended[pid] = ALIVE
        self.end_cause[pid] = 0
        self.merged_into[pid] = -1
        self.capital[pid] = capital
        self.n_settlements[pid] = 0
        self.peak_settlements[pid] = 0
        self.culture[pid] = culture
        self.intermarriage[pid] = intermarriage
        self.cohesion[pid] = cohesion
        self.elite_ratio[pid] = elite_ratio
        self.deposit[pid] = -1
        self.marker[pid] = 0.0 if marker is None else marker
        return pid

    def alive_ids(self) -> np.ndarray:
        return np.flatnonzero(self.ended[: self.n] == ALIVE)


class Monuments:
    """Every monument built: the settlement (and cell) it stands at, the polity that built
    it, the year, its cost, and whether its upkeep was paid this year."""

    def __init__(self) -> None:
        self.settlement = np.zeros(0, dtype=np.int64)
        self.cell = np.zeros(0, dtype=np.int64)
        self.builder = np.zeros(0, dtype=np.int64)
        self.year = np.zeros(0, dtype=np.int64)
        self.cost = np.zeros(0)
        self.maintained = np.zeros(0, dtype=bool)

    def __len__(self) -> int:
        return len(self.settlement)

    def add(self, settlement: int, cell: int, builder: int, year: int, cost: float) -> int:
        self.settlement = np.append(self.settlement, settlement)
        self.cell = np.append(self.cell, cell)
        self.builder = np.append(self.builder, builder)
        self.year = np.append(self.year, year)
        self.cost = np.append(self.cost, cost)
        self.maintained = np.append(self.maintained, True)
        return len(self.settlement) - 1


class Polity(mesa.Agent):
    """A polity. Its state is read from the model's PolityTable."""

    def __init__(self, model, pid: int):
        super().__init__(model)
        self.pid = pid

    @property
    def lineage(self) -> int:
        return int(self.model.polities.lineage[self.pid])

    @property
    def parent(self) -> int:
        return int(self.model.polities.parent[self.pid])

    @property
    def capital(self) -> int:
        return int(self.model.polities.capital[self.pid])

    @property
    def cohesion(self) -> float:
        return float(self.model.polities.cohesion[self.pid])

    @property
    def culture(self) -> dict[str, float]:
        traits = self.model.config.founders.culture_traits
        return dict(zip(traits, self.model.polities.culture[self.pid].tolist()))

    @property
    def members(self) -> np.ndarray:
        t = self.model.settlements
        ids = t.alive_ids()
        return ids[t.polity[ids] == self.pid]
