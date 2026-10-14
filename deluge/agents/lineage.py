"""Lineage registry: bookkeeping of descent, not a Mesa agent.

Every settlement carries a lineage id (its founding group) and a parent id (the
settlement it split from, -1 for a founding band), so descent is a tree that can be
rebuilt at any year from the settlement table. The registry holds the founders; the
teachers are added as a lineage in Round 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deluge.agents.settlement import ALIVE, SettlementTable


@dataclass(frozen=True)
class Founder:
    lineage: int
    name: str
    settlement: int  # id of the founding band
    cell: int
    population: int
    year: int


class LineageRegistry:
    def __init__(self) -> None:
        self.founders: list[Founder] = []

    def add_founder(self, name: str, settlement: int, cell: int, population: int, year: int) -> int:
        lineage = len(self.founders)
        self.founders.append(Founder(lineage, name, settlement, cell, population, year))
        return lineage

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.founders)

    def population(self, table: SettlementTable) -> np.ndarray:
        """Living population of each lineage."""
        ids = table.alive_ids()
        return np.bincount(table.lineage[ids], weights=table.population[ids], minlength=len(self.founders))

    def settlement_counts(self, table: SettlementTable) -> np.ndarray:
        ids = table.alive_ids()
        return np.bincount(table.lineage[ids], minlength=len(self.founders))

    def tree(self, table: SettlementTable, year: int | None = None) -> list[tuple[int, int, int, int, int]]:
        """Edges (settlement, parent, lineage, founded, ended) of every settlement founded
        by ``year`` (all if None); ended is -1 for settlements alive then."""
        n = table.n
        rows = []
        for sid in range(n):
            founded = int(table.founded[sid])
            if year is not None and founded > year:
                continue
            ended = int(table.ended[sid])
            if year is not None and ended != ALIVE and ended > year:
                ended = ALIVE
            rows.append((sid, int(table.parent[sid]), int(table.lineage[sid]), founded, ended))
        return rows

    def ancestry(self, table: SettlementTable, sid: int) -> list[int]:
        """Settlement ids from ``sid`` back to its founding band."""
        chain = [sid]
        while table.parent[chain[-1]] >= 0:
            chain.append(int(table.parent[chain[-1]]))
        return chain
