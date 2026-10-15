"""Lineage registry: bookkeeping of descent, not a Mesa agent.

Every settlement carries a lineage id (its founding group) and a parent id (the
settlement it split from, -1 for a founding band), so descent is a tree that can be
rebuilt at any year from the settlement table. The registry holds the five founding
groups with their culture vectors; the teachers, the sixth founder, leave no settlements
(their visits are kept in ``model.teachers_state``). Polities form a second tree: each
records its parent polity, back to the founding polity of its group (``polity_tree``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from deluge.agents.polity import END_CAUSES
from deluge.agents.settlement import ALIVE, SettlementTable


@dataclass(frozen=True)
class Founder:
    lineage: int
    name: str
    settlement: int  # id of the founding band
    cell: int
    population: int
    year: int
    culture: dict  # trait -> value in [0, 1], drawn at year 0 (spec section 2)


class LineageRegistry:
    def __init__(self) -> None:
        self.founders: list[Founder] = []

    def add_founder(self, name: str, settlement: int, cell: int, population: int, year: int,
                    culture: dict) -> int:
        lineage = len(self.founders)
        self.founders.append(Founder(lineage, name, settlement, cell, population, year, culture))
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

    # --- polities ----------------------------------------------------------------------------

    @staticmethod
    def polity_ancestry(polities, pid: int) -> list[int]:
        """Polity ids from ``pid`` back to its founding polity (parent -1)."""
        chain = [pid]
        while polities.parent[chain[-1]] >= 0:
            chain.append(int(polities.parent[chain[-1]]))
        return chain

    def polity_tree(self, model) -> pd.DataFrame:
        """The polity lineage tree: one row per polity ever formed, with its parent (-1 for
        the five founding polities), founding group, years founded and ended (-1 while
        alive), how it ended, the polity it merged into, and its peak size."""
        P = model.polities
        n = P.n
        return pd.DataFrame({
            "polity": np.arange(n),
            "parent": P.parent[:n].copy(),
            "lineage": P.lineage[:n].copy(),
            "group": [self.founders[int(g)].name for g in P.lineage[:n]],
            "founded": P.founded[:n].copy(),
            "ended": P.ended[:n].copy(),
            "end_cause": [END_CAUSES[int(c)] for c in P.end_cause[:n]],
            "merged_into": P.merged_into[:n].copy(),
            "peak_settlements": P.peak_settlements[:n].copy(),
            "peak_population": P.peak_population[:n].copy(),
        })
