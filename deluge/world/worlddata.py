"""The common world interface every world source produces (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import numpy as np


class Terrain(IntEnum):
    SEA = 0
    COAST = 1
    PLAIN = 2
    FOREST = 3
    HILLS = 4
    MOUNTAIN = 5
    DESERT = 6
    STEPPE = 7
    WETLAND = 8

    @property
    def key(self) -> str:
        """Lower-case name used for per-terrain tables in config."""
        return self.name.lower()


def terrain_table(table, dtype=float) -> np.ndarray:
    """Per-terrain config mapping -> lookup array indexed by Terrain code."""
    missing = [t.key for t in Terrain if t.key not in table]
    if missing:
        raise KeyError(f"terrain table missing {missing}")
    return np.array([table[t.key] for t in Terrain], dtype=dtype)


@dataclass(frozen=True)
class WorldData:
    """Static world fields, each a read-only array of shape (width, height).

    ``base_temp`` and ``base_rain`` are the baseline climate; the year's temperature is
    ``base_temp + temp_anomaly(year)`` and rainfall is ``base_rain`` times the year's
    drought factor (``world/climate.py``). ``terrain`` is classified at the year-0 sea
    level; the sea itself is re-derived every year from ``elevation < sea_level(year)``.
    """

    width: int
    height: int
    elevation: np.ndarray  # metres, relative to present sea level
    terrain: np.ndarray  # Terrain codes
    river: np.ndarray  # flow accumulation, 0 = no river
    base_temp: np.ndarray  # mean annual degC, baseline climate
    base_rain: np.ndarray  # mm/year, baseline climate
    fertility: np.ndarray  # 0..1 soil quality
    wild_food: np.ndarray  # 0..1 forage richness

    def __post_init__(self):
        for name in self.field_names():
            arr = getattr(self, name)
            if arr.shape != (self.width, self.height):
                raise ValueError(f"{name}: shape {arr.shape} != {(self.width, self.height)}")
            arr.setflags(write=False)

    @staticmethod
    def field_names() -> tuple[str, ...]:
        return tuple(f.name for f in fields(WorldData) if f.name not in ("width", "height"))

    def fields(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.field_names()}
