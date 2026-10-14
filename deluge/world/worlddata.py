"""The common world interface every world source produces (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum

import numpy as np


MINERALS = ("copper", "tin", "iron_ore", "coal", "oil")  # deposit kinds, WorldData field names


class Terrain(IntEnum):
    """Terrain classes. Coast is not a class: it is a per-year flag (land next to the
    year's sea), so it moves with the sea level."""

    SEA = 0
    PLAIN = 1
    FOREST = 2
    HILLS = 3
    MOUNTAIN = 4
    DESERT = 5
    STEPPE = 6
    WETLAND = 7
    TUNDRA = 8
    ICE = 9

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
    level; the sea itself is re-derived every year from ``elevation < sea_level(year)``,
    and so is the coast flag.
    """

    width: int
    height: int
    elevation: np.ndarray  # metres, relative to present sea level
    terrain: np.ndarray  # Terrain codes
    river: np.ndarray  # flow accumulation, 0 = no river
    river_order: np.ndarray  # Strahler stream order of river cells, 0 = no river
    flow_to: np.ndarray  # flat id of the downstream cell, -1 for sea cells (year-0 sea)
    base_temp: np.ndarray  # mean annual degC, baseline climate
    base_rain: np.ndarray  # mm/year, baseline climate
    fertility: np.ndarray  # 0..1 soil quality
    wild_food: np.ndarray  # 0..1 forage richness
    # Mineral deposits (``MINERALS``): abundance per cell, 0 = no deposit.
    copper: np.ndarray
    tin: np.ndarray
    iron_ore: np.ndarray
    coal: np.ndarray
    oil: np.ndarray

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

    def deposits(self) -> np.ndarray:
        """(cells, minerals) abundance, flat cell ids, column order ``MINERALS``."""
        return np.stack([getattr(self, m).ravel() for m in MINERALS], axis=1)


@dataclass(frozen=True)
class GenerationReport:
    """How a procedural world was made: diagnostics for reports and tests, not sim state."""

    attempts: int
    rejected: dict  # attempts rejected, by geography rule
    plate_id: np.ndarray  # flat, plate of each cell (-1 when tectonics is off)
    convergent: np.ndarray  # flat bool, convergent boundary cells
    divergent: np.ndarray  # flat bool, divergent boundary cells
    deposit: np.ndarray  # flat, metres of sediment deposited by erosion
    constructed_bridge: tuple[int, ...]  # cell ids of a constructed bridge, () if natural

