"""Carrying capacity per cell for each subsistence mode (spec section 5).

    K_mode = A_cell * Y_mode(fertility, wild_food, temp, rain) * T_mode(techniques)

The yield's static part (soil, forage richness, terrain) is precomputed once; each year
it is scaled by temperature and water suitability ramps. Densities in config are people
per km2 and are placeholders until calibration (round 12). ``T_mode`` is 1 until
knowledge exists (round 4); callers pass per-mode multiplier arrays then.
"""

from __future__ import annotations

import numpy as np

from deluge.config import CapacityConfig
from deluge.world.procedural import ramp
from deluge.world.worlddata import Terrain, WorldData, terrain_table

MODES = ("forage", "herd", "farm")


class CarryingCapacity:
    def __init__(self, cfg: CapacityConfig, world: WorldData):
        self.cfg = cfg
        terrain = world.terrain.ravel()
        land = terrain != Terrain.SEA
        area = cfg.cell_area_km2
        f, h, m = cfg.forage, cfg.herd, cfg.farm
        wild = world.wild_food.ravel()
        fert = world.fertility.ravel()
        self.static = {
            "forage": area * (f.min_density + (f.max_density - f.min_density) * wild) * land,
            "herd": area * terrain_table(h.terrain)[terrain]
            * (h.min_density + (h.max_density - h.min_density) * fert),
            "farm": area * terrain_table(m.terrain)[terrain]
            * (m.min_density + (m.max_density - m.min_density) * fert),
        }
        self.farm_extra_water = m.river_water_mm_per_flow * world.river.ravel()

    def compute(self, temperature: np.ndarray, rainfall: np.ndarray, is_sea: np.ndarray,
                tech: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """People each cell can support this year under each mode (flat arrays)."""
        f, h, m = self.cfg.forage, self.cfg.herd, self.cfg.farm
        land = ~is_sea
        out = {
            "forage": self.static["forage"] * ramp(temperature, f.temp_zero_c, f.temp_full_c)
            * ramp(rainfall, f.rain_zero_mm, f.rain_full_mm),
            "herd": self.static["herd"] * ramp(temperature, h.temp_zero_c, h.temp_full_c)
            * ramp(rainfall, h.rain_zero_mm, h.rain_full_mm),
            "farm": self.static["farm"] * ramp(temperature, m.temp_zero_c, m.temp_full_c)
            * ramp(rainfall + self.farm_extra_water, m.rain_zero_mm, m.rain_full_mm),
        }
        for mode in MODES:
            k = out[mode] * land
            if tech is not None and mode in tech:
                k = k * tech[mode]
            out[mode] = k
        return out
