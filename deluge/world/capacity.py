"""Carrying capacity per cell for each subsistence mode (spec section 5).

    K_mode = A_cell * Y_mode(fertility, wild_food, temp, rain) * T_mode(techniques)

The yield's static part (soil, forage richness, terrain) is precomputed once; each year
it is scaled by temperature and water suitability ramps, and foraging gains a bonus on
cells next to the year's sea. Densities in config are people per km2 and are
placeholders until calibration (round 12); the cell area comes from the grid geometry.
``T_mode`` is 1 until knowledge exists (round 4); callers pass per-mode multiplier arrays
then.

Only cells that can ever be land (``cells``) are computed; every other cell is sea all
run long and has zero capacity.
"""

from __future__ import annotations

import numpy as np

from deluge.config import CapacityConfig
from deluge.world.procedural import ramp
from deluge.world.worlddata import Terrain, WorldData, terrain_table

MODES = ("forage", "herd", "farm")


class CarryingCapacity:
    def __init__(self, cfg: CapacityConfig, world: WorldData, cell_area_km2: float,
                 cells: np.ndarray | None = None):
        self.cfg = cfg
        self.area = cell_area_km2
        self.n = world.width * world.height
        self.cells = np.arange(self.n) if cells is None else np.asarray(cells)
        sel = self.cells
        terrain = world.terrain.ravel()[sel]
        f, h, m = cfg.forage, cfg.herd, cfg.farm
        self.forage_suit = terrain_table(f.terrain)[terrain]
        self.wild = world.wild_food.ravel()[sel]
        fert = world.fertility.ravel()[sel]
        self.static_herd = cell_area_km2 * h.density_scale * terrain_table(h.terrain)[terrain] * (
            h.min_density + (h.max_density - h.min_density) * fert)
        self.static_farm = cell_area_km2 * m.density_scale * terrain_table(m.terrain)[terrain] * (
            m.min_density + (m.max_density - m.min_density) * fert)
        self.farm_extra_water = m.river_water_mm_per_flow * world.river.ravel()[sel]
        self.not_ice = terrain != Terrain.ICE

    def compute(self, temperature: np.ndarray, rainfall: np.ndarray, is_sea: np.ndarray,
                is_coast: np.ndarray, tech: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """People each cell can support this year under each mode (flat arrays over all
        cells; ``tech`` multipliers are flat arrays over all cells too)."""
        f, h, m = self.cfg.forage, self.cfg.herd, self.cfg.farm
        sel = self.cells
        t, r = temperature[sel], rainfall[sel]
        habitable = ~is_sea[sel] & self.not_ice
        wild = np.minimum(1.0, self.wild + f.coast_bonus * is_coast[sel])
        forage = self.area * f.density_scale * self.forage_suit * (
            f.min_density + (f.max_density - f.min_density) * wild)
        local = {
            "forage": forage * ramp(t, f.temp_zero_c, f.temp_full_c) * ramp(r, f.rain_zero_mm, f.rain_full_mm),
            "herd": self.static_herd * ramp(t, h.temp_zero_c, h.temp_full_c)
            * ramp(r, h.rain_zero_mm, h.rain_full_mm),
            "farm": self.static_farm * ramp(t, m.temp_zero_c, m.temp_full_c)
            * ramp(r + self.farm_extra_water, m.rain_zero_mm, m.rain_full_mm),
        }
        out = {}
        for mode in MODES:
            k = local[mode] * habitable
            if tech is not None and mode in tech:
                k = k * tech[mode][sel]
            full = np.zeros(self.n)
            full[sel] = k
            out[mode] = full
        return out
