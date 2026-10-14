"""Placement of the five founding forager groups at year 0.

Each group starts as one band on a land cell in the top share of year-0 foraging
capacity, at least ``min_distance_cells`` hex steps from every other group, with the
groups spread over at least ``min_landmasses`` landmasses. Cells and sizes are drawn from
the history seed's "founders" stream; cells can be pinned in config.
"""

from __future__ import annotations

import numpy as np

from deluge.config import FoundersConfig, LandmassConfig
from deluge.world import geography
from deluge.world.grid import HexGeometry


class PlacementError(RuntimeError):
    pass


def candidate_cells(capacity: np.ndarray, land: np.ndarray, top_share: float) -> np.ndarray:
    """Land cells whose capacity is in the top ``top_share`` of land cells."""
    land_ids = np.flatnonzero(land & (capacity > 0))
    k = capacity[land_ids]
    cut = np.quantile(k, 1.0 - top_share)
    return land_ids[k >= cut]


def landmass_labels(geom: HexGeometry, elev: np.ndarray, sea_level: float, cfg: LandmassConfig) -> np.ndarray:
    """Landmass index per cell (-1 for sea and for islands below the landmass size floor)."""
    rep = geography.landmass_report(geom, elev, sea_level, cfg)
    labels = rep.labels.copy()
    labels[labels >= len(rep.landmasses)] = -1
    return labels


def place_founders(geom: HexGeometry, capacity: np.ndarray, land: np.ndarray, landmass: np.ndarray,
                   cfg: FoundersConfig, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    """(cells, populations) of the founding groups, in the order of ``cfg.names``."""
    n = len(cfg.names)
    sizes = rng.integers(cfg.population_min, cfg.population_max + 1, size=n).tolist()
    if cfg.pinned_cells:
        if len(cfg.pinned_cells) != n:
            raise PlacementError(f"founders.pinned_cells needs {n} cells, got {len(cfg.pinned_cells)}")
        cells = [geom.cid(x, y) for x, y in cfg.pinned_cells]
        if not all(land[c] and capacity[c] > 0 for c in cells):
            raise PlacementError("a pinned founder cell is not habitable land at year 0")
        return cells, sizes
    cand = candidate_cells(capacity, land, cfg.capacity_top_share)
    cand = cand[landmass[cand] >= 0]  # on a landmass, not a small island
    for _ in range(cfg.max_attempts):
        chosen: list[int] = []
        pool = cand
        for _ in range(n):
            if len(pool) == 0:
                break
            c = int(rng.choice(pool))
            chosen.append(c)
            pool = pool[geom.hex_distance(pool, c) >= cfg.min_distance_cells]
        if len(chosen) == n and len(set(landmass[chosen].tolist())) >= cfg.min_landmasses:
            return chosen, sizes
    raise PlacementError(f"no founder placement met the rules in {cfg.max_attempts} attempts")
