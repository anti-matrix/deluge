"""Physical bounds for calibration (Round 12): population density ceilings per subsistence
mode, from ``configs/bounds.yaml`` (``output.bounds_file``, relative to the default config).

Read-only: nothing here enters the simulation. The run recorder (``data/collectors.py``)
measures each year against the ceilings; ``scripts/acceptance/bounds.py`` checks a batch.

- A settlement's ceiling: the highest ceiling of the modes it practises (at least
  ``knowledge.use_share`` of its people). A mode's ceiling is its ``base`` or the highest
  ``raised_by`` value of a technique the settlement holds. Density over ceiling above 1:
  more people on the cell than that mode ever fed on Earth.
- World ceiling: the habitable cells' area x the highest ceiling of any mode available and
  technique held anywhere that year.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from deluge.agents.settlement import MODES
from deluge.config import DEFAULT_CONFIG_PATH


def bounds_path(cfg) -> Path:
    return Path(DEFAULT_CONFIG_PATH).parent / cfg.output.bounds_file


def load_bounds(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Ceilings:
    """Density ceilings per mode (people per km2), compiled against the knowledge graph."""

    def __init__(self, bounds: dict, graph) -> None:
        dc = bounds["density_ceilings"]
        self.base = np.array([float(dc[m]["base"]) for m in MODES])
        self.raised = np.zeros((len(graph.names), len(MODES)))  # (T, modes), 0 where none
        for m, mode in enumerate(MODES):
            for tech, value in (dc[mode].get("raised_by") or {}).items():
                self.raised[graph.index(tech), m] = float(value)
        self.graph = graph

    def per_mode(self, knows: np.ndarray) -> np.ndarray:
        """(S, modes) ceiling for settlements holding ``knows`` (S, T)."""
        raised = np.where(knows[:, :, None], self.raised[None, :, :], 0.0).max(axis=1)
        return np.maximum(self.base[None, :], raised)

    def settlement(self, shares: np.ndarray, knows: np.ndarray, use_share: float) -> np.ndarray:
        """(S,) each settlement's ceiling: the highest of the modes it practises (at least
        ``use_share`` of its people; foraging when none reaches it)."""
        practised = shares >= use_share
        practised[~practised.any(axis=1), 0] = True
        return np.where(practised, self.per_mode(knows), 0.0).max(axis=1)

    def ratio(self, population: np.ndarray, shares: np.ndarray, knows: np.ndarray, area_km2: float,
              use_share: float) -> np.ndarray:
        """(S,) each settlement's density over its ceiling."""
        return population / area_km2 / self.settlement(shares, knows, use_share)

    def world(self, knows_anywhere: np.ndarray, habitable_cells: int, area_km2: float) -> float:
        """Most people the habitable cells could hold with the modes and techniques known anywhere."""
        k = knows_anywhere[None, :]
        available = self.graph.mode_available(k)[0]
        return float(habitable_cells * area_km2 * self.per_mode(k)[0][available].max())
