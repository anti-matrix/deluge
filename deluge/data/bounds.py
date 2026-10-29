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
- Growth ceiling (Round 14): the mean yearly growth over any 500-year window before industry,
  of the world and of each lineage (``window_rates``).
- War deaths (Round 14): a battle's dead against the people present (``battle_death_limit``).
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


# --- growth ceiling and war deaths (Round 14) ------------------------------------------------------


def window_rates(pop, window: int, min_people: float = 0.0, until: int | None = None):
    """(rates, starts): the mean yearly growth rate (P(t + w) / P(t))^(1/w) - 1 of each window of
    ``window`` years of the yearly series ``pop`` (index = year) that starts with at least
    ``min_people`` people and ends before ``until`` (None: the series' end)."""
    pop = np.asarray(pop, dtype=float)
    end = len(pop) if until is None else min(len(pop), max(int(until), 0))
    starts = np.arange(0, max(end - window, 0))
    if len(starts) == 0:
        return np.zeros(0), starts
    a, b = pop[starts], pop[starts + window]
    ok = (a >= max(min_people, 1e-9)) & (b > 0)
    rates = np.full(len(starts), np.nan)
    rates[ok] = (b[ok] / a[ok]) ** (1.0 / window) - 1.0
    return rates, starts


def largest_window_rate(pop, window: int, min_people: float = 0.0, until: int | None = None) -> tuple[float, int]:
    """(the largest window rate, the year its window starts); (nan, -1) when no window counts."""
    rates, starts = window_rates(pop, window, min_people, until)
    if len(rates) == 0 or np.isnan(rates).all():
        return float("nan"), -1
    k = int(np.nanargmax(rates))
    return float(rates[k]), int(starts[k])


def battle_death_limit(target_people, present_people, mobilization, fighting_age_share: float):
    """Most dead a battle can have (``war_deaths`` in configs/bounds.yaml): the fighting-age share
    of the target's people plus the attacker's mobilized force within logistics reach."""
    return (fighting_age_share * np.asarray(target_people, dtype=float)
            + np.asarray(present_people, dtype=float) * np.asarray(mobilization, dtype=float))


def near_capacity_rates(samples, near_share: float, window: int, until: int | None = None):
    """The near-capacity growth windows of the settlement samples (columns year, settlement,
    population, capacity): one row per settlement sampled at t and t + ``window`` that held at
    least ``near_share`` of its capacity at t, with its mean yearly growth rate; windows ending
    at or after ``until`` are left out."""
    import pandas as pd

    s = samples[["year", "settlement", "lineage", "population", "capacity"]]
    a = s[(s.capacity > 0) & (s.population >= near_share * s.capacity)]
    b = s[["year", "settlement", "population"]].assign(year=s.year - window)
    w = a.merge(b, on=["year", "settlement"], suffixes=("", "_end"))
    if until is not None:
        w = w[w.year + window < until]
    w = w[(w.population > 0) & (w.population_end > 0)]
    return pd.DataFrame({"year": w.year, "settlement": w.settlement, "lineage": w.lineage,
                         "start": w.population, "end": w.population_end,
                         "rate": (w.population_end / w.population) ** (1.0 / window) - 1.0})


def frontier_speeds(frontier, window: int):
    """Per lineage and sampled year t: the frontier's mean advance (km a year) from t to
    t + ``window`` (columns lineage, year, km_per_year), from frontier.parquet."""
    import pandas as pd

    f = frontier[["year", "lineage", "frontier_km"]].dropna()
    e = f.assign(year=f.year - window)
    w = f.merge(e, on=["year", "lineage"], suffixes=("", "_end"))
    return pd.DataFrame({"lineage": w.lineage, "year": w.year, "from_km": w.frontier_km, "to_km": w.frontier_km_end,
                         "km_per_year": (w.frontier_km_end - w.frontier_km) / window})
