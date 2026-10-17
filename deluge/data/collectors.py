"""Data collection (spec section 12): the Mesa DataCollector (``model.parquet``, every
year) and ``RunRecorder``, which a headless run calls after every step for the periodic
tables:

- ``lineages.parquet`` every ``output.lineages_every_years``: per founding lineage its
  people, settlements, territory (claimed cells), polities, techniques held (distinct, and
  the population-weighted mean per settlement), people farming, and its people-years since
  year 0 in all and on cells in drought (climate luck);
- ``polities.parquet`` every ``output.polities_every_years``: per living polity its lineage,
  parent, founding year, size, cohesion, pressure, culture vector and the mean of each brain
  output over the years since its previous row;
- per technique and lineage, the first year a settlement of the lineage held it
  (``first_held``, the milestone report);
- ``ceilings.parquet`` every year (Round 12): the largest density over ceiling of a
  settlement (density ceilings of ``configs/bounds.yaml``), and the world's people against the sum of
  cell ceilings (``data/bounds.py``).

The recorder only reads the model. It is pickled with the model in checkpoints, so a
resumed run continues its tables.
"""

from __future__ import annotations

from functools import partial

import mesa
import numpy as np
import pandas as pd

from deluge.agents.settlement import MODES
from deluge.brains.network import OUTPUTS
from deluge.data.bounds import Ceilings, bounds_path, load_bounds

MODEL_REPORTERS = (
    "year",
    "sea_level",
    "temp_anomaly",
    "land_cells",
    "drought_cells",
    "k_forage",
    "k_herd",
    "k_farm",
    "population",
    "n_settlements",
    "farming_settlements",
    "techniques_known",
    "n_polities",
    "mean_cohesion",
    "largest_polity_settlements",
    "largest_polity_population",
    "wars_active",
)

YEAR_STAT_REPORTERS = ("famine_deaths", "moves", "foundings", "dissolved", "drowned_lost",
                       "techniques_learned", "techniques_lost", "losses_after_crash", "latent_expired",
                       "polities_founded", "secessions", "fragmentations", "mergers", "polity_extinctions",
                       "monuments_built", "battles", "raids", "campaigns", "conquests", "capitals_taken",
                       "polities_destroyed", "war_deaths", "sea_crossings", "sea_battles", "surplus")


def year_stat(name: str, model) -> float:
    return float(model.stats[name])


def lineage_population(index: int, model) -> float:
    return float(model.lineage_population[index])


def mode_population(index: int, model) -> float:
    return float(model.mode_population[index])


def brain_output_mean(index: int, model) -> float:
    return float(model.brain_output_mean[index])


def make_datacollector(lineage_names: tuple[str, ...] = ()) -> mesa.DataCollector:
    """Reporters are module-level functions (bound with partial), not lambdas, so the
    model stays picklable for checkpoints. ``brain_<output>``: the mean of each brain
    output over the living polities at this year's decision."""
    reporters: dict = {name: name for name in MODEL_REPORTERS}
    for name in YEAR_STAT_REPORTERS:
        reporters[name] = partial(year_stat, name)
    for i, name in enumerate(OUTPUTS):
        reporters[f"brain_{name}"] = partial(brain_output_mean, i)
    for i, mode in enumerate(MODES):
        reporters[f"pop_{mode}"] = partial(mode_population, i)
    for i, name in enumerate(lineage_names):
        reporters[f"pop_{name}"] = partial(lineage_population, i)
    return mesa.DataCollector(model_reporters=reporters)


class RunRecorder:
    """The periodic tables of one run (module docstring). Call ``observe(model)`` after
    every step; ``year`` in the rows is the year that step simulated."""

    def __init__(self, model) -> None:
        out = model.config.output
        self.lineages_every = out.lineages_every_years
        self.polities_every = out.polities_every_years
        self.lineage_rows: list[dict] = []
        self.polity_rows: list[dict] = []
        n_tech = len(model.knowledge.names)
        self.first_held = np.full((len(model.lineages.founders), n_tech), -1, dtype=np.int64)
        self.output_sum = np.zeros((0, len(OUTPUTS)))  # per polity id, since its last row
        self.output_years = np.zeros(0, dtype=np.int64)
        n_lin = len(model.lineages.founders)
        self.people_years = np.zeros(n_lin)  # per lineage, summed over the years so far
        self.drought_people_years = np.zeros(n_lin)  # of them, on cells in drought that year
        self.ceilings = Ceilings(load_bounds(bounds_path(model.config)), model.knowledge)
        self.ceiling_rows: list[tuple] = []

    def observe(self, model) -> None:
        year = model.year - 1
        t, P = model.settlements, model.polities
        ids = t.alive_ids()
        lin = t.lineage[ids]
        n_lin = self.first_held.shape[0]
        held = np.zeros(self.first_held.shape, dtype=bool)
        np.logical_or.at(held, lin, t.knows[ids])
        self.first_held[held & (self.first_held < 0)] = year
        pids = P.alive_ids()
        if P.n > len(self.output_years):
            grow = P.n - len(self.output_years)
            self.output_sum = np.vstack([self.output_sum, np.zeros((grow, len(OUTPUTS)))])
            self.output_years = np.concatenate([self.output_years, np.zeros(grow, dtype=np.int64)])
        self.output_sum[pids] += model.brains.output[pids]
        self.output_years[pids] += 1
        pop = t.population[ids]
        dry = model.shocks.rain_factor[t.cell[ids]] < 1.0
        self.people_years += np.bincount(lin, weights=pop, minlength=n_lin)
        self.drought_people_years += np.bincount(lin, weights=pop * dry, minlength=n_lin)
        self._ceilings(model, year, ids, pop)
        if year % self.lineages_every == 0:
            self._lineages(model, year, ids, lin, held, n_lin)
        if year % self.polities_every == 0:
            self._polities(model, year, pids)

    def _ceilings(self, model, year: int, ids, pop) -> None:
        """One row a year: the largest density over ceiling of a settlement (its cell and
        density), the world's people and the sum of cell ceilings (data/bounds.py)."""
        t, area = model.settlements, model.cell_area_km2
        habitable = int(np.count_nonzero(np.maximum.reduce([model.capacity[m] for m in MODES]) > 0))
        if len(ids) == 0:
            self.ceiling_rows.append((year, 0.0, 0.0, -1, 0.0, 0.0))
            return
        knows = t.knows[ids]
        ratio = self.ceilings.ratio(pop, t.shares[ids], knows, area, model.config.knowledge.use_share)
        i = int(np.argmax(ratio))
        world = self.ceilings.world(knows.any(axis=0), habitable, area)
        self.ceiling_rows.append((year, float(pop.sum()), world, int(t.cell[ids[i]]), float(ratio[i]),
                                  float(pop[i] / area)))

    def ceilings_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.ceiling_rows, columns=["year", "population", "world_ceiling", "max_cell",
                                                        "max_ratio", "max_density"])

    def _lineages(self, model, year: int, ids, lin, held, n_lin: int) -> None:
        t, P = model.settlements, model.polities
        pop = t.population[ids]
        claimed = model.polity_layer.data.ravel().astype(np.int64) - 1  # polity per cell, -1 none
        cells = np.bincount(P.lineage[claimed[claimed >= 0]], minlength=n_lin)
        pids = P.alive_ids()
        polities = np.bincount(P.lineage[pids], minlength=n_lin)
        known = t.knows[ids].sum(axis=1)
        people = np.bincount(lin, weights=pop, minlength=n_lin)
        farming = np.bincount(lin, weights=pop * t.shares[ids, MODES.index("farm")], minlength=n_lin)
        settlements = np.bincount(lin, minlength=n_lin)
        mean_known = np.bincount(lin, weights=pop * known, minlength=n_lin) / np.maximum(people, 1e-12)
        for g, name in enumerate(model.lineages.names):
            self.lineage_rows.append({
                "year": year, "lineage": g, "group": name, "population": float(people[g]),
                "settlements": int(settlements[g]), "territory_cells": int(cells[g]), "polities": int(polities[g]),
                "techniques_distinct": int(held[g].sum()), "techniques_mean": float(mean_known[g]),
                "farming_population": float(farming[g]),
                "people_years": float(self.people_years[g]),  # summed from year 0 to this year
                "drought_people_years": float(self.drought_people_years[g]),
            })

    def _polities(self, model, year: int, pids) -> None:
        P = model.polities
        traits = model.config.founders.culture_traits
        n = np.maximum(self.output_years[pids], 1)[:, None]
        means = self.output_sum[pids] / n
        for i, pid in enumerate(pids.tolist()):
            row = {"year": year, "polity": pid, "lineage": int(P.lineage[pid]), "parent": int(P.parent[pid]),
                   "founded": int(P.founded[pid]), "settlements": int(P.n_settlements[pid]),
                   "population": float(P.population[pid]), "cohesion": float(P.cohesion[pid]),
                   "pressure": float(P.pressure[pid]), "elite_ratio": float(P.elite_ratio[pid])}
            row.update({f"culture_{tr}": float(v) for tr, v in zip(traits, P.culture[pid])})
            row.update({f"brain_{o}": float(v) for o, v in zip(OUTPUTS, means[i])})
            self.polity_rows.append(row)
        self.output_sum[pids] = 0.0
        self.output_years[pids] = 0

    def lineages_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.lineage_rows)

    def polities_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.polity_rows)

    def first_held_frame(self, model) -> pd.DataFrame:
        """Per technique: the first year any settlement held it, the first independent
        invention, and the first year per lineage (-1: never)."""
        g = model.knowledge
        df = pd.DataFrame({"technique": g.names, "label": g.labels,
                           "first_held": model.first_known_year.copy(),
                           "first_invented": model.first_invented_year.copy()})
        for i, name in enumerate(model.lineages.names):
            df[f"first_held_{name}"] = self.first_held[i]
        return df
