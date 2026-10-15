"""Data collection: Mesa DataCollector setup and Parquet table writers.

Later rounds add the remaining tables in spec section 12.
"""

from __future__ import annotations

from functools import partial

import mesa

from deluge.agents.settlement import MODES

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
)

YEAR_STAT_REPORTERS = ("famine_deaths", "moves", "foundings", "dissolved", "drowned_lost",
                       "techniques_learned", "techniques_lost", "losses_after_crash", "latent_expired",
                       "polities_founded", "secessions", "fragmentations", "mergers", "polity_extinctions",
                       "monuments_built")


def year_stat(name: str, model) -> float:
    return float(model.stats[name])


def lineage_population(index: int, model) -> float:
    return float(model.lineage_population[index])


def mode_population(index: int, model) -> float:
    return float(model.mode_population[index])


def make_datacollector(lineage_names: tuple[str, ...] = ()) -> mesa.DataCollector:
    """Reporters are module-level functions (bound with partial), not lambdas, so the
    model stays picklable for checkpoints."""
    reporters: dict = {name: name for name in MODEL_REPORTERS}
    for name in YEAR_STAT_REPORTERS:
        reporters[name] = partial(year_stat, name)
    for i, mode in enumerate(MODES):
        reporters[f"pop_{mode}"] = partial(mode_population, i)
    for i, name in enumerate(lineage_names):
        reporters[f"pop_{name}"] = partial(lineage_population, i)
    return mesa.DataCollector(model_reporters=reporters)
