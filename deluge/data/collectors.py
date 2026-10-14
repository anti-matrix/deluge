"""Data collection: Mesa DataCollector setup and Parquet table writers.

Later rounds add the remaining tables in spec section 12.
"""

from __future__ import annotations

import mesa

MODEL_REPORTERS = (
    "year",
    "sea_level",
    "temp_anomaly",
    "land_cells",
    "drought_cells",
    "k_forage",
    "k_herd",
    "k_farm",
)


def make_datacollector() -> mesa.DataCollector:
    return mesa.DataCollector(model_reporters={name: name for name in MODEL_REPORTERS})
