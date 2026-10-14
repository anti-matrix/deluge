"""Data collection: Mesa DataCollector setup and Parquet table writers.

Round 0 records only the year; later rounds add the tables in spec section 12.
"""

from __future__ import annotations

import mesa


def make_datacollector() -> mesa.DataCollector:
    return mesa.DataCollector(model_reporters={"year": "year"})
