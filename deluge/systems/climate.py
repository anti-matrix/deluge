"""Phase 1, climate: the year's sea level, temperature, rainfall and shocks."""

from __future__ import annotations

import numpy as np


def run(model) -> None:
    year = model.year
    model.sea_level = float(model.sea_level_series[year])
    model.is_sea = model.elevation_flat < model.sea_level
    model.shocks.step(year, ~model.is_sea)
    model.temp_anomaly = float(model.temp_anomaly_series[year]) + model.shocks.temp_shock
    model.temperature = model.base_temp_flat + model.temp_anomaly
    model.rainfall = model.base_rain_flat * model.shocks.rain_factor
    model.land_cells = int(model.is_sea.size - np.count_nonzero(model.is_sea))
    model.drought_cells = int(np.count_nonzero(model.shocks.rain_factor < 1.0))
