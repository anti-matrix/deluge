"""Phase 1, climate: the year's sea level, coast, temperature, rainfall and shocks."""

from __future__ import annotations

import numpy as np


def run(model) -> None:
    year = model.year
    level = float(model.sea_level_series[year])
    if level != model.sea_level:  # sea and coast only change while the sea moves
        model.sea_level = level
        model.is_sea = model.elevation_flat < level
        model.sea_layer.data[...] = model.is_sea.reshape(model.sea_layer.data.shape)
        # Coast: land next to sea, evaluated only where land can exist.
        cells = model.ever_land
        padded = np.append(model.is_sea, False)
        coast = np.zeros_like(model.is_sea)
        coast[cells] = ~model.is_sea[cells] & padded[model.ever_land_neighbors].any(axis=1)
        model.is_coast = coast
        model.land_cells = int(model.is_sea.size - np.count_nonzero(model.is_sea))
    model.shocks.step(year, ~model.is_sea)
    model.temp_anomaly = float(model.temp_anomaly_series[year]) + model.shocks.temp_shock
    model.temperature = model.base_temp_flat + model.temp_anomaly
    model.rainfall = model.base_rain_flat * model.shocks.rain_factor
    model.drought_cells = int(np.count_nonzero(model.shocks.rain_factor < 1.0))
