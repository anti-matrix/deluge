"""DelugeModel: the yearly step loop.

One step is one year. Each yearly phase (spec section 4) is a function ``run(model)``
in ``deluge/systems/``; they are appended to ``PHASES`` in fixed order as rounds add them:

    1 climate, 2 capacity, 3 subsistence and demography, 4 migration and settlement,
    5 knowledge, 6 brains decide, 7 diplomacy and war, 8 cohesion and politics,
    9 memory, 10 data collection.

``model.year`` is the year being simulated: a step runs every phase for that year,
collects data, then advances the year.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mesa
import numpy as np

from deluge.config import DEFAULT_CONFIG_PATH, Config, load_config
from deluge.data.collectors import make_datacollector
from deluge.rng import RngStreams
from deluge.systems import capacity, climate
from deluge.world import generate_world
from deluge.world.capacity import CarryingCapacity
from deluge.world.climate import ClimateShocks, sea_level_curve, temp_anomaly_series
from deluge.world.grid import HexGeometry, build_mesa_grid, hex_area_km2, hex_unit_km

PHASES: tuple[Callable[[DelugeModel], None], ...] = (climate.run, capacity.run)


class DelugeModel(mesa.Model):
    """Inputs are keyword arguments so SolaraViz can instantiate the model.

    ``config`` takes precedence over ``config_path``; ``seed`` and ``years`` override
    the config's values when given.
    """

    def __init__(
        self,
        *,
        config: Config | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        seed: int | None = None,
        years: int | None = None,
    ) -> None:
        cfg = config if config is not None else load_config(config_path)
        run_overrides = {}
        if seed is not None:
            run_overrides["seed"] = seed
        if years is not None:
            run_overrides["years"] = years
        if run_overrides:
            cfg = cfg.with_overrides({"run": run_overrides})

        streams = RngStreams(cfg.run.seed, cfg.rng.streams)
        # Mesa's model.rng / model.random draw from their own child stream of the root seed.
        super().__init__(rng=streams["mesa"])

        self.config = cfg
        self.streams = streams
        self.year = 0

        # World (static).
        self.geom = HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)
        self.unit_km = hex_unit_km(cfg.grid.height, cfg.world.lat_south, cfg.world.lat_north,
                                   cfg.world.km_per_degree)
        self.cell_area_km2 = hex_area_km2(self.unit_km)
        self.world = generate_world(cfg, streams["world"])
        self.grid = build_mesa_grid(self.geom, self.world.fields(), random=self.random)
        self.elevation_flat = self.world.elevation.ravel()
        self.base_temp_flat = self.world.base_temp.ravel()
        self.base_rain_flat = self.world.base_rain.ravel()

        # Climate over time.
        self.sea_level_series = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
        self.temp_anomaly_series = temp_anomaly_series(
            cfg.climate.temperature, cfg.run.years, streams["climate"]
        )
        self.shocks = ClimateShocks(cfg.climate.shocks, self.geom.positions * self.unit_km, streams["shocks"])
        # Cells that are land at some point of the run; all others stay sea throughout.
        self.ever_land = np.flatnonzero(self.elevation_flat >= self.sea_level_series.min())
        self.ever_land_neighbors = self.geom.neighbors[self.ever_land]
        self.carrying_capacity = CarryingCapacity(cfg.capacity, self.world, self.cell_area_km2,
                                                  self.ever_land)

        # Per-year state, filled by the phases.
        self.sea_level = float("nan")  # set by the climate phase
        self.is_sea = self.elevation_flat < float(self.sea_level_series[0])
        self.is_coast = ~self.is_sea & self.geom.neighbor_any(self.is_sea)
        self.temp_anomaly = 0.0
        self.temperature = self.base_temp_flat.copy()
        self.rainfall = self.base_rain_flat.copy()
        self.capacity: dict = {}
        self.land_cells = 0
        self.drought_cells = 0
        self.k_forage = self.k_herd = self.k_farm = 0.0

        self.datacollector = make_datacollector()

    def step(self) -> None:
        for phase in PHASES:
            phase(self)
        self.datacollector.collect(self)
        self.year += 1
        if self.year >= self.config.run.years:
            self.running = False

    def run(self) -> None:
        """Step until the configured number of years has elapsed."""
        while self.running:
            self.step()
