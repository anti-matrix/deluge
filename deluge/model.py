"""DelugeModel: the yearly step loop.

One step is one year. Each yearly phase (spec section 4) is a function ``run(model)``
in ``deluge/systems/``; they are appended to ``PHASES`` in fixed order as rounds add them:

    1 climate, 2 capacity, 3 subsistence and demography, 4 migration and settlement,
    5 knowledge, 6 brains decide, 7 diplomacy and war, 8 cohesion and politics,
    9 memory, 10 data collection.

Round 0 has no phases: the model only advances the year and collects data.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mesa

from deluge.config import DEFAULT_CONFIG_PATH, Config, load_config
from deluge.data.collectors import make_datacollector
from deluge.rng import RngStreams

PHASES: tuple[Callable[[DelugeModel], None], ...] = ()


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
        self.datacollector = make_datacollector()

    def step(self) -> None:
        for phase in PHASES:
            phase(self)
        self.year += 1
        self.datacollector.collect(self)
        if self.year >= self.config.run.years:
            self.running = False

    def run(self) -> None:
        """Step until the configured number of years has elapsed."""
        while self.running:
            self.step()
