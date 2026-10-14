"""DelugeModel: the yearly step loop.

One step is one year. Each yearly phase (spec section 4) is a function ``run(model)``
in ``deluge/systems/``; they are appended to ``PHASES`` in fixed order as rounds add them:

    1 climate, 2 capacity, 3 subsistence and demography, 4 migration and settlement,
    5 knowledge, 6 brains decide, 7 diplomacy and war, 8 cohesion and politics,
    9 memory, 10 data collection.

Phase 3 starts by moving bands whose cell the sea took this year (they move before they
eat), then runs subsistence and demography.

``model.year`` is the year being simulated: a step runs every phase for that year,
collects data, then advances the year.

Two seeds: ``world_seed`` draws the map (the "world" stream), ``history_seed`` every
other stream, Mesa's own included. The same world can so host many histories.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mesa
import numpy as np
from mesa.discrete_space import PropertyLayer

from deluge.agents.founders import landmass_labels, place_founders
from deluge.agents.lineage import LineageRegistry
from deluge.agents.settlement import MODES, Settlement, SettlementTable
from deluge.agents.teachers import TeachersState
from deluge.config import DEFAULT_CONFIG_PATH, Config, load_config
from deluge.data.collectors import make_datacollector
from deluge.data.events import EventLog, entity
from deluge.rng import RngStreams
from deluge.systems import capacity, climate, demography, knowledge, subsistence
from deluge.systems.knowledge import KnowledgeGraph, usable_mask
from deluge.world import generate_world
from deluge.world.capacity import CarryingCapacity
from deluge.world.climate import ClimateShocks, sea_level_curve, temp_anomaly_series
from deluge.world.grid import HexGeometry, as_field, build_mesa_grid, hex_area_km2, hex_unit_km

PHASES: tuple[Callable[[DelugeModel], None], ...] = (
    climate.run,
    capacity.run,
    demography.relocate_drowned,
    subsistence.run,
    demography.grow,
    demography.migrate,
    knowledge.run,
)

YEAR_STATS = ("famine_deaths", "moves", "foundings", "dissolved", "drowned_lost", "techniques_learned",
              "techniques_lost", "losses_after_crash", "latent_expired")


class DelugeModel(mesa.Model):
    """Inputs are keyword arguments so the browser UI can instantiate the model.

    ``config`` takes precedence over ``config_path``; ``world_seed``, ``history_seed``
    and ``years`` override the config's values when given.
    """

    def __init__(
        self,
        *,
        config: Config | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        world_seed: int | None = None,
        history_seed: int | None = None,
        years: int | None = None,
    ) -> None:
        cfg = config if config is not None else load_config(config_path)
        run_overrides = {k: v for k, v in (("world_seed", world_seed), ("history_seed", history_seed),
                                           ("years", years)) if v is not None}
        if run_overrides:
            cfg = cfg.with_overrides({"run": run_overrides})

        streams = RngStreams(cfg.run.history_seed, cfg.rng.streams)
        world_stream = RngStreams(cfg.run.world_seed, cfg.rng.streams)["world"]
        # Mesa's model.rng / model.random draw from their own child stream of the history seed.
        super().__init__(rng=streams["mesa"])

        self.config = cfg
        self.streams = streams
        self.year = 0

        # World (static).
        self.geom = HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)
        self.unit_km = hex_unit_km(cfg.grid.height, cfg.world.lat_south, cfg.world.lat_north,
                                   cfg.world.km_per_degree)
        self.cell_area_km2 = hex_area_km2(self.unit_km)
        self.world = generate_world(cfg, world_stream)
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
        # The sea flag as a grid PropertyLayer (the browser map draws it); the climate
        # phase updates its data in place, never replaces it.
        self.sea_layer = PropertyLayer.from_data("sea", as_field(self.geom, self.is_sea))
        self.grid.add_property_layer(self.sea_layer)
        self.temp_anomaly = 0.0
        self.temperature = self.base_temp_flat.copy()
        self.rainfall = self.base_rain_flat.copy()
        self.capacity: dict = {}
        self.land_cells = 0
        self.drought_cells = 0
        self.k_forage = self.k_herd = self.k_farm = 0.0

        # People and knowledge.
        self.knowledge = KnowledgeGraph.from_config(cfg.knowledge)
        self.events = EventLog(cfg.events.min_magnitude)
        self.settlements = SettlementTable(len(self.knowledge.names))
        self.settlement_agents: dict[int, Settlement] = {}
        self.occupant = np.full(self.geom.n_cells, -1, dtype=np.int64)  # settlement id per cell
        self.forage_health = np.ones(self.geom.n_cells)  # 1 = forage untouched by overuse
        self.lineages = LineageRegistry()
        self.stats = dict.fromkeys(YEAR_STATS, 0.0)
        n_tech = len(self.knowledge.names)
        self.first_known_year = np.full(n_tech, -1)  # first year any settlement held each technique
        self.first_invented_year = np.full(n_tech, -1)  # first independent invention
        self.first_farming_year = np.full(len(cfg.founders.names), -1)  # per group, farming >= use_share
        self._place_founders()
        self.lineage_openness = np.array([f.culture["openness"] for f in self.lineages.founders])
        self.teachers_state = TeachersState(cfg.teachers, len(self.lineages.founders), self.knowledge.index,
                                            streams["teachers"])
        self.population_layer = PropertyLayer.from_data("population", as_field(self.geom, self.cell_population()))
        self.grid.add_property_layer(self.population_layer)
        self.knowledge_layer = PropertyLayer.from_data("knowledge", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.knowledge_layer)
        self.latent_layer = PropertyLayer.from_data("latent_knowledge", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.latent_layer)
        self._update_people_summary()

        self.datacollector = make_datacollector(self.lineages.names)

    # --- settlements -----------------------------------------------------------------------

    def _place_founders(self) -> None:
        """The five founding bands, on cells chosen by year-0 foraging capacity."""
        cfg = self.config
        temp0 = self.base_temp_flat + float(self.temp_anomaly_series[0])
        k0 = self.carrying_capacity.compute(temp0, self.base_rain_flat, self.is_sea, self.is_coast)["forage"]
        landmass = landmass_labels(self.geom, self.elevation_flat, float(self.sea_level_series[0]),
                                   cfg.world.landmasses)
        self.founder_capacity, self.founder_landmass = k0, landmass  # what placement saw
        rng = self.streams["founders"]
        cells, sizes = place_founders(self.geom, k0, ~self.is_sea, landmass, cfg.founders, rng)
        f = cfg.founders
        for name, cell, size in zip(f.names, cells, sizes):
            culture = {trait: float(rng.uniform(f.culture_min, f.culture_max)) for trait in f.culture_traits}
            sid = self.found_settlement(cell, float(size), lineage=len(self.lineages.founders), parent=-1)
            self.lineages.add_founder(name, sid, cell, size, self.year, culture)

    def found_settlement(self, cell: int, population: float, lineage: int, parent: int,
                         store: float = 0.0, knows=None, strength=None, shares=None, practised=None) -> int:
        if self.occupant[cell] >= 0 or self.is_sea[cell]:
            raise ValueError(f"cell {cell} is not free land")
        sid = self.settlements.add(cell, population, lineage, parent, self.year, store, knows, strength, shares,
                                   practised)
        self.occupant[cell] = sid
        self.settlement_agents[sid] = Settlement(self, sid, self.grid[self.geom.xy(cell)])
        return sid

    def move_settlement(self, sid: int, cell: int) -> None:
        if self.occupant[cell] >= 0 or self.is_sea[cell]:
            raise ValueError(f"cell {cell} is not free land")
        self.occupant[self.settlements.cell[sid]] = -1
        self.occupant[cell] = sid
        self.settlements.cell[sid] = cell
        self.settlement_agents[sid].cell = self.grid[self.geom.xy(cell)]

    def end_settlement(self, sid: int, cause: str, host: int = -1) -> None:
        """Dissolve a settlement; its remaining people are counted as lost unless the
        caller moved them elsewhere first (``host``: the band it joined)."""
        t = self.settlements
        if cause == "dissolved":
            self.stats["dissolved"] += 1
        if cause in ("dissolved", "drowned"):
            self.log_event("died", [("settlement", sid)], int(t.cell[sid]), float(t.population[sid]))
        self.teachers_state.settlement_ended(sid, host)
        self.occupant[t.cell[sid]] = -1
        t.ended[sid] = self.year
        t.population[sid] = 0.0
        t.store[sid] = 0.0
        self.settlement_agents.pop(sid).remove()

    def cell_population(self) -> np.ndarray:
        """People per cell (flat)."""
        t = self.settlements
        ids = t.alive_ids()
        out = np.zeros(self.geom.n_cells)
        out[t.cell[ids]] = t.population[ids]
        return out

    # --- events and knowledge bookkeeping ---------------------------------------------------

    def log_event(self, action: str, actors, place: int, magnitude: float) -> bool:
        """Log a raw event; ``actors`` are (kind, local id) pairs."""
        return self.events.log(self.year, action, [entity(kind, i) for kind, i in actors], place, magnitude)

    def note_technique_gained(self, technique: int, year: int, invented: bool) -> None:
        self.stats["techniques_learned"] += 1
        if self.first_known_year[technique] < 0:
            self.first_known_year[technique] = year
        if invented and self.first_invented_year[technique] < 0:
            self.first_invented_year[technique] = year

    def note_technique_lost(self, technique: int, after_crash: bool) -> None:
        self.stats["techniques_lost"] += 1
        self.stats["losses_after_crash"] += after_crash

    def _update_people_summary(self) -> None:
        t = self.settlements
        ids = t.alive_ids()
        pop = t.population[ids]
        shares = t.shares[ids]
        self.population = float(pop.sum())
        self.n_settlements = len(ids)
        self.lineage_population = self.lineages.population(t)
        self.mode_population = pop @ shares  # people working each mode
        farming = shares[:, MODES.index("farm")] >= self.config.knowledge.use_share
        self.farming_settlements = int(farming.sum())
        for lineage in np.unique(t.lineage[ids[farming]]).tolist():
            if self.first_farming_year[lineage] < 0:
                self.first_farming_year[lineage] = self.year
        self.techniques_known = float(t.knows[ids].sum(axis=1).mean()) if len(ids) else 0.0
        self.population_layer.data[...] = as_field(self.geom, self.cell_population())
        # Techniques per cell: usable where the settlement is, and latent (held, unusable).
        known = np.zeros(self.geom.n_cells)
        latent = np.zeros(self.geom.n_cells)
        if len(ids) and self.capacity:
            usable = usable_mask(self, ids)
            known[t.cell[ids]] = (t.knows[ids] & usable).sum(axis=1)
            latent[t.cell[ids]] = (t.knows[ids] & ~usable).sum(axis=1)
        self.knowledge_layer.data[...] = as_field(self.geom, known)
        self.latent_layer.data[...] = as_field(self.geom, latent)

    # --- stepping --------------------------------------------------------------------------

    def step(self) -> None:
        self.stats = dict.fromkeys(YEAR_STATS, 0.0)
        for phase in PHASES:
            phase(self)
        self._update_people_summary()
        self.datacollector.collect(self)
        self.year += 1
        if self.year >= self.config.run.years:
            self.running = False

    def run(self) -> None:
        """Step until the configured number of years has elapsed."""
        while self.running:
            self.step()
