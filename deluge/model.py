"""DelugeModel: the yearly step loop.

One step is one year. Each yearly phase (spec section 4) is a function ``run(model)``
in ``deluge/systems/``; they are appended to ``PHASES`` in fixed order as rounds add them:

    1 climate, 2 capacity, 3 subsistence and demography, 4 migration and settlement,
    5 knowledge, 6 brains decide, 7 diplomacy and war, 8 cohesion and politics,
    9 memory, 10 data collection.

Phase 3 starts by moving bands whose cell the sea took this year (they move before they
eat), then runs subsistence and demography. Phase 6 evaluates every polity brain in one
batch (``systems/decide.py``); its outputs steer the year's war, tribute and monuments
and, until the next decision, expansion, migration of the core, openness to newcomers
and subsistence shift. Phase 7 runs war (assimilation of
conquered settlements, then the brains' war choice and battles). Phase 8 runs cohesion
(tribute, monuments, elites, pressure, cohesion), then politics (secession, fragmentation,
mergers, culture drift, territory), then the drift and mixing of appearance and dialect.
Phase 9 writes the year's events into the polities' memory stores and, once per
generation, runs lossy transmission (``systems/memory.py``).

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
from deluge.agents.polity import END_CAUSES, Monuments, Polity, PolityTable
from deluge.agents.settlement import MODES, Settlement, SettlementTable
from deluge.agents.teachers import TeachersState
from deluge.brains import evolution, features, network
from deluge.config import DEFAULT_CONFIG_PATH, Config, load_config
from deluge.data.campaigns import CampaignLog
from deluge.data.collectors import make_datacollector
from deluge.data.events import EventLog, entity
from deluge.gazetteer import OccupancyLog
from deluge.rng import RngStreams
from deluge.systems import (capacity, climate, cohesion, decide, demography, identity, knowledge, memory,
                            politics, subsistence, war)
from deluge.systems.knowledge import KnowledgeGraph, usable_mask
from deluge.world import generate_world
from deluge.world.capacity import CarryingCapacity
from deluge.world.climate import ClimateShocks, sea_level_curve, temp_anomaly_series
from deluge.world.grid import (HexGeometry, as_field, build_mesa_grid, hex_area_km2, hex_unit_km,
                               repair_unpickled_grid)
from deluge.world.worlddata import MINERALS

PHASES: tuple[Callable[[DelugeModel], None], ...] = (
    climate.run,
    capacity.run,
    demography.relocate_drowned,
    subsistence.run,
    demography.grow,
    demography.migrate,
    knowledge.run,
    decide.run,
    war.run,
    cohesion.run,
    politics.run,
    identity.run,
    memory.run,
)

YEAR_STATS = ("famine_deaths", "moves", "foundings", "dissolved", "drowned_lost", "techniques_learned",
              "techniques_lost", "losses_after_crash", "latent_expired", "polities_founded", "secessions",
              "fragmentations", "mergers", "polity_extinctions", "monuments_built", "battles", "raids",
              "campaigns", "conquests", "capitals_taken", "polities_destroyed", "war_deaths", "sea_crossings",
              "sea_battles", "surplus")


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
        # Mineral deposits: (cells, minerals) abundance, 0 = none. Static unless depletion is on.
        self.deposits = self.world.deposits().copy()
        self.resources_layer = PropertyLayer.from_data("resources", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.resources_layer)
        self.update_deposits()
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
        self.memory = memory.MemoryState(self)
        self.settlements = SettlementTable(len(self.knowledge.names), n_minerals=len(MINERALS),
                                           n_appearance=len(cfg.ancestry.appearance_traits),
                                           n_dialect=cfg.ancestry.dialect_dims,
                                           n_traits=len(cfg.founders.culture_traits))
        self.settlement_agents: dict[int, Settlement] = {}
        self.occupant = np.full(self.geom.n_cells, -1, dtype=np.int64)  # settlement id per cell
        # Who stood where, for the names of places only (``deluge/gazetteer.py``); never read by
        # the simulation (private: left out of the state digest, kept in checkpoints).
        self._occupancy = OccupancyLog()
        self._campaigns = CampaignLog()  # every attack, for the war display only (``data/campaigns.py``)
        self.forage_health = np.ones(self.geom.n_cells)  # 1 = forage untouched by overuse
        self.lineages = LineageRegistry()
        self.polities = PolityTable(len(cfg.founders.culture_traits), len(MINERALS))
        self.polity_agents: dict[int, Polity] = {}
        self.brain_layout = network.Layout(features.input_names(cfg.brains), cfg.brains.hidden, network.OUTPUTS,
                                           features.TARGET_FEATURES)
        self.brains = network.BrainTable(self.brain_layout, cfg.brains.recurrent_units,
                                         cfg.brains.features.growth_window_years)
        self.brain_fan_in = self.brain_layout.fan_in()
        self.brain_marker_mask = self.brain_layout.mask(features.MARKER_FEATURES, features.MARKER_TARGET_FEATURES)
        self.brain_output_mean = np.zeros(len(network.OUTPUTS))
        self.monuments = Monuments()
        self.wars: dict[tuple[int, int], int] = {}  # polity pair (low id first) -> year of its last battle
        self.wars_active = 0
        self.contact_cells = self.geom.cells_within(cfg.polities.contact_radius)
        self.territory_cells = self.geom.cells_within(cfg.polities.territory_radius)
        self.stats = dict.fromkeys(YEAR_STATS, 0.0)
        n_tech = len(self.knowledge.names)
        self.first_known_year = np.full(n_tech, -1)  # first year any settlement held each technique
        self.first_invented_year = np.full(n_tech, -1)  # first independent invention
        self.first_farming_year = np.full(len(cfg.founders.names), -1)  # per group, farming >= use_share
        self._place_founders()
        politics.update_territory(self)
        self.teachers_state = TeachersState(cfg.teachers, len(self.lineages.founders), self.knowledge.index,
                                            streams["teachers"])
        self.population_layer = PropertyLayer.from_data("population", as_field(self.geom, self.cell_population()))
        self.grid.add_property_layer(self.population_layer)
        self.knowledge_layer = PropertyLayer.from_data("knowledge", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.knowledge_layer)
        self.latent_layer = PropertyLayer.from_data("latent_knowledge", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.latent_layer)
        self.lineage_layer = PropertyLayer.from_data("lineage", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.lineage_layer)
        self.polity_layer = PropertyLayer.from_data("polity", as_field(self.geom, np.zeros(self.geom.n_cells)))
        self.grid.add_property_layer(self.polity_layer)
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
        f, a = cfg.founders, cfg.ancestry
        sids = []
        for name, cell, size in zip(f.names, cells, sizes):
            culture = {trait: float(rng.uniform(f.culture_min, f.culture_max)) for trait in f.culture_traits}
            lineage = len(self.lineages.founders)
            sid = self.found_settlement(cell, float(size), lineage=lineage, parent=-1, polity=-1)
            self.lineages.add_founder(name, sid, cell, size, self.year, culture)
            intermarriage = float(self.streams["politics"].uniform(a.intermarriage_min, a.intermarriage_max))
            pid = self.found_polity(lineage, -1, sid, np.array([culture[tr] for tr in f.culture_traits]),
                                    intermarriage, cfg.polities.cohesion.founder_cohesion)
            self.settlements.polity[sid] = pid
            self.log_event("founded", [("polity", pid), ("settlement", sid)], cell, float(size))
            sids.append(sid)
        identity.draw_founders(self, sids)

    def found_settlement(self, cell: int, population: float, lineage: int, parent: int,
                         store: float = 0.0, knows=None, strength=None, shares=None, practised=None,
                         polity: int | None = None) -> int:
        """A new settlement. ``polity`` defaults to the parent's; a daughter inherits the
        parent's appearance, dialect and culture, and, in the parent's polity, its grievance
        and subject status. Without a parent it takes its polity's culture."""
        if self.occupant[cell] >= 0 or self.is_sea[cell]:
            raise ValueError(f"cell {cell} is not free land")
        if polity is None:
            polity = int(self.settlements.polity[parent]) if parent >= 0 else -1
        t = self.settlements
        sid = t.add(cell, population, lineage, parent, self.year, store, knows, strength, shares, practised, polity)
        if parent >= 0:
            identity.inherit(self, sid, parent)
            memory.inherit(self, sid, parent)
            t.culture[sid] = t.culture[parent]
            if polity == t.polity[parent]:
                t.grievance[sid] = t.grievance[parent]
                t.subject_of[sid] = t.subject_of[parent]
        elif polity >= 0:
            t.culture[sid] = self.polities.culture[polity]
        self.occupant[cell] = sid
        self._occupancy.add(len(self.events), cell, sid)
        self.settlement_agents[sid] = Settlement(self, sid, self.grid[self.geom.xy(cell)])
        return sid

    # --- polities --------------------------------------------------------------------------

    def found_polity(self, lineage: int, parent: int, capital: int, culture: np.ndarray, intermarriage: float,
                     cohesion: float, marker: np.ndarray | None = None) -> int:
        """A new polity with ``capital``; a capital with no polity yet takes its culture.

        Its brain: a mutated copy of the parent's, with the parent's priorities until its
        first decision; without a parent, a fresh random brain (``brains`` stream) and its
        response to zero inputs. ``marker``: its marker parameters (zero if None)."""
        if self.settlements.polity[capital] < 0:
            self.settlements.culture[capital] = culture
        P, B = self.polities, self.brains
        pid = P.add(lineage, parent, self.year, capital, culture, intermarriage, cohesion,
                    self.config.polities.elites.initial_ratio, marker)
        rng = self.streams["brains"]
        if parent >= 0:
            P.load[pid] = P.load[parent]
            B.born(pid, evolution.mutate(B.genome[parent], self.config.brains, rng), B.output[parent],
                   B.target_w[parent], B.state[parent])
        else:
            genome = evolution.founder_genome(self.brain_fan_in, self.brain_marker_mask, self.config.brains, rng)
            out, target_w, _ = network.forward(self.brain_layout, genome[None, :],
                                               np.zeros((1, self.brain_layout.n_in)))
            B.born(pid, genome, out[0], target_w[0], np.zeros(B.state.shape[1]))
        self.polity_agents[pid] = Polity(self, pid)
        self.stats["polities_founded"] += 1
        return pid

    def end_polity(self, pid: int, cause: str, merged_into: int = -1, people: float | None = None) -> None:
        """End a polity: ``extinct`` (no settlements left), ``fragmented`` (broken into its
        regions), ``merged`` (absorbed by ``merged_into``) or ``conquered`` (destroyed by
        ``merged_into``). Every end but a merger is logged as the polity's end (``died``, with
        the conqueror as second actor; ``people`` defaults to its last counted population); a
        merger is logged by the caller as ``joined``."""
        P = self.polities
        if cause != "merged":
            actors = [("polity", pid)] + ([("polity", merged_into)] if cause == "conquered" else [])
            self.log_event("died", actors, int(self.settlements.cell[P.capital[pid]]),
                           float(P.population[pid] if people is None else people))
        if cause == "extinct":
            self.stats["polity_extinctions"] += 1
        P.ended[pid] = self.year
        P.end_cause[pid] = END_CAUSES.index(cause)
        P.merged_into[pid] = merged_into
        memory.polity_ended(self, pid)
        self.polity_agents.pop(pid).remove()

    def build_monument(self, pid: int, sid: int, cost: float) -> int:
        memory.invoke(self, pid, (entity("polity", pid), entity("settlement", sid)))
        cell = int(self.settlements.cell[sid])
        mid = self.monuments.add(sid, cell, pid, self.year, cost)
        self.stats["monuments_built"] += 1
        self.log_event("built", [("polity", pid), ("monument", mid), ("settlement", sid)], cell, cost)
        return mid

    def openness_of(self, ids: np.ndarray) -> np.ndarray:
        """The openness trait of each settlement's polity."""
        j = self.config.founders.culture_traits.index("openness")
        return self.polities.culture[self.settlements.polity[ids], j]

    def move_settlement(self, sid: int, cell: int) -> None:
        if self.occupant[cell] >= 0 or self.is_sea[cell]:
            raise ValueError(f"cell {cell} is not free land")
        self.occupant[self.settlements.cell[sid]] = -1
        self.occupant[cell] = sid
        self._occupancy.add(len(self.events), int(self.settlements.cell[sid]), -1)
        self._occupancy.add(len(self.events), cell, sid)
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
        memory.settlement_ended(self, sid, host)
        self.occupant[t.cell[sid]] = -1
        self._occupancy.add(len(self.events), int(t.cell[sid]), -1)
        t.ended[sid] = self.year
        t.population[sid] = 0.0
        t.store[sid] = 0.0
        self.settlement_agents.pop(sid).remove()

    def claimed_cells(self, ids: np.ndarray | None = None) -> np.ndarray:
        """Per cell: the polity claiming it, -1 for none. A polity claims its members' cells and
        the land within ``polities.territory_radius`` of them (``politics``: its territory); a
        cell in reach of several members goes to the nearest (then the lowest settlement id)."""
        t = self.settlements
        ids = t.alive_ids() if ids is None else ids
        out = np.full(self.geom.n_cells, -1, dtype=np.int64)
        if len(ids) == 0:
            return out
        own = t.cell[ids]
        ring = self.territory_cells[own]  # (S, k), -1 padding
        cells = np.concatenate([own, ring.ravel()])
        sids = np.concatenate([ids, np.repeat(ids, ring.shape[1])])
        ok = cells >= 0
        cells, sids = cells[ok], sids[ok]
        ok = ~self.is_sea[cells]
        cells, sids = cells[ok], sids[ok]
        d = self.geom.hex_distance(cells, t.cell[sids])
        order = np.lexsort((sids, d, cells))
        cells, sids = cells[order], sids[order]
        first = np.r_[True, cells[1:] != cells[:-1]]
        out[cells[first]] = t.polity[sids[first]]
        return out

    def cell_population(self) -> np.ndarray:
        """People per cell (flat)."""
        t = self.settlements
        ids = t.alive_ids()
        out = np.zeros(self.geom.n_cells)
        out[t.cell[ids]] = t.population[ids]
        return out

    def update_deposits(self) -> None:
        """Recompute, from ``deposits``, what each cell can reach and the resources layer.

        ``deposit_source``: (cells, minerals), the nearest deposit cell of each mineral
        within ``world.minerals.access_radius_cells`` hex steps, -1 if none (local reach, the
        stand-in for trade; polity territory adds access, ``politics.update_territory``,
        refreshed here too). Resources layer: 0 no deposit,
        1..len(MINERALS) one mineral (MINERALS order), len(MINERALS) + 1 several."""
        has = self.deposits > 0
        radius = self.config.world.minerals.access_radius_cells
        self.deposit_source = np.full(has.shape, -1, dtype=np.int64)
        for m in range(len(MINERALS)):
            if has[:, m].any():
                steps, nearest = self.geom.distance_from(has[:, m])
                self.deposit_source[:, m] = np.where((steps >= 0) & (steps <= radius), nearest, -1)
        code = np.where(has.sum(axis=1) > 1, len(MINERALS) + 1, has.argmax(axis=1) + 1)
        self.resources_layer.data[...] = as_field(self.geom, np.where(has.any(axis=1), code, 0).astype(float))
        if hasattr(self, "polities"):  # polity territories reach deposits too (set up after the world)
            politics.update_territory(self)
    # --- events and knowledge bookkeeping ---------------------------------------------------

    def log_event(self, action: str, actors, place: int, magnitude: float, link: int = -1, origin: int = -1) -> int:
        """Log a raw event; ``actors`` are (kind, local id) pairs; each settlement actor's
        people are noted with it, and each polity's or settlement's polity's people as last
        counted (``EventLog``). Returns the event's index, -1 if it was not logged."""
        t, P = self.settlements, self.polities
        people = [float(t.population[i]) if kind == "settlement" else np.nan for kind, i in actors]
        polity_people = []
        for kind, i in actors:
            p = int(i) if kind == "polity" else int(t.polity[i]) if kind == "settlement" else -1
            v = float(P.population[p]) if p >= 0 else 0.0
            polity_people.append(v if v > 0 else np.nan)
        ok = self.events.log(self.year, action, [entity(kind, i) for kind, i in actors], place, magnitude, link,
                             people, polity_people, origin)
        return len(self.events) - 1 if ok else -1

    def log_crossing(self, sea_cells: int, actors, place: int, link: int, origin: int = -1) -> None:
        """A sea crossing from cell ``origin`` that is part of event ``link`` (``crossed``: sea
        cells crossed)."""
        if sea_cells > 0:
            self.stats["sea_crossings"] += 1
            self.log_event("crossed", actors, place, float(sea_cells), link, origin)

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
        lineage = np.zeros(self.geom.n_cells)  # founding group + 1 per settled cell, 0 = nobody
        lineage[t.cell[ids]] = t.lineage[ids] + 1
        self.lineage_layer.data[...] = as_field(self.geom, lineage)
        # Polity layer: 1 + the polity claiming the cell, 0 = unclaimed (viz/portrayal.py draws it).
        P = self.polities
        pol = t.polity[ids]
        self.polity_layer.data[...] = as_field(self.geom, self.claimed_cells(ids) + 1.0)
        pids = P.alive_ids()
        self.n_polities = len(pids)
        self.mean_cohesion = float(P.cohesion[pids].mean()) if len(pids) else 0.0
        sizes = np.bincount(pol, minlength=P.n)
        self.largest_polity_settlements = int(sizes.max()) if len(ids) else 0
        self.largest_polity_population = float(np.bincount(pol, weights=t.population[ids], minlength=P.n).max()) \
            if len(ids) else 0.0
        # Techniques per cell: usable where the settlement is, and latent (held, unusable).
        known = np.zeros(self.geom.n_cells)
        latent = np.zeros(self.geom.n_cells)
        if len(ids) and self.capacity:
            usable = usable_mask(self, ids)
            known[t.cell[ids]] = (t.knows[ids] & usable).sum(axis=1)
            latent[t.cell[ids]] = (t.knows[ids] & ~usable).sum(axis=1)
        self.knowledge_layer.data[...] = as_field(self.geom, known)
        self.latent_layer.data[...] = as_field(self.geom, latent)

    # --- checkpoints -------------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Pickling leaves out the war reach cache (``war.land_reach``), a pure function of
        the sea that is rebuilt as needed, and the names and gazetteer made for the views
        (``naming``, ``gazetteer``)."""
        state = self.__dict__.copy()
        state.pop("_reach_cache", None)
        state.pop("_names", None)
        state.pop("_gazetteer", None)
        return state

    def __setstate__(self, state: dict) -> None:
        """Unpickling (checkpoints): Mesa 3.5's grid gives every unpickled cell a class of its
        own that pickle cannot find again, so a restored model could not be pickled twice;
        ``repair_unpickled_grid`` puts the cells back on one class."""
        self.__dict__.update(state)
        repair_unpickled_grid(self.grid)

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
