"""Summary statistics of a generated world, for reports (and tests)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from deluge.config import Config
from deluge.world import geography
from deluge.world.climate import preset_horizon, sea_level_curve
from deluge.world.grid import HexGeometry, hex_area_km2, hex_unit_km
from deluge.world.worlddata import GenerationReport, Terrain, WorldData


@dataclass
class BridgeSummary:
    crest_m: float
    year_drowned: int
    during_pulse: int  # pulse number (1-based), 0 if not during a pulse
    lat: float
    col: int
    row: int
    land_cells_each_side: tuple[int, int]
    constructed: bool


@dataclass
class WorldSummary:
    seed: int
    attempts: int
    cell_km: float  # flat-to-flat hex width
    cell_area_km2: float
    land_cells_year0: int
    drowned_by_year: dict[int, dict[str, float]]
    landmasses: list[int]
    islands: int
    inland_seas: list[int]
    enclosed_seas: list[int]
    bridges: list[BridgeSummary]
    river_cells: int
    rivers_by_order: dict[int, int]  # Strahler streams per order
    rivers_longer_than_15: int
    longest_river: int
    max_row_aligned_run: int
    row_aligned_4step_share: float
    land_share_by_class: dict[str, float]
    land_below_500m_share: float
    land_above_2000m_share: float
    max_elevation_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(cfg: Config, world: WorldData, report: GenerationReport, years=(1000, 2000, 3000)
              ) -> WorldSummary:
    geom = HexGeometry(world.width, world.height, cfg.grid.torus)
    unit_km = hex_unit_km(cfg.grid.height, cfg.world.lat_south, cfg.world.lat_north, cfg.world.km_per_degree)
    area = hex_area_km2(unit_km)
    curve = sea_level_curve(cfg.climate.sea_level, preset_horizon(cfg.climate.sea_level))
    sea0 = float(curve[0])
    e = world.elevation.ravel()
    land0 = e >= sea0
    drowned = {}
    for y in years:
        n = int((land0 & (e < curve[y])).sum())
        drowned[int(y)] = {"sea_level_m": float(curve[y]), "cells": n, "km2": n * area,
                           "share_of_year0_land": n / land0.sum()}
    lm = geography.landmass_report(geom, e, sea0, cfg.world.landmasses)
    sb = geography.sea_basin_report(geom, e, sea0, cfg.world.seas)
    pulses = geography.pulse_windows(curve, cfg.world.land_bridges.pulse_rate_factor)
    floor = cfg.world.land_bridges.min_share_of_land * land0.sum()
    lat = geom.latitude(cfg.world.lat_south, cfg.world.lat_north)
    built = set(report.constructed_bridge)
    bridges = []
    for b in geography.land_bridges(geom, e, curve, floor, pulses):
        x, y = geom.xy(b.crest_cell)
        bridges.append(BridgeSummary(round(b.crest_m, 1), b.year_drowned, b.pulse + 1,
                                     round(float(lat[b.crest_cell]), 1), int(x), int(y), b.sizes,
                                     b.crest_cell in built))
    river = world.river.ravel()
    recv = world.flow_to.ravel()
    lengths = geography.river_lengths(e, river, recv)
    runs = geography.row_aligned_runs(geom, river, recv)
    terrain = world.terrain.ravel()
    shares = {t.key: float((terrain == t).sum() / land0.sum()) for t in Terrain if t != Terrain.SEA}
    above0 = e[land0]
    return WorldSummary(
        seed=cfg.run.seed,
        attempts=report.attempts,
        cell_km=float(np.sqrt(3) * unit_km),
        cell_area_km2=float(area),
        land_cells_year0=int(land0.sum()),
        drowned_by_year=drowned,
        landmasses=list(lm.landmasses),
        islands=lm.islands,
        inland_seas=list(sb.inland_seas),
        enclosed_seas=list(sb.enclosed_basins),
        bridges=bridges,
        river_cells=int((river > 0).sum()),
        rivers_by_order=geography.streams_by_order(world.river_order.ravel(), recv),
        rivers_longer_than_15=sum(1 for n in lengths.values() if n > 15),
        longest_river=max(lengths.values(), default=0),
        max_row_aligned_run=runs[0],
        row_aligned_4step_share=runs[1],
        land_share_by_class=shares,
        land_below_500m_share=float((above0 < 500).mean()),
        land_above_2000m_share=float((above0 > 2000).mean()),
        max_elevation_m=float(e.max()),
    )
