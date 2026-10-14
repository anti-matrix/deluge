"""Round 3: foragers and demography (spec sections 6 and 13).

Measured on world seed 1. Model setup takes about 9 s (world generation), so the runs
are shared through module fixtures.
"""

import pickle

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deluge.agents.founders import PlacementError, candidate_cells, place_founders
from deluge.agents.settlement import ALIVE
from deluge.config import load_config
from deluge.model import DelugeModel
from deluge.systems.demography import next_population
from deluge.world.climate import Drought

WORLD = 1
HISTORIES = (1, 2, 3, 4, 5)
# Round 3 behaviour is that of foragers: these tests switch off the teachers and
# invention (Round 4), so no settlement learns herding or farming.
FORAGERS = {"teachers": {"enabled": False}, "knowledge": {"invention_scale": 0.0}}


def foragers(**kw) -> DelugeModel:
    return DelugeModel(config=load_config(overrides=FORAGERS), **kw)


def check_invariants(m: DelugeModel) -> None:
    t = m.settlements
    ids = t.alive_ids()
    cells = t.cell[ids]
    assert not m.is_sea[cells].any(), "settlement on a sea cell"
    assert len(np.unique(cells)) == len(cells), "two settlements on one cell"
    assert (m.occupant[cells] == ids).all()
    assert (m.occupant >= 0).sum() == len(ids)
    assert (t.population[ids] >= 0).all()
    assert set(m.settlement_agents) == set(ids.tolist())
    for sid in ids[:: max(1, len(ids) // 50)].tolist():  # Mesa agent sits on the table's cell
        assert m.settlement_agents[sid].cell.coordinate == m.geom.xy(int(t.cell[sid]))


@pytest.fixture(scope="module")
def run3000():
    """World 1, history 1 to year 3,000 (through the sea-level rise), invariants checked
    every year. Built for 3,100 years so later years can be stepped from it."""
    m = foragers(world_seed=WORLD, history_seed=1, years=3100)
    check_invariants(m)
    while m.year < 3000:
        m.step()
        check_invariants(m)
    return m


# --- seeds and placement ---------------------------------------------------------------------


def test_history_seed_keeps_the_world(run3000):
    other = foragers(world_seed=WORLD, history_seed=2, years=1)
    np.testing.assert_array_equal(other.world.elevation, run3000.world.elevation)
    assert [f.cell for f in other.lineages.founders] != [f.cell for f in run3000.lineages.founders]


def test_founder_placement_rules(run3000):
    m = run3000
    cfg = m.config.founders
    cand = set(candidate_cells(m.founder_capacity, ~(m.elevation_flat < m.sea_level_series[0]),
                               cfg.capacity_top_share).tolist())
    land = m.elevation_flat >= m.sea_level_series[0]
    for h in range(1, 11):
        cells, sizes = place_founders(m.geom, m.founder_capacity, land, m.founder_landmass, cfg,
                                      np.random.default_rng(h))
        assert len(cells) == len(cfg.names)
        assert set(cells) <= cand  # top third of year-0 foraging capacity
        d = m.geom.hex_distance(np.array(cells)[:, None], np.array(cells)[None, :])
        assert (d[~np.eye(len(cells), dtype=bool)] >= cfg.min_distance_cells).all()
        assert len(set(m.founder_landmass[cells].tolist())) >= cfg.min_landmasses
        assert all(cfg.population_min <= s <= cfg.population_max for s in sizes)
    f = m.lineages.founders
    assert [x.name for x in f] == list(cfg.names)
    assert all(m.settlements.parent[x.settlement] == -1 and m.settlements.founded[x.settlement] == 0 for x in f)


def test_pinned_placement(run3000):
    m = run3000
    cfg = load_config().founders
    cells = [m.lineages.founders[i].cell for i in range(5)][::-1]
    pinned = type(cfg)(**{**cfg.__dict__, "pinned_cells": tuple(m.geom.xy(c) for c in cells)})
    land = m.elevation_flat >= m.sea_level_series[0]
    got, _ = place_founders(m.geom, m.founder_capacity, land, m.founder_landmass, pinned, np.random.default_rng(0))
    assert got == cells
    bad = type(cfg)(**{**cfg.__dict__, "pinned_cells": ((0, 0),) * 5})  # map corner: sea
    with pytest.raises(PlacementError):
        place_founders(m.geom, m.founder_capacity, land, m.founder_landmass, bad, np.random.default_rng(0))


# --- demography ---------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.floats(0.0, 1e6), st.floats(0.0, 50.0), st.floats(-8.0, 8.0)),
                min_size=1, max_size=50))
def test_population_never_negative(rows):
    pop, ratio, noise = (np.array(c) for c in zip(*rows))
    d = load_config().demography
    new, deaths = next_population(pop, ratio, np.full(len(pop), d.forage_abundance),
                                  np.full(len(pop), d.forage_max_growth), d, noise)
    assert (new >= 0).all() and (deaths >= 0).all()
    assert (deaths <= pop + 1e-9).all()


def test_bands_grow_toward_capacity_and_stabilize():
    """With splitting switched off, each founding band grows until it holds its cell's
    capacity, then stays there."""
    cfg = load_config(overrides=FORAGERS).with_overrides({"migration": {"split_threshold": 1e9}})
    m = DelugeModel(config=cfg, world_seed=WORLD, history_seed=1, years=4000)
    sids = [f.settlement for f in m.lineages.founders]
    ratios = []
    while m.running:
        m.step()
        if m.year > 3600:
            t = m.settlements
            k = m.capacity["forage"][t.cell[sids]] * m.forage_health[t.cell[sids]]
            ratios.append(t.population[sids] / k)
    ratios = np.array(ratios)  # (years, bands)
    assert (m.settlements.ended[sids] == ALIVE).all()
    assert m.settlements.n == len(sids)  # no daughters
    mean = ratios.mean(axis=0)
    assert ((mean > 0.85) & (mean < 1.05)).all(), mean
    assert (ratios.std(axis=0) / mean < 0.1).all()  # stable, not oscillating


def test_bands_spread_over_centuries(run3000):
    df = run3000.datacollector.get_model_vars_dataframe().set_index("year")
    n = df["n_settlements"]
    assert n[0] == 5
    assert n[1000] > n[0] and n[2000] > 3 * n[1000] and n[2999] > 2 * n[2000]
    t = run3000.settlements
    ids = t.alive_ids()
    origin = np.array([run3000.lineages.founders[lin].cell for lin in t.lineage[ids]])
    # Measured: median 8 steps (~1,400 km), each lineage filling ~190 contiguous cells.
    dist = run3000.geom.hex_distance(t.cell[ids], origin)
    assert np.median(dist) > 5 and dist.max() > 15


def test_population_in_range_and_no_lineage_lost_everywhere():
    """Round 3 acceptance (foragers) on world seed 1, history seeds 1-5: total population
    at year 3,000 between 0.5 and 10 million, and every founding group survives in some run."""
    survived = np.zeros(5, dtype=bool)
    totals = []
    for h in HISTORIES:
        m = foragers(world_seed=WORLD, history_seed=h, years=3000)
        m.run()
        totals.append(m.population)
        survived |= m.lineage_population > 0
    assert all(0.5e6 <= p <= 10e6 for p in totals), totals
    assert survived.all(), survived


def test_drought_causes_famine_and_migration(run3000):
    """A severe drought over a settled region, shaped like the climate shocks (strongest at
    the centre): famine deaths and whole-band moves rise far above the same years
    without it, and the region loses people."""
    base = run3000
    t = base.settlements
    ids = t.alive_ids()
    radius, severity = 800.0, 0.1
    pos = base.shocks.positions
    # Centre on the settled cell with the most land around it, so the drought's edge
    # lies inside the land.
    land = ~base.is_sea
    around = [land[np.linalg.norm(pos - pos[c], axis=1) < 2 * radius].sum() for c in t.cell[ids]]
    center = int(t.cell[ids[int(np.argmax(around))]])
    d = np.linalg.norm(pos - pos[center], axis=1) / radius
    cells = np.flatnonzero(d < 1.0)
    factor = 1.0 - (1.0 - severity) * (1.0 - d[cells] ** 2)
    years = 10

    def follow(with_drought: bool):
        m = pickle.loads(pickle.dumps(base))
        if with_drought:
            m.shocks.droughts.append(Drought(m.year, m.year + years, center, radius, cells, factor))
        deaths = moves = 0.0
        pop_in = []
        for _ in range(years):
            m.step()
            deaths += m.stats["famine_deaths"]
            moves += m.stats["moves"]
            pop_in.append(m.cell_population()[cells].sum())
        return deaths, moves, pop_in

    dry_deaths, dry_moves, dry_pop = follow(True)
    wet_deaths, wet_moves, wet_pop = follow(False)
    assert dry_deaths > 10 * max(wet_deaths, 1.0), (dry_deaths, wet_deaths)
    assert dry_moves > wet_moves + 5, (dry_moves, wet_moves)
    assert dry_pop[-1] < 0.8 * wet_pop[-1]


# --- lineage --------------------------------------------------------------------------------


def test_every_settlement_traces_to_a_founder(run3000):
    m = run3000
    t = m.settlements
    founders = {f.settlement: f.lineage for f in m.lineages.founders}
    for sid in range(t.n):
        chain = m.lineages.ancestry(t, sid)
        assert chain[-1] in founders
        assert founders[chain[-1]] == t.lineage[sid]
        assert all(t.founded[a] <= t.founded[b] for a, b in zip(chain[1:], chain[:-1]))
    edges = m.lineages.tree(t, year=1000)
    assert len(edges) == (t.founded[: t.n] <= 1000).sum()
    alive_then = [e for e in edges if e[4] == ALIVE]
    assert len(alive_then) == m.datacollector.get_model_vars_dataframe().set_index("year").loc[1000, "n_settlements"]


def test_lineage_population_sums_to_total(run3000):
    df = run3000.datacollector.get_model_vars_dataframe()
    groups = df[[f"pop_{n}" for n in run3000.lineages.names]].sum(axis=1)
    np.testing.assert_allclose(groups, df["population"], rtol=1e-9)
