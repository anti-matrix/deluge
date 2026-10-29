"""Round 14 step 1: the calibration fixes and the new bounds.

- Monuments take ``monument_build_years`` and a polity builds one at a time.
- The attacker fights, and loses people, only with its settlements present at the battle: those
  within logistics reach of the target, and the one the attack set out from.
- A battle's dead stay within the war-deaths bound (``configs/bounds.yaml``).
- The report-only bounds' helpers: near-capacity growth windows, spread speed, land distance.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import small_model
from deluge.brains.network import OUT
from deluge.data.bounds import battle_death_limit, bounds_path, frontier_speeds, load_bounds, near_capacity_rates
from deluge.data.collectors import land_steps
from deluge.data.events import ACTION_CODE
from deluge.systems import cohesion, war
from test_polities import free_land, held, interior_cell, lone_polity, place


def test_a_monument_takes_its_years_and_one_is_built_at_a_time():
    m = held(years=10)
    pid, cap = lone_polity(m, interior_cell(m, 3), pop=5000.0)
    g = m.knowledge
    m.settlements.knows[cap, g.index("megalithic_construction")] = True
    P, pc = m.polities, m.config.polities
    built, y0 = [], m.year
    for _ in range(100):
        m.brains.output[pid, OUT["monuments"]] = 1.0
        P.treasury[pid] = 1e9  # far more than any monument takes: only the years hold it back
        n = len(m.monuments)
        cohesion.run(m)
        assert P.monument_started[pid] >= 0 or len(m.monuments) > n  # at most one under way
        built += [m.year] * (len(m.monuments) - n)
        m.year += 1
    assert len(built) >= 3 and len(set(built)) == len(built)  # never two in a year
    assert all(b - a >= pc.monument_build_years for a, b in zip(built, built[1:]))
    assert built[0] - y0 >= pc.monument_build_years  # the first began in the first year
    # Its cost: the build years x the labour put in when it began (at least monument_cost).
    put = pc.monument_max_share * 1.0 * 1e9 * (1.0 - pc.treasury_spoilage)
    assert m.monuments.cost[0] == pytest.approx(max(pc.monument_cost, pc.monument_build_years * put), rel=0.01)


def test_the_attacker_fights_only_with_the_settlements_present():
    m = held(years=10)
    origin = interior_cell(m, 3)
    a, sa = lone_polity(m, origin, pop=20000.0)
    b, sb = lone_polity(m, int(free_land(m, origin, 1)[0]), lineage=1, pop=1000.0)
    F = war.forces(m)
    before = war.strengths(m, F, np.array([a]), np.array([b]), np.array([sb]), np.array([sa]))[0][0]
    # A far settlement of the attacker, beyond its logistics reach of the target: no strength, no dead.
    reach_cells = int(np.ceil(F.reach[a] / m.unit_km)) + 2
    far_cells = [c for c in np.flatnonzero(~m.is_sea & (m.occupant < 0))
                 if m.geom.hex_distance(c, m.settlements.cell[sb]) > reach_cells]
    far = place(m, far_cells[0], a, pop=500000.0)
    F = war.forces(m)
    after = war.strengths(m, F, np.array([a]), np.array([b]), np.array([sb]), np.array([sa]))[0][0]
    assert after == pytest.approx(before)
    assert war.present_people(m, F, np.array([a]), np.array([sb]), np.array([sa]))[0] == m.settlements.population[sa]
    # The attack set out from the far settlement (as across the sea): it is present.
    assert war.present_people(m, F, np.array([a]), np.array([sb]), np.array([far]))[0] == pytest.approx(
        m.settlements.population[sa] + m.settlements.population[far])
    far_before = float(m.settlements.population[far])
    war.battle(m, F, a, b, sa, sb, campaign=False)
    assert m.settlements.population[far] == far_before  # its people did not fight


@pytest.fixture(scope="module")
def warring():
    m = small_model({"polities": {"secession_base": 0.5}, "war": {"attack_rate": 5.0}}, years=300)  # warlike: battles happen
    m.run()
    return m


def test_battle_dead_stay_within_the_people_present(warring):
    c = warring._campaigns.columns()
    assert len(c["year"]) > 0
    share = load_bounds(bounds_path(warring.config))["war_deaths"]["fighting_age_share"]
    limit = battle_death_limit(c["target_people"], c["present_people"], c["mobilization"], share)
    assert np.all(c["dead_attacker"] + c["dead_defender"] <= limit * (1 + 1e-9))


def test_near_capacity_windows_and_spread_speed():
    s = pd.DataFrame({"year": [0, 0, 500, 500, 1000], "settlement": [1, 2, 1, 2, 1], "lineage": [0, 0, 0, 0, 0],
                      "cell": [5, 6, 5, 6, 5], "population": [600.0, 100.0, 900.0, 400.0, 1800.0],
                      "capacity": [1000.0, 1000.0, 1000.0, 1000.0, 3000.0]})
    w = near_capacity_rates(s, 0.5, 500)
    # Settlement 1 from 0 (600 of 1000) and from 500 (900 of 1000); settlement 2 was below half.
    assert sorted(zip(w.year, w.settlement)) == [(0, 1), (500, 1)]
    assert w.rate.tolist() == pytest.approx([(900 / 600) ** (1 / 500) - 1, (1800 / 900) ** (1 / 500) - 1])
    assert near_capacity_rates(s, 0.5, 500, until=1000).year.tolist() == [0]  # windows ending before industry
    f = pd.DataFrame({"year": [0, 500, 1000], "lineage": [0, 0, 0], "frontier_km": [0.0, 400.0, 1400.0]})
    sp = frontier_speeds(f, 500)
    assert sp.km_per_year.tolist() == pytest.approx([0.8, 2.0])


def test_land_steps_go_over_land_only(warring):
    m = warring
    land = ~m.is_sea
    start = int(np.flatnonzero(land)[len(np.flatnonzero(land)) // 2])
    d = land_steps(m.geom.neighbors, land, start, m.geom)
    reached = np.flatnonzero(d >= 0)
    assert d[start] == 0 and land[reached].all()
    assert np.all(d[reached] >= m.geom.hex_distance(reached, start))  # never shorter than the straight line
    # From a drowned start: from the nearest land, at its distance.
    sea = int(np.flatnonzero(~land)[0])
    d2 = land_steps(m.geom.neighbors, land, sea, m.geom)
    assert d2[sea] == -1 and d2[d2 >= 0].min() == m.geom.hex_distance(np.flatnonzero(land), sea).min()


def test_the_run_records_the_new_bound_tables(warring):
    from deluge.data.collectors import RunRecorder

    m = small_model(years=120)
    rec = RunRecorder(m)
    while m.running:
        m.step()
        rec.observe(m)
    s, f, p = rec.samples_frame(), rec.frontier_frame(), rec.scale_frame()
    assert set(s.year) == {0, 50, 100} and (s.capacity > 0).all()
    assert set(f.year) == set(range(0, 120, 25)) and set(f.lineage) == set(range(len(m.lineages.names)))
    assert list(p.year) == [0] and {"population", "settlements", "writing", "hierarchy", "held"} <= set(p.columns)
    assert ACTION_CODE["built"] >= 0
