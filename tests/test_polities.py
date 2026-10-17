"""Round 5: polities, cohesion, secession, fragmentation, mergers, monuments, territory and
the polity lineage tree (spec section 8).

Scenarios run on the small test world. ``held`` keeps settlements in place and fed so that
only the polity rules act; settlements and polities are placed by hand where a test needs
a given geometry.
"""

import numpy as np
import pytest

from conftest import small_model
from deluge.agents.polity import ALIVE, END_CAUSES
from deluge.data import chronicle
from deluge.data.events import ACTIONS, decode, entity
from deluge.systems import cohesion, politics
from deluge.systems.knowledge import resource_access
from deluge.world.worlddata import MINERALS

HELD = {
    "teachers": {"enabled": False},
    "demography": {"famine_mortality": 0.0, "demographic_turnover": 0.0},
    "migration": {"famine_move_ratio": 0.0, "split_threshold": 1e9},
    "knowledge": {"invention_scale": 0.0},
    "war": {"enabled": False},
}


def held(polities=None, years=100, more=None, **kw):
    over = dict(HELD)
    if polities:
        over["polities"] = polities
    over.update(more or {})
    m = small_model(over, years=years, **kw)
    m.step()
    return m


def events(m, action):
    return [m.events.record(i) for i in range(len(m.events)) if ACTIONS[m.events.action[i]] == action]


def free_land(m, origin, steps, exclude=()):
    """Free land cells exactly ``steps`` hex steps from ``origin``."""
    cells = np.arange(m.geom.n_cells)
    d = m.geom.hex_distance(cells, origin)
    ok = (d == steps) & ~m.is_sea & (m.occupant < 0) & ~np.isin(cells, list(exclude))
    return cells[ok]


def place(m, cell, pid, pop=1000.0):
    """A settlement of polity ``pid`` on ``cell`` (lineage of the polity)."""
    sid = m.found_settlement(int(cell), pop, int(m.polities.lineage[pid]), -1, polity=pid)
    m.settlements.peak_population[sid] = pop
    return sid


def lone_polity(m, cell, lineage=0, pop=1000.0, culture=None, cohesion_value=0.5):
    """A new polity with one settlement on ``cell``."""
    sid = m.found_settlement(int(cell), pop, lineage, -1, polity=-1)
    c = np.full(len(m.config.founders.culture_traits), 0.5) if culture is None else np.asarray(culture, float)
    pid = m.found_polity(lineage, -1, sid, c, 0.5, cohesion_value)
    m.settlements.polity[sid] = pid
    return pid, sid


def interior_cell(m, radius):
    """A free land cell with at least three free land cells at every distance 1..``radius``,
    and its immediate neighbours all free land."""
    cells = np.arange(m.geom.n_cells)
    free = ~m.is_sea & (m.occupant < 0)
    for c in cells[free].tolist():
        d = m.geom.hex_distance(cells, c)
        if free[d == 1].all() and all((free & (d == k)).sum() >= 3 for k in range(2, radius + 1)):
            return c
    raise AssertionError("no open interior on the small world")


# --- founding and the tree ---------------------------------------------------------------------


def test_each_founding_group_starts_as_one_polity():
    m = small_model(years=5)
    P, t = m.polities, m.settlements
    assert P.n == 5 and len(P.alive_ids()) == 5
    for f in m.lineages.founders:
        pid = int(t.polity[f.settlement])
        assert P.lineage[pid] == f.lineage and P.parent[pid] == -1 and P.capital[pid] == f.settlement
        traits = m.config.founders.culture_traits
        np.testing.assert_allclose(P.culture[pid], [f.culture[tr] for tr in traits])
    assert len(m.polity_agents) == 5
    assert len(events(m, "founded")) == 5  # one per founding polity, from its band


def test_daughters_join_their_parents_polity():
    m = small_model(years=400)
    m.run()
    t = m.settlements
    ids = t.alive_ids()
    assert len(ids) > 5
    alive = set(m.polities.alive_ids().tolist())
    assert set(t.polity[ids].tolist()) <= alive
    daughters = ids[t.parent[ids] >= 0]
    same = t.polity[daughters] == t.polity[t.parent[daughters]]
    seceded = m.polities.parent[t.polity[daughters]] >= 0
    assert (same | seceded | (t.ended[t.parent[daughters]] >= 0)).all()


def test_every_polity_traces_to_a_founder_and_the_tree_exports():
    m = small_model({"polities": {"secession_base": 0.2}}, years=500)
    m.run()
    P = m.polities
    assert P.n > 8 and (P.parent[: P.n] >= 0).sum() >= 4
    # The founding polities (their bands may since have been conquered).
    founders = {int(pid): int(P.lineage[pid]) for pid in np.flatnonzero(P.parent[: P.n] == -1)}
    assert sorted(founders.values()) == [f.lineage for f in m.lineages.founders]
    for pid in range(P.n):
        chain = m.lineages.polity_ancestry(P, pid)
        assert chain[-1] in founders and P.lineage[pid] == founders[chain[-1]]
    tree = m.lineages.polity_tree(m)
    assert len(tree) == P.n
    assert set(tree.columns) >= {"polity", "parent", "lineage", "founded", "ended", "end_cause", "merged_into",
                                 "peak_settlements", "peak_population"}
    assert (tree["parent"] < tree["polity"]).all()  # parents are always older
    assert set(tree["end_cause"]) <= set(END_CAUSES)


# --- secession and fragmentation ---------------------------------------------------------------


def test_seceders_in_contact_form_one_polity():
    """Capital, then far from it two settlements in contact and one on its own: everyone but
    the capital secedes; the pair forms one polity, the loner another."""
    m = held({"secession_base": 1e3, "secession_max": 1.0})
    origin = interior_cell(m, 9)
    pid, cap = lone_polity(m, origin, cohesion_value=0.0)
    a = int(free_land(m, origin, 6)[0])
    b = int(free_land(m, a, 1, exclude=[origin])[0])
    far = [c for c in free_land(m, origin, 7).tolist()
           if m.geom.hex_distance(c, a) > 4 and m.geom.hex_distance(c, b) > 4][0]
    sa, sb, sc = place(m, a, pid), place(m, b, pid), place(m, far, pid)
    n0 = m.polities.n
    politics.secession(m)
    t, P = m.settlements, m.polities
    assert t.polity[cap] == pid
    assert t.polity[sa] == t.polity[sb] != pid
    assert t.polity[sc] not in (pid, t.polity[sa])
    for new in (t.polity[sa], t.polity[sc]):
        assert P.parent[new] == pid and P.lineage[new] == P.lineage[pid] and new >= n0
    founded = [r for r in events(m, "founded") if decode(r["actors"][0])[0] == "polity"]
    assert {r["actors"][1] for r in founded[-2:]} == {entity("polity", pid)}


def test_secession_rises_with_distance_and_low_cohesion():
    m = held()
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin)
    near = place(m, free_land(m, origin, 2)[0], pid)
    far = place(m, free_land(m, origin, 8)[0], pid)
    pc = m.config.polities
    P, t = m.polities, m.settlements

    def odds(sid):
        d = m.geom.hex_distance(int(t.cell[sid]), origin)
        return (pc.secession_base * (d / pc.secession_distance_reference) ** pc.secession_distance_exponent
                * (1 - P.cohesion[pid]) * (1 + pc.secession_pressure_weight * P.pressure[pid]))

    assert odds(far) > odds(near) > 0
    before = odds(far)
    P.cohesion[pid] = 0.9
    assert odds(far) < before


def test_fragmentation_breaks_a_polity_into_its_regions():
    m = held({"fragment_rate": 1.0, "region_spacing": 4})
    origin = interior_cell(m, 9)
    pid, cap = lone_polity(m, origin)
    members = [cap, place(m, free_land(m, origin, 1)[0], pid)]
    east = int(free_land(m, origin, 6)[0])
    members += [place(m, east, pid, 2000.0), place(m, free_land(m, east, 1, exclude=[origin])[0], pid)]
    P = m.polities
    cohesion.housekeeping(m)
    P.n_settlements[pid] = len(members)
    P.cohesion[pid], P.pressure[pid] = 0.0, 1.0
    politics.fragmentation(m)
    t = m.settlements
    assert P.ended[pid] == m.year and END_CAUSES[P.end_cause[pid]] == "fragmented"
    new = sorted(set(t.polity[members].tolist()))
    assert len(new) == 2 and all(P.parent[n] == pid for n in new)
    assert t.polity[members[0]] == t.polity[members[1]] and t.polity[members[2]] == t.polity[members[3]]
    died = [r for r in events(m, "died") if r["actors"] == (entity("polity", pid),)]
    assert died


def test_children_inherit_mutated_culture():
    m = held({"secession_base": 1e3, "secession_max": 1.0})
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin, cohesion_value=0.0)
    sid = place(m, free_land(m, origin, 6)[0], pid)
    politics.secession(m)
    child = int(m.settlements.polity[sid])
    P = m.polities
    diff = np.abs(P.culture[child] - P.culture[pid])
    assert 0 < diff.max() < 6 * m.config.polities.culture_mutation
    assert P.cohesion[child] == m.config.polities.cohesion.child_cohesion


def test_empty_polity_goes_extinct():
    m = held()
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    m.end_settlement(sid, "dissolved")
    m.step()
    P = m.polities
    assert P.ended[pid] >= 0 and END_CAUSES[P.end_cause[pid]] == "extinct"
    assert pid not in m.polity_agents


# --- mergers -----------------------------------------------------------------------------------


def test_no_mergers_with_the_memory_consequences_off():
    """Mergers need shared-memory affinity (Round 8); the control switch zeroes it."""
    m = held({"merge_rate": 1.0}, years=60, more={"memory": {"control": True}})
    origin = interior_cell(m, 4)
    lone_polity(m, origin)
    lone_polity(m, free_land(m, origin, 1)[0], lineage=1)
    assert len(politics.contact_pairs(m)) >= 1
    m.run()
    assert m.datacollector.get_model_vars_dataframe()["mergers"].sum() == 0
    assert not (m.polities.end_cause[: m.polities.n] == END_CAUSES.index("merged")).any()


def test_merge_mechanism():
    m = held()
    origin = interior_cell(m, 4)
    a, sa = lone_polity(m, origin, pop=5000.0, culture=np.zeros(6))
    b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=1000.0, culture=np.ones(6))
    cohesion.run(m)
    m.polities.treasury[b] = 10.0
    treasury = m.polities.treasury[a] + 10.0
    big = politics.merge(m, a, b)
    P, t = m.polities, m.settlements
    assert big == a and t.polity[sb] == a
    assert END_CAUSES[P.end_cause[b]] == "merged" and P.merged_into[b] == a
    assert P.treasury[a] == pytest.approx(treasury)
    assert set(np.unique(P.culture[a]).tolist()) <= {0.0, 1.0}  # own values, or the smaller's after crossover
    joined = events(m, "joined")[-1]
    assert joined["actors"] == (entity("polity", b), entity("polity", a))


# --- cohesion, tribute, elites, monuments --------------------------------------------------------


def test_frontier_with_distant_cultures_builds_cohesion():
    """Two polities in contact: culturally distant neighbours raise f and cohesion; an
    identical neighbour gives f = 0 and cohesion only decays."""
    results = {}
    for name, other in (("distant", np.ones(6)), ("alike", np.zeros(6))):
        m = held(years=200)
        origin = interior_cell(m, 4)
        a, _ = lone_polity(m, origin, culture=np.zeros(6))
        lone_polity(m, free_land(m, origin, 1)[0], lineage=1, culture=other)
        for _ in range(100):
            m.step()
        results[name] = (float(m.polities.frontier[a]), float(m.polities.cohesion[a]))
    assert results["distant"][0] == 1.0 and results["alike"][0] == 0.0
    assert results["distant"][1] > 0.5 > results["alike"][1]


def test_tribute_elites_and_monuments():
    m = held({"monument_cost": 50.0, "cohesion": {"monument_cohesion": 0.05}}, years=200)
    P, t, g = m.polities, m.settlements, m.knowledge
    origin = interior_cell(m, 3)
    culture = np.full(6, 0.5)
    traits = m.config.founders.culture_traits
    culture[traits.index("hierarchy")] = 1.0
    culture[traits.index("ritual_investment")] = 1.0
    pid, sid = lone_polity(m, origin, pop=5000.0, culture=culture)
    from test_brains import scripted
    scripted(m, pid, tribute=1.0, monuments=1.0)
    t.knows[sid, g.index("megalithic_construction")] = True
    t.strength[sid, g.index("megalithic_construction")] = 1.0
    e0 = float(P.elite_ratio[pid])
    for _ in range(30):
        t.store[sid] = 2000.0
        m.step()
    assert P.tribute[pid] > 0 and P.treasury[pid] > 0
    assert P.elite_ratio[pid] > e0  # good times: tribute-takers grow
    built = [r for r in events(m, "built") if r["actors"][0] == entity("polity", pid)]
    assert built and m.monuments.maintained[m.monuments.builder == pid].any()
    assert P.prestige[pid] > 0
    name = chronicle.Namer(m)
    line = chronicle.line(built[0], name)
    assert f"{name(entity('polity', pid))} built a monument" in line


def test_no_monuments_without_megalithic_construction():
    m = held({"monument_cost": 50.0}, years=60)
    pid, sid = lone_polity(m, interior_cell(m, 3), pop=5000.0, culture=np.ones(6))
    m.settlements.knows[sid] = False
    for _ in range(30):
        m.settlements.store[sid] = 2000.0
        m.step()
    assert not [r for r in events(m, "built") if r["actors"][0] == entity("polity", pid)]


# --- territory and deposits -------------------------------------------------------------------------


def test_deposit_access_widens_to_the_polity_territory():
    """A copper deposit beyond a settlement's own access radius but next to another
    settlement of its polity gives access; a settlement of another polity at the same
    distance has none."""
    m = held()
    radius = m.config.world.minerals.access_radius_cells
    origin = interior_cell(m, 2)
    pid, home = lone_polity(m, origin)
    cells = np.arange(m.geom.n_cells)
    far = cells[(m.geom.hex_distance(cells, origin) == radius + 4) & ~m.is_sea & (m.occupant < 0)]
    outpost_cell = int(far[0])
    deposit = int(free_land(m, outpost_cell, 1)[0])
    stranger_cell = [c for c in free_land(m, deposit, 1).tolist()
                     if m.geom.hex_distance(c, origin) > radius and c != outpost_cell][0]
    outpost = place(m, outpost_cell, pid)
    other, stranger = lone_polity(m, stranger_cell, lineage=1)
    m.deposits[:] = 0.0
    m.deposits[deposit, MINERALS.index("copper")] = 1.0
    m.update_deposits()
    assert m.deposit_source[origin, MINERALS.index("copper")] < 0  # beyond the home's own reach
    politics.update_territory(m)
    acc = resource_access(m, np.array([home, outpost, stranger]))[:, MINERALS.index("copper")]
    assert acc[0] and acc[1]  # the polity controls the deposit's cell
    assert acc[2]  # the stranger is itself next to it (own reach)
    m.deposits[:] = 0.0
    m.deposits[deposit, MINERALS.index("copper")] = 1.0
    m.update_deposits()
    m.settlements.polity[outpost] = other  # the outpost changes hands
    politics.update_territory(m)
    acc = resource_access(m, np.array([home]))[:, MINERALS.index("copper")]
    assert not acc[0]


# --- outputs ---------------------------------------------------------------------------------------------


def test_polity_layer_and_chronicle():
    m = small_model({"polities": {"secession_base": 0.5}}, years=300)
    m.run()
    t, P = m.settlements, m.polities
    ids = t.alive_ids()
    layer = m.polity_layer.data.ravel()
    np.testing.assert_array_equal(layer[t.cell[ids]], 1 + t.polity[ids])  # a member's cell is its polity's
    claimed = np.flatnonzero(layer > 0)
    assert len(claimed) > len(ids) and not m.is_sea[claimed].any()
    # Every claimed cell lies within the territory radius of a settlement of the claiming polity.
    r = m.config.polities.territory_radius
    for c in claimed[:200].tolist():
        members = ids[t.polity[ids] == layer[c] - 1]
        assert m.geom.hex_distance(t.cell[members], c).min() <= r
    view = chronicle.Chronicle(m)
    view.sync()
    assert " people (A) were founded from " in view.markdown()  # a polity founded: always told
    assert len(view.select(groups={0})) > 0
    df = m.datacollector.get_model_vars_dataframe()
    assert {"n_polities", "mean_cohesion", "secessions", "fragmentations", "mergers"} <= set(df.columns)
    assert df["n_polities"].iloc[-1] == len(P.alive_ids())


def test_common_enemy_matches_the_set_definition():
    """Two polities share an enemy: some third polity at war with both (``model.wars``)."""
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 30))
        wars = {}
        for _ in range(int(rng.integers(0, 40))):
            p, q = sorted(rng.choice(n, 2, replace=False).tolist())
            wars[(p, q)] = 0
        enemies = {}
        for p, q in wars:
            enemies.setdefault(p, set()).add(q)
            enemies.setdefault(q, set()).add(p)
        a, b = rng.integers(0, n, 50), rng.integers(0, n, 50)
        expected = [float(bool((enemies.get(p, set()) & enemies.get(q, set())) - {p, q}))
                    for p, q in zip(a.tolist(), b.tolist())]
        got = politics.common_enemy(SimpleNamespace(wars=wars), a, b)
        assert got.tolist() == expected
