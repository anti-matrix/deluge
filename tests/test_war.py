"""Round 6: war and conquest (spec section 9); the choice follows the brains from Round 7.

Scenarios run on the small test world with settlements held in place and fed
(``test_polities.held``); war is switched on where a test needs the yearly choice, and
battles are otherwise resolved directly with ``war.battle``. Where a test needs a given
choice, the polity gets a scripted brain (``test_brains.scripted``).
"""

import numpy as np
import pytest

from conftest import small_model
from deluge.agents.polity import END_CAUSES
from deluge.data import chronicle
from deluge.data.events import entity
from deluge.systems import cohesion, identity, politics, war
from deluge.world.worlddata import Terrain, terrain_table
from test_polities import HELD, events, free_land, held, interior_cell, lone_polity, place

TRAITS = ("hierarchy", "militarism", "openness", "kin_vs_civic", "ritual_investment", "mobility_preference")


def culture(**values):
    c = np.full(len(TRAITS), 0.5)
    for k, v in values.items():
        c[TRAITS.index(k)] = v
    return c


def mark(m, sid, value):
    """Give a hand-placed settlement distinct appearance and dialect markers."""
    t = m.settlements
    t.appearance_mean[sid] = value
    t.dialect[sid] = 2.0 * value - 1.0
    return sid


def pair(m, gap=1, attacker_pop=20000.0, defender_pop=1000.0, defender_cell=None, attacker_cell=None):
    """An attacker polity and a one-settlement defender polity ``gap`` steps apart."""
    if defender_cell is None:
        defender_cell = interior_cell(m, max(gap, 2))
    if attacker_cell is None:
        attacker_cell = int(free_land(m, defender_cell, gap)[0])
    a, sa = lone_polity(m, attacker_cell, pop=attacker_pop, culture=culture(militarism=1.0))
    b, sb = lone_polity(m, defender_cell, lineage=1, pop=defender_pop, culture=culture(militarism=0.0))
    mark(m, sa, 0.9)
    mark(m, sb, 0.1)
    return a, sa, b, sb


def win_rate(m, a, sa, b, sb, trials=400, campaign=False):
    """Share of ``trials`` raids (or campaigns) ``a`` wins, each from the same starting state."""
    t = m.settlements
    pop = t.population.copy()
    store = t.store.copy()
    wins = 0
    for _ in range(trials):
        t.population[:] = pop
        t.store[:] = store
        wins += war.battle(m, war.forces(m), a, b, sa, sb, campaign)
    return wins / trials


# --- strength -----------------------------------------------------------------------------------


def test_strength_follows_the_formula():
    m = held()
    a, sa, b, sb = pair(m, gap=2)
    w = m.config.war
    F = war.forces(m)
    s_att, s_def = war.strengths(m, F, np.array([a]), np.array([b]), np.array([sb]))
    t, P = m.settlements, m.polities
    mob_a = min(w.mobilization_base + w.mobilization_militarism * 1.0 + w.mobilization_hierarchy * 0.5,
                w.mobilization_max)
    km = 2 * m.unit_km
    assert s_att[0] == pytest.approx(20000.0 * mob_a * 1.0 * (0.5 + P.cohesion[a]) * np.exp(-km / w.logistics_km))
    terrain = terrain_table(w.terrain_defense)[m.world.terrain.ravel()[t.cell[sb]]]
    mob_b = w.mobilization_base + w.mobilization_hierarchy * 0.5
    assert s_def[0] == pytest.approx(1000.0 * mob_b * (0.5 + P.cohesion[b]) * terrain * w.home_bonus)
    # Military techniques and cohesion raise it.
    g = m.knowledge
    t.knows[sa, g.index("copper_bronze")] = True
    P.cohesion[a] = 1.0
    stronger = war.strengths(m, war.forces(m), np.array([a]), np.array([b]), np.array([sb]))[0][0]
    assert stronger == pytest.approx(s_att[0] * 1.3 * 1.5 / (0.5 + 0.5))
    # Horses extend the reach from the capital.
    t.knows[sa, g.index("horse_riding")] = True
    assert war.forces(m).reach[a] == pytest.approx(w.logistics_km * 2.0)


def test_win_chance_is_logistic_in_the_strength_ratio():
    k = 2.0
    assert war.win_chance(1.0, 1.0, k) == pytest.approx(0.5)
    assert war.win_chance(3.0, 1.0, k) == pytest.approx(0.9)
    assert war.win_chance(0.0, 1.0, k) == 0.0
    assert war.win_chance(2.0, 1.0, k) > war.win_chance(1.5, 1.0, k)


# --- controlled scenarios: terrain and distance ---------------------------------------------------


def test_terrain_measurably_affects_outcomes():
    """The same attack on a defender on open plain and on rough ground (hills, mountain or
    wetland): the attacker wins far less often on rough ground."""
    m = held()
    defense = terrain_table(m.config.war.terrain_defense)
    terrain = m.world.terrain.ravel()
    free = ~m.is_sea & (m.occupant < 0)
    nb = m.geom.neighbors

    def site(codes):
        """A free cell of the given terrain with a free land neighbour (for the attacker)."""
        for c in np.flatnonzero(free & np.isin(terrain, codes)).tolist():
            ok = [n for n in nb[c].tolist() if n >= 0 and free[n]]
            if ok:
                return c, ok[0]
        raise AssertionError("no such site on the small world")

    plain = site([Terrain.PLAIN])
    rough = site([t for t in Terrain if defense[t] >= 1.5])
    rates = {}
    for name, (cell, att) in (("plain", plain), ("rough", rough)):
        mm = held()
        a, sa, b, sb = pair(mm, defender_cell=cell, attacker_cell=att, attacker_pop=800.0, defender_pop=1000.0)
        rates[name] = win_rate(mm, a, sa, b, sb)
    assert rates["plain"] - rates["rough"] > 0.15, rates


def test_distance_from_the_capital_measurably_affects_outcomes():
    """The same attack launched by a polity whose capital is next to the target, and by
    one whose capital is far away (the attacking settlement still next to the target)."""
    rates = {}
    for name, gap in (("near", 1), ("far", 8)):
        m = held()
        target = interior_cell(m, 9)
        cap = int(free_land(m, target, gap)[0])
        a, sa, b, sb = pair(m, defender_cell=target, attacker_cell=cap, attacker_pop=3000.0)
        src = sa
        if gap > 1:
            src = place(m, free_land(m, target, 1)[0], a, pop=500.0)
        rates[name] = win_rate(m, a, src, b, sb)
    assert rates["near"] - rates["far"] > 0.3, rates


# --- raids and campaigns -------------------------------------------------------------------------


def test_raid_takes_stores_but_no_land():
    m = held()
    a, sa, b, sb = pair(m, attacker_pop=50000.0)
    t = m.settlements
    t.store[sb] = 400.0
    store_a = float(t.store[sa])
    won = war.battle(m, war.forces(m), a, b, sa, sb, campaign=False)
    assert won  # overwhelming strength
    assert t.polity[sb] == b
    loot = m.config.war.raid_loot_share * 400.0
    assert t.store[sb] == pytest.approx(400.0 - loot) and t.store[sa] == pytest.approx(store_a + loot)
    won_events = events(m, "won")
    assert won_events[-1]["actors"] == (entity("polity", a), entity("polity", b), entity("settlement", sb))
    assert m.wars and m.polities.war_recent[a] == m.polities.war_recent[b] == 1.0
    assert m.stats["raids"] == 1 and m.stats["battles"] == 1


def test_battle_losses_scale_with_the_opponents_strength():
    m = held()
    a, sa, b, sb = pair(m, attacker_pop=5000.0, defender_pop=5000.0)
    t = m.settlements
    w = m.config.war
    F = war.forces(m)
    s_att, s_def = (float(v[0]) for v in war.strengths(m, F, np.array([a]), np.array([b]), np.array([sb])))
    pa, pb = float(t.population[sa]), float(t.population[sb])
    won = war.battle(m, F, a, b, sa, sb, campaign=False)
    lost_a, lost_b = pa - t.population[sa], pb - t.population[sb]
    fa, fb = (1.0, w.loser_casualty_factor) if won else (w.loser_casualty_factor, 1.0)
    assert lost_a == pytest.approx(min(w.casualty_rate * s_def * fa, w.attacker_casualty_cap * pa * F.mob[a]))
    assert lost_b == pytest.approx(min(w.casualty_rate * s_att * fb, w.defender_casualty_cap * pb))
    assert m.stats["war_deaths"] == pytest.approx(lost_a + lost_b)


def test_campaign_conquers_with_grievance_and_settlers():
    m = held()
    defender = interior_cell(m, 3)
    a, sa, b, sb = pair(m, defender_cell=defender, attacker_pop=50000.0)
    cap_b = sb
    sb2 = mark(m, place(m, free_land(m, defender, 2, exclude=[m.settlements.cell[sa]])[0], b, pop=800.0), 0.1)
    t, P = m.settlements, m.polities
    own_culture = t.culture[sb2].copy()
    app_before = t.appearance_mean[sb2].copy()
    won = war.battle(m, war.forces(m), a, b, sa, sb2, campaign=True)
    assert won
    assert t.polity[sb2] == a and t.subject_of[sb2] == a
    assert t.grievance[sb2] == m.config.war.grievance_conquest
    np.testing.assert_array_equal(t.culture[sb2], own_culture)  # keeps its own culture
    assert not np.allclose(t.appearance_mean[sb2], app_before)  # settlers blended in
    took = events(m, "took")[-1]
    assert took["actors"] == (entity("polity", a), entity("settlement", sb2), entity("polity", b))
    assert t.polity[cap_b] == b and P.ended[b] < 0  # the capital was not the target


def test_capital_capture_destroys_a_small_polity():
    m = held()
    a, sa, b, sb = pair(m, attacker_pop=50000.0)
    won = war.battle(m, war.forces(m), a, b, sa, sb, campaign=True)
    assert won
    P = m.polities
    assert P.ended[b] == m.year and END_CAUSES[P.end_cause[b]] == "conquered" and P.merged_into[b] == a
    assert events(m, "lost")[-1]["actors"] == (entity("polity", b), entity("settlement", sb), entity("polity", a))
    assert events(m, "died")[-1]["actors"] == (entity("polity", b), entity("polity", a))
    assert m.stats["capitals_taken"] == 1 and m.stats["polities_destroyed"] == 1
    lines = chronicle.lines(m, 10, actions={"lost", "died", "took"})
    assert any("lost its capital" in s for s in lines) and any("taken by" in s for s in lines)


def test_capital_capture_moves_the_capital_of_a_large_polity():
    m = held()
    defender = interior_cell(m, 4)
    a, sa, b, sb = pair(m, defender_cell=defender, attacker_pop=50000.0, defender_pop=1000.0)
    far = place(m, free_land(m, defender, 4)[0], b, pop=60000.0)
    P = m.polities
    P.cohesion[b] = 0.8
    won = war.battle(m, war.forces(m), a, b, sa, sb, campaign=True)
    assert won
    assert P.ended[b] < 0 and P.capital[b] == far
    assert P.cohesion[b] == pytest.approx(0.8 * (1 - m.config.war.capital_cohesion_loss))


# --- after conquest -------------------------------------------------------------------------------


def test_assimilation_grievance_and_dialect():
    """A conquered settlement's culture and dialect move toward its ruler's, faster under an
    open ruler; grievance decays; it stops being a subject once assimilated."""
    gaps = {}
    for openness in (0.1, 1.0):
        m = held({"secession_base": 0.0}, years=2000)
        defender = interior_cell(m, 3)
        a, sa, b, sb = pair(m, defender_cell=defender)
        sb2 = mark(m, place(m, free_land(m, defender, 2, exclude=[m.settlements.cell[sa]])[0], b, pop=800.0), 0.1)
        t, P = m.settlements, m.polities
        P.culture[a] = culture(militarism=1.0, openness=openness)
        P.culture[b] = culture(militarism=0.0, openness=0.0, hierarchy=0.0)
        t.culture[sb2] = P.culture[b]
        t.polity[sb2], t.subject_of[sb2], t.grievance[sb2] = a, a, 1.0
        d0 = np.linalg.norm(t.dialect[sb2] - t.dialect[sa])
        for _ in range(100):
            m.step()
        gaps[openness] = float(np.linalg.norm(t.culture[sb2] - P.culture[a]))
        assert np.linalg.norm(t.dialect[sb2] - t.dialect[sa]) < d0
        assert t.grievance[sb2] == pytest.approx((1 - m.config.war.grievance_decay) ** 100, rel=1e-6)
    assert gaps[1.0] < gaps[0.1]
    # Once grievance is low and culture close, the subject is assimilated.
    t.grievance[sb2] = 0.0
    t.culture[sb2] = P.culture[a]
    m.step()
    assert t.subject_of[sb2] == -1


def test_members_hold_their_polity_culture_and_seceders_take_their_own():
    m = held({"secession_base": 1e3, "secession_max": 1.0})
    defender = interior_cell(m, 9)
    a, sa, b, sb = pair(m, defender_cell=defender)
    t, P = m.settlements, m.polities
    subject = place(m, free_land(m, defender, 7)[0], a, pop=800.0)
    t.subject_of[subject], t.grievance[subject] = a, 1.0
    t.culture[subject] = np.zeros(len(TRAITS))
    war.assimilate(m)
    np.testing.assert_array_equal(t.culture[sa], P.culture[a])
    assert np.abs(t.culture[subject]).max() < 0.05  # still its own
    politics.secession(m)
    child = int(t.polity[subject])
    assert child != a and t.subject_of[subject] == -1 and t.grievance[subject] == 0.0
    assert np.abs(P.culture[child]).max() < 6 * m.config.polities.culture_mutation  # from its own culture


def test_grievance_raises_secession_and_lowers_cohesion():
    cohesions, seceded = {}, {}
    for g in (0.0, 1.0):
        m = held({"secession_base": 0.05, "secession_max": 1.0})
        origin = interior_cell(m, 6)
        pid, _ = lone_polity(m, origin, pop=5000.0)
        sid = place(m, free_land(m, origin, 5)[0], pid)
        t, P = m.settlements, m.polities
        t.grievance[sid] = g
        cohesion.run(m)
        cohesions[g] = float(P.cohesion[pid])
        P.cohesion[pid], P.pressure[pid] = 0.5, 0.0
        seceded[g] = 0
        for _ in range(400):
            t.polity[sid], t.grievance[sid] = pid, g
            politics.secession(m)
            seceded[g] += int(t.polity[sid] != pid)
    assert cohesions[1.0] < cohesions[0.0]
    # Expected 0.025 vs 0.025 x (1 + secession_grievance_weight) per try.
    assert seceded[1.0] > 2 * seceded[0.0] > 0, seceded


def test_conquered_techniques_reach_the_ruler():
    m = held(years=300)
    defender = interior_cell(m, 3)
    a, sa, b, sb = pair(m, defender_cell=defender)
    sb2 = mark(m, place(m, free_land(m, defender, 2, exclude=[m.settlements.cell[sa]])[0], b, pop=800.0), 0.1)
    t, g = m.settlements, m.knowledge
    j = g.index("sky_calendar")  # no prerequisites, no minerals
    t.knows[sb2, j], t.strength[sb2, j] = True, 1.0
    assert not t.knows[sa, j]
    t.polity[sb2], t.subject_of[sb2], t.grievance[sb2] = a, a, 1.0
    for _ in range(60):
        war.assimilate(m)
    assert t.knows[sa, j]


def test_recent_war_raises_frontier_pressure():
    m = held()
    a, sa, b, sb = pair(m, gap=6)  # not in contact: no cultural frontier
    P = m.polities
    cohesion.run(m)
    assert P.frontier[a] == 0.0
    P.war_recent[a] = 2.0
    cohesion.run(m)
    cc = m.config.polities.cohesion
    assert P.frontier[a] == pytest.approx(cc.war_frontier_weight * 2.0 / (2.0 + cc.war_frontier_half))


# --- choice ----------------------------------------------------------------------------------------


def test_aggression_drives_attacks_and_wars_stay_capped():
    """An aggressive brain attacks its neighbours, whose brains (aggression 0) never attack;
    an attacker never fights more than max_wars polities within the war memory."""
    from test_brains import SATURATE, scripted

    over = dict(HELD, war={"attack_rate": 1.0, "max_wars": 1, "battle_log_min_losses": 0.0})
    m = small_model(over, years=200)
    m.step()
    origin = interior_cell(m, 4)
    a, _ = lone_polity(m, origin, pop=3000.0)
    b, _ = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=3000.0)
    c, _ = lone_polity(m, free_land(m, origin, 2)[0], lineage=2, pop=3000.0)
    scripted(m, a, aggression=1.0, campaign=0.0, target={"bias": SATURATE})
    for pid in (b, c):
        scripted(m, pid, aggression=0.0)
    for _ in range(100):
        m.step()
    battles = [(r["year"], r["actors"][0], r["actors"][1]) for r in events(m, "fought")]
    by_a = [(y, d) for y, x, d in battles if x == entity("polity", a)]
    assert by_a
    assert not [x for _, x, _ in battles if x in (entity("polity", b), entity("polity", c))]
    memory = m.config.war.war_memory_years
    for y, _ in by_a:
        assert len({d for yy, d in by_a if y - memory <= yy <= y}) <= 1


def test_war_choice_ignores_appearance_and_dialect_while_marker_weights_are_zero():
    """Same seeds and an active war between brains whose marker weights are zero (the
    founding default, frozen so no mutation moves them); one run has every settlement's
    appearance and dialect replaced: every war decision and outcome is identical."""
    from test_brains import SATURATE, scripted

    over = dict(HELD, war={"attack_rate": 1.0}, brains={"frozen": True}, polities={"marker_mutation": 0.0})

    def scenario():
        m = small_model(over, years=200)
        m.step()
        origin = interior_cell(m, 4)
        a, _ = lone_polity(m, origin, pop=3000.0, culture=culture(militarism=1.0))
        b, _ = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=3000.0, culture=culture(militarism=1.0))
        for pid in (a, b):
            scripted(m, pid, aggression=1.0, target={"bias": SATURATE})
        return m

    x, y = scenario(), scenario()
    t = y.settlements
    rng = np.random.default_rng(5)
    ids = t.alive_ids()
    t.appearance_mean[ids] = rng.random(t.appearance_mean[ids].shape)
    t.dialect[ids] = rng.normal(0.0, 5.0, t.dialect[ids].shape)
    for _ in range(60):
        x.step()
        y.step()
    dx, dy = x.datacollector.get_model_vars_dataframe(), y.datacollector.get_model_vars_dataframe()
    assert dx["battles"].sum() > 0
    np.testing.assert_array_equal(dx.to_numpy(), dy.to_numpy())
    n = x.settlements.n
    for col in ("population", "store", "polity", "grievance", "subject_of", "culture"):
        np.testing.assert_array_equal(getattr(x.settlements, col)[:n], getattr(y.settlements, col)[:n])


def test_war_switch():
    from test_brains import SATURATE, scripted

    m = small_model(dict(HELD, war={"enabled": False, "attack_rate": 1.0}), years=100)
    m.step()
    origin = interior_cell(m, 4)
    a, _ = lone_polity(m, origin, pop=3000.0, culture=culture(militarism=1.0))
    scripted(m, a, aggression=1.0, target={"bias": SATURATE})
    lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=3000.0)
    m.run()
    assert m.datacollector.get_model_vars_dataframe()["battles"].sum() == 0


def test_identity_measures_are_untouched_by_markers_through_war():
    """``identity.blend`` is the only marker writer war uses (the no-reader test checks
    the source); the settlers' weight follows the people moved."""
    m = held()
    a, sa, b, sb = pair(m)
    t = m.settlements
    h0, s0 = t.appearance_mean[sb].copy(), t.appearance_mean[sa].copy()
    identity.blend(m, sb, sa, people=t.population[sb])  # as many settlers as residents
    np.testing.assert_allclose(t.appearance_mean[sb], (h0 + s0) / 2)
