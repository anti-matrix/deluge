"""Round 14 step 4: alliances (``systems/alliances.py``), driven by the polity brains.

Scores come from each brain's alliance head (``brains.alliance_w``); the tests set those
weights directly (a large bias: the score is ~1, a negative one: ~0).
"""

import numpy as np
import pytest

from deluge.brains.features import ALLIANCE_FEATURES
from deluge.brains.network import OUT
from deluge.data.bounds import battle_death_limit, bounds_path, load_bounds
from deluge.data.events import ACTION_CODE, entity
from deluge.naming import Speaker
from deluge.systems import alliances, war
from test_polities import free_land, held, interior_cell, lone_polity


def like(m, p, value: bool) -> None:
    """Polity ``p``'s alliance head scores everyone ~1 (True) or ~0 (False)."""
    w = np.zeros(len(ALLIANCE_FEATURES))
    w[ALLIANCE_FEATURES.index("bias")] = 20.0 if value else -20.0
    m.brains.alliance_w[p] = w


def neighbours(m, n=2, pop=3000.0):
    origin = interior_cell(m, 4)
    cells = [origin] + [int(c) for c in free_land(m, origin, 1)[: n - 1]]
    out = [lone_polity(m, c, lineage=k % 5, pop=pop) for k, c in enumerate(cells)]
    return [p for p, _ in out], [s for _, s in out]


def events(m, verb):
    a = np.asarray(m.events.action)
    return [i for i in np.flatnonzero(a == ACTION_CODE[verb]).tolist()]


def seek(m, p, value=1.0):
    m.brains.output[p, OUT["alliance_seeking"]] = value


def test_a_seeking_polity_proposes_and_the_other_accepts_or_not():
    m = held(years=10)
    (p, q), _ = neighbours(m)
    m.year = 7  # not a re-scoring year
    seek(m, p)
    seek(m, q, 0.0)
    like(m, p, True)
    like(m, q, False)
    alliances.run(m)
    assert not m.alliances  # q's score for p is below the threshold: no
    like(m, q, True)
    alliances.run(m)
    assert alliances.allied(m, p, q) and m.alliances[(min(p, q), max(p, q))] == 7
    (i,) = events(m, "allied")
    assert m.events.actors[i] == (entity("polity", p), entity("polity", q))
    text = Speaker(m).event(i)
    assert f"{Speaker(m).entity(entity('polity', p))} and " in text and " swore alliance (6,000 people)." in text
    # Without the seeking output above the threshold, nobody proposes.
    m2 = held(years=10)
    (p2, q2), _ = neighbours(m2)
    seek(m2, p2, 0.4)
    like(m2, p2, True)
    like(m2, q2, True)
    alliances.run(m2)
    assert not m2.alliances


def test_alliances_are_rescored_each_generation_and_end_when_a_score_falls():
    m = held(years=10)
    (p, q), _ = neighbours(m)
    m.year = 7
    seek(m, p)
    like(m, p, True)
    like(m, q, True)
    alliances.run(m)
    assert alliances.allied(m, p, q)
    seek(m, p, 0.0)
    like(m, q, False)
    m.year = 24
    alliances.run(m)  # not a re-scoring year: it holds
    assert alliances.allied(m, p, q)
    m.year = 25
    alliances.run(m)
    assert not alliances.allied(m, p, q)
    (i,) = events(m, "alliance_ended")
    assert m.events.actors[i][0] == entity("polity", q)  # the side whose score fell
    assert m.alliance_history[-1][2:] == (7, 25, "rescored", q)


def test_attacking_an_ally_is_a_betrayal():
    m = held(years=10)
    (p, q), (sp, sq) = neighbours(m)
    m.year = 7
    seek(m, p)
    like(m, p, True)
    like(m, q, True)
    alliances.run(m)
    assert alliances.keeps_faith(m, [p], [q])[0]  # no target while it values the alliance
    like(m, p, False)
    assert not alliances.keeps_faith(m, [p], [q])[0]
    before = float(m.settlements.grievance[sq])
    import deluge.systems.war as w

    orig = w.choose
    w.choose = lambda model, F: [(p, q, sp, sq, False, 0)]
    try:
        w.fight(m)
    finally:
        w.choose = orig
    assert not alliances.allied(m, p, q)
    (i,) = events(m, "betrayed")
    assert m.events.actors[i] == (entity("polity", p), entity("polity", q))
    assert m.settlements.grievance[sq] >= before + m.config.alliances.betrayal_grievance - 1e-9
    assert " broke faith with " in Speaker(m).event(i)
    assert m.alliance_history[-1][4:] == ("betrayed", p)


def test_an_ally_may_join_the_defence_and_shares_the_dead():
    m = held(years=10)
    (a, b, c), (sa, sb, sc) = neighbours(m, 3, pop=20000.0)
    m.year = 7
    m.wars[(min(a, b), max(a, b))] = m.wars[(min(a, c), max(a, c))] = 6  # a is their enemy
    for x in (b, c):
        seek(m, x)
        like(m, x, True)
        m.brains.alliance_w[x, ALLIANCE_FEATURES.index("at_war")] = -40.0  # no alliance with an enemy
    like(m, a, False)
    alliances.run(m)
    assert alliances.allied(m, b, c) and not alliances.allied(m, a, b)
    m.brains.output[c, OUT["campaign"]] = 1.0  # it always joins
    F = war.forces(m)
    _, sd_alone = war.strengths(m, F, np.array([a]), np.array([b]), np.array([sb]), np.array([sa]))
    pop_c = float(m.settlements.population[sc])
    war.battle(m, F, a, b, sa, sb, campaign=False)
    (i,) = events(m, "joined_war")
    assert m.events.actors[i] == (entity("polity", c), entity("polity", b), entity("polity", a),
                                  entity("settlement", sb))
    assert m.events.link[i] == events(m, "fought")[-1]
    assert m.settlements.population[sc] < pop_c  # the ally lost people
    assert (min(a, c), max(a, c)) in m.wars
    assert "came to the aid of" in Speaker(m).event(i)
    log = m._campaigns.columns()
    share = load_bounds(bounds_path(m.config))["war_deaths"]["fighting_age_share"]
    dead = log["dead_attacker"][-1] + log["dead_defender"][-1] + log["dead_allies"][-1]
    assert log["dead_allies"][-1] > 0 and log["allied_mobilized"][-1] > 0
    assert dead <= battle_death_limit(log["target_people"][-1], log["present_people"][-1], log["mobilization"][-1],
                                      share, log["allied_mobilized"][-1]) + 1e-6
    assert sd_alone[0] > 0


def test_no_cap_by_default_and_a_cap_for_experiments():
    m = held(years=10)
    origin = interior_cell(m, 4)
    hub, _ = lone_polity(m, origin, pop=3000.0)
    others = [lone_polity(m, int(c), lineage=1, pop=3000.0)[0] for c in free_land(m, origin, 1)[:5]]
    assert m.config.alliances.max_alliances == 0
    m.year = 7
    for x in [hub] + others:
        like(m, x, True)
    seek(m, hub)
    for _ in range(5):
        alliances.run(m)
    assert len(alliances.allies_of(m, hub)) == 5  # no limit
    m2 = held(years=10, more={"alliances": {"max_alliances": 3}})
    hub2, _ = lone_polity(m2, origin, pop=3000.0)
    others2 = [lone_polity(m2, int(c), lineage=1, pop=3000.0)[0] for c in free_land(m2, origin, 1)[:5]]
    m2.year = 7
    for x in [hub2] + others2:
        like(m2, x, True)
    seek(m2, hub2)
    for _ in range(5):
        alliances.run(m2)
    assert len(alliances.allies_of(m2, hub2)) == 3


def test_long_alliances_raise_affinity():
    m = held(years=10)
    (p, q), _ = neighbours(m)
    m.year = 7
    seek(m, p)
    like(m, p, True)
    like(m, q, True)
    alliances.run(m)
    m.year = 17
    young = alliances.affinity(m, [p], [q])[0]
    m.year = 307
    old = alliances.affinity(m, [p], [q])[0]
    assert 0 < young < old <= m.config.alliances.affinity_weight
    assert alliances.affinity(m, [p], [p + 1000 if False else q])[0] == old


def test_chronicle_tells_alliances():
    from deluge.data import chronicle

    m = held(years=10)
    (p, q), _ = neighbours(m)
    m.year = 7
    seek(m, p)
    like(m, p, True)
    like(m, q, True)
    alliances.run(m)
    view = chronicle.Chronicle(m)
    view.sync()
    told = {e["event"] for e in view.entries}
    assert events(m, "allied")[0] in told
