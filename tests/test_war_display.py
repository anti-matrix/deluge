"""Round 13 step 4: the war display (``viz/portrayal.war_display``, from ``data/campaigns.py``).

- An arrow runs from the settlement an attack set out from to the one attacked; it is dashed
  if and only if the attack's sea crossing is in the log, linked to the attack's battle event.
- A supply line runs from the attacker's capital to where it set out (none from the capital),
  its opacity the logistics decay of the attacker's strength.
- The banners match the year's attacks: a sword and its count for attacks made, a shield and
  its count for attacks suffered.
"""

import numpy as np
import pytest

from conftest import small_model
from deluge.data.events import ACTION_CODE
from deluge.systems import war
from deluge.viz import portrayal
from deluge.viz.app import VIZ
from test_polities import free_land, lone_polity, place
from test_war import strait


@pytest.fixture(scope="module")
def warring():
    m = small_model({"polities": {"secession_base": 0.5}}, years=300)
    m.run()
    assert len(m._campaigns) > 0
    return m


def crossed_link(m, log, k) -> bool:
    """Whether attack ``k`` of the log has a sea crossing linked to its battle event."""
    x = int(log["crossed"][k])
    ev = m.events
    return x >= 0 and ev.action[x] == ACTION_CODE["crossed"] and ev.link[x] == log["event"][k]


def test_arrows_run_from_the_set_out_settlement_dashed_across_the_sea():
    m, a, sa, b, sb = strait(2, sailing=True)
    war.battle(m, war.forces(m), a, b, sa, sb, True, sea=2)
    arrow = portrayal.war_display(m, VIZ, m.year)["arrows"][-1]
    pos, t = m.geom.positions, m.settlements
    assert arrow[:4] == [float(pos[t.cell[sa], 0]), float(pos[t.cell[sa], 1]),
                         float(pos[t.cell[sb], 0]), float(pos[t.cell[sb], 1])]
    log = m._campaigns.columns()
    assert crossed_link(m, log, arrow[9]) and arrow[4] is True and arrow[6:8] == [a, b]
    # Over land: solid.
    m2 = small_model(years=2)
    m2.run()
    c1 = int(np.flatnonzero(~m2.is_sea & (m2.occupant < 0))[40])
    p1, s1 = lone_polity(m2, c1, pop=20000.0)
    p2, s2 = lone_polity(m2, int(free_land(m2, c1, 2)[0]), lineage=1, pop=1000.0)
    war.battle(m2, war.forces(m2), p1, p2, s1, s2, False, sea=0)
    arrow = portrayal.war_display(m2, VIZ, m2.year)["arrows"][-1]
    assert arrow[4] is False and not crossed_link(m2, m2._campaigns.columns(), arrow[9])


def test_dash_style_matches_the_crossed_link_through_a_run(warring):
    m = warring
    log = m._campaigns.columns()
    n = 0
    for year in range(0, m.year, 3):
        for arrow in portrayal.war_display(m, VIZ, year)["arrows"]:
            k = arrow[9]
            assert arrow[4] == crossed_link(m, log, k)
            assert year - VIZ.war_fade_years < log["year"][k] <= year and log["attacker"][k] == arrow[6]
            n += 1
    assert n > 0


def test_supply_line_opacity_is_the_logistics_decay(warring):
    # The war model's distance term: strength = force x exp(-km from the capital / reach).
    m2, a, sa, b, sb = strait(2, sailing=True)
    t = m2.settlements
    F = war.forces(m2)
    sa_now, _ = war.strengths(m2, F, np.array([a]), np.array([b]), np.array([sb]))
    force = (F.people * F.mob * F.tech * (0.5 + m2.polities.cohesion[: len(F.people)]))[a]
    war.battle(m2, F, a, b, sa, sb, False, sea=2)
    log = m2._campaigns.columns()
    km = m2.geom.hex_distance(F.capital_cell[a], t.cell[sb]) * m2.unit_km
    assert log["decay"][-1] == np.exp(-km / F.reach[a])
    assert sa_now[0] == pytest.approx(force * log["decay"][-1], rel=1e-12)
    # Every supply line of a run: from the capital to where its attack set out, opacity = its decay.
    m = warring
    lg = m._campaigns.columns()
    pos = m.geom.positions
    lines = 0
    for year in range(0, m.year, 3):
        d = portrayal.war_display(m, VIZ, year)
        for arrow in d["arrows"]:
            k = arrow[9]
            if lg["src"][k] == lg["capital"][k]:
                assert arrow[8] == -1  # set out from the capital: no supply line
                continue
            line = d["supply"][arrow[8]]
            assert line[4] == lg["decay"][k] and line[6] == d["arrows"].index(arrow)
            assert line[:4] == [float(pos[lg["capital_cell"][k], 0]), float(pos[lg["capital_cell"][k], 1]),
                                float(pos[lg["src_cell"][k], 0]), float(pos[lg["src_cell"][k], 1])]
            lines += 1
    # An attack set out from another settlement than the capital: one supply line, from the capital.
    m3 = small_model(years=2)
    m3.run()
    c1 = int(np.flatnonzero(~m3.is_sea & (m3.occupant < 0))[40])
    p1, cap = lone_polity(m3, c1, pop=20000.0)
    c2 = int(free_land(m3, c1, 3)[0])
    s2 = place(m3, c2, p1, 3000.0)
    p3, s3 = lone_polity(m3, int(free_land(m3, c2, 2, exclude=[c1])[0]), lineage=1, pop=500.0)
    war.battle(m3, war.forces(m3), p1, p3, s2, s3, False, sea=0)
    d = portrayal.war_display(m3, VIZ, m3.year)
    arrow = d["arrows"][-1]
    line = d["supply"][arrow[8]]
    lg = m3._campaigns.columns()
    assert lg["src"][-1] == s2 and lg["capital"][-1] == cap and line[4] == lg["decay"][-1] < 1.0
    pos = m3.geom.positions
    assert line[:4] == [float(pos[c1, 0]), float(pos[c1, 1]), float(pos[c2, 0]), float(pos[c2, 1])]


def test_banners_match_the_years_attacks(warring):
    m = warring
    P = m.polities
    shown = 0
    for year in range(0, m.year, 3):
        d = portrayal.war_display(m, VIZ, year)
        made, against = {}, {}
        for arrow in d["arrows"]:
            made[arrow[6]] = made.get(arrow[6], 0) + 1
            against[arrow[7]] = against.get(arrow[7], 0) + 1
        living = {p for p in set(made) | set(against) if P.ended[p] < 0}
        assert {b[3] for b in d["banners"]} == living
        for x, y, color, p, n_made, n_against in d["banners"]:
            assert n_made == made.get(p, 0) and n_against == against.get(p, 0)  # sword / shield counts
            assert color == portrayal.lineage_palette(VIZ)[int(P.lineage[p])]
            shown += 1
    assert shown > 0
    # At the shown year, banners stand on the capitals.
    for x, y, _, p, _, _ in portrayal.war_display(m, VIZ)["banners"]:
        cap = m.geom.positions[m.settlements.cell[P.capital[p]]]
        assert (x, y) == (float(cap[0]), float(cap[1]))


def test_arrows_fade_and_the_log_stays_out_of_the_digest(warring):
    from deluge.data.state import state_digest

    m = warring
    log = m._campaigns.columns()
    year = int(log["year"][-1])
    for k in range(VIZ.war_fade_years + 1):
        alphas = [a[5] for a in portrayal.war_display(m, VIZ, year + k)["arrows"]]
        if k < VIZ.war_fade_years:
            assert alphas and max(alphas) == pytest.approx(1.0 - k / VIZ.war_fade_years, abs=1e-4)
        else:
            assert alphas == []
    saved = m._campaigns
    try:
        before = state_digest(m)
        m._campaigns = None
        assert state_digest(m) == before
    finally:
        m._campaigns = saved
