"""Round 5: appearance and dialect (spec "Ancestry, appearance and identity").

Markers only: the no-capability tests check that no system other than perception and
mixing (``deluge/systems/identity.py``) reads the fields, and that rewriting them changes
nothing else in a run.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from conftest import small_model
from deluge.systems import identity, politics
from test_polities import free_land, held, interior_cell, lone_polity, place

PACKAGE = Path(__file__).resolve().parent.parent / "deluge"
FIELDS = re.compile(r"\.(appearance_mean|appearance_var|dialect|teachers_appearance)\b")
ALLOWED = {"agents/settlement.py", "systems/identity.py"}  # storage, and perception and mixing


def test_no_other_system_reads_the_markers():
    readers = set()
    for path in PACKAGE.rglob("*.py"):
        if FIELDS.search(path.read_text(encoding="utf-8")):
            readers.add(path.relative_to(PACKAGE).as_posix())
    assert readers == ALLOWED


def test_markers_have_no_capability_effect():
    """Same seeds, perception blind (so the only channel the markers have, perception
    through evolved weights, is shut); one run has every founding people's appearance and
    dialect replaced. Everything but the markers is identical after 300 years.
    (``test_brains`` checks the same with perception on and marker weights at zero.)"""
    blind = {"ancestry": {"blind_perception": True}}
    a, b = small_model(blind, years=300), small_model(blind, years=300)
    t = b.settlements
    rng = np.random.default_rng(99)
    ids = t.alive_ids()
    t.appearance_mean[ids] = rng.random(t.appearance_mean[ids].shape)
    t.appearance_var[ids] = rng.random(t.appearance_var[ids].shape) * 0.1
    t.dialect[ids] = rng.normal(0.0, 5.0, t.dialect[ids].shape)
    a.run()
    b.run()
    pd.testing.assert_frame_equal(a.datacollector.get_model_vars_dataframe(), b.datacollector.get_model_vars_dataframe())
    n = a.settlements.n
    assert b.settlements.n == n
    for col in ("population", "store", "cell", "polity", "ended", "knows", "shares"):
        np.testing.assert_array_equal(getattr(a.settlements, col)[:n], getattr(b.settlements, col)[:n])
    pn = a.polities.n
    for col in ("cohesion", "culture", "pressure", "parent", "ended"):
        np.testing.assert_array_equal(getattr(a.polities, col)[:pn], getattr(b.polities, col)[:pn])
    assert len(a.events) == len(b.events)
    assert not np.allclose(a.settlements.appearance_mean[:n], b.settlements.appearance_mean[:n])


def test_founding_peoples_are_distinct():
    m = small_model(years=2)
    a = m.config.ancestry
    t = m.settlements
    sids = [f.settlement for f in m.lineages.founders]
    means = t.appearance_mean[sids]
    d = np.linalg.norm(means[:, None] - means[None], axis=2)[np.triu_indices(len(sids), 1)]
    assert (d >= a.founder_min_distance).all()
    assert ((means >= a.founder_mean_min) & (means <= a.founder_mean_max)).all()
    np.testing.assert_allclose(t.appearance_var[sids], a.founder_variance)
    assert len({tuple(x) for x in t.dialect[sids].round(9)}) == len(sids)
    assert m.teachers_appearance.shape == (len(a.appearance_traits),)
    fixed = small_model({"ancestry": {"teachers_appearance": [0.1, 0.2, 0.3, 0.4]}}, years=2)
    np.testing.assert_allclose(fixed.teachers_appearance, [0.1, 0.2, 0.3, 0.4])


def test_daughters_inherit_and_joining_bands_blend():
    m = held()
    origin = interior_cell(m, 3)
    pid, host = lone_polity(m, origin, pop=3000.0)
    t = m.settlements
    t.appearance_mean[host], t.dialect[host] = 0.2, 0.0
    child = m.found_settlement(int(free_land(m, origin, 1)[0]), 1000.0, int(t.lineage[host]), host)
    np.testing.assert_array_equal(t.appearance_mean[child], t.appearance_mean[host])
    np.testing.assert_array_equal(t.dialect[child], t.dialect[host])
    t.appearance_mean[child], t.dialect[child] = 0.6, 4.0
    from deluge.systems.demography import join_kin
    join_kin(m, child, host)
    np.testing.assert_allclose(t.appearance_mean[host], 0.3)  # 3000 at 0.2 and 1000 at 0.6
    np.testing.assert_allclose(t.dialect[host], 1.0)
    assert (t.appearance_var[host] > t.appearance_var[child] - 1e-12).all()  # mixing adds variance


def _pair(intermarriage, same_polity=True, drift=0.0):
    m = held(years=400)
    over = {"drift_sigma": drift, "dialect_drift_sigma": drift}
    m.config = m.config.with_overrides({"ancestry": over})
    origin = interior_cell(m, 3)
    pid, a = lone_polity(m, origin)
    if same_polity:
        b = place(m, free_land(m, origin, 1)[0], pid)
    else:
        _, b = lone_polity(m, free_land(m, origin, 1)[0], lineage=1)
    m.polities.intermarriage[pid] = intermarriage
    t = m.settlements
    t.appearance_mean[a], t.appearance_mean[b] = 0.1, 0.9
    t.dialect[a], t.dialect[b] = -2.0, 2.0
    return m, a, b


def gap(m, a, b, field):
    x = getattr(m.settlements, field)
    return float(np.linalg.norm(x[a] - x[b]))


def test_appearance_mixes_inside_a_polity_by_intermarriage():
    results = {}
    for im in (0.0, 1.0):
        m, a, b = _pair(im)
        g0 = gap(m, a, b, "appearance_mean")
        for _ in range(100):
            identity.run(m)
            m.year += 1
        results[im] = gap(m, a, b, "appearance_mean") / g0
    assert results[0.0] == 1.0 and results[1.0] < 0.5


def test_appearance_does_not_mix_across_polities_but_dialect_converges():
    m, a, b = _pair(1.0, same_polity=False)
    ga, gd = gap(m, a, b, "appearance_mean"), gap(m, a, b, "dialect")
    for _ in range(100):
        identity.run(m)
        m.year += 1
    assert gap(m, a, b, "appearance_mean") == ga
    assert gap(m, a, b, "dialect") < 0.6 * gd


def test_drift_is_faster_in_small_populations():
    m = held(years=10)
    t = m.settlements
    origin = interior_cell(m, 9)
    pid, small = lone_polity(m, origin, pop=50.0)
    _, big = lone_polity(m, free_land(m, origin, 8)[0], lineage=1, pop=50000.0)
    m.polities.intermarriage[:] = 0.0
    t.appearance_mean[[small, big]] = 0.5
    start = {s: t.appearance_mean[s].copy() for s in (small, big)}
    gen = m.config.ancestry.generation_years
    moved = {small: 0.0, big: 0.0}
    for k in range(1, 41):
        m.year = k * gen
        identity.run(m)
        for s in moved:
            moved[s] += float(np.abs(t.appearance_mean[s] - start[s]).sum())
            start[s] = t.appearance_mean[s].copy()
    assert moved[small] > 3 * moved[big] > 0


def test_peoples_blend_after_a_union():
    """Two founding peoples' polities in contact merge (forced, as mergers are inactive
    this round); their settlements' appearance then converges through intermarriage."""
    m, a, b = _pair(1.0, same_polity=False)
    big = politics.merge(m, int(m.settlements.polity[a]), int(m.settlements.polity[b]))
    m.polities.intermarriage[big] = 1.0
    g0 = gap(m, a, b, "appearance_mean")
    for _ in range(100):
        identity.run(m)
        m.year += 1
    assert gap(m, a, b, "appearance_mean") < 0.5 * g0
    assert m.settlements.lineage[a] != m.settlements.lineage[b]


def test_environmental_selection_switch():
    m = held(years=10)
    assert not m.config.ancestry.selection_enabled
    on = held(years=10)
    on.config = on.config.with_overrides({"ancestry": {"selection_enabled": True, "selection_rate": 0.5,
                                                       "drift_sigma": 0.0}})
    t = on.settlements
    ids = t.alive_ids()
    j = on.config.ancestry.appearance_traits.index("skin_tone")
    lat = np.abs(on.geom.latitude(on.config.world.lat_south, on.config.world.lat_north)[t.cell[ids]])
    a = on.config.ancestry
    optimum = a.selection_optimum_equator + (a.selection_optimum_pole - a.selection_optimum_equator) * lat / 90
    before = np.abs(t.appearance_mean[ids, j] - optimum)
    on.year = a.generation_years
    identity.run(on)
    after = np.abs(t.appearance_mean[ids, j] - optimum)
    assert (after < before + 1e-12).all() and (after < before).any()
