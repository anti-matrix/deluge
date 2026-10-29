"""Round 14 step 3: "How they look" in the polity inspector.

The polity's mean appearance traits, beside its founding people's now and its founders' at
year 0, and how different the polities in contact look to it: the perceived difference its
brain reads for their settlements as targets. Read only (the no-capability tests in
``test_identity.py`` still hold: nothing outside identity.py reads the markers).
"""

import re

import numpy as np
import pytest

from conftest import small_model
from deluge.brains.features import TARGET_FEATURES, target_features
from deluge.systems import identity
from deluge.viz import app, panels


@pytest.fixture(scope="module")
def grown():
    m = small_model({"polities": {"secession_base": 0.5}}, years=200)
    t = m.settlements
    year0 = {int(t.lineage[s]): t.appearance_mean[s].copy() for s in t.alive_ids().tolist()}
    m.run()
    return m, year0


def test_looks_show_the_polity_its_people_and_its_founders(grown):
    m, year0 = grown
    P, t = m.polities, m.settlements
    pid = int(max(P.alive_ids(), key=lambda p: P.population[p]))
    seen = identity.looks(m, pid)
    ids = t.alive_ids()
    mine = ids[t.polity[ids] == pid]
    np.testing.assert_allclose(seen["polity"], t.population[mine] @ t.appearance_mean[mine] / t.population[mine].sum())
    np.testing.assert_allclose(seen["lineage"], identity.lineage_means(m)[0][int(P.lineage[pid])])
    np.testing.assert_allclose(seen["founders"], year0[int(P.lineage[pid])])
    html = panels.inspector(m, pid, app.VIZ)
    assert "How they look" in html
    traits = m.config.ancestry.appearance_traits
    assert re.findall(r'data-trait="(\w+)"', html) == list(traits)
    words = panels.inspector_words(app.VIZ)
    for tr in traits:
        assert words["appearance"][tr][0] in html
    for key in ("polity", "lineage", "founders"):
        for v in seen[key]:
            assert f"{v:.2f}" in html


def test_neighbour_difference_is_what_the_brain_perceives(grown):
    m, _ = grown
    P, t = m.polities, m.settlements
    pid = next(p for p in P.alive_ids().tolist() if identity.neighbour_looks(m, p))
    near = identity.neighbour_looks(m, pid)
    assert [d for _, d in near] == sorted(d for _, d in near)
    # The brain's target feature for each settlement of the first neighbour in contact.
    q, d = near[0]
    ids = t.alive_ids()
    mine = ids[t.polity[ids] == pid]
    contact = np.append(m.occupant, -1)[m.contact_cells[t.cell[mine]]].ravel()
    theirs = np.unique(contact[contact >= 0])
    theirs = theirs[t.polity[theirs] == q]
    f = target_features(m, np.full(len(theirs), pid), np.full(len(theirs), q), theirs, np.zeros(len(theirs)))
    assert d == pytest.approx(f[:, TARGET_FEATURES.index("appearance_distance")].mean())
    line = re.search(r'data-looks="neighbours">([^<]*)<', panels.inspector(m, pid, app.VIZ)).group(1)
    assert f"{d:.2f}" in line and line.count(",") >= 0


def test_blind_perception_says_so():
    m = small_model({"ancestry": {"blind_perception": True}}, years=30)
    m.run()
    pid = int(m.polities.alive_ids()[0])
    assert "blind perception" in panels.inspector(m, pid, app.VIZ)
