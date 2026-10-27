"""Round 13b: the page follows the active run.

The page starts on world 1 / history 1; the seeds are switched to world 5343 / history 165243
and Reset pressed. Then the legend, the chronicle's people filter and the myths dropdown hold
only the new run's peoples and polities, and the chronicle shows entries without the filter
being touched. (Small test world, so the test stays fast; the seeds are the ones played.)
"""

import ipyvuetify as v
import pytest
import solara

from conftest import small_model
from deluge.viz import app, portrayal

OLD, NEW = (1, 1), (5343, 165243)


def _run(seeds, years):
    m = small_model(world_seed=seeds[0], history_seed=seeds[1])
    for _ in range(years):
        m.step()
    return m


@pytest.fixture
def page(monkeypatch):
    old = _run(OLD, 150)
    monkeypatch.setattr(app, "_initial", [old])
    # Reset builds the new run on the small world too, then the test steps it.
    monkeypatch.setattr(app, "DelugeModel", lambda **seeds: small_model(**seeds))
    box, rc = solara.render(app.Page(), handle_error=False)
    yield old, rc
    rc.close()


def _html(rc) -> str:
    return " ".join(w.unsafe_innerHTML for w in rc.find(solara.widgets.HTML).widgets)


def _select(rc, label):
    return rc.find(v.Select, label=label).widget


def _set_seed(rc, label, value):
    w = rc.find(v.TextField, label=label).widget
    w.v_model = str(value)
    w.fire_event("focusout", None)


def test_names_filters_and_myths_follow_the_active_run(page):
    old, rc = page
    old_peoples = portrayal.people_names(old)
    assert _select(rc, "Peoples").v_model == old_peoples
    assert all(n in _html(rc) for n in old_peoples)  # the legend
    _set_seed(rc, "World seed: the map (applies on Reset)", NEW[0])
    _set_seed(rc, "History seed (applies on Reset)", NEW[1])
    rc.find(v.Btn, children=["Reset"]).widget.click()
    rc.find(v.Btn, children=["+100 yr"]).widget.click()
    rc.find(v.Btn, children=["+100 yr"]).widget.click()
    model = _active_model(rc)
    assert (model.config.run.world_seed, model.config.run.history_seed) == NEW and model.year >= 200
    new_peoples = portrayal.people_names(model)
    assert set(new_peoples).isdisjoint(old_peoples)
    html = _html(rc)
    assert all(n in html for n in new_peoples) and not any(n in html for n in old_peoples)
    # The chronicle's filter: every people of the new run, chosen by default; entries shown.
    peoples = _select(rc, "Peoples")
    assert peoples.items == new_peoples and peoples.v_model == new_peoples
    texts = [w.unsafe_innerHTML for w in rc.find(solara.widgets.HTML).widgets]
    assert not any("Nothing yet." in t for t in texts)
    # The myths dropdown: the new run's lineages and living polities with records, nothing old.
    from analysis import bard

    items = _select(rc, "People or polity").items
    want = [n for g, n in enumerate(new_peoples) if bard.has_records(model, "lineage", g)]
    assert items[: len(want)] == want and len(items) > len(want)
    # The run had no record at Reset: the dropdown still picks the first one now, and it can be told.
    assert _select(rc, "People or polity").v_model == items[0]
    assert not rc.find(v.Btn, children=["Tell it as their myth"]).widget.disabled
    assert not any(any(o in it for o in old_peoples) for it in items)
    P = model.polities
    pids = [int(it.rsplit("polity ", 1)[1]) for it in items[len(want):]]
    assert all(P.ended[p] < 0 and model.memory.stores.get(p) for p in pids)
    assert set(pids) == {p for p in P.alive_ids().tolist() if model.memory.stores.get(p)}


def _active_model(rc):
    """The model the page shows now (the one the inspector's and the chronicle's session hold)."""
    import gc

    sessions = [o for o in gc.get_objects() if isinstance(o, app.Session)]
    (model,) = {id(s.model): s.model for s in sessions if s.model.config.run.world_seed == NEW[0]}.values()
    return model
