"""Round 13 step 1: names of places on demand (``deluge/gazetteer.py``).

- A cell is named only while a settlement stands on it, with that settlement's name; an empty
  cell has no name and is told relative to the nearest named place, its land or the sea.
- The occupancy log replays who stood where at every point of the event log.
- A region is named after its most populous settlement at the time of the report.
- No name is made before it is needed.
"""

import pickle

import numpy as np
import pytest

from conftest import small_model
from deluge.data.events import ACTION_CODE, decode, entity
from deluge.gazetteer import DIRECTIONS, gazetteer
from deluge.naming import PLACE_REGION, Speaker


@pytest.fixture(scope="module")
def ran():
    m = small_model(years=200)
    m.run()
    return m


def test_no_names_are_made_before_they_are_needed():
    m = small_model(years=20)
    assert not m.__dict__.get("_names")
    m.run()
    assert not m.__dict__.get("_names")  # running makes none
    sp = Speaker(m)
    sid = int(m.settlements.alive_ids()[0])
    sp.place(int(m.settlements.cell[sid]))
    assert set(m._names) == {(2, sid)}  # only the one asked for: the settlement standing there


def test_a_settled_cell_has_its_settlements_name_and_an_empty_one_none(ran):
    m = ran
    sp, gz = Speaker(m), gazetteer(m)
    t = m.settlements
    for sid in t.alive_ids()[:20].tolist():
        cell = int(t.cell[sid])
        name = sp.entity(entity("settlement", sid))
        assert gz.name_of(cell, sp) == name and sp.place(cell) == f"at {name}"
    empty = np.flatnonzero((m.occupant < 0) & ~m.is_sea)
    for cell in empty[:20].tolist():
        assert gz.name_of(int(cell), sp) is None
        text = sp.place(int(cell))
        assert not text.startswith("at "), text
        assert any(d in text for d in DIRECTIONS) or text in (sp.data["places"]["empty_land"],) \
            or text.startswith("in the land of ")


def test_relative_places_point_the_right_way(ran):
    m = ran
    sp, gz = Speaker(m), gazetteer(m)
    sid = int(m.settlements.alive_ids()[0])
    src = int(m.settlements.cell[sid])
    x, y = m.geom.xy(src)
    occ = m.occupant
    for dx, dy, word in ((1, 0, "east"), (-1, 0, "west"), (0, 2, "north"), (0, -2, "south")):
        c = m.geom.cid(x + dx, y + dy) if 0 <= x + dx < m.geom.width and 0 <= y + dy < m.geom.height else -1
        if c < 0 or occ[c] >= 0:
            continue
        near, _ = gz.nearest(c, None)
        if near == sid:
            assert gz.direction(src, c) == word
            assert gz.relative(c, None, sp).startswith(word)


def test_occupancy_log_replays_the_map(ran):
    m = ran
    gz = gazetteer(m)
    cells = np.arange(m.geom.n_cells)
    assert (gz.occupants(cells, at=len(m.events)) == m.occupant).all()
    # Year 0: the founding bands stood where the founding events place them.
    ev = m.events
    for i in range(len(ev)):
        if ev.year[i] > 0:
            break
        if ACTION_CODE["founded"] == ev.action[i]:
            sid = decode(ev.actors[i][1])[1]
            assert gz.occupant(ev.place[i], at=i) == sid
    # Every move is seen at its event: the band stands on its new cell, which was empty before.
    for i in range(len(ev)):
        if ev.action[i] == ACTION_CODE["moved"]:
            sid = decode(ev.actors[i][0])[1]
            assert gz.occupant(ev.place[i], at=i) == sid
            assert gz.occupant(ev.place[i], at=i, before=True) != sid


def test_arrivals_are_placed_before_the_arrival(ran):
    """A founding or a move is never placed "at" the settlement that arrives with it."""
    m = ran
    sp = Speaker(m)
    ev = m.events
    seen = 0
    for i in range(len(ev)):
        if ev.action[i] not in (ACTION_CODE["founded"], ACTION_CODE["moved"]):
            continue
        kind, sid = decode(ev.actors[i][0])
        if kind != "settlement":
            continue
        assert sp.event_place(i) != f"at {sp.entity(entity('settlement', sid))}"
        seen += 1
    assert seen > 0


def test_regions_take_their_most_populous_settlement(ran):
    m = ran
    sp = Speaker(m)
    t = m.settlements
    ids = t.alive_ids()
    sid = int(ids[np.argmax(t.population[ids])])
    region = m.memory.region(m, int(t.cell[sid]))
    assert sp.place(region, PLACE_REGION) == f"in the country of {sp.entity(entity('settlement', sid))}"


def test_occupancy_survives_checkpoints_and_stays_out_of_the_digest(ran):
    from deluge.data.state import state_digest

    m = ran
    back = pickle.loads(pickle.dumps(m))
    assert list(back._occupancy.stamp) == list(m._occupancy.stamp)
    assert "_gazetteer" not in back.__dict__
    saved = m._occupancy
    try:
        before = state_digest(m)
        m._occupancy = None
        assert state_digest(m) == before
    finally:
        m._occupancy = saved
