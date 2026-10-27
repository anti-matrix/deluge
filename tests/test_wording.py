"""Round 13b: the narrator's and the chronicle's wording.

- A conquest's took / lost / died events of one year, victor and settlement are one chronicle
  entry and one sentence; none of them is told or counted again.
- Singular and plural agree with the number shown ("1 sea cell", "2 sea cells").
- A sea crossing says where it went: "from X to an unpeopled land", "to the land south-east
  of X", never "from X in ...".
"""

import re

import numpy as np
import pytest

from conftest import small_model
from deluge.data import chronicle
from deluge.data.events import ACTIONS, decode, entity
from deluge.gazetteer import Place, destination
from deluge.naming import Speaker, number_form, plural


@pytest.fixture(scope="module")
def warring():
    m = small_model({"polities": {"secession_base": 0.5}}, years=300)
    m.run()
    return m


@pytest.fixture(scope="module")
def conquered():
    """A small polity whose seat is taken: its settlement taken, its seat lost, the people ended."""
    from deluge.systems import war
    from test_war import held, pair

    m = held()
    a, sa, b, sb = pair(m, attacker_pop=50000.0)
    assert war.battle(m, war.forces(m), a, b, sa, sb, campaign=True)
    return m


def test_a_conquest_is_one_sentence(conquered):
    m = conquered
    ev = m.events
    view = chronicle.Chronicle(m)
    view.sync()
    assert view.joined, "no conquest in the run"
    folded = set()
    for main, events in view.joined.items():
        acts = [ACTIONS[ev.action[i]] for i in events]
        assert main == events[0] and len(events) == len(set(acts)) >= 2 and set(acts) <= {"took", "lost", "died"}
        assert len({int(ev.year[i]) for i in events}) == 1 and len({int(ev.place[i]) for i in events}) == 1
        victors = set()
        for i, a in zip(events, acts):
            actors = ev.actors[i]
            victors.add(actors[0] if a == "took" else actors[2] if a == "lost" else actors[1])
        assert len(victors) == 1
        folded.update(events[1:])
        text = view.text({"event": main, "events": events, "suffix": [], "year": int(ev.year[main])})
        sp = Speaker(m, tags=True)
        if "died" in acts:
            ended = ev.actors[events[acts.index("died")]][0]
            assert f" took {sp.entity(ev.actors[events[0]][1])} and destroyed {sp.entity(ended)} (" in text
            assert text.endswith(" left).") and " lost " not in text
    told = {e["event"] for e in view.entries}
    assert not folded & told
    # Every capital lost with its people's end in the same year and place is folded into the taking.
    for e in view.entries:
        i = e["event"]
        if i >= 0 and ACTIONS[ev.action[i]] == "lost" and len(ev.actors[i]) == 3:
            same = [k for k in range(len(ev.year)) if ev.year[k] == ev.year[i] and ev.place[k] == ev.place[i]
                    and ACTIONS[ev.action[k]] == "took" and ev.actors[k][0] == ev.actors[i][2]]
            assert not same, view.text(e)


def test_singular_and_plural_agree():
    assert plural(1, "sea cell") == "1 sea cell" and plural(2, "sea cell") == "2 sea cells"
    assert plural(1, "band") == "1 band" and plural(1500, "person", "people") == "1,500 people"
    assert plural(0.6, "settlement") == "1 settlement"  # as it is shown
    pair = ["{m} sea cell", "{m} sea cells"]
    assert number_form(pair, 1, m="1") == "1 sea cell" and number_form(pair, 2, m="2") == "2 sea cells"
    assert number_form("{m} dead", 1, m="1") == "1 dead"


def test_chronicle_and_records_have_no_number_disagreement(warring):
    from analysis.narrator import record_fields, remembered

    m = warring
    view = chronicle.Chronicle(m)
    view.sync()
    sp = Speaker(m)
    lines = view.markdown().splitlines() + [remembered(sp, record_fields(r))["text"]
                                             for r in m.memory.records.values()]
    bad = re.compile(r"(?<![\d,.])1 (?!person|dead)[a-z-]+s\b|\b(one|a) [a-z]+s (were|have)\b|\b(?:[2-9]|\d{2,}) "
                     r"(sea cell|band|settlement|person|year|monument)\b")
    hits = [s for s in lines if bad.search(s) or "a monument (" in s]
    assert lines and hits == []


def test_a_crossing_says_where_it_went(warring):
    m = warring
    d = Speaker(m).data
    p = d["places"]
    assert destination(Place.say(p, "empty_land"), d) == "to an unpeopled land"
    assert destination(Place.say(p, "near", direction="south-east", name="Mebu"), d) == "to the land south-east of Mebu"
    assert destination(Place.say(p, "far", direction="east", name="Mebu"), d) == "to the land far to the east of Mebu"
    assert destination(Place.say(p, "cell", name="Mebu"), d) == "to Mebu"
    assert destination(Place.say(p, "land", name="Mebu"), d) == "to the land of Mebu"
    # A settlement's crossing, told with each kind of place it can reach.
    sp = Speaker(m)
    t = m.settlements
    s1, s2 = (int(x) for x in t.alive_ids()[:2])
    actors = (entity("settlement", s1), entity("settlement", s2))
    code = ACTIONS.index("crossed")
    for form, kw in (("empty_land", {}), ("near", {"direction": "south-east", "name": sp.entity(actors[1])})):
        text = sp.sentence(code, actors, Place.say(p, form, **kw), "in year 5")
        assert " crossed the sea from " in text and (" to an unpeopled land" in text or " to the land south-east of "
                                                      in text), text
        assert not re.search(r"from \S+ (in|on|at|east|west|north|south|far) ", text), text
    # Every crossing of the run's log and memory: never "from X in/at/on ...".
    view = chronicle.Chronicle(m)
    view.sync()
    for i in np.flatnonzero(np.asarray(m.events.action) == code).tolist():
        s = sp.event(i)
        if " crossed the sea from " in s:
            assert not re.search(r"crossed the sea from [^,(]*? (in|on|at) ", s), s
    assert decode(actors[0])[0] == "settlement"
