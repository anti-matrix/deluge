"""Round 13 step 2: roles of actors, role-built sentences, the loose form and runs.

- Every action code has roles, sentence shapes and a loose form; every role has a shape.
- No chronicle or narrator sentence lists actors with "with"; every logged event is told as a
  sentence of its roles (never in the loose form).
- Memory records keep their actors' slots: a substituted actor takes the replaced one's slot,
  a fused record keeps the winners' slots; a record whose roles no longer fit is told loosely.
- Runs of one event are told once with a count and a span.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import small_model
from deluge.data import chronicle
from deluge.data.events import ACTION_CODE, ACTIONS, ENTITY_KINDS, entity
from deluge.naming import Speaker, data_for
from deluge.roles import FIELD, table_for
from deluge.systems import memory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from analysis import narrator  # noqa: E402


@pytest.fixture(scope="module")
def ran():
    m = small_model(years=400)
    m.run()
    return m


def test_every_action_has_roles_shapes_and_a_loose_form(ran):
    data = data_for(ran)
    rt = table_for(data)
    verbs = {k.split("/")[0] for k in data["roles"]}
    assert verbs == set(ACTIONS) == set(data["sentences"]) == set(data["loose"]["deeds"])
    for key, spec in data["roles"].items():
        verb = key.split("/")[0]
        named = set().union(*(f for _, f in rt.shapes[verb]))
        for entry in spec:
            for role in (entry.values() if isinstance(entry, dict) else [entry]):
                assert role.rstrip("+") in named, (key, role)
    for verb, shapes in data["sentences"].items():
        for s in shapes:
            assert "with" not in s.split(), s
            assert set(FIELD.findall(s)) - {"at", "to", "be", "count"} <= {r.rstrip("+") for spec in data["roles"].values()
                                                                    for e in spec
                                                                    for r in (e.values() if isinstance(e, dict) else [e])}


def test_events_are_told_by_their_roles_and_never_with_a_list(ran):
    m = ran
    sp = Speaker(m, tags=True)
    loose = data_for(m)["loose"]["people"].split("{")[0].strip().lower()
    for i in range(len(m.events)):
        text = sp.event(i)
        assert " with " not in text, text
        assert not text.lower().startswith(loose + " "), text  # an event is never loose
    view = chronicle.Chronicle(m)
    view.sync()
    for k in range(len(view)):
        assert " with " not in view.text(k)
    data = narrator.narrate(m)
    for p in data["peoples"]:
        for r in p["records"]:
            assert " with " not in r["remembered"]["text"]
            assert all(" with " not in h["text"] for h in r["happened"])


def test_examples_read_as_roles(ran):
    m = ran
    sp = Speaker(m)
    P = m.polities
    a, b = (int(x) for x in P.alive_ids()[:2])
    town = int(P.capital[b])
    fought = sp.sentence(ACTION_CODE["fought"], [entity("polity", a), entity("polity", b), entity("settlement", town)])
    assert fought == (f"{sp.entity(entity('polity', a))[:1].upper()}{sp.entity(entity('polity', a))[1:]} attacked "
                      f"{sp.entity(entity('settlement', town))} of {sp.entity(entity('polity', b))}.")
    built = sp.sentence(ACTION_CODE["built"], [entity("polity", a), entity("monument", 0), entity("settlement", town)])
    assert built.endswith(f"raised a monument at {sp.entity(entity('settlement', town))}.")
    t = m.settlements
    s1, s2 = (int(x) for x in t.alive_ids()[:2])
    learned = sp.sentence(ACTION_CODE["learned"], [entity("settlement", s1), entity("technique", 1),
                                                   entity("settlement", s2)])
    assert learned == (f"{sp.entity(entity('settlement', s1))} learned {sp.technique(1)} from "
                       f"{sp.entity(entity('settlement', s2))}.")


def _record(m, action, actors, roles):
    return memory.Memory(10**9, tuple(actors), action, -1, 5, 1.0, 1.0, (), (0,), (None,) * len(actors),
                         year_span=1, place_level=memory.PLACE_NONE, roles=tuple(roles))


def test_records_keep_roles_through_substitution_and_fusion(ran):
    m = ran
    rt = table_for(data_for(m))
    st = m.memory
    pid = next(p for p in m.polities.alive_ids().tolist() if len(st.stores.get(p, ())) >= 2)
    # Substitution: the new actor holds the replaced one's slot.
    mid = st.stores[pid][0]
    rec = st.records[mid]
    before = list(zip(rec.actors, rec.roles))
    saved = (rec.actors, rec.signatures, rec.roles, rec.symbols, dict(rec.actor_votes), st.rng.bit_generator.state)
    try:
        memory._substitute(st, pid, mid, m.year)
        after = list(zip(rec.actors, rec.roles))
        assert [r for _, r in after] == [r for _, r in before]
        changed = [k for k in range(len(after)) if after[k][0] != before[k][0]]
        assert len(changed) <= 1
    finally:
        rec.actors, rec.signatures, rec.roles, rec.symbols, rec.actor_votes, state = saved
        st.rng.bit_generator.state = state
    # Fusion: the winners keep their slots, the winning verb's slot first.
    P, t = m.polities, m.settlements
    a, b = (int(x) for x in P.alive_ids()[:2])
    town = int(P.capital[b])
    battle = [entity("polity", a), entity("polity", b), entity("settlement", town)]
    r1 = _record(m, ACTION_CODE["fought"], battle, rt.slots(ACTION_CODE["fought"], battle))
    r2 = _record(m, ACTION_CODE["starved"], [battle[2]], rt.slots(ACTION_CODE["starved"], [battle[2]]))
    r2.mem_id, r2.vividness = 10**9 + 1, 0.5
    st.records[r1.mem_id], st.records[r2.mem_id] = r1, r2
    st.stores[pid] += [r1.mem_id, r2.mem_id]
    n_ops = len(st.ops)
    try:
        memory._fuse(st, pid, r1.mem_id, r2.mem_id, m.year)
        assert r1.action == ACTION_CODE["fought"]
        assert dict(zip(r1.actors, r1.roles))[battle[2]] == (ACTION_CODE["fought"], ENTITY_KINDS.index("polity"), -1, 2)
        sp = Speaker(m)
        text = sp.tell(r1.action, r1.actors, r1.roles)
        assert "attacked" in text
    finally:
        st.stores[pid] = [x for x in st.stores[pid] if x not in (r1.mem_id, r2.mem_id)]
        st.records.pop(r1.mem_id, None)
        st.records.pop(r2.mem_id, None)
        del st.ops[n_ops:]


def test_broken_roles_are_told_in_the_loose_form(ran):
    m = ran
    sp = Speaker(m)
    rt = table_for(data_for(m))
    P = m.polities
    a, b = (int(x) for x in P.alive_ids()[:2])
    # A battle remembered with a technique in it, and an actor whose slot is another verb's.
    actors = [entity("polity", a), entity("polity", b), entity("technique", 2)]
    slots = rt.slots(ACTION_CODE["fought"], actors[:2]) + [(ACTION_CODE["learned"], 2, -1, 1)]
    text = sp.tell(ACTION_CODE["fought"], actors, slots, "at Somewhere", "in the first age")
    assert text.startswith("Of ") and "a battle" in text and sp.technique(2) in text
    assert text.endswith(", in the first age.")
    assert " with " not in text
    # A thing in a people's role is loose too.
    swapped = [entity("monument", 0), entity("polity", b), entity("settlement", int(P.capital[b]))]
    assert sp.tell(ACTION_CODE["fought"], swapped, rt.slots(ACTION_CODE["fought"], swapped)).startswith("Of ")


def test_runs_are_told_once_with_count_and_span(ran):
    m = ran
    P = m.polities
    pid = int(P.alive_ids()[0])
    sid = int(P.capital[pid])
    cell = int(m.settlements.cell[sid])
    log = m.events
    start, year = len(log), m.year
    m.year += 60
    try:
        for k in range(12):
            log.log(year + 2 * k, "built", [entity("polity", pid), entity("monument", 10**6 + k),
                                            entity("settlement", sid)], cell, 100.0)
        m.year += 50
        view = chronicle.Chronicle(m)
        view.sync()
        mine = [e for e in view.entries if e["event"] >= start]
        assert len(mine) == 1 and len(mine[0]["events"]) == 12
        text = view.text(mine[0])
        sp = Speaker(m, tags=True)
        assert f"from year {year} to {year + 22}" in text.lower()
        assert f"raised 12 monuments at {sp.entity(entity('settlement', sid))}" in text
    finally:
        m.year = year
        for col in (log.year, log.action, log.actors, log.place, log.magnitude, log.link, log.people,
                    log.polity_people, log.origin):
            del col[start:]
