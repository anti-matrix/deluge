"""Round 8: memory records, transmission, symbols, consequences and the control (spec section 11)."""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import small_model
from deluge.data.events import ACTIONS, decode, entity
from deluge.systems import cohesion, memory, politics
from test_polities import free_land, held, interior_cell, lone_polity, place

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from analysis import myths  # noqa: E402

MEMORY_PY = REPO / "deluge" / "systems" / "memory.py"
VOCABULARY = re.compile(
    r"\b(myth\w*|motif\w*|stor(y|ies)|legend\w*|hero\w*|heroine\w*|gods?|goddess\w*|deit(y|ies)|divin\w*|sacred|"
    r"holy|flood\w*|founders?|saint\w*|prophet\w*|tales?|epic\w*|worship\w*|religio\w*|savio(u)?r\w*|monster\w*|"
    r"spirit\w*|miracl\w*|creation|teacher\w*|ancestor\w*|totem\w*|ritual\w*|cult|omen\w*|curse\w*|blessing\w*)\b",
    re.IGNORECASE)


def test_no_myth_or_motif_vocabulary_in_memory_py():
    hits = sorted({m.group(0).lower() for m in VOCABULARY.finditer(MEMORY_PY.read_text(encoding="utf-8"))})
    assert hits == []


def test_memory_code_names_no_verb():
    """Every verb is handled alike: memory.py never names one of the event verbs."""
    text = MEMORY_PY.read_text(encoding="utf-8")
    assert not [a for a in ACTIONS if re.search(rf"[\"']{a}[\"']", text)]


@pytest.fixture(scope="module")
def ran():
    m = small_model(years=400)
    m.run()
    return m


def test_records_come_from_logged_events(ran):
    m, st = ran, ran.memory
    assert st.created > 0 and st.records
    log = m.events
    for rec in st.records.values():
        assert rec.provenance and all(0 <= i < len(log) for i in rec.provenance)
        assert len(rec.signatures) == len(rec.actors)
        if rec.parent < 0 and len(rec.provenance) == 1:
            i = rec.provenance[0]
            assert rec.action == log.action[i]
            assert rec.year <= log.year[i] < rec.year + rec.year_span
    # Every record copied from another has a logged copy or borrowing.
    copied = {row[3] for row in st.ops if memory.OPS[row[1]] in ("copied", "borrowed")}
    assert all(r.mem_id in copied for r in st.records.values() if r.parent >= 0)


def test_stores_stay_within_capacity(ran):
    m, st = ran, ran.memory
    pids = m.polities.alive_ids()
    caps = dict(zip(pids.tolist(), memory.capacity(m, pids).tolist()))
    for pid, store in st.stores.items():
        assert pid in caps and len(store) <= caps[pid]
        assert all(st.records[mid].store == pid for mid in store)


def test_capacity_is_enforced_on_insertion():
    m = small_model({"memory": {"capacity": 2, "record_min_vividness": 0.0}}, years=60)
    m.run()
    assert all(len(s) <= 2 for s in m.memory.stores.values())
    assert m.memory.dropped > 0


def test_writing_raises_capacity():
    m = small_model(years=2)
    pid = int(m.polities.alive_ids()[0])
    base = int(memory.capacity(m, [pid])[0])
    cap = int(m.polities.capital[pid])
    m.settlements.knows[cap, m.knowledge.index("writing")] = True
    assert int(memory.capacity(m, [pid])[0]) == 2 * base


def test_signatures_are_recorded_for_people():
    m = small_model(years=3)
    m.run()
    recs = list(m.memory.records.values())
    assert recs
    for r in recs:
        for a, sig in zip(r.actors, r.signatures):
            kind = decode(a)[0]
            if kind in ("settlement", "polity"):
                n = m.settlements.appearance_mean.shape[1] + m.settlements.dialect.shape[1]
                assert sig is not None and len(sig) == n


def _two_records(m, pid, sid):
    """Two memories of ``pid`` with a shared actor, written straight into its store."""
    st = m.memory
    out = []
    for k, year in enumerate((m.year, m.year + 3)):
        m.events.log(year, "starved", [entity("settlement", sid)], int(m.settlements.cell[sid]), 5000.0 + k)
        out.append(len(m.events) - 1)
    memory._record_events(m)
    return [mid for mid in st.stores[pid] if st.records[mid].provenance[0] in out]


def test_merge_unions_provenance_and_is_logged():
    m = held(years=10, more={"memory": {"merge_chance": 1.0, "detail_loss_chance": 0.0,
                                        "substitution_chance": 0.0, "repeat_gain": 0.0}})
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    a, b = _two_records(m, pid, sid)
    prov = set(m.memory.records[a].provenance) | set(m.memory.records[b].provenance)
    memory._merge(m.memory, pid, a, m.year)
    left = [mid for mid in (a, b) if mid in m.memory.records]
    assert len(left) == 1
    assert set(m.memory.records[left[0]].provenance) == prov
    assert any(memory.OPS[r[1]] == "merged" for r in m.memory.ops)


def test_detail_loss_coarsens_one_field_at_a_time():
    m = held(years=10)
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    mid = _two_records(m, pid, sid)[0]
    rec = m.memory.records[mid]
    for _ in range(12):
        memory._coarsen(m, m.memory, rec, m.year)
    spans = m.config.memory.year_spans
    assert rec.year_span == spans[-1] and rec.place == -1 and rec.magnitude_rounded
    assert sum(memory.OPS[r[1]] == "coarsened" for r in m.memory.ops) == len(spans) - 1 + 2 + 1


def test_substitution_draws_an_entity_of_the_store():
    m = held(years=10)
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    other = place(m, free_land(m, origin, 1)[0], pid)
    m.events.log(m.year, "founded", [entity("settlement", other), entity("settlement", sid)],
                 int(m.settlements.cell[other]), 5000.0)
    memory._record_events(m)
    mid = _two_records(m, pid, sid)[0]
    memory._substitute(m.memory, pid, mid, m.year)
    rec = m.memory.records[mid]
    assert rec.actors == (entity("settlement", other),)
    row = [r for r in m.memory.ops if memory.OPS[r[1]] == "substituted"][-1]
    assert row[5] == entity("settlement", sid) and row[6] == entity("settlement", other)
    assert myths.substituted_actors(m, rec) == [entity("settlement", other)]


def test_splits_copy_the_parent_store():
    m = held({"secession_base": 1e3, "secession_max": 1.0})
    origin = interior_cell(m, 9)
    pid, cap = lone_polity(m, origin, cohesion_value=0.0)
    sid = place(m, free_land(m, origin, 6)[0], pid)
    _two_records(m, pid, sid)
    st = m.memory
    before = sorted(st.records[x].provenance for x in st.stores[pid])
    politics.secession(m)
    child = int(m.settlements.polity[sid])
    assert child != pid
    assert sorted(st.records[x].provenance for x in st.stores[child]) == before
    assert set(st.stores[child]).isdisjoint(st.stores[pid])
    for x in st.stores[child]:
        assert st.records[x].parent in st.stores[pid] or st.records[x].parent >= 0
    held_by_child = [x for x in st.stores[child] if sid in st.records[x].holders]
    assert held_by_child and not any(sid in st.records[x].holders for x in st.stores[pid])


def test_union_adds_the_smaller_store():
    m = held()
    origin = interior_cell(m, 4)
    a, sa = lone_polity(m, origin, pop=5000.0)
    b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=1000.0)
    _two_records(m, b, sb)
    small = list(m.memory.stores[b])
    cohesion.run(m)
    big = politics.merge(m, a, b)
    assert big == a and set(small) <= set(m.memory.stores[a]) and b not in m.memory.stores


def test_mergers_happen_with_shared_symbols_and_a_common_enemy():
    m = held({"merge_rate": 1.0})
    origin = interior_cell(m, 6)
    a, sa = lone_polity(m, origin)
    b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1)
    c, sc = lone_polity(m, free_land(m, origin, 4)[0], lineage=2)
    # One remembered event, held by both.
    m.events.log(m.year, "fought", [entity("polity", a), entity("polity", b)], origin, 5000.0)
    memory._record_events(m)
    memory.derive(m)
    assert memory.affinity(m, np.array([a]), np.array([b]))[0] > 0
    m.wars = {(min(a, c), max(a, c)): m.year, (min(b, c), max(b, c)): m.year}
    assert politics.common_enemy(m, np.array([a]), np.array([b]))[0] == 1.0
    politics.mergers(m)
    assert m.polities.ended[a] >= 0 or m.polities.ended[b] >= 0


def test_brain_inputs_carry_the_summary_only_with_consequences_on():
    from deluge.brains import features

    for control, nonzero in ((False, True), (True, False)):
        m = small_model({"memory": {"control": control}}, years=60)
        m.run()
        pids = m.polities.alive_ids()
        x = features.assemble(m, pids)
        cols = [i for i, n in enumerate(m.brain_layout.inputs) if n.startswith("memory_")]
        assert len(cols) == len(memory.SUMMARY)
        assert bool(np.abs(x[:, cols]).sum() > 0) == nonzero


def test_invocation_amplifies_the_related_records():
    m = held()
    origin = interior_cell(m, 4)
    a, sa = lone_polity(m, origin)
    b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1)
    m.events.log(m.year, "fought", [entity("polity", a), entity("polity", b)], origin, 2000.0)
    memory._record_events(m)
    memory.derive(m)
    mid = m.memory.stores[a][0]
    v0 = m.memory.records[mid].vividness
    memory.invoke(m, a, (entity("polity", b),))
    assert m.memory.records[mid].vividness == pytest.approx(v0 * (1 + m.config.memory.consequences.invocation_gain))


def test_control_matches_a_run_without_memory():
    """Consequence weights at zero: memory is recorded and transmitted, and the run's
    outcomes are identical to a run with memory off."""
    a = small_model({"memory": {"control": True}}, years=300)
    b = small_model({"memory": {"enabled": False}}, years=300)
    a.run()
    b.run()
    assert a.memory.created > 0 and len(a.memory.ops) > 0 and not b.memory.records
    da, db = a.datacollector.get_model_vars_dataframe(), b.datacollector.get_model_vars_dataframe()
    assert da.equals(db)
    assert a.events.to_frame().equals(b.events.to_frame())
    n = a.polities.n
    assert n == b.polities.n
    np.testing.assert_array_equal(a.polities.cohesion[:n], b.polities.cohesion[:n])
    np.testing.assert_array_equal(a.brains.genome[:n], b.brains.genome[:n])


def test_consequences_change_the_run():
    a = small_model(years=400)
    b = small_model({"memory": {"control": True}}, years=400)
    a.run()
    b.run()
    assert not a.datacollector.get_model_vars_dataframe().equals(b.datacollector.get_model_vars_dataframe())


def test_myths_report_renders(ran):

    text = myths.report(ran)
    assert text.startswith("# ") and "## Group" in text and "original event" in text
