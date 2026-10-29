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


def log_people(m, action, actors, place, magnitude, year=None):
    """Log an event as ``model.log_event`` does: each settlement actor's people noted with it."""
    pop = [float(m.settlements.population[i]) if kind == "settlement" else np.nan for kind, i in actors]
    assert m.events.log(m.year if year is None else year, action, [entity(kind, i) for kind, i in actors], place,
                        magnitude, people=pop)
    return len(m.events) - 1


def _two_records(m, pid, sid, years=None):
    """Two memories of ``pid`` with a shared actor, written straight into its store (by
    default in touching years)."""
    st = m.memory
    out = []
    for k, year in enumerate(years or (m.year, m.year + 1)):
        out.append(log_people(m, "starved", [("settlement", sid)], int(m.settlements.cell[sid]), 5000.0 + k, year))
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


def test_merge_needs_meeting_year_spans_and_respects_the_provenance_cap():
    over = {"memory": {"merge_chance": 1.0, "detail_loss_chance": 0.0, "substitution_chance": 0.0,
                       "repeat_gain": 0.0}}
    m = held(years=10, more=over)
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    a, b = _two_records(m, pid, sid, years=(m.year, m.year + 3))  # [y, y+1) and [y+3, y+4): apart
    memory._merge(m.memory, pid, a, m.year)
    assert a in m.memory.records and b in m.memory.records
    m.memory.records[b].year_span = 25  # now [y, y+25) holds both
    m.memory.records[b].year = m.year
    cap = m.config.memory.max_provenance
    m.memory.records[a].provenance = tuple(range(100, 100 + cap))  # the union would be cap + 1
    memory._merge(m.memory, pid, a, m.year)
    assert a in m.memory.records and b in m.memory.records
    m.memory.records[a].provenance = tuple(range(100, 99 + cap))
    memory._merge(m.memory, pid, a, m.year)
    left = [x for x in (a, b) if x in m.memory.records]
    assert len(left) == 1 and len(m.memory.records[left[0]].provenance) == cap


def test_vividness_comes_from_the_share_of_people_affected_and_novelty():
    """One rule for every verb: the share of the holder's people affected, log-scaled, plus a
    bonus the first time a verb, or an actor from outside the holder's polity, enters the store."""
    m = held(years=10, more={"memory": {"repeat_gain": 0.0}})  # no amplification between the records
    cfg = m.config.memory
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin, pop=3000.0)
    other = place(m, free_land(m, origin, 1)[0], pid, pop=1000.0)
    stranger, ssid = lone_polity(m, free_land(m, origin, 4)[0], lineage=1)
    # No place: no shared symbol, so no amplification between them.
    i = log_people(m, "starved", [("settlement", sid)], -1, 5000.0)
    j = log_people(m, "starved", [("settlement", other)], -1, 5000.0)
    k = log_people(m, "moved", [("settlement", other)], -1, 1e6)
    q = log_people(m, "starved", [("polity", pid)], -1, 5000.0)
    o = log_people(m, "starved", [("settlement", sid), ("polity", stranger)], -1, 5000.0)
    o2 = log_people(m, "starved", [("settlement", other), ("polity", stranger)], -1, 5000.0)
    memory._record_events(m)
    v = {m.memory.records[x].provenance[0]: m.memory.records[x].vividness for x in m.memory.stores[pid]}
    b = cfg.novelty_bonus
    assert v[i] == pytest.approx(memory.salience(cfg, 0.75) + b)  # new verb; its own settlement is no novelty
    assert v[j] == pytest.approx(memory.salience(cfg, 0.25))  # nothing new
    assert v[k] == pytest.approx(memory.salience(cfg, 0.25) + b)  # new verb; the magnitude plays no part
    assert v[q] == pytest.approx(cfg.vividness_scale)  # the polity itself: everyone; not an outsider
    assert v[o] == pytest.approx(memory.salience(cfg, 0.75) + b)  # a new actor from outside
    assert v[o2] == pytest.approx(memory.salience(cfg, 0.25))  # that outsider is no longer new
    assert memory.salience(cfg, 0.01) < memory.salience(cfg, 0.1) < memory.salience(cfg, 1.0) == cfg.vividness_scale
    assert not hasattr(cfg, "magnitude_reference")


def test_amplification_diminishes_with_repetition():
    m = held(years=10, more={"memory": {"repeat_gain": 0.5}})
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    st = m.memory
    first = log_people(m, "starved", [("settlement", sid)], origin, 5000.0)
    memory._record_events(m)
    mid = next(x for x in st.stores[pid] if st.records[x].provenance == (first,))
    gains = []
    for k in range(30):
        before = st.records[mid].vividness
        # In other years: identical records of overlapping years would fuse (``merge_identical``).
        log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year + 2 * (k + 1))
        memory._record_events(m)
        gains.append(st.records[mid].vividness - before)
    assert all(g2 < g1 for g1, g2 in zip(gains, gains[1:]))
    v = m.config.memory.vividness_scale  # each repeat: share 1, nothing new
    assert sum(gains) == pytest.approx(0.5 * v * np.log(31))


def test_detail_loss_coarsens_one_field_at_a_time():
    m = held(years=10)
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    mid = _two_records(m, pid, sid)[0]
    rec = m.memory.records[mid]
    for _ in range(12):
        memory._coarsen(m, m.memory, rec, m.year)
    levels = len(m.config.memory.year_span_fractions)
    assert rec.year_level == levels and rec.place == -1 and rec.magnitude_rounded
    assert sum(memory.OPS[r[1]] == "coarsened" for r in m.memory.ops) == levels + 2 + 1


def test_year_coarsens_relative_to_the_records_age():
    """The span is a share of the years since the original event, centred on it: no bins
    anchored at year 0, and an old record's span is wider than a young one's."""
    m = held(years=10, more={"memory": {"detail_loss_chance": 0.0}})
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    st, fr = m.memory, m.config.memory.year_span_fractions
    i = log_people(m, "starved", [("settlement", sid)], -1, 5000.0, year=1234)
    memory._record_events(m)
    rec = next(st.records[x] for x in st.stores[pid] if st.records[x].provenance == (i,))
    spans = []
    for now in (1334, 3234, 9234):  # 100, 2000 and 8000 years later, at the first level each time
        rec.year, rec.year_span, rec.year_level = 1234, 1, 0
        rec.place_level, rec.magnitude_rounded = memory.PLACE_NONE, True  # only the year is open
        memory._coarsen(m, st, rec, now)
        span = rec.year_span
        assert span == max(2, round(fr[0] * (now - 1234)))
        assert rec.year <= 1234 < rec.year + span and abs((rec.year + span / 2) - 1234) <= 1
        spans.append(span)
    assert spans[0] < spans[1] < spans[2]


def test_merge_votes_by_vividness():
    """Fused records take verb, place and actors by a vividness-weighted vote over the records
    fused in, not from whichever record is newest."""
    m = held(years=10, more={"memory": {"repeat_gain": 0.0}})
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    other = place(m, free_land(m, origin, 1)[0], pid)
    st = m.memory
    ids = [log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year),
           log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year + 2),
           log_people(m, "moved", [("settlement", sid), ("settlement", other)], origin, 5000.0, year=m.year + 1)]
    memory._record_events(m)
    from deluge.data.events import ACTION_CODE

    def mids():
        out = {st.records[x].provenance[0]: x for x in st.stores[pid] if len(st.records[x].provenance) == 1}
        return [out[k] for k in ids]

    for order in ((0, 1, 2), (0, 2, 1)):  # either order of fusing
        m2 = held(years=10, more={"memory": {"repeat_gain": 0.0}})
        # Fresh copies of the three records, the newest the most vivid on its own.
        if order != (0, 1, 2):
            m, st = m2, m2.memory
            pid, sid = lone_polity(m, origin)
            other = place(m, free_land(m, origin, 1)[0], pid)
            ids = [log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year),
                   log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year + 2),
                   log_people(m, "moved", [("settlement", sid), ("settlement", other)], origin, 5000.0,
                              year=m.year + 1)]
            memory._record_events(m)
        x = mids()
        for k, v in zip(x, (1.0, 1.0, 1.2)):
            st.records[k].vividness = v
        memory._fuse(st, pid, x[order[0]], x[order[1]], m.year)
        first = next(r for r in st.records.values() if r.store == pid and len(r.provenance) == 2)
        memory._fuse(st, pid, first.mem_id, x[order[2]], m.year)
        rec = next(r for r in st.records.values() if r.store == pid and len(r.provenance) == 3)
        assert rec.action == ACTION_CODE["starved"]  # two records of one verb outvote the more vivid third
        assert rec.actors[0] == entity("settlement", sid)
        assert rec.year == m.year and rec.year_span == 3  # the span covering all three


def test_substitution_draws_an_entity_of_the_store():
    m = held(years=10)
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    other = place(m, free_land(m, origin, 1)[0], pid)
    log_people(m, "founded", [("settlement", other), ("settlement", sid)], int(m.settlements.cell[other]), 5000.0)
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
    rec = m.memory.records[mid]
    gain = m.config.memory.consequences.invocation_gain
    steps = []
    for _ in range(5):
        v0 = rec.vividness
        memory.invoke(m, a, (entity("polity", b),))
        steps.append(rec.vividness - v0)
    assert steps[0] == pytest.approx(gain * rec.base * np.log(2))  # less each time: log((k + 1) / k)
    assert all(s2 < s1 for s1, s2 in zip(steps, steps[1:]))
    assert sum(steps) == pytest.approx(gain * rec.base * np.log(6))


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


def test_consequences_are_scale_free():
    """Cohesion, legitimacy and the summary read each store relative to its own median
    vividness: scaling every record's vividness leaves them unchanged."""
    m = small_model(years=200)
    m.run()
    memory.derive(m)
    pids = m.polities.alive_ids()
    before = (memory.cohesion_term(m, pids), dict(m.memory.legit), memory.summary(m, pids))
    assert before[0].any() and any(before[1].values())
    for r in m.memory.records.values():
        r.vividness *= 100.0
    memory.derive(m)
    after = (memory.cohesion_term(m, pids), dict(m.memory.legit), memory.summary(m, pids))
    np.testing.assert_allclose(after[0], before[0])
    assert after[1] == pytest.approx(before[1])
    np.testing.assert_allclose(after[2], before[2])


def test_myths_report_renders(ran):

    text = myths.report(ran)
    assert text.startswith("# ") and "## Group" in text and "original event" in text


def test_identical_records_always_merge():
    """Records with the same verb, actors and place (same level) whose years overlap fuse into
    one, in the store that took them and in every store each generation; others stay apart."""
    m = held(years=10, more={"memory": {"merge_chance": 0.0, "repeat_gain": 0.0}})
    origin = interior_cell(m, 3)
    pid, sid = lone_polity(m, origin)
    other = place(m, free_land(m, origin, 1)[0], pid)
    st = m.memory
    a = log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year)
    b = log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year)  # identical: fuses
    c = log_people(m, "starved", [("settlement", sid)], origin, 5000.0, year=m.year + 5)  # another year
    d = log_people(m, "starved", [("settlement", other)], origin, 5000.0, year=m.year)  # other actors
    e = log_people(m, "moved", [("settlement", sid)], origin, 5000.0, year=m.year)  # another verb
    memory._record_events(m)
    by = {st.records[x].provenance: st.records[x] for x in st.stores[pid]}
    assert (a, b) in by and (c,) in by and (d,) in by and (e,) in by and len(by) == 4
    assert any(memory.OPS[r[1]] == "merged" for r in st.ops)
    # Once the year coarsens so the spans overlap, the record of another year fuses too.
    rec_c = by[(c,)]
    rec_c.year, rec_c.year_span = m.year - 2, 10
    assert memory.merge_identical(st, pid, m.year) == 1
    assert {st.records[x].provenance for x in st.stores[pid]} == {(a, b, c), (d,), (e,)}
    # The same place at another level is another record.
    f = log_people(m, "starved", [("settlement", other)], origin, 5000.0, year=m.year)
    memory._record_events(m)
    rec_f = next(st.records[x] for x in st.stores[pid] if f in st.records[x].provenance)
    assert len(rec_f.provenance) == 2  # identical to d: fused on arrival
    rec_f.place, rec_f.place_level = st.region(m, origin), memory.PLACE_REGION
    g = log_people(m, "starved", [("settlement", other)], origin, 5000.0, year=m.year)
    memory._record_events(m)
    assert any(st.records[x].provenance == (g,) for x in st.stores[pid])


def test_no_store_holds_identical_records_after_a_run():
    m = small_model(years=300)
    m.run()
    st = m.memory
    for pid, store in st.stores.items():
        seen: dict = {}
        for x in store:
            r = st.records[x]
            seen.setdefault((r.action, frozenset(r.actors), r.place, r.place_level), []).append(r)
        for recs in seen.values():
            recs.sort(key=lambda r: r.year)
            for r1, r2 in zip(recs, recs[1:]):
                assert r2.year >= r1.year + r1.year_span, (pid, r1.mem_id, r2.mem_id)
