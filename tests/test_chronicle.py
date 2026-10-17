"""Round 10 step 3: the chronicle in the narrator's language, and one name per entity.

- The chronicle's rules read only action codes, actor kinds and ids, and numbers: each rule
  fires at its threshold, and renaming everything changes the words but not which events are
  told or how the rest are counted.
- An entity has the same name in the UI (inspector, lineage tree, map hover), the chronicle
  and the narrator's reports.
- The chronicle panel renders in under 200 ms with a log the size of year 4,000's.
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import small_model
from deluge import naming
from deluge.data import chronicle
from deluge.data.events import ENTITY_KINDS, entity

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
YEAR_4000_EVENTS = 240_000  # world seed 1, history seed 1 at year 4,000 (Round 10 measurement)


@pytest.fixture(scope="module")
def ran():
    m = small_model(years=300)
    m.run()
    return m


def told_and_counted(m):
    view = chronicle.Chronicle(m)
    view.sync()
    told = sorted(e["event"] for e in view.entries if e["event"] >= 0)
    counted = sorted((e["year"], e["lineage"], tuple(e["items"])) for e in view.entries if e["event"] < 0)
    return view, told, counted


def test_each_rule_fires_at_its_threshold(ran):
    m = ran
    cfg = m.config.chronicle
    P, t = m.polities, m.settlements
    pid = int(P.alive_ids()[0])
    sid = int(t.alive_ids()[0])
    people = float(P.population[t.polity[sid]])
    other = int([p for p in P.alive_ids() if p != t.polity[sid]][0])
    smallest = min(people, float(P.population[other]))
    start = len(m.events)
    m.year += 50  # a generation later: these events' generation is over at the next sync
    cases = [
        (("founded", [("polity", pid), ("settlement", sid)], 1.0), True),
        (("founded", [("settlement", sid), ("settlement", sid)], 1.0), False),
        (("fought", [("polity", other), ("polity", int(t.polity[sid])), ("settlement", sid)],
          1.01 * cfg.death_share * smallest), True),
        (("fought", [("polity", other), ("polity", int(t.polity[sid])), ("settlement", sid)],
          0.99 * cfg.death_share * smallest), False),
        (("starved", [("settlement", sid)], max(1.01 * cfg.death_share * people, 1000.0)), True),
        (("taught", [("teachers", 0), ("settlement", sid)], 0.0), False),
        (("taught", [("teachers", 0), ("settlement", sid), ("technique", 0)], 1.0), True),
        (("lost", [("polity", pid), ("settlement", sid), ("polity", other)], 10.0), True),
        (("moved", [("settlement", sid)], 10.0), False),
    ]
    expect = {}
    for (action, actors, mag), want in cases:
        i = m.log_event(action, actors, int(t.cell[sid]), mag)
        assert i >= 0, action
        expect[i] = want
    m.year += 50
    try:
        view = chronicle.Chronicle(m)
        view.sync()
        told = {e["event"] for e in view.entries}
        assert {i: i in told for i in expect} == expect
        counted = sum(it[3] for e in view.entries if e["event"] < 0 and e["year"] >= m.year - 100 for it in e["items"])
        assert counted >= sum(1 for w in expect.values() if not w)
    finally:
        m.year -= 100
        log = m.events
        for col in (log.year, log.action, log.actors, log.place, log.magnitude, log.link, log.people,
                    log.polity_people, log.origin):
            del col[start:]


def test_rules_ignore_what_entities_are(ran, monkeypatch):
    """Other names, other phrases, other technique labels: the same events are told and the
    same counts summed (only action codes, kinds, ids and numbers decide)."""
    view, told, counted = told_and_counted(ran)
    words = [view.text(e) for e in view.entries]
    data = naming.data_for(ran)
    other = {**data, "techniques": {k: f"craft number {i}" for i, k in enumerate(data["techniques"])}}
    monkeypatch.setattr(naming, "data_for", lambda model: other)
    monkeypatch.setattr(naming, "proper_name", lambda model, key: f"N{key[0]}x{key[1]}")
    view2, told2, counted2 = told_and_counted(ran)
    assert told2 == told and counted2 == counted
    assert [view2.text(e) for e in view2.entries] != words
    source = (REPO / "deluge" / "data" / "chronicle.py").read_text(encoding="utf-8")
    assert "knowledge.names" not in source and "knowledge.labels" not in source  # nor at what a technique is


def test_same_name_in_ui_chronicle_and_reports(ran):
    from analysis import narrator
    from deluge.config import load_config
    from deluge.viz import panels, portrayal

    m = ran
    viz = load_config().viz
    P, t = m.polities, m.settlements
    pid = int(P.alive_ids()[0])
    g = int(P.lineage[pid])
    cap = int(P.capital[pid])
    plain = naming.Speaker(m)
    polity = plain.entity(entity("polity", pid))
    town = plain.entity(entity("settlement", cap))
    people = plain.people(g)
    assert polity == f"the {naming.proper_name(m, (ENTITY_KINDS.index('polity'), pid))} people"
    # UI: inspector, lineage tree, map hover, legend.
    ui = panels.inspector(m, pid, viz)
    assert polity[:1].upper() + polity[1:] in ui and town in ui and people in ui
    assert polity.lower() in panels.lineage_tree(m, viz).lower()
    snap = portrayal.cell_snapshot(m)
    hover = portrayal.tooltip(m, snap, int(t.cell[cap]))
    assert town in hover and polity in hover
    assert any(n.startswith(people) for n in portrayal.people_names(m))
    # Chronicle (with the group tag) and the narrator's report.
    view = chronicle.Chronicle(m)
    view.sync()
    md = view.markdown()
    assert polity in md and people in md
    report = narrator.narrate(m)
    names = {p["lineage"]: p for p in report["peoples"]}
    if g in names:
        assert names[g]["name"] == people
    lineage_towns = [plain.entity(entity("settlement", int(s))) for s in t.alive_ids() if t.lineage[s] == g]
    if g in names:
        assert names[g]["sample_names"] == lineage_towns[: m.config.reports.sample_names]
    # Names never change as the run goes on (year-0 speech): a fresh model of the same seeds
    # names its peoples and founding polities as the run does at year 300.
    again = small_model(years=300)
    fresh = naming.Speaker(again)
    assert [fresh.people(k) for k in range(5)] == [plain.people(k) for k in range(5)]
    assert fresh.entity(entity("polity", 0)) == plain.entity(entity("polity", 0))


def _grow_log(m, target: int) -> None:
    """Repeat the model's events, later in time, until the log holds ``target`` events."""
    log = m.events
    n = len(log)
    span = max(log.year) + 1
    k = 1
    while len(log) < target:
        off = k * span
        for col, values in ((log.link, [-1] * n), (log.people, log.people[:n]), (log.polity_people, log.polity_people[:n]),
                            (log.origin, log.origin[:n]), (log.year, [y + off for y in log.year[:n]]),
                            (log.action, log.action[:n]), (log.actors, log.actors[:n]), (log.place, log.place[:n]),
                            (log.magnitude, log.magnitude[:n])):
            col.extend(values)
        k += 1
    m.year = max(log.year) + 1


def test_chronicle_panel_renders_in_200_ms_at_year_4000_size():
    from deluge.viz.app import Session, chronicle_panel
    from deluge.viz.portrayal import people_names

    m = small_model(years=200)
    m.run()
    _grow_log(m, YEAR_4000_EVENTS)
    session = Session(m)
    view = chronicle.Chronicle(m)
    peoples = people_names(m)
    t0 = time.perf_counter()
    body, shown, total = chronicle_panel(session, view, peoples, peoples, "", m.config.chronicle.page_entries)
    cold = time.perf_counter() - t0
    assert shown == min(total, m.config.chronicle.page_entries) and body.count("<div") == shown
    # A refresh while playing: the events of the frames since the last one, then the newest page.
    log = m.events
    n = len(log)
    for col in (log.link, log.people, log.polity_people, log.origin, log.year, log.action, log.actors, log.place,
                log.magnitude):
        col.extend(col[n - 4000:n] if col is not log.year else [m.year] * 4000)
    m.year += 1
    t0 = time.perf_counter()
    chronicle_panel(session, view, peoples, peoples, "", m.config.chronicle.page_entries)
    warm = time.perf_counter() - t0
    print(f"chronicle panel at {len(log):,} events: first {1000 * cold:.0f} ms, refresh {1000 * warm:.0f} ms")
    assert warm < 0.2
    assert re.search(r"[A-Z]", body)
