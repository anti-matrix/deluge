"""Round 8b: the deterministic narrator (analysis/narrator.py) and the myths report's Origins."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from conftest import small_model
from test_memory import VOCABULARY

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from analysis import myths, narrator  # noqa: E402

NARRATOR_FILES = (REPO / "analysis" / "narrator.py", REPO / "configs" / "narrator.yaml")


def run(years=200):
    m = small_model(years=years)
    m.run()
    return m


@pytest.fixture(scope="module")
def ran():
    return run()


def test_no_myth_or_motif_vocabulary_in_the_narrator_or_its_phrase_table():
    for path in NARRATOR_FILES:
        hits = sorted({h.group(0).lower() for h in VOCABULARY.finditer(path.read_text(encoding="utf-8"))})
        assert hits == [], path.name


def test_phrase_table_covers_every_technique_mineral_and_verb(ran):
    from deluge.data.events import ACTIONS
    from deluge.world.worlddata import MINERALS

    data = narrator.data_for(ran)
    assert set(data["techniques"]) == set(ran.knowledge.names)
    assert set(data["minerals"]) == set(MINERALS)
    assert set(data["templates"]) == set(ACTIONS)
    assert set(data["technologies"]) >= set(ran.config.teachers.technologies)
    for k in ("onsets", "vowels", "codas", "endings"):
        assert all(isinstance(x, str) for x in data["names"][k])  # quoted: never read as booleans


def test_identical_runs_give_identical_output(ran, tmp_path):
    other = run()
    a, b = narrator.narrate(ran), narrator.narrate(other)
    assert a == b and a["peoples"]
    assert narrator.markdown(a) == narrator.markdown(b)
    md, js = narrator.write(ran, tmp_path / "origins")
    assert md.read_text(encoding="utf-8").startswith("# Origins of the peoples")
    assert json.loads(js.read_text(encoding="utf-8")) == json.loads(json.dumps(a))


def test_names_are_stable_across_reports(ran):
    a, b = narrator.narrate(ran), narrator.narrate(ran)
    assert a == b
    sp = narrator.Speaker(ran, 0, narrator.people_dialect(ran, 0), narrator.data_for(ran))
    sp2 = narrator.Speaker(ran, 0, narrator.people_dialect(ran, 0), narrator.data_for(ran))
    assert all(sp.proper((2, i)) == sp2.proper((2, i)) for i in range(50))
    # A pure function of entity, dialect and seeds.
    spec = narrator.data_for(ran)["names"]
    d = np.linspace(-1, 1, 8)
    assert narrator.name((2, 7), d, (1, 1), spec) == narrator.name((2, 7), d.copy(), (1, 1), spec)
    assert narrator.name((2, 7), d, (1, 1), spec) != narrator.name((2, 7), d, (2, 1), spec) or \
        narrator.name((2, 8), d, (1, 1), spec) != narrator.name((2, 8), d, (2, 1), spec)


def _letters(names):
    c = Counter(ch for n in names for ch in n.lower())
    total = sum(c.values())
    return {k: v / total for k, v in c.items()}


def _tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def test_peoples_with_different_dialects_sound_different(ran):
    spec = narrator.data_for(ran)["names"]
    rng = np.random.default_rng(0)
    da, db = rng.standard_normal(8), rng.standard_normal(8)
    a1 = [narrator.name((2, i), da, (1, 1), spec) for i in range(400)]
    a2 = [narrator.name((2, i), da, (1, 1), spec) for i in range(400, 800)]
    b1 = [narrator.name((2, i), db, (1, 1), spec) for i in range(400)]
    same, apart = _tv(_letters(a1), _letters(a2)), _tv(_letters(a1), _letters(b1))
    assert apart > 2.5 * same
    # Drift: a small change of dialect changes few names, a large one many.
    near = [narrator.name((2, i), da + 0.02, (1, 1), spec) for i in range(400)]
    assert sum(x != y for x, y in zip(a1, near)) < sum(x != y for x, y in zip(a1, b1))
    # The model's peoples: their report-time dialects differ, and so do their names.
    names = {}
    for g in range(len(ran.lineages.names)):
        sp = narrator.Speaker(ran, g, narrator.people_dialect(ran, g), narrator.data_for(ran))
        names[g] = [sp.proper((2, i)) for i in range(300)]
    base = _tv(_letters(names[0][:150]), _letters(names[0][150:]))
    assert _tv(_letters(names[0]), _letters(names[1])) > base


def test_what_they_remember_uses_only_the_record(ran):
    """The remembered text is built from the record's own fields: changing anything else (its
    provenance, its magnitude, the event log) leaves it unchanged; extra fields are refused."""
    data = narrator.data_for(ran)
    entries = [e for g in range(5) for e in narrator.origin_records(ran, g, narrator.History(ran))]
    assert entries
    sp = narrator.Speaker(ran, 0, narrator.people_dialect(ran, 0), data)
    for e in entries[:10]:
        rec = e["record"]
        fields = narrator.record_fields(rec)
        assert set(fields) == set(narrator.REMEMBERED_FIELDS)
        text = narrator.remembered(sp, fields)["text"]
        saved = (rec.provenance, rec.magnitude)
        rec.provenance, rec.magnitude = (0,), -1.0
        try:
            assert narrator.remembered(sp, narrator.record_fields(rec))["text"] == text
        finally:
            rec.provenance, rec.magnitude = saved
        with pytest.raises(KeyError):
            narrator.remembered(sp, {**fields, "provenance": rec.provenance})
        for a in rec.actors:
            assert sp.entity(a) in text
    # First age: a span starting at year 0.
    assert sp.time(0, 25) == data["times"]["first_age"] and sp.time(812) == "in year 812"
    assert sp.time(800, 100) == "in the years 800 to 899"


def test_myths_report_lists_origins_first(ran):
    text = myths.report(ran)
    assert text.index("## Origins") < text.index("## Group")
    hist = myths.History(ran)
    for g in range(5):
        for e in myths.origin_records(ran, g, hist):
            rec = e["record"]
            assert any(ran.events.year[i] <= ran.config.reports.origins_until_year for i in rec.provenance)
            assert f"**#{rec.mem_id}**" in text
            for i in rec.provenance:  # the full provenance is listed
                assert f"    - [{i}] " in text
