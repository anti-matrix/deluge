"""Deterministic narrator: plain-language texts of each people's oldest surviving records
(analysis only; nothing here feeds back into the simulation).

    from analysis.narrator import narrate, write
    data = narrate(model)                      # structured output (a dict, JSON-ready)
    md_path, json_path = write(model, "runs/origins_w1_h1")

For each founding group with living polities, every record from the "Origins" section of the
memory report (``analysis/provenance.py``: surviving records whose provenance includes an
event of years 0 to ``reports.origins_until_year``) is told twice, side by side:

- "What happened": the record's provenance, read from the raw event log, in year order;
- "What they remember": only the fields the record itself holds (its current verb, its actors
  in their order, its place at the level it still has, its year or span), nothing from its
  provenance (``remembered``);

then what memory changed: actors substituted, events fused, details lost.

Wording comes from ``configs/narrator.yaml`` (``reports.narrator_file``): a fixed phrase
table for techniques, minerals and what could be witnessed, one sentence shape per verb, and
the syllable inventory of the name generator (``deluge/naming.py``, shared with the browser UI
and the chronicle). A name is a pure function of the entity, a dialect and the world and
history seeds (``name``); every entity is named in its own founding people's speech as it was
at year 0, so it has the same name in every report and view (``naming.proper_name``). Names
never enter the simulation.

The structured output (``narrate``, written as JSON next to the markdown) is the interface for
a later, optional rewriting pass; it holds every name, field and sentence the markdown shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.provenance import History, origin_records
from deluge.data.events import ACTIONS, entity
# The generator, the phrase table and the speaker live in ``deluge/naming.py`` (shared with the
# UI and the chronicle); they are re-exported here for the reports.
from deluge.naming import (Speaker, data_for, entity_detail, load_data, name, phonotactic,  # noqa: F401
                           syllable_count)
from deluge.systems.memory import PLACE_REGION

REMEMBERED_FIELDS = ("action", "actors", "place", "place_level", "year", "year_span")


def people_dialect(model, g: int) -> np.ndarray:
    """A founding group's dialect when the report is made: the population-weighted mean over
    its living settlements."""
    t = model.settlements
    ids = t.alive_ids()
    ids = ids[t.lineage[ids] == g]
    if len(ids) == 0:
        return np.zeros(t.dialect.shape[1])
    w = np.maximum(t.population[ids], 1e-9)
    return (w @ t.dialect[ids]) / w.sum()


# --- the two texts -------------------------------------------------------------------------------


def happened(model, sp: Speaker, events) -> list[dict]:
    """The original events, in year order (then log order), as sentences; a run of identical
    events in a row (same verb, actors and place) is one sentence over its years, with its
    count and its summed size."""
    ev = model.events
    d = sp.data
    order = sorted(events, key=lambda i: (ev.year[i], i))
    runs: list[list[int]] = []
    stays: dict[tuple, int] = {}  # contacts that passed on nothing: one "stayed with" from the first to the last
    prev = None  # the key of the event before, while its run is the last one
    for i in order:
        key = (ev.action[i], ev.actors[i], ev.place[i])
        if len(ev.actors[i]) <= 2 and ACTIONS[ev.action[i]] in d["empty_templates"]:
            if key in stays:
                runs[stays[key]].append(i)
            else:
                stays[key] = len(runs)
                runs.append([i])
            prev = None
            continue
        if key == prev:
            runs[-1].append(i)
        else:
            runs.append([i])
        prev = key
    out = []
    stay_runs = set(stays.values())
    for k_run, run in enumerate(runs):
        i, last = run[0], run[-1]
        a, b = int(ev.year[i]), int(ev.year[last])
        time = d["times"]["run"].format(year=a, last=b) if b > a else sp.time(a)
        text = sp.sentence(ev.action[i], ev.actors[i], sp.event_place(i), time)
        if len(run) > 1 and k_run not in stay_runs:
            text = text[:-1] + d["run"].format(count=sp.number(len(run))) + "."
        amount = d["amounts"].get(ACTIONS[ev.action[i]])
        if amount:
            total = sum(ev.magnitude[k] for k in run)
            text = text[:-1] + f" ({amount.format(m=f'{total:,.0f}')})."
        out.append({"event": int(i), "events": [int(k) for k in run], "year": a, "last_year": b,
                    "action": ACTIONS[ev.action[i]],
                    "actors": [{"entity": int(x), "name": sp.entity(x)} for x in ev.actors[i]],
                    "place": {"cell": int(ev.place[i]), "name": sp.event_place(i)}, "text": text})
    return out


def remembered(sp: Speaker, fields: dict) -> dict:
    """What a record holds, told from its own fields only (``REMEMBERED_FIELDS``)."""
    unknown = set(fields) - set(REMEMBERED_FIELDS)
    if unknown:
        raise KeyError(f"remembered() reads only {REMEMBERED_FIELDS}, got {sorted(unknown)}")
    place = sp.place(fields["place"], fields["place_level"])
    time = sp.time(fields["year"], fields["year_span"])
    return {"action": ACTIONS[fields["action"]],
            "actors": [{"entity": int(a), "name": sp.entity(a)} for a in fields["actors"]],
            "place": {"level": ("cell", "region", "none")[min(int(fields["place_level"]), 2)],
                      "id": int(fields["place"]), "name": place},
            "year": int(fields["year"]), "year_span": int(fields["year_span"]),
            "text": sp.sentence(fields["action"], fields["actors"], place, time)}


def record_fields(rec) -> dict:
    return {k: getattr(rec, k) for k in REMEMBERED_FIELDS}


def changes(model, sp: Speaker, entry: dict) -> list[str]:
    c = sp.data["changes"]
    rec = entry["record"]
    out = [c["substituted"].format(new=sp.entity(b), old=sp.entity(a), year=y) for y, a, b in entry["substitutions"]]
    if len(rec.provenance) > 1:
        out.append(c["fused"].format(n=len(rec.provenance) - 1))
    original = []
    for i in rec.provenance:
        for a in model.events.actors[i]:
            if a not in rec.actors and a not in original:
                original.append(a)
    if original:
        out.append(c["actors_lost"].format(names=sp.data["joiner"].join(sp.counted(original))))
    if rec.year_span > 1:
        out.append(c["year_lost"].format(span=rec.year_span))
    if rec.place_level == PLACE_REGION:
        out.append(c["place_region"])
    elif rec.place < 0 or rec.place_level > PLACE_REGION:
        out.append(c["place_none"])
    if rec.magnitude_rounded:
        out.append(c["size_lost"])
    return out or [c["none"]]


# --- the report ----------------------------------------------------------------------------------


def sample_names(model, sp: Speaker) -> list[str]:
    """Names of up to ``reports.sample_names`` of the people's living settlements (lowest ids first), in its speech."""
    t = model.settlements
    ids = t.alive_ids()
    ids = ids[t.lineage[ids] == sp.g][: model.config.reports.sample_names]
    return [sp.entity(entity("settlement", int(i))) for i in ids]


def narrate(model) -> dict:
    """The structured output: per people, its name, its dialect and, per origin record, both
    texts and what memory changed."""
    data = data_for(model)
    hist = History(model)
    peoples = []
    for g in range(len(model.lineages.names)):
        entries = origin_records(model, g, hist)
        if not entries:
            continue
        dialect = people_dialect(model, g)
        sp = Speaker(model, g, dialect, data)
        records = []
        for e in entries:
            rec = e["record"]
            records.append({"record": int(rec.mem_id), "stores": sorted(int(p) for p in e["stores"]),
                            "copies": int(e["copies"]), "current": bool(e["current"]),
                            "happened": happened(model, sp, rec.provenance),
                            "remembered": remembered(sp, record_fields(rec)),
                            "changed": changes(model, sp, e)})
        peoples.append({"group": model.lineages.names[g], "lineage": g,
                        "name": sp.entity(entity("lineage", g)), "dialect": [round(float(x), 6) for x in dialect],
                        "sample_names": sample_names(model, sp), "records": records})
    return {"title": "Origins of the peoples", "world_seed": model.config.run.world_seed,
            "history_seed": model.config.run.history_seed, "year": int(model.year),
            "origins_until_year": model.config.reports.origins_until_year, "peoples": peoples}


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def markdown(data: dict) -> str:
    out = [f"# {data['title']}", "",
           f"World seed {data['world_seed']}, history seed {data['history_seed']}, year {data['year']}. Each people's "
           f"surviving records that descend from events of years 0 to {data['origins_until_year']}: what happened "
           "(from the event log) beside what they remember (from the record alone).", ""]
    if not data["peoples"]:
        out += ["No people holds a record from those years.", ""]
    for p in data["peoples"]:
        out += [f"## {p['name']} (group {p['group']})", "",
                "Names of its settlements: " + ", ".join(p["sample_names"]) + ".", ""]
        heading = None
        for k, r in enumerate(p["records"], 1):
            h = (f"Records still holding something of years 0 to {data['origins_until_year']}" if r["current"] else
                 f"Records descending from years 0 to {data['origins_until_year']} by provenance only")
            if h != heading:
                heading = h
                out += [f"### {'(a)' if r['current'] else '(b)'} {h}", ""]
            out += [f"#### {k}. Record #{r['record']} ({r['copies']} cop{'ies' if r['copies'] != 1 else 'y'}, "
                    f"in the stores of polities {', '.join(str(s) for s in r['stores'])})", "",
                    "| What happened | What they remember |", "| --- | --- |"]
            left = "<br>".join(_cell(h["text"]) for h in r["happened"])
            out.append(f"| {left} | {_cell(r['remembered']['text'])} |")
            out += ["", "What memory changed: " + "; ".join(r["changed"]) + ".", ""]
    return "\n".join(out) + "\n"


def write(model, stem: str | Path) -> tuple[Path, Path]:
    """``<stem>.md`` and, next to it, ``<stem>.json`` (the structured output)."""
    data = narrate(model)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    md, js = stem.with_suffix(".md"), stem.with_suffix(".json")
    md.write_text(markdown(data), encoding="utf-8")
    js.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return md, js
