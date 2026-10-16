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
the syllable inventory of the name generator. A name is a pure function of the entity, the
naming people's dialect when the report is made and the world and history seeds
(``name``): the entity and seeds fix the draws, the dialect weighs the inventory, so peoples
sound different and drift apart as their dialects do. Names never enter the simulation.

The structured output (``narrate``, written as JSON next to the markdown) is the interface for
a later, optional rewriting pass; it holds every name, field and sentence the markdown shows.
"""

from __future__ import annotations

import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from analysis.provenance import History, entity_detail, origin_records
from deluge.config import DEFAULT_CONFIG_PATH
from deluge.data.events import ACTIONS, ENTITY_KINDS, decode, entity
from deluge.systems.memory import PLACE_CELL, PLACE_REGION

PLACE_KINDS = ("cell", "region")  # name keys after the entity kinds
REMEMBERED_FIELDS = ("action", "actors", "place", "place_level", "year", "year_span")


@lru_cache(maxsize=4)
def load_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_for(model) -> dict:
    return load_data(str(Path(DEFAULT_CONFIG_PATH).parent / model.config.reports.narrator_file))


# --- names -------------------------------------------------------------------------------------


def _direction(item: str, dims: int) -> np.ndarray:
    """A fixed unit vector per inventory item, hashed from its spelling."""
    v = np.random.default_rng(zlib.crc32(item.encode("utf-8"))).standard_normal(dims)
    return v / np.linalg.norm(v)


def _weights(items, dialect: np.ndarray, scale: float) -> np.ndarray:
    d = np.asarray(dialect, dtype=float)
    w = np.array([np.exp(scale * float(d @ _direction(f"{i}|{it}", len(d)))) for i, it in enumerate(items)])
    return np.cumsum(w / w.sum())


def _pick(items, cdf: np.ndarray, u: float):
    return items[min(int(np.searchsorted(cdf, u, side="right")), len(items) - 1)]


def name(key: tuple[int, int], dialect, seeds: tuple[int, int], spec: dict) -> str:
    """The name of entity ``key`` = (kind code, local id) in the speech of a people with
    ``dialect``: a pure function of the key, the dialect and the (world, history) seeds."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seeds[0]), int(seeds[1]), int(key[0]), int(key[1])]))
    scale = float(spec["dialect_scale"])
    on, vo, co, en = (_weights(spec[k], dialect, scale) for k in ("onsets", "vowels", "codas", "endings"))
    counts = [int(c) for c, _ in spec["syllables"]]
    p = np.cumsum([float(q) for _, q in spec["syllables"]])
    n = counts[min(int(np.searchsorted(p / p[-1], rng.random(), side="right")), len(counts) - 1)]
    draws = rng.random((max(counts), 4))  # always the same number of draws: names drift, not jump
    tail = rng.random(2)
    ending = tail[0] < float(spec["ending_chance"])
    parts = []
    for k in range(n):
        u_on, u_vo, u_has, u_co = draws[k]
        s = _pick(spec["onsets"], on, u_on) + _pick(spec["vowels"], vo, u_vo)
        if u_has < float(spec["coda_chance"]) and not (ending and k == n - 1):
            s += _pick(spec["codas"], co, u_co)
        parts.append(s)
    if ending:
        parts.append(_pick(spec["endings"], en, tail[1]))
    out = ""
    for s in parts:
        if out and len(out) + len(s) > int(spec["max_letters"]):
            break
        out += s
    return out[:1].upper() + out[1:]


class Speaker:
    """One people's way of naming things and saying what happened, at report time."""

    def __init__(self, model, g: int, dialect: np.ndarray, data: dict) -> None:
        self.model, self.g, self.dialect, self.data = model, g, np.asarray(dialect, dtype=float), data
        self.seeds = (model.config.run.world_seed, model.config.run.history_seed)
        self._names: dict[tuple[int, int], str] = {}

    def proper(self, key: tuple[int, int]) -> str:
        if key not in self._names:
            self._names[key] = name(key, self.dialect, self.seeds, self.data["names"])
        return self._names[key]

    def place(self, place: int, level: int = PLACE_CELL) -> str:
        p = self.data["places"]
        if place < 0 or level not in (PLACE_CELL, PLACE_REGION):
            return p["none"]
        kind = len(ENTITY_KINDS) + (0 if level == PLACE_CELL else 1)
        return p["cell" if level == PLACE_CELL else "region"].format(name=self.proper((kind, place)))

    def entity(self, eid: int) -> str:
        kind, i = decode(eid)
        d = self.data
        forms = d["forms"]
        key = (ENTITY_KINDS.index(kind), i)
        if kind == ENTITY_KINDS[0]:
            return forms["first_kind"].format(name=self.proper(key))
        if kind in ("settlement", "polity", "lineage"):
            return forms[kind].format(name=self.proper(key))
        _, what, cell = entity_detail(self.model, eid)
        if kind == "technique":
            return forms[kind].format(phrase=d["techniques"][what])
        if kind in ("technology", "artifact"):
            return forms[kind].format(phrase=d["technologies"][what])
        if kind == "deposit":
            return forms[kind].format(phrase=d["minerals"][what], place=self.place(cell))
        return forms.get(kind, "{name}").format(name=self.proper(key))

    def time(self, year: int, span: int = 1) -> str:
        t = self.data["times"]
        if span > 1 and year == 0:
            return t["first_age"]
        if span > 1:
            return t["span"].format(year=year, last=year + span - 1)
        return t["year"].format(year=year)

    def sentence(self, action: int, actors, place: str, time: str) -> str:
        """One sentence from a verb, actors in order, a place and a time phrase; actors the
        verb's shape leaves out are added at the end, in order."""
        d = self.data
        verb = ACTIONS[action]
        template = d["templates"][verb]
        names = [self.entity(a) for a in actors]
        used = {int(k) for k in "0123456789" if "{" + k + "}" in template}
        if "{rest}" in template:
            used |= set(range(2, len(names)))
        if "{from_}" in template and len(names) > 2:
            used.add(2)
        padded = names + [d["someone"]] * 10
        rest = d["joiner"].join(names[2:]) if len(names) > 2 else d["nothing"]
        from_ = f" from {names[2]}" if len(names) > 2 else ""
        text = template.format(*padded, rest=rest, from_=from_, place=place, time=time)
        others = [n for k, n in enumerate(names) if k not in used]
        if others:
            text += d["with_others"].format(others=d["joiner"].join(others))
        text = text[:1].upper() + text[1:]
        return text + "."


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
    """The original events, in year order (then log order), as sentences."""
    ev = model.events
    out = []
    for i in sorted(events, key=lambda i: (ev.year[i], i)):
        text = sp.sentence(ev.action[i], ev.actors[i], sp.place(ev.place[i]), sp.time(ev.year[i]))
        amount = sp.data["amounts"].get(ACTIONS[ev.action[i]])
        if amount:
            text = text[:-1] + f" ({amount.format(m=f'{ev.magnitude[i]:,.0f}')})."
        out.append({"event": int(i), "year": int(ev.year[i]), "action": ACTIONS[ev.action[i]],
                    "actors": [{"entity": int(a), "name": sp.entity(a)} for a in ev.actors[i]],
                    "place": {"cell": int(ev.place[i]), "name": sp.place(ev.place[i])}, "text": text})
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
        out.append(c["actors_lost"].format(names=sp.data["joiner"].join(sp.entity(a) for a in original)))
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
                            "copies": int(e["copies"]),
                            "happened": happened(model, sp, rec.provenance),
                            "remembered": remembered(sp, record_fields(rec)),
                            "changed": changes(model, sp, e)})
        peoples.append({"group": model.lineages.names[g], "lineage": g,
                        "name": sp.entity(entity("lineage", g)), "dialect": [round(float(x), 6) for x in dialect],
                        "records": records})
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
        out += [f"## {p['name']} (group {p['group']})", ""]
        for k, r in enumerate(p["records"], 1):
            out += [f"### {k}. Record #{r['record']} ({r['copies']} cop{'ies' if r['copies'] != 1 else 'y'}, "
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
