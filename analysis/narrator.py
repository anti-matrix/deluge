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


def _units(s: str, spec: dict) -> list[bool]:
    """The word as sound units, True for a vowel letter; a digraph is one consonant."""
    vowels, digraphs = spec["vowel_letters"], spec["digraphs"]
    out, i = [], 0
    while i < len(s):
        if s[i] in vowels:
            out.append(True)
            i += 1
        else:
            i += 2 if s[i:i + 2] in digraphs else 1
            out.append(False)
    return out


def syllable_count(s: str, spec: dict) -> int:
    """Vowel groups in ``s``."""
    u = _units(s, spec)
    return sum(1 for k, v in enumerate(u) if v and (k == 0 or not u[k - 1]))


def phonotactic(s: str, spec: dict) -> bool:
    """No vowel run longer than ``max_vowel_run``, at most ``max_clusters`` consonant clusters,
    no doubled letter outside ``allowed_doubles``."""
    run = 0
    for ch in s:
        run = run + 1 if ch in spec["vowel_letters"] else 0
        if run > int(spec["max_vowel_run"]):
            return False
    if any(s[k] == s[k + 1] and s[k:k + 2] not in spec["allowed_doubles"] for k in range(len(s) - 1)):
        return False
    u = _units(s, spec)
    clusters = sum(1 for k in range(1, len(u)) if not u[k] and not u[k - 1] and (k == 1 or u[k - 2]))
    return clusters <= int(spec["max_clusters"])


def _chance_pick(pairs, u: float):
    p = np.cumsum([float(q) for _, q in pairs])
    return pairs[min(int(np.searchsorted(p / p[-1], u, side="right")), len(pairs) - 1)][0]


def name(key: tuple[int, int], dialect, seeds: tuple[int, int], spec: dict) -> str:
    """The name of entity ``key`` = (kind code, local id) in the speech of a people with
    ``dialect``: a pure function of the key, the dialect and the (world, history) seeds.

    The dialect weighs the traditions and their pieces; the name is built piece by piece from
    one tradition: an initial (or a first syllable), syllables of the shapes V, CV, CVC, and
    an ending, 2 to 4 syllables in all. A piece that would break the phonotactics
    (``phonotactic``) or ``max_letters`` is replaced by a plainer one: the same syllable without
    its coda, then a bare vowel or consonant-vowel; failing those it is left out."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seeds[0]), int(seeds[1]), int(key[0]), int(key[1])]))
    scale = float(spec["dialect_scale"])
    names = sorted(spec["traditions"])
    trad = spec["traditions"][_pick(names, _weights(names, dialect, scale), rng.random())]
    cdf = {k: _weights(trad[k], dialect, scale) for k in ("initials", "consonants", "clusters", "vowels", "codas",
                                                           "endings")}
    n = int(_chance_pick(spec["syllables"], rng.random()))
    draws = rng.random((6, 7))  # always the same number of draws: names drift, not jump
    head, tail = rng.random(4), rng.random(2)
    limit = int(spec["max_letters"])

    def pick(kind, u):
        return _pick(trad[kind], cdf[kind], u)

    def fits(s):
        return len(s) <= limit and phonotactic(s, spec) and syllable_count(s, spec) <= n

    def syllable(row, out):
        shape = _chance_pick(spec["templates"], row[0])
        v = pick("vowels", row[1])
        c = pick("clusters", row[3]) if row[2] < float(spec["cluster_chance"]) else pick("consonants", row[3])
        full = v if shape == "V" else c + v + (pick("codas", row[4]) if shape == "CVC" else "")
        plain = pick("consonants", row[5]) + v if out and out[-1] in spec["vowel_letters"] else v
        return [full, c + v if shape != "V" else full, plain]

    out = pick("initials", head[1]) if head[0] < float(spec["initial_chance"]) else ""
    ending = pick("endings", tail[1]) if tail[0] < float(spec["ending_chance"]) else ""
    room = n - (syllable_count(ending, spec) if ending else 0)
    for row in draws:
        if syllable_count(out, spec) >= max(room, 1 if ending else 2):
            break
        for piece in syllable(row, out):
            if fits(out + piece):
                out += piece
                break
    if ending and fits(out + ending):
        out += ending
    for row in draws[::-1]:  # at least two syllables
        if syllable_count(out, spec) >= 2:
            break
        for piece in syllable(row, out)[1:]:
            if fits(out + piece):
                out += piece
                break
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

    def number(self, n: int) -> str:
        words = self.data["numbers"]
        return words[n] if n < len(words) else f"{n:,}"

    def counted(self, eids) -> list[str]:
        """Names of ``eids`` in order, identical ones told once with their count."""
        d = self.data
        names = [self.entity(a) for a in eids]
        counts = {n: names.count(n) for n in names}
        out, seen = [], set()
        for a, n in zip(eids, names):
            if n in seen:
                continue
            seen.add(n)
            if counts[n] == 1:
                out.append(n)
                continue
            kind, _ = decode(a)
            form = d["plural_forms"].get(kind, d["plural_forms"]["default"])
            phrase = ""
            if kind in ("technology", "artifact"):
                phrase = d["technologies"][entity_detail(self.model, a)[1]]
            out.append(form.format(count=self.number(counts[n]), phrase=phrase, name=n))
        return out

    def many(self, eid: int) -> bool:
        """Whether the entity is spoken of as many people (verb agreement)."""
        kind = decode(eid)[0]
        kinds = self.data["plural_kinds"]
        return kind in kinds or ("first_kind" in kinds and kind == ENTITY_KINDS[0])

    def sentence(self, action: int, actors, place: str, time: str) -> str:
        """One sentence from a verb, actors in order, a place and a time phrase; actors the
        verb's shape leaves out are added at the end, in order; identical names are counted."""
        d = self.data
        verb = ACTIONS[action]
        template = d["templates"][verb]
        if len(actors) <= 2 and verb in d["empty_templates"]:
            template = d["empty_templates"][verb]
        names = [self.entity(a) for a in actors]
        used = {int(k) for k in "0123456789" if "{" + k + "}" in template}
        if "{rest}" in template:
            used |= set(range(2, len(names)))
        if "{from_}" in template and len(names) > 2:
            used.add(2)
        padded = names + [d["someone"]] * 10
        rest = d["joiner"].join(self.counted(actors[2:])) if len(names) > 2 else d["nothing"]
        from_ = f" from {names[2]}" if len(names) > 2 else ""
        be = d["be"]["many" if actors and self.many(actors[0]) else "one"]
        text = template.format(*padded, rest=rest, from_=from_, place=place, time=time, be=be)
        others = [a for k, a in enumerate(actors) if k not in used]
        if others:
            text += d["with_others"].format(others=d["joiner"].join(self.counted(others)))
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
        text = sp.sentence(ev.action[i], ev.actors[i], sp.place(ev.place[i]), time)
        if len(run) > 1 and k_run not in stay_runs:
            text = text[:-1] + d["run"].format(count=sp.number(len(run))) + "."
        amount = d["amounts"].get(ACTIONS[ev.action[i]])
        if amount:
            total = sum(ev.magnitude[k] for k in run)
            text = text[:-1] + f" ({amount.format(m=f'{total:,.0f}')})."
        out.append({"event": int(i), "events": [int(k) for k in run], "year": a, "last_year": b,
                    "action": ACTIONS[ev.action[i]],
                    "actors": [{"entity": int(x), "name": sp.entity(x)} for x in ev.actors[i]],
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
