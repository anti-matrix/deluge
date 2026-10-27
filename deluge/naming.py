"""Names of things and plain-language sentences, shared by the browser UI, the chronicle and
the reports (``analysis/narrator.py``), so an entity has one name everywhere.

Nothing here enters the simulation: it only reads model state.

- Phrase table: ``configs/narrator.yaml`` (``reports.narrator_file``): the syllable inventory
  of the name generator, plain phrases for techniques, minerals and what could be witnessed,
  how each kind of entity is spoken of, one sentence shape per verb, the chronicle's summary
  phrases and the inspector's words.
- Generator (``name``): a pure function of an entity key, a dialect and the (world, history)
  seeds. The dialect weighs the inventory, so peoples sound different.
- Canonical names (``proper_name``, ``Speaker.entity``): every entity is named in the speech
  of its own founding people as it was at year 0 (``identity.founding_dialect``): a
  settlement and a polity in their founding lineage's, a lineage in its own; the teachers in
  a neutral speech (a zero dialect). So a name never changes during a run and is the same in
  every view. A lineage is "the {name}", a polity "the {name} people", a settlement its name;
  ``tags`` adds the founding group's letter, e.g. "the Keraithaa (B)".
- Places (``deluge/gazetteer.py``): a cell is named only while a settlement stands on it, with
  that settlement's name; an empty cell is told relative to the nearest named place, a region
  after its most important named place. Every name is made when first needed (``_cache``).
"""

from __future__ import annotations

import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from deluge.agents.teachers import ARTIFACT_KINDS
from deluge.config import DEFAULT_CONFIG_PATH
from deluge.data.events import ACTIONS, ENTITY_KINDS, PEOPLE_KINDS, decode
from deluge.roles import FIELD, table_for
from deluge.world.worlddata import MINERALS

PLACE_CELL, PLACE_REGION = 0, 1  # place levels (``systems/memory.py``: PLACE_CELL, PLACE_REGION)
PEOPLE_NAMED = ("lineage", "polity", "settlement")  # named in their founding people's speech


@lru_cache(maxsize=4)
def load_data(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_for(model) -> dict:
    return load_data(str(Path(DEFAULT_CONFIG_PATH).parent / model.config.reports.narrator_file))


# --- the generator --------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _direction(item: str, dims: int) -> np.ndarray:
    """A fixed unit vector per inventory item, hashed from its spelling."""
    v = np.random.default_rng(zlib.crc32(item.encode("utf-8"))).standard_normal(dims)
    return v / np.linalg.norm(v)


_WEIGHTS: dict[tuple, np.ndarray] = {}


def _weights(items, dialect: np.ndarray, scale: float) -> np.ndarray:
    d = np.asarray(dialect, dtype=float)
    key = (tuple(items), d.tobytes(), float(scale))
    cdf = _WEIGHTS.get(key)
    if cdf is None:
        w = np.array([np.exp(scale * float(d @ _direction(f"{i}|{it}", len(d)))) for i, it in enumerate(items)])
        cdf = np.cumsum(w / w.sum())
        _WEIGHTS[key] = cdf
    return cdf


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


def plural(n: float, one: str, many: str | None = None, fmt: str = "{:,.0f}") -> str:
    """``n`` and its noun, singular when ``n`` shows as 1 ("1 sea cell", "2 sea cells");
    ``many`` defaults to ``one`` + "s"."""
    shown = fmt.format(n)
    return f"{shown} {one if shown == '1' else many or one + 's'}"


def number_form(form, n: float, **values) -> str:
    """A phrase table entry that is one string or a pair [one, many], chosen by ``n`` as shown
    (``values["m"]`` when given, else ``n`` rounded), then filled with ``values``."""
    if isinstance(form, list):
        shown = str(values.get("m", f"{n:,.0f}"))
        form = form[0] if shown in ("1", "one") else form[1]
    return form.format(**values)


# --- canonical names --------------------------------------------------------------------------------


def naming_lineage(model, key: tuple[int, int]) -> int:
    """The founding people in whose speech entity ``key`` is named (-1: the neutral speech)."""
    code, i = key
    if code >= len(ENTITY_KINDS):
        return -1
    kind = ENTITY_KINDS[code]
    if kind == "lineage":
        return int(i)
    if kind == "polity":
        return int(model.polities.lineage[i])
    if kind == "settlement":
        return int(model.settlements.lineage[i])
    return -1


def _cache(model) -> dict:
    """Names already made for this model (kept off its pickled state)."""
    return model.__dict__.setdefault("_names", {})


def proper_name(model, key: tuple[int, int]) -> str:
    """The canonical name of entity ``key`` (module docstring)."""
    names = _cache(model)
    out = names.get(key)
    if out is None:
        from deluge.systems import identity

        dialect = identity.founding_dialect(model, naming_lineage(model, key))
        out = name(key, dialect, (model.config.run.world_seed, model.config.run.history_seed),
                   data_for(model)["names"])
        names[key] = out
    return out


def entity_detail(model, eid: int) -> tuple[str, str, int]:
    """(kind, what it is by its registry name, a cell or -1) for an entity: a technique's name in
    the knowledge graph, a technology's or an artifact's kind, a deposit's mineral and cell;
    for the other kinds the name is empty."""
    kind, i = decode(eid)
    if kind == "technique":
        return kind, model.knowledge.names[i], -1
    if kind == "technology":
        return kind, model.config.teachers.technologies[i], -1
    if kind == "artifact":
        return kind, ARTIFACT_KINDS[model.teachers_state.kind[i]], -1
    if kind == "deposit":
        cell, mineral = divmod(i, len(MINERALS))
        return kind, MINERALS[mineral], cell
    return kind, "", -1


class Speaker:
    """Names things and says what happened in plain language.

    ``entity``, ``place`` and the sentences use the canonical names. ``proper`` names a key in
    this speaker's own dialect (``dialect``, default the neutral speech): the generator itself,
    for reports that compare how peoples sound. ``tags``: lineages and polities carry their
    founding group's letter, e.g. "the Keraithaa (B)"."""

    def __init__(self, model, g: int = -1, dialect=None, data: dict | None = None, tags: bool = False) -> None:
        self.model, self.g = model, g
        from deluge.systems import identity

        self.data = data_for(model) if data is None else data
        self.speech = identity.founding_dialect(model, -1) if dialect is None else np.asarray(dialect, dtype=float)
        self.seeds = (model.config.run.world_seed, model.config.run.history_seed)
        self.tags = tags
        self._names: dict[tuple[int, int], str] = {}

    def proper(self, key: tuple[int, int]) -> str:
        if key not in self._names:
            self._names[key] = name(key, self.speech, self.seeds, self.data["names"])
        return self._names[key]

    def tag(self, eid: int) -> str:
        """" (B)": the founding group of a lineage or polity, when ``tags``."""
        kind, i = decode(eid)
        if not self.tags or kind not in ("lineage", "polity"):
            return ""
        g = i if kind == "lineage" else int(self.model.polities.lineage[i])
        return f" ({self.model.lineages.names[g]})"

    def place(self, place: int, level: int = PLACE_CELL, at: int | None = None) -> str:
        """A cell (as it stood when event ``at`` was logged; None: now) or a memory region (now),
        in words (``deluge/gazetteer.py``)."""
        from deluge.gazetteer import gazetteer

        from deluge.gazetteer import Place

        if place < 0 or level not in (PLACE_CELL, PLACE_REGION):
            return Place.say(self.data["places"], "none")
        if level == PLACE_REGION:
            return gazetteer(self.model).region(place, self)
        return gazetteer(self.model).place(place, self, at)

    def event_place(self, i: int) -> str:
        """Where event ``i`` of the log happened, as it could be said then (``Gazetteer.event_place``)."""
        from deluge.gazetteer import gazetteer

        return gazetteer(self.model).event_place(i, self)

    def people(self, g: int) -> str:
        """A founding lineage, as spoken of (with its tag)."""
        from deluge.data.events import entity

        return self.entity(entity("lineage", g))

    def technique(self, j: int) -> str:
        return self.data["techniques"][self.model.knowledge.names[j]]

    def entity(self, eid: int) -> str:
        kind, i = decode(eid)
        d = self.data
        forms = d["forms"]
        key = (ENTITY_KINDS.index(kind), i)
        if kind == ENTITY_KINDS[0]:
            return forms["first_kind"].format(name=proper_name(self.model, key))
        if kind in PEOPLE_NAMED:
            return forms[kind].format(name=proper_name(self.model, key)) + self.tag(eid)
        _, what, cell = entity_detail(self.model, eid)
        if kind == "technique":
            return forms[kind].format(phrase=d["techniques"][what])
        if kind in ("technology", "artifact"):
            return forms[kind].format(phrase=d["technologies"][what])
        if kind == "deposit":
            return forms[kind].format(phrase=d["minerals"][what], place=self.place(cell))
        return forms.get(kind, "{name}").format(name=proper_name(self.model, key))

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

    # --- sentences from roles (``deluge/roles.py``) -------------------------------------------------

    def roles(self, action: int, actors, slots) -> list[str | None]:
        """Each actor's role by its slot; None for a slot of another verb (or none)."""
        rt = table_for(self.data)
        slots = list(slots) + [None] * (len(actors) - len(slots))
        return [rt.role(s, a) if s is not None and s[0] == action else None for s, a in zip(slots, actors)]

    def tell(self, action: int, actors, slots, place: str = "", time: str | None = None, count: int = 1) -> str:
        """One sentence of ``action`` from its actors' roles (``slots``: each actor's slot), where
        it happened (a place phrase, "" for none) and when; ``count`` > 1 tells a run. A set of
        roles with no shape of the verb is told in the loose form (``loose``)."""
        d = self.data
        found = self.shape_of(action, actors, slots, count)
        if found is None:
            return self.loose(action, actors, place, time)
        shape, own_run, by_role = found
        names = {r: d["joiner"].join(self.counted(v)) for r, v in by_role.items()}
        towns = [a for a in actors if decode(a)[0] == "settlement"]
        if place and any(place == d["places"]["cell"].format(name=self.entity(a)) for a in towns):
            place = ""  # the place bears the name of a settlement the sentence names
        lead = next((f for f in FIELD.findall(shape) if f in by_role), None)
        many = lead is not None and (len(by_role[lead]) > 1 or self.many(by_role[lead][0]))
        from deluge.gazetteer import destination

        to = destination(place, d) if place else ""
        text = shape.format(**names, at=f" {place}" if place else "", to=f" {to}" if to else "",
                            be=d["be"]["many" if many else "one"], count=f"{count:,}")
        if count > 1 and not own_run:
            text += d["run"].format(count=self.number(count))
        if time:
            text = f"{time}, {text}"
        return text[:1].upper() + text[1:] + "."

    def shape_of(self, action: int, actors, slots, count: int = 1):
        """(shape, whether a run's own shape, actors by role) of a sentence of these roles, None
        when they make none (the loose form)."""
        rt = table_for(self.data)
        by_role: dict[str, list[int]] = {}
        for a, r in zip(actors, self.roles(action, actors, slots)):
            if r is None or not rt.fits(r, a):
                return None
            by_role.setdefault(r, []).append(a)
        shape, own_run = rt.shape(action, frozenset(by_role), count)
        return None if shape is None else (shape, own_run, by_role)

    def loose(self, action: int, actors, place: str = "", time: str | None = None) -> str:
        """The loose form: "Of the Kavar and the Vahza, and a battle at Adtova, in the first age"."""
        d = self.data
        f = d["loose"]
        agents = [a for a in actors if decode(a)[0] in PEOPLE_KINDS]
        things = [a for a in actors if decode(a)[0] not in PEOPLE_KINDS]
        deed = f["deeds"][ACTIONS[action]]
        if things:
            deed = f["things"].format(deed=deed, things=d["joiner"].join(self.counted(things)))
        if place:
            deed = f"{deed} {place}"
        parts = [f["people"].format(agents=d["joiner"].join(self.counted(agents))), f["deed"].format(deed=deed)] \
            if agents else [deed]
        if time:
            parts.append(time)
        text = ", ".join(parts)
        return text[:1].upper() + text[1:] + "."

    def sentence(self, action: int, actors, place: str = "", time: str | None = None, via: int = -1) -> str:
        """An event's sentence from its verb and actors in log order (``via``: the verb of the
        event it is part of)."""
        return self.tell(action, actors, table_for(self.data).slots(action, actors, via), place, time)

    def event_slots(self, i: int) -> list[tuple]:
        ev = self.model.events
        link = ev.link[i]
        return table_for(self.data).slots(ev.action[i], ev.actors[i], ev.action[link] if link >= 0 else -1)

    def event(self, i: int, amount: bool = True) -> str:
        """Event ``i`` of the log as one sentence, with its size (``amounts``)."""
        return self.run([i], amount)

    def conquest(self, events) -> str:
        """A conquest of one year, victor and settlement (the chronicle's joined entry: its took,
        lost and died events) as one sentence: "In year 1274, the Goxeu people took Geugeu and
        destroyed the Ares people (20,196 people left)."; with no end of a people, the taking of
        their seat; with neither, the taking alone."""
        ev, d = self.model.events, self.data
        by = {ACTIONS[ev.action[i]]: i for i in events}
        took, lost, died = by.get("took"), by.get("lost"), by.get("died")
        if died is None and lost is None:
            return self.run([took])
        first = took if took is not None else lost
        town = ev.actors[first][1]
        if took is not None:
            victor, loser = ev.actors[took][0], ev.actors[took][2]
        else:
            victor, loser = ev.actors[lost][2], ev.actors[lost][0]
        names = {"taker": self.entity(victor), "taken": self.entity(town), "former": self.entity(loser)}
        if died is not None:
            text, form, m = d["conquest"]["destroyed"], d["amounts"]["died"], float(ev.magnitude[died])
        else:
            text, form, m = d["conquest"]["seat"], d["amounts"]["took"], float(ev.magnitude[first])
        text = f"{self.time(int(ev.year[first]))}, {text.format(**names)}"
        return text[:1].upper() + text[1:] + f" ({number_form(form, m, m=f'{m:,.0f}')})."

    def run(self, events, amount: bool = True) -> str:
        """A run of one event (same verb, groups of people and place) told once: the first
        event's roles, the run's years and count, the summed size."""
        ev = self.model.events
        i, last = events[0], events[-1]
        a, b = int(ev.year[i]), int(ev.year[last])
        time = self.data["times"]["run"].format(year=a, last=b) if b > a else self.time(a)
        text = self.tell(ev.action[i], ev.actors[i], self.event_slots(i), self.event_place(i), time, len(events))
        form = self.data["amounts"].get(ACTIONS[ev.action[i]]) if amount else None
        if form:
            total = sum(ev.magnitude[k] for k in events)
            text = text[:-1] + f" ({number_form(form, total, m=f'{total:,.0f}')})."
        return text
