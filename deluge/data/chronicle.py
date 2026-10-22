"""The chronicle: the important events of the raw event log, told in the narrator's language
(``deluge/naming.py``), and the routine ones summed per generation.

An event is told one by one when a rule of ``chronicle`` (config) picks it. Every rule reads
only the event's action code, the kinds and ids of its actors and its numbers (magnitude,
the people of its actors' polities, the cells it names), never what an entity is:

- ``polity_actions`` with a polity as first actor (a polity founded, ended, united);
- ``capital_actions`` and ``always_actions`` (a capital lost; monuments);
- ``share_actions`` whose magnitude is at least ``death_share`` of the smallest population
  among its actors' polities when it was logged (battles, famines, settlements lost or taken);
- ``teachers_actions`` with the teachers as first actor and a magnitude of at least
  ``teachers_min_magnitude``;
- ``first_actions``: the first event anywhere with a given technique among its actors, and the
  first per founding lineage (the lineage of its first people actor);
- ``crossing_actions``: the first that reaches another landmass (land connected that year)
  than the one it set out from, per lineage, and per sea tier anywhere (the tier: the
  techniques' sea_steps, the smallest that covers the sea cells crossed).

Every other event is counted in a summary per generation (``generation_years``) and lineage,
by action, kind of first actor and technique (``summaries`` in the phrase table), once the
generation is over. Entries are kept in time order; a view shows the newest
``page_entries``, then more on demand. The log itself is never trimmed (``EventLog``).

``Chronicle.sync`` reads the log up to one snapshot length (the model may be appending), so
events appended meanwhile wait for the next call.
"""

from __future__ import annotations

import hashlib

import numpy as np

from deluge.data.events import ACTION_CODE, ACTIONS, ENTITY_KINDS, decode, entity
from deluge.naming import Speaker

N_KINDS = len(ENTITY_KINDS)
KIND = {k: i for i, k in enumerate(ENTITY_KINDS)}


class Namer:
    """Names entities as the UI and the reports do (``naming.Speaker``, with group tags)."""

    def __init__(self, model, tags: bool = True) -> None:
        self.model = model
        self.sp = Speaker(model, tags=tags)

    def lineage_of(self, eid: int) -> int:
        """Founding group of a settlement, polity or lineage entity (-1 for other entities)."""
        kind, i = decode(eid)
        if kind == "settlement":
            return int(self.model.settlements.lineage[i])
        if kind == "lineage":
            return i
        if kind == "polity":
            return int(self.model.polities.lineage[i])
        return -1

    def __call__(self, eid: int) -> str:
        return self.sp.entity(eid)


def line(record: dict, name: Namer) -> str:
    """One raw event (``EventLog.record``) as the narrator's sentence, with its size."""
    sp = name.sp
    place = sp.event_place(record["event"]) if "event" in record else sp.place(record["place"])
    text = sp.sentence(ACTION_CODE[record["action"]], record["actors"], place, sp.time(record["year"]))
    form = sp.data["amounts"].get(record["action"])
    if form:
        text = text[:-1] + f" ({form.format(m=f'{record['magnitude']:,.0f}')})."
    return text


def _kinds_locals(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(kind code, local id) of entity ids, -1 where the id is -1."""
    ok = ids >= 0
    return np.where(ok, ids % N_KINDS, -1), np.where(ok, ids // N_KINDS, -1)


class Chronicle:
    """The chronicle of one model (see the module docstring)."""

    def __init__(self, model) -> None:
        self.model = model
        cfg = model.config.chronicle
        self.cfg = cfg
        self.sp = Speaker(model, tags=True)
        self.data = self.sp.data
        codes = lambda names: np.array([ACTION_CODE[a] for a in names], dtype=np.int64)  # noqa: E731
        self.polity_codes = codes(cfg.polity_actions)
        self.capital_codes = codes(cfg.capital_actions)
        self.share_codes = codes(cfg.share_actions)
        self.teachers_codes = codes(cfg.teachers_actions)
        self.always_codes = codes(cfg.always_actions)
        self.first_codes = codes(cfg.first_actions)
        self.crossing_codes = codes(cfg.crossing_actions)
        g = model.knowledge
        steps = g.effect["sea_steps"]
        tiers = sorted({float(s) for s in steps if s > 0})
        # Per sea tier: its sea cells and the technique giving it (lowest index among equals).
        self.tiers = [(s, int(np.flatnonzero(steps == s)[0])) for s in tiers]
        self.n = 0  # events read so far
        # Entries as columns, in time order: the year (a summary: its generation's last year), the
        # type (0 an event told, 1 a summary), the event's index or the summary's number, and the
        # founding lineages taking part (a bit each). Text is made only for entries shown.
        self._year = np.zeros(0, dtype=np.int64)
        self._type = np.zeros(0, dtype=np.int64)
        self._ref = np.zeros(0, dtype=np.int64)
        self._groups = np.zeros(0, dtype=np.int64)
        self._new: list[tuple[np.ndarray, ...]] = []
        self.suffix: dict[int, list[str]] = {}  # event -> what the chronicle adds (firsts)
        self.summaries: list[dict] = []
        self._text: dict[tuple[int, int], str] = {}
        self._pending: list[np.ndarray] = []  # routine rows not yet summed: (gen, lineage, action, kind, detail, actor)
        self.emitted_gen = 0  # generations before this one are summed
        self.first_tech: set[int] = set()
        self.first_tech_lineage: set[tuple[int, int]] = set()
        self.crossed_lineage: set[int] = set()
        self.crossed_tier: set[int] = set()
        self._landmass: dict[bytes, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._year)

    def entry(self, k: int) -> dict:
        """Entry ``k`` (time order): an event told {"event", "year", "groups", "suffix"} or a
        summary {"event": -1, "year", "last", "lineage", "groups", "items"}."""
        if self._type[k] == 0:
            i = int(self._ref[k])
            return {"event": i, "year": int(self._year[k]), "groups": int(self._groups[k]),
                    "suffix": self.suffix.get(i, [])}
        return {"event": -1, **self.summaries[int(self._ref[k])], "groups": int(self._groups[k])}

    @property
    def entries(self) -> list[dict]:
        return [self.entry(k) for k in range(len(self))]

    # --- reading the log ----------------------------------------------------------------------------

    def _snapshot(self) -> int:
        """Number of events complete in every column right now (the model appends the columns
        in turn), read once per sync."""
        log = self.model.events
        return min(len(log.link), len(log.people), len(log.polity_people), len(log.origin), len(log.year),
                   len(log.action), len(log.actors), len(log.place), len(log.magnitude))

    def _lineages(self, kinds: np.ndarray, local: np.ndarray) -> np.ndarray:
        """Founding lineage of each actor (settlement, polity, lineage), -1 otherwise."""
        t, P = self.model.settlements, self.model.polities
        out = np.full(len(kinds), -1, dtype=np.int64)
        s = kinds == KIND["settlement"]
        out[s] = t.lineage[local[s]]
        p = kinds == KIND["polity"]
        out[p] = P.lineage[local[p]]
        lin = kinds == KIND["lineage"]
        out[lin] = local[lin]
        return out

    def _land_labels(self, year: int) -> np.ndarray:
        """Connected land components of ``year`` (its sea level), -1 for sea."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        m = self.model
        year = min(max(int(year), 0), len(m.sea_level_series) - 1)
        land = m.elevation_flat >= float(m.sea_level_series[year])
        key = hashlib.sha1(np.packbits(land).tobytes()).digest()
        labels = self._landmass.get(key)
        if labels is None:
            if not hasattr(self, "_edges"):
                nb = m.geom.neighbors
                c, k = np.nonzero(nb >= 0)
                self._edges = (c, nb[c, k])
            c, t = self._edges
            ok = land[c] & land[t]
            n = m.geom.n_cells
            graph = csr_matrix((np.ones(int(ok.sum())), (c[ok], t[ok])), shape=(n, n))
            _, lab = connected_components(graph, directed=False)
            labels = np.where(land, lab, -1)
            self._landmass[key] = labels
        return labels

    def _tier(self, sea_cells: float) -> int:
        """Index of the smallest sea tier covering ``sea_cells`` (the last one beyond all)."""
        for k, (s, _) in enumerate(self.tiers):
            if sea_cells <= s:
                return k
        return len(self.tiers) - 1

    def sync(self) -> None:
        """Read the events logged since the last call; sum the generations now over."""
        year_now = int(self.model.year)  # read before the log: events of earlier years are all in
        n = self._snapshot()
        if n > self.n:
            self._read(self.n, n)
            self.n = n
        done_gen = year_now // self.cfg.generation_years  # generations before it are over
        if done_gen > self.emitted_gen:
            self._summarize(done_gen)
        self._merge_new()

    def _read(self, i0: int, n: int) -> None:
        log = self.model.events
        cfg = self.cfg
        k = n - i0
        action = np.asarray(log.action[i0:n], dtype=np.int64)
        years = np.asarray(log.year[i0:n], dtype=np.int64)
        mag = np.asarray(log.magnitude[i0:n], dtype=float)
        actors = log.actors[i0:n]
        a = [np.array([x[0] for x in actors], dtype=np.int64),
             np.array([x[1] if len(x) > 1 else -1 for x in actors], dtype=np.int64),
             np.array([x[2] if len(x) > 2 else -1 for x in actors], dtype=np.int64)]
        kinds, local, lin = [], [], []
        for ids in a:
            kk, ll = _kinds_locals(ids)
            kinds.append(kk)
            local.append(ll)
            lin.append(self._lineages(kk, ll))
        groups = np.zeros(k, dtype=np.int64)
        for g in lin:
            groups |= np.where(g >= 0, np.left_shift(1, np.maximum(g, 0)), 0)
        owner = np.where(lin[0] >= 0, lin[0], np.where(lin[1] >= 0, lin[1], lin[2]))
        first_polity = kinds[0] == KIND["polity"]
        told = ((np.isin(action, self.polity_codes) & first_polity) | np.isin(action, self.capital_codes)
                | np.isin(action, self.always_codes)
                | (np.isin(action, self.teachers_codes) & (kinds[0] == KIND["teachers"])
                   & (mag >= cfg.teachers_min_magnitude)))
        share = np.flatnonzero(np.isin(action, self.share_codes) & ~told)
        if len(share):
            pp = log.polity_people
            width = max(len(pp[i0 + r]) for r in share.tolist())
            nan = (np.nan,) * width
            people = np.array([(pp[i0 + r] + nan)[:width] for r in share.tolist()], dtype=float)
            people[~(people > 0)] = np.nan
            known = ~np.isnan(people).all(axis=1)
            smallest = np.nanmin(np.where(known[:, None], people, 1.0), axis=1)
            told[share[known & (mag[share] >= cfg.death_share * smallest)]] = True
        suffix: dict[int, list[str]] = {}
        self._firsts(i0, action, kinds, local, owner, suffix)
        self._crossings(i0, n, action, years, mag, owner, suffix)
        told[[r for r in suffix]] = True
        self.suffix.update((i0 + r, s) for r, s in suffix.items())
        rows = np.flatnonzero(told)
        self._new.append((years[rows], np.zeros(len(rows), dtype=np.int64), i0 + rows, groups[rows]))
        rest = np.flatnonzero(~told)
        if len(rest):
            detail = np.where(kinds[1] == KIND["technique"], local[1], -1)
            rows = np.column_stack([years[rest] // cfg.generation_years, owner[rest], action[rest], kinds[0][rest],
                                    detail[rest], a[0][rest]])
            self._pending.append(rows)

    def _firsts(self, i0: int, action, kinds, local, owner, suffix: dict) -> None:
        """The first event anywhere, and per lineage, with each technique among its actors."""
        first = np.isin(action, self.first_codes)
        if not first.any():
            return
        log = self.model.events
        tech_kind = KIND["technique"]
        # (event row, technique) for every technique actor: the first three actors by array,
        # the rest of the rare longer events one by one.
        rows, techs = [], []
        for j in range(3):
            hit = np.flatnonzero(first & (kinds[j] == tech_kind))
            rows.append(hit)
            techs.append(local[j][hit])
        long = [r for r in np.flatnonzero(first).tolist() if len(log.actors[i0 + r]) > 3]
        extra = [(r, decode(x)[1]) for r in long for x in log.actors[i0 + r][3:] if x % N_KINDS == tech_kind]
        if extra:
            rows.append(np.array([r for r, _ in extra], dtype=np.int64))
            techs.append(np.array([j for _, j in extra], dtype=np.int64))
        r_all, t_all = np.concatenate(rows), np.concatenate(techs)
        if len(r_all) == 0:
            return
        order = np.argsort(r_all, kind="stable")
        r_all, t_all = r_all[order], t_all[order]
        n_lin = len(self.model.lineages.founders)
        # Only the first row of each technique, and of each (technique, lineage), can be a first.
        _, a = np.unique(t_all, return_index=True)
        _, b = np.unique(t_all * (n_lin + 1) + (owner[r_all] + 1), return_index=True)
        cand = np.unique(np.concatenate([a, b]))
        by_row: dict[int, list[int]] = {}
        for k in cand.tolist():
            by_row.setdefault(int(r_all[k]), []).append(int(t_all[k]))
        for r in sorted(by_row):
            g = int(owner[r])
            for j in by_row[r]:
                if j not in self.first_tech:
                    self.first_tech.add(j)
                    if g >= 0:
                        self.first_tech_lineage.add((j, g))
                    suffix.setdefault(r, []).append(self.data["chronicle"]["first_anywhere"].format(
                        phrase=self.sp.technique(j)))
                elif g >= 0 and (j, g) not in self.first_tech_lineage:
                    self.first_tech_lineage.add((j, g))
                    suffix.setdefault(r, []).append(self.data["chronicle"]["first_people"].format(
                        phrase=self.sp.technique(j), people=self.sp.people(g)))

    def _crossings(self, i0: int, n: int, action, years, mag, owner, suffix: dict) -> None:
        """The first crossing to another landmass per lineage, and per sea tier anywhere."""
        sel = np.flatnonzero(np.isin(action, self.crossing_codes))
        if len(sel) == 0 or not self.tiers:
            return
        log = self.model.events
        n_lin = len(self.model.lineages.founders)
        for r in sel.tolist():
            g = int(owner[r])
            tier = self._tier(float(mag[r]))
            if (g < 0 or g in self.crossed_lineage) and tier in self.crossed_tier:
                if len(self.crossed_lineage) == n_lin and len(self.crossed_tier) == len(self.tiers):
                    return
                continue
            origin, place = int(log.origin[i0 + r]), int(log.place[i0 + r])
            if origin < 0 or place < 0:
                continue
            labels = self._land_labels(int(years[r]))
            if labels[origin] < 0 or labels[place] < 0 or labels[origin] == labels[place]:
                continue
            c = self.data["chronicle"]
            if g >= 0 and g not in self.crossed_lineage:
                self.crossed_lineage.add(g)
                suffix.setdefault(r, []).append(c["first_land_people"].format(people=self.sp.people(g)))
            if tier not in self.crossed_tier:
                self.crossed_tier.add(tier)
                suffix.setdefault(r, []).append(c["first_land_tier"].format(
                    phrase=self.sp.technique(self.tiers[tier][1])))

    def _merge_new(self) -> None:
        """Put this sync's entries in time order with the others: by year, events before the
        summaries of their year, then by event index or summary number."""
        if not self._new:
            return
        cols = [np.concatenate([old] + [new[j] for new in self._new])
                for j, old in enumerate((self._year, self._type, self._ref, self._groups))]
        order = np.lexsort((cols[2], cols[1], cols[0]))
        self._year, self._type, self._ref, self._groups = (c[order] for c in cols)
        self._new = []

    def _summarize(self, done_gen: int) -> None:
        """Sum the routine events of generations ``emitted_gen`` .. ``done_gen`` - 1."""
        rows = np.concatenate(self._pending) if self._pending else np.zeros((0, 6), dtype=np.int64)
        over = rows[:, 0] < done_gen
        now, self._pending = rows[over], ([rows[~over]] if (~over).any() else [])
        G = self.cfg.generation_years
        if len(now):
            # Per (generation, lineage, action, kind, detail): events and distinct first actors.
            order = np.lexsort(now.T[::-1])
            now = now[order]
            group_change = np.r_[True, (now[1:, :5] != now[:-1, :5]).any(axis=1)]
            actor_change = group_change | np.r_[True, now[1:, 5] != now[:-1, 5]]
            starts = np.flatnonzero(group_change)
            counts = np.diff(np.r_[starts, len(now)])
            distinct = np.add.reduceat(actor_change.astype(np.int64), starts)
            keys = now[starts, :5]
            by_entry: dict[tuple[int, int], list] = {}
            for (gen, g, act, kind, det), c, d in zip(keys.tolist(), counts.tolist(), distinct.tolist()):
                by_entry.setdefault((gen, g), []).append((act, kind, det, c, d))
            last, ref, groups = [], [], []
            for (gen, g), items in by_entry.items():  # by generation, then lineage
                items.sort(key=lambda it: (-it[3], it[0], it[1], it[2]))
                last.append((gen + 1) * G - 1)
                ref.append(len(self.summaries))
                groups.append((1 << g) if g >= 0 else 0)
                self.summaries.append({"year": gen * G, "last": (gen + 1) * G - 1, "lineage": g, "items": items})
            arr = lambda v: np.array(v, dtype=np.int64)  # noqa: E731
            self._new.append((arr(last), np.ones(len(last), dtype=np.int64), arr(ref), arr(groups)))
        self.emitted_gen = done_gen

    # --- telling -------------------------------------------------------------------------------------

    def _item(self, act: int, kind: int, det: int, count: int, distinct: int) -> str:
        s = self.data["summaries"]
        verb = ACTIONS[act]
        form = s.get(f"{verb}/{ENTITY_KINDS[kind]}") if kind >= 0 else None
        form = form or s.get(verb) or s["default"]
        if isinstance(form, list):  # [one, many], by the number the form shows
            n = distinct if "{actors}" in form[1] else count
            form = form[0] if n == 1 else form[1]
        phrase = self.sp.technique(det) if det >= 0 else ""
        return form.format(count=f"{count:,}", actors=f"{distinct:,}", phrase=phrase, verb=verb)

    def text(self, entry) -> str:
        """An entry (its position, or its dict from ``entry``) as one line of the narrator's
        language (cached)."""
        if not isinstance(entry, dict):
            entry = self.entry(int(entry))
        key = (0, entry["event"]) if entry["event"] >= 0 else (1, (entry["year"], entry["lineage"]))
        t = self._text.get(key)
        if t is None:
            if entry["event"] >= 0:
                t = self.sp.event(entry["event"])
                if entry["suffix"]:
                    t = t[:-1] + "".join(entry["suffix"]) + "."
            else:
                c = self.data["chronicle"]
                items = c["item_joiner"].join(self._item(*it) for it in entry["items"])
                g = entry["lineage"]
                form = c["summary"] if g >= 0 else c["summary_none"]
                t = form.format(year=entry["year"], last=entry["last"], items=items,
                                people=self.sp.people(g) if g >= 0 else "")
                t = t[:1].upper() + t[1:]
            self._text[key] = t
        return t

    def select(self, groups: set[int] | None = None, text: str = "") -> list[int]:
        """Positions of the entries (oldest first) involving any of ``groups`` (all if None; an
        entry of no lineage always) and whose line contains ``text`` (case-insensitive; this
        tells every entry it looks at)."""
        if groups is None:
            idx = np.arange(len(self))
        else:
            want = sum(1 << g for g in groups)
            idx = np.flatnonzero(((self._groups & want) != 0) | (self._groups == 0))
        needle = text.strip().lower()
        if needle:
            idx = np.array([k for k in idx.tolist() if needle in self.text(k).lower()], dtype=np.int64)
        return idx.tolist()

    def latest(self, chosen: list[int], limit: int) -> list[str]:
        """The newest ``limit`` of the ``chosen`` entries, newest first, as lines."""
        return [self.text(k) for k in reversed(chosen[-limit:])] if limit > 0 else []

    def markdown(self) -> str:
        """The whole chronicle, oldest first, as markdown."""
        m = self.model
        head = ["# Chronicle", "", f"World seed {m.config.run.world_seed}, history seed {m.config.run.history_seed}, "
                f"to year {max(m.year - 1, 0):,}. {len(self):,} entries from {self.n:,} events.", ""]
        return "\n".join(head + [f"- {self.text(k)}" for k in range(len(self))]) + "\n"


def lines(model, limit: int, groups: set[int] | None = None) -> list[str]:
    """The newest ``limit`` chronicle entries as lines, newest first."""
    c = Chronicle(model)
    c.sync()
    return c.latest(c.select(groups), limit)


def log_csv(model) -> str:
    """The raw event log as CSV: every event, every column, read up to one snapshot length;
    tuples of actors and people space-separated."""
    import pandas as pd

    log = model.events
    n = min(len(log.link), len(log.people), len(log.polity_people), len(log.origin), len(log.year),
            len(log.action), len(log.actors), len(log.place), len(log.magnitude))
    join = lambda rows: [" ".join(f"{v:g}" for v in x) for x in rows]  # noqa: E731
    df = pd.DataFrame({
        "year": log.year[:n], "action": [ACTIONS[a] for a in log.action[:n]],
        "actors": join(log.actors[:n]), "place": log.place[:n], "magnitude": log.magnitude[:n],
        "link": log.link[:n], "people": join(log.people[:n]), "polity_people": join(log.polity_people[:n]),
        "origin": log.origin[:n]})
    return df.to_csv(index=False)


__all__ = ["Chronicle", "Namer", "entity", "line", "lines", "log_csv"]
