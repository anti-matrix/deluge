"""Where memory records came from: the transformation log indexed per record, and the
records that descend from the first years of the run (analysis only).

Shared by the memory report (``analysis/myths.py``) and the narrator
(``analysis/narrator.py``). Everything here reads the sim's raw fields.
"""

from __future__ import annotations

from collections import defaultdict

from deluge.agents.teachers import ARTIFACT_KINDS
from deluge.data.events import decode
from deluge.systems.memory import OPS
from deluge.world.worlddata import MINERALS


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


class History:
    """Index over the transformation log: the rows of each record and where it came from."""

    def __init__(self, model) -> None:
        self.rows = defaultdict(list)
        self.source: dict[int, int] = {}  # record -> the record it was copied or borrowed from
        self.fused = defaultdict(list)  # record -> records fused into it
        for row in model.memory.ops:
            year, op, store, mem, other, a, b, value = row
            self.rows[mem].append(row)
            name = OPS[op]
            if name in ("copied", "borrowed"):
                self.source[mem] = other
            elif name == "merged":
                self.fused[mem].append(other)

    def lineage_of(self, mem: int) -> list[int]:
        """The record, the records it descends from by copying, and those fused into any of them."""
        out, todo, seen = [], [mem], set()
        while todo:
            m = todo.pop()
            if m in seen:
                continue
            seen.add(m)
            out.append(m)
            if m in self.source:
                todo.append(self.source[m])
            todo.extend(self.fused.get(m, ()))
        return out

    def transformations(self, mem: int) -> list[tuple]:
        rows = [r for m in self.lineage_of(mem) for r in self.rows.get(m, ())]
        return sorted(rows, key=lambda r: (r[0], r[3], r[1]))


def substituted_actors(model, rec) -> list[int]:
    """Actors of a record that took part in none of its original events."""
    original = {a for i in rec.provenance for a in model.events.actors[i]}
    return [a for a in rec.actors if a not in original]


def substitutions(hist: History, mem: int) -> list[tuple[int, int, int]]:
    """The actor substitutions a record went through (its own and those of the records it
    came from), in order: (year, actor replaced, actor put in)."""
    return [(r[0], r[5], r[6]) for r in hist.transformations(mem) if OPS[r[1]] == "substituted"]


def origin_records(model, g: int, hist: History, until: int | None = None) -> list[dict]:
    """The surviving records of founding group ``g`` (in its living polities' stores) whose
    provenance includes an event of years 0 to ``until`` (``reports.origins_until_year``).
    Copies alike in every field and history (sibling stores after splits) are one entry:
    {"record", "stores", "copies", "substitutions", "current"}. ``current`` (a): the record's
    own fields still hold something of those years (``holds_early``); the others (b) descend
    from them by provenance only. (a) come first, then (b), each oldest first."""
    until = model.config.reports.origins_until_year if until is None else until
    st, P, ev = model.memory, model.polities, model.events
    groups: dict[tuple, dict] = {}
    for p in P.alive_ids().tolist():
        if P.lineage[p] != g:
            continue
        for m in st.stores.get(p, ()):
            r = st.records[m]
            if not any(ev.year[i] <= until for i in r.provenance):
                continue
            subs = tuple(substitutions(hist, m))
            key = (r.provenance, r.actors, r.action, r.year, r.year_span, r.place, r.place_level, subs)
            e = groups.get(key)
            if e is None:
                groups[key] = {"record": r, "stores": [p], "copies": 1, "substitutions": list(subs),
                               "current": holds_early(model, r, until)}
            else:
                e["copies"] += 1
                if p not in e["stores"]:
                    e["stores"].append(p)
                if r.mem_id < e["record"].mem_id:
                    e["record"] = r
    return sorted(groups.values(), key=lambda e: (not e["current"], min(ev.year[i] for i in e["record"].provenance),
                                                  e["record"].mem_id))


def holds_early(model, rec, until: int) -> bool:
    """Whether a record's current fields still hold something of years 0 to ``until``: its
    year span reaches into them, or it keeps the verb and an actor of one of its original
    events from those years."""
    ev = model.events
    if rec.year <= until:
        return True
    early = [i for i in rec.provenance if ev.year[i] <= until]
    return any(ev.action[i] == rec.action and set(ev.actors[i]) & set(rec.actors) for i in early)
