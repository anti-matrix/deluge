"""Human-readable memory report (spec section 11, "Provenance and reading results").

    from analysis.myths import report
    text = report(model)                       # markdown, from a model at any year

For each founding lineage, then for each living polity: the heaviest symbols of its memory
store, the records attached to each, and for every record the original events it descends
from (years, actors, places, from the raw event log) and the transformations it went
through (``model.memory.ops``, its own and those of the records it was copied from or fused
with). Records keep the actors' appearance and dialect signatures; the report shows the
actors only.

The report reads the sim's raw fields and names entities the way the chronicle does
(``deluge/data/chronicle.py``); it adds no interpretation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from deluge.data.chronicle import Namer, line
from deluge.data.events import ACTIONS, decode
from deluge.systems.memory import FIELDS, OPS, PLACE_CELL, PLACE_REGION


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


def _place(model, rec) -> str:
    if rec.place < 0:
        return "place lost"
    if rec.place_level == PLACE_CELL:
        x, y = model.geom.xy(rec.place)
        return f"cell ({x}, {y})"
    if rec.place_level == PLACE_REGION:
        r = model.config.memory.region_cells
        cols = model.memory.region_cols
        ry, rx = divmod(rec.place, cols)
        return f"region x {rx * r}-{rx * r + r - 1}, y {ry * r}-{ry * r + r - 1}"
    return "place lost"


def _year(rec) -> str:
    return f"year {rec.year}" if rec.year_span == 1 else f"years {rec.year}-{rec.year + rec.year_span - 1}"


def symbol_label(model, name: Namer, s: int) -> str:
    kind, value = model.memory.symbol_key[s]
    if kind == "entity":
        return name(value)
    if kind == "cell":
        x, y = model.geom.xy(value)
        return f"cell ({x}, {y})"
    return f"event {value} ({line(model.events.record(value), name)})"


def _op_text(model, name: Namer, row) -> str:
    year, op, store, mem, other, a, b, value = row
    op = OPS[op]
    if op == "copied":
        return f"{year}: copied into polity {store}'s store from polity {a}'s at a split"
    if op == "borrowed":
        src = f"polity {a}'s store" if a >= 0 else "the records of a settlement whose store had ended"
        return f"{year}: borrowed into polity {store}'s store from {src}"
    if op == "absorbed":
        return f"{year}: joined polity {store}'s store from polity {a}'s in a union"
    if op == "coarsened":
        field = FIELDS[a]
        if field == "year":
            return f"{year}: year coarsened to a {b}-year span"
        if field == "place":
            return f"{year}: place coarsened to {'its region' if b == PLACE_REGION else 'none'}"
        return f"{year}: magnitude rounded to {value:,.0f}"
    if op == "merged":
        return f"{year}: fused with record #{other}"
    if op == "substituted":
        return f"{year}: actor {name(a)} replaced by {name(b)}"
    if op == "repeated":
        return f"{year}: vividness +{value:.2f} from similar events"
    return f"{year}: vividness +{value:.2f} from use"


def substituted_actors(model, rec) -> list[int]:
    """Actors of a record that took part in none of its original events."""
    original = {a for i in rec.provenance for a in model.events.actors[i]}
    return [a for a in rec.actors if a not in original]


def _record(model, name: Namer, hist: History, rec, max_events: int = 6, max_ops: int = 12) -> list[str]:
    actors = ", ".join(name(a) for a in rec.actors) or "no actors"
    out = [f"- **#{rec.mem_id}** `{ACTIONS[rec.action]}`: {actors}; {_place(model, rec)}; {_year(rec)}; "
           f"magnitude {rec.magnitude:,.0f}; vividness {rec.vividness:.2f}; held by {len(rec.holders)} settlements"]
    prov = list(rec.provenance)
    out.append(f"  - from {len(prov)} original event{'s' if len(prov) != 1 else ''}:")
    for i in prov[:max_events]:
        out.append(f"    - [{i}] {line(model.events.record(i), name)} ({_place_of_event(model, i)})")
    if len(prov) > max_events:
        out.append(f"    - ... and {len(prov) - max_events} more")
    subs = substituted_actors(model, rec)
    if subs:
        out.append("  - actors in none of its original events: " + ", ".join(name(a) for a in subs))
    ops = hist.transformations(rec.mem_id)
    if not ops:
        out.append("  - transformations: none")
        return out
    shown = [r for r in ops if OPS[r[1]] not in ("repeated", "invoked")]
    counts = defaultdict(int)
    gains = defaultdict(float)
    for r in ops:
        counts[OPS[r[1]]] += 1
        if OPS[r[1]] in ("repeated", "invoked"):
            gains[OPS[r[1]]] += r[7]
    out.append("  - transformations (this record and the records it came from): "
               + ", ".join(f"{k} {counts[k]}" for k in OPS if counts[k])
               + "".join(f"; vividness +{v:.1f} {'from similar events' if k == 'repeated' else 'from use'}"
                         for k, v in sorted(gains.items())))
    if shown:
        text = "; ".join(f"#{r[3]} {_op_text(model, name, r)}" for r in shown[:max_ops])
        more = f"; ... {len(shown) - max_ops} more" if len(shown) > max_ops else ""
        out.append(f"    - {text}{more}")
    return out


def _place_of_event(model, i: int) -> str:
    c = model.events.place[i]
    if c < 0:
        return "no place"
    x, y = model.geom.xy(c)
    return f"cell ({x}, {y})"


def _store_symbols(model, pid: int) -> list[tuple[int, float]]:
    w = model.memory.weights.get(pid)
    if w is None:  # not derived yet: compute from the store
        w = {}
        for m in model.memory.stores.get(pid, ()):
            r = model.memory.records[m]
            for s in r.symbols:
                w[s] = w.get(s, 0.0) + r.vividness
    return sorted(w.items(), key=lambda kv: (-kv[1], kv[0]))


def polity_section(model, pid: int, name: Namer, hist: History, n_symbols: int = 3, n_records: int = 1) -> list[str]:
    from deluge.systems.memory import capacity

    P, st = model.polities, model.memory
    mids = st.stores.get(pid, [])
    out = [f"### Polity {pid} (group {model.lineages.names[int(P.lineage[pid])]})", "",
           f"Founded year {int(P.founded[pid])}, parent polity {int(P.parent[pid])}; {int(P.n_settlements[pid])} "
           f"settlements, {P.population[pid]:,.0f} people; store {len(mids)} / {int(capacity(model, [pid])[0])} "
           f"records.", ""]
    shown: set[int] = set()
    for s, w in _store_symbols(model, pid)[:n_symbols]:
        out.append(f"**Symbol {s}** (weight {w:.2f}): {symbol_label(model, name, s)}")
        out.append("")
        recs = sorted((st.records[m] for m in mids if s in st.records[m].symbols), key=lambda r: (-r.vividness, r.mem_id))
        for rec in recs[:n_records]:
            if rec.mem_id in shown:
                out.append(f"- #{rec.mem_id} (above)")
                continue
            shown.add(rec.mem_id)
            out += _record(model, name, hist, rec)
        if len(recs) > n_records:
            out.append(f"- ... {len(recs) - n_records} more records attached")
        out.append("")
    if not mids:
        out += ["(empty store)", ""]
    return out


def lineage_section(model, g: int, name: Namer, hist: History, n_symbols: int = 8, n_records: int = 3) -> list[str]:
    P, st = model.polities, model.memory
    pids = [int(p) for p in P.alive_ids() if P.lineage[p] == g]
    out = [f"## Group {model.lineages.names[g]}", ""]
    if not pids:
        return out + ["No living polity.", ""]
    total: dict[int, float] = defaultdict(float)
    holders: dict[int, int] = defaultdict(int)
    for p in pids:
        for s, w in _store_symbols(model, p):
            total[s] += w
            holders[s] += 1
    n_rec = sum(len(st.stores.get(p, ())) for p in pids)
    out += [f"{len(pids)} living polities, {n_rec} records in their stores.", ""]
    for s, w in sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))[:n_symbols]:
        out.append(f"**Symbol {s}** (summed weight {w:.2f}, in {holders[s]} of {len(pids)} stores): "
                   f"{symbol_label(model, name, s)}")
        out.append("")
        recs = sorted((st.records[m] for p in pids for m in st.stores.get(p, ()) if s in st.records[m].symbols),
                      key=lambda r: (-r.vividness, r.mem_id))
        seen_prov: set[tuple] = set()
        k = 0
        for rec in recs:
            if rec.provenance in seen_prov:  # copies of one record in sibling stores: show one
                continue
            seen_prov.add(rec.provenance)
            out += [f"- in polity {rec.store}:"] + ["  " + x for x in _record(model, name, hist, rec)]
            k += 1
            if k >= n_records:
                break
        out.append("")
    return out


def report(model, polities: bool = True, title: str | None = None) -> str:
    st = model.memory
    name = Namer(model)
    hist = History(model)
    P = model.polities
    pids = P.alive_ids()
    out = [f"# {title or 'Memory report'}", "",
           f"World seed {model.config.run.world_seed}, history seed {model.config.run.history_seed}, year {model.year}. "
           f"{len(pids)} living polities; {len(st.records)} live records ({st.created} written from events in all); "
           f"{len(st.symbol_key)} symbols; {len(st.ops)} logged transformations; {len(model.events)} events logged.",
           "",
           "Symbols are integer handles for an entity, a cell or an event; a symbol's weight is the summed "
           "vividness of the records attached to it. Every record lists the original events it descends from "
           "(index in the event log) and the transformations it went through.", ""]
    for g in range(len(model.lineages.founders)):
        out += lineage_section(model, g, name, hist)
    if polities:
        out += ["## Living polities", ""]
        order = sorted(pids.tolist(), key=lambda p: (-float(P.population[p]), p))
        for p in order:
            out += polity_section(model, int(p), name, hist)
    return "\n".join(out) + "\n"


def surviving_events(model) -> dict[int, list[int]]:
    """Per founding group: the event indices its living stores still descend from."""
    st, P = model.memory, model.polities
    out = {}
    for g in range(len(model.lineages.founders)):
        ev = set()
        for p in P.alive_ids().tolist():
            if P.lineage[p] == g:
                for m in st.stores.get(p, ()):
                    ev.update(st.records[m].provenance)
        out[g] = sorted(ev)
    return out


def describe_signature(sig) -> str:
    if sig is None:
        return "-"
    return ", ".join("nan" if not np.isfinite(v) else f"{v:.2f}" for v in sig)
