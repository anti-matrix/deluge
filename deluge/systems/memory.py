"""Phase 9: memory records, lossy transmission, symbols and their consequences (spec section 11).

The module supplies a substrate, generic copying errors and generic consequences. It
never looks at what an event was about: every verb, entity kind and place is handled alike,
and nothing here names a kind of event, actor or outcome (a test checks the vocabulary).

Records (``Memory``) are built from the raw event log. Each year the events logged since
the last memory phase are read and offered to the store of every living polity that took
part (as a polity actor, or through one of its settlements). One rule sets a record's
starting vividness for every verb alike (no per-verb weights): the share of the holder's
people the event affected (all of them when the polity is an actor itself, else the people
its settlement actors had when the event was logged, over the polity's people), log-scaled:
``vividness_scale`` x log(1 + share / ``share_reference``) / log(1 + 1 / share_reference),
plus ``novelty_bonus`` if the verb has never been recorded in that store and again if some
actor from outside the holder's polity (``_outside``) never has. Records at least ``record_min_vividness`` are written. Each record keeps
the event's actors, verb, place, year and magnitude, the appearance and dialect signature
of every actor at the time (``identity.actor_signatures``) and its provenance: the indices
of the original events in the log. ``vividness_max`` caps vividness as a safety bound.

Holders: the settlements whose people carry a record. A new record is held by the
settlement actors of its polity (else by the capital); daughters inherit their parent's,
a band joining another hands them on, and each generation records spread to members in
contact (``spread_rate``) and are lost (``holding_loss``). A conquered settlement keeps the
records it holds; when its old store ends they live on with it alone.

Symbols are integer handles. An entity, a place (cell) or an event gets one when it first
enters memory with vividness at least ``symbol_min_vividness``; a record points to the
symbols of its actors, place and events. A symbol's weight in a store is the summed
vividness of the store's records attached to it.

Transmission, every ``generation_years`` years, per store: decay; detail loss (the year to
a broader span relative to the record's age, ``year_span_fractions`` of the years since its
original events, centred on their mean year; the place to its region and then to none; the
magnitude to one significant figure); merge of two records sharing an actor, the place or a
symbol whose year spans overlap or touch, unless the union of their provenance would exceed
``max_provenance`` original events (provenance the union; the year span the one covering
both; the verb, the place and the actors, up to ``max_actors``, by a vividness-weighted vote
over every record fused in so far, each record's tally weighing its vividness; symbols the
more vivid one's, then the other's, up to ``max_symbols``); actor
substitution by an entity of the store drawn by its total vividness; the
floor and the capacity (``capacity`` x the capital's ``memory_capacity`` multipliers);
borrowing between polities in contact and from subject settlements to their rulers, with
detail loss applied to the copy. Amplification diminishes with its count: the k-th new
record sharing the verb and a symbol with a held one adds ``repeat_gain`` x its vividness x
log((k + 1) / k) to it, and the k-th invocation (below) adds ``invocation_gain`` x the
record's starting vividness x log((k + 1) / k), so n of them add in all a multiple of
log(n + 1).
Splits copy the parent's store to the child (new ids, the child's members carry the copies
over); a union adds the smaller partner's store to the larger's.

Every transformation is a row of ``ops`` (``OPS``): copies, borrowings and unions when
they happen, the transmission operators and the summed amplification of each record on
transmission years.

Consequences, each weighted by ``consequences`` (all zero with ``control``):

- cohesion: + weight x S / (S + half) x (1 - C), S the vividness of the records held by at
  least ``shared_majority`` of the polity's people (``cohesion_term``);
- legitimacy: secession odds x (1 - weight x L), L = w / (w + half), w the weight of the
  symbols of the capital, the polity, its parent polities and its founding lineage
  (``legitimacy``);
- every vividness reference (the two halves, the summary's symbol-weight reference) is a
  multiple of the store's own median record vividness (``scale``), so it is scale-free;
- affinity: the overlap of two stores' heaviest symbols drives mergers (with a common
  enemy, ``politics``) and speeds a subject's assimilation (``subject_affinity``);
- invocation: before a battle, a monument or a union the polity's heaviest symbols among
  its records involving the other party are used: their records gain, less each time
  (``invoke``);
- brain input: ``SUMMARY`` (``summary``).

Derived values (symbol weights, the shared term, legitimacy, the summary, subject
overlaps) are recomputed every generation; a new polity starts from its parent's.
Everything random draws from the ``memory`` stream only, so with the weights at zero a run
matches one without memory.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from deluge.data.events import ACTIONS, PEOPLE_KINDS, decode, entity
from deluge.systems import identity

SUMMARY = ("heaviest_symbol", "outsider_share", "mean_age")
OPS = ("copied", "borrowed", "absorbed", "coarsened", "merged", "substituted", "repeated", "invoked")
FIELDS = ("year", "place", "magnitude")
PLACE_CELL, PLACE_REGION, PLACE_NONE = 0, 1, 2


@dataclass(slots=True)
class Memory:
    mem_id: int
    actors: tuple[int, ...]
    action: int
    place: int
    year: int
    magnitude: float
    vividness: float
    symbols: tuple[int, ...]
    provenance: tuple[int, ...]
    signatures: tuple = ()      # per actor: appearance means + dialect at the time, or None
    year_span: int = 1          # the year stands for [year, year + span)
    year_level: int = 0         # how many times the year has coarsened (year_span_fractions)
    place_level: int = PLACE_CELL
    magnitude_rounded: bool = False
    store: int = -1             # polity whose store holds it, -1 once that store has ended
    parent: int = -1            # the record it was copied from
    repeated: float = 0.0       # vividness gained since the last transmission
    invoked: float = 0.0
    holders: set = field(default_factory=set)
    base: float = 0.0           # starting vividness
    repeats: int = 0            # repetitions and invocations so far (amplification diminishes with them)
    invocations: int = 0
    # Vividness-weighted tallies of the constituent records' fields (merges vote with them):
    # verb -> weight, (place, place level) -> weight, actor -> weight.
    action_votes: dict = field(default_factory=dict)
    place_votes: dict = field(default_factory=dict)
    actor_votes: dict = field(default_factory=dict)

    def tally(self) -> None:
        """Start the tallies from the record's own fields, at its vividness."""
        self.action_votes = {self.action: self.vividness}
        self.place_votes = {(self.place, self.place_level): self.vividness}
        self.actor_votes = {a: self.vividness for a in self.actors}


class MemoryState:
    """Every live record, the stores, the holders, the symbols and the transformation log."""

    def __init__(self, model) -> None:
        cfg = model.config.memory
        if cfg.enabled and model.config.brains.memory_inputs != len(SUMMARY):
            raise ValueError(f"brains.memory_inputs must be {len(SUMMARY)} (memory SUMMARY)")
        self.cfg = cfg
        self.enabled = cfg.enabled
        w = cfg.consequences
        zero = cfg.control or not cfg.enabled
        self.w = {name: 0.0 if zero else float(getattr(w, name))
                  for name in ("cohesion_weight", "legitimacy_weight", "merger_weight", "assimilation_weight",
                               "invocation_gain", "brain_input_weight")}
        self.rng = model.streams["memory"]
        self.records: dict[int, Memory] = {}
        self.stores: dict[int, list[int]] = {}
        self.seen: dict[int, tuple[set, set]] = {}  # per store: verbs and actors ever recorded in it (novelty)
        self.held: dict[int, set[int]] = {}
        self.next_id = 0
        self.cursor = 0
        self.symbol_of: dict[tuple[str, int], int] = {}
        self.symbol_key: list[tuple[str, int]] = []
        self.ops: list[tuple] = []
        self.created = 0
        self.dropped = 0
        # Derived per polity, recomputed every generation.
        self.weights: dict[int, dict[int, float]] = {}
        self.heavy: dict[int, dict[int, float]] = {}
        self.shared: dict[int, float] = {}
        self.legit: dict[int, float] = {}
        self.scale: dict[int, float] = {}  # median vividness of each store's records
        self.summary_of: dict[int, np.ndarray] = {}
        self.subject_overlap: dict[int, float] = {}
        self.coverage: dict[int, dict[int, float]] = {}  # per polity: record -> share of its people holding it
        gx = -(-model.geom.width // cfg.region_cells)
        self.region_cols = gx

    # --- bookkeeping --------------------------------------------------------------------------

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def _hold(self, sid: int, mid: int) -> None:
        self.records[mid].holders.add(sid)
        self.held.setdefault(sid, set()).add(mid)

    def _delete(self, mid: int) -> None:
        rec = self.records.pop(mid)
        for s in rec.holders:
            h = self.held.get(s)
            if h is not None:
                h.discard(mid)
        self.dropped += 1

    def _remove_from_store(self, mid: int) -> None:
        rec = self.records[mid]
        if rec.store >= 0:
            self.stores[rec.store].remove(mid)
        self._delete(mid)

    def log(self, year: int, op: str, store: int, mem: int, other: int = -1, a: int = -1, b: int = -1,
            value: float = 0.0) -> None:
        self.ops.append((year, OPS.index(op), store, mem, other, a, b, value))

    def symbol(self, key: tuple[str, int], create: bool) -> int:
        s = self.symbol_of.get(key, -1)
        if s < 0 and create:
            s = len(self.symbol_key)
            self.symbol_of[key] = s
            self.symbol_key.append(key)
        return s

    def region(self, model, cell: int) -> int:
        x, y = model.geom.xy(cell)
        r = self.cfg.region_cells
        return (int(y) // r) * self.region_cols + int(x) // r


# --- capacity ----------------------------------------------------------------------------------


def capacity(model, pids) -> np.ndarray:
    """Store capacity of each polity: ``capacity`` x its capital's memory_capacity multipliers."""
    pids = np.asarray(pids, dtype=np.int64)
    knows = model.settlements.knows[model.polities.capital[pids]]
    k = model.config.memory.capacity * model.knowledge.multiplier(knows, "memory_capacity")
    return np.floor(k + 1e-9).astype(np.int64)


def _weakest(state: MemoryState, store: list[int]) -> int:
    """The store's record with the least (vividness, id)."""
    records = state.records
    vivid = [records[m].vividness for m in store]
    low = min(vivid)
    return min(m for m, v in zip(store, vivid) if v == low)


def _trim(state: MemoryState, pid: int, cap: int) -> None:
    store = state.stores[pid]
    while len(store) > cap:
        state._remove_from_store(_weakest(state, store))


def _insert(model, state: MemoryState, pid: int, rec: Memory, cap: int) -> bool:
    """Put ``rec`` in ``pid``'s store; when full, the weakest record (``rec`` included) drops."""
    store = state.stores.setdefault(pid, [])
    rec.store = pid
    state.records[rec.mem_id] = rec
    store.append(rec.mem_id)
    if len(store) > cap:
        weakest = _weakest(state, store)
        state._remove_from_store(weakest)
        return weakest != rec.mem_id
    return True


# --- recording ---------------------------------------------------------------------------------


def salience(cfg, share: float) -> float:
    """Starting vividness from the share of the holder's people affected (log-scaled)."""
    share = min(max(share, 0.0), 1.0)
    return cfg.vividness_scale * np.log1p(share / cfg.share_reference) / np.log1p(1.0 / cfg.share_reference)


def _outside(model, pid: int, eid: int) -> bool:
    """Whether actor ``eid`` is a group of people from outside polity ``pid``: another polity,
    a settlement of another polity, another founding lineage, or any other group of people."""
    kind, local = decode(eid)
    if kind == "polity":
        return local != pid
    if kind == "settlement":
        return int(model.settlements.polity[local]) != pid
    if kind == "lineage":
        return local != int(model.polities.lineage[pid])
    return kind in PEOPLE_KINDS


def _record_events(model) -> None:
    state: MemoryState = model.memory
    cfg = state.cfg
    log = model.events
    t, P = model.settlements, model.polities
    n = len(log)
    start, state.cursor = state.cursor, n
    if start == n:
        return
    vmax = cfg.vividness_max
    pids = P.alive_ids()
    caps = dict(zip(pids.tolist(), capacity(model, pids).tolist()))  # polities that end this phase keep theirs
    signature_of: dict[int, tuple] = {}  # actors' signatures this year
    ids = t.alive_ids()
    people = np.bincount(t.polity[ids], weights=t.population[ids], minlength=P.n)
    for i in range(start, n):
        code = log.action[i]
        actors = log.actors[i]
        counts = log.people[i]
        pols: list[int] = []
        members: dict[int, list[int]] = {}
        affected: dict[int, float] = {}
        whole: set[int] = set()
        for a, k in zip(actors, counts):
            kind, local = decode(a)
            if kind == "polity":
                p = local
                whole.add(p)
            elif kind == "settlement":
                p = int(t.polity[local])
                if p >= 0:
                    if t.ended[local] < 0:
                        members.setdefault(p, []).append(local)
                    if np.isfinite(k):
                        affected[p] = affected.get(p, 0.0) + k
            else:
                continue
            if p >= 0 and P.ended[p] < 0 and p not in pols:
                pols.append(p)
        if not pols:
            continue
        place = log.place[i]
        keys = [("entity", a) for a in actors] + ([("cell", place)] if place >= 0 else []) + [("event", i)]
        signatures = None
        for p in pols:
            total = float(people[p]) if p < len(people) else 0.0
            share = 1.0 if p in whole else (affected.get(p, 0.0) / total if total > 0 else 1.0)
            verbs, seen = state.seen.setdefault(p, (set(), set()))
            novel = (code not in verbs) + any(a not in seen and _outside(model, p, a) for a in actors)
            v = min(salience(cfg, share) + cfg.novelty_bonus * novel, vmax)
            if v < cfg.record_min_vividness:
                continue
            strong = v >= cfg.symbol_min_vividness
            symbols = tuple(s for s in (state.symbol(k, strong) for k in keys) if s >= 0)
            if signatures is None:
                signatures = identity.actor_signatures(model, actors, signature_of)
            # Amplification by repetition, diminishing: held records sharing the verb and a symbol.
            if symbols:
                sym = set(symbols)
                for mid in state.stores.get(p, ()):
                    r = state.records[mid]
                    if r.action == code and sym.intersection(r.symbols):
                        r.repeats += 1
                        gain = min(r.vividness + cfg.repeat_gain * v * np.log1p(1.0 / r.repeats), vmax) - r.vividness
                        r.vividness += gain
                        r.repeated += gain
            rec = Memory(state._new_id(), tuple(actors), code, place, log.year[i], log.magnitude[i], v, symbols,
                         (i,), signatures, base=v)
            state.created += 1
            verbs.add(code)
            seen.update(actors)
            if _insert(model, state, p, rec, caps[p]):
                for s in members.get(p) or [int(P.capital[p])]:
                    state._hold(s, rec.mem_id)


# --- transmission operators ---------------------------------------------------------------------


def _coarsen(model, state: MemoryState, rec: Memory, year: int) -> None:
    """Detail loss: one field that still has detail coarsens."""
    cfg = state.cfg
    open_fields = []
    if rec.year_level < len(cfg.year_span_fractions):
        open_fields.append(0)
    if rec.place_level < PLACE_NONE:
        open_fields.append(1)
    if not rec.magnitude_rounded:
        open_fields.append(2)
    if not open_fields:
        return
    f = open_fields[int(state.rng.integers(len(open_fields)))]
    if f == 0:
        # Relative to the record's age: a share of the years since its original events, centred on them.
        center = float(np.mean([model.events.year[i] for i in rec.provenance]))
        span = max(rec.year_span + 1, int(round(cfg.year_span_fractions[rec.year_level] * (year - center))))
        rec.year_level += 1
        rec.year_span = span
        rec.year = max(int(round(center - span / 2)), 0)
        new = span
    elif f == 1:
        if rec.place_level == PLACE_CELL and rec.place >= 0:
            rec.place = state.region(model, rec.place)
            rec.place_level = PLACE_REGION
        else:
            rec.place = -1
            rec.place_level = PLACE_NONE
        if rec.place_votes:  # the tally now speaks of the coarsened place only
            rec.place_votes = {(rec.place, rec.place_level): sum(rec.place_votes.values())}
        new = rec.place_level
    else:
        rec.magnitude = float(f"{rec.magnitude:.1g}")
        rec.magnitude_rounded = True
        new = 1
    state.log(year, "coarsened", rec.store, rec.mem_id, a=f, b=new, value=rec.magnitude if f == 2 else rec.year)


def _shares_detail(a: Memory, b: Memory) -> bool:
    return (bool(set(a.actors).intersection(b.actors))
            or (a.place >= 0 and a.place == b.place and a.place_level == b.place_level)
            or bool(set(a.symbols).intersection(b.symbols)))


def _vote(tallies: list[tuple[dict, float]]) -> dict:
    """Sum the tallies, each rescaled so it weighs its record's vividness."""
    out: dict = {}
    for votes, w in tallies:
        total = sum(votes.values())
        for k, v in votes.items():
            out[k] = out.get(k, 0.0) + (v * w / total if total > 0 else 0.0)
    return out


def _ranked(votes: dict) -> list:
    """Keys by weight, heaviest first (ties: the lower key)."""
    return sorted(votes, key=lambda k: (-votes[k], k))


def _spans_meet(a: Memory, b: Memory) -> bool:
    """The year spans [year, year + span) overlap or touch."""
    return a.year <= b.year + b.year_span and b.year <= a.year + a.year_span


def _mergeable(state: MemoryState, a: Memory, b: Memory) -> bool:
    return (_shares_detail(a, b) and _spans_meet(a, b)
            and len(set(a.provenance) | set(b.provenance)) <= state.cfg.max_provenance)


def _merge(state: MemoryState, pid: int, mid: int, year: int) -> None:
    rec = state.records[mid]
    store = state.stores[pid]
    records = state.records
    # ``_mergeable``, its cheapest test first.
    lo, hi = rec.year, rec.year + rec.year_span
    actors, symbols, provenance = set(rec.actors), set(rec.symbols), set(rec.provenance)
    place = rec.place if rec.place >= 0 else None
    limit = state.cfg.max_provenance
    others = []
    for m in store:
        b = records[m]
        if m == mid or not (lo <= b.year + b.year_span and b.year <= hi):
            continue
        if not (actors.intersection(b.actors) or (place is not None and b.place == place
                                                  and b.place_level == rec.place_level)
                or symbols.intersection(b.symbols)):
            continue
        if len(provenance.union(b.provenance)) <= limit:
            others.append(m)
    if not others:
        return
    w = np.array([state.records[m].vividness for m in others])
    other = others[int(np.searchsorted(np.cumsum(w), state.rng.random() * w.sum(), side="right").clip(0, len(w) - 1))]
    _fuse(state, pid, mid, other, year)


def _fuse(state: MemoryState, pid: int, mid: int, other: int, year: int) -> None:
    """Fuse records ``mid`` and ``other`` of ``pid``'s store into the more vivid one."""
    a, b = state.records[mid], state.records[other]
    if (b.vividness, -b.mem_id) > (a.vividness, -a.mem_id):
        a, b = b, a
    cfg = state.cfg
    # Verb, place and actors: a vividness-weighted vote over every record fused in so far.
    for r in (a, b):
        if not r.action_votes:
            r.tally()
    wa, wb = a.vividness, b.vividness
    action_votes = _vote([(a.action_votes, wa), (b.action_votes, wb)])
    place_votes = _vote([(a.place_votes, wa), (b.place_votes, wb)])
    actor_votes = _vote([(a.actor_votes, wa), (b.actor_votes, wb)])
    sig_of = dict(zip(b.actors, b.signatures))
    sig_of.update(zip(a.actors, a.signatures))
    a.action = _ranked(action_votes)[0]
    a.place, a.place_level = _ranked(place_votes)[0]
    actors = [x for x in _ranked(actor_votes) if x in sig_of][: cfg.max_actors]
    a.actors, a.signatures = tuple(actors), tuple(sig_of[x] for x in actors)
    a.action_votes, a.place_votes, a.actor_votes = action_votes, place_votes, actor_votes
    # Year: the span covering both (spans are relative to each record's age, never fixed bins).
    start = min(a.year, b.year)
    end = max(a.year + a.year_span, b.year + b.year_span)
    a.year, a.year_span, a.year_level = start, end - start, max(a.year_level, b.year_level)
    a.vividness = min(a.vividness + state.cfg.merge_carry * b.vividness, state.cfg.vividness_max)
    a.symbols = tuple(dict.fromkeys(a.symbols + b.symbols))[: max(cfg.max_symbols, len(a.symbols))]
    a.provenance = tuple(sorted(set(a.provenance) | set(b.provenance)))
    a.repeated += b.repeated
    a.invoked += b.invoked
    a.repeats += b.repeats
    a.invocations += b.invocations
    a.base = max(a.base, b.base)
    for s in list(b.holders):
        state._hold(s, a.mem_id)
    state.log(year, "merged", pid, a.mem_id, other=b.mem_id)
    state._remove_from_store(b.mem_id)
    state.dropped -= 1  # fused, not lost


def _substitute(state: MemoryState, pid: int, mid: int, year: int) -> None:
    rec = state.records[mid]
    if not rec.actors:
        return
    total: dict[int, float] = {}
    where: dict[int, tuple] = {}
    for m in state.stores[pid]:
        r = state.records[m]
        for x, sg in zip(r.actors, r.signatures):
            total[x] = total.get(x, 0.0) + r.vividness
            if x not in where:
                where[x] = sg
    pool = [x for x in total if x not in rec.actors]
    if not pool:
        return
    w = np.array([total[x] for x in pool])
    new = pool[int(np.searchsorted(np.cumsum(w), state.rng.random() * w.sum(), side="right").clip(0, len(w) - 1))]
    k = int(state.rng.integers(len(rec.actors)))
    old = rec.actors[k]
    rec.actors = rec.actors[:k] + (new,) + rec.actors[k + 1:]
    rec.signatures = rec.signatures[:k] + (where[new],) + rec.signatures[k + 1:]
    if rec.actor_votes:  # the substituted actor takes over the old one's votes
        rec.actor_votes[new] = rec.actor_votes.get(new, 0.0) + rec.actor_votes.pop(old, 0.0)
    s_old, s_new = state.symbol(("entity", old), False), state.symbol(("entity", new), False)
    syms = [s for s in rec.symbols if s != s_old]
    if s_new >= 0 and s_new not in syms:
        syms.append(s_new)
    rec.symbols = tuple(syms)
    state.log(year, "substituted", pid, mid, a=old, b=new)


def _new_copy(model, state: MemoryState, rec: Memory, pid: int, factor: float, op: str) -> Memory:
    """A copy of ``rec`` meant for ``pid``'s store, logged, not yet inserted."""
    new = Memory(state._new_id(), rec.actors, rec.action, rec.place, rec.year, rec.magnitude,
                 rec.vividness * factor, rec.symbols, rec.provenance, rec.signatures, rec.year_span,
                 rec.year_level, rec.place_level, rec.magnitude_rounded, parent=rec.mem_id,
                 base=rec.base * factor, repeats=rec.repeats, invocations=rec.invocations,
                 action_votes=dict(rec.action_votes), place_votes=dict(rec.place_votes),
                 actor_votes=dict(rec.actor_votes))
    verbs, seen = state.seen.setdefault(pid, (set(), set()))
    verbs.add(rec.action)
    seen.update(rec.actors)
    state.log(model.year, op, pid, new.mem_id, other=rec.mem_id, a=rec.store)
    return new


def _copy(model, state: MemoryState, rec: Memory, pid: int, factor: float, op: str, cap: int,
          holders) -> int:
    """A copy of ``rec`` into ``pid``'s store (-1 if it did not fit)."""
    new = _new_copy(model, state, rec, pid, factor, op)
    if not _insert(model, state, pid, new, cap):
        return -1
    for s in holders:
        state._hold(s, new.mem_id)
    return new.mem_id


def _pick(state: MemoryState, mids: list[int]) -> int:
    w = np.array([state.records[m].vividness for m in mids])
    return mids[int(np.searchsorted(np.cumsum(w), state.rng.random() * w.sum(), side="right").clip(0, len(w) - 1))]


def _borrow(model) -> None:
    from deluge.systems import politics
    from deluge.systems.cohesion import trait_index

    state: MemoryState = model.memory
    cfg = state.cfg
    t, P = model.settlements, model.polities
    year = model.year
    pairs = politics.contact_pairs(model)
    openness = P.culture[:, trait_index(model, "openness")]
    for a, b in pairs.tolist():
        for giver, taker in ((a, b), (b, a)):
            if state.rng.random() >= cfg.borrow_contact_rate * openness[taker]:
                continue
            mids = state.stores.get(giver, [])
            if not mids or P.ended[taker] >= 0:
                continue
            src = state.records[_pick(state, mids)]
            new = _copy(model, state, src, taker, cfg.borrow_vividness, "borrowed",
                        int(capacity(model, [taker])[0]), [int(P.capital[taker])])
            if new >= 0 and state.rng.random() < cfg.detail_loss_chance:
                _coarsen(model, state, state.records[new], year)
    ids = t.alive_ids()
    subjects = ids[(t.subject_of[ids] >= 0) & (t.subject_of[ids] == t.polity[ids])]
    for sid in subjects.tolist():
        ruler = int(t.polity[sid])
        foreign = sorted(m for m in state.held.get(sid, ()) if state.records[m].store != ruler)
        if not foreign or state.rng.random() >= cfg.borrow_conquest_rate:
            continue
        src = state.records[_pick(state, foreign)]
        new = _copy(model, state, src, ruler, cfg.borrow_vividness, "borrowed", int(capacity(model, [ruler])[0]),
                    [sid])
        if new >= 0 and state.rng.random() < cfg.detail_loss_chance:
            _coarsen(model, state, state.records[new], year)


def _spread(model) -> None:
    """Records spread to members in contact with a holder and are lost by holders."""
    from deluge.systems.cohesion import settlement_contacts

    state: MemoryState = model.memory
    cfg = state.cfg
    t = model.settlements
    state.coverage = {}
    ids = t.alive_ids()
    pol = t.polity[ids]
    order = np.argsort(pol, kind="stable")
    ids, pol = ids[order], pol[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pol)) + 1, len(ids)]
    contacts_all = settlement_contacts(model, ids)
    index = np.full(t.n + 1, -1, dtype=np.int64)
    for j in range(len(starts) - 1):
        lo, hi = starts[j], starts[j + 1]
        pid = int(pol[lo])
        mids = state.stores.get(pid)
        if not mids:
            continue
        members = ids[lo:hi]
        n = len(members)
        index[members] = np.arange(n)
        H = np.zeros((n + 1, len(mids)), dtype=bool)
        for c, m in enumerate(mids):
            hs = state.records[m].holders
            for s in hs:
                i = index[s] if s < t.n else -1
                if i >= 0 and t.polity[s] == pid and t.ended[s] < 0:
                    H[i, c] = True
        nb = contacts_all[lo:hi]
        nb_idx = np.where(nb >= 0, index[np.maximum(nb, 0)], -1)
        nb_idx = np.where(nb_idx >= 0, nb_idx, n)  # the padding row holds nothing
        near = H[nb_idx].any(axis=1)
        own = H[:n]
        gain = ~own & near & (state.rng.random((n, len(mids))) < cfg.spread_rate)
        lose = own & (state.rng.random((n, len(mids))) < cfg.holding_loss)
        share = t.population[members] @ ((own | gain) & ~lose) / max(float(model.polities.population[pid]), 1.0)
        state.coverage[pid] = dict(zip(mids, share.tolist()))
        for i, c in zip(*np.nonzero(gain)):
            state._hold(int(members[i]), mids[c])
        for i, c in zip(*np.nonzero(lose)):
            s, m = int(members[i]), mids[c]
            state.records[m].holders.discard(s)
            state.held[s].discard(m)
        index[members] = -1
    # Records whose store has ended live on with their holders only.
    for mid in [m for m, r in state.records.items() if r.store < 0]:
        rec = state.records[mid]
        for s in sorted(rec.holders):
            if state.rng.random() < cfg.holding_loss:
                rec.holders.discard(s)
                state.held[s].discard(mid)
        if not rec.holders:
            state._delete(mid)


def transmit(model) -> None:
    state: MemoryState = model.memory
    cfg = state.cfg
    P = model.polities
    year = model.year
    pids = P.alive_ids()
    caps = dict(zip(pids.tolist(), capacity(model, pids).tolist()))
    for pid in pids.tolist():
        store = state.stores.get(pid)
        if not store:
            continue
        for mid in store:
            rec = state.records[mid]
            rec.vividness *= 1.0 - cfg.decay
            if rec.repeated > 0:
                state.log(year, "repeated", pid, mid, value=rec.repeated)
            if rec.invoked > 0:
                state.log(year, "invoked", pid, mid, value=rec.invoked)
            rec.repeated = rec.invoked = 0.0
        for mid in list(store):
            if mid not in state.records:
                continue  # fused this generation
            rec = state.records[mid]
            if state.rng.random() < cfg.detail_loss_chance:
                _coarsen(model, state, rec, year)
            if state.rng.random() < cfg.substitution_chance:
                _substitute(state, pid, mid, year)
            if state.rng.random() < cfg.merge_chance:
                _merge(state, pid, mid, year)
        if len(store) >= caps[pid]:
            for mid in [m for m in store if state.records[m].vividness < cfg.floor]:
                state._remove_from_store(mid)
        _trim(state, pid, caps[pid])
    _borrow(model)
    _spread(model)
    derive(model)


# --- splits, unions, settlements ----------------------------------------------------------------


def copy_store(model, parent: int, child: int, members) -> None:
    """A split: the child's store is a copy of the parent's; its members carry the copies."""
    state: MemoryState = model.memory
    if not state.enabled:
        return
    cap = int(capacity(model, [child])[0])
    mine = set(int(s) for s in members)
    store = state.stores.setdefault(child, [])
    verbs, seen = state.seen.get(parent, (set(), set()))
    state.seen[child] = (set(verbs), set(seen))
    copies = []
    for mid in list(state.stores.get(parent, ())):
        rec = state.records[mid]
        moving = sorted(rec.holders & mine)
        copies.append((_new_copy(model, state, rec, child, 1.0, "copied"), moving))
        for s in moving:
            rec.holders.discard(s)
            state.held[s].discard(mid)
    # Inserting them one by one, each time dropping the weakest when over capacity, keeps
    # the strongest ``cap`` of the store and the copies by (vividness, id): keep those at once.
    pool = store + [new.mem_id for new, _ in copies]
    if len(pool) > cap:
        vivid = {m: state.records[m].vividness for m in store}
        vivid.update((new.mem_id, new.vividness) for new, _ in copies)
        keep = set(sorted(pool, key=lambda m: (vivid[m], m))[len(pool) - cap:])
    else:
        keep = None
    for m in list(store):
        if keep is not None and m not in keep:
            state._remove_from_store(m)
    for new, moving in copies:
        if keep is not None and new.mem_id not in keep:
            state.dropped += 1
            continue
        new.store = child
        state.records[new.mem_id] = new
        store.append(new.mem_id)
        for s in moving:
            state._hold(s, new.mem_id)
    for d in (state.weights, state.heavy, state.shared, state.legit, state.summary_of, state.scale):
        if parent in d:
            d[child] = d[parent].copy() if hasattr(d[parent], "copy") else d[parent]


def absorb(model, big: int, small: int) -> None:
    """A union: the smaller partner's records join the larger's store (then the capacity)."""
    state: MemoryState = model.memory
    if not state.enabled:
        return
    store = state.stores.setdefault(big, [])
    verbs, seen = state.seen.setdefault(big, (set(), set()))
    v2, s2 = state.seen.pop(small, (set(), set()))
    verbs.update(v2)
    seen.update(s2)
    for mid in state.stores.pop(small, []):
        state.records[mid].store = big
        store.append(mid)
        state.log(model.year, "absorbed", big, mid, a=small)
    _trim(state, big, int(capacity(model, [big])[0]))


def polity_ended(model, pid: int) -> None:
    """A store ends with its polity; records some living settlement holds live on with it."""
    state: MemoryState = model.memory
    if not state.enabled:
        return
    t = model.settlements
    state.seen.pop(pid, None)
    for mid in state.stores.pop(pid, []):
        rec = state.records[mid]
        rec.holders = {s for s in rec.holders if t.ended[s] < 0}
        rec.store = -1
        if not rec.holders:
            state._delete(mid)
    for d in (state.weights, state.heavy, state.shared, state.legit, state.summary_of, state.scale):
        d.pop(pid, None)


def inherit(model, child: int, parent: int) -> None:
    """A daughter settlement carries its parent's records."""
    state: MemoryState = model.memory
    if not state.enabled:
        return
    for mid in sorted(state.held.get(parent, ())):
        state._hold(child, mid)


def settlement_ended(model, sid: int, host: int = -1) -> None:
    """A settlement ends; a band joining ``host`` hands its records on."""
    state: MemoryState = model.memory
    if not state.enabled:
        return
    for mid in sorted(state.held.pop(sid, ())):
        rec = state.records[mid]
        rec.holders.discard(sid)
        if host >= 0:
            state._hold(host, mid)
        if rec.store < 0 and not rec.holders:
            state._delete(mid)


# --- derived values and consequences ------------------------------------------------------------


def _lineage(model, eid: int) -> int:
    kind, local = decode(eid)
    if kind == "settlement":
        return int(model.settlements.lineage[local])
    if kind == "polity":
        return int(model.polities.lineage[local])
    if kind == "lineage":
        return local
    return -1


def _overlap(a: dict[int, float], b: dict[int, float]) -> float:
    """Weighted Jaccard: sum of min over sum of max (= sum a + sum b - sum of min)."""
    if not a or not b:
        return 0.0
    common = a.keys() & b.keys()
    if not common:
        return 0.0
    lo = sum(min(a[k], b[k]) for k in common)
    hi = sum(a.values()) + sum(b.values()) - lo
    return lo / hi if hi > 0 else 0.0


def _heaviest(weights: dict[int, float], k: int) -> dict[int, float]:
    """The ``k`` heaviest symbols, heaviest first (ties: the lower symbol)."""
    return dict(heapq.nsmallest(k, weights.items(), key=lambda kv: (-kv[1], kv[0])))


def _saturate(x: float, half: float) -> float:
    """x / (x + half), 0 when both are 0."""
    return x / (x + half) if x + half > 0 else 0.0


def derive(model) -> None:
    """Symbol weights, heaviest symbols, shared term, legitimacy, summary, subject overlaps."""
    state: MemoryState = model.memory
    cfg = state.cfg
    t, P = model.settlements, model.polities
    year = model.year
    state.weights, state.heavy, state.shared, state.legit, state.summary_of = {}, {}, {}, {}, {}
    state.scale = {}
    not_people = -2
    people_lineage: dict[int, int] = {}  # actor -> its founding lineage (-1: none) if a group of people, else -2
    records = state.records
    majority = cfg.shared_majority

    for pid in P.alive_ids().tolist():
        mids = state.stores.get(pid, [])
        weights: dict[int, float] = {}
        total_v = shared = outsider = age = 0.0
        coverage = state.coverage.get(pid, {})
        own_lineage = int(P.lineage[pid])
        for m in mids:
            r = records[m]
            v = r.vividness
            total_v += v
            for s in r.symbols:
                weights[s] = weights.get(s, 0.0) + v
            if coverage.get(m, 0.0) >= majority:
                shared += v
            for x in r.actors:
                lin = people_lineage.get(x)
                if lin is None:
                    lin = _lineage(model, x) if decode(x)[0] in PEOPLE_KINDS else not_people
                    people_lineage[x] = lin
                if lin != not_people and lin != own_lineage:
                    outsider += v
                    break
            age += v * (year - r.year)
        heavy = _heaviest(weights, cfg.heavy_symbols)
        # The store's own scale: the median vividness of its records (references are in its units).
        scale = float(np.median([state.records[m].vividness for m in mids])) if mids else 0.0
        state.weights[pid], state.heavy[pid], state.shared[pid] = weights, heavy, shared
        state.scale[pid] = scale
        keys = [("entity", entity("settlement", int(P.capital[pid]))), ("entity", entity("lineage", own_lineage))]
        keys += [("entity", entity("polity", p)) for p in model.lineages.polity_ancestry(P, pid)]
        wl = sum(weights.get(state.symbol(k, False), 0.0) for k in keys if state.symbol(k, False) >= 0)
        state.legit[pid] = _saturate(wl, cfg.consequences.legitimacy_half_medians * scale)
        top = max(weights.values()) if weights else 0.0
        top_ref = cfg.summary_weight_medians * scale
        state.summary_of[pid] = np.array([
            np.tanh(top / top_ref) if top_ref > 0 else 0.0,
            outsider / total_v if total_v > 0 else 0.0,
            np.tanh(age / total_v / cfg.summary_age_reference) if total_v > 0 else 0.0,
        ])
    state.subject_overlap = {}
    if state.w["assimilation_weight"] > 0:
        ids = t.alive_ids()
        subjects = ids[(t.subject_of[ids] >= 0) & (t.subject_of[ids] == t.polity[ids])]
        done: dict[tuple, float] = {}  # (records held, ruler) -> overlap
        for sid in subjects.tolist():
            ruler = int(t.polity[sid])
            key = (frozenset(state.held.get(sid, ())), ruler)
            if key not in done:
                mine: dict[int, float] = {}
                for m in key[0]:
                    r = records[m]
                    for s in r.symbols:
                        mine[s] = mine.get(s, 0.0) + r.vividness
                mine = _heaviest(mine, cfg.heavy_symbols)
                done[key] = _overlap(mine, state.heavy.get(ruler, {}))
            state.subject_overlap[sid] = done[key]


def cohesion_term(model, pids: np.ndarray) -> np.ndarray:
    """The shared-memory cohesion term's S / (S + half) x weight, per polity (zeros when off)."""
    state: MemoryState = model.memory
    w = state.w["cohesion_weight"]
    if w == 0.0:
        return np.zeros(len(pids))
    k = state.cfg.consequences.cohesion_half_medians
    return w * np.array([_saturate(state.shared.get(p, 0.0), k * state.scale.get(p, 0.0))
                         for p in np.asarray(pids).tolist()])


def legitimacy(model, pol: np.ndarray) -> np.ndarray:
    """weight x L of each entry's polity (zeros when off)."""
    state: MemoryState = model.memory
    w = state.w["legitimacy_weight"]
    if w == 0.0:
        return np.zeros(len(pol))
    vals = np.zeros(model.polities.n)
    for p, v in state.legit.items():
        vals[p] = v
    return w * vals[np.asarray(pol, dtype=np.int64)]


def affinity(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min(1, weight x overlap of the heaviest symbols) of polity pairs (zeros when off)."""
    state: MemoryState = model.memory
    w = state.w["merger_weight"]
    if w == 0.0:
        return np.zeros(len(a))
    return np.array([min(1.0, w * _overlap(state.heavy.get(p, {}), state.heavy.get(q, {})))
                     for p, q in zip(np.asarray(a).tolist(), np.asarray(b).tolist())])


def subject_affinity(model, sids: np.ndarray) -> np.ndarray:
    """weight x overlap of each subject's records with its ruler's store (zeros when off)."""
    state: MemoryState = model.memory
    w = state.w["assimilation_weight"]
    if w == 0.0:
        return np.zeros(len(sids))
    return w * np.array([state.subject_overlap.get(s, 0.0) for s in np.asarray(sids).tolist()])


def summary(model, pids: np.ndarray) -> np.ndarray:
    """(len(pids), SUMMARY) brain inputs x brain_input_weight (zeros when off)."""
    state: MemoryState = model.memory
    w = state.w["brain_input_weight"]
    out = np.zeros((len(pids), len(SUMMARY)))
    if w == 0.0:
        return out
    for i, p in enumerate(np.asarray(pids).tolist()):
        v = state.summary_of.get(p)
        if v is not None:
            out[i] = w * v
    return out


def invoke(model, pid: int, related) -> None:
    """Before an act involving the entities ``related``: the polity's heaviest symbols among
    its records involving them are used, and every record attached to them gains."""
    state: MemoryState = model.memory
    gain = state.w["invocation_gain"]
    if gain == 0.0:
        return
    mids = state.stores.get(pid)
    if not mids:
        return
    rel = set(related)
    weights = state.weights.get(pid, {})
    syms: set[int] = set()
    for m in mids:
        r = state.records[m]
        if rel.intersection(r.actors):
            syms.update(r.symbols)
    if not syms:
        return
    used = set(sorted(syms, key=lambda s: (-weights.get(s, 0.0), s))[: state.cfg.consequences.invocation_symbols])
    vmax = state.cfg.vividness_max
    for m in mids:
        r = state.records[m]
        if used.intersection(r.symbols):
            r.invocations += 1
            new = min(r.vividness + gain * r.base * np.log1p(1.0 / r.invocations), vmax)
            r.invoked += new - r.vividness
            r.vividness = new


# --- the phase ---------------------------------------------------------------------------------


def run(model) -> None:
    state: MemoryState = model.memory
    if not state.enabled:
        return
    _record_events(model)
    P = model.polities
    pids = P.alive_ids()
    for pid, cap in zip(pids.tolist(), capacity(model, pids).tolist()):
        if pid in state.stores:
            _trim(state, pid, cap)
    if model.year > 0 and model.year % state.cfg.generation_years == 0:
        transmit(model)


# --- reading (analysis, tests) ------------------------------------------------------------------


def records_frame(model):
    """Every live record, one row each (polity -1: its store has ended)."""
    import pandas as pd

    state: MemoryState = model.memory
    rows = [(r.mem_id, r.store, list(r.actors), ACTIONS[r.action], r.place, r.place_level, r.year, r.year_span,
             r.magnitude, r.vividness, list(r.symbols), list(r.provenance), r.parent, sorted(r.holders))
            for r in state.records.values()]
    return pd.DataFrame(rows, columns=["mem_id", "polity", "actors", "action", "place", "place_level", "year",
                                       "year_span", "magnitude", "vividness", "symbols", "provenance", "parent",
                                       "holders"])


def ops_frame(model):
    """The transformation log (``OPS``)."""
    import pandas as pd

    df = pd.DataFrame(model.memory.ops, columns=["year", "op", "polity", "mem_id", "other", "a", "b", "value"])
    df["op"] = pd.Categorical.from_codes(df["op"].to_numpy(dtype=np.int64), categories=list(OPS))
    return df
