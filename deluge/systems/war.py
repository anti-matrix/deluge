"""Phase 7: war and conquest (spec section 9).

Each year, first the conquered settlements assimilate (``assimilate``), then, with
``war.enabled``, the polities choose and fight their battles (``fight``).

Choice (the polity brains, Round 7). The candidate targets of a polity are the foreign
settlements within reach of one of its own (``war_contacts``): over land, a path of at most
``polities.contact_radius`` steps over land cells; from a settlement practising sailing
(holding a technique with ``sea_steps`` it can use where it is, on the coast), also a path
entering at most that many land cells and at most its sea reach in sea cells: the highest
sea_steps it practises (its sea tier: sailing, open-sea voyaging, ocean navigation), the same
tiers migration uses. Distance from the capital still weakens the attack. Polities separated
only by sea cannot fight otherwise. Where a target is within reach of several of the
polity's settlements, one over land attacks, else the one with the shortest crossing; an
attack across sea is logged as ``crossed`` (sea cells) after the battle's event. The polity's
per-neighbour head scores each candidate from its target features
(``brains/features.py``: estimated win chance, stores, fertility, culture distance,
perceived appearance and dialect distance, already at war, a capital). Each polity takes
its best-scoring target and attacks it with chance ``attack_rate`` x its aggression output
x score x min(food per head, 1) (its surplus capacity to fight); a campaign rather than a
raid with chance = its campaign output. A polity already at war with ``max_wars`` polities
only attacks those. The teachers are not chosen as a target: they hold no stores or land
(``TeachersState.attacked`` stays for a later round).

Strength S = P x mobilization x M_tech x (0.5 + cohesion) x D. Mobilization rises with
militarism and hierarchy; M_tech is the product of the capital's military multipliers
(bronze, horses, iron, ...) and the teachers' weapons while a member holds them; D =
exp(-km from the capital / (``logistics_km`` x the capital's raid_range multipliers:
horses, sailing, writing)). The attacker fights with its whole polity at D; the defender
with the target settlement's people in full and the rest of its polity at its own D, times
the terrain's ``terrain_defense``, the target's defense multiplier (sedentary building) and
``home_bonus``.

Resolution: the attacker wins with chance Sa^k / (Sa^k + Sd^k), k = ``win_steepness``. Each
side loses ``casualty_rate`` x the other's strength (more for the loser), capped; the
attacker's dead are spread over its settlements by population, the defender's come from the
target. Military techniques may pass between the two settlements that fought. A raid won
carries off part of the target's store. A campaign won takes the target and possibly its
neighbours of the same polity, with settlers from the attacking settlement; if the target
was the loser's capital, the loser loses cohesion and moves its capital, or, with too few
people left, is destroyed and the conqueror takes everything it had.

After conquest a settlement keeps its own culture and techniques, gets ``grievance`` and is
a subject (``subject_of``) of its conqueror. Each year its culture moves toward its ruler's
at ``assimilation_rate`` x the ruler's openness (x (1 + memory weight x shared-memory
overlap, ``memory.subject_affinity``)), its techniques pass to the ruler's capital at
``subject_diffusion_rate``, and its grievance decays; once grievance is low and its culture
close to its ruler's it is assimilated. Grievance raises secession and lowers cohesion
(``politics``, ``cohesion``); each battle counts as recent war in frontier pressure, and
as a win or a loss in the brain inputs. Members that are not subjects hold their polity's
culture. Every member's grievance also moves by ``polities.marker_grievance_rate`` x its
polity's grievance marker parameters . its perceived appearance and dialect distance from
the capital (zero in the founders; floored at 0).

Events (raw): ``fought`` (attacker polity, defender polity, target settlement; dead) for
battles with at least ``battle_log_min_losses`` dead; ``won`` (attacker, defender, target;
rations taken) for every raid won; ``took`` (conqueror, settlement, former polity; people)
for every settlement conquered; ``lost`` (polity, its capital, conqueror; people) for a
captured capital; ``died`` (polity, conqueror) for a destroyed polity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from deluge.agents.polity import marker_index
from deluge.brains.features import target_features
from deluge.brains.network import OUT, target_scores
from deluge.data.events import entity
from deluge.jit import njit
from deluge.systems import identity, knowledge, memory
from deluge.systems.cohesion import culture_distance, trait_index
from deluge.world.worlddata import terrain_table

WIN_EPS = 1e-12


def subject_memory_overlap(model, sids: np.ndarray) -> np.ndarray:
    """Weighted shared-memory overlap between subject settlements and their rulers
    (``memory.subject_affinity``: zero with the memory consequences off)."""
    return memory.subject_affinity(model, sids)


def win_chance(sa, sd, steepness: float):
    sa, sd = np.asarray(sa, float), np.asarray(sd, float)
    return np.where(sa > 0, 1.0 / (1.0 + (sd / np.maximum(sa, WIN_EPS)) ** steepness), 0.0)


# --- assimilation -----------------------------------------------------------------------------


def assimilate(model) -> None:
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    ids = t.alive_ids()
    if len(ids) == 0:
        return
    pol = t.polity[ids]
    subject = (t.subject_of[ids] >= 0) & (t.subject_of[ids] == pol)
    native = ids[~subject]
    t.culture[native] = P.culture[t.polity[native]]
    t.subject_of[native] = -1
    subjects = ids[subject]
    if len(subjects):
        ruler = t.polity[subjects]
        rate = (w.assimilation_rate * model.openness_of(subjects)
                * (1.0 + subject_memory_overlap(model, subjects)))
        t.culture[subjects] += rate[:, None] * (P.culture[ruler] - t.culture[subjects])
        # Techniques the conquered hold pass to the ruler's capital.
        capital = P.capital[ruler]
        offer = t.knows[subjects] & ~t.knows[capital] & g.acquirable[None, :]
        if offer.any():
            draw = model.streams["war"].random(offer.shape) < w.subject_diffusion_rate
            for i, j in zip(*np.nonzero(offer & draw)):
                knowledge.grant(model, int(capital[i]), int(j), int(subjects[i]))
    t.grievance[ids] *= 1.0 - w.grievance_decay
    app, dia = identity.core_distances(model, ids)
    e = P.marker[pol][:, marker_index("grievance_appearance", "grievance_dialect")]
    rate = model.config.polities.marker_grievance_rate
    t.grievance[ids] = np.maximum(t.grievance[ids] + rate * (e[:, 0] * app + e[:, 1] * dia), 0.0)
    gap = culture_distance(t.culture[ids], P.culture[pol])
    done = subject & (t.grievance[ids] < w.grievance_floor) & (gap < w.assimilated_distance)
    t.subject_of[ids[done]] = -1


# --- strength ---------------------------------------------------------------------------------


def _mobilization_and_tech(model, ids: np.ndarray, pol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per polity id: mobilization share and military multiplier (capital's techniques,
    teachers' weapons)."""
    w, cfg = model.config.war, model.config
    t, P, g = model.settlements, model.polities, model.knowledge
    n = P.n
    culture = P.culture[:n]
    mob = np.clip(w.mobilization_base + w.mobilization_militarism * culture[:, trait_index(model, "militarism")]
                  + w.mobilization_hierarchy * culture[:, trait_index(model, "hierarchy")], 0.0, w.mobilization_max)
    tech = g.multiplier(t.knows[P.capital[:n]], "military")
    armed = np.zeros(n, dtype=bool)
    armed[pol[model.teachers_state.weapons_holders(ids)]] = True
    return mob, np.where(armed, tech * cfg.teachers.weapons_military, tech)


def polity_strength(model) -> np.ndarray:
    """Per polity id: people x mobilization x M_tech x (0.5 + cohesion), before distance and
    terrain (the brains' strength-ratio inputs)."""
    t, P = model.settlements, model.polities
    ids = t.alive_ids()
    pol = t.polity[ids]
    people = np.bincount(pol, weights=t.population[ids], minlength=P.n)
    mob, tech = _mobilization_and_tech(model, ids, pol)
    return people * mob * tech * (0.5 + P.cohesion[: P.n])


def forces(model) -> SimpleNamespace:
    """Per-polity numbers (indexed by polity id) that strength is built from, this year."""
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    ids = t.alive_ids()
    pol = t.polity[ids]
    n = P.n
    people = np.bincount(pol, weights=t.population[ids], minlength=n)
    fed = np.bincount(pol, weights=t.population[ids] * t.food_ratio[ids], minlength=n) / np.maximum(people, 1.0)
    mob, tech = _mobilization_and_tech(model, ids, pol)
    capital = P.capital[:n]
    reach = w.logistics_km * g.multiplier(t.knows[capital], "raid_range")
    members: dict[int, list[int]] = {}
    for sid, p in zip(ids.tolist(), pol.tolist()):
        members.setdefault(p, []).append(sid)
    return SimpleNamespace(people=people, fed=fed, mob=mob, tech=tech, reach=reach,
                           capital_cell=t.cell[capital].copy(), members=members,
                           terrain=terrain_table(w.terrain_defense)[model.world.terrain.ravel()])


def strengths(model, F, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strength of attacker polities ``a`` and defender polities ``b`` at target settlements ``x``."""
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    cell = t.cell[x]
    km_a = model.geom.hex_distance(F.capital_cell[a], cell) * model.unit_km
    km_b = model.geom.hex_distance(F.capital_cell[b], cell) * model.unit_km
    n = len(F.people)
    force = F.people * F.mob * F.tech * (0.5 + P.cohesion[:n])  # per polity, the same products in the same order
    sa = force[a] * np.exp(-km_a / F.reach[a])
    local = t.population[x]
    rest = np.maximum(F.people[b] - local, 0.0) * np.exp(-km_b / F.reach[b])
    sd = ((local + rest) * F.mob[b] * F.tech[b] * (0.5 + P.cohesion[b]) * F.terrain[cell]
          * _multiplier_of(g, t, x, "defense") * w.home_bonus)
    return sa, sd


def _multiplier_of(g, t, x: np.ndarray, name: str) -> np.ndarray:
    """``g.multiplier(t.knows[x], name)``. With at most one technique carrying the effect, the
    product has one factor, so it is read per settlement from that technique's column (the
    same values: a sum of one nonzero term is exact whatever the matrix shape)."""
    logs = np.log(g.effect[name])
    terms = np.flatnonzero(logs != 0.0)
    if len(terms) > 1:
        return g.multiplier(t.knows[x], name)
    if len(terms) == 0:
        return np.ones(len(x))
    j = int(terms[0])
    return np.exp(np.array([0.0, logs[j]]))[t.knows[x, j].astype(np.intp)]


# --- reach over land and sea ---------------------------------------------------------------------

AXIAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
_BALLS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ball(radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Axial offsets within ``radius`` hex steps (the centre first) and, per offset, the
    index of each of its six neighbours in that list (len(offsets) where outside)."""
    if radius not in _BALLS:
        offs = [(0, 0)] + [(dq, dr) for dq in range(-radius, radius + 1) for dr in range(-radius, radius + 1)
                           if 0 < (abs(dq) + abs(dr) + abs(dq + dr)) // 2 <= radius]
        index = {o: j for j, o in enumerate(offs)}
        nb = np.array([[index.get((q + a, r + b), len(offs)) for a, b in AXIAL_STEPS] for q, r in offs])
        _BALLS[radius] = (np.array(offs, dtype=np.int64), nb)
    return _BALLS[radius]


def sea_reach(model, cells: np.ndarray, land_steps: int, sea_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From each of ``cells``: (targets, sea) over the hex ball of ``land_steps`` + max
    ``sea_steps`` steps; ``sea`` is the fewest sea cells crossed on a path entering at most
    ``land_steps`` land cells and at most that origin's ``sea_steps`` sea cells (a large
    number where none does), targets -1 off the map."""
    geom = model.geom
    budget = np.asarray(sea_steps, dtype=np.int64)
    radius = land_steps + (int(budget.max()) if len(budget) else 0)
    offs, nb = _ball(radius)
    q, r = geom.axial
    Q = q[cells][:, None] + offs[None, :, 0]
    R = r[cells][:, None] + offs[None, :, 1]
    X = Q + (R + (R & 1)) // 2
    valid = (R >= 0) & (R < geom.height) & (X >= 0) & (X < geom.width)
    targets = np.where(valid, X * geom.height + R, -1)
    wet = valid & model.is_sea[np.maximum(targets, 0)]
    dry = valid & ~wet
    big = np.int64(1 << 30)
    f = np.full((len(cells), len(offs) + 1), big, dtype=np.int64)  # the last column: outside the ball
    f[:, 0] = 0
    limit = budget[:, None]

    def over_sea():
        for _ in range(int(budget.max()) if len(budget) else 0):
            g = f[:, nb].min(axis=2) + 1
            better = wet & (g < f[:, :-1]) & (g <= limit)
            if not better.any():
                return
            f[:, :-1] = np.where(better, g, f[:, :-1])

    over_sea()
    for _ in range(land_steps):
        g = f[:, nb].min(axis=2)
        better = dry & (g < f[:, :-1])
        f[:, :-1] = np.where(better, g, f[:, :-1])
        over_sea()
    return targets, f[:, :-1]


@njit(cache=True)
def _relax_over_sea(neighbors, is_sea, cur, seeds, budget, reached, n_reached, buf_a, buf_b):
    """Lower ``cur`` over sea cells (+1 per sea cell entered, at most ``budget``) from
    ``seeds`` (cells whose value just fell), level by level; returns the new ``n_reached``."""
    big = 1 << 30
    # Seeds by value (a counting sort over 0..budget).
    count = np.zeros(budget + 2, np.int64)
    for s in seeds:
        count[cur[s] + 1] += 1
    for k in range(budget + 1):
        count[k + 1] += count[k]
    ordered = np.empty(len(seeds), np.int64)
    fill = count.copy()
    for s in seeds:
        ordered[fill[cur[s]]] = s
        fill[cur[s]] += 1
    i, ns = 0, len(ordered)
    now, n_now = buf_a, 0  # cells pushed at value k
    nxt = buf_b
    for k in range(budget + 1):
        n_next = 0
        for phase in range(2):
            if phase == 0:
                lo, hi = i, ns
            else:
                lo, hi = 0, n_now
            j = lo
            while j < hi:
                c = ordered[j] if phase == 0 else now[j]
                if phase == 0 and cur[c] != k:
                    break  # the seeds of value k are done
                j += 1
                if cur[c] != k or k + 1 > budget:
                    continue  # stale entry, or no sea step left
                for e in range(neighbors.shape[1]):
                    m = neighbors[c, e]
                    if m >= 0 and is_sea[m] and k + 1 < cur[m]:
                        if cur[m] == big:
                            reached[n_reached] = m
                            n_reached += 1
                        cur[m] = k + 1
                        nxt[n_next] = m
                        n_next += 1
            if phase == 0:
                i = j
        now, nxt = nxt, now
        n_now = n_next
        if i >= ns and n_now == 0:
            break
    return n_reached


@njit(cache=True)
def _reach_one(neighbors, is_sea, origin, land_steps, budget):
    """(targets, sea) from ``origin``: every cell a path entering at most ``land_steps`` land
    cells and at most ``budget`` sea cells reaches (the origin itself counts as neither), with
    the fewest sea cells crossed; targets ascending. Layer by layer: the cells reached with at
    most l land cells entered, closed over sea steps, then one more land step."""
    n = is_sea.shape[0]
    big = 1 << 30
    cur = np.full(n, big, np.int64)
    reached = np.empty(n, np.int64)
    buf_a = np.empty(n, np.int64)
    buf_b = np.empty(n, np.int64)
    cur[origin] = 0
    reached[0] = origin
    n_reached = _relax_over_sea(neighbors, is_sea, cur, np.array([origin], np.int64), budget, reached, 1,
                                buf_a, buf_b)
    changed = np.empty(n, np.int64)
    for _ in range(land_steps):
        old = cur.copy()
        n_changed = 0
        for r in range(n_reached):
            c = reached[r]
            v = old[c]
            for e in range(neighbors.shape[1]):
                m = neighbors[c, e]
                if m >= 0 and not is_sea[m] and v < cur[m]:
                    if cur[m] == big:
                        reached[n_reached] = m
                        n_reached += 1
                    if cur[m] == old[m]:
                        changed[n_changed] = m
                        n_changed += 1
                    cur[m] = v
        if n_changed == 0:
            break
        n_reached = _relax_over_sea(neighbors, is_sea, cur, changed[:n_changed].copy(), budget, reached,
                                    n_reached, buf_a, buf_b)
    targets = np.sort(reached[:n_reached])
    return targets, cur[targets]


def reach_lists(model, cells: np.ndarray, land_steps: int, sea_steps: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per origin of ``cells``: (targets, sea) of ``sea_reach`` kept to the cells within reach
    (on the map, at most that origin's ``sea_steps`` sea cells), sea cells included;
    ``_reach_one`` computes them. The sim itself reads only the land cells, cached
    (``land_reach``)."""
    neighbors, is_sea = model.geom.neighbors, model.is_sea
    return [_reach_one(neighbors, is_sea, int(c), land_steps, int(b))
            for c, b in zip(np.asarray(cells).tolist(), np.asarray(sea_steps).tolist())]


def land_reach(model, cells: np.ndarray, land_steps: int, sea_steps: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """``reach_lists`` kept to land cells: per origin (land targets as int32, sea cells
    crossed as int16), ascending. Cached on the model per (cell, land_steps, sea_steps); when
    the sea changes, only the entries whose hex ball (``land_steps`` + sea_steps) holds a cell
    that changed are dropped, so an entry's land cells are always this year's."""
    state = model.__dict__.setdefault("_reach_cache", {"sea": None, "cache": {}})
    cache = state["cache"]
    if state["sea"] is None or not np.array_equal(state["sea"], model.is_sea):
        if state["sea"] is not None and cache:
            flipped = np.flatnonzero(state["sea"] != model.is_sea)
            keys = list(cache)
            origin = np.array([k[0] for k in keys], dtype=np.int64)
            radius = np.array([k[1] + k[2] for k in keys], dtype=np.int64)
            near = np.zeros(len(keys), dtype=bool)
            for f in flipped.tolist():
                near |= model.geom.hex_distance(origin, f) <= radius
                if near.all():
                    break
            for k in np.flatnonzero(near).tolist():
                del cache[keys[k]]
        state["sea"] = model.is_sea.copy()
    cells = np.asarray(cells, dtype=np.int64)
    budgets = np.asarray(sea_steps, dtype=np.int64)
    out: list = [None] * len(cells)
    get = cache.get
    neighbors, is_sea = model.geom.neighbors, model.is_sea
    for k, (c, b) in enumerate(zip(cells.tolist(), budgets.tolist())):
        key = (c, land_steps, b)
        hit = get(key)
        if hit is None:
            targets, sea = _reach_one(neighbors, is_sea, c, land_steps, b)
            land = ~is_sea[targets]
            hit = (targets[land].astype(np.int32), sea[land].astype(np.int16))
            cache[key] = hit
        out[k] = hit
    return out


def sea_budget(model, ids: np.ndarray) -> np.ndarray:
    """Sea cells each settlement's attack can cross: the highest sea_steps (sea tier) of the
    techniques it holds and can use where it is (on the coast), the same tiers migration uses.
    Strength still decays with distance from the capital (``strengths``), so long crossings
    are possible but weak."""
    g, t = model.knowledge, model.settlements
    out = np.zeros(len(ids), dtype=np.int64)
    holds = (t.knows[ids] & (g.effect["sea_steps"] > 0)[None, :]).any(axis=1)
    if holds.any():
        from deluge.systems.knowledge import usable_mask

        h = ids[holds]
        out[holds] = np.rint(g.highest(t.knows[h] & usable_mask(model, h), "sea_steps")).astype(np.int64)
    return out


def _contact_lists(model, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lengths, land cells, sea cells crossed) within war reach of each settlement of ``ids``,
    concatenated in ``ids`` order. Settlements sit on land cells only, so the reach entries'
    land targets are all that can hold one."""
    t = model.settlements
    entries = land_reach(model, t.cell[ids], model.config.polities.contact_radius, sea_budget(model, ids))
    lens = np.fromiter((len(e[0]) for e in entries), dtype=np.int64, count=len(entries))
    return lens, np.concatenate([e[0] for e in entries]), np.concatenate([e[1] for e in entries])


@njit(cache=True)
def _war_candidates(lens, targets, sea, src, src_pol, perm, occupant, polity, n_settlements):
    """Foreign settlements within reach of each polity's settlements, one per (polity, target):
    the attacking settlement with the fewest sea cells to cross, then the lowest id. ``perm``:
    the rows ordered by polity (stable). Returns (polity, target, settlement, sea) by polity,
    then target."""
    starts = np.zeros(len(lens) + 1, np.int64)
    for r in range(len(lens)):
        starts[r + 1] = starts[r] + lens[r]
    big = 1 << 62
    best_sea = np.full(n_settlements, big, np.int64)
    best_src = np.full(n_settlements, big, np.int64)
    touched = np.empty(n_settlements, np.int64)
    out_a = np.empty(len(targets), np.int64)
    out_x = np.empty(len(targets), np.int64)
    out_s = np.empty(len(targets), np.int64)
    out_sea = np.empty(len(targets), np.int64)
    n_out = 0
    i = 0
    while i < len(perm):
        a = src_pol[perm[i]]
        n_touched = 0
        while i < len(perm) and src_pol[perm[i]] == a:
            r = perm[i]
            s = src[r]
            for j in range(starts[r], starts[r + 1]):
                x = occupant[targets[j]]
                if x < 0 or x == s or polity[x] == a:
                    continue
                d = sea[j]
                if best_sea[x] == big:
                    touched[n_touched] = x
                    n_touched += 1
                if d < best_sea[x] or (d == best_sea[x] and s < best_src[x]):
                    best_sea[x] = d
                    best_src[x] = s
            i += 1
        xs = np.sort(touched[:n_touched])
        for x in xs:
            out_a[n_out] = a
            out_x[n_out] = x
            out_s[n_out] = best_src[x]
            out_sea[n_out] = best_sea[x]
            n_out += 1
            best_sea[x] = big
            best_src[x] = big
    return out_a[:n_out], out_x[:n_out], out_s[:n_out], out_sea[:n_out]


def war_contacts(model, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows, targets, sea): each settlement of ``ids`` (by row) that another settlement is
    within reach of for war: over land within ``polities.contact_radius`` hex steps of land,
    or, from a settlement practising sailing, also across at most its ``sea_budget`` sea cells;
    ``sea`` is the fewest sea cells crossed (0 over land)."""
    empty = np.zeros(0, dtype=np.int64)
    if len(ids) == 0:
        return empty, empty, empty
    lens, targets, sea = _contact_lists(model, ids)
    rows = np.repeat(np.arange(len(ids), dtype=np.int64), lens)
    occ = model.occupant[targets]
    keep = (occ >= 0) & (occ != ids[rows])
    rows, xs, seas = rows[keep], occ[keep], sea[keep]
    if len(rows) == 0:
        return empty, empty, empty
    order = np.lexsort((xs, rows))
    return rows[order], xs[order], seas[order]


# --- choice -----------------------------------------------------------------------------------


def at_war(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pair of polity ids: they are at war (``model.wars``, keys low id first)."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if not model.wars or len(a) == 0:
        return np.zeros(len(a), dtype=bool)
    keys = np.array(list(model.wars), dtype=np.int64).reshape(-1, 2)
    n = np.int64(max(int(keys.max()), int(a.max()), int(b.max())) + 1)
    wars = np.sort(keys[:, 0] * n + keys[:, 1])
    pairs = np.minimum(a, b) * n + np.maximum(a, b)
    at = np.minimum(np.searchsorted(wars, pairs), len(wars) - 1)
    return wars[at] == pairs


def at_war_counts(model) -> np.ndarray:
    counts = np.zeros(model.polities.n, dtype=np.int64)
    for a, b in model.wars:
        counts[a] += 1
        counts[b] += 1
    return counts


def choose(model, F) -> list[tuple[int, int, int, int, bool, int]]:
    """This year's attacks: (attacker, defender, attacking settlement, target, campaign, sea
    cells crossed), in attacker id order."""
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    ids = t.alive_ids()
    if len(ids) == 0:
        return []
    # Foreign settlements in reach (``war_contacts``), one attacking settlement per (polity,
    # target): over land if one is, else the shortest crossing (then the lowest id).
    pol = t.polity[ids]
    lens, targets, sea = _contact_lists(model, ids)
    a, x, src, sea = _war_candidates(lens, targets, sea, ids, pol, np.argsort(pol, kind="stable"),
                                     model.occupant, t.polity[: t.n], t.n)
    if len(a) == 0:
        return []
    b = t.polity[x]
    busy = at_war_counts(model) >= w.max_wars
    if busy[a].any():
        allowed = ~busy[a]
        held = np.flatnonzero(~allowed)
        allowed[held] = at_war(model, a[held], b[held])
        src, x, a, b, sea = src[allowed], x[allowed], a[allowed], b[allowed], sea[allowed]
        if len(a) == 0:
            return []
    # Candidates come by attacker id. Each attacker draws whether it attacks (chance attack_rate
    # x aggression x its best target's score x food) and whether by campaign. A score is at most
    # 1, so an attacker whose draw is at least attack_rate x aggression x food cannot attack:
    # only the others' targets are scored (the same values: every step is per candidate).
    brains = model.brains
    first = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    attackers = a[first]
    out = brains.output[attackers]
    fed = np.clip(F.fed[attackers], 0.0, 1.0)
    attack_draw = rng.random(len(attackers))
    campaign = rng.random(len(attackers)) < out[:, OUT["campaign"]]
    rate = w.attack_rate * out[:, OUT["aggression"]]
    maybe = attack_draw < rate * fed
    if not maybe.any():
        return []
    sel = np.repeat(maybe, np.diff(np.r_[first, len(a)]))
    src, x, a, b, sea = src[sel], x[sel], a[sel], b[sel], sea[sel]
    sa, sd = strengths(model, F, a, b, x)
    p_win = win_chance(sa, sd, w.win_steepness)
    score = target_scores(target_features(model, a, b, x, p_win), brains.target_w[a])
    order = np.lexsort((-score, a))
    best = order[np.r_[True, a[order][1:] != a[order][:-1]]]  # top score per attacker, attacker id order
    k = np.flatnonzero(maybe)
    attack = attack_draw[k] < rate[k] * score[best] * fed[k]
    return [(int(a[j]), int(b[j]), int(src[j]), int(x[j]), bool(c), int(sea[j]))
            for j, go, c in zip(best.tolist(), attack.tolist(), campaign[k].tolist()) if go]


# --- resolution -------------------------------------------------------------------------------


def take(model, F, a: int, b: int, x: int, src: int) -> None:
    """Polity ``a`` takes settlement ``x`` from ``b``; settlers come from ``src`` (-1: none)."""
    w = model.config.war
    t = model.settlements
    people = float(t.population[x])
    t.polity[x] = a
    t.subject_of[x] = a
    t.grievance[x] = max(float(t.grievance[x]), w.grievance_conquest)
    if src >= 0 and t.ended[src] < 0:
        settlers = min(w.settler_share * people, w.settler_source_cap * float(t.population[src]))
        if settlers > 0:
            identity.blend(model, x, src, settlers)
            t.population[src] -= settlers
            t.population[x] += settlers
    F.members[b].remove(x)
    F.members.setdefault(a, []).append(x)
    F.people[b] -= people
    F.people[a] += people
    model.stats["conquests"] += 1
    model.log_event("took", [("polity", a), ("settlement", x), ("polity", b)], int(t.cell[x]), people)


def campaign_won(model, F, a: int, b: int, src: int, x: int) -> None:
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    capital = int(P.capital[b])
    before = max(float(F.people[b]), 0.0)
    take(model, F, a, b, x, src)
    near = np.append(model.occupant, -1)[model.contact_cells[t.cell[x]]]
    for y in near[near >= 0].tolist():
        if t.polity[y] == b and y != capital and rng.random() < w.neighbor_capture_chance:
            take(model, F, a, b, int(y), src)
    if x != capital:
        return
    model.stats["capitals_taken"] += 1
    model.log_event("lost", [("polity", b), ("settlement", x), ("polity", a)], int(t.cell[x]), float(t.population[x]))
    rest = F.members[b]
    left = max(float(F.people[b]), 0.0)
    if rest and left >= w.destroy_ratio * F.people[a]:
        new = max(rest, key=lambda s: (t.population[s], -s))
        P.capital[b] = new
        F.capital_cell[b] = t.cell[new]
        P.cohesion[b] *= 1.0 - w.capital_cohesion_loss
        return
    for y in list(rest):
        take(model, F, a, b, y, -1)
    model.stats["polities_destroyed"] += 1
    model.end_polity(b, "conquered", merged_into=a, people=before)


def exchange_military(model, s1: int, s2: int) -> None:
    """After a battle, each military technique one side's settlement holds may pass to the other's."""
    t, g = model.settlements, model.knowledge
    rng = model.streams["war"]
    for giver, taker in ((s1, s2), (s2, s1)):
        for j in np.flatnonzero(g.military & t.knows[giver] & ~t.knows[taker]).tolist():
            if rng.random() < model.config.war.enemy_military_diffusion:
                knowledge.grant(model, taker, j, giver)


def battle(model, F, a: int, b: int, src: int, x: int, campaign: bool, sea: int = 0) -> bool:
    """Resolve one raid or campaign, launched from ``src`` across ``sea`` sea cells;
    returns whether the attacker won."""
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    memory.invoke(model, a, (entity("polity", b), entity("settlement", x)))
    sa, sd = (float(v[0]) for v in strengths(model, F, np.array([a]), np.array([b]), np.array([x])))
    won = bool(rng.random() < win_chance(sa, sd, w.win_steepness))
    loss_a = min(w.casualty_rate * sd * (1.0 if won else w.loser_casualty_factor),
                 w.attacker_casualty_cap * F.people[a] * F.mob[a])
    loss_d = min(w.casualty_rate * sa * (w.loser_casualty_factor if won else 1.0),
                 w.defender_casualty_cap * float(t.population[x]))
    own = np.array(F.members[a], dtype=np.int64)
    pop = t.population[own]
    if pop.sum() > 0 and loss_a > 0:
        t.population[own] = pop - loss_a * pop / pop.sum()
    t.population[x] -= loss_d
    F.people[a] -= loss_a
    F.people[b] -= loss_d
    model.stats["battles"] += 1
    model.stats["campaigns" if campaign else "raids"] += 1
    model.stats["war_deaths"] += loss_a + loss_d
    P.war_recent[a] += 1.0
    P.war_recent[b] += 1.0
    P.wins_recent[a if won else b] += 1.0
    P.losses_recent[b if won else a] += 1.0
    model.wars[(min(a, b), max(a, b))] = model.year
    cell = int(t.cell[x])
    link = -1  # the battle's event: fought, else what the win took (won, took)
    if loss_a + loss_d >= w.battle_log_min_losses:
        link = model.log_event("fought", [("polity", a), ("polity", b), ("settlement", x)], cell, loss_a + loss_d)
    exchange_military(model, src, x)
    if won:
        before = len(model.events)
        if campaign:
            campaign_won(model, F, a, b, src, x)
        else:
            loot = w.raid_loot_share * float(t.store[x])
            t.store[x] -= loot
            t.store[src] += loot
            model.log_event("won", [("polity", a), ("polity", b), ("settlement", x)], cell, loot)
        if link < 0 and len(model.events) > before:
            link = before
    if sea > 0:
        model.stats["sea_battles"] += 1
        model.log_crossing(sea, [("polity", a), ("settlement", src), ("settlement", x), ("polity", b)], cell, link)
    return won


def fight(model) -> None:
    F = forces(model)
    t, P = model.settlements, model.polities
    for a, b, src, x, campaign, sea in choose(model, F):
        # Skip what this year's earlier battles changed: a polity gone, a target taken.
        if P.ended[a] >= 0 or P.ended[b] >= 0 or t.polity[x] != b or t.polity[src] != a:
            continue
        battle(model, F, a, b, src, x, campaign, sea)


def run(model) -> None:
    w = model.config.war
    assimilate(model)
    P = model.polities
    P.war_recent[: P.n] *= 1.0 - w.war_recent_decay
    keep = 1.0 - model.config.brains.features.war_record_decay
    P.wins_recent[: P.n] *= keep
    P.losses_recent[: P.n] *= keep
    if w.enabled:
        fight(model)
    year = model.year
    model.wars = {pair: y for pair, y in model.wars.items()
                  if year - y <= w.war_memory_years and P.ended[pair[0]] < 0 and P.ended[pair[1]] < 0}
    model.wars_active = len(model.wars)
