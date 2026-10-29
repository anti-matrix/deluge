"""Phase 8b: secession, fragmentation, mergers, culture drift and territory (spec section 8).

- Secession: each member settlement but the capital secedes with probability
  ``secession_base`` x (hex steps to the capital / reference) ** exponent x (1 - cohesion)
  x (1 + ``secession_pressure_weight`` x pressure) x (1 + ``secession_grievance_weight`` x
  the settlement's grievance from conquest, ``systems/war.py``) x the marker factor, capped.
  Marker factor: 1 + the polity's secession marker parameters . the settlement's perceived
  appearance and dialect distance and culture distance from the core, clipped to [0,
  ``marker_factor_max``] (1 in the founders, whose parameters are zero). Seceding
  settlements of one polity that are in contact with each other (``contact_radius``,
  transitively) form one new polity.
- Fragmentation: a polity whose cohesion is below ``fragment_cohesion`` while its pressure
  is above ``fragment_pressure`` breaks, with chance ``fragment_rate`` a year, into its
  regions: regional seats are its largest settlements at least ``region_spacing`` steps
  apart (the capital first), every member joins its nearest seat, and each region becomes
  a new polity; the old one ends (``fragmented``).
- Every new polity keeps the parent's lineage, inherits its intermarriage rate and its
  members' own culture (the parent's, except in conquered members that kept theirs) with
  Gaussian mutation (``culture_mutation``), the parent's marker parameters with Gaussian
  mutation (``marker_mutation``, bounded by ``marker_effect_max``) and a mutated copy of
  the parent's brain (``model.found_polity``, ``brains/evolution.py``), starts at
  ``child_cohesion`` and the initial elite ratio, and records its parent, so the polity
  tree stays complete, and a copy of the parent's memory store (``memory.copy_store``).
  Secession odds are lowered by the polity's legitimacy from memory (``memory.legitimacy``).
- Mergers: two polities in contact merge with chance ``merge_rate`` x affinity x common
  threat; affinity is min(1, ``memory.consequences.merger_weight`` x the overlap of their
  heaviest memory symbols) and the threat 1 when both are at war with the same third polity
  (``model.wars``), else 0. With the memory consequences off no merger happens. ``merge``
  does the union: each partner first invokes its memories of the other, the larger partner
  absorbs the smaller, keeps its culture and brain (each a uniform crossover with the
  smaller one's with chance ``merge_crossover_chance``) and adds the smaller one's memory
  store to its own, and the smaller ends (``merged``).
- Culture drift: culture moves toward the mean culture of the polities in contact.
- Territory: the cells of the members and those within ``territory_radius`` of them; a
  deposit there gives every member access (spec section 5).
"""

from __future__ import annotations

import numpy as np

from deluge.agents.polity import marker_index
from deluge.brains import evolution
from deluge.data.events import entity
from deluge.systems import identity, memory
from deluge.systems.cohesion import culture_distance, settlement_contacts


def mutated(model, culture: np.ndarray, intermarriage: float, marker: np.ndarray
            ) -> tuple[np.ndarray, float, np.ndarray]:
    pc = model.config.polities
    sd = pc.culture_mutation
    rng = model.streams["politics"]
    c = np.clip(culture + rng.normal(0.0, sd, size=culture.shape), 0.0, 1.0)
    im = float(np.clip(intermarriage + rng.normal(0.0, sd), 0.0, 1.0))
    mk = np.clip(marker + rng.normal(0.0, pc.marker_mutation, size=marker.shape),
                 -pc.marker_effect_max, pc.marker_effect_max)
    return c, im, mk


def spin_off(model, parent: int, members: list[int]) -> int:
    """A new polity from ``members`` of ``parent`` (logged as founded from it). Its culture
    is the members' own (population-weighted), mutated: the parent's, except where
    conquered members kept their own. The members are no longer subjects."""
    P, t = model.polities, model.settlements
    members_arr = np.asarray(members, dtype=np.int64)
    capital = int(members_arr[np.argmax(t.population[members_arr])])
    w = np.maximum(t.population[members_arr], 1e-12)
    own = (w @ t.culture[members_arr]) / w.sum()
    culture, im, marker = mutated(model, own, float(P.intermarriage[parent]), P.marker[parent])
    pid = model.found_polity(int(P.lineage[parent]), parent, capital, culture, im,
                             model.config.polities.cohesion.child_cohesion, marker)
    t.polity[members_arr] = pid
    t.culture[members_arr] = culture
    t.subject_of[members_arr] = -1
    t.grievance[members_arr] = 0.0
    memory.copy_store(model, parent, pid, members_arr)
    model.log_event("founded", [("polity", pid), ("polity", parent)], int(t.cell[capital]),
                    float(t.population[members_arr].sum()))
    return pid


def _components(model, members: np.ndarray) -> list[list[int]]:
    """Groups of ``members`` connected through contact (union-find)."""
    if len(members) == 1:
        return [[int(members[0])]]
    index = {int(s): i for i, s in enumerate(members.tolist())}
    root = list(range(len(members)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    contacts = settlement_contacts(model, members)
    for i, row in enumerate(contacts.tolist()):
        for s in row:
            j = index.get(s)
            if j is not None:
                ri, rj = find(i), find(j)
                if ri != rj:
                    root[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i, s in enumerate(members.tolist()):
        groups.setdefault(find(i), []).append(int(s))
    return list(groups.values())


def secession(model) -> None:
    pc = model.config.polities
    P, t = model.polities, model.settlements
    ids = t.alive_ids()
    pol = t.polity[ids]
    capital = P.capital[pol]
    steps = model.geom.hex_distance(t.cell[ids], t.cell[capital]).astype(float)
    app, dia = identity.core_distances(model, ids)
    e = P.marker[pol][:, marker_index("secession_appearance", "secession_dialect", "secession_culture")]
    gap = culture_distance(t.culture[ids], P.culture[pol])
    factor = np.clip(1.0 + e[:, 0] * app + e[:, 1] * dia + e[:, 2] * gap, 0.0, pc.marker_factor_max)
    p = (pc.secession_base * (steps / pc.secession_distance_reference) ** pc.secession_distance_exponent
         * (1.0 - P.cohesion[pol]) * (1.0 + pc.secession_pressure_weight * P.pressure[pol])
         * (1.0 + pc.secession_grievance_weight * t.grievance[ids]) * factor
         * (1.0 - memory.legitimacy(model, pol)))
    p = np.where(ids == capital, 0.0, np.minimum(p, pc.secession_max))
    leaving = model.streams["politics"].random(len(ids)) < p
    if not leaving.any():
        return
    for parent in np.unique(pol[leaving]).tolist():
        for group in _components(model, ids[leaving & (pol == parent)]):
            spin_off(model, parent, group)
            model.stats["secessions"] += 1


def regions(model, pid: int) -> list[list[int]]:
    """Members of ``pid`` grouped around regional seats (see the module docstring)."""
    P, t = model.polities, model.settlements
    members = model.polity_agents[pid].members
    capital = int(P.capital[pid])
    order = members[np.argsort(-t.population[members], kind="stable")]
    seats = [capital]
    for s in order.tolist():
        if s != capital and (model.geom.hex_distance(t.cell[np.array(seats)], int(t.cell[s]))
                             >= model.config.polities.region_spacing).all():
            seats.append(int(s))
    seat_cells = t.cell[np.array(seats)]
    d = model.geom.hex_distance(t.cell[members][:, None], seat_cells[None, :])
    nearest = np.argmin(d, axis=1)
    return [members[nearest == i].tolist() for i in range(len(seats))]


def fragmentation(model) -> None:
    pc = model.config.polities
    P = model.polities
    pids = P.alive_ids()
    rng = model.streams["politics"]
    at_risk = pids[(P.cohesion[pids] < pc.fragment_cohesion) & (P.pressure[pids] > pc.fragment_pressure)
                   & (P.n_settlements[pids] >= 2)]
    for pid in at_risk.tolist():
        if rng.random() >= pc.fragment_rate:
            continue
        parts = regions(model, pid)
        if len(parts) < 2:
            continue
        for part in parts:
            spin_off(model, pid, part)
        model.end_polity(pid, "fragmented")
        model.stats["fragmentations"] += 1


# --- mergers ---------------------------------------------------------------------------------


def shared_memory_overlap(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Affinity of polity pairs from shared memory (``memory.affinity``: zero with the
    memory consequences off)."""
    return memory.affinity(model, a, b)


def common_enemy(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1 where both polities of a pair are at war (``model.wars``) with the same third polity."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    enemies: dict[int, set[int]] = {}
    for p, q in model.wars:
        enemies.setdefault(p, set()).add(q)
        enemies.setdefault(q, set()).add(p)
    # Two polities share an enemy r exactly when both are among r's enemies (r is neither:
    # no polity is its own enemy).
    shared = [(p, q) for r, es in enemies.items() if len(es) > 1 for p in es for q in es if p < q and r not in (p, q)]
    shared += [(p, p) for p, es in enemies.items() if es - {p}]  # a polity paired with itself
    if not shared or len(a) == 0:
        return np.zeros(len(a))
    n = np.int64(max(max(max(pq) for pq in shared), int(a.max()), int(b.max())) + 1)
    codes = np.unique(np.array([p * n + q for p, q in shared], dtype=np.int64))
    pairs = np.minimum(a, b) * n + np.maximum(a, b)
    at = np.minimum(np.searchsorted(codes, pairs), len(codes) - 1)
    return (codes[at] == pairs).astype(float)


def contact_pairs(model) -> np.ndarray:
    """(n, 2) distinct pairs of living polities with settlements in contact, a < b."""
    t = model.settlements
    ids = t.alive_ids()
    contacts = settlement_contacts(model, ids)
    i, k = np.nonzero(contacts >= 0)
    a, b = t.polity[ids[i]], t.polity[contacts[i, k]]
    n = np.int64(model.polities.n)
    code = np.unique(np.minimum(a, b) * n + np.maximum(a, b))  # sorted as (low, high) pairs
    pairs = np.column_stack([code // n, code % n])
    return pairs[pairs[:, 0] != pairs[:, 1]]


def merge(model, a: int, b: int) -> int:
    """Union of polities ``a`` and ``b``: the larger by population absorbs the smaller.
    Returns the surviving polity."""
    P, t = model.polities, model.settlements
    big, small = (a, b) if P.population[a] >= P.population[b] else (b, a)
    memory.invoke(model, big, (entity("polity", small),))
    memory.invoke(model, small, (entity("polity", big),))
    rng = model.streams["politics"]
    if rng.random() < model.config.polities.merge_crossover_chance:
        take = rng.random(P.culture.shape[1]) < 0.5
        P.culture[big] = np.where(take, P.culture[small], P.culture[big])
    if rng.random() < model.config.polities.merge_crossover_chance:
        B = model.brains
        B.replace(big, evolution.crossover(B.genome[big], B.genome[small], model.streams["brains"]), model.year)
    wb, ws = max(float(P.population[big]), 1.0), max(float(P.population[small]), 1.0)
    for col in (P.cohesion, P.elite_ratio):
        col[big] = (wb * col[big] + ws * col[small]) / (wb + ws)
    P.treasury[big] += P.treasury[small]
    P.monument_fund[big] += P.monument_fund[small]
    ids = t.alive_ids()
    members = ids[t.polity[ids] == small]
    t.polity[members] = big
    P.population[big] += P.population[small]
    P.n_settlements[big] += len(members)
    model.log_event("joined", [("polity", small), ("polity", big)], int(t.cell[P.capital[big]]),
                    float(t.population[members].sum()))
    memory.absorb(model, big, small)
    model.end_polity(small, "merged", merged_into=big)
    model.stats["mergers"] += 1
    return big


def mergers(model) -> None:
    from deluge.systems import alliances

    if model.memory.w["merger_weight"] == 0.0 and not model.alliances:  # affinity is zero: skip the contact pairs
        return
    pairs = contact_pairs(model)
    if len(pairs) == 0:
        return
    threat = common_enemy(model, pairs[:, 0], pairs[:, 1])
    affinity = np.zeros(len(pairs))
    hot = threat > 0  # affinity only matters where there is a common enemy
    # Shared memory, and (Round 14) the years two polities have been allied, capped at 1.
    affinity[hot] = np.minimum(1.0, shared_memory_overlap(model, pairs[hot, 0], pairs[hot, 1])
                               + alliances.affinity(model, pairs[hot, 0], pairs[hot, 1]))
    odds = model.config.polities.merge_rate * affinity * threat
    if not (odds > 0).any():
        return
    draw = model.streams["politics"].random(len(pairs)) < odds
    P = model.polities
    for a, b in pairs[draw].tolist():
        if P.ended[a] < 0 and P.ended[b] < 0:  # neither already merged this year
            merge(model, a, b)


# --- culture drift and territory ---------------------------------------------------------------


def drift_culture(model) -> None:
    rate = model.config.polities.culture_contact_drift
    P, t = model.polities, model.settlements
    ids = t.alive_ids()
    pol = t.polity[ids]
    contacts = settlement_contacts(model, ids)
    other = np.where(contacts >= 0, t.polity[np.maximum(contacts, 0)], -1)
    foreign = (other >= 0) & (other != pol[:, None])
    if not foreign.any():
        return
    i, k = np.nonzero(foreign)
    own, oth = pol[i], other[i, k]
    n = P.n
    weight = np.bincount(own, minlength=n).astype(float)
    target = np.zeros_like(P.culture)
    other_culture = P.culture[oth]
    for j in range(P.culture.shape[1]):
        target[:n, j] = np.bincount(own, weights=other_culture[:, j], minlength=n)
    touched = np.flatnonzero(weight > 0)
    target[touched] /= weight[touched][:, None]
    border_share = (np.bincount(pol, weights=foreign.any(axis=1).astype(float), minlength=n)
                    / np.maximum(np.bincount(pol, minlength=n), 1))
    step = (rate * border_share[touched])[:, None]
    P.culture[touched] += step * (target[touched] - P.culture[touched])


def update_territory(model) -> None:
    """Each polity's deposit per mineral: a deposit cell within ``territory_radius`` of a
    member settlement (-1 for none)."""
    P = model.polities
    P.deposit[: P.n] = -1
    occ = np.append(model.occupant, -1)
    pol_of = np.append(model.settlements.polity, -1)
    for m in range(model.deposits.shape[1]):
        cells = np.flatnonzero(model.deposits[:, m] > 0)
        if len(cells) == 0:
            continue
        ball = np.column_stack([cells, model.territory_cells[cells]])  # (D, k + 1)
        holder = pol_of[occ[ball]]  # -1 for no settlement (the padding reads the -1 row)
        d, j = np.nonzero(holder >= 0)
        if len(d) == 0:
            continue
        owners, first = np.unique(holder[d, j], return_index=True)
        P.deposit[owners, m] = cells[d[first]]


def run(model) -> None:
    secession(model)
    fragmentation(model)
    mergers(model)
    drift_culture(model)
    update_territory(model)
