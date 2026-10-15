"""Phase 8b: secession, fragmentation, mergers, culture drift and territory (spec section 8).

- Secession: each member settlement but the capital secedes with probability
  ``secession_base`` x (hex steps to the capital / reference) ** exponent x (1 - cohesion)
  x (1 + ``secession_pressure_weight`` x pressure), capped. Seceding settlements of one
  polity that are in contact with each other (``contact_radius``, transitively) form one
  new polity. The weight the polity's culture gives to cultural, dialect and appearance
  distance from the core is zero until perception is wired (Round 7).
- Fragmentation: a polity whose cohesion is below ``fragment_cohesion`` while its pressure
  is above ``fragment_pressure`` breaks, with chance ``fragment_rate`` a year, into its
  regions: regional seats are its largest settlements at least ``region_spacing`` steps
  apart (the capital first), every member joins its nearest seat, and each region becomes
  a new polity; the old one ends (``fragmented``).
- Every new polity keeps the parent's lineage, inherits its culture and intermarriage rate
  with Gaussian mutation (``culture_mutation``), starts at ``child_cohesion`` and the
  initial elite ratio, and records its parent, so the polity tree stays complete. (Brains
  and memory copies join in Rounds 7 and 8.)
- Mergers: two polities in contact merge with chance ``merge_rate`` x affinity x common
  threat; affinity is their shared-memory overlap (Round 8) and the threat a common enemy
  (Round 6). Both are zero until then, so no merger happens yet; ``merge`` does the union:
  the larger partner absorbs the smaller, keeps its culture (uniform crossover with the
  smaller one's with chance ``merge_crossover_chance``), and the smaller ends (``merged``).
- Culture drift: culture moves toward the mean culture of the polities in contact.
- Territory: the cells of the members and those within ``territory_radius`` of them; a
  deposit there gives every member access (spec section 5).
"""

from __future__ import annotations

import numpy as np

from deluge.systems.cohesion import settlement_contacts


def mutated(model, culture: np.ndarray, intermarriage: float) -> tuple[np.ndarray, float]:
    sd = model.config.polities.culture_mutation
    rng = model.streams["politics"]
    c = np.clip(culture + rng.normal(0.0, sd, size=culture.shape), 0.0, 1.0)
    im = float(np.clip(intermarriage + rng.normal(0.0, sd), 0.0, 1.0))
    return c, im


def spin_off(model, parent: int, members: list[int]) -> int:
    """A new polity from ``members`` of ``parent`` (logged as founded from it)."""
    P, t = model.polities, model.settlements
    members_arr = np.asarray(members, dtype=np.int64)
    capital = int(members_arr[np.argmax(t.population[members_arr])])
    culture, im = mutated(model, P.culture[parent], float(P.intermarriage[parent]))
    pid = model.found_polity(int(P.lineage[parent]), parent, capital, culture, im,
                             model.config.polities.cohesion.child_cohesion)
    t.polity[members_arr] = pid
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
    p = (pc.secession_base * (steps / pc.secession_distance_reference) ** pc.secession_distance_exponent
         * (1.0 - P.cohesion[pol]) * (1.0 + pc.secession_pressure_weight * P.pressure[pol]))
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


MERGER_INPUTS_EXIST = False  # set by the rounds that add shared memory (8) and war (6)


def shared_memory_overlap(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Affinity of polity pairs from shared memory: zero until memory exists (Round 8)."""
    return np.zeros(len(a))


def common_enemy(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whether polity pairs face a common enemy: zero until war exists (Round 6)."""
    return np.zeros(len(a))


def contact_pairs(model) -> np.ndarray:
    """(n, 2) distinct pairs of living polities with settlements in contact, a < b."""
    t = model.settlements
    ids = t.alive_ids()
    contacts = settlement_contacts(model, ids)
    i, k = np.nonzero(contacts >= 0)
    a, b = t.polity[ids[i]], t.polity[contacts[i, k]]
    pairs = np.unique(np.column_stack([np.minimum(a, b), np.maximum(a, b)]), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


def merge(model, a: int, b: int) -> int:
    """Union of polities ``a`` and ``b``: the larger by population absorbs the smaller.
    Returns the surviving polity."""
    P, t = model.polities, model.settlements
    big, small = (a, b) if P.population[a] >= P.population[b] else (b, a)
    rng = model.streams["politics"]
    if rng.random() < model.config.polities.merge_crossover_chance:
        take = rng.random(P.culture.shape[1]) < 0.5
        P.culture[big] = np.where(take, P.culture[small], P.culture[big])
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
    model.end_polity(small, "merged", merged_into=big)
    model.stats["mergers"] += 1
    return big


def mergers(model) -> None:
    if not MERGER_INPUTS_EXIST:  # both inputs are zero: skip building the contact pairs
        return
    pairs = contact_pairs(model)
    if len(pairs) == 0:
        return
    affinity = shared_memory_overlap(model, pairs[:, 0], pairs[:, 1])
    threat = common_enemy(model, pairs[:, 0], pairs[:, 1])
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
