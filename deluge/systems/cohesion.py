"""Phase 8a: tribute, monuments, elites, structural-demographic pressure and cohesion
(spec section 8).

Per living polity, each year, in this order:

1. Membership: a polity with no settlements left ends (``extinct``); a polity whose
   capital ended or left takes its largest settlement as the new capital.
2. Tribute: each member gives ``tribute_max_rate`` x its polity brain's tribute output x
   its marker factor of its store to the treasury (at most all of it), which spoils at
   ``treasury_spoilage``. Marker factor: 1 + the polity's tribute marker parameters . the
   member's perceived appearance and dialect distance from the capital, clipped to [0,
   ``marker_factor_max``] (1 in the founders, whose parameters are zero).
3. Monuments: once the capital holds megalithic construction, ``monument_max_share`` x
   the brain's monument output of the treasury goes to a monument fund. One monument at a
   time is under construction (Round 14): it begins with the first labour put in, needs
   ``monument_build_years`` x that year's labour (at least ``monument_cost``), and is built at
   the capital (a ``built`` event) once that much is saved, never sooner than
   ``monument_build_years`` after it began. Every monument standing in
   the polity's settlements costs ``monument_upkeep`` x its cost a year; those the
   treasury can pay are maintained this year (oldest first), the rest are not. Maintained
   monuments raise cohesion and are counted as prestige.
4. Elites: the elite ratio grows logistically at ``elites.growth`` x min(tribute per
   head / tribute_reference, 1) and falls at ``elites.decline`` in years when the polity's
   people are short of food on average. Elite pressure = max(elite_ratio / positions - 1, 0)
   with positions = ``elites.positions`` x (0.5 + hierarchy).
5. Pressure: population pressure (people over the capacity of the members' cells: 0 at
   ``population_reference`` of capacity, 1 at capacity and above) plus elite pressure,
   minus state strength ((tribute - spending) per head), floored at 0.
6. Frontier pressure f: of the members in contact with another polity's settlements
   (within ``contact_radius``), the share facing a culturally distant one (culture
   distance ramping from ``distant_culture_min`` to ``distant_culture_full``), plus recent
   war: ``war_frontier_weight`` x w / (w + ``war_frontier_half``), w the polity's decaying
   count of battles (``systems/war.py``), capped at 1.
7. Cohesion: C' = C + r0 f C (1 - C) - delta C + monuments, with delta rising with size
   (scaled down by hierarchy and administration techniques), mean distance from the
   capital, inequality of food per head between members, elite pressure, pressure and the
   members' grievance from conquest (population-weighted mean), plus
   ``marker_cohesion_decay`` x the cohesion marker parameters . the members' mean perceived
   appearance and dialect distance from the capital (delta floored at 0).
   Then the shared-memory term: C + ``memory.cohesion_term`` x (1 - C) (spec section 11;
   zero with the memory consequences off).

The load (people over the capacity of the members' cells) is kept for the brains' headroom
input.
"""

from __future__ import annotations

import numpy as np

from deluge.agents.polity import marker_index
from deluge.brains.network import OUT
from deluge.systems import identity, memory
from deluge.systems.subsistence import fed_capacity, mode_capacity

TRAIT_DIST_EPS = 1e-12


def trait_index(model, name: str) -> int:
    return model.config.founders.culture_traits.index(name)


def culture_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between culture vectors over sqrt(number of traits): 0..1."""
    return np.linalg.norm(a - b, axis=-1) / np.sqrt(a.shape[-1])


def settlement_contacts(model, ids: np.ndarray) -> np.ndarray:
    """(S, k) ids of the settlements in contact with each of ``ids`` (within
    ``polities.contact_radius``), -1 padding."""
    t = model.settlements
    return np.append(model.occupant, -1)[model.contact_cells[t.cell[ids]]]


def housekeeping(model) -> None:
    P, t = model.polities, model.settlements
    ids = t.alive_ids()
    pids = P.alive_ids()
    n = np.bincount(t.polity[ids], minlength=P.n)
    for pid in pids[n[pids] == 0].tolist():
        model.end_polity(pid, "extinct")
    pids = P.alive_ids()
    cap = P.capital[pids]
    lost = (t.ended[cap] >= 0) | (t.polity[cap] != pids)
    if lost.any():
        pol, pop = t.polity[ids], t.population[ids]
        order = np.lexsort((-pop, pol))  # by polity, largest first
        first = {}
        for i in order.tolist():
            first.setdefault(int(pol[i]), int(ids[i]))
        for pid in pids[lost].tolist():
            P.capital[pid] = first[pid]


def run(model) -> None:
    housekeeping(model)
    pc, cc = model.config.polities, model.config.polities.cohesion
    P, t, g = model.polities, model.settlements, model.knowledge
    ids = t.alive_ids()
    pids = P.alive_ids()
    if len(pids) == 0:
        return
    pol = t.polity[ids]
    pop = t.population[ids]
    npol = P.n
    culture = P.culture
    hierarchy = culture[:, trait_index(model, "hierarchy")]
    brain = model.brains.output
    app, dia = identity.core_distances(model, ids)
    marker = P.marker[pol]

    n_set = np.bincount(pol, minlength=npol).astype(float)
    people = np.bincount(pol, weights=pop, minlength=npol)
    P.n_settlements[pids] = n_set[pids].astype(np.int64)
    P.population[pids] = people[pids]
    P.peak_population[pids] = np.maximum(P.peak_population[pids], people[pids])
    P.peak_settlements[pids] = np.maximum(P.peak_settlements[pids], P.n_settlements[pids])
    safe_people = np.maximum(people, 1.0)

    # Tribute.
    e = marker[:, marker_index("tribute_appearance", "tribute_dialect")]
    factor = np.clip(1.0 + e[:, 0] * app + e[:, 1] * dia, 0.0, pc.marker_factor_max)
    take = np.minimum(pc.tribute_max_rate * brain[pol, OUT["tribute"]] * factor, 1.0) * t.store[ids]
    t.store[ids] -= take
    tribute = np.bincount(pol, weights=take, minlength=npol)
    P.tribute[pids] = tribute[pids]
    P.treasury[pids] = P.treasury[pids] * (1.0 - pc.treasury_spoilage) + tribute[pids]

    # Monuments: fund and build.
    spending = np.zeros(npol)
    capital = P.capital[pids]
    can_build = t.knows[capital, g.index("megalithic_construction")]
    put = np.where(can_build, pc.monument_max_share * brain[pids, OUT["monuments"]] * P.treasury[pids], 0.0)
    P.treasury[pids] -= put
    P.monument_fund[pids] += put
    spending[pids] += put
    # At most one monument under construction per polity: it begins when labour is first put
    # in, needs monument_build_years x that year's labour (at least monument_cost), and is done
    # once that much is saved and not sooner than monument_build_years after it began.
    begin = (P.monument_started[pids] < 0) & (put > 0)
    P.monument_started[pids[begin]] = model.year
    P.monument_target[pids[begin]] = np.maximum(pc.monument_cost, pc.monument_build_years * put[begin])
    building = P.monument_started[pids] >= 0
    done = (building & (P.monument_fund[pids] >= P.monument_target[pids])
            & (model.year - P.monument_started[pids] >= pc.monument_build_years))
    for pid in pids[done].tolist():
        cost = float(P.monument_target[pid])
        P.monument_fund[pid] -= cost
        P.monument_started[pid] = -1
        P.monument_target[pid] = 0.0
        model.build_monument(pid, int(P.capital[pid]), cost)
    # Upkeep, oldest monuments first, as far as the treasury goes.
    mon = model.monuments
    maintained = np.zeros(npol)
    P.prestige[pids] = 0.0
    if len(mon):
        standing = t.ended[mon.settlement] < 0
        owner = np.where(standing, t.polity[mon.settlement], -1)
        upkeep = pc.monument_upkeep * mon.cost
        mon.maintained[:] = False
        idx = np.flatnonzero(owner >= 0)
        if len(idx):
            order = idx[np.lexsort((idx, owner[idx]))]  # by owner, oldest first
            cum = np.cumsum(upkeep[order])
            starts = np.r_[0, np.flatnonzero(np.diff(owner[order])) + 1]
            base = np.repeat(cum[starts] - upkeep[order][starts], np.diff(np.r_[starts, len(order)]))
            paid = (cum - base) <= P.treasury[owner[order]]
            mon.maintained[order[paid]] = True
            cost_paid = np.bincount(owner[order[paid]], weights=upkeep[order[paid]], minlength=npol)
            P.treasury[pids] -= cost_paid[pids]
            spending += cost_paid
            maintained = np.bincount(owner[order[paid]], minlength=npol).astype(float)
            P.prestige[pids] = np.bincount(owner[order[paid]], weights=mon.cost[order[paid]], minlength=npol)[pids]
    P.spending[pids] = spending[pids]

    # Elites.
    pe = model.config.polities.elites
    fed = np.bincount(pol, weights=pop * t.food_ratio[ids], minlength=npol) / safe_people
    e = P.elite_ratio[pids]
    good = np.clip(tribute[pids] / safe_people[pids] / pe.tribute_reference, 0.0, 1.0)
    e = e + pe.growth * good * e * (1.0 - e)
    e = np.where(fed[pids] < 1.0, e * (1.0 - pe.decline), e)
    P.elite_ratio[pids] = np.clip(e, 0.0, 1.0)
    positions = pe.positions * (0.5 + hierarchy[pids])
    elite_pressure = np.maximum(P.elite_ratio[pids] / positions - 1.0, 0.0)

    # Structural-demographic pressure.
    pp = model.config.polities.pressure
    k = fed_capacity(t.shares[ids], mode_capacity(model, t.knows[ids], t.cell[ids]), model.config.demography)
    capacity = np.bincount(pol, weights=k, minlength=npol)
    load = people[pids] / np.maximum(capacity[pids], 1.0)
    P.load[pids] = load
    pop_pressure = np.clip((load - pp.population_reference) / (1.0 - pp.population_reference), 0.0, 1.0)
    strength = np.clip((tribute[pids] - spending[pids]) / safe_people[pids] / pp.state_reference, -1.0, 1.0)
    pressure = np.maximum(pp.population_weight * pop_pressure + pp.elite_weight * elite_pressure
                          - pp.state_weight * strength, 0.0)
    P.pressure[pids] = pressure

    # Frontier pressure.
    contacts = settlement_contacts(model, ids)
    other = np.where(contacts >= 0, t.polity[np.maximum(contacts, 0)], -1)
    foreign = (other >= 0) & (other != pol[:, None])
    dist = culture_distance(culture[pol][:, None, :], culture[np.maximum(other, 0)])
    distant = np.clip((dist - cc.distant_culture_min) / max(cc.distant_culture_full - cc.distant_culture_min,
                                                            TRAIT_DIST_EPS), 0.0, 1.0)
    facing = np.where(foreign, distant, 0.0).max(axis=1, initial=0.0)
    border = foreign.any(axis=1)
    n_border = np.bincount(pol, weights=border.astype(float), minlength=npol)
    f = np.bincount(pol, weights=facing, minlength=npol)[pids] / np.maximum(n_border[pids], 1.0)
    w = P.war_recent[pids]
    f = np.minimum(f + cc.war_frontier_weight * w / (w + cc.war_frontier_half), 1.0)
    P.frontier[pids] = f

    # Cohesion.
    admin = g.multiplier(t.knows[capital], "admin_cap")
    size = n_set[pids] / (cc.size_reference_settlements * admin * (0.5 + hierarchy[pids]))
    steps = model.geom.hex_distance(t.cell[ids], t.cell[P.capital[pol]]).astype(float)
    mean_steps = np.bincount(pol, weights=pop * steps, minlength=npol)[pids] / safe_people[pids]
    logf = np.log(np.maximum(t.food_ratio[ids], 1e-3))
    mu = np.bincount(pol, weights=pop * logf, minlength=npol) / safe_people
    inequality = np.sqrt(np.maximum(np.bincount(pol, weights=pop * (logf - mu[pol]) ** 2, minlength=npol)
                                    / safe_people, 0.0))[pids]
    grievance = np.bincount(pol, weights=pop * t.grievance[ids], minlength=npol)[pids] / safe_people[pids]
    e = P.marker[pids][:, marker_index("cohesion_appearance", "cohesion_dialect")]
    mean_app = np.bincount(pol, weights=pop * app, minlength=npol)[pids] / safe_people[pids]
    mean_dia = np.bincount(pol, weights=pop * dia, minlength=npol)[pids] / safe_people[pids]
    delta = (cc.base_decay + cc.size_decay * size + cc.distance_decay * mean_steps / cc.distance_reference
             + cc.inequality_decay * inequality + cc.elite_decay * elite_pressure + cc.pressure_decay * pressure
             + cc.grievance_decay * grievance + pc.marker_cohesion_decay * (e[:, 0] * mean_app + e[:, 1] * mean_dia))
    delta = np.maximum(delta, 0.0)
    c = P.cohesion[pids]
    m = maintained[pids]
    c = c + cc.growth_rate * f * c * (1.0 - c) - delta * c + cc.monument_cohesion * m / (m + cc.monument_half) * (1 - c)
    c = c + memory.cohesion_term(model, pids) * (1.0 - c)
    P.cohesion[pids] = np.clip(c, 0.0, 1.0)
