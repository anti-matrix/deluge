"""Births, deaths, famine and migration (spec section 6, phases 3 and 4).

- ``relocate_drowned`` (start of phase 3): bands whose cell went under the sea this year
  move to the best free land cell they can reach, or join a nearby band of their own
  lineage, before they eat; otherwise they dissolve.
- ``grow`` (phase 3b): net growth rises with food per head up to the settlement's maximum
  (the share-weighted mode maxima: ~0.3%/yr foragers, ~1% farmers); below one ration,
  deaths rise in proportion to the shortfall (softened while a medicine artifact lasts).
  Random demographic fluctuation scales with the square root of the population, so
  small bands can die out. Bands below ``min_population`` dissolve.
- ``migrate`` (phase 4): starving bands move whole (sedentary farmers only after
  ``farmer_famine_years`` famine years in a row), to a free cell or, where the land
  around is full, into a band of their own lineage that has more food per head; crowded
  bands split and found a daughter settlement on the best free cell in reach.

Reach is ``search_radius`` hex steps over land, plus technique bonuses; with sailing a
path may cross a few sea cells. A cell's appeal is a utility over expected food per head
there (for the settlement's own modes and techniques), distance and crowding. War
danger enters in Round 6.
"""

from __future__ import annotations

import numpy as np

from deluge.agents.settlement import ALIVE, MODES
from deluge.config import DemographyConfig
from deluge.systems.subsistence import fed_capacity, food_per_head, mode_capacity, mode_parameters


def next_population(pop: np.ndarray, food_ratio: np.ndarray, abundance: np.ndarray, max_growth: np.ndarray,
                    cfg: DemographyConfig, noise: np.ndarray,
                    famine_factor: np.ndarray | float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(new population, famine deaths) for settlements with this food per head, food
    abundance and growth ceiling; ``noise`` is a standard normal draw per settlement,
    ``famine_factor`` scales famine deaths. Never negative."""
    fed = np.minimum(food_ratio, abundance)
    growth = max_growth * np.clip((fed - 1.0) / np.maximum(abundance - 1.0, 1e-12), 0.0, 1.0)
    famine = np.minimum(cfg.famine_max_mortality, cfg.famine_mortality * np.maximum(1.0 - food_ratio, 0.0))
    deaths = pop * famine * famine_factor
    fluct = np.sqrt(np.maximum(pop, 0.0) * cfg.demographic_turnover) * noise
    new = np.maximum(pop * (1.0 + growth) - deaths + fluct, 0.0)
    return new, np.minimum(deaths, pop)


def grow(model) -> None:
    d = model.config.demography
    t = model.settlements
    ids = t.alive_ids()
    a, r = mode_parameters(d)
    shares = t.shares[ids]
    noise = model.streams["demography"].standard_normal(len(ids))
    medicine = model.teachers_state.medicine_holders(ids)
    factor = np.where(medicine, model.config.teachers.medicine_famine_mortality, 1.0)
    pop, ratio = t.population[ids], t.food_ratio[ids]
    new, deaths = next_population(pop, ratio, shares @ a, shares @ r, d, noise, factor)
    t.population[ids] = new
    t.peak_population[ids] = np.maximum(t.peak_population[ids], new)
    t.famine_years[ids] = np.where(ratio < 1.0, t.famine_years[ids] + 1, 0)
    model.stats["famine_deaths"] += float(deaths.sum())
    for i in np.flatnonzero(deaths > 0).tolist():
        sid = int(ids[i])
        model.log_event("starved", [("settlement", sid)], int(t.cell[sid]), float(deaths[i]))
        if medicine[i]:
            spared = float(deaths[i] / factor[i] - deaths[i])
            for art in model.teachers_state.medicine_artifacts(sid):
                model.log_event("used", [("artifact", art), ("settlement", sid)], int(t.cell[sid]), spared)
    for sid in ids[new < d.min_population].tolist():
        model.end_settlement(sid, "dissolved")


# --- migration ------------------------------------------------------------------------------


def reach(model, sid: int, origin: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cells, steps): land cells a band can reach from ``origin`` (its own cell by
    default) within its search radius, crossing at most its sea_steps sea cells (the
    origin itself excluded)."""
    m, g, t = model.config.migration, model.knowledge, model.settlements
    knows = t.knows[sid]
    radius = m.search_radius + int(g.total(knows, "migration_radius"))
    sea_steps = int(g.total(knows, "sea_steps"))
    origin = int(t.cell[sid]) if origin is None else origin
    sea = model.is_sea
    nbrs = model.geom.neighbor_lists
    seen = {origin: 0}
    land_steps = {}
    frontier = [(origin, 0)]  # (cell, sea cells crossed)
    for step in range(1, radius + 1):
        nxt = []
        for c, wet in frontier:
            for n in nbrs[c]:
                if n in seen:
                    continue
                w = wet + int(sea[n])
                if w > sea_steps:
                    continue
                seen[n] = step
                nxt.append((n, w))
                if not sea[n]:
                    land_steps[n] = step
        frontier = nxt
    return (np.fromiter(land_steps.keys(), dtype=np.int64, count=len(land_steps)),
            np.fromiter(land_steps.values(), dtype=np.int64, count=len(land_steps)))


def expected_food(model, sid: int, cells: np.ndarray, people: float) -> np.ndarray:
    """Food per head ``people`` of settlement ``sid`` could expect on each cell, with its
    own modes and techniques."""
    t = model.settlements
    knows = np.repeat(t.knows[sid][None, :], len(cells), axis=0)
    k = mode_capacity(model, knows, cells)
    shares = np.repeat(t.shares[sid][None, :], len(cells), axis=0)
    return food_per_head(shares, k, people, model.config.demography)


def cell_utility(model, sid: int, cells: np.ndarray, steps: np.ndarray, people: float) -> np.ndarray:
    m = model.config.migration
    expected = expected_food(model, sid, cells, people)
    occupied = np.append(model.occupant >= 0, False)
    crowd = occupied[model.geom.neighbors[cells]].sum(axis=1)
    with np.errstate(divide="ignore"):
        return np.log(expected) - m.distance_cost * steps - m.crowding_cost * crowd


def best_free_cell(model, sid: int, people: float, min_ratio: float, origin: int | None = None) -> tuple[int, float]:
    """The free reachable land cell with the highest utility where expected food per head
    is at least ``min_ratio``: (cell, utility), or (-1, -inf)."""
    cells, steps = reach(model, sid, origin)
    free = model.occupant[cells] < 0
    cells, steps = cells[free], steps[free]
    if len(cells) == 0:
        return -1, -np.inf
    ok = expected_food(model, sid, cells, people) >= min_ratio
    if not ok.any():
        return -1, -np.inf
    cells, steps = cells[ok], steps[ok]
    u = cell_utility(model, sid, cells, steps, people)
    i = int(np.argmax(u))
    return int(cells[i]), float(u[i])


def best_kin_to_join(model, sid: int) -> int:
    """The reachable band of the same lineage where food per head after taking in band
    ``sid`` would be highest, if that beats staying; -1 if none does."""
    t, d = model.settlements, model.config.demography
    origin, pop = int(t.cell[sid]), float(t.population[sid])
    cells, _ = reach(model, sid)
    hosts = model.occupant[cells]
    hosts = hosts[hosts >= 0]
    hosts = hosts[t.lineage[hosts] == t.lineage[sid]]
    if len(hosts) == 0:
        return -1
    k_host = mode_capacity(model, t.knows[hosts], t.cell[hosts])
    after = food_per_head(t.shares[hosts], k_host, t.population[hosts] + pop, d)
    here = expected_food(model, sid, np.array([origin]), pop)[0]
    i = int(np.argmax(after))
    return int(hosts[i]) if after[i] > here else -1


def join_kin(model, sid: int, host: int) -> None:
    """Band ``sid`` joins band ``host``: people, store and techniques pool, artifacts
    pass to the host."""
    t = model.settlements
    pop = float(t.population[sid])
    t.population[host] += pop
    t.peak_population[host] = max(t.peak_population[host], t.population[host])
    t.store[host] += t.store[sid]
    t.strength[host] = np.maximum(t.strength[host], t.strength[sid])
    t.knows[host] |= t.knows[sid]
    t.practised[host] |= t.practised[sid]
    model.log_event("joined", [("settlement", sid), ("settlement", host)], int(t.cell[host]), pop)
    model.end_settlement(sid, "joined", host=host)


def move(model, sid: int, cell: int) -> None:
    pop = float(model.settlements.population[sid])
    model.move_settlement(sid, cell)
    model.stats["moves"] += 1
    model.log_event("moved", [("settlement", sid)], cell, pop)


def relocate_drowned(model) -> None:
    t = model.settlements
    ids = t.alive_ids()
    drowned = ids[model.is_sea[t.cell[ids]]]
    for sid in drowned.tolist():
        pop = float(t.population[sid])
        cell, _ = best_free_cell(model, sid, pop, 0.0)
        if cell >= 0:
            move(model, sid, cell)
            continue
        cells, _ = reach(model, sid)
        host = model.occupant[cells]
        kin = host[(host >= 0) & (t.lineage[np.maximum(host, 0)] == t.lineage[sid])]
        if len(kin):
            join_kin(model, sid, int(kin[np.argmax(t.population[kin])]))
        else:
            model.stats["drowned_lost"] += pop
            model.end_settlement(sid, "drowned")


def free_land_nearby(model) -> np.ndarray:
    """Per cell: some free land cell lies within the largest reach any band can have.
    Reach never extends past that many hex steps, so a band whose cell is False here
    cannot find a free cell; skipping its search changes nothing but the run time."""
    radius = model.config.migration.search_radius + int(model.knowledge.effect["migration_radius"].sum())
    near = ~model.is_sea & (model.occupant < 0)
    for _ in range(radius):
        near = near | model.geom.neighbor_any(near)
    return near


def migrate(model) -> None:
    m, d = model.config.migration, model.config.demography
    t = model.settlements
    year = model.year
    farm = MODES.index("farm")
    near_free = free_land_nearby(model)

    # Starving bands move whole, if somewhere reachable promises more: a free cell, or
    # else a band of their own lineage to join. Sedentary farmers wait out short famines.
    ids = t.alive_ids()
    starving = t.food_ratio[ids] < m.famine_move_ratio
    sedentary = t.shares[ids, farm] >= m.sedentary_farm_share
    starving &= ~sedentary | (t.famine_years[ids] >= m.farmer_famine_years)
    for sid in ids[starving].tolist():
        if t.ended[sid] != ALIVE:
            continue  # already joined by another band this year
        origin = int(t.cell[sid])
        pop = float(t.population[sid])
        if near_free[origin]:
            here = cell_utility(model, sid, np.array([origin]), np.zeros(1, dtype=np.int64), pop)[0]
            cell, util = best_free_cell(model, sid, pop, 0.0)
            if cell >= 0 and util > here:
                move(model, sid, cell)
                near_free = free_land_nearby(model)
                continue
        host = best_kin_to_join(model, sid)
        if host >= 0:
            join_kin(model, sid, host)
            model.stats["moves"] += 1
            near_free = free_land_nearby(model)

    # Crowded bands split (founding only fills cells, so the mask stays conservative).
    ids = t.alive_ids()
    k = fed_capacity(t.shares[ids], mode_capacity(model, t.knows[ids], t.cell[ids]), d)
    crowded = ids[(t.population[ids] >= m.split_threshold * k) & (t.next_split_year[ids] <= year)]
    for sid in crowded.tolist():
        leaving = m.split_fraction * float(t.population[sid])
        cell = -1
        if near_free[t.cell[sid]]:
            cell, _ = best_free_cell(model, sid, leaving, m.min_found_food_ratio)
        if cell < 0 or leaving < d.min_population:
            t.next_split_year[sid] = year + m.split_retry_years
            continue
        share = leaving / float(t.population[sid])
        store = share * float(t.store[sid])
        t.population[sid] -= leaving
        t.store[sid] -= store
        child = model.found_settlement(cell, leaving, int(t.lineage[sid]), sid, store,
                                       knows=t.knows[sid].copy(), strength=t.strength[sid].copy(),
                                       shares=t.shares[sid].copy(), practised=t.practised[sid].copy())
        model.stats["foundings"] += 1
        model.log_event("founded", [("settlement", child), ("settlement", sid)], cell, leaving)
