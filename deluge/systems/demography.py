"""Births, deaths, famine and migration (spec section 6, phases 3 and 4).

- ``relocate_drowned`` (start of phase 3): bands whose cell went under the sea this year
  move to the best free land cell they can reach, or join a nearby band of their own
  lineage, before they eat; otherwise they dissolve.
- ``grow`` (phase 3b): net growth rises with food per head up to the forager maximum;
  below one ration, deaths rise in proportion to the shortfall. Random demographic
  fluctuation scales with the square root of the population, so small bands can die
  out. Bands below ``min_population`` dissolve.
- ``migrate`` (phase 4): starving forager bands move whole, to a free cell or, where the
  land around is full, into a band of their own lineage that has more food per head;
  crowded bands split and found a daughter settlement on the best free cell within
  ``search_radius``.

A cell's appeal is a utility over expected food per head there, distance (hex steps over
land) and crowding (occupied neighbours). War danger enters in Round 6.
"""

from __future__ import annotations

import numpy as np

from deluge.config import DemographyConfig


def next_population(pop: np.ndarray, food_ratio: np.ndarray, cfg: DemographyConfig,
                    noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(new population, famine deaths) for bands with this food per head; ``noise`` is a
    standard normal draw per band. Never negative."""
    a = cfg.forage_abundance
    fed = np.minimum(food_ratio, a)
    growth = cfg.forage_max_growth * np.clip((fed - 1.0) / (a - 1.0), 0.0, 1.0)
    famine = np.minimum(cfg.famine_max_mortality, cfg.famine_mortality * np.maximum(1.0 - food_ratio, 0.0))
    deaths = pop * famine
    fluct = np.sqrt(np.maximum(pop, 0.0) * cfg.demographic_turnover) * noise
    new = np.maximum(pop * (1.0 + growth) - deaths + fluct, 0.0)
    return new, deaths


def grow(model) -> None:
    d = model.config.demography
    t = model.settlements
    ids = t.alive_ids()
    noise = model.streams["demography"].standard_normal(len(ids))
    new, deaths = next_population(t.population[ids], t.food_ratio[ids], d, noise)
    t.population[ids] = new
    model.stats["famine_deaths"] += float(deaths.sum())
    for sid in ids[new < d.min_population].tolist():
        model.end_settlement(sid, "dissolved")


# --- migration ------------------------------------------------------------------------------


def reachable_land(model, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """(cells, steps): land cells reachable from ``origin`` over land within
    ``search_radius`` hex steps (the origin itself excluded)."""
    radius = model.config.migration.search_radius
    land = ~model.is_sea
    nbrs = model.geom.neighbor_lists
    seen = {origin: 0}
    frontier = [origin]
    for step in range(1, radius + 1):
        nxt = []
        for c in frontier:
            for n in nbrs[c]:
                if n not in seen and land[n]:
                    seen[n] = step
                    nxt.append(n)
        frontier = nxt
    del seen[origin]
    return np.fromiter(seen.keys(), dtype=np.int64, count=len(seen)), np.fromiter(
        seen.values(), dtype=np.int64, count=len(seen))


def cell_utility(model, cells: np.ndarray, steps: np.ndarray, people: float) -> np.ndarray:
    """Utility of moving ``people`` to each of ``cells`` (``steps`` away)."""
    m, d = model.config.migration, model.config.demography
    k = model.capacity["forage"][cells] * model.forage_health[cells]
    expected = np.minimum(d.forage_abundance, k / max(people, 1e-12))
    occupied = np.append(model.occupant >= 0, False)
    crowd = occupied[model.geom.neighbors[cells]].sum(axis=1)
    with np.errstate(divide="ignore"):
        return np.log(expected) - m.distance_cost * steps - m.crowding_cost * crowd


def best_free_cell(model, origin: int, people: float, min_ratio: float) -> tuple[int, float]:
    """The free reachable land cell with the highest utility where expected food per head
    is at least ``min_ratio``: (cell, utility), or (-1, -inf)."""
    cells, steps = reachable_land(model, origin)
    free = model.occupant[cells] < 0
    cells, steps = cells[free], steps[free]
    if len(cells) == 0:
        return -1, -np.inf
    k = model.capacity["forage"][cells] * model.forage_health[cells]
    ok = k / max(people, 1e-12) >= min_ratio
    if not ok.any():
        return -1, -np.inf
    cells, steps = cells[ok], steps[ok]
    u = cell_utility(model, cells, steps, people)
    i = int(np.argmax(u))
    return int(cells[i]), float(u[i])


def best_kin_to_join(model, sid: int) -> int:
    """The reachable band of the same lineage where food per head after taking in band
    ``sid`` would be highest, if that beats staying; -1 if none does."""
    t, d = model.settlements, model.config.demography
    origin, pop = int(t.cell[sid]), float(t.population[sid])
    cells, _ = reachable_land(model, origin)
    hosts = model.occupant[cells]
    hosts = hosts[(hosts >= 0)]
    hosts = hosts[t.lineage[hosts] == t.lineage[sid]]
    if len(hosts) == 0:
        return -1
    k_host = model.capacity["forage"][t.cell[hosts]] * model.forage_health[t.cell[hosts]]
    after = np.minimum(d.forage_abundance, k_host / (t.population[hosts] + pop))
    k_here = model.capacity["forage"][origin] * model.forage_health[origin]
    here = min(d.forage_abundance, k_here / max(pop, 1e-12))
    i = int(np.argmax(after))
    return int(hosts[i]) if after[i] > here else -1


def relocate_drowned(model) -> None:
    t = model.settlements
    ids = t.alive_ids()
    drowned = ids[model.is_sea[t.cell[ids]]]
    for sid in drowned.tolist():
        origin = int(t.cell[sid])
        pop = float(t.population[sid])
        cell, _ = best_free_cell(model, origin, pop, 0.0)
        if cell >= 0:
            model.move_settlement(sid, cell)
            model.stats["moves"] += 1
            continue
        cells, _ = reachable_land(model, origin)
        host = model.occupant[cells]
        kin = host[(host >= 0) & (t.lineage[np.maximum(host, 0)] == t.lineage[sid])]
        if len(kin):
            target = int(kin[np.argmax(t.population[kin])])
            t.population[target] += pop
            t.store[target] += t.store[sid]
            t.population[sid] = 0.0
            model.end_settlement(sid, "joined")
        else:
            model.stats["drowned_lost"] += pop
            model.end_settlement(sid, "drowned")


def migrate(model) -> None:
    m, d = model.config.migration, model.config.demography
    t = model.settlements
    year = model.year

    # Starving forager bands move whole, if somewhere reachable promises more: a free
    # cell, or else a band of their own lineage to join.
    ids = t.alive_ids()
    starving = ids[t.food_ratio[ids] < m.famine_move_ratio]
    for sid in starving.tolist():
        if t.ended[sid] != -1:
            continue  # already joined by another band this year
        origin = int(t.cell[sid])
        pop = float(t.population[sid])
        here = cell_utility(model, np.array([origin]), np.zeros(1, dtype=np.int64), pop)[0]
        cell, util = best_free_cell(model, origin, pop, 0.0)
        if cell >= 0 and util > here:
            model.move_settlement(sid, cell)
            model.stats["moves"] += 1
            continue
        host = best_kin_to_join(model, sid)
        if host >= 0:
            t.population[host] += pop
            t.store[host] += t.store[sid]
            t.population[sid] = 0.0
            model.end_settlement(sid, "joined")
            model.stats["moves"] += 1

    # Crowded bands split.
    ids = t.alive_ids()
    k = model.capacity["forage"][t.cell[ids]] * model.forage_health[t.cell[ids]]
    crowded = ids[(t.population[ids] >= m.split_threshold * k) & (t.next_split_year[ids] <= year)]
    for sid in crowded.tolist():
        leaving = m.split_fraction * float(t.population[sid])
        cell, _ = best_free_cell(model, int(t.cell[sid]), leaving, m.min_found_food_ratio)
        if cell < 0 or leaving < d.min_population:
            t.next_split_year[sid] = year + m.split_retry_years
            continue
        share = leaving / float(t.population[sid])
        store = share * float(t.store[sid])
        t.population[sid] -= leaving
        t.store[sid] -= store
        model.found_settlement(cell, leaving, int(t.lineage[sid]), sid, store)
        model.stats["foundings"] += 1
