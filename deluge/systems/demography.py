"""Births, deaths, famine and migration (spec section 6, phases 3 and 4).

- ``relocate_drowned`` (start of phase 3): bands whose cell went under the sea this year
  move to the best free land cell they can reach, or join a nearby band of their own
  lineage, before they eat; otherwise they dissolve.
- ``grow`` (phase 3b): net growth rises with food per head up to the settlement's maximum
  (the share-weighted mode maxima: ~0.3%/yr foragers, ~1% farmers); below one ration,
  deaths rise in proportion to the shortfall (softened while a medicine artifact lasts,
  and by techniques with a ``famine_mortality`` effect).
  Random demographic fluctuation scales with the square root of the population, so
  small bands can die out. Bands below ``min_population`` dissolve.
- ``migrate`` (phase 4): starving bands move whole (sedentary farmers only after
  ``farmer_famine_years`` famine years in a row), to a free cell or, where the land
  around is full, into a band of their own lineage that has more food per head; a band
  of another polity takes them in with chance = its polity brain's newcomers output.
  Each polity's capital band, unless sedentary, moves whole with chance
  ``core_move_rate`` x the brain's core migration output to a better free cell in reach.
  Crowded bands split and found a daughter settlement on the best free cell in reach;
  crowded means at least ``split_threshold`` + ``split_threshold_range`` x (0.5 - the
  brain's expand output) of the cell's capacity.

Reach is ``search_radius`` land cells entered, plus technique bonuses; with sailing a
path may also cross up to the band's sea reach in sea cells (the highest sea_steps of its
techniques, its sea tier: sailing, open-sea voyaging, ocean navigation), counted apart from
the land radius, as war counts them (``war.sea_reach``). A move,
a founding or a join that needs such a path (``sea_cells_crossed``) is logged as
``crossed`` with the fewest sea cells crossed, right after its own event; for the core's
migration the band's polity is the second actor. A cell's appeal is a utility over expected food per head
there (for the settlement's own modes and techniques), distance and crowding. War
danger enters in Round 6.
"""

from __future__ import annotations

import numpy as np

from deluge.agents.settlement import ALIVE, MODES
from deluge.brains.network import OUT
from deluge.config import DemographyConfig
from deluge.jit import njit
from deluge.systems import identity
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
    tech = model.knowledge.multiplier(t.knows[ids], "famine_mortality")  # e.g. modern medicine
    pop, ratio = t.population[ids], t.food_ratio[ids]
    new, deaths = next_population(pop, ratio, shares @ a, shares @ r, d, noise, factor * tech)
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


def _reach_field(model, sid: int, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """(land cells, sea cells crossed) within band ``sid``'s reach from ``origin``
    (``war.land_reach``): its land search radius and its sea_steps counted apart, as war
    counts them."""
    from deluge.systems.war import land_reach

    m, g, t = model.config.migration, model.knowledge, model.settlements
    knows = t.knows[sid]
    radius = m.search_radius + int(g.total(knows, "migration_radius"))
    sea_steps = int(g.highest(knows, "sea_steps"))
    return land_reach(model, np.array([origin]), radius, np.array([sea_steps]))[0]


def reach(model, sid: int, origin: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(cells, steps): land cells a band can reach from ``origin`` (its own cell by
    default) entering at most its search radius of land cells and crossing at most its
    sea_steps sea cells (the origin itself excluded); steps: hex distance."""
    t = model.settlements
    origin = int(t.cell[sid]) if origin is None else origin
    targets, _ = _reach_field(model, sid, origin)
    cells = targets[targets != origin].astype(np.int64)
    return cells, model.geom.hex_distance(cells, np.full(len(cells), origin))


def sea_cells_crossed(model, sid: int, origin: int, dest: int) -> int:
    """The fewest sea cells a path from ``origin`` to land cell ``dest`` within band
    ``sid``'s reach crosses (0 when a path over land alone reaches it; the origin itself not
    counted). Used to log the band's sea crossings."""
    targets, sea = _reach_field(model, sid, origin)
    hit = np.flatnonzero(targets == dest)
    return int(sea[hit[0]]) if len(hit) else 0


def expected_food(model, sid: int, cells: np.ndarray, people: float) -> np.ndarray:
    """Food per head ``people`` of settlement ``sid`` could expect on each cell, with its
    own modes and techniques (one technique row for every cell)."""
    t = model.settlements
    k = mode_capacity(model, t.knows[sid], cells)
    return food_per_head(t.shares[sid], k, people, model.config.demography)


def cell_utility(model, sid: int, cells: np.ndarray, steps: np.ndarray, people: float,
                 expected: np.ndarray | None = None) -> np.ndarray:
    """Utility of each cell for band ``sid`` (``expected``: its expected food per head
    there, when already computed)."""
    m = model.config.migration
    if expected is None:
        expected = expected_food(model, sid, cells, people)
    crowd = (np.append(model.occupant, -1)[model.geom.neighbors[cells]] >= 0).sum(axis=1)
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
    expected = expected_food(model, sid, cells, people)
    ok = expected >= min_ratio
    if not ok.any():
        return -1, -np.inf
    cells, steps = cells[ok], steps[ok]
    u = cell_utility(model, sid, cells, steps, people, expected[ok])
    i = int(np.argmax(u))
    return int(cells[i]), float(u[i])


def welcoming(model, sid: int, hosts: np.ndarray) -> np.ndarray:
    """The ``hosts`` that would take in band ``sid``: those of its own polity, and those of
    another polity with chance = that polity brain's newcomers output."""
    t = model.settlements
    foreign = t.polity[hosts] != t.polity[sid]
    if not foreign.any():
        return hosts
    accept = model.brains.output[t.polity[hosts], OUT["newcomers"]]
    draw = model.streams["migration"].random(len(hosts))
    return hosts[~foreign | (draw < accept)]


def best_kin_to_join(model, sid: int) -> int:
    """The reachable band of the same lineage that would take band ``sid`` in
    (``welcoming``) where food per head after taking it in would be highest, if that beats
    staying; -1 if none does."""
    t, d = model.settlements, model.config.demography
    origin, pop = int(t.cell[sid]), float(t.population[sid])
    cells, _ = reach(model, sid)
    hosts = model.occupant[cells]
    hosts = hosts[hosts >= 0]
    hosts = hosts[t.lineage[hosts] == t.lineage[sid]]
    if len(hosts):
        hosts = welcoming(model, sid, hosts)
    if len(hosts) == 0:
        return -1
    k_host = mode_capacity(model, t.knows[hosts], t.cell[hosts])
    after = food_per_head(t.shares[hosts], k_host, t.population[hosts] + pop, d)
    here = expected_food(model, sid, np.array([origin]), pop)[0]
    i = int(np.argmax(after))
    return int(hosts[i]) if after[i] > here else -1


def join_kin(model, sid: int, host: int) -> None:
    """Band ``sid`` joins band ``host``: people, store and techniques pool, artifacts
    pass to the host, appearance and dialect mix by numbers of people. The band leaves its
    polity (it may be another polity of the same founding group)."""
    t = model.settlements
    pop = float(t.population[sid])
    crossed = sea_cells_crossed(model, sid, int(t.cell[sid]), int(t.cell[host]))
    identity.blend(model, host, sid)
    t.population[host] += pop
    t.peak_population[host] = max(t.peak_population[host], t.population[host])
    t.store[host] += t.store[sid]
    t.strength[host] = np.maximum(t.strength[host], t.strength[sid])
    t.knows[host] |= t.knows[sid]
    t.practised[host] |= t.practised[sid]
    i = model.log_event("joined", [("settlement", sid), ("settlement", host)], int(t.cell[host]), pop)
    model.log_crossing(crossed, [("settlement", sid), ("settlement", host)], int(t.cell[host]), i)
    model.end_settlement(sid, "joined", host=host)


def move(model, sid: int, cell: int, core: int = -1) -> None:
    """Band ``sid`` moves whole to ``cell`` (``core``: the polity whose capital band it is,
    when the core migrates)."""
    t = model.settlements
    pop = float(t.population[sid])
    crossed = sea_cells_crossed(model, sid, int(t.cell[sid]), cell)
    model.move_settlement(sid, cell)
    model.stats["moves"] += 1
    i = model.log_event("moved", [("settlement", sid)], cell, pop)
    model.log_crossing(crossed, [("settlement", sid)] + ([("polity", core)] if core >= 0 else []), cell, i)


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
            kin = welcoming(model, sid, kin)
        if len(kin):
            join_kin(model, sid, int(kin[np.argmax(t.population[kin])]))
        else:
            model.stats["drowned_lost"] += pop
            model.end_settlement(sid, "drowned")


def free_land_nearby(model) -> np.ndarray:
    """Per cell: some free land cell lies within the largest land reach any band can have.
    Without sea_steps, reach never extends past that many hex steps, so a band whose cell
    is False here cannot find a free cell; skipping its search changes nothing but the run
    time (``may_find_free``)."""
    radius = model.config.migration.search_radius + int(model.knowledge.effect["migration_radius"].sum())
    return _dilate(model.geom.neighbors, ~model.is_sea & (model.occupant < 0), radius)


@njit(cache=True)
def _dilate(neighbors, flags, steps):
    """``flags`` grown ``steps`` times over hex neighbours (each step: a cell or one of its
    neighbours is set)."""
    cur = flags.copy()
    for _ in range(steps):
        nxt = cur.copy()
        for c in range(len(cur)):
            if not cur[c]:
                for e in range(neighbors.shape[1]):
                    m = neighbors[c, e]
                    if m >= 0 and cur[m]:
                        nxt[c] = True
                        break
        cur = nxt
    return cur


def may_find_free(model, near_free: np.ndarray, sid: int) -> bool:
    """Whether band ``sid``'s search for a free cell can succeed: free land in land reach,
    or it holds sea_steps (its reach over sea is not bounded by ``free_land_nearby``)."""
    t = model.settlements
    return bool(near_free[t.cell[sid]]) or model.knowledge.highest(t.knows[sid], "sea_steps") > 0


def move_cores(model, near_free: np.ndarray) -> None:
    """Migration of the core: a polity's capital band, unless sedentary, moves whole with
    chance ``core_move_rate`` x its brain's core migration output, if a free cell in reach
    promises more than staying."""
    m, t, P = model.config.migration, model.settlements, model.polities
    pids = P.alive_ids()
    if len(pids) == 0:
        return
    farm = MODES.index("farm")
    draw = model.streams["migration"].random(len(pids))
    going = pids[draw < m.core_move_rate * model.brains.output[pids, OUT["core_migration"]]]
    for pid in going.tolist():
        cap = int(P.capital[pid])
        if t.ended[cap] != ALIVE or t.polity[cap] != pid or t.shares[cap, farm] >= m.sedentary_farm_share:
            continue
        origin = int(t.cell[cap])
        if not may_find_free(model, near_free, cap):
            continue
        pop = float(t.population[cap])
        here = cell_utility(model, cap, np.array([origin]), np.zeros(1, dtype=np.int64), pop)[0]
        cell, util = best_free_cell(model, cap, pop, 0.0)
        if cell >= 0 and util > here:
            move(model, cap, cell, core=pid)
            near_free = free_land_nearby(model)


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
        if may_find_free(model, near_free, sid):
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

    move_cores(model, near_free)
    near_free = free_land_nearby(model)

    # Crowded bands split (founding only fills cells, so the mask stays conservative).
    ids = t.alive_ids()
    k = fed_capacity(t.shares[ids], mode_capacity(model, t.knows[ids], t.cell[ids]), d)
    threshold = m.split_threshold + m.split_threshold_range * (0.5 - model.brains.output[t.polity[ids], OUT["expand"]])
    crowded = ids[(t.population[ids] >= threshold * k) & (t.next_split_year[ids] <= year)]
    for sid in crowded.tolist():
        leaving = m.split_fraction * float(t.population[sid])
        cell = -1
        if may_find_free(model, near_free, sid):
            cell, _ = best_free_cell(model, sid, leaving, m.min_found_food_ratio)
        if cell < 0 or leaving < d.min_population:
            t.next_split_year[sid] = year + m.split_retry_years
            continue
        crossed = sea_cells_crossed(model, sid, int(t.cell[sid]), cell)
        share = leaving / float(t.population[sid])
        store = share * float(t.store[sid])
        t.population[sid] -= leaving
        t.store[sid] -= store
        child = model.found_settlement(cell, leaving, int(t.lineage[sid]), sid, store,
                                       knows=t.knows[sid].copy(), strength=t.strength[sid].copy(),
                                       shares=t.shares[sid].copy(), practised=t.practised[sid].copy())
        model.stats["foundings"] += 1
        i = model.log_event("founded", [("settlement", child), ("settlement", sid)], cell, leaving)
        model.log_crossing(crossed, [("settlement", child), ("settlement", sid)], cell, i)
