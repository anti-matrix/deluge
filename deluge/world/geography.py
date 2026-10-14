"""Geographic analysis of an elevation field: landmasses, sea basins, sea-level pulses and
land bridges. Used to check generated worlds against the geography rules in config and to
report on them; nothing here changes the sim except ``construct_bridge``, which edits the
starting map when no natural bridge drowns in a pulse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deluge.config import LandBridgeConfig, LandmassConfig, SeaBasinConfig
from deluge.world.grid import HexGeometry


# --- landmasses ------------------------------------------------------------------------------


@dataclass(frozen=True)
class LandmassReport:
    labels: np.ndarray  # component label per cell, -1 for sea; 0 is the largest
    sizes: tuple[int, ...]  # all land components, largest first
    landmasses: tuple[int, ...]  # components at or above the size floor
    islands: int  # components below the floor
    ok: bool


def landmass_report(geom: HexGeometry, elev: np.ndarray, sea_level_m: float,
                    cfg: LandmassConfig) -> LandmassReport:
    land = elev >= sea_level_m
    labels = geom.components(land)
    sizes = tuple(int(s) for s in np.bincount(labels[labels >= 0])) if land.any() else ()
    floor = cfg.min_share_of_land * land.sum()
    big = tuple(s for s in sizes if s >= floor)
    islands = len(sizes) - len(big)
    ok = (
        cfg.min_count <= len(big) <= cfg.max_count
        and big[0] / big[-1] >= cfg.min_size_ratio
        and islands >= cfg.min_islands
    ) if big else False
    return LandmassReport(labels, sizes, big, islands, ok)


# --- seas ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeaBasinReport:
    inland_seas: tuple[int, ...]  # sea bodies not connected to the main ocean, sizes
    enclosed_basins: tuple[int, ...]  # open-water basins joined to the ocean only by straits
    ok: bool


def sea_basin_report(geom: HexGeometry, elev: np.ndarray, sea_level_m: float,
                     cfg: SeaBasinConfig) -> SeaBasinReport:
    sea = elev < sea_level_m
    labels = geom.components(sea)
    sizes = np.bincount(labels[labels >= 0]) if sea.any() else np.array([], dtype=int)
    inland = tuple(int(s) for s in sizes[1:] if s >= cfg.min_basin_cells)
    steps, _ = geom.distance_from(~sea)
    open_water = sea & (steps >= cfg.open_water_min_steps)
    olabels = geom.components(open_water)
    osizes = np.bincount(olabels[olabels >= 0]) if open_water.any() else np.array([], dtype=int)
    enclosed = tuple(int(s) for s in osizes[1:] if s >= cfg.min_basin_cells)
    ok = (not cfg.required) or bool(inland or enclosed)
    return SeaBasinReport(inland, enclosed, ok)


# --- sea-level pulses ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Pulse:
    start_year: int
    end_year: int  # exclusive
    start_level: float
    end_level: float


def pulse_windows(sea_level: np.ndarray, rate_factor: float) -> tuple[Pulse, ...]:
    """Runs of years in which the sea rises faster than ``rate_factor`` times the mean
    rate of its rising years."""
    rate = np.diff(sea_level)
    rising = rate > 0
    if not rising.any():
        return ()
    fast = rate > rate_factor * rate[rising].mean()
    pulses = []
    year = 0
    while year < len(fast):
        if fast[year]:
            start = year
            while year < len(fast) and fast[year]:
                year += 1
            pulses.append(Pulse(start, year, float(sea_level[start]), float(sea_level[year])))
        else:
            year += 1
    return tuple(pulses)


# --- rivers ---------------------------------------------------------------------------------------


def river_lengths(elev: np.ndarray, river: np.ndarray, receivers: np.ndarray) -> dict[int, int]:
    """For each river mouth (a river cell whose receiver is not a river cell), the length
    in river cells of the longest river path that ends there."""
    cells = np.flatnonzero(river > 0)
    is_river = river > 0
    longest = {int(c): 1 for c in cells}
    for c in cells[np.argsort(-elev[cells], kind="stable")].tolist():
        r = int(receivers[c])
        if r >= 0 and is_river[r]:
            longest[r] = max(longest[r], longest[c] + 1)
    return {c: n for c, n in longest.items() if receivers[c] < 0 or not is_river[receivers[c]]}


def row_aligned_runs(geom: HexGeometry, river: np.ndarray, receivers: np.ndarray,
                     segment: int = 4) -> tuple[int, float]:
    """(longest run of consecutive river steps going the same way along a hex row, share
    of river cells from which the next ``segment`` steps all go the same way along the
    row)."""
    x, y = geom.coords
    is_river = river > 0
    cells = np.flatnonzero(is_river)
    rec = receivers
    step = np.zeros(len(river), dtype=np.int64)  # +1 / -1 along the row, 0 otherwise
    ok = rec >= 0
    idx = np.flatnonzero(ok)
    same_row = y[rec[idx]] == y[idx]
    step[idx[same_row]] = x[rec[idx[same_row]]] - x[idx[same_row]]
    best = 0
    straight = 0
    for c in cells.tolist():
        run, prev, k, cur = 0, 0, 0, c
        first_dir, first_ok = step[c], step[c] != 0
        while cur >= 0 and is_river[cur] and rec[cur] >= 0:
            s = step[cur]
            run = run + 1 if s != 0 and s == prev else (1 if s != 0 else 0)
            prev = s
            best = max(best, run)
            if k < segment and s != first_dir:
                first_ok = False
            k += 1
            cur = int(rec[cur])
        if first_ok and k >= segment:
            straight += 1
    return best, straight / max(len(cells), 1)


# --- land bridges --------------------------------------------------------------------------------


@dataclass(frozen=True)
class LandBridge:
    crest_cell: int  # the cell whose drowning breaks the connection
    crest_m: float  # the connection is lost once the sea rises above this
    year_drowned: int
    sizes: tuple[int, int]  # land cells on each side when it drowns
    pulse: int  # index into the pulse windows, -1 if not during a pulse


def land_bridges(geom: HexGeometry, elev: np.ndarray, sea_level: np.ndarray, floor_cells: float,
                 pulses: tuple[Pulse, ...]) -> tuple[LandBridge, ...]:
    """Every point where rising sea splits one land area into two that both have at least
    ``floor_cells`` cells.

    Land is added cell by cell from the highest down (union-find). When a cell joins two
    separate land areas, that cell is the bridge's crest: for any sea level above it the
    two areas are apart, at or below it they are one.
    """
    sea0, top = float(sea_level[0]), float(sea_level.max())
    land_ids = np.flatnonzero(elev >= sea0)
    order = land_ids[np.argsort(-elev[land_ids], kind="stable")]
    parent = {}
    size = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nbrs = geom.neighbor_lists
    bridges = []
    for c in order.tolist():
        parent[c] = c
        size[c] = 1
        roots = {find(n) for n in nbrs[c] if n in parent}
        if len(roots) >= 2:
            ranked = sorted(roots, key=lambda r: -size[r])
            h = float(elev[c])
            if sea0 <= h < top and size[ranked[1]] >= floor_cells:
                drowned = int(np.argmax(sea_level > h))
                pulse = next((i for i, p in enumerate(pulses) if p.start_year <= drowned < p.end_year), -1)
                bridges.append(LandBridge(c, h, drowned, (size[ranked[0]], size[ranked[1]]), pulse))
        for r in roots:
            parent[r] = c
            size[c] += size[r]
    return tuple(bridges)


def construct_bridge(geom: HexGeometry, elev: np.ndarray, sea_level: np.ndarray,
                     pulses: tuple[Pulse, ...], floor_cells: float, cfg: LandBridgeConfig,
                     rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, ...]]:
    """Raise a narrow land bridge between the two nearest large landmasses (as they stand
    after a pulse), with its crest inside that pulse's sea-level span, so it drowns during
    the pulse. Returns (new elevation, bridge cells), or (elev, ()) if no pair is close
    enough."""
    if not pulses:
        return elev, ()
    pulse = pulses[int(rng.integers(len(pulses)))]
    lo = max(pulse.start_level, float(sea_level[0])) + cfg.crest_margin_m
    hi = pulse.end_level - cfg.crest_margin_m
    if hi <= lo:
        return elev, ()
    crest = float(rng.uniform(lo, hi))

    land_after = elev >= pulse.end_level
    labels = geom.components(land_after)
    sizes = np.bincount(labels[labels >= 0]) if land_after.any() else np.array([])
    big = [i for i, s in enumerate(sizes) if s >= floor_cells]
    best = None
    for i, a in enumerate(big):
        steps, src = geom.distance_from(labels == a)
        for b in big[i + 1:]:
            cells_b = np.flatnonzero(labels == b)
            j = cells_b[np.argmin(steps[cells_b])]
            if 0 < steps[j] <= cfg.max_length_steps and (best is None or steps[j] < best[0]):
                best = (int(steps[j]), a, int(j))
    if best is None:
        return elev, ()
    _, a, end = best
    # Walk back from the far shore toward landmass a, always to a neighbour one step closer.
    steps, _ = geom.distance_from(labels == a)
    path = []
    c = end
    while steps[c] > 1:
        c = min((n for n in geom.neighbor_lists[c] if steps[n] == steps[c] - 1),
                key=lambda n: (elev[n] < pulse.end_level, -elev[n]))
        path.append(c)
    path.reverse()
    out = elev.copy()
    k = len(path)
    for i, c in enumerate(path):
        out[c] = crest + cfg.end_rise_m_per_step * min(i, k - 1 - i)
    flank = min(cfg.flank_floor_m, float(np.nextafter(sea_level[0], -np.inf)))  # stays sea
    on_path = set(path)
    for c in path:
        for n in geom.neighbor_lists[c]:
            if n not in on_path and out[n] < float(sea_level[0]):
                out[n] = max(out[n], flank)
    return out, tuple(path)
