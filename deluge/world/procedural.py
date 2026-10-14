"""Procedural world generator (spec section 5).

Each stage is a pure function of the previous arrays, its config section and (where it
draws randomness) an RNG stream. All arrays here are flat per-cell arrays indexed by the
cell id of ``HexGeometry``; ``generate_world`` reshapes them into ``WorldData`` fields.

Stage order: continent mask + noise -> elevation calibration -> erosion -> landmass check
-> depression filling -> temperature -> rainfall -> flow accumulation (rivers) -> terrain
-> fertility and wild food. Rivers come after rainfall because runoff is rain-weighted.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from deluge.config import (
    BaseTemperatureConfig,
    Config,
    ContinentConfig,
    ElevationConfig,
    ErosionConfig,
    FertilityConfig,
    LandmassConfig,
    NoiseConfig,
    RainfallConfig,
    RiverConfig,
    TerrainConfig,
    WildFoodConfig,
)
from deluge.world import noise
from deluge.world.climate import sea_level_curve
from deluge.world.grid import SQRT3, HexGeometry
from deluge.world.worlddata import Terrain, WorldData, terrain_table


class WorldGenerationError(RuntimeError):
    pass


# --- coordinates -----------------------------------------------------------------------


def normalized_positions(geom: HexGeometry) -> np.ndarray:
    """Hex centers scaled so the map spans u in [0, 1]; v keeps the aspect ratio."""
    return geom.positions / (SQRT3 * geom.width)


def map_aspect(geom: HexGeometry) -> float:
    return (1.5 * geom.height) / (SQRT3 * geom.width)


# --- elevation -------------------------------------------------------------------------


def _simplex(seed: int, uv: np.ndarray, frequency: float, offset: np.ndarray) -> np.ndarray:
    """OpenSimplex noise at each point of uv."""
    return noise.noise2(noise.permutation(seed), uv[:, 0] * frequency + offset[0],
                        uv[:, 1] * frequency + offset[1])


def fbm(uv: np.ndarray, rng: np.random.Generator, cfg: NoiseConfig, ridged: bool = False) -> np.ndarray:
    """Fractal simplex noise, roughly in [-1, 1]; ``ridged`` gives sharp crests in [0, 1]
    ((1 - |noise|) ** ridge_sharpness) that read as mountain ranges."""
    seed = int(rng.integers(0, 2**31 - 1))
    total = np.zeros(len(uv))
    amp, freq, norm = 1.0, cfg.ridge_frequency if ridged else cfg.base_frequency, 0.0
    for _ in range(cfg.ridge_octaves if ridged else cfg.octaves):
        offset = rng.uniform(0.0, 1000.0, size=2)
        n = _simplex(seed, uv, freq, offset)
        total += amp * ((1.0 - np.abs(n)) ** cfg.ridge_sharpness if ridged else n)
        norm += amp
        amp *= cfg.persistence
        freq *= cfg.lacunarity
    return total / norm


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def continent_mask(geom: HexGeometry, rng: np.random.Generator, cfg: ContinentConfig) -> np.ndarray:
    """A few low-frequency elliptical blobs of uneven size, domain-warped so their shapes
    are irregular. Returns 0 outside every blob, rising to 1 in their interiors."""
    uv = normalized_positions(geom)
    aspect = map_aspect(geom)
    count = int(rng.integers(cfg.count_min, cfg.count_max + 1))
    radii = np.sort(rng.uniform(cfg.radius_min, cfg.radius_max, size=count))[::-1]
    blobs: list[tuple[np.ndarray, float, float, float]] = []  # center, radius, elongation, angle
    lo = np.array([cfg.edge_margin, cfg.edge_margin])
    hi = np.array([1.0 - cfg.edge_margin, aspect - cfg.edge_margin])
    for r in radii:
        elong = float(rng.uniform(1.0, cfg.elongation_max))
        angle = float(rng.uniform(0.0, np.pi))
        for _ in range(cfg.placement_tries):  # a blob that finds no room is dropped
            c = rng.uniform(lo, hi)
            if all(np.linalg.norm(c - o) >= r + ro + cfg.min_separation for o, ro, _, _ in blobs):
                blobs.append((c, float(r), elong, angle))
                break
    warp_seed = int(rng.integers(0, 2**31 - 1))
    offsets = rng.uniform(0.0, 1000.0, size=(cfg.warp_octaves, 2, 2))
    warp = np.zeros_like(uv)
    amp, freq, norm = 1.0, cfg.warp_frequency, 0.0
    for o in range(cfg.warp_octaves):  # fractal warp gives ragged coasts
        for axis in range(2):
            warp[:, axis] += amp * _simplex(warp_seed, uv, freq, offsets[o, axis])
        norm += amp
        amp *= cfg.warp_persistence
        freq *= cfg.warp_lacunarity
    warped = uv + cfg.warp_strength * warp / norm
    mask = np.zeros(len(uv))
    for c, r, elong, angle in blobs:
        rel = warped - c
        ca, sa = np.cos(angle), np.sin(angle)
        s = np.sqrt(elong)  # axis ratio = elong, area kept at pi r^2
        a = (rel[:, 0] * ca + rel[:, 1] * sa) / (r * s)
        b = (-rel[:, 0] * sa + rel[:, 1] * ca) * s / r
        d = np.sqrt(a * a + b * b)
        mask = np.maximum(mask, smoothstep((1.0 - d) / cfg.rim_width))
    return mask


def raw_elevation(geom: HexGeometry, rng: np.random.Generator, noise: NoiseConfig,
                  continents: ContinentConfig) -> np.ndarray:
    """Unitless relief: continent mask + fractal noise + ridged noise inside continents,
    pushed down near the map edges."""
    uv = normalized_positions(geom)
    aspect = map_aspect(geom)
    mask = continent_mask(geom, rng, continents)
    detail = fbm(uv, rng, noise)
    ridges = fbm(uv, rng, noise, ridged=True)
    edge_dist = np.minimum.reduce([uv[:, 0], 1.0 - uv[:, 0], uv[:, 1], aspect - uv[:, 1]])
    edge = np.clip(1.0 - edge_dist / noise.edge_falloff_width, 0.0, 1.0)
    return (noise.mask_weight * mask + noise.noise_weight * detail
            + noise.ridge_weight * ridges * mask - noise.edge_falloff_weight * edge)


def calibrate_elevation(raw: np.ndarray, land_share: float, ref_sea_m: float,
                        cfg: ElevationConfig) -> np.ndarray:
    """Map relief to metres by rank (a hypsometric curve), so ``land_share`` of cells lie
    at or above ``ref_sea_m`` whatever the relief's distribution. Only the order of cells
    matters, so the spatial pattern (ranges, valleys, coasts) is kept.

    The lowest ``shelf_share_of_land`` of land cells become the shelf between
    ``ref_sea_m`` and present sea level, the land a rising sea will drown; the rest rises
    from 0 m to ``land_max_m`` as rank ** ``land_exponent``.
    """
    n = len(raw)
    order = np.argsort(raw, kind="stable")
    n_land = int(round(land_share * n))
    land_ids = order[n - n_land:]  # ascending relief
    sea_ids = order[: n - n_land]
    out = np.empty_like(raw, dtype=float)

    n_shelf = int(round(cfg.shelf_share_of_land * n_land))
    shelf_ids, high_ids = land_ids[:n_shelf], land_ids[n_shelf:]
    if n_shelf:
        q = np.arange(n_shelf) / n_shelf
        out[shelf_ids] = ref_sea_m + (0.0 - ref_sea_m) * q
    base = max(ref_sea_m, 0.0)
    if len(high_ids):
        q = (np.arange(len(high_ids)) + 1) / len(high_ids)
        out[high_ids] = base + (cfg.land_max_m - base) * q**cfg.land_exponent
    if len(sea_ids):
        top = ref_sea_m - cfg.shelf_step_m
        d = 1.0 - np.arange(len(sea_ids)) / max(len(sea_ids) - 1, 1)  # 1 deepest .. 0 shallowest
        out[sea_ids] = top - (top - cfg.ocean_min_m) * d**cfg.ocean_exponent
    return out


def erode(geom: HexGeometry, elev: np.ndarray, sea_level_m: float, rng: np.random.Generator,
          cfg: ErosionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Light droplet erosion. Returns (eroded elevation, deposited sediment in metres)."""
    nbrs = geom.neighbor_lists
    e = elev.tolist()
    dep = [0.0] * len(e)
    land = np.flatnonzero(elev >= sea_level_m)
    if len(land) == 0 or cfg.droplets <= 0:
        return elev.copy(), np.zeros_like(elev)
    starts = rng.choice(land, size=cfg.droplets).tolist()
    for c in starts:
        sed = 0.0
        for _ in range(cfg.max_steps):
            ec = e[c]
            n = min(nbrs[c], key=e.__getitem__)
            en = e[n]
            if en >= ec:  # pit: fill up to the lowest neighbour, the rest settles as lost
                d = min(sed, en - ec)
                e[c] += d
                dep[c] += d
                sed = 0.0
                break
            drop = ec - en
            cap = cfg.capacity_factor * drop
            if sed > cap:
                d = cfg.deposit_rate * (sed - cap)
                e[c] += d
                dep[c] += d
                sed -= d
            else:
                cut = min(cfg.erode_rate * (cap - sed), cfg.max_erode_fraction * drop)
                e[c] -= cut
                sed += cut
            c = n
            if e[c] < sea_level_m:  # reached the sea; sediment is carried offshore
                sed = 0.0
                break
        if sed > 0.0 and e[c] >= sea_level_m:  # ran out of steps on land
            e[c] += sed
            dep[c] += sed
    return np.asarray(e), np.asarray(dep)


@dataclass(frozen=True)
class LandmassReport:
    labels: np.ndarray  # component label per cell, -1 for sea; 0 is the largest
    sizes: tuple[int, ...]  # all land components, largest first
    landmasses: tuple[int, ...]  # components at or above the size floor
    ok: bool


def landmass_report(geom: HexGeometry, elev: np.ndarray, sea_level_m: float,
                    cfg: LandmassConfig) -> LandmassReport:
    land = elev >= sea_level_m
    labels = geom.components(land)
    sizes = tuple(int(s) for s in np.bincount(labels[labels >= 0])) if land.any() else ()
    floor = cfg.min_share_of_land * land.sum()
    big = tuple(s for s in sizes if s >= floor)
    ok = (
        cfg.min_count <= len(big) <= cfg.max_count
        and len(big) > 0
        and big[0] / big[-1] >= cfg.min_size_ratio
    )
    return LandmassReport(labels, sizes, big, ok)


def fill_depressions(geom: HexGeometry, elev: np.ndarray, sea_level_m: float,
                     epsilon: float) -> np.ndarray:
    """Priority-flood fill (Barnes et al. 2014, with epsilon) seeded from the sea, so every
    land cell has a strictly descending path to a sea cell."""
    nbrs = geom.neighbor_lists
    filled = elev.astype(float).copy()
    sea = elev < sea_level_m
    seeds = np.flatnonzero(sea)
    if len(seeds) == 0:
        raise WorldGenerationError("no sea cells to drain to")
    closed = sea.copy()
    heap = [(float(filled[c]), int(c)) for c in seeds]
    heapq.heapify(heap)
    while heap:
        h, c = heapq.heappop(heap)
        for n in nbrs[c]:
            if closed[n]:
                continue
            closed[n] = True
            if filled[n] <= h + epsilon:
                filled[n] = h + epsilon
            heapq.heappush(heap, (float(filled[n]), n))
    # Filled land must stay land.
    land = ~sea
    filled[land] = np.maximum(filled[land], sea_level_m)
    return filled


def flow_receivers(geom: HexGeometry, elev: np.ndarray, sea_level_m: float) -> np.ndarray:
    """Steepest-descent downstream neighbor of each land cell (-1 for sea cells and pits)."""
    padded = np.append(elev, np.inf)
    nb = geom.neighbors
    nb_elev = padded[nb]  # off-map neighbors read +inf
    k = np.argmin(nb_elev, axis=1)
    lowest = nb_elev[np.arange(len(elev)), k]
    recv = nb[np.arange(len(elev)), k]
    recv = np.where((lowest < elev) & (elev >= sea_level_m), recv, -1)
    return recv


def flow_accumulation(elev: np.ndarray, receivers: np.ndarray, runoff: np.ndarray,
                      sea_level_m: float) -> np.ndarray:
    """Upstream runoff summed down the receiver tree (cells x metres/year)."""
    land = elev >= sea_level_m
    acc = np.where(land, runoff, 0.0).astype(float)
    order = np.flatnonzero(land)
    order = order[np.argsort(-elev[order], kind="stable")]
    recv = receivers.tolist()
    a = acc.tolist()
    for c in order.tolist():
        r = recv[c]
        if r >= 0:
            a[r] += a[c]
    return np.asarray(a)


# --- climate fields ----------------------------------------------------------------------


def base_temperature(lat: np.ndarray, elev: np.ndarray, cfg: BaseTemperatureConfig) -> np.ndarray:
    """Latitude curve plus lapse rate on land above present sea level."""
    t = cfg.equator_c + (cfg.pole_c - cfg.equator_c) * (np.abs(lat) / 90.0) ** cfg.lat_exponent
    return t - cfg.lapse_rate_c_per_km * np.maximum(elev, 0.0) / 1000.0


def latitude_rain(lat: np.ndarray, cfg: RainfallConfig) -> np.ndarray:
    r = np.full(lat.shape, cfg.base_mm, dtype=float)
    for center, amp, width in cfg.bands:
        r += amp * np.exp(-(((lat - center) / width) ** 2))
    return r


def wind_direction(lat: float, cfg: RainfallConfig) -> int:
    for lo, hi, direction in cfg.winds:
        if lo <= lat < hi or (hi >= 90.0 and lat >= lo):
            return int(direction)
    raise WorldGenerationError(f"no wind band covers latitude {lat}")


def base_rainfall(geom: HexGeometry, elev: np.ndarray, lat: np.ndarray, sea_level_m: float,
                  cfg: RainfallConfig) -> np.ndarray:
    """Latitude bands carried along each row by the prevailing wind. Air recharges over sea,
    dries over land, rains out when forced uphill and leaves a shadow behind mountains."""
    lat_rain = latitude_rain(lat, cfg)
    rain = np.zeros_like(elev, dtype=float)
    for y in range(geom.height):
        row = [geom.cid(x, y) for x in range(geom.width)]
        if wind_direction(float(lat[row[0]]), cfg) < 0:
            row.reverse()
        m = cfg.upwind_moisture
        prev_h = sea_level_m
        for c in row:
            h = float(elev[c])
            if h < sea_level_m:
                rain[c] = lat_rain[c] * m
                m += cfg.ocean_recharge * (1.0 - m)
                prev_h = sea_level_m
            else:
                uplift_km = max(0.0, h - prev_h) / 1000.0
                rain[c] = lat_rain[c] * m * (1.0 + cfg.orographic_gain_per_km * uplift_km)
                m = max(cfg.min_moisture,
                        m * (1.0 - cfg.land_loss) * math.exp(-cfg.shadow_loss_per_km * uplift_km))
                prev_h = h
    for _ in range(cfg.smoothing_passes):
        rain = geom.neighbor_mean(rain)
    return rain


def runoff_m(rain: np.ndarray, temp: np.ndarray, cfg: RiverConfig) -> np.ndarray:
    et = cfg.et_base_mm + cfg.et_per_c_mm * np.maximum(temp, 0.0)
    return np.maximum(rain - et, 0.0) / 1000.0


# --- terrain and resources -------------------------------------------------------------------


def classify_terrain(geom: HexGeometry, elev: np.ndarray, temp: np.ndarray, rain: np.ndarray,
                     flow: np.ndarray, sea_level_m: float, cfg: TerrainConfig) -> np.ndarray:
    """Whittaker-style climate biome, overridden by relief, wetness and the coast."""
    sea = elev < sea_level_m
    t_pos = np.maximum(temp, 0.0)
    terrain = np.full(elev.shape, int(Terrain.FOREST), dtype=np.int8)
    terrain[rain < cfg.forest_rain_mm + cfg.forest_rain_per_c * t_pos] = Terrain.PLAIN
    terrain[rain < cfg.steppe_rain_mm + cfg.steppe_rain_per_c * t_pos] = Terrain.STEPPE
    terrain[rain < cfg.desert_rain_mm + cfg.desert_rain_per_c * t_pos] = Terrain.DESERT
    terrain[temp < cfg.tundra_c] = Terrain.STEPPE
    coast = ~sea & geom.neighbor_any(sea)
    terrain[coast] = Terrain.COAST
    wet = (~sea & (elev - sea_level_m <= cfg.wetland_max_m_above_sea)
           & ((flow >= cfg.wetland_min_flow) | (rain >= cfg.wetland_min_rain_mm)))
    terrain[wet] = Terrain.WETLAND
    terrain[elev >= cfg.hills_m] = Terrain.HILLS
    terrain[elev >= cfg.mountain_m] = Terrain.MOUNTAIN
    terrain[sea] = Terrain.SEA
    return terrain


def ramp(x: np.ndarray, zero: float, full: float) -> np.ndarray:
    """0 at ``zero``, 1 at ``full``, linear between, clipped."""
    return np.clip((x - zero) / (full - zero), 0.0, 1.0)


def fertility_field(terrain: np.ndarray, flow: np.ndarray, deposit: np.ndarray, temp: np.ndarray,
                    cfg: FertilityConfig) -> np.ndarray:
    f = terrain_table(cfg.base)[terrain]
    f = f + cfg.river_bonus * np.clip(np.log1p(flow) / np.log1p(cfg.river_flow_ref), 0.0, 1.0)
    f = f + cfg.sediment_bonus * np.clip(deposit / cfg.sediment_ref_m, 0.0, 1.0)
    f = f * ramp(temp, cfg.cold_zero_c, cfg.cold_full_c)
    f[terrain == Terrain.SEA] = 0.0
    return np.clip(f, 0.0, 1.0)


def wild_food_field(geom: HexGeometry, terrain: np.ndarray, flow: np.ndarray, temp: np.ndarray,
                    cfg: WildFoodConfig) -> np.ndarray:
    w = terrain_table(cfg.base)[terrain]
    land = terrain != Terrain.SEA
    forest = terrain == Terrain.FOREST
    open_land = land & ~forest
    edge = (forest & geom.neighbor_any(open_land)) | (open_land & geom.neighbor_any(forest))
    w = w + cfg.forest_edge_bonus * edge + cfg.river_bonus * (flow > 0)
    w = w * np.maximum(cfg.cold_floor, ramp(temp, cfg.cold_zero_c, cfg.cold_full_c))
    w[~land] = 0.0
    return np.clip(w, 0.0, 1.0)


# --- pipeline -------------------------------------------------------------------------------


def generate_elevation(geom: HexGeometry, cfg: Config, rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw worlds until the landmass rules hold. Returns (elevation, deposit, attempts)."""
    w = cfg.world
    if w.tectonics.enabled:
        raise NotImplementedError("world.tectonics.enabled: tectonics is not implemented yet")
    for attempt in range(1, w.max_attempts + 1):
        raw = raw_elevation(geom, rng, w.noise, w.continents)
        elev = calibrate_elevation(raw, w.land_share, w.reference_sea_level_m, w.elevation)
        if not landmass_report(geom, elev, w.reference_sea_level_m, w.landmasses).ok:
            continue
        eroded, deposit = erode(geom, elev, w.reference_sea_level_m, rng, w.erosion)
        # Re-apply the hypsometric curve: erosion reorders cells (carved valleys sink in
        # rank) but the land share and shelf share stay exact.
        eroded = calibrate_elevation(eroded, w.land_share, w.reference_sea_level_m, w.elevation)
        if landmass_report(geom, eroded, w.reference_sea_level_m, w.landmasses).ok:
            return eroded, deposit, attempt
    raise WorldGenerationError(
        f"no world met the landmass rules in {w.max_attempts} attempts; check world.landmasses"
    )


def generate_world(cfg: Config, rng: np.random.Generator) -> WorldData:
    w = cfg.world
    geom = HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)
    sea0 = float(sea_level_curve(cfg.climate.sea_level, 0)[0])

    elev, deposit, _ = generate_elevation(geom, cfg, rng)
    elev = fill_depressions(geom, elev, sea0, w.fill_epsilon_m)

    lat = geom.latitude(w.lat_south, w.lat_north)
    temp = base_temperature(lat, elev, w.temperature)
    rain = base_rainfall(geom, elev, lat, sea0, w.rainfall)

    receivers = flow_receivers(geom, elev, sea0)
    acc = flow_accumulation(elev, receivers, runoff_m(rain, temp, w.rivers), sea0)
    river = np.where((elev >= sea0) & (acc >= w.rivers.min_flow), acc, 0.0)

    terrain = classify_terrain(geom, elev, temp, rain, acc, sea0, w.terrain)
    fertility = fertility_field(terrain, river, deposit, temp, w.fertility)
    wild_food = wild_food_field(geom, terrain, river, temp, w.wild_food)

    shape = geom.shape
    return WorldData(
        width=geom.width,
        height=geom.height,
        elevation=elev.reshape(shape),
        terrain=terrain.reshape(shape),
        river=river.reshape(shape),
        base_temp=temp.reshape(shape),
        base_rain=rain.reshape(shape),
        fertility=fertility.reshape(shape),
        wild_food=wild_food.reshape(shape),
    )
