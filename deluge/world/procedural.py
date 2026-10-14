"""Procedural world generator (spec section 5).

Each stage is a pure function of the previous arrays, its config section and (where it
draws randomness) an RNG stream. Arrays are flat per-cell arrays indexed by the cell id of
``HexGeometry``; ``generate_world`` reshapes them into ``WorldData`` fields.

Stage order:
 1. plates (tectonics.py) and continent relief: warped multi-octave noise, plate bias,
    boundary ranges, plateaus, rifts and basins;
 2. continuous elevation through a hypsometric curve (knots in config);
 3. light droplet erosion;
 4. priority-flood depression filling with an epsilon gradient and random tie-breaks;
 5. geography checks (landmasses, islands, seas, a land bridge that drowns in a pulse),
    retried with fresh draws until they hold;
 6. distance to the sea, temperature, rainfall from a 2D moisture transport;
 7. runoff and flow accumulation (rivers);
 8. terrain classes, fertility, wild food.
"""

from __future__ import annotations

import heapq

import numpy as np

from deluge.config import (
    BaseTemperatureConfig,
    Config,
    ElevationConfig,
    ErosionConfig,
    FertilityConfig,
    RainfallConfig,
    RiverConfig,
    ShapeConfig,
    TerrainConfig,
    WildFoodConfig,
)
from deluge.world import geography, noise, tectonics
from deluge.world.climate import preset_horizon, sea_level_curve
from deluge.world.grid import SQRT3, HexGeometry, hex_unit_km
from deluge.world.tectonics import Plates, map_aspect, normalized_positions
from deluge.world.worlddata import GenerationReport, Terrain, WorldData, terrain_table


class WorldGenerationError(RuntimeError):
    pass


# --- relief and elevation ---------------------------------------------------------------------


def continent_relief(geom: HexGeometry, rng: np.random.Generator, cfg: ShapeConfig,
                     plates: Plates | None) -> np.ndarray:
    """Unitless relief before tectonics: domain-warped fractal noise (its high octaves
    give peninsulas, bays and island chains), a smoothed continental bias per plate, and a
    drop toward the map edge."""
    uv = normalized_positions(geom)
    warped = noise.warp(uv, rng, cfg.warp_frequency, cfg.warp_octaves, cfg.warp_persistence,
                        cfg.warp_lacunarity, cfg.warp_strength)
    relief = noise.fbm(warped, rng, cfg.base_frequency, cfg.octaves, cfg.persistence, cfg.lacunarity)
    if plates is not None:
        bias = plates.bias[plates.plate_id].astype(float)
        for _ in range(cfg.plate_bias_smoothing_passes):
            bias = geom.neighbor_mean(bias)
        relief = relief + cfg.plate_bias_weight * bias
    # Edge falloff measured on the warped coordinates, so coasts near the border bend
    # instead of running along it.
    # A second falloff on the true coordinates keeps land from reaching the border band.
    aspect = map_aspect(geom)

    def falloff(p):
        dist = np.minimum.reduce([p[:, 0], 1.0 - p[:, 0], p[:, 1], aspect - p[:, 1]])
        return np.clip(1.0 - dist / cfg.edge_falloff_width, 0.0, 1.0)

    relief = relief - cfg.edge_falloff_weight * np.maximum(falloff(warped), falloff(uv))
    # A band of open sea along the border: cells there have missing neighbours, which
    # would force rivers and coasts to run along the edge.
    x, y = geom.coords
    border = np.minimum.reduce([x, geom.width - 1 - x, y, geom.height - 1 - y])
    relief[border < cfg.edge_sea_cells] = relief.min() - 1.0
    return relief


def _land_threshold(values: np.ndarray, land_share: float) -> float:
    n = len(values)
    return float(np.sort(values)[n - int(round(land_share * n))])


def normalized_relief(relief: np.ndarray, land_share: float) -> np.ndarray:
    """z = (r - t) / (median land relief - t) on land and (r - t) / (t - median sea relief)
    at sea, where ``t`` puts ``land_share`` of cells on land: z = 0 is the coast, z = 1
    the median land cell, z = -1 the median sea cell. Piecewise affine, so slopes keep
    their shape."""
    t = _land_threshold(relief, land_share)
    land = relief >= t
    med_land = np.median(relief[land])
    med_sea = np.median(relief[~land])
    return np.where(land, (relief - t) / max(med_land - t, 1e-12), (relief - t) / max(t - med_sea, 1e-12))


def hypsometric_elevation(z: np.ndarray, land_share: float, ref_sea_m: float,
                          cfg: ElevationConfig) -> np.ndarray:
    """Continuous map from normalized relief to metres.

    ``z`` is shifted (not rescaled) so that ``land_share`` of cells have z >= 0, then
    mapped through the piecewise-linear knots in config, with z = 0 at ``ref_sea_m``.
    """
    land_z0 = cfg.land_knots[0]
    sea_z0 = cfg.sea_knots[0]
    if land_z0 != (0.0, ref_sea_m) or sea_z0 != (0.0, ref_sea_m):
        raise ValueError("elevation knots must start at [0, reference_sea_level_m]")
    z = z - _land_threshold(z, land_share)
    land = z >= 0.0
    lk = np.asarray(cfg.land_knots)
    sk = np.asarray(sorted(cfg.sea_knots))
    out = np.where(land, _interp_extrapolate(z, lk), _interp_extrapolate(z, sk))
    # Sea cells strictly below the reference level, land at or above it.
    out[~land] = np.minimum(out[~land], np.nextafter(ref_sea_m, -np.inf))
    out[land] = np.maximum(out[land], ref_sea_m)
    return out


def add_detail(geom: HexGeometry, elev: np.ndarray, ref_sea_m: float, rng: np.random.Generator,
               cfg: ElevationConfig) -> np.ndarray:
    """Small-scale relief on land (hummocks, spurs, low divides) so that flow wanders and
    converges instead of running straight down a smooth slope. Its amplitude is
    ``detail_m`` plus ``detail_fraction`` of the height above the reference level, so
    uplands get proportionally rougher. Land stays land."""
    uv = normalized_positions(geom)
    n = noise.fbm(uv, rng, cfg.detail_frequency, cfg.detail_octaves, cfg.detail_persistence,
                  cfg.detail_lacunarity)
    land = elev >= ref_sea_m
    amp = cfg.detail_m + cfg.detail_fraction * (elev - ref_sea_m)
    out = elev.copy()
    out[land] = np.maximum(ref_sea_m, elev[land] + (amp * n)[land])
    return out


def _interp_extrapolate(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear through knots, extended linearly past the outer ones."""
    xs, ys = knots[:, 0], knots[:, 1]
    y = np.interp(x, xs, ys)
    lo_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    hi_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + (x - xs[0]) * lo_slope, y)
    return np.where(x > xs[-1], ys[-1] + (x - xs[-1]) * hi_slope, y)


def erode(geom: HexGeometry, elev: np.ndarray, sea_level_m: float, rng: np.random.Generator,
          cfg: ErosionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Light droplet erosion. Returns (eroded elevation, deposited sediment in metres).

    Each droplet steps to a random lower neighbour, chosen with probability proportional
    to drop ** ``direction_exponent``. Always taking the steepest neighbour would carve
    valleys straight along the hex axes wherever the slope is smooth.
    """
    nbrs = geom.neighbor_lists
    e = elev.tolist()
    dep = [0.0] * len(e)
    land = np.flatnonzero(elev >= sea_level_m)
    if len(land) == 0 or cfg.droplets <= 0:
        return elev.copy(), np.zeros_like(elev)
    starts = rng.choice(land, size=cfg.droplets).tolist()
    draws = rng.random(cfg.droplets * cfg.max_steps).tolist()
    k = cfg.direction_exponent
    for i, c in enumerate(starts):
        sed = 0.0
        for s in range(cfg.max_steps):
            ec = e[c]
            lower = [(n, (ec - e[n]) ** k) for n in nbrs[c] if e[n] < ec]
            if not lower:  # pit: fill up to the lowest neighbour, the rest settles as lost
                d = min(sed, min(e[n] for n in nbrs[c]) - ec)
                e[c] += d
                dep[c] += d
                sed = 0.0
                break
            u = draws[i * cfg.max_steps + s] * sum(wt for _, wt in lower)
            for n, wt in lower:
                u -= wt
                if u <= 0.0:
                    break
            en = e[n]
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
        if sed > 0.0 and e[c] >= sea_level_m:
            e[c] += sed
            dep[c] += sed
    out = np.asarray(e)
    # Erosion never turns land into sea or sea into land.
    was_land = elev >= sea_level_m
    out[was_land] = np.maximum(out[was_land], sea_level_m)
    out[~was_land] = elev[~was_land]
    return out, np.asarray(dep)


# --- hydrology --------------------------------------------------------------------------------


def fill_depressions(geom: HexGeometry, elev: np.ndarray, sea_level_m: float, epsilon: float,
                     rng: np.random.Generator, random_factor: float) -> np.ndarray:
    """Priority-flood fill (Barnes et al. 2014, with epsilon) seeded from the sea, so every
    land cell has a strictly descending path to a sea cell.

    A raised cell gets ``epsilon * (1 + random_factor * U)`` above the cell that reached
    it, with U uniform per cell. With a constant epsilon, flats fill in concentric rings
    and the drainage across them runs in straight lines along the hex axes; random
    increments make the fill a random-weight shortest-path tree, so drainage across flats
    wanders while every step still drops by at least epsilon.
    """
    nbrs = geom.neighbor_lists
    filled = elev.astype(float).copy()
    sea = elev < sea_level_m
    if not sea.any():
        raise WorldGenerationError("no sea cells to drain to")
    rise = (epsilon * (1.0 + random_factor * rng.random(geom.n_cells))).tolist()
    closed = sea.copy()
    heap = [(float(filled[c]), int(c)) for c in np.flatnonzero(sea)]
    heapq.heapify(heap)
    f = filled.tolist()
    while heap:
        h, c = heapq.heappop(heap)
        for n in nbrs[c]:
            if closed[n]:
                continue
            closed[n] = True
            if f[n] <= h + epsilon:
                f[n] = h + rise[n]
            heapq.heappush(heap, (f[n], n))
    filled = np.asarray(f)
    land = ~sea
    filled[land] = np.maximum(filled[land], sea_level_m)
    return filled


def flow_receivers(geom: HexGeometry, elev: np.ndarray, sea_level_m: float) -> np.ndarray:
    """Steepest-descent downstream neighbour of each land cell (-1 for sea cells and pits)."""
    padded = np.append(elev, np.inf)
    nb = geom.neighbors
    nb_elev = padded[nb]  # off-map neighbours read +inf
    k = np.argmin(nb_elev, axis=1)
    lowest = nb_elev[np.arange(len(elev)), k]
    recv = nb[np.arange(len(elev)), k]
    return np.where((lowest < elev) & (elev >= sea_level_m), recv, -1)


def flow_directions(geom: HexGeometry, elev: np.ndarray, sea_level_m: float,
                    rng: np.random.Generator, exponent: float) -> np.ndarray:
    """Downstream neighbour of each land cell, drawn among its lower neighbours with
    probability proportional to drop ** ``exponent`` (-1 for sea cells and pits).

    Steepest descent on a hex grid sends every cell on a smooth slope along the hex axis
    nearest the gradient, so rivers run dead straight; a weighted draw lets them zigzag
    across such slopes while every step still goes downhill.
    """
    padded = np.append(elev, np.inf)
    nb = geom.neighbors
    drop = elev[:, None] - padded[nb]  # off-map neighbours: -inf
    weight = np.where(drop > 0, np.maximum(drop, 0.0) ** exponent, 0.0)
    total = weight.sum(axis=1)
    u = rng.random(len(elev)) * total
    k = np.argmax(np.cumsum(weight, axis=1) >= u[:, None], axis=1)
    recv = nb[np.arange(len(elev)), k]
    return np.where((total > 0) & (elev >= sea_level_m), recv, -1)


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


def runoff_m(rain: np.ndarray, temp: np.ndarray, cfg: RiverConfig) -> np.ndarray:
    et = cfg.et_base_mm + cfg.et_per_c_mm * np.maximum(temp, 0.0)
    return np.maximum(rain - et, 0.0) / 1000.0


# --- climate fields ----------------------------------------------------------------------------


def distance_to_sea_km(geom: HexGeometry, sea: np.ndarray, unit_km: float) -> np.ndarray:
    steps, _ = geom.distance_from(sea)
    steps = np.where(steps < 0, geom.width + geom.height, steps)
    return steps * SQRT3 * unit_km


def latitude_temperature(lat: np.ndarray, cfg: BaseTemperatureConfig) -> np.ndarray:
    return cfg.equator_c + (cfg.pole_c - cfg.equator_c) * (np.abs(lat) / 90.0) ** cfg.lat_exponent


def base_temperature(lat: np.ndarray, elev: np.ndarray, dist_sea_km: np.ndarray,
                     cfg: BaseTemperatureConfig) -> np.ndarray:
    """Latitude curve, continentality (interiors deviate further from a neutral
    temperature: hot interiors hotter, cold interiors colder), lapse rate on land above
    present sea level."""
    t_lat = latitude_temperature(lat, cfg)
    cont = 1.0 - np.exp(-dist_sea_km / cfg.continentality_scale_km)
    t = t_lat + cfg.continentality_amplitude * cont * (t_lat - cfg.continentality_neutral_c)
    return t - cfg.lapse_rate_c_per_km * np.maximum(elev, 0.0) / 1000.0


def _octaves(cfg: RainfallConfig) -> tuple[int, float, float]:
    return cfg.noise_octaves, cfg.noise_persistence, cfg.noise_lacunarity


def latitude_rain(lat: np.ndarray, cfg: RainfallConfig) -> np.ndarray:
    r = np.full(lat.shape, cfg.base_mm, dtype=float)
    for center, amp, width in cfg.bands:
        r += amp * np.exp(-(((lat - center) / width) ** 2))
    return r


def wind_vectors(lat: np.ndarray, angle_noise_deg: np.ndarray, cfg: RainfallConfig) -> np.ndarray:
    """Unit vector the air moves along, per cell: the band's direction rotated by noise."""
    uv = np.zeros((len(lat), 2))
    for lo, hi, u, v in cfg.winds:
        sel = (lat >= lo) & (lat < hi) if hi < 90.0 else (lat >= lo)
        uv[sel] = (u, v)
    a = np.deg2rad(angle_noise_deg)
    rot = np.stack([uv[:, 0] * np.cos(a) - uv[:, 1] * np.sin(a),
                    uv[:, 0] * np.sin(a) + uv[:, 1] * np.cos(a)], axis=1)
    return rot / np.linalg.norm(rot, axis=1, keepdims=True)


def neighbor_directions(geom: HexGeometry) -> np.ndarray:
    """(n_cells, 6, 2) unit vectors from each cell toward each neighbour slot, including
    slots off the map edge."""
    from deluge.world.grid import _EVEN_ROW_OFFSETS, _ODD_ROW_OFFSETS

    x, y = geom.coords
    out = np.zeros((geom.n_cells, 6, 2))
    for parity, offsets in ((0, _EVEN_ROW_OFFSETS), (1, _ODD_ROW_OFFSETS)):
        sel = (y % 2) == parity
        for k, (dx, dy) in enumerate(offsets):
            nx, ny = x[sel] + dx, y[sel] + dy
            px = SQRT3 * (nx + 0.5 * (ny % 2 == 0))
            py = 1.5 * ny
            out[sel, k, 0] = (px - geom.positions[sel, 0]) / SQRT3
            out[sel, k, 1] = (py - geom.positions[sel, 1]) / SQRT3
    return out


def base_rainfall(geom: HexGeometry, elev: np.ndarray, lat: np.ndarray, sea_level_m: float,
                  dist_sea_km: np.ndarray, unit_km: float, rng: np.random.Generator,
                  cfg: RainfallConfig) -> np.ndarray:
    """Rain from moisture carried by the prevailing winds over the hex grid.

    Wind bands follow latitude, but the band edges wobble and the wind direction turns
    with low-frequency noise. Air is fully moist over the sea and beyond the map edge;
    over land each cell takes moisture from its upwind neighbours (weighted by how well
    each lines up with the wind), loses a share per kilometre, rains more where it is
    forced uphill and is drained by that uplift, leaving a rain shadow downwind. The
    transport is solved by repeated sweeps until it settles. Rain then falls further with
    distance from any sea and gets a mild multiplicative noise.
    """
    uv = normalized_positions(geom)
    lat_w = lat + cfg.band_wobble_deg * noise.fbm(uv, rng, cfg.band_wobble_frequency, *_octaves(cfg))
    angle = cfg.wind_angle_noise_deg * noise.fbm(uv, rng, cfg.wind_noise_frequency, *_octaves(cfg))
    wind = wind_vectors(lat_w, angle, cfg)

    dirs = neighbor_directions(geom)  # cell -> neighbour
    align = np.maximum(0.0, -(dirs * wind[:, None, :]).sum(axis=2))  # neighbour upwind of cell
    weight = align**cfg.wind_sharpness
    weight /= weight.sum(axis=1, keepdims=True)

    nb = geom.neighbors
    valid = nb >= 0
    sea = elev < sea_level_m
    # Orographic lift responds to the large-scale relief, not to hummocks: rain and
    # shadows use a smoothed surface.
    h = np.where(sea, sea_level_m, elev)
    for _ in range(cfg.orographic_smoothing_passes):
        h = np.where(sea, sea_level_m, geom.neighbor_mean(h))
    h_nb = np.where(valid, h[np.where(valid, nb, 0)], sea_level_m)
    uplift_km = np.maximum(0.0, h[:, None] - h_nb) / 1000.0
    step_km = SQRT3 * unit_km
    keep = (1.0 - cfg.land_loss_per_1000km * step_km / 1000.0) * np.exp(-cfg.shadow_loss_per_km * uplift_km)

    m = np.full(geom.n_cells, cfg.upwind_moisture)
    for _ in range(cfg.iterations):
        m_nb = np.where(valid, m[np.where(valid, nb, 0)], cfg.upwind_moisture)
        new = np.maximum(cfg.min_moisture, (weight * m_nb * keep).sum(axis=1))
        new[sea] = cfg.upwind_moisture
        if np.allclose(new, m, atol=1e-6):
            m = new
            break
        m = new
    m_nb = np.where(valid, m[np.where(valid, nb, 0)], cfg.upwind_moisture)
    incoming = (weight * m_nb).sum(axis=1)
    lift = (weight * uplift_km).sum(axis=1)
    incoming[sea] = cfg.upwind_moisture
    rain = latitude_rain(lat_w, cfg) * incoming * (1.0 + cfg.orographic_gain_per_km * lift)
    cont = np.exp(-dist_sea_km / cfg.continentality_scale_km)
    rain *= (1.0 - cfg.continentality_weight) + cfg.continentality_weight * cont
    rain *= np.exp(cfg.rain_noise_amplitude * noise.fbm(uv, rng, cfg.rain_noise_frequency, *_octaves(cfg)))
    for _ in range(cfg.smoothing_passes):
        rain = geom.neighbor_mean(rain)
    return rain


# --- terrain and resources -------------------------------------------------------------------


def classify_terrain(elev: np.ndarray, temp: np.ndarray, rain: np.ndarray, flow: np.ndarray,
                     sea_level_m: float, cfg: TerrainConfig) -> np.ndarray:
    """Whittaker-style climate biome, overridden by cold, wetness and relief."""
    sea = elev < sea_level_m
    t_pos = np.maximum(temp, 0.0)
    terrain = np.full(elev.shape, int(Terrain.FOREST), dtype=np.int8)
    terrain[rain < cfg.forest_rain_mm + cfg.forest_rain_per_c * t_pos] = Terrain.PLAIN
    terrain[rain < cfg.steppe_rain_mm + cfg.steppe_rain_per_c * t_pos] = Terrain.STEPPE
    terrain[rain < cfg.desert_rain_mm + cfg.desert_rain_per_c * t_pos] = Terrain.DESERT
    terrain[temp < cfg.tundra_c] = Terrain.TUNDRA
    wet = ((elev - sea_level_m <= cfg.wetland_max_m_above_sea) & (temp >= cfg.tundra_c)
           & ((flow >= cfg.wetland_min_flow) | (rain >= cfg.wetland_min_rain_mm)))
    terrain[wet] = Terrain.WETLAND
    terrain[elev >= cfg.hills_m] = Terrain.HILLS
    terrain[elev >= cfg.mountain_m] = Terrain.MOUNTAIN
    terrain[temp < cfg.ice_c] = Terrain.ICE
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
    f[(terrain == Terrain.SEA) | (terrain == Terrain.ICE)] = 0.0
    return np.clip(f, 0.0, 1.0)


def wild_food_field(geom: HexGeometry, terrain: np.ndarray, flow: np.ndarray, temp: np.ndarray,
                    cfg: WildFoodConfig) -> np.ndarray:
    """Static forage richness. The coast bonus is applied yearly by the capacity model,
    because the coast moves with the sea."""
    w = terrain_table(cfg.base)[terrain]
    land = terrain != Terrain.SEA
    forest = terrain == Terrain.FOREST
    open_land = land & ~forest
    edge = (forest & geom.neighbor_any(open_land)) | (open_land & geom.neighbor_any(forest))
    w = w + cfg.forest_edge_bonus * edge + cfg.river_bonus * (flow > 0)
    w = w * np.maximum(cfg.cold_floor, ramp(temp, cfg.cold_zero_c, cfg.cold_full_c))
    w[~land | (terrain == Terrain.ICE)] = 0.0
    return np.clip(w, 0.0, 1.0)


# --- pipeline -------------------------------------------------------------------------------


def world_unit_km(cfg: Config) -> float:
    return hex_unit_km(cfg.grid.height, cfg.world.lat_south, cfg.world.lat_north, cfg.world.km_per_degree)


def generate_terrain(geom: HexGeometry, cfg: Config, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray, Plates | None, tuple[int, ...], int, dict[str, int]]:
    """Draw worlds until the geography rules hold.

    Returns (elevation, deposit, plates, constructed bridge cells, attempts, rejections
    by rule)."""
    w = cfg.world
    unit_km = world_unit_km(cfg)
    # The whole preset, not just the run's years: the world must not depend on run length.
    curve = sea_level_curve(cfg.climate.sea_level, preset_horizon(cfg.climate.sea_level))
    sea0 = float(curve[0])
    pulses = geography.pulse_windows(curve, w.land_bridges.pulse_rate_factor)
    rejected = {"seas": 0, "landmass_count": 0, "land_bridge": 0, "landmass_rules": 0}
    for attempt in range(1, w.max_attempts + 1):
        plates = tectonics.make_plates(geom, rng, w.tectonics) if w.tectonics.enabled else None
        z = normalized_relief(continent_relief(geom, rng, w.shape, plates), w.land_share)
        if plates is not None:
            # Tectonic relief is in normalized units, so ranges reach similar heights in
            # every world regardless of how rugged the base relief is.
            z = z + tectonics.tectonic_relief(geom, plates, rng, w.tectonics, unit_km)
        elev = hypsometric_elevation(z, w.land_share, w.reference_sea_level_m, w.elevation)
        elev = add_detail(geom, elev, w.reference_sea_level_m, rng, w.elevation)
        if not geography.sea_basin_report(geom, elev, sea0, w.seas).ok:
            rejected["seas"] += 1
            continue
        # Cheap screen before erosion: erosion keeps land and sea at year 0 as they are,
        # and a constructed bridge can join two landmasses into one.
        pre = geography.landmass_report(geom, elev, sea0, w.landmasses)
        if not (w.landmasses.min_count <= len(pre.landmasses) <= w.landmasses.max_count + 1):
            rejected["landmass_count"] += 1
            continue
        elev, deposit = erode(geom, elev, sea0, rng, w.erosion)
        elev = fill_depressions(geom, elev, sea0, w.fill_epsilon_m, rng, w.fill_random_factor)

        floor = w.land_bridges.min_share_of_land * (elev >= sea0).sum()
        bridges = [b for b in geography.land_bridges(geom, elev, curve, floor, pulses) if b.pulse >= 0]
        built: tuple[int, ...] = ()
        if len(bridges) < w.land_bridges.min_count and w.land_bridges.construct_if_missing:
            elev, built = geography.construct_bridge(geom, elev, curve, pulses, floor, w.land_bridges, rng)
            if built:
                elev = fill_depressions(geom, elev, sea0, w.fill_epsilon_m, rng, w.fill_random_factor)
                bridges = [b for b in geography.land_bridges(geom, elev, curve, floor, pulses) if b.pulse >= 0]
        if len(bridges) < w.land_bridges.min_count:
            rejected["land_bridge"] += 1
            continue
        if not geography.landmass_report(geom, elev, sea0, w.landmasses).ok:
            rejected["landmass_rules"] += 1
            continue
        return elev, deposit, plates, built, attempt, rejected
    raise WorldGenerationError(
        f"no world met the geography rules in {w.max_attempts} attempts; check world.landmasses, "
        "world.seas and world.land_bridges"
    )


def generate_world_with_report(cfg: Config, rng: np.random.Generator) -> tuple[WorldData, GenerationReport]:
    w = cfg.world
    geom = HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)
    unit_km = world_unit_km(cfg)
    sea0 = float(sea_level_curve(cfg.climate.sea_level, 0)[0])

    elev, deposit, plates, built, attempts, rejected = generate_terrain(geom, cfg, rng)
    sea = elev < sea0
    lat = geom.latitude(w.lat_south, w.lat_north)
    dist_km = distance_to_sea_km(geom, sea, unit_km)
    temp = base_temperature(lat, elev, dist_km, w.temperature)
    rain = base_rainfall(geom, elev, lat, sea0, dist_km, unit_km, rng, w.rainfall)

    receivers = flow_directions(geom, elev, sea0, rng, w.rivers.direction_exponent)
    acc = flow_accumulation(elev, receivers, runoff_m(rain, temp, w.rivers), sea0)
    river = np.where(~sea & (acc >= w.rivers.min_flow), acc, 0.0)

    terrain = classify_terrain(elev, temp, rain, acc, sea0, w.terrain)
    fertility = fertility_field(terrain, river, deposit, temp, w.fertility)
    wild_food = wild_food_field(geom, terrain, river, temp, w.wild_food)

    shape = geom.shape
    world = WorldData(
        width=geom.width,
        height=geom.height,
        elevation=elev.reshape(shape),
        terrain=terrain.reshape(shape),
        river=river.reshape(shape),
        flow_to=receivers.reshape(shape),
        base_temp=temp.reshape(shape),
        base_rain=rain.reshape(shape),
        fertility=fertility.reshape(shape),
        wild_food=wild_food.reshape(shape),
    )
    none = np.zeros(geom.n_cells, dtype=bool)
    report = GenerationReport(
        attempts=attempts,
        rejected=rejected,
        plate_id=plates.plate_id if plates is not None else np.full(geom.n_cells, -1),
        convergent=plates.convergent if plates is not None else none,
        divergent=plates.divergent if plates is not None else none,
        deposit=deposit,
        constructed_bridge=built,
    )
    return world, report


def generate_world(cfg: Config, rng: np.random.Generator) -> WorldData:
    return generate_world_with_report(cfg, rng)[0]
