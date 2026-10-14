"""World generation: each pipeline stage with fixed seeds, plus the acceptance checks of the
reworked generator (geography rules, tectonic ranges, continuous hypsometry, drainage,
irregular climate)."""

import dataclasses

import numpy as np
import pytest

from conftest import make_world
from deluge.config import load_config
from deluge.rng import RngStreams
from deluge.world import generate_world, geography, procedural as P, tectonics
from deluge.world.climate import sea_level_curve
from deluge.world.grid import HexGeometry
from deluge.world.worlddata import Terrain

SEA0 = -60.0


def flat(world, name):
    return getattr(world, name).ravel()


# --- determinism ----------------------------------------------------------------------------


def test_same_seed_same_world(cfg, worlds):
    again = make_world(cfg, 1)
    for name, arr in worlds[1].fields().items():
        np.testing.assert_array_equal(arr, getattr(again, name), err_msg=name)


def test_different_seeds_differ(worlds):
    assert not np.array_equal(worlds[1].elevation, worlds[2].elevation)


def test_world_fields_are_read_only(worlds):
    with pytest.raises(ValueError):
        worlds[1].elevation[0, 0] = 1.0


def test_world_independent_of_run_length(cfg, worlds):
    short = cfg.with_overrides({"run": {"years": 3}})
    np.testing.assert_array_equal(make_world(short, 3).elevation, worlds[3].elevation)


def test_world_independent_of_other_streams(cfg):
    more = cfg.with_overrides({"rng": {"streams": list(cfg.rng.streams) + ["extra"]}})
    a = P.generate_world(cfg, RngStreams(3, cfg.rng.streams)["world"])
    b = P.generate_world(more, RngStreams(3, more.rng.streams)["world"])
    np.testing.assert_array_equal(a.elevation, b.elevation)


# --- relief and elevation --------------------------------------------------------------------


def test_normalized_relief_anchors(cfg):
    r = np.random.default_rng(0).normal(size=5000)
    z = P.normalized_relief(r, 0.35)
    land = z >= 0
    assert land.sum() == round(0.35 * 5000)
    assert np.median(z[land]) == pytest.approx(1.0, abs=1e-3)
    assert np.median(z[~land]) == pytest.approx(-1.0, abs=1e-3)
    order = np.argsort(r)
    assert np.all(np.diff(z[order]) >= 0)


def test_hypsometric_elevation_is_continuous_and_monotone(cfg):
    z = np.linspace(-4, 8, 2001)
    e = P.hypsometric_elevation(z, 0.35, SEA0, cfg.world.elevation)
    assert (e >= SEA0).sum() == round(0.35 * len(z))
    assert np.all(np.diff(e) > 0)  # strictly increasing: no flats
    assert np.max(np.diff(e)) < 200  # no jumps


def test_knots_must_start_at_reference(cfg):
    bad = dataclasses.replace(cfg.world.elevation, land_knots=((0.0, 0.0), (1.0, 100.0)))
    with pytest.raises(ValueError):
        P.hypsometric_elevation(np.zeros(10), 0.35, SEA0, bad)


def test_land_share_near_target(worlds, cfg):
    for seed, w in worlds.items():
        share = (flat(w, "elevation") >= SEA0).mean()
        assert abs(share - cfg.world.land_share) < 0.01, seed


def test_hypsometry(worlds):
    """Most land is low (0-500 m), with a tail of highlands; shelves are wide."""
    tails = 0
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        land = e[e >= 0]
        assert (land < 500).mean() > 0.55, seed
        tails += e.max() > 2500
        sea = e < SEA0
        shelf = sea & (e >= -120)
        assert shelf.sum() > 0.05 * sea.sum(), seed
    assert tails >= len(worlds) - 1


def test_coastal_plains_drown_in_broad_areas(worlds, geom, cfg):
    """Land lost by year 3000 is not a one-cell ring: much of it lies two or more steps
    from the year-3000 coast."""
    sl = sea_level_curve(cfg.climate.sea_level, 3000)
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        drowned = (e >= sl[0]) & (e < sl[3000])
        steps, _ = geom.distance_from(e >= sl[3000])
        assert (steps[drowned] >= 2).mean() > 0.25, seed
        assert drowned.sum() > 0.1 * (e >= sl[0]).sum(), seed


def test_erosion_carves_and_keeps_coasts(cfg, geom):
    z = P.normalized_relief(P.continent_relief(geom, np.random.default_rng(2), cfg.world.shape, None), 0.35)
    elev = P.hypsometric_elevation(z, 0.35, SEA0, cfg.world.elevation)
    a, dep = P.erode(geom, elev, SEA0, np.random.default_rng(3), cfg.world.erosion)
    b, _ = P.erode(geom, elev, SEA0, np.random.default_rng(3), cfg.world.erosion)
    np.testing.assert_array_equal(a, b)
    land = elev >= SEA0
    np.testing.assert_array_equal(a >= SEA0, land)
    np.testing.assert_array_equal(a[~land], elev[~land])
    assert (a[land] < elev[land]).sum() > 200
    assert (dep >= 0).all() and dep.max() > 0


# --- tectonics -----------------------------------------------------------------------------------


def test_plates_and_boundaries(cfg, geom):
    t = cfg.world.tectonics
    plates = tectonics.make_plates(geom, np.random.default_rng(4), t)
    n = len(plates.velocity)
    assert t.plates_min <= n <= t.plates_max
    assert set(np.unique(plates.plate_id)) <= set(range(n))
    assert plates.convergent.any() and plates.divergent.any()
    assert not (plates.convergent & plates.divergent).any()
    relief = tectonics.tectonic_relief(geom, plates, np.random.default_rng(5), t, 100.0)
    assert relief[plates.convergent].mean() > 0.5 * t.mountain_height * 0.2
    assert relief[plates.divergent].mean() < 0


def test_mountains_follow_convergent_boundaries(generated, geom, cfg):
    near = []
    coastal_range_worlds = 0
    for seed, (w, rep) in generated.items():
        high = np.isin(flat(w, "terrain"), [Terrain.MOUNTAIN, Terrain.HILLS, Terrain.ICE]) & (
            flat(w, "elevation") >= cfg.world.terrain.hills_m)
        steps, _ = geom.distance_from(rep.convergent)
        near.append((steps[high] <= 4).mean())
        sea = flat(w, "elevation") < SEA0
        sea_steps, _ = geom.distance_from(sea)
        coastal_range_worlds += bool((high & (sea_steps <= 2)).any())
        # More than one range: high ground splits into several separate areas.
        labels = geom.components(high)
        sizes = np.bincount(labels[labels >= 0])
        assert (sizes >= 5).sum() >= 2, seed
    assert np.mean(near) > 0.6, near
    assert coastal_range_worlds >= len(generated) - 1


def test_tectonics_can_be_switched_off(cfg):
    off = cfg.with_overrides({"world": {
        "tectonics": {"enabled": False},
        "seas": {"required": False},
        "land_bridges": {"min_count": 0},
        "landmasses": {"min_count": 1, "max_count": 50, "min_islands": 0, "min_size_ratio": 1.0},
    }})
    w, rep = make_world(off, 1, with_report=True)
    assert (rep.plate_id == -1).all()
    assert abs((flat(w, "elevation") >= SEA0).mean() - cfg.world.land_share) < 0.01


def test_earth_source_is_a_stub(cfg):
    earth = cfg.with_overrides({"world": {"source": "earth"}})
    with pytest.raises(NotImplementedError):
        generate_world(earth, np.random.default_rng(0))


def test_too_strict_rules_raise(cfg):
    strict = cfg.with_overrides({"world": {"max_attempts": 2,
                                           "landmasses": {"min_count": 40, "max_count": 40}}})
    with pytest.raises(P.WorldGenerationError):
        make_world(strict, 1)


# --- geography rules --------------------------------------------------------------------------


def test_geography_rules_hold(worlds, geom, cfg):
    w_cfg = cfg.world
    curve = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
    pulses = geography.pulse_windows(curve, w_cfg.land_bridges.pulse_rate_factor)
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        lm = geography.landmass_report(geom, e, SEA0, w_cfg.landmasses)
        assert lm.ok, (seed, lm.landmasses, lm.islands)
        assert w_cfg.landmasses.min_count <= len(lm.landmasses) <= w_cfg.landmasses.max_count
        sb = geography.sea_basin_report(geom, e, SEA0, w_cfg.seas)
        assert sb.inland_seas or sb.enclosed_basins, seed
        floor = w_cfg.land_bridges.min_share_of_land * (e >= SEA0).sum()
        in_pulse = [b for b in geography.land_bridges(geom, e, curve, floor, pulses) if b.pulse >= 0]
        assert in_pulse, seed
        for b in in_pulse:
            p = pulses[b.pulse]
            assert p.start_year <= b.year_drowned < p.end_year
            assert curve[b.year_drowned - 1] <= b.crest_m < curve[b.year_drowned]
            # Once the crest is under water its land neighbours lie in separate land areas.
            above = geom.components(e > b.crest_m)
            sides = {int(above[n]) for n in geom.neighbor_lists[b.crest_cell] if above[n] >= 0}
            assert len(sides) >= 2, (seed, b)


def test_pulse_windows(cfg):
    curve = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
    pulses = geography.pulse_windows(curve, cfg.world.land_bridges.pulse_rate_factor)
    assert [(p.start_year, p.end_year) for p in pulses] == [(500, 700), (1500, 1750)]


def test_bridge_detection_on_a_synthetic_isthmus():
    geom = HexGeometry(20, 7)
    x, y = geom.coords
    e = np.full(geom.n_cells, -500.0)
    e[(x >= 2) & (x <= 7) & (y >= 1) & (y <= 5)] = 100.0
    e[(x >= 12) & (x <= 17) & (y >= 1) & (y <= 5)] = 100.0
    e[(x >= 8) & (x <= 11) & (y == 3)] = -30.0  # isthmus crest at -30 m
    curve = np.linspace(-60.0, 0.0, 601)
    pulses = geography.pulse_windows(np.concatenate([np.full(10, -60.0), curve]), 2.0)
    bridges = geography.land_bridges(geom, e, curve, 10, ())
    assert len(bridges) == 1
    assert bridges[0].crest_m == -30.0
    assert curve[bridges[0].year_drowned] > -30.0 >= curve[bridges[0].year_drowned - 1]
    assert pulses == ()  # a steady rise has no pulses


# --- hydrology ------------------------------------------------------------------------------


def test_filled_surface_drains_everywhere(cfg, geom):
    z = P.normalized_relief(P.continent_relief(geom, np.random.default_rng(4), cfg.world.shape, None), 0.35)
    elev = P.hypsometric_elevation(z, 0.35, SEA0, cfg.world.elevation)
    filled = P.fill_depressions(geom, elev, SEA0, cfg.world.fill_epsilon_m, np.random.default_rng(1),
                                cfg.world.fill_random_factor)
    assert (filled >= elev).all()
    np.testing.assert_array_equal(filled[elev < SEA0], elev[elev < SEA0])
    recv = P.flow_directions(geom, filled, SEA0, np.random.default_rng(2), cfg.world.rivers.direction_exponent)
    land = filled >= SEA0
    assert (recv[land] >= 0).all()
    assert (filled[recv[land]] < filled[land]).all()


def test_flats_drain_without_straight_rows(cfg):
    """A dead-flat plain with the sea along one side: every cell drains, and the paths
    across it wander instead of running straight along the rows."""
    geom = HexGeometry(40, 24)
    x, _ = geom.coords
    elev = np.where(x == 0, -100.0, 0.0)
    filled = P.fill_depressions(geom, elev, SEA0, cfg.world.fill_epsilon_m, np.random.default_rng(0),
                                cfg.world.fill_random_factor)
    recv = P.flow_directions(geom, filled, SEA0, np.random.default_rng(1), cfg.world.rivers.direction_exponent)
    land = filled >= SEA0
    assert (recv[land] >= 0).all()
    flow = P.flow_accumulation(filled, recv, np.ones(geom.n_cells), SEA0)
    longest, share = geography.row_aligned_runs(geom, flow, recv)
    # Measured with the default config: longest 7-10, share 3-5% (constant epsilon: > 20).
    assert share < 0.08 and longest <= 12, (longest, share)
    # Constant-epsilon filling would have run every path straight to the sea.
    ring = P.fill_depressions(geom, elev, SEA0, cfg.world.fill_epsilon_m, np.random.default_rng(0), 0.0)
    steep = P.flow_receivers(geom, ring, SEA0)
    assert geography.row_aligned_runs(geom, flow, steep)[0] > 20


def test_rivers_flow_downhill_to_the_sea(worlds, geom):
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        river = flat(w, "river")
        recv = flat(w, "flow_to")
        land = e >= SEA0
        assert (river[~land] == 0).all()
        assert (recv[~land] == -1).all() and (recv[land] >= 0).all()
        assert (e[recv[land]] < e[land]).all()
        for c in np.flatnonzero(river > 0):
            n = 0
            while e[c] >= SEA0:
                c = recv[c]
                n += 1
                assert n < geom.n_cells
        down = recv[river > 0]
        on_land = e[down] >= SEA0
        assert (river[down[on_land]] >= river[river > 0][on_land]).all()


def test_rivers_are_long_branched_and_not_row_aligned(worlds, geom):
    long_rivers = []
    for seed, w in worlds.items():
        e, river, recv = flat(w, "elevation"), flat(w, "river"), flat(w, "flow_to")
        lengths = geography.river_lengths(e, river, recv)
        long_rivers.append(sum(1 for n in lengths.values() if n > 15))
        is_river = river > 0
        tributaries = np.bincount(recv[is_river & (recv >= 0)], minlength=len(e))
        assert (tributaries[is_river] >= 2).sum() > 20, seed  # confluences
        longest, share = geography.row_aligned_runs(geom, river, recv)
        # Measured on seeds 1-10: share 0.4-2.7%, longest 5-10 (steepest descent: up to 3%, 11).
        assert share < 0.03, (seed, share)
        assert longest <= 12, (seed, longest)
    assert min(long_rivers) >= 1, long_rivers


# --- climate fields ----------------------------------------------------------------------


def test_temperature_lapse_latitude_and_continentality(cfg):
    t = cfg.world.temperature
    lat = np.array([0.0, 0.0, 30.0, 60.0, -60.0, 60.0, 5.0, 5.0])
    elev = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 3000.0, 0.0, 3000.0])
    temp = P.base_temperature(lat, elev, dist, t)
    assert temp[0] - temp[1] == pytest.approx(t.lapse_rate_c_per_km)
    assert temp[0] > temp[2] > temp[3]
    assert temp[3] == pytest.approx(temp[4])
    assert temp[5] < temp[3]  # cold interior colder
    assert temp[7] > temp[6]  # hot interior hotter


def _plain_rain_cfg(cfg):
    """Rainfall config with noise, smoothing and continentality off, so the synthetic
    tests see the transport mechanism alone."""
    return dataclasses.replace(cfg.world.rainfall, smoothing_passes=0, band_wobble_deg=0.0,
                               wind_angle_noise_deg=0.0, rain_noise_amplitude=0.0,
                               continentality_weight=0.0, orographic_smoothing_passes=0)


def test_rain_shadow_behind_a_ridge(cfg):
    """Westerlies cross a ridge: the windward slope gets more rain than the flats before
    it, and the lee gets less than the windward side and the flats."""
    rcfg = _plain_rain_cfg(cfg)
    geom = HexGeometry(40, 8)
    x, _ = geom.coords
    elev = np.where(x < 5, -1000.0, 100.0)
    elev = np.where((x >= 15) & (x <= 17), 3000.0, elev)
    lat = np.full(geom.n_cells, 45.0)  # westerlies blow toward +x
    dist = np.zeros(geom.n_cells)
    rain = P.base_rainfall(geom, elev, lat, SEA0, dist, 100.0, np.random.default_rng(0), rcfg)
    flats = rain[(x >= 8) & (x <= 12)].mean()
    windward = rain[x == 15].mean()
    lee = rain[(x >= 19) & (x <= 23)].mean()
    assert windward > flats
    assert lee < 0.6 * windward
    assert lee < flats


def test_rain_falls_inland(cfg):
    rcfg = dataclasses.replace(_plain_rain_cfg(cfg), continentality_weight=cfg.world.rainfall.continentality_weight)
    geom = HexGeometry(60, 8)
    x, _ = geom.coords
    elev = np.where((x < 5) | (x > 55), -1000.0, 50.0)
    lat = np.full(geom.n_cells, 45.0)
    dist = P.distance_to_sea_km(geom, elev < SEA0, 100.0)
    rain = P.base_rainfall(geom, elev, lat, SEA0, dist, 100.0, np.random.default_rng(0), rcfg)
    assert rain[x == 6].mean() > rain[x == 30].mean()


def test_rain_is_not_striped(worlds, geom, cfg):
    """Latitude alone explains well under all of the rain pattern on land."""
    lat_row = geom.coords[1]
    for seed, w in worlds.items():
        rain = flat(w, "base_rain")
        land = flat(w, "elevation") >= SEA0
        r, rows = rain[land], lat_row[land]
        row_mean = np.bincount(rows, weights=r) / np.maximum(np.bincount(rows), 1)
        explained = 1.0 - ((r - row_mean[rows]) ** 2).sum() / ((r - r.mean()) ** 2).sum()
        assert explained < 0.6, (seed, explained)


def test_biome_bands(worlds, geom, cfg):
    lat = geom.latitude(cfg.world.lat_south, cfg.world.lat_north)
    desert = {"eq": 0, "sub": 0}
    forest = {"eq": 0, "sub": 0}
    total = {"eq": 0, "sub": 0}
    for w in worlds.values():
        t = flat(w, "terrain")
        land = t != Terrain.SEA
        for band, sel in (("eq", np.abs(lat) < 8), ("sub", (np.abs(lat) > 18) & (np.abs(lat) < 35))):
            total[band] += (land & sel).sum()
            desert[band] += ((t == Terrain.DESERT) & sel).sum()
            forest[band] += ((t == Terrain.FOREST) & sel).sum()
        temp = flat(w, "base_temp")
        low = land & (flat(w, "elevation") < 300)
        assert temp[low & (np.abs(lat) < 10)].mean() > temp[low & (np.abs(lat) > 45)].mean() + 15
    assert desert["sub"] / total["sub"] > 1.5 * desert["eq"] / total["eq"]
    assert forest["eq"] / total["eq"] > forest["sub"] / total["sub"]


# --- terrain and resources ---------------------------------------------------------------------


def test_terrain_classes(worlds, cfg):
    tc = cfg.world.terrain
    for seed, w in worlds.items():
        t = flat(w, "terrain")
        e = flat(w, "elevation")
        temp = flat(w, "base_temp")
        assert set(np.unique(t)) <= {int(v) for v in Terrain}
        np.testing.assert_array_equal(t == Terrain.SEA, e < SEA0)
        land = e >= SEA0
        np.testing.assert_array_equal(t[land] == Terrain.ICE, temp[land] < tc.ice_c)
        not_ice = land & (temp >= tc.ice_c)
        assert (t[not_ice & (e >= tc.mountain_m)] == Terrain.MOUNTAIN).all()
        cold_low = not_ice & (temp < tc.tundra_c) & (e < tc.hills_m)
        assert (t[cold_low] == Terrain.TUNDRA).all()


def test_fertility_and_wild_food_ranges(worlds):
    for seed, w in worlds.items():
        t = flat(w, "terrain")
        for name in ("fertility", "wild_food"):
            v = flat(w, name)
            assert v.min() >= 0 and v.max() <= 1, (seed, name)
            assert (v[(t == Terrain.SEA) | (t == Terrain.ICE)] == 0).all(), (seed, name)
        assert (flat(w, "fertility")[t == Terrain.TUNDRA] < 0.2).all()


def test_floodplains_are_fertile(worlds):
    river_f, dry_f = [], []
    for w in worlds.values():
        t = flat(w, "terrain")
        f = flat(w, "fertility")
        r = flat(w, "river")
        major = r >= 10.0  # floodplains of substantial rivers, not every small stream
        lowland = np.isin(t, [Terrain.PLAIN, Terrain.FOREST, Terrain.STEPPE])
        river_f.append(f[lowland & major])
        dry_f.append(f[lowland & (r == 0)])
    assert np.concatenate(river_f).mean() > np.concatenate(dry_f).mean() + 0.1
