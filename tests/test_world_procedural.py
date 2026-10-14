"""World generation: each pipeline stage with fixed seeds, plus Round 1 acceptance checks
(continents, mountain ranges, rivers flowing downhill to the sea, biome bands, land share)."""

import dataclasses

import numpy as np
import pytest

from conftest import WORLD_SEEDS, make_world
from deluge.config import load_config
from deluge.rng import RngStreams
from deluge.world import generate_world, procedural as P
from deluge.world.grid import HexGeometry
from deluge.world.worlddata import Terrain

SEA0 = -60.0


def flat(world, name):
    return getattr(world, name).ravel()


# --- determinism ----------------------------------------------------------------------------


def test_same_seed_same_world(cfg, worlds):
    again = make_world(cfg, 0)
    for name, arr in worlds[0].fields().items():
        np.testing.assert_array_equal(arr, getattr(again, name), err_msg=name)


def test_different_seeds_differ(worlds):
    assert not np.array_equal(worlds[0].elevation, worlds[1].elevation)


def test_world_fields_are_read_only(worlds):
    with pytest.raises(ValueError):
        worlds[0].elevation[0, 0] = 1.0


# --- elevation stages ------------------------------------------------------------------------


def test_raw_elevation_deterministic(cfg, geom):
    a = P.raw_elevation(geom, np.random.default_rng(5), cfg.world.noise, cfg.world.continents)
    b = P.raw_elevation(geom, np.random.default_rng(5), cfg.world.noise, cfg.world.continents)
    np.testing.assert_array_equal(a, b)


def test_calibrate_hits_land_and_shelf_share(cfg, geom):
    raw = np.random.default_rng(1).normal(size=geom.n_cells)
    elev = P.calibrate_elevation(raw, 0.35, SEA0, cfg.world.elevation)
    land = elev >= SEA0
    assert land.sum() == round(0.35 * geom.n_cells)
    shelf = land & (elev < 0.0)
    assert shelf.sum() == round(cfg.world.elevation.shelf_share_of_land * land.sum())
    assert elev.max() == pytest.approx(cfg.world.elevation.land_max_m)
    assert elev.min() == pytest.approx(cfg.world.elevation.ocean_min_m)
    # Rank-preserving: higher relief never ends lower.
    order = np.argsort(raw)
    assert np.all(np.diff(elev[order]) >= 0)


def test_erosion_carves_and_deposits(cfg, geom):
    raw = P.raw_elevation(geom, np.random.default_rng(2), cfg.world.noise, cfg.world.continents)
    elev = P.calibrate_elevation(raw, 0.35, SEA0, cfg.world.elevation)
    eroded, dep = P.erode(geom, elev, SEA0, np.random.default_rng(3), cfg.world.erosion)
    again, dep2 = P.erode(geom, elev, SEA0, np.random.default_rng(3), cfg.world.erosion)
    np.testing.assert_array_equal(eroded, again)
    assert (dep >= 0).all() and dep.max() > 0
    land = elev >= SEA0
    assert (eroded[land] < elev[land]).sum() > 100  # valleys were cut
    # Mass is removed or moved, never created (sediment reaching the sea is lost).
    assert eroded.sum() <= elev.sum() + 1e-6
    np.testing.assert_array_equal(eroded[~land], elev[~land])


def test_land_share_near_target(worlds, cfg):
    for seed, w in worlds.items():
        share = (flat(w, "elevation") >= SEA0).mean()
        assert abs(share - cfg.world.land_share) < 0.005, seed


def test_two_to_four_uneven_landmasses(worlds, cfg, geom):
    rules = cfg.world.landmasses
    for seed, w in worlds.items():
        rep = P.landmass_report(geom, flat(w, "elevation"), SEA0, rules)
        assert rules.min_count <= len(rep.landmasses) <= rules.max_count, (seed, rep.sizes)
        assert rep.landmasses[0] / rep.landmasses[-1] >= rules.min_size_ratio, (seed, rep.sizes)


def test_shelf_floods_by_present_sea_level(worlds, cfg):
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        land0 = e >= SEA0
        shelf = land0 & (e < 0.0)
        assert abs(shelf.sum() / land0.sum() - cfg.world.elevation.shelf_share_of_land) < 0.02, seed


def test_tectonics_switch_not_implemented(cfg):
    on = cfg.with_overrides({"world": {"tectonics": {"enabled": True}}})
    with pytest.raises(NotImplementedError):
        generate_world(on, np.random.default_rng(0))


def test_earth_source_is_a_stub(cfg):
    earth = cfg.with_overrides({"world": {"source": "earth"}})
    with pytest.raises(NotImplementedError):
        generate_world(earth, np.random.default_rng(0))


def test_too_strict_landmass_rules_raise(cfg):
    strict = cfg.with_overrides({"world": {"max_attempts": 2,
                                           "landmasses": {"min_count": 9, "max_count": 9}}})
    with pytest.raises(P.WorldGenerationError):
        make_world(strict, 0)


# --- mountains ----------------------------------------------------------------------------------


def test_mountains_form_ranges_not_single_peaks(worlds, geom, cfg):
    """Mountain cells cluster into connected ranges, and ranges are elongated."""
    for seed, w in worlds.items():
        mountain = flat(w, "terrain") == Terrain.MOUNTAIN
        labels = geom.components(mountain)
        sizes = np.bincount(labels[labels >= 0])
        assert sizes.max() >= 15, seed
        # Elongation of the largest range: ratio of principal axes of its cell positions.
        pts = geom.positions[labels == 0]
        ev = np.sort(np.linalg.eigvalsh(np.cov(pts.T)))
        assert ev[1] / max(ev[0], 1e-9) > 1.5, (seed, ev)


# --- hydrology ------------------------------------------------------------------------------


def test_filled_surface_drains_everywhere(cfg, geom):
    raw = P.raw_elevation(geom, np.random.default_rng(4), cfg.world.noise, cfg.world.continents)
    elev = P.calibrate_elevation(raw, 0.35, SEA0, cfg.world.elevation)
    filled = P.fill_depressions(geom, elev, SEA0, cfg.world.fill_epsilon_m)
    assert (filled >= elev).all()
    np.testing.assert_array_equal(filled[elev < SEA0], elev[elev < SEA0])
    recv = P.flow_receivers(geom, filled, SEA0)
    land = filled >= SEA0
    assert (recv[land] >= 0).all()
    assert (filled[recv[land]] < filled[land]).all()


def test_rivers_flow_downhill_to_the_sea(worlds, geom):
    for seed, w in worlds.items():
        e = flat(w, "elevation")
        river = flat(w, "river")
        recv = P.flow_receivers(geom, e, SEA0)
        land = e >= SEA0
        assert (river[~land] == 0).all()
        cells = np.flatnonzero(river > 0)
        assert len(cells) >= 0.02 * land.sum(), seed
        for c in cells:
            path = 0
            while e[c] >= SEA0:
                n = recv[c]
                assert n >= 0, (seed, c)
                assert e[n] < e[c]
                c = n
                path += 1
                assert path < geom.n_cells
        # Flow never shrinks downstream.
        down = recv[cells]
        on_land = e[down] >= SEA0
        assert (river[down[on_land]] >= river[cells[on_land]]).all()


def test_rivers_form_networks(worlds, geom):
    """Rivers join into drainage networks: in a typical world the longest river gathers
    ten or more river cells upstream; even dry worlds have rivers of several cells."""
    longest = []
    for seed, w in worlds.items():
        river = flat(w, "river")
        e = flat(w, "elevation")
        recv = P.flow_receivers(geom, e, SEA0)
        cells = set(np.flatnonzero(river > 0).tolist())
        upstream = {c: 0 for c in cells}
        for c in sorted(cells, key=lambda c: -e[c]):
            if recv[c] in upstream:
                upstream[recv[c]] += upstream[c] + 1
        longest.append(max(upstream.values()))
    assert np.median(longest) >= 10, longest
    assert min(longest) >= 5, longest


# --- climate fields ----------------------------------------------------------------------


def test_temperature_lapse_and_latitude(cfg):
    t = cfg.world.temperature
    lat = np.array([0.0, 0.0, 30.0, 60.0, -60.0])
    elev = np.array([0.0, 1000.0, 0.0, 0.0, 0.0])
    temp = P.base_temperature(lat, elev, t)
    assert temp[0] - temp[1] == pytest.approx(t.lapse_rate_c_per_km)
    assert temp[0] > temp[2] > temp[3]
    assert temp[3] == pytest.approx(temp[4])


def test_rain_shadow_behind_a_ridge(cfg):
    """Westerlies cross a ridge: the windward slope gets more rain than the flats before
    it, and the lee side gets less than the windward side."""
    rcfg = dataclasses.replace(cfg.world.rainfall, smoothing_passes=0)
    geom = HexGeometry(40, 3)
    x, _ = geom.coords
    elev = np.where(x < 5, -1000.0, 100.0)
    elev = np.where((x >= 15) & (x <= 17), 3000.0, elev)
    lat = np.full(geom.n_cells, 45.0)  # westerlies blow toward +x
    rain = P.base_rainfall(geom, elev, lat, SEA0, rcfg)
    flats = rain[(x >= 8) & (x <= 12)].mean()
    windward = rain[x == 15].mean()
    lee = rain[(x >= 19) & (x <= 23)].mean()
    assert windward > flats
    assert lee < 0.6 * windward
    assert lee < flats


def test_biome_bands(worlds):
    """Over many worlds, deserts are commonest in the subtropics and rare at the equator;
    forest is commoner at the equator than in the subtropics; temperature falls poleward."""
    geom = HexGeometry(100, 60)
    lat = geom.latitude(-60.0, 70.0)
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
    assert desert["sub"] / total["sub"] > 2 * desert["eq"] / total["eq"]
    assert forest["eq"] / total["eq"] > forest["sub"] / total["sub"]


# --- terrain and resources ---------------------------------------------------------------------


def test_terrain_classes(worlds, cfg):
    tc = cfg.world.terrain
    for seed, w in worlds.items():
        t = flat(w, "terrain")
        e = flat(w, "elevation")
        assert set(np.unique(t)) <= {int(v) for v in Terrain}
        np.testing.assert_array_equal(t == Terrain.SEA, e < SEA0)
        assert (t[e >= tc.mountain_m] == Terrain.MOUNTAIN).all()
        assert (t[(e >= tc.hills_m) & (e < tc.mountain_m)] == Terrain.HILLS).all()


def test_fertility_and_wild_food_ranges(worlds):
    for seed, w in worlds.items():
        sea = flat(w, "terrain") == Terrain.SEA
        for name in ("fertility", "wild_food"):
            v = flat(w, name)
            assert v.min() >= 0 and v.max() <= 1, (seed, name)
            assert (v[sea] == 0).all(), (seed, name)


def test_floodplains_are_fertile(worlds):
    river_f, dry_f = [], []
    for w in worlds.values():
        t = flat(w, "terrain")
        f = flat(w, "fertility")
        r = flat(w, "river")
        lowland = np.isin(t, [Terrain.PLAIN, Terrain.FOREST, Terrain.STEPPE, Terrain.COAST])
        river_f.append(f[lowland & (r > 0)])
        dry_f.append(f[lowland & (r == 0)])
    assert np.concatenate(river_f).mean() > np.concatenate(dry_f).mean() + 0.1


def test_wild_food_highest_in_wetlands_coasts_and_edges(worlds):
    for w in worlds.values():
        t = flat(w, "terrain")
        wf = flat(w, "wild_food")
        assert wf[t == Terrain.DESERT].mean() < wf[t == Terrain.COAST].mean()
        assert wf[t == Terrain.MOUNTAIN].mean() < wf[t == Terrain.FOREST].mean()


@pytest.mark.parametrize("seed", WORLD_SEEDS[:3])
def test_world_independent_of_other_streams(seed):
    """The world draws only from its own stream: adding streams does not change it."""
    cfg = load_config()
    more = cfg.with_overrides({"rng": {"streams": list(cfg.rng.streams) + ["extra"]}})
    a = P.generate_world(cfg, RngStreams(seed, cfg.rng.streams)["world"])
    b = P.generate_world(more, RngStreams(seed, more.rng.streams)["world"])
    np.testing.assert_array_equal(a.elevation, b.elevation)
