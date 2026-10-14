import numpy as np

from deluge.world.capacity import MODES, CarryingCapacity
from deluge.world.grid import HexGeometry, hex_area_km2, hex_unit_km
from deluge.world.procedural import ramp
from deluge.world.worlddata import Terrain

SEA0 = -60.0


def _area(cfg):
    return hex_area_km2(hex_unit_km(cfg.grid.height, cfg.world.lat_south, cfg.world.lat_north,
                                    cfg.world.km_per_degree))


def _year0(world, cfg, anomaly=0.0, cells=None):
    geom = HexGeometry(world.width, world.height)
    cap = CarryingCapacity(cfg.capacity, world, _area(cfg), cells)
    temp = world.base_temp.ravel() + anomaly
    rain = world.base_rain.ravel()
    sea = world.elevation.ravel() < SEA0
    coast = ~sea & geom.neighbor_any(sea)
    return cap, temp, rain, sea, coast, cap.compute(temp, rain, sea, coast)


def test_cell_area_from_geometry(cfg):
    area = _area(cfg)
    assert 20_000 < area < 35_000  # 160 x 96 over 130 degrees of latitude


def test_zero_on_sea_and_ice(worlds, cfg):
    for w in worlds.values():
        _, _, _, sea, _, k = _year0(w, cfg)
        ice = w.terrain.ravel() == Terrain.ICE
        for mode in MODES:
            assert (k[mode] >= 0).all()
            assert (k[mode][sea] == 0).all()
            assert (k[mode][ice] == 0).all()


def test_tundra_cannot_be_farmed_and_forages_poorly(worlds, cfg):
    for w in worlds.values():
        _, _, _, _, _, k = _year0(w, cfg)
        t = w.terrain.ravel()
        tundra = t == Terrain.TUNDRA
        if tundra.any():
            assert (k["farm"][tundra] == 0).all()
            plain = t == Terrain.PLAIN
            assert k["forage"][tundra].mean() < k["forage"][plain].mean()


def test_densities_in_spec_ranges(worlds, cfg):
    """Where climate is fully suitable: foraging 0.1-1 person/km2 before the calibration
    factor, farming on good land (plains with fertile soil) 10-50 person/km2 before its
    calibration factor."""
    c = cfg.capacity
    area = _area(cfg)
    for w in worlds.values():
        _, temp, rain, sea, _, k = _year0(w, cfg)
        t = w.terrain.ravel()
        full_forage = (~sea & (t == Terrain.PLAIN)
                       & (ramp(temp, c.forage.temp_zero_c, c.forage.temp_full_c) == 1)
                       & (ramp(rain, c.forage.rain_zero_mm, c.forage.rain_full_mm) == 1))
        # Round 3 calibration scales the spec's forager densities by density_scale.
        dens = k["forage"][full_forage] / area / c.forage.density_scale
        assert dens.min() >= c.forage.min_density - 1e-9
        assert dens.max() <= c.forage.max_density + 1e-9
        good = (~sea & (t == Terrain.PLAIN) & (w.fertility.ravel() > 0.6)
                & (ramp(temp, c.farm.temp_zero_c, c.farm.temp_full_c) == 1))
        if good.any():
            # Round 4 calibration scales the spec's farming densities by density_scale (the
            # calibrated density on good river land is checked by scripts/round4_acceptance.py).
            farm = k["farm"][good] / area / c.farm.density_scale
            assert farm.max() <= c.farm.max_density + 1e-9
            assert np.median(farm[farm > 0]) >= c.farm.min_density


def test_coast_bonus_raises_foraging(worlds, cfg):
    w = worlds[1]
    cap, temp, rain, sea, coast, k = _year0(w, cfg)
    inland = cap.compute(temp, rain, sea, np.zeros_like(coast))
    assert (k["forage"] >= inland["forage"] - 1e-9).all()
    assert k["forage"][coast].sum() > inland["forage"][coast].sum()


def test_cold_reduces_farming(worlds, cfg):
    w = worlds[1]
    *_, warm = _year0(w, cfg, 0.0)
    *_, cold = _year0(w, cfg, -5.0)
    assert cold["farm"].sum() < warm["farm"].sum()
    assert (cold["farm"] <= warm["farm"] + 1e-9).all()


def test_subset_matches_full_computation(worlds, cfg):
    w = worlds[1]
    ever_land = np.flatnonzero(w.elevation.ravel() >= SEA0)
    *_, full = _year0(w, cfg)
    *_, part = _year0(w, cfg, cells=ever_land)
    for mode in MODES:
        np.testing.assert_allclose(part[mode], full[mode])


def test_tech_multiplier_scales(worlds, cfg):
    w = worlds[1]
    cap, temp, rain, sea, coast, base = _year0(w, cfg)
    boosted = cap.compute(temp, rain, sea, coast, tech={"farm": np.full(len(temp), 2.0)})
    np.testing.assert_allclose(boosted["farm"], 2 * base["farm"])
    np.testing.assert_array_equal(boosted["forage"], base["forage"])
