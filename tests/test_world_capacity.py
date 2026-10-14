import numpy as np

from deluge.world.capacity import MODES, CarryingCapacity
from deluge.world.procedural import ramp
from deluge.world.worlddata import Terrain


def _year0(world, cfg, anomaly=0.0):
    cap = CarryingCapacity(cfg.capacity, world)
    temp = world.base_temp.ravel() + anomaly
    rain = world.base_rain.ravel()
    sea = world.elevation.ravel() < -60.0
    return cap, temp, rain, sea, cap.compute(temp, rain, sea)


def test_zero_on_sea_and_nonnegative(worlds, cfg):
    for w in worlds.values():
        _, _, _, sea, k = _year0(w, cfg)
        for mode in MODES:
            assert (k[mode] >= 0).all()
            assert (k[mode][sea] == 0).all()


def test_densities_in_spec_ranges(worlds, cfg):
    """Where climate is fully suitable: foraging 0.1-1 person/km2, farming on good land
    (plains with fertile soil) 10-50 person/km2."""
    c = cfg.capacity
    for w in worlds.values():
        _, temp, rain, sea, k = _year0(w, cfg)
        land = ~sea
        f_ok = land & (ramp(temp, c.forage.temp_zero_c, c.forage.temp_full_c) == 1) \
            & (ramp(rain, c.forage.rain_zero_mm, c.forage.rain_full_mm) == 1)
        dens = k["forage"][f_ok] / c.cell_area_km2
        assert dens.min() >= c.forage.min_density - 1e-9
        assert dens.max() <= c.forage.max_density + 1e-9
        t = w.terrain.ravel()
        good = (land & (t == Terrain.PLAIN) & (w.fertility.ravel() > 0.6)
                & (ramp(temp, c.farm.temp_zero_c, c.farm.temp_full_c) == 1))
        if good.any():
            farm = k["farm"][good] / c.cell_area_km2
            assert farm.max() <= c.farm.max_density + 1e-9
            assert np.median(farm[farm > 0]) >= c.farm.min_density


def test_cold_reduces_farming(worlds, cfg):
    w = worlds[0]
    _, _, _, _, warm = _year0(w, cfg, 0.0)
    _, _, _, _, cold = _year0(w, cfg, -5.0)
    assert cold["farm"].sum() < warm["farm"].sum()
    assert (cold["farm"] <= warm["farm"] + 1e-9).all()


def test_mountains_cannot_be_farmed(worlds, cfg):
    for w in worlds.values():
        _, _, _, _, k = _year0(w, cfg)
        assert (k["farm"][w.terrain.ravel() == Terrain.MOUNTAIN] == 0).all()


def test_tech_multiplier_scales(worlds, cfg):
    w = worlds[0]
    cap, temp, rain, sea, base = _year0(w, cfg)
    boosted = cap.compute(temp, rain, sea, tech={"farm": np.full(len(temp), 2.0)})
    np.testing.assert_allclose(boosted["farm"], 2 * base["farm"])
    np.testing.assert_array_equal(boosted["forage"], base["forage"])
