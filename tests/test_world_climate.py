import dataclasses

import numpy as np
import pytest

from deluge.world.climate import (
    ClimateShocks,
    red_noise,
    sea_level_curve,
    temp_anomaly_series,
)
from deluge.world.grid import HexGeometry


def test_sea_level_curve_default(cfg):
    sl = sea_level_curve(cfg.climate.sea_level, 11000)
    assert len(sl) == 11001
    assert sl[0] == pytest.approx(-60.0)
    assert sl[3000] == pytest.approx(0.0)
    assert np.all(np.diff(sl) >= 0)
    assert np.all(sl[3000:] == 0.0)


def test_sea_level_has_two_fast_pulses(cfg):
    sl = sea_level_curve(cfg.climate.sea_level, 3000)
    rate = np.diff(sl)
    mean_rate = (sl[3000] - sl[0]) / 3000
    fast = rate > 2.5 * mean_rate
    # Count separate runs of fast years.
    starts = np.flatnonzero(fast[1:] & ~fast[:-1]) + 1
    runs = len(starts) + int(fast[0])
    assert runs == 2


def test_modern_melt_preset(cfg):
    sl = sea_level_curve(dataclasses.replace(cfg.climate.sea_level, preset="modern_melt"), 100)
    assert np.all(sl == 70.0)


def test_coasts_drown_across_first_3000_years(worlds, cfg):
    sl = sea_level_curve(cfg.climate.sea_level, 11000)
    for seed, w in worlds.items():
        e = w.elevation.ravel()
        sea = [int((e < sl[y]).sum()) for y in (0, 1000, 2000, 3000, 6000, 11000)]
        assert sea[0] < sea[1] < sea[2] < sea[3], (seed, sea)
        assert sea[3] == sea[4] == sea[5]
        lost = sea[3] - sea[0]
        assert lost >= 0.1 * (e >= sl[0]).sum(), (seed, sea)


def test_temp_anomaly_phases(cfg):
    t = cfg.climate.temperature
    series = temp_anomaly_series(t, 11000, np.random.default_rng(0))
    cold = series[100:1100].mean()
    assert t.cold_snap_depth_min - 0.5 < cold < t.cold_snap_depth_max + 0.5
    optimum = series[3500:5500].mean()
    assert optimum > 0.5 * t.optimum_peak
    late = series[10000:].mean()
    assert late < optimum
    assert abs(series[1800:2400].mean()) < 0.6


def test_temp_anomaly_deterministic(cfg):
    a = temp_anomaly_series(cfg.climate.temperature, 2000, np.random.default_rng(3))
    b = temp_anomaly_series(cfg.climate.temperature, 2000, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_red_noise_is_autocorrelated():
    x = red_noise(0.97, 0.1, 20000, np.random.default_rng(1))
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert r1 > 0.9


def test_shocks_bounds_and_determinism(cfg):
    geom = HexGeometry(40, 20)
    land = np.ones(geom.n_cells, dtype=bool)
    sc = cfg.climate.shocks

    def run(seed):
        s = ClimateShocks(sc, geom.positions, np.random.default_rng(seed))
        factors, temps, durations = [], [], []
        for year in range(3000):
            s.step(year, land)
            factors.append(s.rain_factor.copy())
            temps.append(s.temp_shock)
            durations += [d.end - d.start for d in s.droughts]
        return np.array(factors), np.array(temps), durations

    f, t, durations = run(7)
    f2, t2, _ = run(7)
    np.testing.assert_array_equal(f, f2)
    np.testing.assert_array_equal(t, t2)
    assert f.min() > 0 and f.max() <= 1.0
    assert (f < 1).any(axis=1).mean() > 0.2  # droughts occur
    assert min(durations) >= sc.drought_duration_min
    assert max(durations) <= sc.drought_duration_max
    assert t.max() <= 0.0
    # Spatially correlated: a drought hits a contiguous patch, not scattered cells.
    year = int(np.argmax((f < 1).sum(axis=1)))
    hit = f[year] < 1
    assert geom.components(hit).max() < 3


def test_no_shocks_when_probabilities_zero(cfg):
    geom = HexGeometry(10, 10)
    sc = dataclasses.replace(cfg.climate.shocks, drought_annual_probability=0.0,
                             volcanic_annual_probability=0.0)
    s = ClimateShocks(sc, geom.positions, np.random.default_rng(0))
    for year in range(500):
        s.step(year, np.ones(geom.n_cells, dtype=bool))
        assert (s.rain_factor == 1.0).all() and s.temp_shock == 0.0
