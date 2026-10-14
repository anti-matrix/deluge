import dataclasses

import numpy as np
import pytest

from deluge.world import geography
from deluge.world.climate import (
    ClimateShocks,
    red_noise,
    sea_level_curve,
    temp_anomaly_curve,
    temp_anomaly_series,
)
from deluge.world.grid import HexGeometry


def test_sea_level_curve_default(cfg):
    sl = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
    assert len(sl) == cfg.run.years + 1 == 12901
    assert sl[0] == pytest.approx(-60.0)
    assert sl[3000] == pytest.approx(0.0)
    assert np.all(np.diff(sl) >= 0)
    assert np.all(sl[3000:] == 0.0)


def test_sea_level_has_two_fast_pulses(cfg):
    sl = sea_level_curve(cfg.climate.sea_level, 3000)
    assert len(geography.pulse_windows(sl, cfg.world.land_bridges.pulse_rate_factor)) == 2


def test_modern_melt_preset_has_same_pulses(cfg):
    sl_cfg = dataclasses.replace(cfg.climate.sea_level, preset="modern_melt")
    sl = sea_level_curve(sl_cfg, cfg.run.years)
    assert sl[0] == pytest.approx(0.0)
    assert sl[3000] == pytest.approx(70.0)
    assert np.all(sl[3000:] == 70.0)
    assert np.all(np.diff(sl) >= 0)
    default = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
    rate = cfg.world.land_bridges.pulse_rate_factor
    a = [(p.start_year, p.end_year) for p in geography.pulse_windows(sl, rate)]
    b = [(p.start_year, p.end_year) for p in geography.pulse_windows(default, rate)]
    assert a == b
    # Same shape, scaled from 60 m to 70 m of rise.
    np.testing.assert_allclose((sl - sl[0]) / 70.0, (default - default[0]) / 60.0, atol=1e-3)


def test_coasts_drown_across_first_3000_years(small_world, cfg):
    sl = sea_level_curve(cfg.climate.sea_level, cfg.run.years)
    for seed, w in ((0, small_world),):
        e = w.elevation.ravel()
        sea = [int((e < sl[y]).sum()) for y in (0, 1000, 2000, 3000, 6000, 12900)]
        assert sea[0] < sea[1] < sea[2] < sea[3], (seed, sea)
        assert sea[3] == sea[4] == sea[5]


def test_temp_anomaly_phases(cfg):
    """Cold snap, recovery, warm optimum (years 3,000-7,500), slow cooling, then a warm and
    a cold spell before the present (spec section 5)."""
    t = cfg.climate.temperature
    series = temp_anomaly_series(t, cfg.run.years, np.random.default_rng(0))
    assert len(series) == 12901
    cold = series[100:1100].mean()
    assert t.cold_snap_depth_min - 0.5 < cold < t.cold_snap_depth_max + 0.5
    optimum = series[3500:7000].mean()
    assert optimum > 0.5 * t.optimum_peak
    late = series[9000:11800].mean()
    assert late < optimum
    assert abs(series[1800:2400].mean()) < 0.6
    curve = temp_anomaly_curve(t, cfg.run.years, -4.0)  # no red noise
    assert curve[7000] == pytest.approx(t.optimum_peak)
    assert curve[12050] > curve[11850] and curve[12500] < curve[11850]  # warm, then cold spell
    assert curve[12050] - curve[12500] > 0.5 * (t.warm_spell_offset - t.cold_spell_offset)
    assert curve[12900] == pytest.approx(t.late_cooling_end_value)


def test_temp_anomaly_independent_of_run_length(cfg):
    t = cfg.climate.temperature
    np.testing.assert_array_equal(temp_anomaly_curve(t, 3000, -4.0), temp_anomaly_curve(t, cfg.run.years, -4.0)[:3001])

def test_temp_anomaly_deterministic(cfg):
    a = temp_anomaly_series(cfg.climate.temperature, 2000, np.random.default_rng(3))
    b = temp_anomaly_series(cfg.climate.temperature, 2000, np.random.default_rng(3))
    np.testing.assert_array_equal(a, b)


def test_red_noise_is_autocorrelated():
    x = red_noise(0.97, 0.1, 20000, np.random.default_rng(1))
    r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert r1 > 0.9


def test_shocks_bounds_and_determinism(cfg):
    geom = HexGeometry(60, 30)
    unit_km = 100.0
    land = np.ones(geom.n_cells, dtype=bool)
    sc = cfg.climate.shocks

    def run(seed):
        s = ClimateShocks(sc, geom.positions * unit_km, np.random.default_rng(seed))
        factors, temps, droughts = [], [], []
        for year in range(3000):
            s.step(year, land)
            factors.append(s.rain_factor.copy())
            temps.append(s.temp_shock)
            droughts += [(d.end - d.start, d.radius) for d in s.droughts]
        return np.array(factors), np.array(temps), droughts

    f, t, droughts = run(7)
    f2, t2, _ = run(7)
    np.testing.assert_array_equal(f, f2)
    np.testing.assert_array_equal(t, t2)
    assert f.min() > 0 and f.max() <= 1.0
    assert (f < 1).any(axis=1).mean() > 0.2  # droughts occur
    durations = [d for d, _ in droughts]
    radii = [r for _, r in droughts]
    assert min(durations) >= sc.drought_duration_min
    assert max(durations) <= sc.drought_duration_max
    assert sc.drought_radius_min_km <= min(radii) and max(radii) <= sc.drought_radius_max_km
    assert t.max() <= 0.0
    # Spatially correlated: a single drought hits one contiguous patch.
    year = next(y for y in range(len(f)) if (f[y] < 1).any())
    assert geom.components(f[year] < 1).max() < 3


def test_no_shocks_when_probabilities_zero(cfg):
    geom = HexGeometry(10, 10)
    sc = dataclasses.replace(cfg.climate.shocks, drought_annual_probability=0.0,
                             volcanic_annual_probability=0.0)
    s = ClimateShocks(sc, geom.positions * 100.0, np.random.default_rng(0))
    for year in range(500):
        s.step(year, np.ones(geom.n_cells, dtype=bool))
        assert (s.rain_factor == 1.0).all() and s.temp_shock == 0.0
