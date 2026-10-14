"""Climate over time: sea-level curve, temperature anomaly, and shocks (spec section 5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deluge.config import SeaLevelConfig, ShockConfig, TempAnomalyConfig


def sea_level_curve(cfg: SeaLevelConfig, years: int) -> np.ndarray:
    """Sea level in metres relative to present for years 0..years (piecewise linear)."""
    points = cfg.presets[cfg.preset]
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(f"sea-level preset {cfg.preset!r}: years must increase")
    return np.interp(np.arange(years + 1, dtype=float), xs, ys)


def preset_horizon(cfg: SeaLevelConfig) -> int:
    """Last year given in the active sea-level preset."""
    return int(max(p[0] for p in cfg.presets[cfg.preset]))


def _spell(years: np.ndarray, start: int, end: int, offset: float, ramp: int) -> np.ndarray:
    """``offset`` inside [start, end], ramping in and out over ``ramp`` years inside it."""
    if not start + ramp < end - ramp:
        raise ValueError("climate.temperature: a spell must be longer than two ramps")
    return np.interp(years, [start, start + ramp, end - ramp, end], [0.0, offset, offset, 0.0],
                     left=0.0, right=0.0)


def temp_anomaly_curve(cfg: TempAnomalyConfig, years: int, cold_snap_depth: float) -> np.ndarray:
    """Deterministic part of the anomaly: cold snap, recovery, optimum, slow cooling to
    ``late_cooling_end_year`` (flat after), plus the warm and cold spells on top. Fixed in
    calendar years, so a short run sees the same curve as the start of a long one."""
    xs = [
        cfg.cold_snap_start,
        cfg.cold_snap_end,
        cfg.cold_snap_end + cfg.cold_snap_ramp_years,
        cfg.optimum_start - cfg.optimum_ramp_years,
        cfg.optimum_start,
        cfg.optimum_end,
        cfg.late_cooling_end_year,
    ]
    ys = [cold_snap_depth, cold_snap_depth, 0.0, 0.0, cfg.optimum_peak, cfg.optimum_peak,
          cfg.late_cooling_end_value]
    if np.any(np.diff(xs) <= 0) or cfg.warm_spell_end > cfg.cold_spell_start:
        raise ValueError("climate.temperature: phase years must increase")
    t = np.arange(years + 1, dtype=float)
    return (np.interp(t, xs, ys)
            + _spell(t, cfg.warm_spell_start, cfg.warm_spell_end, cfg.warm_spell_offset, cfg.spell_ramp_years)
            + _spell(t, cfg.cold_spell_start, cfg.cold_spell_end, cfg.cold_spell_offset, cfg.spell_ramp_years))


def red_noise(phi: float, sigma: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) series started from its stationary distribution."""
    eps = rng.normal(0.0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1.0 - phi * phi, 1e-12))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    return out


def temp_anomaly_series(cfg: TempAnomalyConfig, years: int, rng: np.random.Generator) -> np.ndarray:
    """Global temperature anomaly (degC) for years 0..years, curve plus red noise."""
    depth = float(rng.uniform(cfg.cold_snap_depth_min, cfg.cold_snap_depth_max))
    curve = temp_anomaly_curve(cfg, years, depth)
    return curve + red_noise(cfg.red_noise_phi, cfg.red_noise_sigma, years + 1, rng)


@dataclass
class Drought:
    start: int
    end: int  # exclusive
    center: int  # cell id
    radius: float  # km
    footprint_cells: np.ndarray
    footprint_factor: np.ndarray  # rain multiplier on those cells


@dataclass
class VolcanicWinter:
    start: int
    end: int  # exclusive
    temp_drop: float


class ClimateShocks:
    """Yearly draws of regional droughts and rare volcanic winters.

    Droughts are spatially correlated: a disc of cells around a random land cell, strongest
    at the center. Overlapping droughts multiply.
    """

    def __init__(self, cfg: ShockConfig, positions_km: np.ndarray, rng: np.random.Generator):
        self.cfg = cfg
        self.positions = positions_km
        self.rng = rng
        self.droughts: list[Drought] = []
        self.volcanic: list[VolcanicWinter] = []
        self.rain_factor = np.ones(len(positions_km))
        self.temp_shock = 0.0

    def step(self, year: int, land: np.ndarray) -> None:
        cfg, rng = self.cfg, self.rng
        self.droughts = [d for d in self.droughts if d.end > year]
        self.volcanic = [v for v in self.volcanic if v.end > year]

        if rng.random() < cfg.drought_annual_probability:
            land_ids = np.flatnonzero(land)
            if len(land_ids):
                center = int(rng.choice(land_ids))
                radius = float(rng.uniform(cfg.drought_radius_min_km, cfg.drought_radius_max_km))
                severity = float(rng.uniform(cfg.drought_rain_factor_min, cfg.drought_rain_factor_max))
                duration = int(rng.integers(cfg.drought_duration_min, cfg.drought_duration_max + 1))
                d = np.linalg.norm(self.positions - self.positions[center], axis=1) / radius
                cells = np.flatnonzero(d < 1.0)
                factor = 1.0 - (1.0 - severity) * (1.0 - d[cells] ** 2)
                self.droughts.append(Drought(year, year + duration, center, radius, cells, factor))
        if rng.random() < cfg.volcanic_annual_probability:
            duration = int(rng.integers(cfg.volcanic_duration_min, cfg.volcanic_duration_max + 1))
            drop = float(rng.uniform(cfg.volcanic_temp_drop_min, cfg.volcanic_temp_drop_max))
            self.volcanic.append(VolcanicWinter(year, year + duration, drop))

        if self.droughts:
            factor = np.ones(len(self.positions))
            for d in self.droughts:
                factor[d.footprint_cells] *= d.footprint_factor
            self.rain_factor = factor
        elif self.rain_factor.min() < 1.0:
            self.rain_factor = np.ones(len(self.positions))
        self.temp_shock = float(sum(v.temp_drop for v in self.volcanic))
