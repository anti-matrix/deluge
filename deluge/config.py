"""Config loader: YAML files into frozen dataclasses.

Unknown and missing keys are rejected so a typo in a config never silently falls back to
a default. Mappings become read-only ``FrozenDict``s and lists become tuples, so a loaded
config is immutable all the way down.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class FrozenDict(dict):
    """Read-only dict that still pickles (MappingProxyType does not)."""

    def _readonly(self, *args, **kwargs):
        raise TypeError("config mappings are read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly
    __ior__ = _readonly

    def __reduce__(self):
        return (FrozenDict, (dict(self),))


@dataclass(frozen=True)
class RunConfig:
    seed: int
    years: int


@dataclass(frozen=True)
class RngConfig:
    streams: tuple[str, ...]


@dataclass(frozen=True)
class GridConfig:
    width: int
    height: int
    torus: bool


@dataclass(frozen=True)
class LandmassConfig:
    min_count: int
    max_count: int
    min_share_of_land: float
    min_size_ratio: float


@dataclass(frozen=True)
class ContinentConfig:
    count_min: int
    count_max: int
    radius_min: float
    radius_max: float
    edge_margin: float
    min_separation: float
    placement_tries: int
    warp_frequency: float
    warp_octaves: int
    warp_persistence: float
    warp_lacunarity: float
    warp_strength: float
    elongation_max: float
    rim_width: float


@dataclass(frozen=True)
class NoiseConfig:
    octaves: int
    base_frequency: float
    ridge_frequency: float
    ridge_octaves: int
    ridge_sharpness: float
    persistence: float
    lacunarity: float
    mask_weight: float
    noise_weight: float
    ridge_weight: float
    edge_falloff_width: float
    edge_falloff_weight: float


@dataclass(frozen=True)
class ElevationConfig:
    land_max_m: float
    ocean_min_m: float
    land_exponent: float
    ocean_exponent: float
    shelf_share_of_land: float
    shelf_step_m: float


@dataclass(frozen=True)
class TectonicsConfig:
    enabled: bool


@dataclass(frozen=True)
class ErosionConfig:
    droplets: int
    max_steps: int
    capacity_factor: float
    erode_rate: float
    deposit_rate: float
    max_erode_fraction: float


@dataclass(frozen=True)
class BaseTemperatureConfig:
    equator_c: float
    pole_c: float
    lat_exponent: float
    lapse_rate_c_per_km: float


@dataclass(frozen=True)
class RainfallConfig:
    base_mm: float
    bands: tuple[tuple[float, float, float], ...]
    winds: tuple[tuple[float, float, int], ...]
    upwind_moisture: float
    ocean_recharge: float
    land_loss: float
    orographic_gain_per_km: float
    shadow_loss_per_km: float
    min_moisture: float
    smoothing_passes: int


@dataclass(frozen=True)
class RiverConfig:
    et_base_mm: float
    et_per_c_mm: float
    min_flow: float


@dataclass(frozen=True)
class TerrainConfig:
    mountain_m: float
    hills_m: float
    tundra_c: float
    desert_rain_mm: float
    desert_rain_per_c: float
    steppe_rain_mm: float
    steppe_rain_per_c: float
    forest_rain_mm: float
    forest_rain_per_c: float
    wetland_max_m_above_sea: float
    wetland_min_flow: float
    wetland_min_rain_mm: float


@dataclass(frozen=True)
class FertilityConfig:
    base: Mapping[str, float]
    river_bonus: float
    river_flow_ref: float
    sediment_bonus: float
    sediment_ref_m: float
    cold_zero_c: float
    cold_full_c: float


@dataclass(frozen=True)
class WildFoodConfig:
    base: Mapping[str, float]
    forest_edge_bonus: float
    river_bonus: float
    cold_zero_c: float
    cold_full_c: float
    cold_floor: float


@dataclass(frozen=True)
class WorldConfig:
    source: str
    lat_south: float
    lat_north: float
    land_share: float
    reference_sea_level_m: float
    max_attempts: int
    landmasses: LandmassConfig
    continents: ContinentConfig
    noise: NoiseConfig
    elevation: ElevationConfig
    tectonics: TectonicsConfig
    erosion: ErosionConfig
    fill_epsilon_m: float
    temperature: BaseTemperatureConfig
    rainfall: RainfallConfig
    rivers: RiverConfig
    terrain: TerrainConfig
    fertility: FertilityConfig
    wild_food: WildFoodConfig


@dataclass(frozen=True)
class SeaLevelConfig:
    preset: str
    presets: Mapping[str, tuple[tuple[float, float], ...]]


@dataclass(frozen=True)
class TempAnomalyConfig:
    cold_snap_start: int
    cold_snap_end: int
    cold_snap_depth_min: float
    cold_snap_depth_max: float
    cold_snap_ramp_years: int
    optimum_start: int
    optimum_end: int
    optimum_peak: float
    optimum_ramp_years: int
    late_cooling_end_value: float
    red_noise_phi: float
    red_noise_sigma: float


@dataclass(frozen=True)
class ShockConfig:
    drought_annual_probability: float
    drought_duration_min: int
    drought_duration_max: int
    drought_radius_min: float
    drought_radius_max: float
    drought_rain_factor_min: float
    drought_rain_factor_max: float
    volcanic_annual_probability: float
    volcanic_duration_min: int
    volcanic_duration_max: int
    volcanic_temp_drop_min: float
    volcanic_temp_drop_max: float


@dataclass(frozen=True)
class ClimateConfig:
    sea_level: SeaLevelConfig
    temperature: TempAnomalyConfig
    shocks: ShockConfig


@dataclass(frozen=True)
class ForageConfig:
    min_density: float
    max_density: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float


@dataclass(frozen=True)
class HerdConfig:
    min_density: float
    max_density: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float
    terrain: Mapping[str, float]


@dataclass(frozen=True)
class FarmConfig:
    min_density: float
    max_density: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float
    river_water_mm_per_flow: float
    terrain: Mapping[str, float]


@dataclass(frozen=True)
class CapacityConfig:
    cell_area_km2: float
    forage: ForageConfig
    herd: HerdConfig
    farm: FarmConfig


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str


@dataclass(frozen=True)
class Config:
    run: RunConfig
    rng: RngConfig
    grid: GridConfig
    world: WorldConfig
    climate: ClimateConfig
    capacity: CapacityConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def hash(self) -> str:
        """Stable hash of the full config, for run metadata."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with nested overrides applied, e.g. {"run": {"seed": 7}}."""
        return _convert(Config, _deep_merge(self.to_dict(), overrides), "config")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _convert(Config, raw, "config")


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_to_plain(v) for v in obj]
    return obj


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _convert(hint: Any, value: Any, where: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build_dataclass(hint, value, where)
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise TypeError(f"{where}: expected {len(args)} items, got {value!r}")
        return tuple(_convert(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin in (collections.abc.Mapping, dict):
        if not isinstance(value, Mapping):
            raise TypeError(f"{where}: expected a mapping, got {value!r}")
        key_t, val_t = args
        return FrozenDict(
            {_convert(key_t, k, where): _convert(val_t, v, f"{where}.{k}") for k, v in value.items()}
        )
    return _coerce(hint, value, where)


def _build_dataclass(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    unknown = set(raw) - set(names)
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")
    missing = set(names) - set(raw)
    if missing:
        raise KeyError(f"{where}: missing keys {sorted(missing)}")
    return cls(**{n: _convert(hints[n], raw[n], f"{where}.{n}") for n in names})


def _coerce(hint: Any, value: Any, where: str) -> Any:
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected bool, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected int, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected float, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected str, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported config type {hint!r}")
