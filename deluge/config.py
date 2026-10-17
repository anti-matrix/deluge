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
    world_seed: int
    history_seed: int
    years: int


@dataclass(frozen=True)
class RngConfig:
    streams: tuple[str, ...]


@dataclass(frozen=True)
class GridConfig:
    width: int
    height: int
    torus: bool


# --- world -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LandmassConfig:
    min_count: int
    max_count: int
    min_share_of_land: float
    min_size_ratio: float
    min_islands: int


@dataclass(frozen=True)
class SeaBasinConfig:
    required: bool
    open_water_min_steps: int
    min_basin_cells: int


@dataclass(frozen=True)
class LandBridgeConfig:
    min_count: int
    min_share_of_land: float
    pulse_rate_factor: float
    construct_if_missing: bool
    max_length_steps: int
    crest_margin_m: float
    end_rise_m_per_step: float
    flank_floor_m: float


@dataclass(frozen=True)
class ShapeConfig:
    base_frequency: float
    octaves: int
    persistence: float
    lacunarity: float
    warp_frequency: float
    warp_octaves: int
    warp_persistence: float
    warp_lacunarity: float
    warp_strength: float
    plate_bias_weight: float
    plate_bias_smoothing_passes: int
    edge_falloff_width: float
    pole_falloff_deg: float
    edge_falloff_weight: float
    edge_sea_cells: int


@dataclass(frozen=True)
class TectonicsConfig:
    enabled: bool
    plates_min: int
    plates_max: int
    noise_persistence: float
    noise_lacunarity: float
    boundary_warp_frequency: float
    boundary_warp_octaves: int
    boundary_warp_strength: float
    speed_min: float
    speed_max: float
    convergence_threshold: float
    mountain_height: float
    mountain_width_km: float
    peak_noise_frequency: float
    peak_noise_octaves: int
    peak_noise_sharpness: float
    peak_noise_weight: float
    plateau_height: float
    plateau_width_km: float
    plateau_falloff_km: float
    rift_depth: float
    rift_width_km: float
    basin_depth: float
    basin_width_km: float


@dataclass(frozen=True)
class ElevationConfig:
    land_knots: tuple[tuple[float, float], ...]
    sea_knots: tuple[tuple[float, float], ...]
    detail_m: float
    detail_fraction: float
    detail_frequency: float
    detail_octaves: int
    detail_persistence: float
    detail_lacunarity: float


@dataclass(frozen=True)
class ErosionConfig:
    droplets: int
    max_steps: int
    direction_exponent: float
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
    continentality_neutral_c: float
    continentality_amplitude: float
    continentality_scale_km: float


@dataclass(frozen=True)
class WesterliesConfig:
    lat_min: float
    lat_max: float
    u: float
    v_poleward: float
    rain_mm: float
    edge_deg: float
    land_loss_factor: float


@dataclass(frozen=True)
class RainfallConfig:
    base_mm: float
    bands: tuple[tuple[float, float, float], ...]
    winds: tuple[tuple[float, float, float, float], ...]
    westerlies: WesterliesConfig
    wind_sharpness: float
    noise_octaves: int
    noise_persistence: float
    noise_lacunarity: float
    band_wobble_deg: float
    band_wobble_frequency: float
    wind_angle_noise_deg: float
    wind_noise_frequency: float
    rain_noise_amplitude: float
    rain_noise_frequency: float
    upwind_moisture: float
    land_loss_per_1000km: float
    orographic_gain_per_km: float
    shadow_loss_per_km: float
    shadow_min_uplift_m: float
    orographic_smoothing_passes: int
    min_moisture: float
    continentality_weight: float
    continentality_scale_km: float
    iterations: int
    smoothing_passes: int


@dataclass(frozen=True)
class RiverConfig:
    et_base_mm: float
    et_per_c_mm: float
    min_flow: float
    direction_exponent: float


@dataclass(frozen=True)
class TerrainConfig:
    mountain_m: float
    hills_m: float
    ice_c: float
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
class MineralTypeConfig:
    share: float
    belt: float
    upland: float
    lowland: float
    stable: float
    sediment: float
    shelf: float
    abundance_min: float
    abundance_max: float


@dataclass(frozen=True)
class MineralsConfig:
    belt_km: float
    upland_m: float
    lowland_m: float
    sediment_ref_m: float
    noise_frequency: float
    noise_octaves: int
    noise_persistence: float
    noise_lacunarity: float
    noise_weight: float
    access_radius_cells: int
    depletion: bool
    depletion_per_settlement_year: float
    types: Mapping[str, MineralTypeConfig]


@dataclass(frozen=True)
class WorldConfig:
    source: str
    lat_south: float
    lat_north: float
    km_per_degree: float
    land_share: float
    reference_sea_level_m: float
    max_attempts: int
    landmasses: LandmassConfig
    seas: SeaBasinConfig
    land_bridges: LandBridgeConfig
    shape: ShapeConfig
    tectonics: TectonicsConfig
    elevation: ElevationConfig
    erosion: ErosionConfig
    fill_epsilon_m: float
    fill_random_factor: float
    temperature: BaseTemperatureConfig
    rainfall: RainfallConfig
    rivers: RiverConfig
    terrain: TerrainConfig
    fertility: FertilityConfig
    wild_food: WildFoodConfig
    minerals: MineralsConfig


# --- climate over time ------------------------------------------------------------------------


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
    late_cooling_end_year: int
    late_cooling_end_value: float
    warm_spell_start: int
    warm_spell_end: int
    warm_spell_offset: float
    cold_spell_start: int
    cold_spell_end: int
    cold_spell_offset: float
    spell_ramp_years: int
    red_noise_phi: float
    red_noise_sigma: float


@dataclass(frozen=True)
class ShockConfig:
    drought_annual_probability: float
    drought_duration_min: int
    drought_duration_max: int
    drought_radius_min_km: float
    drought_radius_max_km: float
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


# --- carrying capacity ------------------------------------------------------------------------


@dataclass(frozen=True)
class ForageConfig:
    min_density: float
    max_density: float
    density_scale: float
    coast_bonus: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float
    terrain: Mapping[str, float]


@dataclass(frozen=True)
class HerdConfig:
    min_density: float
    max_density: float
    density_scale: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float
    terrain: Mapping[str, float]


@dataclass(frozen=True)
class FarmConfig:
    min_density: float
    max_density: float
    density_scale: float
    temp_zero_c: float
    temp_full_c: float
    rain_zero_mm: float
    rain_full_mm: float
    river_water_mm_per_flow: float
    terrain: Mapping[str, float]


@dataclass(frozen=True)
class CapacityConfig:
    forage: ForageConfig
    herd: HerdConfig
    farm: FarmConfig


# --- people -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FoundersConfig:
    names: tuple[str, ...]
    population_min: int
    population_max: int
    capacity_top_share: float
    min_distance_cells: int
    min_landmasses: int
    max_attempts: int
    pinned_cells: tuple[tuple[int, int], ...]
    culture_traits: tuple[str, ...]
    culture_min: float
    culture_max: float


@dataclass(frozen=True)
class DemographyConfig:
    forage_abundance: float
    herd_abundance: float
    farm_abundance: float
    forage_max_growth: float
    herd_max_growth: float
    farm_max_growth: float
    mode_shift_rate: float
    mode_brain_bias: float
    farm_stickiness: float
    famine_mortality: float
    famine_max_mortality: float
    demographic_turnover: float
    store_cap_years: float
    spoilage: float
    min_population: float
    wild_food_degrade: float
    wild_food_recover: float
    wild_food_floor: float


@dataclass(frozen=True)
class MigrationConfig:
    split_threshold: float
    split_fraction: float
    split_retry_years: int
    search_radius: int
    distance_cost: float
    crowding_cost: float
    min_found_food_ratio: float
    famine_move_ratio: float
    sedentary_farm_share: float
    farmer_famine_years: int
    split_threshold_range: float
    core_move_rate: float


# --- knowledge, teachers, events --------------------------------------------------------------


@dataclass(frozen=True)
class TechniqueConfig:
    label: str
    late: bool
    resources: tuple[str, ...]
    category: str
    prerequisites: tuple[str, ...]
    invention_rate: float
    used_with: str
    unlocks: str
    effects: Mapping[str, float]


@dataclass(frozen=True)
class KnowledgeConfig:
    graph_file: str
    techniques: Mapping[str, TechniqueConfig]
    late_techniques_enabled: bool
    invention_scale: float
    invention_reference_population: float
    invention_population_exponent: float
    division_of_labor_invention_bonus: float
    diffusion_rate: float
    use_share: float
    size_use_population: float
    decay_unused: float
    decay_latent: float
    decay_small: float
    small_reference_population: float
    small_factor_cap: float
    practice_gain: float
    crash_share: float


@dataclass(frozen=True)
class TeachersConfig:
    enabled: bool
    arrival_year: int
    first_group: int
    years_per_group: int
    knowledge: tuple[str, ...]
    teach_rate: float
    technologies: tuple[str, ...]
    artifact_chance: float
    artifacts_min: int
    artifacts_max: int
    artifact_types: tuple[str, ...]
    artifact_lifespan_min: int
    artifact_lifespan_max: int
    tools_production: float
    medicine_famine_mortality: float
    weapons_military: float


@dataclass(frozen=True)
class EventsConfig:
    min_magnitude: Mapping[str, float]


# --- polities, cohesion, ancestry ---------------------------------------------------------------


@dataclass(frozen=True)
class CohesionConfig:
    founder_cohesion: float
    child_cohesion: float
    growth_rate: float
    distant_culture_min: float
    distant_culture_full: float
    base_decay: float
    size_decay: float
    size_reference_settlements: float
    distance_decay: float
    distance_reference: float
    inequality_decay: float
    elite_decay: float
    pressure_decay: float
    monument_cohesion: float
    monument_half: float
    war_frontier_weight: float
    war_frontier_half: float
    grievance_decay: float


@dataclass(frozen=True)
class PressureConfig:
    population_reference: float
    population_weight: float
    elite_weight: float
    state_weight: float
    state_reference: float


@dataclass(frozen=True)
class EliteConfig:
    initial_ratio: float
    growth: float
    tribute_reference: float
    decline: float
    positions: float


@dataclass(frozen=True)
class PolitiesConfig:
    contact_radius: int
    territory_radius: int
    culture_mutation: float
    culture_contact_drift: float
    tribute_max_rate: float
    treasury_spoilage: float
    monument_max_share: float
    monument_cost: float
    monument_upkeep: float
    secession_base: float
    secession_distance_reference: float
    secession_distance_exponent: float
    secession_pressure_weight: float
    secession_grievance_weight: float
    secession_max: float
    fragment_cohesion: float
    fragment_pressure: float
    fragment_rate: float
    region_spacing: int
    merge_rate: float
    merge_crossover_chance: float
    marker_mutation: float
    marker_effect_max: float
    marker_cohesion_decay: float
    marker_grievance_rate: float
    marker_factor_max: float
    cohesion: CohesionConfig
    pressure: PressureConfig
    elites: EliteConfig


@dataclass(frozen=True)
class AncestryConfig:
    appearance_traits: tuple[str, ...]
    founder_mean_min: float
    founder_mean_max: float
    founder_min_distance: float
    founder_variance: float
    founder_max_attempts: int
    teachers_appearance: tuple[float, ...]
    generation_years: int
    drift_sigma: float
    drift_reference_population: float
    drift_sigma_max: float
    mixing_rate: float
    intermarriage_min: float
    intermarriage_max: float
    dialect_dims: int
    dialect_founder_sigma: float
    dialect_drift_sigma: float
    dialect_contact_rate: float
    dialect_assimilation_rate: float
    selection_enabled: bool
    selection_trait: str
    selection_rate: float
    selection_optimum_equator: float
    selection_optimum_pole: float
    blind_perception: bool
    perceived_appearance_reference: float
    perceived_dialect_reference: float


@dataclass(frozen=True)
class WarConfig:
    enabled: bool
    attack_rate: float
    max_wars: int
    war_memory_years: int
    store_half_years: float
    mobilization_base: float
    mobilization_militarism: float
    mobilization_hierarchy: float
    mobilization_max: float
    logistics_km: float
    home_bonus: float
    terrain_defense: Mapping[str, float]
    win_steepness: float
    casualty_rate: float
    loser_casualty_factor: float
    attacker_casualty_cap: float
    defender_casualty_cap: float
    raid_loot_share: float
    neighbor_capture_chance: float
    settler_share: float
    settler_source_cap: float
    capital_cohesion_loss: float
    destroy_ratio: float
    battle_log_min_losses: float
    war_recent_decay: float
    enemy_military_diffusion: float
    subject_diffusion_rate: float
    grievance_conquest: float
    grievance_decay: float
    grievance_floor: float
    assimilation_rate: float
    assimilated_distance: float


@dataclass(frozen=True)
class BrainFeaturesConfig:
    log_population_center: float
    log_population_scale: float
    store_half_years: float
    elite_reference: float
    settlements_reference: float
    growth_window_years: int
    technique_reference: float
    strength_ratio_scale: float
    hostile_reference: float
    war_record_reference: float
    war_record_decay: float
    temp_anomaly_scale: float


@dataclass(frozen=True)
class BrainsConfig:
    hidden: int
    recurrent_units: int
    memory_inputs: int
    init_sigma: float
    marker_init_sigma: float
    mutation_sigma: float
    jump_chance: float
    jump_sigma: float
    frozen: bool
    features: BrainFeaturesConfig


# --- memory -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryConsequencesConfig:
    cohesion_weight: float
    cohesion_half_medians: float
    legitimacy_weight: float
    legitimacy_half_medians: float
    merger_weight: float
    assimilation_weight: float
    invocation_gain: float
    invocation_symbols: int
    brain_input_weight: float


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool
    control: bool
    generation_years: int
    capacity: int
    vividness_scale: float
    share_reference: float
    novelty_bonus: float
    record_min_vividness: float
    vividness_max: float
    symbol_min_vividness: float
    decay: float
    floor: float
    detail_loss_chance: float
    year_span_fractions: tuple[float, ...]
    region_cells: int
    merge_chance: float
    merge_carry: float
    max_provenance: int
    max_actors: int
    max_symbols: int
    substitution_chance: float
    repeat_gain: float
    borrow_contact_rate: float
    borrow_conquest_rate: float
    borrow_vividness: float
    spread_rate: float
    holding_loss: float
    shared_majority: float
    heavy_symbols: int
    summary_weight_medians: float
    summary_age_reference: float
    consequences: MemoryConsequencesConfig


# --- output and rendering ----------------------------------------------------------------------


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str
    lineages_every_years: int
    polities_every_years: int
    memory_snapshot_every_years: int
    checkpoint_every_years: int
    checkpoints_kept: int
    batch_workers: int
    batch_run_memory_mb: int
    batch_memory_reserve_mb: int
    bounds_file: str


@dataclass(frozen=True)
class VizConfig:
    play_interval_ms: int
    play_interval_max_ms: int
    render_interval: int
    render_interval_max: int
    step_buttons: tuple[int, ...]
    lineage_colors: tuple[str, ...]
    polity_shade_by: tuple[str, ...]
    polity_density_min: float
    polity_density_max: float
    polity_shade_levels: int
    polity_light: float
    polity_dark: float
    border_peoples_width: float
    border_polities_width: float
    border_outline_width: float
    border_color: str
    border_outline_color: str
    resource_colors: Mapping[str, str]
    knowledge_color: str
    knowledge_vmax: float
    latent_color: str
    latent_vmax: float
    population_color: str
    population_vmax: float
    population_alpha_curve: float
    polity_output_min: float
    polity_output_max: float
    mode_colors: Mapping[str, str]
    cohesion_cmap: str
    capital_px: float
    monument_px: float
    monument_color: str
    war_color: str
    war_px: float
    teachers_color: str
    teachers_px: float
    map_height_px: int
    max_zoom: float
    panel_interval: int
    chart_points: int
    chart_width_px: int
    chart_height_px: int
    tree_width_px: int
    tree_row_px: float
    tree_max_height_px: int
    inspector_symbols: int
    inspector_events: int
    inspector_origins: int
    inspector_file: str


@dataclass(frozen=True)
class RenderConfig:
    dpi: int
    hillshade_azimuth_deg: float
    hillshade_altitude_deg: float
    vertical_exaggeration: float
    hillshade_ambient: float
    river_min_order: int
    river_width_min: float
    river_width_max: float
    coast_linewidth: float
    flood_years: tuple[int, ...]
    population_years: tuple[int, ...]
    population_density_min: float
    population_density_max: float
    knowledge_years: tuple[int, ...]


@dataclass(frozen=True)
class ReportsConfig:
    origins_until_year: int
    narrator_file: str
    sample_names: int


@dataclass(frozen=True)
class ChronicleConfig:
    generation_years: int
    death_share: float
    page_entries: int
    polity_actions: tuple[str, ...]
    capital_actions: tuple[str, ...]
    share_actions: tuple[str, ...]
    teachers_actions: tuple[str, ...]
    teachers_min_magnitude: float
    always_actions: tuple[str, ...]
    first_actions: tuple[str, ...]
    crossing_actions: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    run: RunConfig
    rng: RngConfig
    grid: GridConfig
    world: WorldConfig
    climate: ClimateConfig
    capacity: CapacityConfig
    founders: FoundersConfig
    demography: DemographyConfig
    migration: MigrationConfig
    knowledge: KnowledgeConfig
    teachers: TeachersConfig
    events: EventsConfig
    polities: PolitiesConfig
    ancestry: AncestryConfig
    war: WarConfig
    brains: BrainsConfig
    memory: MemoryConfig
    output: OutputConfig
    viz: VizConfig
    render: RenderConfig
    reports: ReportsConfig
    chronicle: ChronicleConfig

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    def hash(self) -> str:
        """Stable hash of the full config, for run metadata."""
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with nested overrides applied, e.g. {"run": {"history_seed": 7}}."""
        return _convert(Config, _deep_merge(self.to_dict(), overrides), "config")


def read_overrides(path: str | Path) -> dict[str, Any]:
    """A YAML file of nested overrides (only the keys it changes), e.g.
    ``configs/worlds/small_test.yaml``."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    knowledge = raw.get("knowledge")
    if isinstance(knowledge, dict) and "techniques" not in knowledge and "graph_file" in knowledge:
        # The knowledge graph is a separate data file, embedded so the config (and its
        # hash) is complete on its own.
        with open(Path(path).parent / knowledge["graph_file"], encoding="utf-8") as f:
            knowledge["techniques"] = (yaml.safe_load(f) or {}).get("techniques", {})
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
