"""Static PNG maps of world layers (matplotlib, headless). Rendering only reads world data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import Normalize, to_rgba  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from deluge.config import RenderConfig  # noqa: E402
from deluge.world.grid import SQRT3, HexGeometry  # noqa: E402
from deluge.world.worlddata import Terrain, WorldData  # noqa: E402

TERRAIN_COLORS = {
    Terrain.SEA: "#2b5d8a",
    Terrain.PLAIN: "#b9cf7a",
    Terrain.FOREST: "#4f8a45",
    Terrain.HILLS: "#a99363",
    Terrain.MOUNTAIN: "#8a7b70",
    Terrain.DESERT: "#e6cc8f",
    Terrain.STEPPE: "#cfc38a",
    Terrain.WETLAND: "#6aa89c",
    Terrain.TUNDRA: "#b7bfa9",
    Terrain.ICE: "#f4f6f8",
}
SEA_COLOR = "#2b5d8a"
LAND_COLOR = "#d8cfa8"
FLOODED_COLOR = "#e07b39"
NO_VALUE_COLOR = "#bdbdbd"
COAST_COLOR = "#1d2f40"
RIVER_COLOR = "#2a6fb8"


def _axes(geom: HexGeometry, title: str, dpi: int, size=(14, 8.2)):
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    _style(ax, geom, title)
    return fig, ax


def _style(ax, geom: HexGeometry, title: str) -> None:
    pos = geom.positions
    ax.set_xlim(pos[:, 0].min() - 1.2, pos[:, 0].max() + 1.2)
    ax.set_ylim(pos[:, 1].min() - 1.2, pos[:, 1].max() + 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def _hexes(ax, geom: HexGeometry, colors):
    coll = PolyCollection(geom.hex_vertices(), linewidths=0, antialiased=False)
    coll.set_facecolor(colors)
    coll.set_edgecolor("face")
    ax.add_collection(coll)
    return coll


def _lat_axis(ax, geom: HexGeometry, lat_south: float, lat_north: float) -> None:
    ys = geom.positions[:, 1]
    y0, y1 = ys.min(), ys.max()
    ticks = [lat for lat in range(-60, 71, 20) if lat_south <= lat <= lat_north]
    ax.set_yticks([y0 + (lat - lat_south) / (lat_north - lat_south) * (y1 - y0) for lat in ticks])
    ax.set_yticklabels([f"{abs(t)}°{'N' if t > 0 else 'S' if t < 0 else ''}" for t in ticks])


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --- physical map -------------------------------------------------------------------------------


def hillshade(geom: HexGeometry, elev: np.ndarray, sea_level: float, unit_km: float,
              rcfg: RenderConfig) -> np.ndarray:
    """Lambertian shade factor per cell from the least-squares slope over the six
    neighbours, normalized so flat ground keeps its base colour."""
    h = np.maximum(elev, sea_level)
    nb = geom.neighbors
    valid = nb >= 0
    dh = np.where(valid, h[np.where(valid, nb, 0)] - h[:, None], 0.0)
    dp = (geom.positions[np.where(valid, nb, 0)] - geom.positions[:, None, :]) * unit_km * 1000.0
    dp = np.where(valid[:, :, None], dp, 0.0)
    spacing = SQRT3 * unit_km * 1000.0
    g = (dh[:, :, None] * dp).sum(axis=1) / (3.0 * spacing**2) * rcfg.vertical_exaggeration
    normal = np.stack([-g[:, 0], -g[:, 1], np.ones(len(h))], axis=1)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    az, alt = np.deg2rad(rcfg.hillshade_azimuth_deg), np.deg2rad(rcfg.hillshade_altitude_deg)
    light = np.array([np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)])
    shade = np.clip(normal @ light, 0.0, 1.0) / np.sin(alt)
    return rcfg.hillshade_ambient + (1.0 - rcfg.hillshade_ambient) * shade


def _sea_colors(depth: np.ndarray) -> np.ndarray:
    """Bathymetry: pale over the shelf, dark in the deep ocean."""
    t = np.clip(np.log1p(np.maximum(depth, 0.0)) / np.log1p(6000.0), 0.0, 1.0)
    shallow = np.array(to_rgba("#9cc7de"))
    deep = np.array(to_rgba("#173a5c"))
    return shallow[None, :] * (1 - t[:, None]) + deep[None, :] * t[:, None]


def _coast_segments(geom: HexGeometry, land: np.ndarray) -> np.ndarray:
    """Hex edges between land and sea cells."""
    nb = geom.neighbors
    valid = nb >= 0
    other_sea = valid & ~land[np.where(valid, nb, 0)]
    c, k = np.nonzero(land[:, None] & other_sea)
    p, q = geom.positions[c], geom.positions[nb[c, k]]
    mid = (p + q) / 2.0
    d = (q - p) / SQRT3
    perp = np.stack([-d[:, 1], d[:, 0]], axis=1) * 0.5
    return np.stack([mid - perp, mid + perp], axis=1)


def _river_lines(geom: HexGeometry, world: WorldData, land: np.ndarray, rcfg: RenderConfig):
    """River segments of Strahler order ``river_min_order`` and up, wider with order."""
    order = world.river_order.ravel()
    recv = world.flow_to.ravel()
    cells = np.flatnonzero((order >= rcfg.river_min_order) & land & (recv >= 0))
    pos = geom.positions
    ends = pos[recv[cells]].copy()
    into_sea = ~land[recv[cells]]
    ends[into_sea] = (pos[cells[into_sea]] + pos[recv[cells[into_sea]]]) / 2.0
    segs = np.stack([pos[cells], ends], axis=1)
    span = max(int(order.max()) - rcfg.river_min_order, 1)
    widths = rcfg.river_width_min + (rcfg.river_width_max - rcfg.river_width_min) * (
        order[cells] - rcfg.river_min_order) / span
    return segs, widths


def physical_colors(geom: HexGeometry, world: WorldData, sea_level: float, unit_km: float,
                    rcfg: RenderConfig) -> tuple[np.ndarray, np.ndarray]:
    elev = world.elevation.ravel()
    land = elev >= sea_level
    terrain = world.terrain.ravel()
    base = np.array([to_rgba(TERRAIN_COLORS[Terrain(t)]) for t in range(len(Terrain))])
    colors = base[terrain].copy()
    shade = hillshade(geom, elev, sea_level, unit_km, rcfg)
    colors[:, :3] = np.clip(colors[:, :3] * shade[:, None], 0.0, 1.0)
    colors[~land] = _sea_colors(sea_level - elev[~land])
    return colors, land


def physical_map(geom: HexGeometry, world: WorldData, sea_level: float, title: str, path: Path,
                 unit_km: float, rcfg: RenderConfig, lats: tuple[float, float],
                 marks: list[tuple[int, str]] = ()) -> Path:
    fig, ax = _axes(geom, title, rcfg.dpi)
    _draw_physical(ax, geom, world, sea_level, unit_km, rcfg, marks=marks)
    handles = [Patch(color=TERRAIN_COLORS[t], label=t.key) for t in Terrain if t != Terrain.SEA]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.85, ncol=3)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def _draw_physical(ax, geom, world, sea_level, unit_km, rcfg, flooded_from: float | None = None,
                   marks=()):
    colors, land = physical_colors(geom, world, sea_level, unit_km, rcfg)
    if flooded_from is not None:
        elev = world.elevation.ravel()
        drowned = (elev >= flooded_from) & ~land
        colors[drowned] = to_rgba(FLOODED_COLOR)
    _hexes(ax, geom, colors)
    segs, widths = _river_lines(geom, world, land, rcfg)
    ax.add_collection(LineCollection(segs, colors=RIVER_COLOR, linewidths=widths, capstyle="round"))
    ax.add_collection(LineCollection(_coast_segments(geom, land), colors=COAST_COLOR,
                                     linewidths=rcfg.coast_linewidth))
    for cell, label in marks:
        x, y = geom.positions[cell]
        ax.plot([x], [y], marker="o", markersize=11, markerfacecolor="none",
                markeredgecolor="#d7191c", markeredgewidth=2.0)
        ax.annotate(label, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=8,
                    color="#d7191c", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))


def flooding_panels(geom: HexGeometry, world: WorldData, sea_levels: dict[int, float], path: Path,
                    unit_km: float, cell_area_km2: float, rcfg: RenderConfig,
                    lats: tuple[float, float], marks: list[tuple[int, str]] = ()) -> Path:
    """Physical map per year; land drowned since the first year shown in orange."""
    years = sorted(sea_levels)
    elev = world.elevation.ravel()
    first = sea_levels[years[0]]
    land0 = elev >= first
    fig, axes = plt.subplots(2, 2, figsize=(18, 11.5), dpi=rcfg.dpi)
    for ax, year in zip(axes.ravel(), years):
        lvl = sea_levels[year]
        drowned = int((land0 & (elev < lvl)).sum())
        _style(ax, geom, f"year {year}: sea level {lvl:+.0f} m, drowned since year {years[0]}: "
                         f"{drowned} cells, {drowned * cell_area_km2 / 1e6:.2f} M km2 "
                         f"({drowned / land0.sum():.1%})")
        _draw_physical(ax, geom, world, lvl, unit_km, rcfg, flooded_from=first, marks=marks)
        _lat_axis(ax, geom, *lats)
    fig.legend(handles=[Patch(color=FLOODED_COLOR, label=f"land drowned since year {years[0]}")],
               loc="lower center", ncol=1)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


# --- single layers -----------------------------------------------------------------------------


def field_map(geom: HexGeometry, values: np.ndarray, sea: np.ndarray, title: str, path: Path,
              cmap: str, norm, label: str, lats: tuple[float, float], dpi: int) -> Path:
    """Continuous field on land; sea in blue, land with no finite value (e.g. zero
    capacity on a log scale) in grey."""
    fig, ax = _axes(geom, title, dpi)
    v = values.astype(float)
    cm = plt.get_cmap(cmap)
    colors = np.empty((len(v), 4))
    ok = ~sea & np.isfinite(v)
    colors[ok] = cm(norm(v[ok]))
    colors[~sea & ~ok] = to_rgba(NO_VALUE_COLOR)
    colors[sea] = to_rgba(SEA_COLOR)
    _hexes(ax, geom, colors)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.8, label=label)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def terrain_map(geom: HexGeometry, terrain: np.ndarray, title: str, path: Path,
                lats: tuple[float, float], dpi: int) -> Path:
    fig, ax = _axes(geom, title, dpi)
    _hexes(ax, geom, [TERRAIN_COLORS[Terrain(t)] for t in terrain])
    counts = np.bincount(terrain, minlength=len(Terrain))
    handles = [Patch(color=TERRAIN_COLORS[t], label=f"{t.key} ({counts[t]})") for t in Terrain]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85, ncol=2)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def elevation_map(geom: HexGeometry, elev: np.ndarray, sea_level: float, title: str, path: Path,
                  lats: tuple[float, float], dpi: int) -> Path:
    fig, ax = _axes(geom, title, dpi)
    land = elev >= sea_level
    colors = np.empty((len(elev), 4))
    top = max(elev.max(), sea_level + 1)
    land_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 64)))
    colors[land] = land_cmap(Normalize(sea_level, top)(elev[land]))
    colors[~land] = _sea_colors(sea_level - elev[~land])
    _hexes(ax, geom, colors)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(sea_level, top), cmap=land_cmap), ax=ax,
                 shrink=0.8, label="land elevation (m, present sea level = 0)")
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def categorical_map(geom: HexGeometry, title: str, path: Path, lats: tuple[float, float],
                    legend: list[tuple[str, str]], colors, dpi: int) -> Path:
    fig, ax = _axes(geom, title, dpi)
    _hexes(ax, geom, colors)
    ax.legend(handles=[Patch(color=c, label=t) for t, c in legend], loc="lower left", fontsize=8,
              framealpha=0.85)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def climate_curves(sea_level: np.ndarray, temp_anomaly: np.ndarray, path: Path, dpi: int) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), dpi=dpi, sharex=True)
    years = np.arange(len(sea_level))
    a1.plot(years, sea_level, color="#2b5d8a")
    a1.set_ylabel("sea level (m vs present)")
    a1.grid(alpha=0.3)
    a2.plot(years[: len(temp_anomaly)], temp_anomaly, color="#c0392b", linewidth=0.6)
    a2.axhline(0, color="k", linewidth=0.5)
    a2.set_ylabel("temperature anomaly (°C)")
    a2.set_xlabel("year")
    a2.grid(alpha=0.3)
    a1.set_title("Climate forcing over the run")
    return _save(fig, path)
