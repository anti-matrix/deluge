"""Static PNG maps of world layers (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from deluge.world.grid import HexGeometry  # noqa: E402
from deluge.world.procedural import flow_receivers  # noqa: E402
from deluge.world.worlddata import Terrain, WorldData  # noqa: E402

TERRAIN_COLORS = {
    Terrain.SEA: "#2b5d8a",
    Terrain.COAST: "#e8d9a0",
    Terrain.PLAIN: "#b5cf6b",
    Terrain.FOREST: "#3f7f3a",
    Terrain.HILLS: "#a68c5b",
    Terrain.MOUNTAIN: "#7d7068",
    Terrain.DESERT: "#e3c27a",
    Terrain.STEPPE: "#cbbf7e",
    Terrain.WETLAND: "#5fa39a",
}
SEA_COLOR = "#2b5d8a"
LAND_COLOR = "#d8cfa8"
FLOODED_COLOR = "#e07b39"
NO_VALUE_COLOR = "#bdbdbd"


def _axes(geom: HexGeometry, title: str):
    fig, ax = plt.subplots(figsize=(13, 7.4), dpi=110)
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


def field_map(geom: HexGeometry, values: np.ndarray, sea: np.ndarray, title: str, path: Path,
              cmap: str, norm, label: str, lats: tuple[float, float]) -> Path:
    """Continuous field on land; sea in blue, land with no finite value (e.g. zero
    capacity on a log scale) in grey."""
    fig, ax = _axes(geom, title)
    v = values.astype(float)
    cm = plt.get_cmap(cmap)
    colors = np.empty((len(v), 4))
    ok = ~sea & np.isfinite(v)
    colors[ok] = cm(norm(v[ok]))
    colors[~sea & ~ok] = matplotlib.colors.to_rgba(NO_VALUE_COLOR)
    colors[sea] = matplotlib.colors.to_rgba(SEA_COLOR)
    _hexes(ax, geom, colors=colors)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm), ax=ax, shrink=0.8, label=label)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def terrain_map(geom: HexGeometry, terrain: np.ndarray, title: str, path: Path,
                lats: tuple[float, float]) -> Path:
    fig, ax = _axes(geom, title)
    _hexes(ax, geom, colors=[TERRAIN_COLORS[Terrain(t)] for t in terrain])
    counts = np.bincount(terrain, minlength=len(Terrain))
    handles = [Patch(color=TERRAIN_COLORS[t], label=f"{t.key} ({counts[t]})") for t in Terrain]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.85)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def elevation_map(geom: HexGeometry, elev: np.ndarray, sea_level: float, title: str, path: Path,
                  lats: tuple[float, float]) -> Path:
    fig, ax = _axes(geom, title)
    land = elev >= sea_level
    colors = np.empty((len(elev), 4))
    colors[land] = plt.get_cmap("terrain")(
        0.25 + 0.75 * Normalize(sea_level, max(elev.max(), sea_level + 1))(elev[land]))
    colors[~land] = plt.get_cmap("Blues_r")(
        0.15 + 0.6 * Normalize(elev.min(), sea_level)(elev[~land]))
    _hexes(ax, geom, colors=colors)
    top = max(elev.max(), sea_level + 1)
    land_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "land", plt.get_cmap("terrain")(np.linspace(0.25, 1.0, 64)))
    sm = plt.cm.ScalarMappable(norm=Normalize(sea_level, top), cmap=land_cmap)
    fig.colorbar(sm, ax=ax, shrink=0.8, label="land elevation (m, present sea level = 0)")
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def river_map(geom: HexGeometry, world: WorldData, sea_level: float, title: str, path: Path,
              lats: tuple[float, float]) -> Path:
    fig, ax = _axes(geom, title)
    elev = world.elevation.ravel()
    sea = elev < sea_level
    shade = plt.get_cmap("Greys")(0.15 + 0.45 * Normalize(sea_level, elev.max())(np.maximum(elev, sea_level)))
    shade[sea] = matplotlib.colors.to_rgba(SEA_COLOR)
    _hexes(ax, geom, colors=shade)
    river = world.river.ravel()
    recv = flow_receivers(geom, elev, sea_level)
    cells = np.flatnonzero(river > 0)
    pos = geom.positions
    segs = [(pos[c], pos[recv[c]]) for c in cells if recv[c] >= 0]
    widths = [0.6 + 1.6 * np.log1p(river[c]) / np.log1p(river.max()) for c in cells if recv[c] >= 0]
    ax.add_collection(LineCollection(segs, colors="#1f78d1", linewidths=widths, capstyle="round"))
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def categorical_map(geom: HexGeometry, title: str, path: Path, lats: tuple[float, float],
                    legend: list[tuple[str, str]], colors) -> Path:
    fig, ax = _axes(geom, title)
    _hexes(ax, geom, colors=colors)
    ax.legend(handles=[Patch(color=c, label=t) for t, c in legend], loc="lower left", fontsize=8,
              framealpha=0.85)
    _lat_axis(ax, geom, *lats)
    return _save(fig, path)


def land_sea_panels(geom: HexGeometry, elev: np.ndarray, sea_levels: dict[int, float], path: Path,
                    lats: tuple[float, float]) -> Path:
    """One panel per year: land, sea, and land lost since the first year shown."""
    years = sorted(sea_levels)
    first = elev >= sea_levels[years[0]]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.4), dpi=100)
    for ax, year in zip(axes.ravel(), years):
        land = elev >= sea_levels[year]
        colors = np.where(land, LAND_COLOR, SEA_COLOR).astype(object)
        colors[first & ~land] = FLOODED_COLOR
        _style(ax, geom, f"year {year}: sea level {sea_levels[year]:+.0f} m, "
                         f"land {land.sum()} cells ({(first & ~land).sum()} flooded since year {years[0]})")
        _hexes(ax, geom, colors=list(colors))
        _lat_axis(ax, geom, *lats)
    fig.legend(handles=[Patch(color=LAND_COLOR, label="land"), Patch(color=SEA_COLOR, label="sea"),
                        Patch(color=FLOODED_COLOR, label=f"flooded since year {years[0]}")],
               loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def land_sea_map(geom: HexGeometry, elev: np.ndarray, year: int, sea_level: float,
                 first_sea_level: float, path: Path, lats: tuple[float, float]) -> Path:
    land = elev >= sea_level
    flooded = (elev >= first_sea_level) & ~land
    colors = np.where(land, LAND_COLOR, SEA_COLOR).astype(object)
    colors[flooded] = FLOODED_COLOR
    return categorical_map(
        geom, f"Land and sea, year {year} (sea level {sea_level:+.0f} m): "
                    f"{land.sum()} land cells, {flooded.sum()} flooded since year 0",
        path, lats, [("land", LAND_COLOR), ("sea", SEA_COLOR), ("flooded since year 0", FLOODED_COLOR)],
        list(colors))


def climate_curves(sea_level: np.ndarray, temp_anomaly: np.ndarray, path: Path) -> Path:
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.5), dpi=110, sharex=True)
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
