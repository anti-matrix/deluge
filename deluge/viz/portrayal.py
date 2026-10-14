"""Map layer portrayal for the browser UI.

The map is drawn by Mesa's own ``SpaceRenderer`` (matplotlib backend), which draws every
property layer of the HexGrid as a PolyCollection on its hex mesh. Its mesh shifts even
rows right, like Mesa's connectivity and ``deluge.world.grid``; it does not use
``cell.position`` (see docs/MESA_API_NOTES.md). Three layers are drawn, in the order
the model added them: terrain, the year's sea over it, and people per cell on top.
Everything here only reads model state.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
from mesa.visualization.components import PropertyLayerStyle

from deluge.config import VizConfig
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS, _lat_axis
from deluge.world.worlddata import Terrain

TERRAIN_CMAP = ListedColormap([TERRAIN_COLORS[t] for t in Terrain], name="terrain_classes")
MAP_LAYERS = ("terrain", "sea", "population")


def population_cmap(vcfg: VizConfig, n: int = 256) -> ListedColormap:
    """One colour whose opacity rises like log(1 + c x): Mesa normalizes layer values
    linearly, so this keeps bands of a few hundred people visible next to full cells."""
    x = np.linspace(0.0, 1.0, n)
    c = vcfg.population_alpha_curve
    alpha = np.log1p(c * x) / np.log1p(c)
    rgb = np.tile(to_rgb(vcfg.population_color), (n, 1))
    return ListedColormap(np.column_stack([rgb, alpha]), name="population")


def make_layer_style(vcfg: VizConfig):
    """``fn(layer) -> PropertyLayerStyle | None`` for the layers the map draws; None
    hides all other layers."""
    pop_cmap = population_cmap(vcfg)

    def style(layer) -> PropertyLayerStyle | None:
        if layer.name == "terrain":
            # One colour per class: class k spans [k - 0.5, k + 0.5).
            return PropertyLayerStyle(colormap=TERRAIN_CMAP, alpha=1.0, colorbar=False,
                                      vmin=-0.5, vmax=len(Terrain) - 0.5)
        if layer.name == "sea":
            # Mesa maps the value to opacity: sea (1) fully covers the terrain, land (0) is clear.
            return PropertyLayerStyle(color=SEA_COLOR, alpha=1.0, colorbar=False, vmin=0.0, vmax=1.0)
        if layer.name == "population":
            return PropertyLayerStyle(colormap=pop_cmap, alpha=1.0, colorbar=False,
                                      vmin=0.0, vmax=vcfg.population_vmax)
        return None

    return style


def make_axes_styler(model, vcfg: VizConfig):
    """Post-process hook for the renderer's axes: size, limits, latitude labels.

    Mesa applies it once per renderer; the limits stay because each frame only replaces
    the drawn artists."""

    def style(ax) -> None:
        fig = ax.get_figure()
        fig.set_size_inches(vcfg.figure_width_in, vcfg.figure_height_in)
        fig.set_dpi(vcfg.figure_dpi)
        pos = model.geom.positions
        ax.set_xlim(pos[:, 0].min() - 1.2, pos[:, 0].max() + 1.2)
        ax.set_ylim(pos[:, 1].min() - 1.2, pos[:, 1].max() + 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        _lat_axis(ax, model.geom, model.config.world.lat_south, model.config.world.lat_north)

    return style


def shown_year(model) -> int:
    """The year the map shows: the last simulated year (year 0 before the first step,
    when the model already holds year 0's sea)."""
    return max(model.year - 1, 0)


def status_text(model) -> str:
    year = shown_year(model)
    lines = [
        f"**Year {year:,}** of {model.config.run.years:,}",
        f"Sea level: {model.sea_level_series[year]:+.1f} m",
        f"Land cells: {int(model.is_sea.size - model.is_sea.sum()):,}",
        f"Temperature anomaly: {model.temp_anomaly:+.2f} °C",
        f"Population: {model.population:,.0f} in {model.n_settlements:,} settlements",
        "By group: " + ", ".join(f"{name} {p:,.0f}" for name, p in
                                 zip(model.lineages.names, model.lineage_population)),
        f"World seed: {model.config.run.world_seed}, history seed: {model.config.run.history_seed}",
    ]
    return "  \n".join(lines)
