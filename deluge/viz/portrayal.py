"""Map layer portrayal for the browser UI.

The map is drawn by Mesa's own ``SpaceRenderer`` (matplotlib backend), which draws every
property layer of the HexGrid as a PolyCollection on its hex mesh. Its mesh shifts even
rows right, like Mesa's connectivity and ``deluge.world.grid``; it does not use
``cell.position`` (see docs/MESA_API_NOTES.md). Two layers are drawn: terrain, then the
year's sea on top of it. Everything here only reads model state.
"""

from __future__ import annotations

from matplotlib.colors import ListedColormap
from mesa.visualization.components import PropertyLayerStyle

from deluge.config import VizConfig
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS, _lat_axis
from deluge.world.worlddata import Terrain

TERRAIN_CMAP = ListedColormap([TERRAIN_COLORS[t] for t in Terrain], name="terrain_classes")
MAP_LAYERS = ("terrain", "sea")


def map_layer_style(layer) -> PropertyLayerStyle | None:
    """PropertyLayerStyle for the layers the map draws; None hides all other layers."""
    if layer.name == "terrain":
        # One colour per class: class k spans [k - 0.5, k + 0.5).
        return PropertyLayerStyle(colormap=TERRAIN_CMAP, alpha=1.0, colorbar=False,
                                  vmin=-0.5, vmax=len(Terrain) - 0.5)
    if layer.name == "sea":
        # Mesa maps the value to opacity: sea (1) fully covers the terrain, land (0) is clear.
        return PropertyLayerStyle(color=SEA_COLOR, alpha=1.0, colorbar=False, vmin=0.0, vmax=1.0)
    return None


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
        f"Seed: {model.config.run.seed}",
    ]
    return "  \n".join(lines)
