"""Map layer portrayal for the browser UI.

The map is drawn by Mesa's own ``SpaceRenderer`` (matplotlib backend), which draws every
property layer of the HexGrid as a PolyCollection on its hex mesh. Its mesh shifts even
rows right, like Mesa's connectivity and ``deluge.world.grid``; it does not use
``cell.position`` (see docs/MESA_API_NOTES.md). Three layers are drawn, in the order
the model added them: terrain, the year's sea over it, and one overlay on top (people per
cell, the polity of each settlement (its founding group's colour, one shade per polity),
techniques usable per cell, latent techniques per cell: held where they cannot be used,
the founding group of each settlement, or mineral deposits). Everything here only reads
model state.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap, to_rgb, to_rgba
from mesa.visualization.components import PropertyLayerStyle

from deluge.config import VizConfig
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS, _lat_axis
from deluge.world.worlddata import MINERALS, Terrain

TERRAIN_CMAP = ListedColormap([TERRAIN_COLORS[t] for t in Terrain], name="terrain_classes")
MAP_LAYERS = ("terrain", "sea")
OVERLAYS = ("population", "polity", "knowledge", "latent_knowledge", "lineage", "resources")  # model layers drawn over the map


def alpha_cmap(color: str, curve: float, n: int = 256) -> ListedColormap:
    """One colour whose opacity rises like log(1 + c x) (linearly for c = 0): Mesa
    normalizes layer values linearly, so this keeps small values visible."""
    x = np.linspace(0.0, 1.0, n)
    alpha = np.log1p(curve * x) / np.log1p(curve) if curve > 0 else x
    rgb = np.tile(to_rgb(color), (n, 1))
    return ListedColormap(np.column_stack([rgb, alpha]))


def categorical_cmap(colors) -> ListedColormap:
    """Code 0 clear, code k (1..n) the k-th colour, fully opaque."""
    return ListedColormap([(0.0, 0.0, 0.0, 0.0)] + [to_rgba(c) for c in colors])


def resource_colors(vcfg: VizConfig) -> list[str]:
    """Resources overlay colours in code order: MINERALS, then several."""
    return [vcfg.resource_colors[m] for m in MINERALS] + [vcfg.resource_colors["several"]]


def shade(color: str, factor: float) -> tuple[float, float, float, float]:
    """``color`` darkened toward black (factor < 0) or lightened toward white (factor > 0)."""
    rgb = np.array(to_rgb(color))
    rgb = rgb * (1.0 + factor) if factor < 0 else rgb + (1.0 - rgb) * factor
    return (*rgb.tolist(), 1.0)


def polity_colors(vcfg: VizConfig) -> list[tuple]:
    """Polity overlay colours in code order (code 1 + lineage x shades + shade): each
    founding group's colour in ``polity_shades`` shades."""
    n, r = vcfg.polity_shades, vcfg.polity_shade_range
    factors = np.linspace(-r, r, n) if n > 1 else np.zeros(1)
    return [shade(c, float(f)) for c in vcfg.lineage_colors for f in factors]


def legend_html(vcfg: VizConfig, overlay: str, group_names) -> str:
    """Colour key for the categorical overlays ('' for the others)."""
    if overlay == "lineage":
        items = [(f"group {n}", c) for n, c in zip(group_names, vcfg.lineage_colors)]
    elif overlay == "polity":
        items = [(f"polities of group {n} (shades: one per polity)", c) for n, c in zip(group_names, vcfg.lineage_colors)]
    elif overlay == "resources":
        items = list(zip([m.replace("_", " ") for m in MINERALS] + ["several"], resource_colors(vcfg)))
    else:
        return ""
    return " ".join(f'<span style="display:inline-block;width:0.9em;height:0.9em;background:{c};'
                    f'border:1px solid #555;vertical-align:middle"></span> {label}' for label, c in items)


def make_layer_style(vcfg: VizConfig, overlay: str = "population"):
    """``fn(layer) -> PropertyLayerStyle | None`` for the layers the map draws: terrain,
    sea and the chosen ``overlay``; None hides all other layers."""
    if overlay not in OVERLAYS:
        raise ValueError(f"overlay must be one of {OVERLAYS}")
    overlay_style = {
        "population": PropertyLayerStyle(colormap=alpha_cmap(vcfg.population_color, vcfg.population_alpha_curve),
                                         alpha=1.0, colorbar=False, vmin=0.0, vmax=vcfg.population_vmax),
        "knowledge": PropertyLayerStyle(colormap=alpha_cmap(vcfg.knowledge_color, 0.0), alpha=1.0,
                                        colorbar=False, vmin=0.0, vmax=vcfg.knowledge_vmax),
        "latent_knowledge": PropertyLayerStyle(colormap=alpha_cmap(vcfg.latent_color, 0.0), alpha=1.0,
                                               colorbar=False, vmin=0.0, vmax=vcfg.latent_vmax),
        # Categorical: code k spans [k - 0.5, k + 0.5).
        "lineage": PropertyLayerStyle(colormap=categorical_cmap(vcfg.lineage_colors), alpha=1.0, colorbar=False,
                                      vmin=-0.5, vmax=len(vcfg.lineage_colors) + 0.5),
        "polity": PropertyLayerStyle(colormap=categorical_cmap(polity_colors(vcfg)), alpha=1.0, colorbar=False,
                                     vmin=-0.5, vmax=len(polity_colors(vcfg)) + 0.5),
        "resources": PropertyLayerStyle(colormap=categorical_cmap(resource_colors(vcfg)), alpha=1.0,
                                        colorbar=False, vmin=-0.5, vmax=len(MINERALS) + 1.5),
    }[overlay]

    def style(layer) -> PropertyLayerStyle | None:
        if layer.name == "terrain":
            # One colour per class: class k spans [k - 0.5, k + 0.5).
            return PropertyLayerStyle(colormap=TERRAIN_CMAP, alpha=1.0, colorbar=False,
                                      vmin=-0.5, vmax=len(Terrain) - 0.5)
        if layer.name == "sea":
            # Mesa maps the value to opacity: sea (1) fully covers the terrain, land (0) is clear.
            return PropertyLayerStyle(color=SEA_COLOR, alpha=1.0, colorbar=False, vmin=0.0, vmax=1.0)
        if layer.name == overlay:
            return overlay_style
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
    modes = ", ".join(f"{m} {p:,.0f}" for m, p in zip(("foraging", "herding", "farming"), model.mode_population))
    ts = model.teachers_state
    teachers = ("present" if ts.present else "departed" if ts.departed and ts.visits
                else "not arrived" if model.config.teachers.enabled else "disabled")
    lines = [
        f"**Year {year:,}** of {model.config.run.years:,}",
        f"Sea level: {model.sea_level_series[year]:+.1f} m",
        f"Land cells: {int(model.is_sea.size - model.is_sea.sum()):,}",
        f"Temperature anomaly: {model.temp_anomaly:+.2f} °C",
        f"Population: {model.population:,.0f} in {model.n_settlements:,} settlements",
        "By group: " + ", ".join(f"{name} {p:,.0f}" for name, p in
                                 zip(model.lineages.names, model.lineage_population)),
        f"By mode: {modes}",
        f"Techniques per settlement: {model.techniques_known:.1f}; teachers: {teachers}",
        f"Polities: {model.n_polities:,}; mean cohesion {model.mean_cohesion:.2f}; largest "
        f"{model.largest_polity_settlements:,} settlements, {model.largest_polity_population:,.0f} people",
        f"World seed: {model.config.run.world_seed}, history seed: {model.config.run.history_seed}",
    ]
    return "  \n".join(lines)
