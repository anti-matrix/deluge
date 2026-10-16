"""Browser UI entry point (spec section 12).

    solara run deluge/viz/app.py

Shows the hex map (terrain, with the year's sea drawn over it, and a chosen overlay:
people or techniques known per cell, the founding group of each settlement, mineral
deposits, or the polities: claimed cells in their founding group's hue shaded by a chosen
variable, with borders; with a colour key), Mesa's play / pause / step / reset controls,
buttons that step 1, 10 or 100 years, a year counter, plots of sea level, population per
founding group and per subsistence mode, and the number of polities, and a chronicle of the whole event log (a page
at a time, newest first; filtered by group, event type, year range and text; downloadable
as a text file). The world seed (the map) and the history seed are
integer fields that take effect on Reset.

Built from Mesa's own visualization pieces (SpaceRenderer and its component,
ModelController, the plot component) in a plain Solara layout rather than ``SolaraViz``:
with the installed ipyvue 3 / Vue 3 frontend, the draggable grid that ``SolaraViz`` puts
its components in (``solara.GridDraggable``, a Vue 2 plugin) fails to load and the main
area stays blank (see docs/MESA_API_NOTES.md).

The UI only reads model state and steps the model through ``DelugeModel.step``, the same
call a headless run makes, so a UI run and a headless run with the same seeds agree.
"""

from __future__ import annotations

import solara
from mesa.visualization import SpaceRenderer, make_plot_component
from mesa.visualization.solara_viz import ModelController
from mesa.visualization.utils import force_update, update_counter

from deluge.agents.settlement import MODES
from deluge.config import load_config
from deluge.data import chronicle
from deluge.data.events import ACTIONS
from deluge.model import DelugeModel
from deluge.viz import portrayal

CONFIG = load_config()
VIZ = CONFIG.viz
SEEDS = ("world_seed", "history_seed")


def advance(model: DelugeModel, years: int) -> None:
    """Step the model up to ``years`` times, stopping at the end of the run."""
    for _ in range(years):
        if not model.running:
            break
        model.step()


def make_renderer(model: DelugeModel, overlay: str = "population") -> SpaceRenderer:
    """Mesa's SpaceRenderer with the terrain, sea and overlay layers drawn (no grid
    lines: at 15,000 cells they would hide the map). The axes styling only depends on the
    grid size, which is the same for every seed, so Mesa can reuse it after a Reset."""
    renderer = SpaceRenderer(model, backend="matplotlib")
    renderer.setup_propertylayer(portrayal.make_layer_style(VIZ, overlay))
    renderer.post_process = portrayal.make_axes_styler(model, VIZ)
    renderer.draw_propertylayer()
    return renderer


_sea_level_plot, _ = make_plot_component("sea_level")
# Population per founding group, in the lineage palette (the map and legends use the same).
GROUP_PLOT_COLORS = {f"pop_{name}": color for name, color in zip(CONFIG.founders.names, portrayal.lineage_palette(VIZ))}
_group_plot, _ = make_plot_component(GROUP_PLOT_COLORS)
if set(VIZ.polity_shade_by) - set(portrayal.SHADE_BY):
    raise KeyError(f"viz.polity_shade_by: unknown {sorted(set(VIZ.polity_shade_by) - set(portrayal.SHADE_BY))}")
_mode_plot, _ = make_plot_component([f"pop_{mode}" for mode in MODES])
_polity_plot, _ = make_plot_component("n_polities")


@solara.component
def Status(model: DelugeModel):
    update_counter.get()
    solara.Markdown(portrayal.status_text(model))


@solara.component
def StepButtons(model: DelugeModel):
    update_counter.get()

    def stepper(years: int):
        def on_click():
            advance(model, years)
            force_update()

        return on_click

    with solara.Row():
        for years in VIZ.step_buttons:
            solara.Button(f"+{years} yr", color="primary", on_click=stepper(years),
                          disabled=not model.running)


@solara.component
def SeedInputs(model_parameters):
    """Integer fields for the two seeds; Mesa's Reset builds the model from them."""

    def setter(name: str):
        def on_value(value):
            if value is not None:
                model_parameters.value = {**model_parameters.value, name: int(value)}

        return on_value

    solara.InputInt("World seed: the map (applies on Reset)", value=model_parameters.value["world_seed"],
                    on_value=setter("world_seed"))
    solara.InputInt("History seed (applies on Reset)", value=model_parameters.value["history_seed"],
                    on_value=setter("history_seed"))


@solara.component
def OverlaySelect(renderer, overlay, shade_by):
    def on_value(value: str):
        overlay.set(value)
        renderer.value.propertylayer_portrayal = portrayal.make_layer_style(VIZ, value)
        force_update()

    solara.Select("Map overlay", values=list(portrayal.OVERLAYS), value=overlay.value, on_value=on_value)
    if overlay.value == "polity":
        def on_shade(value: str):
            shade_by.set(value)
            force_update()

        solara.Select("Shade by", values=list(VIZ.polity_shade_by), value=shade_by.value, on_value=on_shade)
    legend = portrayal.legend_html(VIZ, overlay.value, CONFIG.founders.names)
    if legend:
        solara.HTML(tag="div", unsafe_innerHTML=legend)


def draw_map(model: DelugeModel, renderer: SpaceRenderer, overlay: str, shade_by: str):
    """One frame of the map, as Mesa's SpaceRendererComponent draws it (matplotlib), plus the
    polity overlay: clear the artists, draw the property layers, style the axes once, then the
    polity fill and borders. Returns the figure."""
    ax = renderer.canvas
    for artist in [*ax.lines, *ax.collections, *ax.patches, *ax.images, *ax.artists]:
        artist.remove()
    for cbar in renderer.backend_renderer._active_colorbars:
        cbar.remove()
    renderer.backend_renderer._active_colorbars.clear()
    if renderer.propertylayer_mesh:
        renderer.draw_propertylayer()
    if renderer.post_process and not renderer._post_process_applied:
        renderer.post_process(ax)
        renderer._post_process_applied = True
    if overlay == "polity":
        portrayal.draw_polities(ax, model, VIZ, shade_by)
    return ax.get_figure()


@solara.component
def MapView(model: DelugeModel, renderer: SpaceRenderer, overlay: str, shade_by: str):
    update_counter.get()
    renderer.space = model.grid
    fig = draw_map(model, renderer, overlay, shade_by)
    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight",
                            dependencies=[update_counter.value, overlay, shade_by])


@solara.component
def Chronicle(model: DelugeModel):
    """The whole event log, a page at a time (newest first), filtered by group, event
    type, year range and text; the full chronicle downloads as a text file."""
    update_counter.get()
    view = solara.use_memo(lambda: chronicle.Chronicle(model), dependencies=[model])
    names = list(model.lineages.names)
    groups, set_groups = solara.use_state(names)
    actions, set_actions = solara.use_state(list(ACTIONS))
    year_min, set_year_min = solara.use_state(None)
    year_max, set_year_max = solara.use_state(None)
    text, set_text = solara.use_state("")
    page, set_page = solara.use_state(0)

    def reset_page(setter):
        def on_value(value):
            setter(value)
            set_page(0)

        return on_value

    solara.SelectMultiple("Groups", values=groups, all_values=names, on_value=reset_page(set_groups))
    solara.SelectMultiple("Event types", values=actions, all_values=list(ACTIONS), on_value=reset_page(set_actions))
    with solara.Row():
        solara.InputInt("From year", value=year_min, optional=True, clearable=True,
                        on_value=reset_page(set_year_min))
        solara.InputInt("To year", value=year_max, optional=True, clearable=True,
                        on_value=reset_page(set_year_max))
        solara.InputText("Search", value=text, on_value=reset_page(set_text), clearable=True)
    chosen = view.select(groups={names.index(g) for g in groups}, actions=set(actions),
                         year_min=year_min, year_max=year_max, text=text or "")
    pages = view.pages(chosen, VIZ.chronicle_lines)
    shown_page = min(page, pages - 1)
    with solara.Row():
        solara.Button("Newer", on_click=lambda: set_page(max(shown_page - 1, 0)), disabled=shown_page == 0)
        solara.Button("Older", on_click=lambda: set_page(shown_page + 1), disabled=shown_page >= pages - 1)
        solara.Text(f"Page {shown_page + 1:,} of {pages:,}: {len(chosen):,} of {len(view):,} events")
        solara.FileDownload(lambda: view.text(), filename="chronicle.txt", label="Download whole chronicle",
                            mime_type="text/plain")
    shown = view.page(chosen, shown_page, VIZ.chronicle_lines)
    with solara.Div(style={"max-height": "420px", "overflow-y": "auto"}):
        solara.Markdown("  \n".join(shown) if shown else "No events.")


@solara.component
def Plots(model: DelugeModel):
    update_counter.get()
    with solara.Row():
        _sea_level_plot(model)
        _group_plot(model)
        _mode_plot(model)
        _polity_plot(model)


_initial: list = []


def initial_state() -> tuple[DelugeModel, SpaceRenderer]:
    """The model and renderer the page opens with, built on first use (not at import, so
    importing this module stays cheap)."""
    if not _initial:
        model = DelugeModel(world_seed=CONFIG.run.world_seed, history_seed=CONFIG.run.history_seed)
        _initial.extend((model, make_renderer(model)))
    return _initial[0], _initial[1]


@solara.component
def Page():
    initial_model, initial_renderer = initial_state()
    model = solara.use_reactive(initial_model)
    renderer = solara.use_reactive(initial_renderer)
    model_parameters = solara.use_reactive({name: getattr(CONFIG.run, name) for name in SEEDS})
    overlay = solara.use_reactive("population")
    shade_by = solara.use_reactive(VIZ.polity_shade_by[0])
    play_interval = solara.use_reactive(VIZ.play_interval_ms)
    render_interval = solara.use_reactive(VIZ.render_interval)
    use_threads = solara.use_reactive(False)

    solara.Title("Deluge")
    # Plain columns: with the Vue 3 frontend, solara.Sidebar overlays the main area
    # instead of pushing it aside, which hides the left part of the map.
    with solara.Columns([1, 3]):
        with solara.Column():
            with solara.Card("Controls"):
                solara.SliderInt("Pause between frames (ms)", value=play_interval, min=1,
                                 max=VIZ.play_interval_max_ms)
                solara.SliderInt("Years per frame / Step", value=render_interval, min=1,
                                 max=VIZ.render_interval_max)
                ModelController(model, renderer=renderer, model_parameters=model_parameters,
                                play_interval=play_interval, render_interval=render_interval,
                                use_threads=use_threads)
                StepButtons(model.value)
            with solara.Card("Model parameters"):
                SeedInputs(model_parameters)
            with solara.Card("Map"):
                OverlaySelect(renderer, overlay, shade_by)
            with solara.Card("Year"):
                Status(model.value)
        with solara.Column():
            MapView(model.value, renderer.value, overlay.value, shade_by.value)
            Plots(model.value)
            with solara.Card("Chronicle"):
                Chronicle(model.value)

