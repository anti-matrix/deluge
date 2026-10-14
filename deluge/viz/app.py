"""Browser UI entry point (spec section 12, Round 2 skeleton).

    solara run deluge/viz/app.py

Shows the hex map (terrain, with the year's sea drawn over it), Mesa's play / pause /
step / reset controls, buttons that step 1, 10 or 100 years, a year counter and the sea
level curve so far. The seed slider takes effect on Reset.

Built from Mesa's own visualization pieces (SpaceRenderer and its component,
ModelController, ModelCreator, the plot component) in a plain Solara layout rather than
``SolaraViz``: with the installed ipyvue 3 / Vue 3 frontend, the draggable grid that
``SolaraViz`` puts its components in (``solara.GridDraggable``, a Vue 2 plugin) fails to
load and the main area stays blank (see docs/MESA_API_NOTES.md).

The UI only reads model state and steps the model through ``DelugeModel.step``, the same
call a headless run makes, so a UI run and a headless run with the same seed agree.
"""

from __future__ import annotations

import solara
from mesa.visualization import SpaceRenderer, make_plot_component
from mesa.visualization.solara_viz import ModelController, ModelCreator, SpaceRendererComponent
from mesa.visualization.utils import force_update, update_counter

from deluge.config import load_config
from deluge.model import DelugeModel
from deluge.viz import portrayal

CONFIG = load_config()
VIZ = CONFIG.viz


def advance(model: DelugeModel, years: int) -> None:
    """Step the model up to ``years`` times, stopping at the end of the run."""
    for _ in range(years):
        if not model.running:
            break
        model.step()


def make_renderer(model: DelugeModel) -> SpaceRenderer:
    """Mesa's SpaceRenderer with the terrain and sea layers drawn (no grid lines: at
    15,000 cells they would hide the map). The axes styling only depends on the grid
    size, which is the same for every seed, so Mesa can reuse it after a Reset."""
    renderer = SpaceRenderer(model, backend="matplotlib")
    renderer.setup_propertylayer(portrayal.map_layer_style)
    renderer.post_process = portrayal.make_axes_styler(model, VIZ)
    renderer.draw_propertylayer()
    return renderer


MODEL_PARAMS = {
    "seed": {
        "type": "SliderInt",
        "label": "Seed (applies on Reset)",
        "value": CONFIG.run.seed,
        "min": VIZ.seed_min,
        "max": VIZ.seed_max,
        "step": 1,
    },
}

_sea_level_plot, _ = make_plot_component("sea_level")


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
def SeaLevelPlot(model: DelugeModel):
    update_counter.get()
    _sea_level_plot(model)


@solara.component
def Page():
    model = solara.use_reactive(initial_model)
    renderer = solara.use_reactive(initial_renderer)
    model_parameters = solara.use_reactive({})
    play_interval = solara.use_reactive(VIZ.play_interval_ms)
    render_interval = solara.use_reactive(VIZ.render_interval)
    use_threads = solara.use_reactive(False)

    solara.Title("Deluge")
    # Plain columns: with the Vue 3 frontend, solara.Sidebar overlays the main area
    # instead of pushing it aside, which hides the left part of the map.
    with solara.Columns([1, 3]):
        Controls(model, renderer, model_parameters, play_interval, render_interval, use_threads)
        with solara.Column():
            SpaceRendererComponent(model.value, renderer.value)
            SeaLevelPlot(model.value)


@solara.component
def Controls(model, renderer, model_parameters, play_interval, render_interval, use_threads):
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
            ModelCreator(model, MODEL_PARAMS, model_parameters=model_parameters)
        with solara.Card("Year"):
            Status(model.value)


initial_model = DelugeModel(seed=CONFIG.run.seed)
initial_renderer = make_renderer(initial_model)
