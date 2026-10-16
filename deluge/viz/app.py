"""Browser UI entry point (spec section 12).

    solara run deluge/viz/app.py

The map is a canvas drawn in the browser (``viz/hexmap.py``): zoom with the wheel or the
buttons, pan by dragging, hover a cell for its terrain, elevation, settlement, polity,
lineage and population, click a claimed cell to inspect its polity. Layers: polity (shaded
by population density, technology level or subsistence output), terrain, population
density, subsistence mode, cohesion, a chosen technique's spread, founding group, mineral
deposits, techniques known; overlays: capitals, monuments, the teachers' path during the
contact phase, active wars as arrows between capitals (dashed across the sea). Side panels:
population per founding group, polity count, sea level and subsistence modes over time, a
lineage tree snapshot, the polity inspector and the chronicle of the whole event log.

Speed: while playing, a worker thread steps the model ``years per frame`` years at a time
and, between steps, builds the next frame (``Frame``: the map's colours, borders and marks,
the hover data, the status text, and every ``viz.panel_interval`` frames the charts and the
tree). The browser draws whatever frame is newest, so stepping never waits for drawing. The
model is only touched under one lock, so a frame always shows one consistent year.

The UI only reads model state and steps the model through ``DelugeModel.step``, the same
call a headless run makes, so a UI run and a headless run with the same seeds agree.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import solara
import solara.lab

from deluge.agents.settlement import MODES
from deluge.config import load_config
from deluge.data import chronicle
from deluge.data.events import ACTIONS
from deluge.model import DelugeModel
from deluge.viz import panels, portrayal
from deluge.viz.hexmap import HexMap, b64

CONFIG = load_config()
VIZ = CONFIG.viz
SEEDS = ("world_seed", "history_seed")
if set(VIZ.polity_shade_by) - set(portrayal.SHADE_BY):
    raise KeyError(f"viz.polity_shade_by: unknown {sorted(set(VIZ.polity_shade_by) - set(portrayal.SHADE_BY))}")
# Population per founding group, in the lineage palette (the map, the tree and legends use the same).
GROUP_PLOT_COLORS = {f"pop_{name}": color for name, color in zip(CONFIG.founders.names, portrayal.lineage_palette(VIZ))}


def advance(model: DelugeModel, years: int) -> None:
    """Step the model up to ``years`` times, stopping at the end of the run."""
    for _ in range(years):
        if not model.running:
            break
        model.step()


# --- frames -------------------------------------------------------------------------------------------


@dataclass
class View:
    layer: str = "polity"
    shade_by: str = VIZ.polity_shade_by[0]
    technique: str = "agriculture"


@dataclass
class Frame:
    year: int = 0
    colors: str = ""
    lines: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)
    status: str = ""
    charts: list = field(default_factory=list)
    tree: str = ""
    panels_version: int = 0
    seconds: float = 0.0


def build_frame(model: DelugeModel, view: View, wars: portrayal.WarTracker, previous: Frame | None,
                panels_too: bool) -> Frame:
    """Everything the page draws for the model's current year (call between steps)."""
    t0 = time.perf_counter()
    f = Frame(year=portrayal.shown_year(model))
    f.colors = b64(portrayal.cell_colors(model, VIZ, view.layer, view.shade_by, view.technique), "uint8")
    f.lines = [{"segments": b64(seg.reshape(-1), "float32"), "width": w, "color": c}
               for seg, w, c in portrayal.line_sets(model, VIZ, view.layer)]
    f.marks = portrayal.marks(model, VIZ, wars)
    f.cells = portrayal.cell_snapshot(model)
    f.status = portrayal.status_text(model)
    if panels_too or previous is None:
        f.charts = charts(model)
        f.tree = panels.lineage_tree(model, VIZ)
        f.panels_version = (previous.panels_version + 1) if previous else 0
    else:
        f.charts, f.tree, f.panels_version = previous.charts, previous.tree, previous.panels_version
    f.seconds = time.perf_counter() - t0
    return f


def charts(model: DelugeModel) -> list[str]:
    names = CONFIG.founders.names
    s = panels.series(model, [f"pop_{n}" for n in names] + ["n_polities", "sea_level"] + [f"pop_{m}" for m in MODES])
    xs = s["year"]
    return [
        panels.line_chart(VIZ, "Population per founding group", xs,
                          [(n, s[f"pop_{n}"], GROUP_PLOT_COLORS[f"pop_{n}"]) for n in names]),
        panels.line_chart(VIZ, "Polities", xs, [("polities", s["n_polities"], "#333333")]),
        panels.line_chart(VIZ, "Population per subsistence mode", xs,
                          [(m, s[f"pop_{m}"], VIZ.mode_colors[m]) for m in MODES]),
        panels.line_chart(VIZ, "Sea level (m, + 100)", xs, [("sea level", [v + 100.0 for v in s["sea_level"]],
                                                              "#2166ac")]),
    ]


class Session:
    """One browser session's model, the lock around it, and the play loop."""

    def __init__(self, model: DelugeModel) -> None:
        self.model = model
        self.lock = threading.Lock()
        self.wars = portrayal.WarTracker()
        self.frames = 0

    def frame(self, view: View, previous: Frame | None, panels_too: bool = True) -> Frame:
        with self.lock:
            return build_frame(self.model, view, self.wars, previous, panels_too)

    def step(self, years: int, view: View, previous: Frame | None, panels_too: bool) -> Frame:
        with self.lock:
            advance(self.model, years)
            self.frames += 1
            return build_frame(self.model, view, self.wars, previous, panels_too)


_initial: list = []


def initial_model() -> DelugeModel:
    """The model the page opens with, built on first use (not at import, so importing this
    module stays cheap)."""
    if not _initial:
        _initial.append(DelugeModel(world_seed=CONFIG.run.world_seed, history_seed=CONFIG.run.history_seed))
    return _initial[0]


# --- components -----------------------------------------------------------------------------------------


@solara.component
def MapView(session: Session, frame: Frame, selected_cell: solara.Reactive[int], on_pick):
    tooltip, set_tooltip = solara.use_state("")
    model = session.model

    def on_hover(cell: int):
        try:
            set_tooltip(portrayal.tooltip(model, frame.cells, cell))
        except Exception:  # a frame from a model being replaced: no tooltip
            set_tooltip("")

    HexMap.element(grid_width=model.geom.width, grid_height=model.geom.height, height_px=VIZ.map_height_px,
                   max_zoom=VIZ.max_zoom, colors=frame.colors, lines=frame.lines, marks=frame.marks,
                   tooltip=tooltip, selected=selected_cell.value, on_hover=on_hover, on_click=on_pick)


@solara.component
def LayerSelect(view: solara.Reactive[View], refresh):
    v = view.value
    techniques = list(initial_model().knowledge.names)

    def setter(name):
        def on_value(value):
            view.set(View(**{**v.__dict__, name: value}))
            refresh()
        return on_value

    solara.Select("Map layer", values=list(portrayal.LAYERS), value=v.layer, on_value=setter("layer"))
    if v.layer == "polity":
        solara.Select("Shade by", values=list(VIZ.polity_shade_by), value=v.shade_by, on_value=setter("shade_by"))
    if v.layer == "technique":
        solara.Select("Technique", values=techniques, value=v.technique, on_value=setter("technique"))
    legend = portrayal.legend_html(VIZ, v.layer, CONFIG.founders.names, v.technique)
    if legend:
        solara.HTML(tag="div", unsafe_innerHTML=legend)
    solara.HTML(tag="div", unsafe_innerHTML=(
        '<div style="font-size:12px;margin-top:4px">Overlays: <b>dots</b> capitals, <b>squares</b> monuments, '
        f'<span style="color:{VIZ.teachers_color}">dotted line and triangle</span>: the teachers during the contact '
        f'phase, <span style="color:{VIZ.war_color}">arrows</span>: active wars, attacker to defender (dashed: '
        'across the sea). Wheel or +/- to zoom, drag to pan, click a cell to inspect its polity.</div>'))


@solara.component
def SeedInputs(seeds: solara.Reactive[dict]):
    def setter(name: str):
        def on_value(value):
            if value is not None:
                seeds.set({**seeds.value, name: int(value)})
        return on_value

    solara.InputInt("World seed: the map (applies on Reset)", value=seeds.value["world_seed"],
                    on_value=setter("world_seed"))
    solara.InputInt("History seed (applies on Reset)", value=seeds.value["history_seed"],
                    on_value=setter("history_seed"))


@solara.component
def Chronicle(model: DelugeModel, version: int):
    """The whole event log, a page at a time (newest first), filtered by group, event
    type, year range and text; the full chronicle downloads as a text file. Refreshed with
    the side panels, not every frame."""
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
        solara.InputInt("From year", value=year_min, optional=True, clearable=True, on_value=reset_page(set_year_min))
        solara.InputInt("To year", value=year_max, optional=True, clearable=True, on_value=reset_page(set_year_max))
        solara.InputText("Search", value=text, on_value=reset_page(set_text), clearable=True)
    chosen = solara.use_memo(lambda: view.select(groups={names.index(g) for g in groups}, actions=set(actions),
                                                 year_min=year_min, year_max=year_max, text=text or ""),
                             dependencies=[version, tuple(groups), tuple(actions), year_min, year_max, text])
    pages = view.pages(chosen, VIZ.chronicle_lines)
    shown_page = min(page, pages - 1)
    with solara.Row():
        solara.Button("Newer", on_click=lambda: set_page(max(shown_page - 1, 0)), disabled=shown_page == 0)
        solara.Button("Older", on_click=lambda: set_page(shown_page + 1), disabled=shown_page >= pages - 1)
        solara.Text(f"Page {shown_page + 1:,} of {pages:,}: {len(chosen):,} events")
        solara.FileDownload(lambda: view.text(), filename="chronicle.txt", label="Download whole chronicle",
                            mime_type="text/plain")
    shown = view.page(chosen, shown_page, VIZ.chronicle_lines)
    with solara.Div(style={"max-height": "420px", "overflow-y": "auto"}):
        solara.Markdown("  \n".join(shown) if shown else "No events.")


@solara.component
def Inspector(session: Session, pid: int, version: int):
    def compute():
        with session.lock:
            return panels.inspector(session.model, pid, VIZ)

    content = solara.use_memo(compute, dependencies=[session, pid, version])
    solara.HTML(tag="div", unsafe_innerHTML=content)


@solara.component
def Page():
    session = solara.use_memo(lambda: Session(initial_model()), dependencies=[])
    session_r = solara.use_reactive(session)
    s: Session = session_r.value
    view = solara.use_reactive(View())
    frame = solara.use_reactive(None)
    playing = solara.use_reactive(False)
    years_per_frame = solara.use_reactive(VIZ.render_interval)
    pause_ms = solara.use_reactive(VIZ.play_interval_ms)
    seeds = solara.use_reactive({name: getattr(CONFIG.run, name) for name in SEEDS})
    selected_cell = solara.use_reactive(-1)
    selected_pid = solara.use_reactive(-1)
    timing = solara.use_reactive("")

    if frame.value is None:
        frame.value = s.frame(view.value, None)

    def refresh():
        frame.set(s.frame(view.value, frame.value, panels_too=True))

    def step_years(n: int):
        frame.set(s.step(n, view.value, frame.value, panels_too=True))

    def play_loop():
        start, y0 = time.perf_counter(), s.model.year
        k = 0
        while playing.value and s.model.running:
            k += 1
            f = s.step(years_per_frame.value, view.value, frame.value, panels_too=k % VIZ.panel_interval == 0)
            frame.set(f)
            dt = time.perf_counter() - start
            timing.set(f"{s.model.year - y0:,} years in {dt:,.1f} s ({(s.model.year - y0) / max(dt, 1e-9):,.1f} "
                       f"years/s); frame build {1000 * f.seconds:.0f} ms")
            if pause_ms.value:
                time.sleep(pause_ms.value / 1000)
        if playing.value:
            playing.set(False)
        frame.set(s.frame(view.value, frame.value, panels_too=True))

    solara.lab.use_task(play_loop, dependencies=[playing.value], prefer_threaded=True)

    def reset():
        playing.set(False)
        with s.lock:
            model = DelugeModel(**seeds.value)
        new = Session(model)
        session_r.set(new)
        selected_cell.set(-1)
        selected_pid.set(-1)
        frame.set(new.frame(view.value, None))

    def pick(cell: int):
        snap = frame.value.cells
        p = int(snap["polity"][cell])
        if p < 0:
            p = int(snap["settlement_polity"][cell])
        selected_cell.set(cell)
        selected_pid.set(p)

    f: Frame = frame.value
    model = s.model
    solara.Title("Deluge")
    with solara.Columns([1, 3]):
        with solara.Column():
            with solara.Card("Controls"):
                with solara.Row():
                    solara.Button("Reset", color="primary", on_click=reset)
                    solara.Button("❚❚ Pause" if playing.value else "▶ Play", color="primary",
                                  on_click=lambda: playing.set(not playing.value), disabled=not model.running)
                    solara.Button(f"Step {years_per_frame.value}", on_click=lambda: step_years(years_per_frame.value),
                                  disabled=playing.value or not model.running)
                with solara.Row():
                    for years in VIZ.step_buttons:
                        solara.Button(f"+{years} yr", on_click=lambda years=years: step_years(years),
                                      disabled=playing.value or not model.running)
                solara.SliderInt("Years per frame", value=years_per_frame, min=1, max=VIZ.render_interval_max)
                solara.SliderInt("Pause between frames (ms)", value=pause_ms, min=0, max=VIZ.play_interval_max_ms)
                if timing.value:
                    solara.Text(timing.value)
            with solara.Card("Model parameters"):
                SeedInputs(seeds)
            with solara.Card("Map"):
                LayerSelect(view, refresh)
            with solara.Card("Year"):
                solara.Markdown(f.status)
        with solara.Column():
            MapView(s, f, selected_cell, pick)
            with solara.Row(style={"flex-wrap": "wrap"}):
                for svg in f.charts:
                    solara.HTML(tag="div", unsafe_innerHTML=svg)
            with solara.Columns([1, 1]):
                with solara.Card("Lineage tree"):
                    solara.HTML(tag="div", unsafe_innerHTML=f.tree)
                with solara.Card("Polity inspector"):
                    Inspector(s, selected_pid.value, f.panels_version if not playing.value else -1)
            with solara.Card("Chronicle"):
                Chronicle(model, f.panels_version)
