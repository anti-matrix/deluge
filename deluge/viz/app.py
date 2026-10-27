"""Browser UI entry point (spec section 12).

    solara run deluge/viz/app.py
    DELUGE_CHECKPOINT=runs/<run>/checkpoints/year_04000.pkl solara run deluge/viz/app.py   # open a checkpoint

The map is a canvas drawn in the browser (``viz/hexmap.py``): zoom with the wheel or the
buttons, pan by dragging, hover a cell for its terrain, elevation, settlement, polity,
lineage and population, click a claimed cell to inspect its polity. Layers: polity (shaded
by population density, technology level or subsistence output), terrain, population
density, subsistence mode, cohesion, a chosen technique's spread, founding group, mineral
deposits, techniques known; overlays: capitals, monuments, the teachers' path during the
contact phase, the war display (campaign arrows, supply lines, war banners; each switched on
its own, hover to light up one war). Side panels:
population per founding group, polity count, sea level and subsistence modes over time, a
lineage tree snapshot (polity names on hover), the polity inspector and the chronicle, every
entity named as in the reports (``deluge/naming.py``).

Speed: while playing, a worker thread steps the model ``years per frame`` years at a time
and, between steps, builds the next frame (``Frame``: the map's colours, borders and marks,
the hover data, the status text, and every ``viz.panel_interval`` frames the charts and the
tree). The browser draws whatever frame is newest, so stepping never waits for drawing. The
model is only touched under one lock, so a frame always shows one consistent year.

The UI only reads model state and steps the model through ``DelugeModel.step``, the same
call a headless run makes, so a UI run and a headless run with the same seeds agree.
"""

from __future__ import annotations

import html
import itertools
import os
import pickle
import threading
import time
from dataclasses import dataclass, field

import solara
import solara.lab

from deluge.agents.settlement import MODES
from deluge.config import load_config
from deluge.data import chronicle
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
    war: tuple = portrayal.WAR_OVERLAYS  # the parts of the war display switched on


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


def build_frame(model: DelugeModel, view: View, previous: Frame | None, panels_too: bool) -> Frame:
    """Everything the page draws for the model's current year (call between steps)."""
    t0 = time.perf_counter()
    f = Frame(year=portrayal.shown_year(model))
    f.colors = b64(portrayal.cell_colors(model, VIZ, view.layer, view.shade_by, view.technique), "uint8")
    f.lines = [{"segments": b64(seg.reshape(-1), "float32"), "width": w, "color": c}
               for seg, w, c in portrayal.line_sets(model, VIZ, view.layer)]
    f.marks = portrayal.marks(model, VIZ, view.war)
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


_SESSIONS = itertools.count()


class Session:
    """One run shown in the browser: its model, the lock around it, and the play loop."""

    def __init__(self, model: DelugeModel) -> None:
        self.model = model
        self.lock = threading.Lock()
        self.frames = 0
        # The components that keep choices of their own (the chronicle's filter, the myths'
        # selection, the inspector's tellings) are keyed by this, so a new run rebuilds them
        # from its own peoples, polities and names.
        self.key = next(_SESSIONS)

    def frame(self, view: View, previous: Frame | None, panels_too: bool = True) -> Frame:
        with self.lock:
            return build_frame(self.model, view, previous, panels_too)

    def step(self, years: int, view: View, previous: Frame | None, panels_too: bool) -> Frame:
        with self.lock:
            advance(self.model, years)
            self.frames += 1
            return build_frame(self.model, view, previous, panels_too)


_initial: list = []


def initial_model() -> DelugeModel:
    """The model the page opens with, built on first use (not at import, so importing this
    module stays cheap): a new one, or the model of the checkpoint (``scripts/run.py``) named by
    the environment variable ``DELUGE_CHECKPOINT``."""
    if not _initial:
        path = os.environ.get("DELUGE_CHECKPOINT")
        if path:
            with open(path, "rb") as f:
                state = pickle.load(f)
            _initial.append(state["model"] if isinstance(state, dict) else state)
        else:
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
def LayerSelect(view: solara.Reactive[View], refresh, model: DelugeModel):
    """The map's layer, shade and overlay choices, and the legend: of the active run's model."""
    v = view.value
    techniques = list(model.knowledge.names)
    peoples = solara.use_memo(lambda: portrayal.people_names(model), dependencies=[model])

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
    legend = portrayal.legend_html(VIZ, v.layer, peoples, v.technique)
    if legend:
        solara.HTML(tag="div", unsafe_innerHTML=legend)
    solara.SelectMultiple("War overlays", values=list(v.war), all_values=list(portrayal.WAR_OVERLAYS),
                          on_value=lambda value: setter("war")(tuple(o for o in portrayal.WAR_OVERLAYS if o in value)))
    solara.HTML(tag="div", unsafe_innerHTML=(
        '<div style="font-size:12px;margin-top:4px">Overlays: <b>dots</b> capitals, <b>squares</b> monuments, '
        f'<span style="color:{VIZ.teachers_color}">dotted line and triangle</span>: the teachers during the contact '
        f'phase. War: <span style="color:{VIZ.war_color}">arrows</span> from the settlement an attack set out from '
        f'to the one attacked (dashed: across the sea; fading over {VIZ.war_fade_years} years), '
        f'<span style="color:{VIZ.supply_color}">thin lines</span>: supply from the attacker\'s capital, as faint as '
        'its reach is stretched; <b>flags</b> on capitals at war: a sword for attacks made, a shield for attacks '
        'suffered, with their counts. Hover an arrow or a flag to light up its war. Wheel or +/- to zoom, drag to '
        'pan, click a cell to inspect its polity.</div>'))


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


def chronicle_panel(session: Session, view: chronicle.Chronicle, peoples: list[str], chosen_peoples,
                    text: str, limit: int) -> tuple[str, int, int]:
    """What the chronicle panel shows: (HTML of the newest ``limit`` matching entries, entries
    shown, entries matching). Reads the events logged since the last call first."""
    with session.lock:
        view.sync()
    groups = {peoples.index(g) for g in chosen_peoples if g in peoples}
    chosen = view.select(groups, text or "")
    lines = view.latest(chosen, limit)
    body = "".join(f'<div style="margin:0 0 3px 0">{html.escape(s)}</div>' for s in lines) or "<i>Nothing yet.</i>"
    return body, len(lines), len(chosen)


@solara.component
def Chronicle(session: Session, version: int):
    """The chronicle (``deluge/data/chronicle.py``): important events in the narrator's
    language, routine ones summed per generation, newest first: the latest
    ``chronicle.page_entries``, more on demand. Filtered by people and text; downloads the
    full raw log (CSV) and the whole chronicle (markdown). Refreshed with the side panels."""
    model = session.model
    view = solara.use_memo(lambda: chronicle.Chronicle(model), dependencies=[model])
    peoples = solara.use_memo(lambda: portrayal.people_names(model), dependencies=[model])
    chosen_peoples, set_peoples = solara.use_state(peoples)
    text, set_text = solara.use_state("")
    limit, set_limit = solara.use_state(CONFIG.chronicle.page_entries)

    def reset(setter):
        def on_value(value):
            setter(value)
            set_limit(CONFIG.chronicle.page_entries)
        return on_value

    def locked(make):
        """A download: read the new events under the lock, write the file outside it (the
        chronicle and the log are only appended to, so playing goes on meanwhile)."""
        def data():
            with session.lock:
                view.sync()
            return make()
        return data

    body, shown, total = solara.use_memo(
        lambda: chronicle_panel(session, view, peoples, chosen_peoples, text, limit),
        dependencies=[version, tuple(chosen_peoples), text, limit])
    with solara.Row():
        solara.SelectMultiple("Peoples", values=chosen_peoples, all_values=peoples, on_value=reset(set_peoples))
        solara.InputText("Search", value=text, on_value=reset(set_text), clearable=True)
    with solara.Row():
        solara.Text(f"Latest {shown:,} of {total:,} entries")
        solara.Button("Show older", on_click=lambda: set_limit(limit + CONFIG.chronicle.page_entries),
                      disabled=shown >= total)
        solara.FileDownload(locked(lambda: chronicle.log_csv(model)), filename="events.csv",
                            label="Download full log", mime_type="text/csv")
        solara.FileDownload(locked(view.markdown), filename="chronicle.md", label="Download chronicle",
                            mime_type="text/markdown")
    with solara.Div(style={"max-height": "420px", "overflow-y": "auto", "font-size": "13px"}):
        solara.HTML(tag="div", unsafe_innerHTML=body)


@solara.component
def Inspector(session: Session, pid: int, version: int):
    def compute():
        with session.lock:
            return panels.inspector(session.model, pid, VIZ)

    content = solara.use_memo(compute, dependencies=[session, pid, version])
    solara.HTML(tag="div", unsafe_innerHTML=content)
    if 0 <= pid < session.model.polities.n:
        Bard(session, "polity", pid, version)


# --- the bard (analysis/bard.py): on demand only ----------------------------------------------------------


def bard_input(session: Session, kind: str, i: int) -> dict:
    """The bard's input for a polity or a founding lineage (read under the lock)."""
    from analysis import bard

    with session.lock:
        model = session.model
        return bard.polity_input(model, i) if kind == "polity" else bard.lineage_input(model, i)


def bard_tell(inp: dict) -> dict:
    """The bard's telling with the configured backend (outside the lock: an llm call may take
    time; it falls back to formulaic with a notice when there is no endpoint)."""
    from analysis import bard

    return bard.tell(inp)


@solara.component
def Bard(session: Session, kind: str, i: int, version: int):
    """"Tell it as their myth": the bard's telling beside the narrator's plain text, on demand;
    the bard's input to copy or download."""
    import json

    from analysis import bard
    from deluge.viz.copybutton import CopyButton

    inp = solara.use_memo(lambda: bard_input(session, kind, i), dependencies=[session, kind, i, version])
    told, set_told = solara.use_state({})
    exported = json.dumps(bard.export_input(inp), ensure_ascii=False, indent=1)
    mine = told.get((kind, i, tuple(r["version"] for r in inp["records"])))

    def tell():
        out = bard_tell(inp)
        set_told({**told, (kind, i, tuple(r["version"] for r in inp["records"])): out})

    with solara.Row(style={"align-items": "center", "flex-wrap": "wrap"}):
        solara.Button("Tell it as their myth", on_click=tell, disabled=not inp["records"])
        CopyButton.element(text=exported, label="Copy bard input", title="what an LLM would receive, with the instructions")
        solara.FileDownload(exported, filename=f"bard_input_{kind}_{i}.json", label="bard_input.json",
                            mime_type="application/json")
    if not inp["records"]:
        solara.Text("Nothing to tell: no record.")
    if mine:
        with solara.Columns([1, 1]):
            with solara.Column():
                solara.HTML(tag="div", unsafe_innerHTML="<b>What they remember (the narrator)</b>" + "".join(
                    f'<div style="margin:2px 0">{html.escape(r["text"])}</div>' for r in inp["records"]))
            with solara.Column():
                note = f'<div style="color:#a33;font-size:12px">{html.escape(mine["notice"])}</div>' if mine["notice"] else ""
                solara.HTML(tag="div", unsafe_innerHTML=(
                    f"<b>{html.escape(mine['label'][:1].upper() + mine['label'][1:])}</b>{note}"
                    f'<div style="white-space:pre-wrap">{html.escape(mine["text"])}</div>'))


def myths_export(session: Session) -> str:
    """"Myths of the peoples" (markdown): every founding people's origins and the largest living
    polities, with the bard's tellings made so far (no new call)."""
    from analysis import bard

    with session.lock:
        model = session.model
        P = model.polities
        pids = sorted(P.alive_ids().tolist(), key=lambda q: (-float(P.population[q]), q))[: VIZ.myths_polities]
        inputs = [bard.lineage_input(model, g) for g in range(len(model.lineages.founders))]
        inputs += [bard.polity_input(model, q) for q in pids]
        return bard.myths_markdown(model, inputs)


def myth_choices(session: Session) -> list[tuple[str, str, int]]:
    """Who can be told (label, kind, id): every founding lineage and every living polity that
    holds any memory record; the lineages first, then the polities by people."""
    from analysis import bard
    from deluge.data.events import entity
    from deluge.naming import Speaker

    with session.lock:
        model = session.model
        P = model.polities
        say = Speaker(model, tags=True)
        names = portrayal.people_names(model)
        out = [(names[g], "lineage", g) for g in range(len(names))
               if bard.has_records(model, "lineage", g)]
        pids = sorted((p for p in P.alive_ids().tolist() if bard.has_records(model, "polity", p)),
                      key=lambda q: (-float(P.population[q]), q))
        out += [(f"{say.entity(entity('polity', p))}, polity {p}", "polity", p) for p in pids]
        return out


@solara.component
def Myths(session: Session, version: int):
    """The myth of any people or living polity with a memory, on demand (``Bard``), and the export."""
    choices = solara.use_memo(lambda: myth_choices(session), dependencies=[session, version])
    labels = [c[0] for c in choices]
    picked, set_chosen = solara.use_state(None)
    # The first choice until one is picked (the list grows as the run makes its first records).
    chosen = picked if picked in labels else (labels[0] if labels else None)
    with solara.Row():
        solara.Select("People or polity", values=labels, value=chosen, on_value=set_chosen)
        solara.FileDownload(lambda: myths_export(session), filename="myths_of_the_peoples.md",
                            label="Myths of the peoples", mime_type="text/markdown")
    if chosen in labels:
        _, kind, i = choices[labels.index(chosen)]
        Bard(session, kind, i, version)
    elif not labels:
        solara.Text("Nothing to tell: no people holds a record yet.")


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
                LayerSelect(view, refresh, model)
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
                    Inspector(s, selected_pid.value, f.panels_version if not playing.value else -1).key(
                        f"inspector-{s.key}")
            with solara.Card("Chronicle"):
                Chronicle(s, f.panels_version).key(f"chronicle-{s.key}")
            with solara.Card("Myths of the peoples"):
                Myths(s, f.panels_version if not playing.value else -1).key(f"myths-{s.key}")
