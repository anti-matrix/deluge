"""Rounds 2 and 9: the browser UI.

The map is drawn in the browser from what ``viz/portrayal.py`` computes; these tests check
those colours, borders, marks and hover texts against model state, the side panels, and
that stepping through the UI gives the same history as a headless run.
"""

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgb
from mesa.visualization.solara_viz import _build_model_init_kwargs

from conftest import small_config, small_model
from deluge.viz import app, panels, portrayal
from deluge.viz.hexmap import HexMap
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS
from deluge.world.worlddata import Terrain

SEED = 3
VIZ = app.VIZ


@pytest.fixture(scope="module")
def model():
    return small_model()


@pytest.fixture(scope="module")
def grown():
    """A small-world run with several polities of several groups."""
    m = small_model({"polities": {"secession_base": 0.5}}, years=250)
    m.run()
    return m


def rgb255(c):
    return np.rint(np.array(to_rgb(c)) * 255)


def test_terrain_layer_draws_terrain_and_the_years_sea(model):
    colors = portrayal.cell_colors(model, VIZ, "terrain")
    assert colors.shape == (model.geom.n_cells, 3) and colors.dtype == np.uint8
    terrain = model.world.terrain.ravel()
    land = ~model.is_sea
    expected = np.array([rgb255(TERRAIN_COLORS[Terrain(t)]) for t in terrain[land]])
    np.testing.assert_array_equal(colors[land], expected)
    np.testing.assert_array_equal(colors[model.is_sea], np.tile(rgb255(SEA_COLOR), (model.is_sea.sum(), 1)))


@pytest.mark.parametrize("layer", portrayal.LAYERS)
def test_every_layer_draws(grown, layer):
    base = portrayal.cell_colors(grown, VIZ, "terrain")
    colors = portrayal.cell_colors(grown, VIZ, layer, technique="agriculture")
    assert colors.shape == base.shape
    t = grown.settlements
    cells = t.cell[t.alive_ids()]
    if layer not in ("terrain", "resources"):
        assert (colors[cells] != base[cells]).any(), layer  # the layer shows on settled cells
    sea = grown.is_sea
    if layer not in ("polity", "cohesion", "resources"):
        np.testing.assert_array_equal(colors[sea], base[sea])  # the sea stays sea


def test_polity_layer_fills_claimed_cells_by_group_and_shade(grown):
    m = grown
    colors = portrayal.cell_colors(m, VIZ, "polity")
    for shade_by in VIZ.polity_shade_by:
        cells, rgba = portrayal.polity_fill(m, VIZ, shade_by)
        claimed = np.flatnonzero(m.polity_layer.data.ravel() > 0)
        np.testing.assert_array_equal(cells, claimed)
        lineage = m.polities.lineage[m.polity_layer.data.ravel().astype(int)[cells] - 1]
        shades = portrayal.polity_shades(VIZ)
        for g in np.unique(lineage).tolist():  # every colour of a group is one of its own shades
            own = {tuple(np.round(s, 6)) for s in shades[g]}
            assert all(tuple(np.round(c, 6)) in own for c in rgba[lineage == g])
        # Darker with more of the variable.
        share = portrayal.SHADE_BY[shade_by](m, VIZ)[cells]
        bright = rgba[:, :3].sum(axis=1)
        g0 = lineage == lineage[0]
        order = np.argsort(share[g0], kind="stable")
        assert (np.diff(bright[g0][order]) <= 1e-9).all()
    cells, rgba = portrayal.polity_fill(m, VIZ)
    np.testing.assert_array_equal(colors[cells], np.rint(rgba[:, :3] * 255))
    assert list(portrayal.SHADE_BY) == list(VIZ.polity_shade_by)


def test_polity_borders_thick_between_groups_thin_within_and_no_enclave_outlines(grown):
    m = grown
    peoples, polities = portrayal.polity_borders(m)
    assert len(peoples) + len(polities) > 0
    owner = portrayal.claimed(m)
    lone = portrayal.enclaves(m)
    pos = m.geom.positions
    for seg in np.concatenate([peoples, polities])[:200]:
        assert np.linalg.norm(seg[1] - seg[0]) == pytest.approx(1.0)
        near = np.argsort(np.linalg.norm(pos - seg.mean(axis=0), axis=1))[:2]
        assert owner[near[0]] != owner[near[1]] and (owner[near] >= 0).all()
        assert not lone[near].any()
    sets = portrayal.line_sets(m, VIZ, "polity")
    widths = [w for _, w, _ in sets]
    assert VIZ.border_peoples_width in widths or VIZ.border_polities_width in widths
    assert all(c == VIZ.border_outline_color for _, _, c in sets[::2])  # each over a light outline first
    assert portrayal.line_sets(m, VIZ, "terrain") == []
    # An enclave: a polity's single cell touching none of its own, while it claims others.
    nb = m.geom.neighbors
    for c in np.flatnonzero(lone)[:20].tolist():
        p = owner[c]
        assert not any(owner[n] == p for n in nb[c] if n >= 0) and (owner == p).sum() > 1


def test_marks_show_capitals_monuments_wars_and_the_teachers(grown):
    m = grown
    wars = portrayal.WarTracker()
    mk = portrayal.marks(m, VIZ, wars)
    assert len(mk["dots"]) == len(m.polities.alive_ids())
    palette = portrayal.lineage_palette(VIZ)
    assert {d[2] for d in mk["dots"]} <= set(palette)
    for a in mk["arrows"]:
        assert len(a) == 5 and isinstance(a[4], bool)
    assert len(mk["arrows"]) <= len(m.wars)
    # The teachers during the contact phase: a path and their position.
    early = small_model(years=30)
    app.advance(early, 3)
    if early.teachers_state.present:
        mk = portrayal.marks(early, VIZ)
        assert mk["here"] is not None and mk["path"]


def test_war_tracker_reads_direction_and_sea_crossings():
    from deluge.data.events import entity

    m = small_model(years=2)
    w = portrayal.WarTracker()
    i = len(m.events)
    m.events.log(1, "fought", [entity("polity", 2), entity("polity", 1), entity("settlement", 0)], 0, 5000.0)
    m.events.log(1, "crossed", [entity("polity", 2), entity("settlement", 0), entity("settlement", 1),
                                entity("polity", 1)], 0, 2.0, link=i)
    w.update(m)
    assert w.last[(1, 2)] == (1, 2, 1, True)
    m.events.log(2, "fought", [entity("polity", 1), entity("polity", 2), entity("settlement", 0)], 0, 5000.0)
    w.update(m)
    assert w.last[(1, 2)] == (2, 1, 2, False)


def test_tooltip_names_terrain_elevation_settlement_polity_lineage_population(grown):
    m = grown
    snap = portrayal.cell_snapshot(m)
    t = m.settlements
    sid = int(t.alive_ids()[0])
    cell = int(t.cell[sid])
    text = portrayal.tooltip(m, snap, cell)
    x, y = m.geom.xy(cell)
    from deluge.data.events import entity
    from deluge.naming import Speaker

    say = Speaker(m, tags=True)
    assert f"{say.place(cell)}</b> (cell {x}, {y})" in text and "elevation" in text
    assert f"{say.entity(entity('settlement', sid))}, of {say.entity(entity('polity', int(t.polity[sid])))}" in text
    assert say.people(int(t.lineage[sid])) in text and "population" in text
    sea = int(np.flatnonzero(m.is_sea)[0])
    assert "sea" in portrayal.tooltip(m, snap, sea)


def test_side_panels_render(grown):
    m = grown
    charts = app.charts(m)
    assert len(charts) == 4 and all(c.startswith("<svg") for c in charts)
    for name, color in app.GROUP_PLOT_COLORS.items():
        assert color in charts[0]
    tree = panels.lineage_tree(m, VIZ)
    assert tree.startswith("<svg") and f"{len(m.polities.alive_ids())} living" in tree
    pid = int(m.polities.alive_ids()[0])
    html = panels.inspector(m, pid, VIZ)
    for part in ("Their ways", "What they lean toward this year", "What they remember", "<details"):
        assert part in html
    for trait in m.config.founders.culture_traits:
        assert trait.replace("_", " ") in html
    assert "<title>" in tree  # polity names on hover
    assert "Click a claimed cell" in panels.inspector(m, -1, VIZ)


def test_frames_carry_everything_the_page_draws(grown):
    session = app.Session(grown)
    for layer in portrayal.LAYERS:
        f = session.frame(app.View(layer=layer), None)
        assert f.colors and f.status and len(f.charts) == 4 and f.tree
        assert (len(f.lines) > 0) == (layer == "polity")
    widget = HexMap(grid_width=grown.geom.width, grid_height=grown.geom.height, colors=f.colors, lines=f.lines,
                    marks=f.marks)
    assert widget.template_file[1] == "hexmap.vue"


def test_ui_and_headless_runs_agree():
    """A model made the way the UI's Reset makes it (seeds as keyword arguments), stepped
    through the UI's play loop and step buttons (frames built between steps), matches a
    headless run with the same seeds."""
    template = small_model(history_seed=0, years=2)
    params = {"world_seed": SEED, "history_seed": SEED}
    kwargs = _build_model_init_kwargs(template, params, add_scenario_when_empty=True,
                                      require_model_accepts_scenario=True)
    assert kwargs == params
    ui = type(template)(config=small_config(), **kwargs)
    session = app.Session(ui)
    years = 300
    frame = None
    for n in (1, 10, 100, 25, 25, 25, 25, 25, 25, 39):
        frame = session.step(n, app.View(), frame, panels_too=True)
    headless = small_model(world_seed=SEED, history_seed=SEED, years=years)
    headless.run()
    assert ui.year == headless.year == years
    pd.testing.assert_frame_equal(ui.datacollector.get_model_vars_dataframe(),
                                  headless.datacollector.get_model_vars_dataframe())
    assert ui.events.to_frame().equals(headless.events.to_frame())
    n = headless.settlements.n
    assert ui.settlements.n == n
    for col in ("population", "cell", "lineage", "parent", "ended", "polity"):
        np.testing.assert_array_equal(getattr(ui.settlements, col)[:n], getattr(headless.settlements, col)[:n])


def test_advance_stops_at_end_of_run():
    m = small_model(years=5)
    app.advance(m, 100)
    assert m.year == 5 and not m.running


@pytest.mark.slow
def test_sea_layer_follows_sea_level():
    m = small_model(years=3100)
    app.advance(m, 3001)
    np.testing.assert_array_equal(m.is_sea, m.elevation_flat < m.sea_level_series[3000])
    colors = portrayal.cell_colors(m, VIZ, "terrain")
    np.testing.assert_array_equal(colors[m.is_sea], np.tile(rgb255(SEA_COLOR), (m.is_sea.sum(), 1)))
    assert portrayal.shown_year(m) == 3000
    assert "Year 3,000" in portrayal.status_text(m)


def test_app_page_builds():
    assert app.Page is not None
    assert app.SEEDS == ("world_seed", "history_seed")


def test_legends_and_plots_use_one_lineage_palette():
    palette = portrayal.lineage_palette(VIZ)
    assert palette == ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")  # A red, B blue, C green, D purple, E orange
    names = app.CONFIG.founders.names
    assert app.GROUP_PLOT_COLORS == {f"pop_{n}": c for n, c in zip(names, palette)}
    for layer, label in (("lineage", "the {}"), ("polity", "polities of the {}")):
        html = portrayal.legend_html(VIZ, layer, [f"the {n}" for n in names])
        rows = html.split('<div style="display:flex')[1:]
        assert len(rows) == len(names)
        for row, n, c in zip(rows, names, palette):
            assert f"background:{c}" in row and row.index(f"background:{c}") < row.index(label.format(n))
    assert "iron ore" in portrayal.legend_html(VIZ, "resources", names)
    for layer in ("subsistence mode", "technique", "cohesion"):
        assert portrayal.legend_html(VIZ, layer, names, "agriculture")


def test_chronicle_reads_one_snapshot_while_events_are_appended():
    """The model can append events while the browser chronicle reads (Round 6 bug:
    "operands could not be broadcast together"): a sync reads every column up to the length
    all of them have; the rest waits for the next sync."""
    from deluge.data import chronicle

    m = small_model(years=20)
    for _ in range(5):
        m.step()
    view = chronicle.Chronicle(m)
    n = len(m.events)
    m.events.link.append(-1)  # an event half appended: its first column only
    view.sync()
    assert view.n == n
    m.events.link.pop()
    m.log_event("moved", [("settlement", 0)], 0, 1.0)
    view.sync()
    assert view.n == n + 1
    assert len(chronicle.log_csv(m).splitlines()) == n + 2  # header and every event
