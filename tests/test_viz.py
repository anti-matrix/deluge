"""Round 2: the browser UI skeleton.

Mesa's SpaceRenderer draws the map; these tests check that its hex mesh lines up with
Deluge's geometry (and so with Mesa's connectivity, not ``cell.position``), that the
drawn colours are the model's terrain and sea, and that stepping through the UI gives the
same history as a headless run.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from mesa.visualization.solara_viz import _build_model_init_kwargs  # noqa: E402

from conftest import small_config, small_model  # noqa: E402
from deluge.model import DelugeModel  # noqa: E402
from deluge.viz import app, portrayal  # noqa: E402
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS  # noqa: E402
from deluge.world.worlddata import Terrain  # noqa: E402

SEED = 3


@pytest.fixture(scope="module")
def model():
    return small_model()


def mesh_order_to_cid(geom):
    """Mesa's hex mesh and its layer colours run row by row (``data.T.ravel()``)."""
    rows, cols = np.divmod(np.arange(geom.n_cells), geom.width)
    return cols * geom.height + rows


def test_mesa_hex_mesh_matches_geometry(model):
    renderer = app.make_renderer(model)
    hexes = np.asarray(renderer.space_drawer.hexagons)  # (n, 6, 2)
    centers = hexes.mean(axis=1)
    geom = model.geom
    np.testing.assert_allclose(centers, geom.positions[mesh_order_to_cid(geom)], atol=1e-9)
    # So every neighbour pair of the simulation is drawn as touching hexes.
    order = np.empty(geom.n_cells, dtype=int)
    order[mesh_order_to_cid(geom)] = np.arange(geom.n_cells)
    nb = geom.neighbors
    c, k = np.nonzero(nb >= 0)
    d = np.linalg.norm(centers[order[c]] - centers[order[nb[c, k]]], axis=1)
    np.testing.assert_allclose(d, np.sqrt(3.0), atol=1e-9)


def test_map_draws_terrain_sea_and_population(model):
    renderer = app.make_renderer(model)
    terrain_coll, sea_coll, pop_coll = renderer.canvas.collections[-3:]
    cid = mesh_order_to_cid(model.geom)
    terrain = model.world.terrain.ravel()[cid]
    expected = np.array([to_rgba(TERRAIN_COLORS[Terrain(t)]) for t in terrain])
    np.testing.assert_allclose(terrain_coll.get_facecolor(), expected, atol=1e-6)
    sea = sea_coll.get_facecolor()
    is_sea = model.is_sea[cid]
    np.testing.assert_allclose(sea[is_sea], np.tile(to_rgba(SEA_COLOR), (is_sea.sum(), 1)), atol=1e-6)
    assert (sea[~is_sea, 3] == 0).all()  # land: sea layer fully transparent
    # People: the five founding bands are visible, empty cells are clear.
    people = model.cell_population()[cid]
    alpha = pop_coll.get_facecolor()[:, 3]
    assert (people > 0).sum() == 5
    assert (alpha[people > 0] > 0.2).all()
    assert (alpha[people == 0] == 0).all()


def test_only_map_layers_and_the_chosen_overlay_are_drawn(model):
    for overlay in portrayal.OVERLAYS:
        style = portrayal.make_layer_style(app.VIZ, overlay)
        drawn = {name for name, layer in model.grid._mesa_property_layers.items()
                 if name != "empty" and style(layer) is not None}
        # The polity overlay is drawn by portrayal.draw_polities, not as a Mesa layer.
        assert drawn == set(portrayal.MAP_LAYERS) | ({overlay} if overlay != "polity" else set())


@pytest.mark.slow
def test_sea_layer_follows_sea_level():
    m = small_model(years=3100)
    layer = m.grid._mesa_property_layers["sea"]
    data = layer.data
    sea0 = int(layer.data.sum())
    app.advance(m, 3001)
    assert layer.data is data  # updated in place: Mesa's per-cell accessors stay bound
    np.testing.assert_array_equal(layer.data.ravel(), m.is_sea)
    np.testing.assert_array_equal(m.is_sea, m.elevation_flat < m.sea_level_series[3000])
    assert layer.data.sum() > sea0  # coasts drowned
    assert portrayal.shown_year(m) == 3000
    assert "Year 3,000" in portrayal.status_text(m)


def test_ui_and_headless_runs_agree():
    """A model made the way the UI's Reset makes it (seeds as keyword arguments), stepped
    through the UI's step buttons, matches a headless run with the same seeds. Small test
    world; the UI itself builds the default config the same way."""
    ui_model_template = small_model(history_seed=0, years=2)
    params = {"world_seed": SEED, "history_seed": SEED}
    kwargs = _build_model_init_kwargs(ui_model_template, params, add_scenario_when_empty=True,
                                      require_model_accepts_scenario=True)
    assert kwargs == params
    ui = type(ui_model_template)(config=small_config(), **kwargs)
    years = 300
    for n in (1, 10, 100, 189):
        app.advance(ui, n)
    headless = small_model(world_seed=SEED, history_seed=SEED, years=years)
    headless.run()
    pd.testing.assert_frame_equal(ui.datacollector.get_model_vars_dataframe(),
                                  headless.datacollector.get_model_vars_dataframe())
    np.testing.assert_array_equal(ui.is_sea, headless.is_sea)
    for mode in ("forage", "herd", "farm"):
        np.testing.assert_array_equal(ui.capacity[mode], headless.capacity[mode])
    n = headless.settlements.n
    assert ui.settlements.n == n
    for col in ("population", "cell", "lineage", "parent", "ended"):
        np.testing.assert_array_equal(getattr(ui.settlements, col)[:n], getattr(headless.settlements, col)[:n])


def test_advance_stops_at_end_of_run():
    m = small_model(years=5)
    app.advance(m, 100)
    assert m.year == 5 and not m.running


@pytest.mark.slow
def test_initial_model_uses_config_seeds():
    run = app.initial_state()[0].config.run
    assert (run.world_seed, run.history_seed) == (app.CONFIG.run.world_seed, app.CONFIG.run.history_seed)


def test_app_page_builds():
    assert app.Page is not None
    assert app.SEEDS == ("world_seed", "history_seed")


def test_lineage_overlay_colors_each_settlement_by_group(model):
    """The lineage layer holds founding group + 1 on settled cells (0 elsewhere) and the
    overlay draws group k in the k-th configured colour."""
    t = model.settlements
    ids = t.alive_ids()
    data = model.lineage_layer.data.ravel()
    assert (data[t.cell[ids]] == t.lineage[ids] + 1).all()
    assert (data > 0).sum() == len(ids)
    renderer = app.make_renderer(model, overlay="lineage")
    coll = renderer.canvas.collections[-1]
    cid = mesh_order_to_cid(model.geom)
    face = coll.get_facecolor()
    lin = data[cid]
    for k, color in enumerate(app.VIZ.lineage_colors):
        if (lin == k + 1).any():
            np.testing.assert_allclose(face[lin == k + 1], np.tile(to_rgba(color), ((lin == k + 1).sum(), 1)), atol=1e-6)
    assert (face[lin == 0, 3] == 0).all()
    assert len(set(app.VIZ.lineage_colors)) == len(app.CONFIG.founders.names)
    assert "group A" in portrayal.legend_html(app.VIZ, "lineage", app.CONFIG.founders.names)


@pytest.fixture(scope="module")
def grown():
    """A small-world run with several polities of several groups."""
    m = small_model({"polities": {"secession_base": 0.05}}, years=250)
    m.run()
    return m


def _gid(ax, gid):
    return [c for c in ax.collections if c.get_gid() == gid]


def test_polity_overlay_fills_claimed_cells_by_group_and_density(grown):
    """Each claimed cell in its founding group's hue; brighter or darker by population
    density on a log scale (darker: more people); unclaimed cells unfilled."""
    m, vcfg = grown, app.VIZ
    renderer = app.make_renderer(m, overlay="polity")
    fig = app.draw_map(m, renderer, "polity", "population density")
    (fill,) = _gid(fig.axes[0], "polity_fill")
    cells, colors = portrayal.polity_fill(m, vcfg)
    claimed = np.flatnonzero(m.polity_layer.data.ravel() > 0)
    np.testing.assert_array_equal(cells, claimed)
    np.testing.assert_allclose(fill.get_facecolor(), colors, atol=1e-6)
    lineage = m.polities.lineage[m.polity_layer.data.ravel().astype(int)[cells] - 1]
    shades = portrayal.polity_shades(vcfg)
    for g in np.unique(lineage).tolist():  # every colour of a group is one of its own shades
        own = {tuple(np.round(s, 6)) for s in shades[g]}
        assert all(tuple(np.round(c, 6)) in own for c in colors[lineage == g])
    # Darker with more people: brightness falls with the density share.
    share = portrayal.density_share(m, vcfg)[cells]
    bright = colors[:, :3].sum(axis=1)
    g0 = lineage == lineage[0]
    order = np.argsort(share[g0], kind="stable")
    assert (np.diff(bright[g0][order]) <= 1e-9).all()  # never brighter with more people
    ramp = np.array(shades[0])[:, :3].sum(axis=1)
    assert (np.diff(ramp) < 0).all()  # the steps themselves darken strictly
    assert list(portrayal.SHADE_BY) == list(vcfg.polity_shade_by) == ["population density"]
    # The Mesa layer for polities is not drawn: the fill above replaces it.
    style = portrayal.make_layer_style(vcfg, "polity")
    assert style(m.polity_layer) is None


def test_polity_borders_thick_between_groups_thin_within(grown):
    m, vcfg = grown, app.VIZ
    peoples, polities = portrayal.polity_borders(m)
    assert len(peoples) + len(polities) > 0
    claimed = m.polity_layer.data.ravel().astype(int) - 1
    pos = m.geom.positions
    # Each segment is a shared hex side (length 1) between two claimed cells of different polities.
    for seg in np.concatenate([peoples, polities])[:100]:
        assert np.linalg.norm(seg[1] - seg[0]) == pytest.approx(1.0)
        mid = seg.mean(axis=0)
        near = np.argsort(np.linalg.norm(pos - mid, axis=1))[:2]
        assert claimed[near[0]] != claimed[near[1]] and (claimed[near] >= 0).all()
    fig = app.draw_map(m, app.make_renderer(m, overlay="polity"), "polity", "population density")
    ax = fig.axes[0]
    for gid, width, seg in (("people_borders", vcfg.border_peoples_width, peoples),
                            ("polity_borders", vcfg.border_polities_width, polities)):
        if len(seg):
            (line,) = _gid(ax, gid)
            (outline,) = _gid(ax, gid + "_outline")
            assert line.get_linewidth()[0] == pytest.approx(width)
            assert outline.get_linewidth()[0] == pytest.approx(width + vcfg.border_outline_width)
            assert outline.get_zorder() < line.get_zorder()
    assert vcfg.border_peoples_width > vcfg.border_polities_width


def test_map_plot_and_legend_use_one_lineage_palette(grown):
    vcfg = app.VIZ
    palette = portrayal.lineage_palette(vcfg)
    assert palette == ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00")  # A red, B blue, C green, D purple, E orange
    names = app.CONFIG.founders.names
    # The population-per-group plot.
    assert app.GROUP_PLOT_COLORS == {f"pop_{n}": c for n, c in zip(names, palette)}
    # The legends: each row its swatch then its own label, in palette order.
    for overlay, label in (("lineage", "group {}"), ("polity", "polities of group {}")):
        html = portrayal.legend_html(vcfg, overlay, names)
        rows = html.split('<div style="display:flex')[1:]
        assert len(rows) == len(names)
        for row, n, c in zip(rows, names, palette):
            assert f"background:{c}" in row and row.index(f"background:{c}") < row.index(label.format(n))
    assert "one per polity" not in portrayal.legend_html(vcfg, "polity", names)
    # The map: the lineage overlay draws group k in palette[k]; the polity fill in shades of it.
    m = grown
    face = app.make_renderer(m, overlay="lineage").canvas.collections[-1].get_facecolor()
    lin = m.lineage_layer.data.ravel()[mesh_order_to_cid(m.geom)]
    for k, c in enumerate(palette):
        if (lin == k + 1).any():
            np.testing.assert_allclose(face[lin == k + 1][0], to_rgba(c), atol=1e-6)
    shades = portrayal.polity_shades(vcfg)
    for k, c in enumerate(palette):
        base = np.array(to_rgba(c)[:3])
        # The shade ramp runs from lighter to darker through the palette colour itself.
        assert np.abs(np.array(shades[k])[:, :3] - base).sum(axis=1).min() < 0.15
    assert "Polities:" in portrayal.status_text(m)
    assert "iron ore" in portrayal.legend_html(vcfg, "resources", names)


def test_chronicle_filter_reads_one_snapshot_while_events_are_appended():
    """The model can append events while the browser chronicle filters (Round 6 bug:
    "operands could not be broadcast together"). Events appended between the column
    reads of one select wait for the next select."""
    from deluge.data import chronicle

    m = small_model(years=20)
    for _ in range(5):
        m.step()
    view = chronicle.Chronicle(m)

    class AppendingWhileRead:
        """An actions filter whose iteration appends events, as a running model would."""

        def __iter__(self):
            for _ in range(3):
                m.log_event("moved", [("settlement", 0)], 0, 1.0)
            return iter(["moved", "founded"])

    n = len(m.events)
    sel = view.select(actions=AppendingWhileRead(), groups={0, 1, 2, 3, 4}, year_min=0)
    assert len(m.events) == n + 3 and (sel < n).all()
    # Half-appended record (year written, other columns not yet): left for the next call.
    m.events.year.append(m.year)
    later = view.select(actions={"moved"})
    m.events.year.pop()
    assert later.max() == n + 2
    assert len(view.text().splitlines()) == n + 3
