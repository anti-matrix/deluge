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

from deluge.model import DelugeModel  # noqa: E402
from deluge.viz import app, portrayal  # noqa: E402
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS  # noqa: E402
from deluge.world.worlddata import Terrain  # noqa: E402

SEED = 3


@pytest.fixture(scope="module")
def model():
    return DelugeModel(seed=SEED)


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


def test_map_draws_terrain_then_sea(model):
    renderer = app.make_renderer(model)
    terrain_coll, sea_coll = renderer.canvas.collections[-2:]
    cid = mesh_order_to_cid(model.geom)
    terrain = model.world.terrain.ravel()[cid]
    expected = np.array([to_rgba(TERRAIN_COLORS[Terrain(t)]) for t in terrain])
    np.testing.assert_allclose(terrain_coll.get_facecolor(), expected, atol=1e-6)
    sea = sea_coll.get_facecolor()
    is_sea = model.is_sea[cid]
    np.testing.assert_allclose(sea[is_sea], np.tile(to_rgba(SEA_COLOR), (is_sea.sum(), 1)), atol=1e-6)
    assert (sea[~is_sea, 3] == 0).all()  # land: sea layer fully transparent


def test_only_map_layers_are_drawn(model):
    drawn = {name for name, layer in model.grid._mesa_property_layers.items()
             if name != "empty" and portrayal.map_layer_style(layer) is not None}
    assert drawn == set(portrayal.MAP_LAYERS)


def test_sea_layer_follows_sea_level():
    m = DelugeModel(seed=SEED)
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
    """A model made the way the UI's Reset makes it, stepped through the UI's step
    buttons, matches a headless run with the same seed."""
    ui_model_template = DelugeModel(seed=0, years=2)
    kwargs = _build_model_init_kwargs(ui_model_template, {"seed": SEED}, add_scenario_when_empty=True,
                                      require_model_accepts_scenario=True)
    assert kwargs == {"seed": SEED}
    ui = type(ui_model_template)(**kwargs)
    years = 1800
    for n in (1, 10, 100, 1689):
        app.advance(ui, n)
    headless = DelugeModel(seed=SEED, years=years)
    headless.run()
    pd.testing.assert_frame_equal(ui.datacollector.get_model_vars_dataframe(),
                                  headless.datacollector.get_model_vars_dataframe())
    np.testing.assert_array_equal(ui.is_sea, headless.is_sea)
    for mode in ("forage", "herd", "farm"):
        np.testing.assert_array_equal(ui.capacity[mode], headless.capacity[mode])


def test_advance_stops_at_end_of_run():
    m = DelugeModel(seed=SEED, years=5)
    app.advance(m, 100)
    assert m.year == 5 and not m.running


def test_app_page_builds():
    assert app.Page is not None
    assert app.initial_model.config.run.seed == app.CONFIG.run.seed
    assert list(app.MODEL_PARAMS) == ["seed"]
