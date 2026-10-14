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
        assert drawn == set(portrayal.MAP_LAYERS) | {overlay}


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
