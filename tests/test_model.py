import pickle
import time

import mesa
import numpy as np
import pandas as pd

from deluge.model import DelugeModel

# Round 0 required 11,000 *empty* steps in under 5 s. The steps now run the climate and
# capacity phases (about 6.4 s at 160 x 96 on the dev machine); this guard catches regressions while
# leaving room on slower CPUs. The spec's overall budget is 10 minutes per run.
FULL_RUN_BUDGET_S = 10.0


def test_model_is_mesa_model():
    m = DelugeModel(years=3)
    assert isinstance(m, mesa.Model)
    assert m.year == 0


def test_full_run_steps_and_budget():
    m = DelugeModel(seed=3)
    t0 = time.perf_counter()
    m.run()
    elapsed = time.perf_counter() - t0
    assert m.year == 11000
    assert m.steps == 11000
    assert not m.running
    assert elapsed < FULL_RUN_BUDGET_S, f"11,000 steps took {elapsed:.2f}s"
    df = m.datacollector.get_model_vars_dataframe()
    assert list(df["year"]) == list(range(11000))


def test_same_seed_identical_output():
    a, b = DelugeModel(seed=3, years=500), DelugeModel(seed=3, years=500)
    a.run()
    b.run()
    pd.testing.assert_frame_equal(
        a.datacollector.get_model_vars_dataframe(), b.datacollector.get_model_vars_dataframe()
    )
    assert a.streams.state() == b.streams.state()
    for name, arr in a.world.fields().items():
        np.testing.assert_array_equal(arr, getattr(b.world, name))
    np.testing.assert_array_equal(a.rng.random(8), b.rng.random(8))
    assert a.random.random() == b.random.random()


def test_different_seeds_differ():
    a, b = DelugeModel(seed=3, years=50), DelugeModel(seed=4, years=50)
    a.run()
    b.run()
    assert not np.array_equal(a.world.elevation, b.world.elevation)
    assert not np.array_equal(a.rng.random(8), b.rng.random(8))


def test_yearly_state_follows_sea_level():
    m = DelugeModel(seed=3, years=3001)
    m.run()
    df = m.datacollector.get_model_vars_dataframe().set_index("year")
    assert df.loc[0, "sea_level"] == -60.0
    assert df.loc[3000, "sea_level"] == 0.0
    assert df.loc[0, "land_cells"] > df.loc[1000, "land_cells"] > df.loc[3000, "land_cells"]
    np.testing.assert_array_equal(m.is_sea, m.elevation_flat < 0.0)
    np.testing.assert_array_equal(m.is_coast, ~m.is_sea & m.geom.neighbor_any(m.is_sea))
    assert (m.capacity["farm"][m.is_sea] == 0).all()


def test_mesa_grid_carries_world_layers():
    m = DelugeModel(seed=3, years=1)
    cell = m.grid._cells[(10, 20)]
    assert cell.elevation == m.world.elevation[10, 20]
    assert cell.terrain == m.world.terrain[10, 20]


def test_model_pickles():
    m = DelugeModel(seed=3, years=20)
    m.run()
    clone = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(clone.world.elevation, m.world.elevation)
    assert clone.config == m.config
