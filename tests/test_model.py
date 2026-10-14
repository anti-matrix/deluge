import pickle
import time

import mesa
import numpy as np
import pandas as pd
import pytest

from conftest import small_model
from deluge.model import DelugeModel

# Round 0 required 11,000 *empty* steps in under 5 s; the run is 12,900 years since Round
# 4b. Since Round 4 the steps run every system through knowledge and the teachers, with
# farming populations in the hundreds of millions (minutes at 160 x 96 on the dev machine). The guard is the spec's budget
# of 10 minutes per run (section 14; Round 11 profiles and speeds up).
FULL_RUN_BUDGET_S = 600.0


def test_model_is_mesa_model():
    m = small_model(years=3)
    assert isinstance(m, mesa.Model)
    assert m.year == 0


@pytest.mark.slow
def test_full_run_steps_and_budget():
    m = DelugeModel(world_seed=3, history_seed=3)
    years = m.config.run.years
    t0 = time.perf_counter()
    m.run()
    elapsed = time.perf_counter() - t0
    assert m.year == years
    assert m.steps == years
    assert not m.running
    assert elapsed < FULL_RUN_BUDGET_S, f"{years:,} steps took {elapsed:.2f}s"
    df = m.datacollector.get_model_vars_dataframe()
    assert list(df["year"]) == list(range(years))


def test_same_seed_identical_output():
    a, b = small_model(years=300), small_model(years=300)
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


@pytest.mark.slow
def test_different_seeds_differ():
    a, b = small_model(world_seed=3, history_seed=3, years=50), small_model(world_seed=2, history_seed=4, years=50)
    a.run()
    b.run()
    assert not np.array_equal(a.world.elevation, b.world.elevation)
    assert not np.array_equal(a.rng.random(8), b.rng.random(8))


@pytest.mark.slow
def test_yearly_state_follows_sea_level():
    m = DelugeModel(world_seed=3, history_seed=3, years=3001)
    m.run()
    df = m.datacollector.get_model_vars_dataframe().set_index("year")
    assert df.loc[0, "sea_level"] == -60.0
    assert df.loc[3000, "sea_level"] == 0.0
    assert df.loc[0, "land_cells"] > df.loc[1000, "land_cells"] > df.loc[3000, "land_cells"]
    np.testing.assert_array_equal(m.is_sea, m.elevation_flat < 0.0)
    np.testing.assert_array_equal(m.is_coast, ~m.is_sea & m.geom.neighbor_any(m.is_sea))
    assert (m.capacity["farm"][m.is_sea] == 0).all()


def test_yearly_sea_and_coast_flags():
    """The sea and coast flags follow the sea-level curve (small world, first centuries)."""
    m = small_model(years=301)
    m.run()
    df = m.datacollector.get_model_vars_dataframe().set_index("year")
    assert df.loc[0, "sea_level"] == -60.0
    np.testing.assert_allclose(df["sea_level"], m.sea_level_series[:301])
    np.testing.assert_array_equal(m.is_sea, m.elevation_flat < m.sea_level_series[300])
    np.testing.assert_array_equal(m.is_coast, ~m.is_sea & m.geom.neighbor_any(m.is_sea))
    assert (m.capacity["farm"][m.is_sea] == 0).all()


def test_mesa_grid_carries_world_layers():
    m = small_model(years=1)
    cell = m.grid._cells[(10, 20)]
    assert cell.elevation == m.world.elevation[10, 20]
    assert cell.terrain == m.world.terrain[10, 20]


def test_model_pickles():
    m = small_model(years=20)
    m.run()
    clone = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(clone.world.elevation, m.world.elevation)
    assert clone.config == m.config
