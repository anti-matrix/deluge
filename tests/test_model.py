import time

import mesa
import numpy as np
import pandas as pd

from deluge.model import DelugeModel


def test_model_is_mesa_model():
    m = DelugeModel(years=3)
    assert isinstance(m, mesa.Model)
    assert m.year == 0


def test_steps_full_run_under_5_seconds():
    m = DelugeModel(seed=42)
    t0 = time.perf_counter()
    m.run()
    elapsed = time.perf_counter() - t0
    assert m.year == 11000
    assert m.steps == 11000
    assert not m.running
    assert elapsed < 5.0, f"11,000 empty steps took {elapsed:.2f}s"


def test_same_seed_identical_output():
    a, b = DelugeModel(seed=7, years=500), DelugeModel(seed=7, years=500)
    a.run()
    b.run()
    pd.testing.assert_frame_equal(
        a.datacollector.get_model_vars_dataframe(), b.datacollector.get_model_vars_dataframe()
    )
    assert a.streams.state() == b.streams.state()
    np.testing.assert_array_equal(a.rng.random(8), b.rng.random(8))
    assert a.random.random() == b.random.random()


def test_mesa_rng_derives_from_seed():
    a, b = DelugeModel(seed=1, years=1), DelugeModel(seed=2, years=1)
    assert not np.array_equal(a.rng.random(8), b.rng.random(8))
