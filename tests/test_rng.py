import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from deluge.rng import RngStreams

NAMES = ("mesa", "world", "climate", "war")


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=0, max_value=2**63 - 1))
def test_same_seed_same_draws(seed):
    a, b = RngStreams(seed, NAMES), RngStreams(seed, NAMES)
    for name in NAMES:
        np.testing.assert_array_equal(a[name].random(16), b[name].random(16))


def test_different_seeds_differ():
    a, b = RngStreams(1, NAMES), RngStreams(2, NAMES)
    assert not np.array_equal(a["world"].random(16), b["world"].random(16))


def test_streams_are_distinct():
    s = RngStreams(42, NAMES)
    draws = [tuple(s[n].random(8)) for n in NAMES]
    assert len(set(draws)) == len(NAMES)


def test_adding_or_reordering_streams_does_not_shift_others():
    base = RngStreams(42, ("world", "climate"))
    grown = RngStreams(42, ("war", "climate", "memory", "world"))
    for name in ("world", "climate"):
        np.testing.assert_array_equal(base[name].random(16), grown[name].random(16))


def test_drawing_from_one_stream_does_not_shift_another():
    a, b = RngStreams(42, NAMES), RngStreams(42, NAMES)
    a["war"].random(1000)
    np.testing.assert_array_equal(a["climate"].random(16), b["climate"].random(16))


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        RngStreams(0, ("world", "world"))
