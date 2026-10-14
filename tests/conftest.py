"""Shared fixtures.

Fast tests use the small test world (``configs/worlds/small_test.yaml``: 64 x 40 cells,
fixed world seed). Full-size worlds (``cfg``, ``geom``, ``generated``, ``worlds``) are for
tests marked ``slow``, which a plain ``pytest`` skips (``pytest -m slow`` runs them).
"""

import os
from pathlib import Path

import pytest

from deluge.config import DEFAULT_CONFIG_PATH, load_config, read_overrides
from deluge.model import DelugeModel
from deluge.rng import RngStreams
from deluge.world import procedural
from deluge.world.grid import HexGeometry

WORLD_SEEDS = (1, 2, 3, 4, 5)
SMALL_OVERRIDES_PATH = Path(DEFAULT_CONFIG_PATH).parent / "worlds" / "small_test.yaml"
SMALL_OVERRIDES = read_overrides(SMALL_OVERRIDES_PATH)


def pytest_xdist_auto_num_workers(config):
    """``-n auto``: one worker per core but one."""
    return max(1, (os.cpu_count() or 2) - 1)


def small_config(overrides: dict | None = None):
    """The small test world's config, with further nested overrides."""
    cfg = load_config(overrides=SMALL_OVERRIDES)
    return cfg.with_overrides(overrides) if overrides else cfg


def small_model(overrides: dict | None = None, **kw) -> DelugeModel:
    """A model on the small test world (``kw``: world_seed, history_seed, years)."""
    return DelugeModel(config=small_config(overrides), **kw)


def make_world(cfg, seed, with_report=False):
    world, report = procedural.generate_world_with_report(cfg, RngStreams(seed, cfg.rng.streams)["world"])
    return (world, report) if with_report else world


@pytest.fixture(scope="session")
def small_cfg():
    return small_config()


@pytest.fixture(scope="session")
def small_geom(small_cfg):
    return HexGeometry(small_cfg.grid.width, small_cfg.grid.height, small_cfg.grid.torus)


@pytest.fixture(scope="session")
def small_generated(small_cfg):
    """The small test world and its generation report."""
    return make_world(small_cfg, small_cfg.run.world_seed, with_report=True)


@pytest.fixture(scope="session")
def small_world(small_generated):
    return small_generated[0]


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def geom(cfg):
    return HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)


@pytest.fixture(scope="session")
def generated(cfg):
    """Default-config worlds and their generation reports, made once per session (slow tests)."""
    return {seed: make_world(cfg, seed, with_report=True) for seed in WORLD_SEEDS}


@pytest.fixture(scope="session")
def worlds(generated):
    return {seed: wr[0] for seed, wr in generated.items()}
