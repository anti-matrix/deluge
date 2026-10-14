import pytest

from deluge.config import load_config
from deluge.rng import RngStreams
from deluge.world import procedural
from deluge.world.grid import HexGeometry

WORLD_SEEDS = (1, 2, 3, 4, 5)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def geom(cfg):
    return HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)


def make_world(cfg, seed, with_report=False):
    world, report = procedural.generate_world_with_report(cfg, RngStreams(seed, cfg.rng.streams)["world"])
    return (world, report) if with_report else world


@pytest.fixture(scope="session")
def generated(cfg):
    """Default-config worlds and their generation reports, made once per session."""
    return {seed: make_world(cfg, seed, with_report=True) for seed in WORLD_SEEDS}


@pytest.fixture(scope="session")
def worlds(generated):
    return {seed: wr[0] for seed, wr in generated.items()}
