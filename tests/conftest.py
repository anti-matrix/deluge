import pytest

from deluge.config import load_config
from deluge.rng import RngStreams
from deluge.world import procedural
from deluge.world.grid import HexGeometry

WORLD_SEEDS = tuple(range(10))


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def geom(cfg):
    return HexGeometry(cfg.grid.width, cfg.grid.height, cfg.grid.torus)


def make_world(cfg, seed):
    return procedural.generate_world(cfg, RngStreams(seed, cfg.rng.streams)["world"])


@pytest.fixture(scope="session")
def worlds(cfg):
    """Default-config worlds for several seeds, generated once per session."""
    return {seed: make_world(cfg, seed) for seed in WORLD_SEEDS}
