"""World layer: grid, world sources, climate, carrying capacity."""

from __future__ import annotations

import numpy as np

from deluge.config import Config
from deluge.world.worlddata import Terrain, WorldData

__all__ = ["Terrain", "WorldData", "generate_world"]


def generate_world(cfg: Config, rng: np.random.Generator) -> WorldData:
    """Build the world from the configured source. Every source returns the same WorldData."""
    if cfg.world.source == "procedural":
        from deluge.world import procedural

        return procedural.generate_world(cfg, rng)
    if cfg.world.source == "earth":
        from deluge.world import earth

        return earth.generate_world(cfg, rng)
    raise ValueError(f"unknown world.source {cfg.world.source!r}")
