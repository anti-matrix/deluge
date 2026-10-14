"""Real-topography importer (optional, round 12).

Will resample local ETOPO/GEBCO elevation, HydroSHEDS rivers and WorldClim/CHELSA
climate rasters to the hex grid, then run the terrain, fertility and wild-food stages of
``world/procedural.py``. It must return the same ``WorldData`` as the procedural source.
"""

from __future__ import annotations

import numpy as np

from deluge.config import Config
from deluge.world.worlddata import WorldData


def generate_world(cfg: Config, rng: np.random.Generator) -> WorldData:
    raise NotImplementedError("world.source 'earth' is planned for round 12; use 'procedural'")
