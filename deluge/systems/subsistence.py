"""Phase 3a, subsistence: food production, consumption and storage (spec section 6).

Foraging only until Round 4. Food is counted in rations (one person-year of food). A band
gathers up to ``forage_abundance`` rations per person, but never more than its cell's
foraging capacity (this year's climate and drought already in it) times the cell's
forage health. Food available = production + what is left of the store after spoilage;
food per head (``food_ratio``) drives growth and famine in the demography phase. Surplus
goes to the store up to a small cap. Foraging above capacity wears the cell's forage
health down; it recovers slowly when the pressure is off.
"""

from __future__ import annotations

import numpy as np


def run(model) -> None:
    d = model.config.demography
    t = model.settlements
    ids = t.alive_ids()
    cells = t.cell[ids]
    pop = t.population[ids]
    k = model.capacity["forage"][cells] * model.forage_health[cells]
    production = np.minimum(pop * d.forage_abundance, k)
    available = production + t.store[ids] * (1.0 - d.spoilage)
    ratio = available / np.maximum(pop, 1e-12)
    t.food_ratio[ids] = ratio
    t.store[ids] = np.minimum(np.maximum(available - pop, 0.0), pop * d.store_cap_years)

    # Overuse feedback on the cells' forage health (1 = untouched).
    overuse = np.zeros(model.geom.n_cells)
    overuse[cells] = np.clip(pop / np.maximum(k, 1e-12) - 1.0, 0.0, 1.0)
    h = model.forage_health
    h -= d.wild_food_degrade * overuse
    h += d.wild_food_recover * (1.0 - h)
    np.clip(h, d.wild_food_floor, 1.0, out=h)
