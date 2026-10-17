"""Phase 3a, subsistence: mode shares, food production, consumption and storage (spec section 6).

Food is counted in rations (one person-year of food). A settlement splits its people
over the modes it can practise (foraging always; herding and farming once a technique
unlocks them) in ``shares``. With ``K_m`` its cell's capacity under mode m (this year's
climate in it; technique multipliers; forage health for foraging), each mode's workers
gather up to ``a_m`` rations each but never more than that mode's capacity: food per head
from production = ``sum_m min(share_m a_m, K_m / population)``, times technique and tool
bonuses. Workers in a mode the cell cannot support (``K_m = 0``) produce nothing. Food
available = production + what is left of the store after spoilage; food per head
(``food_ratio``) drives growth and famine in the demography phase. A settlement's
capacity (``fed_capacity``) is the population its mode mix feeds at one ration per head;
for a pure forager band it is the cell's forage capacity.

Mode choice: each year a settlement moves ``mode_shift_rate`` of its people toward the
mode whose shift promises the fastest growth (farming grows fastest but yields less per
worker; foraging the reverse), or stays. Its polity brain influences the choice: a shift
toward farming (herding) gets ``mode_brain_bias`` x (farming (herding) output - 0.5)
added to its expected growth. Leaving farming is sticky (``farm_stickiness``).
"""

from __future__ import annotations

import numpy as np

from deluge.agents.settlement import MODES
from deluge.brains.network import OUT
from deluge.systems.knowledge import production_multiplier


def mode_parameters(cfg) -> tuple[np.ndarray, np.ndarray]:
    """(abundance, max growth) per mode, in MODES order."""
    a = np.array([cfg.forage_abundance, cfg.herd_abundance, cfg.farm_abundance])
    r = np.array([cfg.forage_max_growth, cfg.herd_max_growth, cfg.farm_max_growth])
    return a, r


def mode_capacity(model, knows: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """(S, modes) capacity of each row's cell for a settlement holding ``knows``."""
    g, cap = model.knowledge, model.capacity
    calendar = g.multiplier(knows, "drought_loss") < 1.0
    farm = np.where(calendar, cap["farm_calendar"][cells], cap["farm"][cells])
    farm = farm * g.multiplier(knows, "farm_yield")
    river = model.world.river.ravel()[cells] > 0
    farm = farm * np.where(river, g.multiplier(knows, "river_farm_yield"), 1.0)
    return np.column_stack([
        cap["forage"][cells] * model.forage_health[cells],
        cap["herd"][cells] * g.multiplier(knows, "herd_yield"),
        farm,
    ])


def food_per_head(shares: np.ndarray, k: np.ndarray, pop, cfg) -> np.ndarray:
    """Rations produced per head (before bonuses): each mode's workers are capped by that
    mode's capacity. ``pop`` is per row (or a scalar)."""
    a, _ = mode_parameters(cfg)
    per = np.asarray(pop, dtype=float)
    per = per[:, None] if per.ndim else per
    return np.minimum(shares * a[None, :], k / np.maximum(per, 1e-12)).sum(axis=1)


def fed_capacity(shares: np.ndarray, k: np.ndarray, cfg) -> np.ndarray:
    """Population each row's mode mix feeds at one ration per head: the largest n with
    food per head >= 1, i.e. sum_m min(c_m n, K_m) >= n (c_m = share_m a_m). A sum of
    minima is the minimum over subsets S of modes of C_S n + K_notS, so n = the least
    K_notS / (1 - C_S) over the subsets with C_S < 1 (0 when all c_m sum to less than 1)."""
    a, _ = mode_parameters(cfg)
    c = shares * a[None, :]
    subsets = _mode_subsets(c.shape[1])
    c_in = c @ subsets.T  # (S, subsets)
    k_out = k @ (1.0 - subsets).T
    with np.errstate(divide="ignore"):
        bound = np.where(c_in < 1.0, k_out / (1.0 - c_in), np.inf)
    return bound.min(axis=1)


def _mode_subsets(n_modes: int) -> np.ndarray:
    """(2^n_modes, n_modes) 0/1 rows: every subset of the modes."""
    return ((np.arange(2 ** n_modes)[:, None] >> np.arange(n_modes)[None, :]) & 1).astype(float)


def expected_growth(shares: np.ndarray, k: np.ndarray, pop: np.ndarray, cfg) -> np.ndarray:
    """Net growth per year a settlement could expect with these shares (no store)."""
    a, r = mode_parameters(cfg)
    a_s, r_s = shares @ a, shares @ r
    fed = food_per_head(shares, k, pop, cfg)
    good = r_s * np.clip((fed - 1.0) / (a_s - 1.0), 0.0, 1.0)
    bad = -np.minimum(cfg.famine_max_mortality, cfg.famine_mortality * (1.0 - fed))
    return np.where(fed >= 1.0, good, bad)


def available_shares(shares: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Shares with unavailable modes removed (their people go to the others, foraging if
    nothing else is left)."""
    s = np.where(available, shares, 0.0)
    total = s.sum(axis=1)
    s[total <= 0, 0] = 1.0
    return s / s.sum(axis=1, keepdims=True)


def shift_modes(model, ids: np.ndarray, k: np.ndarray) -> np.ndarray:
    d = model.config.demography
    t = model.settlements
    available = model.knowledge.mode_available(t.knows[ids])
    shares = available_shares(t.shares[ids], available)
    pop = t.population[ids]
    farm = MODES.index("farm")
    brain = model.brains.output[t.polity[ids]]
    bias = np.zeros((len(ids), len(MODES)))
    bias[:, farm] = d.mode_brain_bias * (brain[:, OUT["farming"]] - 0.5)
    bias[:, MODES.index("herd")] = d.mode_brain_bias * (brain[:, OUT["herding"]] - 0.5)
    best = shares
    best_g = expected_growth(shares, k, pop, d)
    for m in range(len(MODES)):
        target = np.zeros(len(MODES))
        target[m] = 1.0
        cand = (1.0 - d.mode_shift_rate) * shares + d.mode_shift_rate * target
        leaving = cand[:, farm] < shares[:, farm]
        if leaving.any():  # sticky: farming shrinks only at a fraction of the rate
            new_farm = shares[:, farm] - (shares[:, farm] - cand[:, farm]) * d.farm_stickiness
            rest = 1.0 - new_farm
            others = cand.copy()
            others[:, farm] = 0.0
            others *= (rest / np.maximum(others.sum(axis=1), 1e-12))[:, None]
            others[:, farm] = new_farm
            cand = np.where(leaving[:, None], others, cand)
        ok = available[:, m]
        g = expected_growth(cand, k, pop, d) + bias[:, m]
        better = ok & (g > best_g + 1e-12)
        best = np.where(better[:, None], cand, best)
        best_g = np.where(better, g, best_g)
    t.shares[ids] = best
    return best


def run(model) -> None:
    d = model.config.demography
    g = model.knowledge
    t = model.settlements
    ids = t.alive_ids()
    cells = t.cell[ids]
    pop = t.population[ids]
    knows = t.knows[ids]
    k = mode_capacity(model, knows, cells)
    shares = shift_modes(model, ids, k)

    production = pop * food_per_head(shares, k, pop, d) * production_multiplier(model, ids)
    model.stats["surplus"] += float(np.maximum(production - pop, 0.0).sum())  # reporting: rations beyond one a head
    spoilage = np.clip(d.spoilage * g.multiplier(knows, "spoilage"), 0.0, 1.0)
    available = production + t.store[ids] * (1.0 - spoilage)
    ratio = available / np.maximum(pop, 1e-12)
    t.food_ratio[ids] = ratio
    cap_years = np.where(knows, g.effect["store_cap_years"][None, :], np.nan)
    cap_years = np.fmax(np.nanmax(np.where(np.isnan(cap_years), -np.inf, cap_years), axis=1), d.store_cap_years)
    t.store[ids] = np.minimum(np.maximum(available - pop, 0.0), pop * cap_years)

    # Overuse feedback on the cells' forage health (1 = untouched).
    overuse = np.zeros(model.geom.n_cells)
    foragers = pop * shares[:, 0]
    overuse[cells] = np.clip(foragers / np.maximum(k[:, 0], 1e-12) - 1.0, 0.0, 1.0)
    h = model.forage_health
    h -= d.wild_food_degrade * overuse
    h += d.wild_food_recover * (1.0 - h)
    np.clip(h, d.wild_food_floor, 1.0, out=h)
