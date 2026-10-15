"""Brain input vector assembly (spec section 10), normalised to about -1..1.

Inputs per polity (``FEATURES``, then ``brains.memory_inputs`` memory slots and
``brains.recurrent_units`` recurrent units):

- Own state: log population, food per head - 1, store level, cohesion, pressure, elite
  ratio, number of settlements, territory growth (relative change in settlements over
  ``growth_window_years``).
- Knowledge: techniques known at the capital per category (subsistence, building, military).
- Neighbours (polities with a settlement in contact with one of its own): strength ratio
  to the strongest and the weakest (ln of own over theirs, 0 with no neighbour), polities at
  war with, mean culture, appearance and dialect distance across its contacts with foreign
  settlements, the same three distances between its members and its core (population-
  weighted: culture from the polity's culture, appearance and dialect from the capital),
  recent battles won and lost.
- Environment: the year's climate anomaly, share of its people in drought, capacity
  headroom (1 - people / capacity, last counted by the cohesion phase).
- Memory: zeros until memory exists (Round 8).

Appearance and dialect distances are the perceived ones (``systems/identity.py``: zero
with ``blind_perception``); the weights reading them start at zero (``MARKER_FEATURES``,
``brains.marker_init_sigma``).

Target features of each candidate war target (``TARGET_FEATURES``), scored by the polity's
per-neighbour head: a constant, the estimated win chance, the target's stores, the
fertility of its cell, culture distance from the attacker, perceived appearance and dialect
distance from the attacker's capital, already at war with its polity, it is its polity's
capital.
"""

from __future__ import annotations

import numpy as np

from deluge.config import BrainsConfig
from deluge.systems import identity
from deluge.systems.cohesion import culture_distance, settlement_contacts

FEATURES = (
    "log_population", "food", "store", "cohesion", "pressure", "elite_ratio", "settlements", "growth",
    "techniques_subsistence", "techniques_building", "techniques_military",
    "strongest_neighbour", "weakest_neighbour", "hostile", "culture_neighbours", "appearance_neighbours",
    "dialect_neighbours", "culture_periphery", "appearance_periphery", "dialect_periphery", "wins", "losses",
    "temp_anomaly", "drought", "headroom",
)
MARKER_FEATURES = ("appearance_neighbours", "dialect_neighbours", "appearance_periphery", "dialect_periphery")
TECHNIQUE_CATEGORIES = ("subsistence", "building", "military")

TARGET_FEATURES = ("bias", "win_chance", "stores", "fertility", "culture_distance", "appearance_distance",
                   "dialect_distance", "at_war", "capital")
MARKER_TARGET_FEATURES = ("appearance_distance", "dialect_distance")

STRENGTH_EPS = 1e-12


def input_names(cfg: BrainsConfig) -> tuple[str, ...]:
    return (FEATURES + tuple(f"memory_{i}" for i in range(cfg.memory_inputs))
            + tuple(f"recurrent_{i}" for i in range(cfg.recurrent_units)))


def _weighted_mean(groups: np.ndarray, values: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    total = np.bincount(groups, weights=weights, minlength=n)
    return np.bincount(groups, weights=weights * values, minlength=n) / np.maximum(total, STRENGTH_EPS)


def assemble(model, pids: np.ndarray) -> np.ndarray:
    """(len(pids), inputs) brain inputs of the living polities ``pids``."""
    from deluge.systems import war  # war imports this module

    cfg = model.config.brains
    fc = cfg.features
    t, P, B = model.settlements, model.polities, model.brains
    names = model.brain_layout.inputs
    col = {name: i for i, name in enumerate(names)}
    x = np.zeros((len(pids), len(names)))
    n = P.n
    ids = t.alive_ids()
    pol = t.polity[ids]
    pop = t.population[ids]

    def put(name: str, values) -> None:
        x[:, col[name]] = values

    # Own state.
    people = np.bincount(pol, weights=pop, minlength=n)
    safe = np.maximum(people, 1.0)
    n_set = np.bincount(pol, minlength=n)
    fed = np.bincount(pol, weights=pop * t.food_ratio[ids], minlength=n) / safe
    store = np.bincount(pol, weights=t.store[ids], minlength=n) / safe
    put("log_population", np.clip((np.log10(safe[pids]) - fc.log_population_center) / fc.log_population_scale,
                                  -1.0, 1.0))
    put("food", np.clip(fed[pids] - 1.0, -1.0, 1.0))
    put("store", store[pids] / (store[pids] + fc.store_half_years))
    put("cohesion", 2.0 * P.cohesion[pids] - 1.0)
    put("pressure", np.tanh(P.pressure[pids]))
    put("elite_ratio", np.tanh(P.elite_ratio[pids] / fc.elite_reference))
    put("settlements", np.tanh(n_set[pids] / fc.settlements_reference))
    hist = B.settlement_history
    fresh = hist[pids, 0] < 0
    hist[pids[fresh]] = n_set[pids[fresh]][:, None]
    slot = model.year % hist.shape[1]
    then = hist[pids, slot]
    put("growth", np.tanh((n_set[pids] - then) / np.maximum(then, 1)))
    hist[pids, slot] = n_set[pids]

    # Knowledge at the capital.
    knows = t.knows[P.capital[pids]]
    techniques = model.config.knowledge.techniques
    for cat in TECHNIQUE_CATEGORIES:
        mask = np.array([techniques[name].category == cat for name in model.knowledge.names])
        put(f"techniques_{cat}", np.tanh(knows[:, mask].sum(axis=1) / fc.technique_reference))

    # Neighbours.
    contacts = settlement_contacts(model, ids)
    other = np.where(contacts >= 0, t.polity[np.maximum(contacts, 0)], -1)
    i, k = np.nonzero((other >= 0) & (other != pol[:, None]))
    s, o, a, b = ids[i], contacts[i, k], pol[i], other[i, k]
    strength = np.maximum(war.polity_strength(model), STRENGTH_EPS)
    strongest = np.full(n, -np.inf)
    weakest = np.full(n, np.inf)
    np.maximum.at(strongest, a, strength[b])
    np.minimum.at(weakest, a, strength[b])
    has = np.isfinite(strongest[pids])
    own = strength[pids]
    put("strongest_neighbour", np.where(has, np.tanh(np.log(own / np.where(has, strongest[pids], 1.0))
                                                     / fc.strength_ratio_scale), 0.0))
    put("weakest_neighbour", np.where(has, np.tanh(np.log(own / np.where(has, weakest[pids], 1.0))
                                                   / fc.strength_ratio_scale), 0.0))
    put("hostile", np.tanh(war.at_war_counts(model)[pids] / fc.hostile_reference))
    ones = np.ones(len(a))
    app, dia = identity.pair_distances(model, s, o)
    put("culture_neighbours", _weighted_mean(a, culture_distance(P.culture[a], t.culture[o]), ones, n)[pids])
    put("appearance_neighbours", _weighted_mean(a, app, ones, n)[pids])
    put("dialect_neighbours", _weighted_mean(a, dia, ones, n)[pids])
    app, dia = identity.core_distances(model, ids)
    put("culture_periphery", _weighted_mean(pol, culture_distance(t.culture[ids], P.culture[pol]), pop, n)[pids])
    put("appearance_periphery", _weighted_mean(pol, app, pop, n)[pids])
    put("dialect_periphery", _weighted_mean(pol, dia, pop, n)[pids])
    put("wins", np.tanh(P.wins_recent[pids] / fc.war_record_reference))
    put("losses", np.tanh(P.losses_recent[pids] / fc.war_record_reference))

    # Environment.
    put("temp_anomaly", np.tanh(model.temp_anomaly / fc.temp_anomaly_scale))
    dry = (model.shocks.rain_factor[t.cell[ids]] < 1.0).astype(float)
    put("drought", _weighted_mean(pol, dry, pop, n)[pids])
    put("headroom", np.clip(1.0 - P.load[pids], -1.0, 1.0))

    # Memory slots stay zero (Round 8); recurrent units carry last year's hidden state.
    if cfg.recurrent_units:
        x[:, col["recurrent_0"]:col["recurrent_0"] + cfg.recurrent_units] = B.state[pids]
    return x


def target_features(model, a: np.ndarray, b: np.ndarray, x: np.ndarray, p_win: np.ndarray) -> np.ndarray:
    """(n, TARGET_FEATURES) of candidate targets: settlement ``x`` of polity ``b`` for
    attacker ``a``, with the estimated win chance ``p_win``."""
    w = model.config.war
    t, P = model.settlements, model.polities
    f = np.zeros((len(a), len(TARGET_FEATURES)))
    col = {name: i for i, name in enumerate(TARGET_FEATURES)}
    years = t.store[x] / np.maximum(t.population[x], 1.0)
    app, dia = identity.pair_distances(model, P.capital[a], x)
    f[:, col["bias"]] = 1.0
    f[:, col["win_chance"]] = p_win
    f[:, col["stores"]] = years / (years + w.store_half_years)
    f[:, col["fertility"]] = model.world.fertility.ravel()[t.cell[x]]
    f[:, col["culture_distance"]] = culture_distance(P.culture[a], t.culture[x])
    f[:, col["appearance_distance"]] = app
    f[:, col["dialect_distance"]] = dia
    f[:, col["at_war"]] = [(min(p, q), max(p, q)) in model.wars for p, q in zip(a.tolist(), b.tolist())]
    f[:, col["capital"]] = x == P.capital[b]
    return f
