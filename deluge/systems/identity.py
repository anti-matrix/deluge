"""Appearance and dialect: perception, drift and mixing (spec "Ancestry, appearance and identity").

Markers only. This module is the only code that reads or writes the settlement table's
``appearance_mean``, ``appearance_var`` and ``dialect`` columns (a test enforces it);
nothing here feeds back into production, fertility, health, strength or any other
capability.

- Perception (``pair_distances``, ``core_distances``): the perceived appearance and
  dialect distance between two settlements, tanh(euclidean distance / reference), 0..1.
  Every other system sees the markers only through these two functions, and only two kinds
  of reader use them: the polity brains (inputs and the per-neighbour target head, whose
  marker weights start at zero, ``brains``) and the marker-effect culture parameters of
  each polity (zero in the founders, ``agents/polity.py``). With ``blind_perception`` every
  perceived distance is zero (the control condition).

- Year 0 (``draw_founders``): each founding people gets its own mean per appearance trait
  (uniform in [founder_mean_min, founder_mean_max], redrawn until every two peoples are at
  least ``founder_min_distance`` apart), a small variance, and an independent dialect
  vector. The teachers get their own appearance draw (fixed by config if given); they
  leave no settlements, so it is only kept for the memory records of later rounds.
- Inheritance (``inherit``): a daughter settlement copies its parent's markers.
- Mixing: when a band joins another, or settlers from the conqueror move into a
  conquered settlement (``blend``), the host's markers move toward the newcomers',
  weighted by the numbers of people. Each year, settlements of the same polity
  in contact (``polities.contact_radius``) exchange a share ``mixing_rate`` x the polity's
  intermarriage rate of their people with the population-weighted mean of those contacts
  (appearance means and variances mix as a population mixture). Dialects converge between
  all settlements in contact, of any polity, at ``dialect_contact_rate``. A conquered
  settlement that is still an unassimilated subject (``subject_of``, ``systems/war.py``)
  also moves its dialect toward its ruler's capital at ``dialect_assimilation_rate`` x the
  ruler's openness; its appearance blends through the settlers and then through the
  ruler's intermarriage like any member's.
- Drift, once per generation (``generation_years``): appearance means wander with sd
  ``drift_sigma`` x sqrt(reference / population) (capped), faster in small populations;
  dialects wander with sd ``dialect_drift_sigma``.
- Optional environmental selection (off by default): one trait is pulled toward a
  latitude-dependent optimum.

A union (merger, `politics.merge`) mixes its people through the absorbing polity's intermarriage.
"""

from __future__ import annotations

import numpy as np

from deluge.config import AncestryConfig


def draw_founders(model, founder_sids: list[int]) -> None:
    """Markers of the founding bands (one per people) and of the teachers."""
    a: AncestryConfig = model.config.ancestry
    rng = model.streams["identity"]
    n, k = len(founder_sids), len(a.appearance_traits)
    for _ in range(a.founder_max_attempts):
        means = rng.uniform(a.founder_mean_min, a.founder_mean_max, size=(n, k))
        d = np.linalg.norm(means[:, None, :] - means[None, :, :], axis=2)
        if (d[np.triu_indices(n, 1)] >= a.founder_min_distance).all():
            break
    else:
        raise RuntimeError(f"no founder appearance draw met founder_min_distance in {a.founder_max_attempts} attempts")
    t = model.settlements
    for sid, mean in zip(founder_sids, means):
        t.appearance_mean[sid] = mean
        t.appearance_var[sid] = a.founder_variance
        t.dialect[sid] = rng.normal(0.0, a.dialect_founder_sigma, size=a.dialect_dims)
    if a.teachers_appearance:
        if len(a.teachers_appearance) != k:
            raise ValueError(f"ancestry.teachers_appearance needs {k} values")
        model.teachers_appearance = np.array(a.teachers_appearance, dtype=float)
    else:
        model.teachers_appearance = rng.uniform(a.founder_mean_min, a.founder_mean_max, size=k)


def pair_distances(model, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perceived (appearance, dialect) distance between settlements ``a`` and ``b`` (arrays
    of ids, paired), each 0..1; zeros with ``blind_perception``."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    cfg: AncestryConfig = model.config.ancestry
    if cfg.blind_perception:
        return np.zeros(len(a)), np.zeros(len(a))
    t = model.settlements
    app = np.linalg.norm(t.appearance_mean[a] - t.appearance_mean[b], axis=1)
    dia = np.linalg.norm(t.dialect[a] - t.dialect[b], axis=1)
    return np.tanh(app / cfg.perceived_appearance_reference), np.tanh(dia / cfg.perceived_dialect_reference)


def core_distances(model, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Perceived (appearance, dialect) distance of each settlement from its polity's capital."""
    return pair_distances(model, ids, model.polities.capital[model.settlements.polity[ids]])


def actor_signatures(model, actors, cache: dict | None = None) -> tuple:
    """Appearance and dialect signature of each event actor now, for the memory records
    (spec "In memory"): a settlement's appearance means then its dialect, a polity's those of
    its capital, the teachers' appearance (they have no dialect: NaN); None for entities
    that are not people and for a lineage. Stored only; nothing reads them back. ``cache``:
    signatures already taken at this moment, by entity id (filled in)."""
    from deluge.data.events import decode

    t = model.settlements
    n_dia = t.dialect.shape[1]
    out = []
    for a in actors:
        if cache is not None and a in cache:
            out.append(cache[a])
            continue
        kind, local = decode(a)
        if kind == "polity":
            kind, local = "settlement", int(model.polities.capital[local])
        if kind == "settlement":
            sig = tuple(np.concatenate((t.appearance_mean[local], t.dialect[local])).round(4).tolist())
        elif kind == "teachers":
            sig = tuple(np.concatenate((model.teachers_appearance, np.full(n_dia, np.nan))).round(4).tolist())
        else:
            sig = None
        if cache is not None:
            cache[a] = sig
        out.append(sig)
    return tuple(out)


def inherit(model, child: int, parent: int) -> None:
    t = model.settlements
    t.appearance_mean[child] = t.appearance_mean[parent]
    t.appearance_var[child] = t.appearance_var[parent]
    t.dialect[child] = t.dialect[parent]


def _mix(mean, var, other_mean, other_var, w):
    """Population mixture: share ``w`` from the other population (``w`` broadcastable)."""
    new = (1.0 - w) * mean + w * other_mean
    new_var = (1.0 - w) * (var + (mean - new) ** 2) + w * (other_var + (other_mean - new) ** 2)
    return new, new_var


def blend(model, host: int, guest: int, people: float | None = None) -> None:
    """Band ``guest`` joins band ``host``, or ``people`` of it settle there (conquest
    settlers); call before the people are added to the host."""
    t = model.settlements
    ph = float(t.population[host])
    pg = float(t.population[guest]) if people is None else float(people)
    w = pg / max(ph + pg, 1e-12)
    t.appearance_mean[host], t.appearance_var[host] = _mix(t.appearance_mean[host], t.appearance_var[host],
                                                          t.appearance_mean[guest], t.appearance_var[guest], w)
    t.dialect[host] = (1.0 - w) * t.dialect[host] + w * t.dialect[guest]


def _contact_weights(model, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each settlement's contacts (``polities.contact_radius``): (ids with -1 read as 0, people
    in every contact, people in the contacts of its own polity), each (S, k)."""
    t = model.settlements
    occ = np.append(model.occupant, -1)[model.contact_cells[t.cell[ids]]]  # (S, k) settlement ids or -1
    o = np.maximum(occ, 0)
    w_all = np.where(occ >= 0, t.population[o], 0.0)
    w_own = np.where(t.polity[o] == t.polity[ids][:, None], w_all, 0.0)
    return o, w_all, w_own


def _weighted(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(has contacts, weights normalised per row)."""
    total = w.sum(axis=1)
    return total > 0, w / np.maximum(total, 1e-12)[:, None]


def run(model) -> None:
    a: AncestryConfig = model.config.ancestry
    t = model.settlements
    ids = t.alive_ids()
    if len(ids) == 0:
        return

    # Mixing inside polities (appearance), contact convergence for everyone (dialect): the
    # population-weighted means of the contacts.
    o, w_all, w_own = _contact_weights(model, ids)
    has, wn = _weighted(w_own)
    mean = np.einsum("sk,skj->sj", wn, t.appearance_mean[o])
    var = np.einsum("sk,skj->sj", wn, t.appearance_var[o])
    w = (a.mixing_rate * model.polities.intermarriage[t.polity[ids]] * has)[:, None]
    t.appearance_mean[ids], t.appearance_var[ids] = _mix(t.appearance_mean[ids], t.appearance_var[ids], mean, var, w)
    has, wn = _weighted(w_all)
    dialect = np.einsum("sk,skj->sj", wn, t.dialect[o])
    wd = (a.dialect_contact_rate * has)[:, None]
    t.dialect[ids] = (1.0 - wd) * t.dialect[ids] + wd * dialect
    # Assimilation: subjects' dialect moves toward their ruler's capital.
    subjects = ids[(t.subject_of[ids] >= 0) & (t.subject_of[ids] == t.polity[ids])]
    if len(subjects):
        ws = (a.dialect_assimilation_rate * model.openness_of(subjects))[:, None]
        capital_dialect = t.dialect[model.polities.capital[t.polity[subjects]]]
        t.dialect[subjects] += ws * (capital_dialect - t.dialect[subjects])

    if model.year == 0 or model.year % a.generation_years:
        return
    rng = model.streams["identity"]
    pop = np.maximum(t.population[ids], 1.0)
    sd = np.minimum(a.drift_sigma * np.sqrt(a.drift_reference_population / pop), a.drift_sigma_max)
    drift = rng.normal(0.0, 1.0, size=(len(ids), len(a.appearance_traits))) * sd[:, None]
    t.appearance_mean[ids] = np.clip(t.appearance_mean[ids] + drift, 0.0, 1.0)
    t.dialect[ids] = t.dialect[ids] + rng.normal(0.0, a.dialect_drift_sigma, size=(len(ids), a.dialect_dims))
    if a.selection_enabled:
        j = a.appearance_traits.index(a.selection_trait)
        lat = np.abs(model.geom.latitude(model.config.world.lat_south, model.config.world.lat_north)[t.cell[ids]])
        optimum = a.selection_optimum_equator + (a.selection_optimum_pole - a.selection_optimum_equator) * lat / 90.0
        t.appearance_mean[ids, j] += a.selection_rate * (optimum - t.appearance_mean[ids, j])


# --- measurement (analysis and tests only) ------------------------------------------------------


def lineage_means(model) -> tuple[np.ndarray, np.ndarray]:
    """Population-weighted (appearance mean, dialect) per founding lineage, NaN for extinct ones."""
    t = model.settlements
    ids = t.alive_ids()
    n = len(model.lineages.founders)
    w = np.bincount(t.lineage[ids], weights=t.population[ids], minlength=n)
    app = np.full((n, t.appearance_mean.shape[1]), np.nan)
    dia = np.full((n, t.dialect.shape[1]), np.nan)
    for g in np.flatnonzero(w > 0):
        sel = ids[t.lineage[ids] == g]
        p = t.population[sel] / w[g]
        app[g] = p @ t.appearance_mean[sel]
        dia[g] = p @ t.dialect[sel]
    return app, dia


def contact_distances(model) -> dict[str, float]:
    """Mean appearance and dialect distance between settlements in contact, split by
    same and different founding lineage (NaN where there are no such pairs)."""
    t = model.settlements
    ids = t.alive_ids()
    occ = np.append(model.occupant, -1)[model.contact_cells[t.cell[ids]]]
    i, k = np.nonzero(occ >= 0)
    a, b = ids[i], occ[i, k]
    app = np.linalg.norm(t.appearance_mean[a] - t.appearance_mean[b], axis=1)
    dia = np.linalg.norm(t.dialect[a] - t.dialect[b], axis=1)
    same = t.lineage[a] == t.lineage[b]
    out = {}
    for name, sel in (("same_lineage", same), ("other_lineage", ~same)):
        out[f"appearance_{name}"] = float(app[sel].mean()) if sel.any() else float("nan")
        out[f"dialect_{name}"] = float(dia[sel].mean()) if sel.any() else float("nan")
        out[f"pairs_{name}"] = int(sel.sum())
    return out


def spread(model) -> dict[str, float]:
    """Population-weighted sd of appearance means and dialects across all settlements, and
    the mean within-settlement appearance variance."""
    t = model.settlements
    ids = t.alive_ids()
    p = t.population[ids] / max(float(t.population[ids].sum()), 1e-12)
    out = {}
    for name, x in (("appearance", t.appearance_mean[ids]), ("dialect", t.dialect[ids])):
        mu = p @ x
        out[f"{name}_sd"] = float(np.sqrt(p @ ((x - mu) ** 2).sum(axis=1)))
    out["appearance_within_var"] = float(p @ t.appearance_var[ids].mean(axis=1))
    return out
