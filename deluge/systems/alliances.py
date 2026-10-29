"""Alliances (Round 14; spec sections 9 and 10: alliance seeking is a brain output, alliances are
resolved with war in phase 7).

Every alliance comes from the polity brains; nothing here says who should ally with whom.

- Score: a polity's alliance head (``brains/features.py`` ``ALLIANCE_FEATURES``, weights
  ``model.brains.alliance_w``) scores another polity, 0..1 (``score``).
- Proposing, each year: a living polity whose ``alliance_seeking`` output is above
  ``seek_threshold`` and that has fewer than ``max_alliances`` allies proposes to the polity in
  contact with it (``politics.contact_pairs``) its head scores best among those not yet its
  allies and with room for one more, if that score is above ``score_threshold``. The other
  accepts if its own score for the proposer is above ``score_threshold`` (``allied``). In id
  order, so one polity's new alliance counts against the next proposal.
- Keeping: every ``rescore_years`` both sides score each other again; the alliance ends when
  either score falls below ``score_threshold`` (``alliance_ended``, the side whose score fell
  first named first). An alliance with a polity that ended is dropped with it.
- In war (``systems/war.py``): allies are not each other's targets while the attacker's
  score for its ally is above the threshold; when it has fallen below and the ally is its
  best target, the attack is a betrayal (``betray``): the alliance ends at once, every
  settlement of the betrayed polity gains ``betrayal_grievance``, and it is logged
  (``betrayed``). An ally of either side may join a battle with chance = its own brain's
  campaign output (``helpers``), adding its strength at the usual logistics decay from its
  capital.
- Affinity: the years two polities have been allied raise their merger affinity
  (``affinity``: weight x years / (years + half), with the shared-memory affinity, capped
  at 1).

``model.alliances``: (low id, high id) -> year it began; ``model.alliance_history``: every
alliance that ended, (low, high, began, ended, how, by): how is "rescored", "betrayed" or
"ended" (a partner ended), by the polity whose score fell or that betrayed (-1 otherwise).
"""

from __future__ import annotations

import numpy as np

from deluge.brains.features import alliance_features
from deluge.brains.network import OUT, target_scores


def key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def allied(model, a: int, b: int) -> bool:
    return key(int(a), int(b)) in model.alliances


def allies_of(model, p: int) -> list[int]:
    """Polity ``p``'s allies, by id."""
    return sorted(b if a == p else a for a, b in model.alliances if p in (a, b))


def score(model, p, q) -> np.ndarray:
    """Polity ``p``'s alliance score for polity ``q`` (arrays, paired), 0..1."""
    p, q = np.asarray(p, dtype=np.int64), np.asarray(q, dtype=np.int64)
    if len(p) == 0:
        return np.zeros(0)
    return target_scores(alliance_features(model, p, q), model.brains.alliance_w[p])


def _end(model, a: int, b: int, how: str, by: int = -1) -> None:
    k = key(a, b)
    began = model.alliances.pop(k)
    model.alliance_history.append((k[0], k[1], began, model.year, how, by))
    model.stats["alliances_ended"] += 1


def run(model) -> None:
    """The year's alliances: drop those of ended polities, re-score each generation, propose and accept."""
    cfg = model.config.alliances
    if not cfg.enabled:
        return
    from deluge.systems import politics

    P = model.polities
    for a, b in [k for k in model.alliances if P.ended[k[0]] >= 0 or P.ended[k[1]] >= 0]:
        _end(model, a, b, "ended")
    if model.alliances and model.year % cfg.rescore_years == 0:
        pairs = sorted(model.alliances)
        a = np.array([p for p, _ in pairs], dtype=np.int64)
        b = np.array([q for _, q in pairs], dtype=np.int64)
        s_ab, s_ba = score(model, a, b), score(model, b, a)
        for (p, q), x, y in zip(pairs, s_ab.tolist(), s_ba.tolist()):
            if x < cfg.score_threshold or y < cfg.score_threshold:
                by = p if x < cfg.score_threshold and (y >= cfg.score_threshold or x <= y) else q
                began = model.alliances[(p, q)]
                _end(model, p, q, "rescored", by)
                model.log_event("alliance_ended", [("polity", by), ("polity", q if by == p else p)],
                                int(model.settlements.cell[P.capital[by]]), float(model.year - began))
    pids = P.alive_ids()
    if len(pids) < 2:
        return
    seeking = pids[model.brains.output[pids, OUT["alliance_seeking"]] > cfg.seek_threshold]
    if len(seeking) == 0:
        return
    pairs = politics.contact_pairs(model)
    if len(pairs) == 0:
        return
    near: dict[int, list[int]] = {}
    for x, y in pairs.tolist():
        near.setdefault(x, []).append(y)
        near.setdefault(y, []).append(x)
    def full(n: int) -> bool:  # max_alliances 0: no limit
        return cfg.max_alliances > 0 and n >= cfg.max_alliances

    count = {p: 0 for p in pids.tolist()}
    for x, y in model.alliances:
        count[x] = count.get(x, 0) + 1
        count[y] = count.get(y, 0) + 1
    for p in seeking.tolist():
        if full(count.get(p, 0)):
            continue
        cand = [q for q in near.get(p, ()) if not allied(model, p, q) and not full(count.get(q, 0))]
        if not cand:
            continue
        s = score(model, np.full(len(cand), p), np.array(cand))
        k = int(np.argmax(s))
        q = cand[k]
        if s[k] <= cfg.score_threshold or score(model, [q], [p])[0] <= cfg.score_threshold:
            continue
        model.alliances[key(p, q)] = model.year
        count[p] = count.get(p, 0) + 1
        count[q] = count.get(q, 0) + 1
        model.stats["alliances_formed"] += 1
        t = model.settlements
        ids = t.alive_ids()
        people = float(t.population[ids[(t.polity[ids] == p) | (t.polity[ids] == q)]].sum())
        model.log_event("allied", [("polity", p), ("polity", q)], int(t.cell[P.capital[p]]), people)


def betray(model, a: int, b: int) -> None:
    """Polity ``a`` attacks its ally ``b``: the alliance ends at once, ``b``'s settlements gain
    grievance, and it is logged."""
    cfg = model.config.alliances
    t, P = model.settlements, model.polities
    began = model.alliances[key(a, b)]
    _end(model, a, b, "betrayed", a)
    model.stats["betrayals"] += 1
    ids = t.alive_ids()
    t.grievance[ids[t.polity[ids] == b]] += cfg.betrayal_grievance
    model.log_event("betrayed", [("polity", a), ("polity", b)], int(t.cell[P.capital[b]]),
                    float(model.year - began))


def keeps_faith(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per pair: ``a`` and ``b`` are allies and ``a`` still scores ``b`` above the threshold (so
    ``b`` is no target of ``a``)."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    out = np.zeros(len(a), dtype=bool)
    if not model.alliances or len(a) == 0:
        return out
    rows = np.array([allied(model, x, y) for x, y in zip(a.tolist(), b.tolist())], dtype=bool)
    if rows.any():
        out[rows] = score(model, a[rows], b[rows]) > model.config.alliances.score_threshold
    return out


def helpers(model, side: int, enemy: int) -> list[int]:
    """Allies of ``side`` that join its battle against ``enemy``: each (not allied with the
    enemy, nor the enemy itself) with chance = its brain's campaign output."""
    if not model.alliances:
        return []
    rng = model.streams["war"]
    out = []
    for c in allies_of(model, side):
        if c == enemy or allied(model, c, enemy) or model.polities.ended[c] >= 0:
            continue
        if rng.random() < model.brains.output[c, OUT["campaign"]]:
            out.append(c)
    return out


def affinity(model, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Merger affinity from years allied: weight x years / (years + half), 0 for non-allies."""
    cfg = model.config.alliances
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    out = np.zeros(len(a))
    if not cfg.enabled or not model.alliances:
        return out
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        began = model.alliances.get(key(x, y))
        if began is not None:
            years = model.year - began
            out[i] = cfg.affinity_weight * years / (years + cfg.affinity_half_years)
    return out
