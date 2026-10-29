"""Round 14 step 5: kin markers and the appearance prior.

- Both per-neighbour heads read dialect similarity and shared descent; their weights start at
  zero in the founders' genomes, so a founder's brain behaves as before until selection moves them.
- ``memory.kin_prior``: appearance similarity adds to alliance scores and subtracts from
  war-target scores by its weight; it changes nothing else (no strength, food, fertility or
  technique), and appearance keeps no capability effect.
"""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from conftest import small_model
from deluge.brains.features import ALLIANCE_FEATURES, KIN_FEATURES, TARGET_FEATURES, shared_descent, target_features
from deluge.systems import alliances, identity, war
from deluge.systems.subsistence import mode_capacity

PACKAGE = Path(__file__).resolve().parent.parent / "deluge"


@pytest.fixture(scope="module")
def grown():
    m = small_model({"polities": {"secession_base": 0.5}}, years=250)
    m.run()
    return m


def test_kin_weights_start_at_zero(grown):
    m = grown
    lay, B = m.brain_layout, m.brains
    s = lay.slices()
    for head, names, wname in (("q", TARGET_FEATURES, "q0"), ("qa", ALLIANCE_FEATURES, "qa0")):
        for f in KIN_FEATURES:
            j = names.index(f)
            for pid in range(5):  # the founders' genomes as born
                g = B.birth_genomes(5)[pid]
                assert np.all(g[s[head][0]].reshape(s[head][1])[:, j] == 0.0)
                assert g[s[wname][0]][j] == 0.0
    # So at birth their output weights on the kin features are zero.
    from deluge.brains import network

    g = B.birth_genomes(5)
    _, tw, aw, _ = network.forward(lay, g, np.random.default_rng(0).normal(size=(5, lay.n_in)))
    assert np.all(tw[:, [TARGET_FEATURES.index(f) for f in KIN_FEATURES]] == 0.0)
    assert np.all(aw[:, [ALLIANCE_FEATURES.index(f) for f in KIN_FEATURES]] == 0.0)


def test_shared_descent_and_dialect_similarity(grown):
    m = grown
    P = m.polities
    child = next(p for p in range(P.n) if P.parent[p] >= 0)
    parent = int(P.parent[child])
    scale = m.config.brains.features.descent_steps_scale
    assert shared_descent(m, [child], [parent])[0] == pytest.approx(np.exp(-1 / scale))
    assert shared_descent(m, [child], [child])[0] == 1.0
    other = next(p for p in range(P.n) if P.lineage[p] != P.lineage[child])
    assert shared_descent(m, [child], [other])[0] == 0.0
    a, b = int(P.capital[child]), int(P.capital[other])
    app, dia = identity.pair_distances(m, [a], [b])
    s_app, s_dia = identity.pair_similarities(m, [a], [b])
    assert s_dia[0] == pytest.approx(1 - dia[0]) and s_app[0] == pytest.approx(1 - app[0])
    f = target_features(m, np.array([child]), np.array([other]), np.array([b]), np.zeros(1))
    assert f[0, TARGET_FEATURES.index("dialect_similarity")] == pytest.approx(1 - dia[0])


def _prior(m, enabled: bool, weight: float = 0.2):
    cfg = m.config
    mem = dataclasses.replace(cfg.memory, kin_prior=dataclasses.replace(cfg.memory.kin_prior, enabled=enabled,
                                                                        weight=weight))
    m.config = dataclasses.replace(cfg, memory=mem)


def test_the_prior_changes_alliance_and_target_scores_only(grown):
    m = grown
    P, t = m.polities, m.settlements
    pids = P.alive_ids()
    p, q = int(pids[0]), int(pids[1])
    x = int(P.capital[q])
    ids = t.alive_ids()
    saved = m.config
    try:
        out = {}
        for on in (False, True):
            _prior(m, on)
            F = war.forces(m)
            out[on] = {
                "strength": war.strengths(m, F, np.array([p]), np.array([q]), np.array([x]))[0][0],
                "polity_strength": war.polity_strength(m).copy(),
                "capacity": mode_capacity(m, t.knows[ids], t.cell[ids]).copy(),
                "fertility": m.world.fertility.copy(),
                "military": m.knowledge.multiplier(t.knows[ids], "military").copy(),
                "alliance": alliances.score(m, [p], [q])[0],
                "prior": identity.kin_prior(m, [P.capital[p]], [x])[0],
            }
        for k in ("strength", "polity_strength", "capacity", "fertility", "military"):
            np.testing.assert_array_equal(out[False][k], out[True][k])
        sim = identity.pair_similarities(m, [P.capital[p]], [x])[0][0]
        assert out[False]["prior"] == 0.0 and out[True]["prior"] == pytest.approx(0.2 * sim)
        logit = lambda s: np.log(s / (1 - s))  # noqa: E731
        assert logit(out[True]["alliance"]) == pytest.approx(logit(out[False]["alliance"]) + 0.2 * sim, rel=1e-6)
    finally:
        m.config = saved


def test_only_alliance_and_target_choice_read_the_prior():
    readers = set()
    for path in PACKAGE.rglob("*.py"):
        if re.search(r"\bkin_prior\b", path.read_text(encoding="utf-8")):
            readers.add(path.relative_to(PACKAGE).as_posix())
    assert readers == {"config.py", "systems/identity.py", "systems/alliances.py", "systems/war.py"}
    war_src = (PACKAGE / "systems" / "war.py").read_text(encoding="utf-8")
    (line,) = [ln for ln in war_src.splitlines() if "kin_prior(" in ln]
    assert line.strip() == "-identity.kin_prior(model, P.capital[a], x))"  # the target score's shift only


def test_with_blind_perception_the_prior_is_inert():
    m = small_model({"ancestry": {"blind_perception": True}}, years=20)
    m.run()
    P = m.polities
    a, b = P.capital[P.alive_ids()[:2]]
    assert identity.kin_prior(m, [a], [b])[0] == 0.0
