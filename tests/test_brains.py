"""Round 7: polity brains and neuroevolution (spec section 10, "Ancestry, appearance and
identity").

Scenarios run on the small test world, mostly with settlements held in place
(``test_polities.held``). ``scripted`` gives a polity a brain whose outputs and target
weights are constants, so a test can set what the brain wants.
"""

import numpy as np
import pytest

from conftest import small_model
from deluge.agents.polity import MARKER_EFFECTS, marker_index
from deluge.brains import evolution, features, network
from deluge.brains.network import OUT, OUTPUTS
from deluge.systems import cohesion, decide, demography, politics, subsistence, war
from test_polities import HELD, free_land, held, interior_cell, lone_polity, place

SATURATE = 30.0  # logit that puts a sigmoid output at 0 or 1 to double precision


def scripted(m, pid, target=None, **outputs):
    """Give polity ``pid`` a brain with constant outputs (0..1, default 0.5) and constant
    target weights (``target``: feature -> weight, default 0)."""
    lay = m.brain_layout
    g = np.zeros(lay.size, dtype=np.float32)
    s = lay.slices()
    b2 = np.zeros(len(OUTPUTS))
    for name, v in outputs.items():
        b2[OUT[name]] = SATURATE if v >= 1 else -SATURATE if v <= 0 else np.log(v / (1 - v))
    q0 = np.zeros(len(lay.target))
    for name, v in (target or {}).items():
        q0[lay.target.index(name)] = v
    g[s["b2"][0]] = b2
    g[s["q0"][0]] = q0
    m.brains.genome[pid] = g
    out, tw, _, _ = network.forward(lay, g[None, :], np.zeros((1, lay.n_in)))
    m.brains.output[pid], m.brains.target_w[pid] = out[0], tw[0]
    return pid


# --- the network ----------------------------------------------------------------------------------


def test_batched_forward_matches_one_by_one():
    m = small_model(years=2)
    lay = m.brain_layout
    rng = np.random.default_rng(1)
    g = rng.normal(0.0, 0.5, (7, lay.size)).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, (7, lay.n_in))
    out, tw, _, h = network.forward(lay, g, x)
    for i in range(7):
        w = {k: v[0] for k, v in lay.unpack(g[i:i + 1].astype(np.float64)).items()}
        hi = np.tanh(x[i] @ w["w1"] + w["b1"])
        np.testing.assert_allclose(h[i], hi)
        np.testing.assert_allclose(out[i], 1.0 / (1.0 + np.exp(-(hi @ w["w2"] + w["b2"]))))
        np.testing.assert_allclose(tw[i], hi @ w["q"] + w["q0"])
    assert out.shape == (7, len(OUTPUTS)) and ((out > 0) & (out < 1)).all()


def test_layout_and_marker_mask():
    m = small_model(years=2)
    lay = m.brain_layout
    cfg = m.config.brains
    assert lay.n_in == len(features.FEATURES) + cfg.memory_inputs + cfg.recurrent_units
    s = lay.slices()
    w1 = np.arange(lay.size)[s["w1"][0]].reshape(s["w1"][1])
    mask = m.brain_marker_mask
    for name in features.MARKER_FEATURES:
        assert mask[w1[lay.inputs.index(name)]].all()
    assert not mask[w1[lay.inputs.index("cohesion")]].any()
    q = np.arange(lay.size)[s["q"][0]].reshape(s["q"][1])
    assert mask[q[:, lay.target.index("appearance_distance")]].all()
    assert not mask[q[:, lay.target.index("win_chance")]].any()
    qa = np.arange(lay.size)[s["qa"][0]].reshape(s["qa"][1])  # Round 14: the alliance head's marker features
    assert mask[qa[:, lay.alliance.index("appearance_distance")]].all() and not mask[qa[:, lay.alliance.index("bias")]].any()
    assert mask.sum() == 4 * cfg.hidden + 2 * (cfg.hidden + 1) + 2 * (cfg.hidden + 1) + 4 * (cfg.hidden + 1)  # + kin


def test_founders_get_independent_random_brains_with_zero_marker_weights():
    a, b = small_model(years=2), small_model(years=2)
    ga, gb = a.brains.genome[:5], b.brains.genome[:5]
    np.testing.assert_array_equal(ga, gb)  # seeded
    assert len({row.tobytes() for row in ga}) == 5  # independent
    assert (ga[:, a.brain_marker_mask] == 0).all() and (ga[:, ~a.brain_marker_mask] != 0).all()
    other = small_model(years=2, history_seed=11)
    assert not np.array_equal(other.brains.genome[:5], ga)
    loose = small_model({"brains": {"marker_init_sigma": 1.0}}, years=2)
    assert (loose.brains.genome[:5][:, loose.brain_marker_mask] != 0).all()
    np.testing.assert_array_equal(loose.brains.genome[:5][:, ~a.brain_marker_mask], ga[:, ~a.brain_marker_mask])
    assert (a.polities.marker[:5] == 0).all()


def test_outputs_vary_across_polities_and_follow_the_inputs():
    m = small_model(years=60)
    m.run()
    out = m.brains.output[m.polities.alive_ids()]
    assert (out.std(axis=0) > 0.01).all()
    df = m.datacollector.get_model_vars_dataframe()
    assert df["brain_aggression"].nunique() > 1  # the same brains, changing inputs


# --- inheritance and mutation -------------------------------------------------------------------


def test_split_inherits_a_mutated_brain_and_the_parents_priorities():
    m = held()
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin, pop=5000.0)
    far = place(m, free_land(m, origin, 8)[0], pid)
    m.brains.output[pid] = 0.3
    child = politics.spin_off(m, pid, [far])
    g, gp = m.brains.genome[child], m.brains.genome[pid]
    d = g - gp
    sigma = m.config.brains.mutation_sigma
    assert 0.5 * sigma < np.median(np.abs(d)) / 0.6745 < 2.0 * sigma  # Gaussian noise of sd sigma
    np.testing.assert_array_equal(m.brains.output[child], m.brains.output[pid])
    assert m.polities.parent[child] == pid


def test_frozen_brains_are_exact_copies_of_the_founders():
    m = held({"secession_base": 1e3, "secession_max": 1.0}, years=60)
    m.config = m.config.with_overrides({"brains": {"frozen": True}})
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin, pop=5000.0)
    for d in (3, 6, 9):
        place(m, free_land(m, origin, d)[0], pid)
    for _ in range(3):
        m.step()
    P = m.polities
    kids = np.flatnonzero(P.parent[: P.n] >= 0)
    assert len(kids) > 0
    for k in kids.tolist():
        founder = m.lineages.polity_ancestry(P, k)[-1]
        np.testing.assert_array_equal(m.brains.genome[k], m.brains.genome[founder])


def test_fragmentation_gives_each_region_its_own_mutation():
    m = held({"fragment_rate": 1.0, "fragment_cohesion": 1.1, "fragment_pressure": -1.0, "region_spacing": 3})
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin, pop=5000.0)
    for d in (4, 8):
        place(m, free_land(m, origin, d)[0], pid)
    m.polities.n_settlements[pid] = 3
    politics.fragmentation(m)
    kids = np.flatnonzero(m.polities.parent[: m.polities.n] == pid)
    assert len(kids) >= 2
    genomes = m.brains.genome[kids]
    assert len({row.tobytes() for row in genomes}) == len(kids)
    assert all(np.abs(row - m.brains.genome[pid]).max() > 0 for row in genomes)


def test_conquest_kills_the_losers_brain():
    m = held()
    origin = interior_cell(m, 3)
    a, sa = lone_polity(m, origin, pop=50000.0)
    b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=500.0)
    kept = m.brains.genome[b].copy()
    assert war.battle(m, war.forces(m), a, b, sa, sb, campaign=True)
    assert m.polities.ended[b] == m.year
    decide.run(m)
    assert b not in m.polities.alive_ids().tolist()
    np.testing.assert_array_equal(m.brains.genome[b], kept)  # kept in the archive only
    assert m.settlements.polity[sb] == a  # its settlement now follows the conqueror's brain


def test_merger_crossover_keeps_the_birth_genome_in_the_archive():
    m = held({"merge_crossover_chance": 1.0})
    origin = interior_cell(m, 3)
    a, _ = lone_polity(m, origin, pop=5000.0)
    b, _ = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=500.0)
    ga, gb = m.brains.genome[a].copy(), m.brains.genome[b].copy()
    big = politics.merge(m, a, b)
    g = m.brains.genome[big]
    assert big == a and ((g == ga) | (g == gb)).all() and (g == gb).any() and (g == ga).any()
    np.testing.assert_array_equal(m.brains.birth_genomes(m.polities.n)[a], ga)
    frame = network.archive_frame(m)
    assert len(frame) == m.polities.n and len(frame.weights[0]) == m.brain_layout.size
    np.testing.assert_array_equal(frame.weights[a], ga)


def test_marker_parameters_are_inherited_with_mutation():
    m = held()
    origin = interior_cell(m, 9)
    pid, _ = lone_polity(m, origin, pop=5000.0)
    m.polities.marker[pid] = 0.5
    child = politics.spin_off(m, pid, [place(m, free_land(m, origin, 8)[0], pid)])
    d = m.polities.marker[child] - 0.5
    assert (d != 0).all() and np.abs(d).max() < 6 * m.config.polities.marker_mutation
    assert len(MARKER_EFFECTS) == m.polities.marker.shape[1]


def test_recurrent_state_switch():
    m = small_model({"brains": {"recurrent_units": 3}}, years=10)
    assert m.brain_layout.n_in == len(features.FEATURES) + m.config.brains.memory_inputs + 3
    m.run()
    pids = m.polities.alive_ids()
    assert (m.brains.state[pids] != 0).any()
    x = features.assemble(m, pids)
    j = m.brain_layout.inputs.index("recurrent_0")
    np.testing.assert_array_equal(x[:, j:j + 3], m.brains.state[pids])


# --- outputs wired into the systems ---------------------------------------------------------------


def test_aggression_and_campaign_outputs_drive_war():
    over = dict(HELD, war={"attack_rate": 1.0, "max_wars": 3, "battle_log_min_losses": 0.0})
    m = small_model(over, years=200)
    m.step()
    origin = interior_cell(m, 4)
    a, _ = lone_polity(m, origin, pop=3000.0)
    b, _ = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=3000.0)
    c, _ = lone_polity(m, free_land(m, origin, 2)[0], lineage=2, pop=3000.0)
    scripted(m, a, aggression=1.0, campaign=0.0, target={"bias": SATURATE})
    for pid in (b, c):
        scripted(m, pid, aggression=0.0, target={"bias": SATURATE})
    F = war.forces(m)
    attacks = war.choose(m, F)
    assert [x[0] for x in attacks] == [a] and not attacks[0][4]  # only a attacks, a raid
    scripted(m, a, aggression=1.0, campaign=1.0, target={"bias": SATURATE})
    assert all(x[4] for x in war.choose(m, F) if x[0] == a)


def test_target_head_can_prefer_by_appearance_and_blind_perception_hides_it():
    def pick(weight, blind=False):
        m = held()
        if blind:
            m.config = m.config.with_overrides({"ancestry": {"blind_perception": True}})
        m.config = m.config.with_overrides({"war": {"attack_rate": 1.0, "enabled": True}})
        origin = interior_cell(m, 4)
        a, sa = lone_polity(m, origin, pop=3000.0)
        b, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1, pop=3000.0)
        c, sc = lone_polity(m, free_land(m, origin, 2)[0], lineage=2, pop=3000.0)
        t = m.settlements
        t.appearance_mean[sa], t.appearance_mean[sb], t.appearance_mean[sc] = 0.5, 0.5, 0.0  # c looks different
        t.dialect[[sa, sb, sc]] = 0.0
        scripted(m, a, aggression=1.0, target={"bias": SATURATE / 2, "appearance_distance": weight})
        for pid in (b, c):
            scripted(m, pid, aggression=0.0)
        F = war.forces(m)
        sa_, sd_ = war.strengths(m, F, np.array([a, a]), np.array([b, c]), np.array([sb, sc]))
        p = war.win_chance(sa_, sd_, m.config.war.win_steepness)
        f = features.target_features(m, np.array([a, a]), np.array([b, c]), np.array([sb, sc]), p)
        s = network.target_scores(f, m.brains.target_w[[a, a]])
        return {"b": s[0], "c": s[1]}, f[:, features.TARGET_FEATURES.index("appearance_distance")]

    s, f = pick(+40.0)
    assert f[1] > f[0] == 0.0 and s["c"] > s["b"]
    s, _ = pick(-40.0)
    assert s["b"] > s["c"]
    s, f = pick(+40.0, blind=True)
    assert (f == 0).all() and s["b"] == s["c"]


def test_tribute_and_monument_outputs():
    taken = {}
    for v in (0.0, 1.0):
        m = held({"monument_cost": 50.0})
        pid, sid = lone_polity(m, interior_cell(m, 3), pop=5000.0)
        g = m.knowledge.index("megalithic_construction")
        m.settlements.knows[sid, g], m.settlements.strength[sid, g] = True, 1.0
        scripted(m, pid, tribute=v, monuments=v)
        m.settlements.store[sid] = 2000.0
        cohesion.run(m)
        taken[v] = (float(m.polities.tribute[pid]), float(m.polities.monument_fund[pid]))
    assert max(taken[0.0]) < 1e-6
    assert taken[1.0][0] == pytest.approx(m.config.polities.tribute_max_rate * 2000.0)
    assert taken[1.0][1] > 0


def test_expand_output_moves_the_split_threshold():
    founded = {}
    for v in (0.0, 1.0):
        m = held()
        m.config = m.config.with_overrides({"migration": {"split_threshold": 0.5}})
        pid, sid = lone_polity(m, interior_cell(m, 4), pop=1000.0)
        scripted(m, pid, expand=v, core_migration=0.0)
        t = m.settlements
        from deluge.systems.subsistence import fed_capacity, mode_capacity
        k = fed_capacity(t.shares[[sid]], mode_capacity(m, t.knows[[sid]], t.cell[[sid]]), m.config.demography)[0]
        t.population[sid] = 0.5 * k  # crowded at expand 1 (threshold 0.25), not at 0 (0.75)
        n0 = t.n
        demography.migrate(m)
        founded[v] = int((t.parent[n0:t.n] == sid).sum())
    assert founded == {0.0: 0, 1.0: 1}


def test_newcomers_output_decides_who_takes_bands_in():
    m = held()
    origin = interior_cell(m, 4)
    pid, band = lone_polity(m, origin)
    host_pid, host = lone_polity(m, free_land(m, origin, 1)[0])  # same lineage, another polity
    own = place(m, free_land(m, origin, 2)[0], pid)
    hosts = np.array([host, own])
    scripted(m, host_pid, newcomers=0.0)
    assert demography.welcoming(m, band, hosts).tolist() == [own]
    scripted(m, host_pid, newcomers=1.0)
    assert demography.welcoming(m, band, hosts).tolist() == [host, own]


def test_core_migration_output_moves_the_capital_band():
    def trial(v):
        m = held()
        m.config = m.config.with_overrides({"migration": {"core_move_rate": 1.0}})
        for cell in np.flatnonzero(~m.is_sea & (m.occupant < 0)).tolist():
            pid, cap = lone_polity(m, cell, pop=200.0)
            scripted(m, pid, core_migration=v)
            here = demography.cell_utility(m, cap, np.array([cell]), np.zeros(1, dtype=np.int64), 200.0)[0]
            _, util = demography.best_free_cell(m, cap, 200.0, 0.0)
            if util > here:
                demography.move_cores(m, demography.free_land_nearby(m))
                return int(m.settlements.cell[cap]) != cell
            m.end_settlement(cap, "dissolved")
            m.end_polity(pid, "extinct")
        raise AssertionError("no cell with a better one in reach")

    assert trial(1.0) and not trial(0.0)


def test_farming_output_biases_the_mode_choice():
    shares = {}
    for v in (0.0, 1.0):
        m = held()
        m.config = m.config.with_overrides({"demography": {"mode_brain_bias": 1.0}})
        pid, sid = lone_polity(m, interior_cell(m, 3), pop=500.0)
        g = m.knowledge.index("agriculture")
        m.settlements.knows[sid, g], m.settlements.strength[sid, g] = True, 1.0
        scripted(m, pid, farming=v, herding=0.0)
        for _ in range(10):
            subsistence.run(m)
        shares[v] = float(m.settlements.shares[sid, 2])
    assert shares[0.0] == 0.0 and shares[1.0] > 0.3


# --- perception and marker effects ----------------------------------------------------------------


def test_perception_inputs_and_the_blind_switch():
    def inputs(blind):
        m = held()
        if blind:
            m.config = m.config.with_overrides({"ancestry": {"blind_perception": True}})
        origin = interior_cell(m, 4)
        a, sa = lone_polity(m, origin)
        _, sb = lone_polity(m, free_land(m, origin, 1)[0], lineage=1)
        far = place(m, free_land(m, origin, 2)[0], a)
        t = m.settlements
        t.appearance_mean[[sa, far]], t.appearance_mean[sb] = 0.2, 0.9
        t.appearance_mean[far, 0] = 0.6
        t.dialect[sa], t.dialect[sb], t.dialect[far] = 0.0, 3.0, 1.0
        x = features.assemble(m, np.array([a]))[0]
        return {name: x[m.brain_layout.inputs.index(name)] for name in features.MARKER_FEATURES}

    seen, blind = inputs(False), inputs(True)
    assert all(v > 0 for v in seen.values()), seen
    assert all(v == 0 for v in blind.values())


def test_marker_effects_change_tribute_and_secession():
    m = held()
    origin = interior_cell(m, 6)
    pid, cap = lone_polity(m, origin, pop=5000.0)
    near, far_looking = place(m, free_land(m, origin, 5)[0], pid), place(m, free_land(m, origin, 5)[1], pid)
    t, P = m.settlements, m.polities
    t.appearance_mean[[cap, near]], t.appearance_mean[far_looking] = 0.5, 0.0
    t.dialect[[cap, near, far_looking]] = 0.0
    scripted(m, pid, tribute=0.5)
    P.marker[pid, marker_index("tribute_appearance")] = 2.0
    t.store[[near, far_looking]] = 1000.0
    cohesion.run(m)
    assert t.store[far_looking] < t.store[near]  # the different-looking member pays more
    P.marker[pid, marker_index("tribute_appearance")] = -2.0
    t.store[[near, far_looking]] = 1000.0
    cohesion.run(m)
    assert t.store[far_looking] > t.store[near]


# --- no capability effect -------------------------------------------------------------------------


def _scrambled_pair(overrides):
    """Two 500-year runs on the same seeds (secession raised so polities split); the second
    has every appearance and dialect scrambled at year 0."""
    overrides = {**overrides, "polities": {**overrides.get("polities", {}), "secession_base": 0.2}}
    a, b = small_model(overrides, years=500), small_model(overrides, years=500)
    t = b.settlements
    rng = np.random.default_rng(99)
    ids = t.alive_ids()
    t.appearance_mean[ids] = rng.random(t.appearance_mean[ids].shape)
    t.dialect[ids] = rng.normal(0.0, 5.0, t.dialect[ids].shape)
    a.run()
    b.run()
    return a, b


def _same_history(a, b):
    np.testing.assert_array_equal(a.datacollector.get_model_vars_dataframe().to_numpy(),
                                  b.datacollector.get_model_vars_dataframe().to_numpy())
    n = a.settlements.n
    assert b.settlements.n == n
    for col in ("population", "store", "polity", "grievance", "shares"):
        np.testing.assert_array_equal(getattr(a.settlements, col)[:n], getattr(b.settlements, col)[:n])
    assert len(a.events) == len(b.events)


def test_blind_perception_makes_markers_inert_even_with_marker_weights():
    """Random marker weights and marker parameters, but blind: scrambling every
    appearance and dialect changes nothing."""
    over = {"ancestry": {"blind_perception": True}, "brains": {"marker_init_sigma": 2.0},
            "polities": {"marker_mutation": 0.5}}
    a, b = _scrambled_pair(over)
    assert a.polities.n > 5 and np.abs(a.polities.marker[: a.polities.n]).max() > 0
    _same_history(a, b)


def test_markers_act_only_through_evolved_weights():
    """Seen markers, but marker weights and parameters held at zero (no mutation): scrambling
    every appearance and dialect changes nothing. With marker weights, it does."""
    over = {"brains": {"frozen": True}, "polities": {"marker_mutation": 0.0}}
    a, b = _scrambled_pair(over)
    assert a.polities.n > 5
    _same_history(a, b)
    over = {"brains": {"marker_init_sigma": 4.0}, "polities": {"secession_base": 0.001}}
    a, b = _scrambled_pair(over)
    assert not np.array_equal(a.datacollector.get_model_vars_dataframe().to_numpy(),
                              b.datacollector.get_model_vars_dataframe().to_numpy())
