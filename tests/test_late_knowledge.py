"""Round 4b: mineral deposits, resource access, the late knowledge graph, and their
chronicle lines (spec sections 5, 7 and 13).

Scenarios run on the small test world with the prerequisites seeded by hand: settlements
are held in place and fed (no famine, no migration, no splitting) so that only the
knowledge rules act. Effect sizes are not tested (uncalibrated in this round).
"""

import numpy as np
import pytest

from conftest import small_model
from deluge.data import chronicle
from deluge.data.events import ACTIONS, decode, entity
from deluge.systems.knowledge import KnowledgeGraph, deposit_id, resources_met, usable_mask
from deluge.world.worlddata import MINERALS

# Settlements stay where they are and keep their people: only knowledge changes.
HELD = {
    "teachers": {"enabled": False},
    "demography": {"famine_mortality": 0.0, "demographic_turnover": 0.0},
    "migration": {"famine_move_ratio": 0.0, "split_threshold": 1e9},
}
LATE = ("printing", "mechanical_power", "chemistry", "gunpowder", "steel", "coal_mining",
        "steam_power", "railways_steamships", "synthetic_fertilizers", "modern_medicine", "electricity",
        "telecommunication", "oil_combustion")


def scenario(overrides=None, years=200):
    """Small world, one step taken (capacity known); returns the model."""
    knowledge = {"invention_scale": 0.0, "late_techniques_enabled": True}
    knowledge.update((overrides or {}).pop("knowledge", {}))
    m = small_model({**HELD, "knowledge": knowledge, **(overrides or {})}, years=years)
    m.step()
    return m


def ancestors(g: KnowledgeGraph, name: str) -> set[str]:
    """The technique and everything it needs, transitively."""
    out, todo = set(), [name]
    while todo:
        n = todo.pop()
        if n not in out:
            out.add(n)
            todo += [g.names[p] for p in np.flatnonzero(g.prereq[g.index(n)])]
    return out


def give(m, sid: int, names, practised: bool = False) -> None:
    g, t = m.knowledge, m.settlements
    for n in names:
        j = g.index(n)
        t.knows[sid, j] = True
        t.strength[sid, j] = 1.0
        t.practised[sid, j] = practised


def free_neighbour(m, sid: int) -> int:
    cell = int(m.settlements.cell[sid])
    for n in m.geom.neighbor_lists[cell]:
        if not m.is_sea[n] and m.occupant[n] < 0:
            return int(n)
    raise AssertionError("no free land next to the settlement")


def set_people(m, sid: int, people: float) -> None:
    m.settlements.population[sid] = people
    m.settlements.peak_population[sid] = people


def events(m, action: str):
    return [m.events.record(i) for i in range(len(m.events)) if ACTIONS[m.events.action[i]] == action]


# --- graph and deposits ----------------------------------------------------------------------------


def test_late_graph_follows_the_spec(small_cfg):
    g = KnowledgeGraph.from_config(small_cfg.knowledge)
    assert set(LATE) <= set(g.names) and len(g.names) == 29

    def prereqs(n):
        return {g.names[p] for p in np.flatnonzero(g.prereq[g.index(n)])}

    assert prereqs("open_sea_voyaging") == {"sailing", "sky_calendar"}
    assert prereqs("ocean_navigation") == {"open_sea_voyaging", "writing", "iron"}
    assert not g.late[g.index("open_sea_voyaging")] and not g.late[g.index("ocean_navigation")]
    assert prereqs("coal_mining") == {"iron", "mechanical_power"}
    assert prereqs("steam_power") == {"coal_mining", "steel", "printing"}
    assert prereqs("synthetic_fertilizers") == {"chemistry", "steam_power"}
    assert prereqs("electricity") == {"steam_power", "copper_bronze"}
    assert prereqs("oil_combustion") == {"steam_power"}

    def needs(n):
        return {MINERALS[r] for r in np.flatnonzero(g.requires[g.index(n)])}

    assert needs("copper_bronze") == {"copper", "tin"}
    assert needs("iron") == {"iron_ore"}
    assert needs("coal_mining") == {"coal"}
    assert needs("oil_combustion") == {"oil"}
    assert sum(1 for n in g.names if needs(n)) == 4
    assert all(g.labels)


def test_unknown_resource_rejected(small_cfg):
    import dataclasses
    techs = dict(small_cfg.knowledge.techniques)
    techs["iron"] = dataclasses.replace(techs["iron"], resources=("gold",))
    with pytest.raises(KeyError):
        KnowledgeGraph.from_config(dataclasses.replace(small_cfg.knowledge, techniques=techs))


def test_deposits_generated_on_land(small_world, small_cfg):
    d = small_world.deposits()
    elev = small_world.elevation.ravel()
    sea0 = small_cfg.climate.sea_level.presets[small_cfg.climate.sea_level.preset][0][1]
    n_land = int((elev >= sea0).sum())
    for i, m in enumerate(MINERALS):
        tc = small_cfg.world.minerals.types[m]
        has = d[:, i] > 0
        assert has.sum() == round(tc.share * n_land), m
        assert (elev[has] >= sea0).all(), m
        assert d[has, i].min() >= tc.abundance_min and d[has, i].max() <= tc.abundance_max


def test_metals_near_mountain_belts(small_generated, small_geom):
    world, report = small_generated
    steps, _ = small_geom.distance_from(report.convergent)
    d = world.deposits()
    land = world.elevation.ravel() >= -60.0
    copper = d[:, MINERALS.index("copper")] > 0
    coal = d[:, MINERALS.index("coal")] > 0
    assert np.median(steps[copper]) < np.median(steps[land])
    assert np.median(steps[copper]) < np.median(steps[coal])


def test_resources_layer(small_cfg):
    m = small_model(years=2)
    code = m.resources_layer.data.ravel()
    has = m.deposits > 0
    assert ((code > 0) == has.any(axis=1)).all()
    single = has.sum(axis=1) == 1
    assert (code[single] == has[single].argmax(axis=1) + 1).all()
    assert (code[has.sum(axis=1) > 1] == len(MINERALS) + 1).all()


# --- access, invention, diffusion, loss --------------------------------------------------------------


def test_late_techniques_are_invented_with_access():
    """A large settlement on a deposit cell holding everything below steam power invents
    steam power, and then the techniques that follow from it."""
    m = scenario({"knowledge": {"invention_scale": 0.05}})
    g, t = m.knowledge, m.settlements
    sid = int(t.alive_ids()[0])
    cell = int(t.cell[sid])
    m.deposits[cell] = 1.0  # every mineral on its cell
    m.update_deposits()
    give(m, sid, ancestors(g, "steam_power") - {"steam_power"})
    set_people(m, sid, 20000.0)
    while m.running and not t.knows[sid, g.index("oil_combustion")]:
        m.step()
    m.step()  # deposits are worked from the year after a technique is gained
    for n in ("steam_power", "oil_combustion"):
        assert t.knows[sid, g.index(n)], n
        assert m.first_invented_year[g.index(n)] >= 1, n
    learned = [r for r in events(m, "learned") if r["actors"][0] == entity("settlement", sid)]
    assert entity("technique", g.index("steam_power")) in {r["actors"][1] for r in learned}
    # The settlement works the deposits its techniques need: one "used" event per mineral.
    used = [r for r in events(m, "used") if decode(r["actors"][0])[0] == "deposit"]
    worked = {decode(r["actors"][0])[1] for r in used}
    assert {deposit_id(cell, MINERALS.index(x)) for x in ("coal", "oil", "iron_ore")} <= worked
    assert all(r["actors"][1] == entity("settlement", sid) for r in used)


def test_resource_access_is_enforced():
    """Access is a deposit within access_radius_cells of the settlement's cell. A settlement
    holding every prerequisite of coal mining, with coal only beyond that radius, never
    invents it even at a high invention rate; one with coal in reach does. Coal mining held
    without access is latent."""
    m = scenario({"knowledge": {"invention_scale": 1.0, "diffusion_rate": 1.0}}, years=40)
    g, t = m.knowledge, m.settlements
    radius = m.config.world.minerals.access_radius_cells
    j = g.index("coal_mining")
    ids = t.alive_ids()
    near, far = int(ids[0]), int(ids[1])
    d = int(m.geom.hex_distance(int(t.cell[near]), int(t.cell[far])))
    assert d > 2 * radius
    m.deposits[:] = 0.0
    ring = [c for c in range(m.geom.n_cells)
            if int(m.geom.hex_distance(c, int(t.cell[near]))) == radius and not m.is_sea[c]]
    m.deposits[ring[0], MINERALS.index("coal")] = 1.0  # at the edge of the near one's reach
    for sid in (near, far):
        m.deposits[int(t.cell[sid]), MINERALS.index("iron_ore")] = 1.0
    m.update_deposits()
    for sid in (near, far):
        give(m, sid, ancestors(g, "coal_mining") - {"coal_mining"})
        set_people(m, sid, 20000.0)
    met = resources_met(m, np.array([near, far]))
    assert met[0, j] and not met[1, j]
    m.run()
    assert t.knows[near, j] and not t.knows[far, j]
    assert m.settlements.works[near, MINERALS.index("coal")] == ring[0]  # works the deposit in reach
    t.knows[far, j] = True
    t.strength[far, j] = 1.0
    assert not usable_mask(m, np.array([far]))[0, j]  # held without access: latent


def test_bronze_needs_copper_and_tin_each_in_reach():
    """Copper and tin on different cells, both within the radius: bronze is invented. Tin
    beyond the radius: it is not."""
    results = []
    for tin_steps in ("in_reach", "out_of_reach"):
        m = scenario({"knowledge": {"invention_scale": 1.0}}, years=30)
        g, t = m.knowledge, m.settlements
        radius = m.config.world.minerals.access_radius_cells
        sid = int(t.alive_ids()[0])
        home = int(t.cell[sid])
        dist = m.geom.hex_distance(np.arange(m.geom.n_cells), home)
        m.deposits[:] = 0.0
        m.deposits[int(np.flatnonzero(dist == 1)[0]), MINERALS.index("copper")] = 1.0
        tin_at = radius if tin_steps == "in_reach" else radius + 1
        m.deposits[int(np.flatnonzero(dist == tin_at)[0]), MINERALS.index("tin")] = 1.0
        m.update_deposits()
        assert (m.deposits[home] == 0).all()
        give(m, sid, ancestors(g, "copper_bronze") - {"copper_bronze"})
        set_people(m, sid, 20000.0)
        m.run()
        results.append(bool(t.knows[sid, g.index("copper_bronze")]))
    assert results == [True, False]

def test_late_technique_diffuses_to_a_neighbour():
    """Printing (no mineral) spreads from a holder to an adjacent settlement holding
    writing, with invention switched off."""
    m = scenario({"knowledge": {"diffusion_rate": 1.0}}, years=60)
    g, t = m.knowledge, m.settlements
    j = g.index("printing")
    src = int(t.alive_ids()[0])
    dst = m.found_settlement(free_neighbour(m, src), 8000.0, int(t.lineage[src]), src)
    give(m, src, ancestors(g, "printing"))
    give(m, dst, ancestors(g, "printing") - {"printing"})
    for sid in (src, dst):
        set_people(m, sid, 8000.0)
    m.run()
    assert t.knows[dst, j]
    assert m.first_invented_year[j] == -1  # not invented: adopted
    learned = [r for r in events(m, "learned")
               if r["actors"][:2] == (entity("settlement", dst), entity("technique", j))]
    assert learned and learned[0]["actors"][2] == entity("settlement", src)


def test_late_technique_is_lost_when_unused():
    """Electricity held by a small band that does not use it decays and is forgotten,
    with a forgot event (it was practised before)."""
    m = scenario(years=400)
    g, t = m.knowledge, m.settlements
    j = g.index("electricity")
    sid = int(t.alive_ids()[0])
    give(m, sid, ["electricity"], practised=True)
    set_people(m, sid, 50.0)
    while m.running and t.knows[sid, j]:
        m.step()
    assert not t.knows[sid, j]
    assert any(r["actors"][:2] == (entity("settlement", sid), entity("technique", j)) for r in events(m, "forgot"))


def test_depletion_switch():
    """Off by default: deposits stay. On: a worked deposit is used up and logged."""
    m = scenario(years=30)
    assert not m.config.world.minerals.depletion
    m_on = scenario({"world": {"minerals": {"depletion": True, "depletion_per_settlement_year": 0.3}}}, years=30)
    for model in (m, m_on):
        g, t = model.knowledge, model.settlements
        sid = int(t.alive_ids()[0])
        cell = int(t.cell[sid])
        model.deposits[cell] = 0.0
        model.deposits[cell, MINERALS.index("coal")] = 0.5
        model.deposits[cell, MINERALS.index("iron_ore")] = 0.5
        model.update_deposits()
        give(model, sid, ancestors(g, "coal_mining"))
        set_people(model, sid, 20000.0)
        model.run()
    cell = int(m.settlements.cell[int(m.settlements.alive_ids()[0])])
    assert m.deposits[cell, MINERALS.index("coal")] == 0.5
    assert not events(m, "depleted")
    gone = events(m_on, "depleted")
    assert {decode(r["actors"][0])[1] % len(MINERALS) for r in gone} == {MINERALS.index("coal"),
                                                                           MINERALS.index("iron_ore")}
    for r in gone:  # each one is used up; once its own are gone the band works others in reach
        cell, mineral = divmod(decode(r["actors"][0])[1], len(MINERALS))
        assert cell == r["place"] and m_on.deposits[cell, mineral] == 0.0
    assert m_on.resources_layer.data.ravel()[gone[0]["place"]] != len(MINERALS) + 1  # layer updated


def test_late_techniques_off_by_default():
    """With knowledge.late_techniques_enabled false (the default), a settlement holding every
    prerequisite of printing neither invents it nor adopts it from a neighbour; early
    techniques still spread."""
    m = small_model({**HELD, "knowledge": {"invention_scale": 1.0, "diffusion_rate": 1.0}}, years=40)
    assert not m.config.knowledge.late_techniques_enabled
    m.step()
    g, t = m.knowledge, m.settlements
    assert g.late.sum() == len(LATE) and {g.names[j] for j in np.flatnonzero(g.late)} == set(LATE)
    assert g.acquirable[g.index("ocean_navigation")]  # not late: open with late techniques off
    src = int(t.alive_ids()[0])
    dst = m.found_settlement(free_neighbour(m, src), 8000.0, int(t.lineage[src]), src)
    give(m, src, ancestors(g, "printing"))
    give(m, dst, ancestors(g, "printing") - {"printing"})
    for sid in (src, dst):
        set_people(m, sid, 8000.0)
    m.run()
    ids = t.alive_ids()
    late = np.flatnonzero(g.late)
    assert not t.knows[dst, g.index("printing")]
    assert (m.first_invented_year[late] == -1).all()
    assert t.knows[ids][:, late].sum() == 1  # only the hand-given printing
    assert (m.first_invented_year[~g.late] >= 0).any()


def test_prerequisites_are_hard():
    """Round 12: with invention and diffusion odds near certain, a settlement gains a
    technique in a year only when it held every prerequisite at the start of that year."""
    m = scenario({"knowledge": {"invention_scale": 1.0, "diffusion_rate": 1.0}}, years=40)
    g, t = m.knowledge, m.settlements
    gained = 0
    for _ in range(30):
        ids = t.alive_ids()
        before = t.knows[ids].copy()
        m.step()
        new = t.knows[ids] & ~before
        assert not (new & ~g.prerequisites_met(before)).any()
        gained += int(new.sum())
    assert gained > 0 and t.knows[t.alive_ids()][:, g.prereq.any(axis=1)].any()


# --- chronicle -----------------------------------------------------------------------------------------


def test_chronicle_shows_late_techniques_and_deposits():
    m = scenario({"knowledge": {"invention_scale": 0.05}}, years=150)
    g, t = m.knowledge, m.settlements
    sid = int(t.alive_ids()[0])
    m.deposits[int(t.cell[sid])] = 1.0
    m.update_deposits()
    give(m, sid, ancestors(g, "steam_power") - {"steam_power"})
    set_people(m, sid, 20000.0)
    m.run()
    view = chronicle.Chronicle(m)
    view.sync()
    text = view.markdown()
    steam = view.sp.technique(g.index("steam_power"))
    assert f"learned {steam}, the first anywhere to know {steam}" in text
    assert "deposit" in text and "first worked" in text  # deposits worked: in the generation summaries
    name = chronicle.Namer(m)
    used = [i for i in range(len(m.events)) if m.events.record(i)["action"] == "used"
            and decode(m.events.actors[i][0])[0] == "deposit"]
    assert any("the coal " in chronicle.line(m.events.record(i), name) for i in used)
