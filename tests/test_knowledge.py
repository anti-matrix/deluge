"""Round 4: knowledge graph, invention, diffusion, loss, the teachers, artifacts, the event
log and the chronicle (spec sections 7, 11 and 13)."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from deluge.agents.teachers import ARTIFACT_KINDS, Teachers
from conftest import small_config
from deluge.data import chronicle
from deluge.data.events import ACTIONS, ENTITY_KINDS, decode, entity
from deluge.model import DelugeModel
from deluge.systems.knowledge import KnowledgeGraph


@pytest.fixture(scope="module")
def cfg():
    return small_config()


@pytest.fixture(scope="module")
def early(cfg):
    """Small test world, its history seed, through the teachers' stay and 250 years after."""
    m = DelugeModel(config=cfg, years=300)
    m.run()
    return m


# --- graph -------------------------------------------------------------------------------------


def test_graph_compiles_in_prerequisite_order(cfg):
    g = KnowledgeGraph.from_config(cfg.knowledge)
    assert len(g.names) == 28
    pos = {t: i for i, t in enumerate(g.order)}
    for t, p in zip(*np.nonzero(g.prereq)):
        assert pos[p] < pos[t]
    assert g.names[g.unlocks.tolist().index(2)] == "agriculture"
    assert set(cfg.teachers.knowledge) <= set(g.names)


def test_graph_rejects_cycles_and_unknown_prerequisites(cfg):
    techs = dict(cfg.knowledge.techniques)
    a = techs["agriculture"]
    cyc = {**techs, "agriculture": dataclasses.replace(a, prerequisites=("storage",))}
    with pytest.raises(ValueError):
        KnowledgeGraph.from_config(dataclasses.replace(cfg.knowledge, techniques=cyc))
    bad = {**techs, "agriculture": dataclasses.replace(a, prerequisites=("bronze_age",))}
    with pytest.raises(KeyError):
        KnowledgeGraph.from_config(dataclasses.replace(cfg.knowledge, techniques=bad))


# --- teachers ----------------------------------------------------------------------------------


def test_teachers_visit_every_group_nearest_first_then_leave(early, cfg):
    ts = early.teachers_state
    tc = cfg.teachers
    assert [v["lineage"] for v in ts.visits] == ts.visited and sorted(ts.visited) == [0, 1, 2, 3, 4]
    for v in ts.visits:
        assert v["end"] - v["start"] in (tc.years_per_group - 1, tc.years_per_group)
    # Nearest-first: each next group is the closest unvisited one at that moment (by the
    # founding bands, which stay put in these early years).
    home = {f.lineage: f.cell for f in early.lineages.founders}
    for i in range(1, len(ts.visits) - 1):
        here = ts.visits[i]["cell"]
        rest = ts.visited[i + 1:]
        d = {g: int(early.geom.hex_distance(home[g], here)) for g in rest}
        assert d[ts.visited[i + 1]] == min(d.values())
    assert ts.departed and not ts.present and ts.agent is None
    assert not [a for a in early.agents if isinstance(a, Teachers)]  # no agent left behind
    ev = pd.DataFrame([early.events.record(i) for i in range(len(early.events))])
    arrivals = ev[ev.action == "arrived"]
    assert len(arrivals) == 5 and arrivals.year.min() == tc.arrival_year
    dep = ev[ev.action == "departed"]
    assert len(dep) == 1 and dep.year.iloc[0] == tc.arrival_year + 5 * tc.years_per_group - 1
    # Flight witnessed at every arrival and at the departure.
    flight = entity("technology", tc.technologies.index("flight"))
    assert all(flight in a for a in pd.concat([arrivals, dep]).actors)
    assert (ev[ev.action == "taught"].year < dep.year.iloc[0] + 1).all()


def test_teachers_never_enter_demography(early):
    """The teachers are no settlement, no lineage member and carry no population."""
    t = early.settlements
    assert set(t.lineage[: t.n].tolist()) <= set(range(5))
    assert len(early.lineages.founders) == 5


@pytest.mark.slow
def test_teaching_subsets_differ_between_groups(cfg):
    subsets = set()
    for h in (1, 2, 3):
        m = DelugeModel(config=cfg, history_seed=h, years=60)
        m.run()
        subsets |= {tuple(sorted(v["taught"])) for v in m.teachers_state.visits}
    assert len(subsets) >= 3


def test_teachers_disabled(cfg):
    off = cfg.with_overrides({"teachers": {"enabled": False}})
    m = DelugeModel(config=off, years=60)
    m.run()
    assert m.teachers_state.visits == []
    used = {ACTIONS[a] for a in m.events.action}
    assert not used & {"arrived", "taught", "departed", "gave", "displayed"}


def test_attack_on_teachers_always_fails(cfg):
    m = DelugeModel(config=cfg, years=60)
    m.step()
    target = m.teachers_state.target
    n = len(m.events)
    assert m.teachers_state.attacked(m, target) is False
    new = [ACTIONS[a] for a in m.events.action[n:]]
    assert new == ["fought", "displayed", "lost"]
    weapons = entity("technology", cfg.teachers.technologies.index("weapons"))
    assert weapons in m.events.actors[n + 1]
    assert m.settlements.population[target] > 0  # no harm beyond the failed attack


def test_artifacts_handover_lifespan_and_breakage(cfg):
    sure = cfg.with_overrides({"teachers": {"artifact_chance": 1.0, "artifact_lifespan_min": 20,
                                            "artifact_lifespan_max": 40}})
    m = DelugeModel(config=sure, years=120)
    m.run()
    ts = m.teachers_state
    assert all(1 <= len(v["artifacts"]) <= 4 for v in ts.visits)
    assert set(ARTIFACT_KINDS[k] for k in ts.kind) <= set(sure.teachers.artifact_types)
    for a in range(len(ts.kind)):
        assert 20 <= ts.expires[a] - ts.given[a] <= 40
        assert not ts.active[a]  # all broke by year 120
    actions = [ACTIONS[a] for a in m.events.action]
    assert actions.count("gave") == 5
    assert actions.count("broke") == sum(1 for h in ts.holder if h >= 0)


def test_artifacts_pass_with_their_settlement(cfg):
    m = DelugeModel(config=cfg, years=2)
    ts = m.teachers_state
    ts.kind.append(0)
    ts.holder.append(0)
    ts.given.append(0)
    ts.expires.append(100)
    ts.active.append(True)
    assert ts.tools_multiplier(np.array([0, 1]))[0] == cfg.teachers.tools_production
    ts.settlement_ended(0, host=3)
    assert ts.holder[-1] == 3
    assert ts.tools_multiplier(np.array([0, 3])).tolist() == [1.0, cfg.teachers.tools_production]


# --- invention, diffusion, loss -------------------------------------------------------------------


def test_unused_technique_is_forgotten_and_small_groups_forget_faster(cfg):
    # Never used: no band reaches size_use_population (small-world bands can).
    off = cfg.with_overrides({"teachers": {"enabled": False}, "knowledge": {"size_use_population": 1e12}})
    m = DelugeModel(config=off, years=2000)
    g, t, k = m.knowledge, m.settlements, cfg.knowledge
    j = g.index("megalithic_construction")  # used only by large settlements
    t.knows[:5, j] = True
    t.strength[:5, j] = 1.0
    years_known = []
    while m.running and t.knows[: t.n, j].any():
        m.step()
        years_known.append(m.year)
    assert not t.knows[: t.n, j].any()
    # Unused, a large group would hold it ~1 / decay_unused years; small bands lose it sooner.
    assert years_known[-1] < 1.0 / k.decay_unused + 5
    # Never practised, so it disappears without a "forgot" event.
    forgot = [r for r in (m.events.record(i) for i in range(len(m.events))) if r["action"] == "forgot"]
    assert not [r for r in forgot if r["actors"][1] == entity("technique", j)]
    assert m.datacollector.get_model_vars_dataframe()["latent_expired"].sum() >= 5


def test_latent_technique_decays_slowly_and_silently(cfg):
    """Sailing held by an inland band is latent: it decays at decay_latent (or not at all
    next to a band that practises it), and neither its gain nor its loss is logged."""
    m = DelugeModel(config=cfg.with_overrides({"teachers": {"enabled": False}}), years=400)
    m.step()
    t, g, k = m.settlements, m.knowledge, cfg.knowledge
    j = g.index("sailing")
    inland = [sid for sid in t.alive_ids().tolist() if not m.is_coast[t.cell[sid]]]
    assert inland, "a founding band inland"
    sid = inland[0]
    t.knows[sid, j] = True
    t.strength[sid, j] = 1.0
    years = 300
    for _ in range(years):
        m.step()
    assert t.ended[sid] != -1 or t.knows[sid, j]  # still held (if the band still exists)
    if t.ended[sid] == -1 and not m.is_coast[t.cell[sid]]:
        assert t.strength[sid, j] >= 1.0 - years * k.decay_latent - 1e-9
        assert not t.practised[sid, j]
    sailing = entity("technique", j)
    own = entity("settlement", sid)
    logged = [r for r in (m.events.record(i) for i in range(len(m.events)))
              if r["action"] in ("learned", "forgot") and r["actors"][:2] == (own, sailing)]
    assert not logged


def test_practised_flag_follows_use(early):
    """A technique counts as practised once used where it is held; a farming band where
    agriculture is usable has practised it. (A band can keep a farming share on a cell where
    farming is latent when the growth comparison is a close call the brain tips, Round 7.)"""
    from deluge.systems.knowledge import usable_mask

    t, g = early.settlements, early.knowledge
    ids = t.alive_ids()
    usable = usable_mask(early, ids)[:, g.index("agriculture")]
    farming = ids[(t.shares[ids, 2] >= early.config.knowledge.use_share) & usable]
    assert len(farming)
    assert t.practised[farming, g.index("agriculture")].all()
    assert not (t.practised[: t.n] & ~t.knows[: t.n]).any()


def test_invention_happens_without_teachers(cfg):
    fast = cfg.with_overrides({"teachers": {"enabled": False}, "knowledge": {"invention_scale": 1e-2}})
    m = DelugeModel(config=fast, years=200)
    m.run()
    assert (m.first_invented_year >= 0).any()
    learned = [r for r in (m.events.record(i) for i in range(len(m.events))) if r["action"] == "learned"]
    assert any(len(r["actors"]) == 2 for r in learned)  # invented: no source settlement


# --- event log and chronicle ---------------------------------------------------------------------


def test_every_event_is_complete_and_the_chronicle_renders(early, tmp_path):
    log = early.events
    assert len(log) > 0
    n_cells = early.geom.n_cells
    for i in range(len(log)):
        r = log.record(i)
        assert 0 <= r["year"] < early.year
        assert r["action"] in ACTIONS
        assert len(r["actors"]) >= 1 and all(decode(a)[0] in ENTITY_KINDS for a in r["actors"])
        assert -1 <= r["place"] < n_cells
        assert np.isfinite(r["magnitude"])
    every = chronicle.lines(early, len(log))
    assert len(every) == len(log) and all(s.startswith("Year ") for s in every)
    years = [int(s.split(":")[0].split()[1]) for s in every]
    assert years == sorted(years, reverse=True)  # newest first
    only_c = chronicle.lines(early, 50, groups={2})
    assert all("group C" in s for s in only_c)
    only_taught = chronicle.lines(early, 50, actions={"taught"})
    assert only_taught and all(" taught " in s for s in only_taught)
    log.write(tmp_path / "events.parquet")
    df = pd.read_parquet(tmp_path / "events.parquet")
    assert list(df.columns) == ["year", "action", "actors", "place", "magnitude", "link", "people"]
    assert len(df) == len(log)


def test_chronicle_has_no_story_vocabulary():
    """Templates describe what the sim did, nothing more (spec section 11)."""
    text = " ".join(chronicle.TEMPLATES.values()).lower() + " ".join(chronicle.MAGNITUDE.values()).lower()
    for word in ("god", "flood", "hero", "myth", "legend", "sacred", "divine", "deluge", "founder"):
        assert word not in text, word


def test_event_threshold(cfg):
    from deluge.data.events import EventLog
    log = EventLog(cfg.events.min_magnitude)
    assert not log.log(0, "starved", [entity("settlement", 1)], 5, cfg.events.min_magnitude["starved"] - 1)
    assert log.log(0, "starved", [entity("settlement", 1)], 5, cfg.events.min_magnitude["starved"])
    assert len(log) == 1


def test_chronicle_view_keeps_every_event_and_pages(early):
    """The view covers the whole log; pages, year range, group, type and text filters
    select from it; the full text has one line per event, oldest first."""
    view = chronicle.Chronicle(early)
    n = len(early.events)
    everything = view.select()
    assert len(everything) == len(view) == n
    size = 7
    pages = [view.page(everything, p, size) for p in range(view.pages(everything, size))]
    assert sum(len(p) for p in pages) == n and all(len(p) == size for p in pages[:-1])
    assert pages[0][0] == view.line(n - 1)  # newest first
    years = np.asarray(early.events.year)
    sel = view.select(year_min=60, year_max=120)
    assert len(sel) == ((years >= 60) & (years <= 120)).sum()
    assert all(60 <= early.events.year[i] <= 120 for i in sel)
    hits = view.select(text="TAUGHT")
    assert len(hits) and all(" taught " in view.line(i) for i in hits)
    assert len(view.select(text="no such words anywhere")) == 0
    assert len(view.select(groups=set(), actions=None)) == 0
    text = view.text()
    rows = text.splitlines()
    assert len(rows) == n and rows[0] == view.line(0) and rows[-1] == view.line(n - 1)
    assert len(early.events) == n  # reading never drops events
