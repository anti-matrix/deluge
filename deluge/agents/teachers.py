"""The teachers' contact phase and the artifacts they leave (spec section 7).

The teachers arrive by air at ``teachers.arrival_year`` at the band of the first group
they visit (drawn from history_seed unless pinned), stay ``years_per_group`` years with
each group, nearest group next, then depart by air and are removed. While present they
are a Mesa CellAgent on the visited band's cell; they never enter demography, migration
or politics.

Each contact year, every technique of their knowledge the visited band lacks (and whose
prerequisites it holds) transfers with chance ``teach_rate`` x the openness of the band's
polity, so
groups come away with different subsets. What the groups witness is logged as raw
events: each arrival and the departure (by flight), each contact year (with the
techniques passed on), medicine when the visited band is starving, weapons only if
attacked (``attacked``; attacks always fail).

When they leave a group, with chance ``artifact_chance`` they hand the band 1 to 4
artifacts (tools, medicine, weapons), each lasting a drawn number of years. Artifacts
belong to the settlement, pass with it when it joins another band, and give their bonus
until they break. Nothing in the knowledge graph leads to them.
"""

from __future__ import annotations

import numpy as np
from mesa.discrete_space import CellAgent

from deluge.agents.settlement import ALIVE
from deluge.config import TeachersConfig

ARTIFACT_KINDS = ("tools", "medicine", "weapons")


class Teachers(CellAgent):
    """The teachers while present: a mobile agent on the visited band's cell."""

    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell


class TeachersState:
    def __init__(self, cfg: TeachersConfig, n_groups: int, technique_index, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n_groups = n_groups
        self.knowledge = [technique_index(n) for n in cfg.knowledge]
        unknown = set(cfg.artifact_types) - set(ARTIFACT_KINDS)
        if unknown:
            raise ValueError(f"teachers.artifact_types: unknown {sorted(unknown)}")
        self.agent: Teachers | None = None
        self.present = False
        self.departed = not cfg.enabled
        self.visited: list[int] = []  # lineages, in visit order
        self.visits: list[dict] = []  # lineage, settlement, start, end, techniques taught, artifacts
        self.group = -1
        self.target = -1  # settlement being taught
        self.years_here = 0
        # Artifacts: one row per artifact.
        self.kind: list[int] = []
        self.holder: list[int] = []
        self.given: list[int] = []
        self.expires: list[int] = []
        self.active: list[bool] = []

    # --- schedule ------------------------------------------------------------------------

    def _home(self, model, lineage: int) -> int:
        """The group's largest living band (-1 if the group is gone)."""
        t = model.settlements
        ids = t.alive_ids()
        ids = ids[t.lineage[ids] == lineage]
        return int(ids[np.argmax(t.population[ids])]) if len(ids) else -1

    def _next_group(self, model) -> int:
        left = [g for g in range(self.n_groups) if g not in self.visited and self._home(model, g) >= 0]
        if not left:
            return -1
        if not self.visited:
            g = self.cfg.first_group
            return g if g >= 0 else int(left[int(self.rng.integers(len(left)))])
        here = int(model.settlements.cell[self.target]) if self.target >= 0 else -1
        homes = [int(model.settlements.cell[self._home(model, g)]) for g in left]
        d = model.geom.hex_distance(np.array(homes), here)
        return int(left[int(np.argmin(d))])

    def _fly_to(self, model, lineage: int) -> None:
        sid = self._home(model, lineage)
        cell = int(model.settlements.cell[sid])
        self.group, self.target, self.years_here = lineage, sid, 0
        self.visited.append(lineage)
        self.visits.append({"lineage": lineage, "settlement": sid, "cell": cell, "start": model.year,
                            "end": None, "taught": [], "artifacts": []})
        xy = model.geom.xy(cell)
        if self.agent is None:
            self.agent = Teachers(model, model.grid[xy])
        else:
            self.agent.cell = model.grid[xy]
        model.log_event("arrived", [("teachers", 0), ("settlement", sid), ("technology", self._tech("flight"))],
                        cell, float(model.settlements.population[sid]))

    def _tech(self, name: str) -> int:
        return self.cfg.technologies.index(name)

    # --- yearly --------------------------------------------------------------------------

    def step(self, model) -> None:
        self._age_artifacts(model)
        cfg = self.cfg
        if self.departed or model.year < cfg.arrival_year:
            return
        if not self.present:
            first = self._next_group(model)
            if first < 0:
                self.departed = True
                return
            self.present = True
            self._fly_to(model, first)
        t = model.settlements
        if t.ended[self.target] != ALIVE:  # the band moved on or joined kin: follow the group
            sid = self._home(model, self.group)
            if sid < 0:
                self._leave_group(model)
                return
            self.target = sid
            self.agent.cell = model.grid[model.geom.xy(int(t.cell[sid]))]
        self._teach(model)
        self.years_here += 1
        if self.years_here >= cfg.years_per_group:
            self._leave_group(model)

    def _teach(self, model) -> None:
        g, t = model.knowledge, model.settlements
        sid = self.target
        openness = float(model.openness_of(np.array([sid]))[0])
        taught = []
        for j in sorted(self.knowledge, key=g.order.index):
            if t.knows[sid, j] or not (t.knows[sid] | ~g.prereq[j]).all():
                continue
            if self.rng.random() < self.cfg.teach_rate * openness:
                t.knows[sid, j] = True
                t.strength[sid, j] = 1.0
                taught.append(j)
                model.note_technique_gained(j, model.year, invented=False)
        cell = int(t.cell[sid])
        model.log_event("taught", [("teachers", 0), ("settlement", sid)] + [("technique", j) for j in taught],
                        cell, float(len(taught)))
        self.visits[-1]["taught"] += [g.names[j] for j in taught]
        if t.food_ratio[sid] < 1.0 and "medicine" in self.cfg.technologies:
            model.log_event("displayed", [("teachers", 0), ("settlement", sid), ("technology", self._tech("medicine"))],
                            cell, float(t.population[sid]))

    def _leave_group(self, model) -> None:
        cfg = self.cfg
        sid = self.target
        visit = self.visits[-1]
        visit["end"] = model.year
        if sid >= 0 and model.settlements.ended[sid] == ALIVE and self.rng.random() < cfg.artifact_chance:
            n = int(self.rng.integers(cfg.artifacts_min, cfg.artifacts_max + 1))
            given = []
            for _ in range(n):
                kind = ARTIFACT_KINDS.index(cfg.artifact_types[int(self.rng.integers(len(cfg.artifact_types)))])
                life = int(self.rng.integers(cfg.artifact_lifespan_min, cfg.artifact_lifespan_max + 1))
                self.kind.append(kind)
                self.holder.append(sid)
                self.given.append(model.year)
                self.expires.append(model.year + life)
                self.active.append(True)
                given.append(len(self.kind) - 1)
                visit["artifacts"].append(ARTIFACT_KINDS[kind])
            model.log_event("gave", [("teachers", 0), ("settlement", sid)] + [("artifact", a) for a in given],
                            int(model.settlements.cell[sid]), float(n))
        nxt = self._next_group(model)
        if nxt >= 0:
            self._fly_to(model, nxt)
            return
        cell = int(model.settlements.cell[sid]) if sid >= 0 else -1
        model.log_event("departed", [("teachers", 0), ("settlement", sid), ("technology", self._tech("flight"))],
                        cell, float(len(self.visited)))
        self.agent.remove()
        self.agent = None
        self.present = False
        self.departed = True
        self.target = -1

    # --- artifacts --------------------------------------------------------------------------

    def _age_artifacts(self, model) -> None:
        for a in range(len(self.kind)):
            if self.active[a] and model.year >= self.expires[a]:
                self.active[a] = False
                h = self.holder[a]
                if h >= 0:
                    model.log_event("broke", [("artifact", a), ("settlement", h)],
                                    int(model.settlements.cell[h]), 1.0)

    def settlement_ended(self, sid: int, host: int) -> None:
        """Artifacts pass with their settlement into the band it joins (host >= 0), or are
        left behind with no holder."""
        for a in range(len(self.kind)):
            if self.holder[a] == sid:
                self.holder[a] = host

    def _held(self, kind: str) -> dict[int, list[int]]:
        k = ARTIFACT_KINDS.index(kind)
        out: dict[int, list[int]] = {}
        for a in range(len(self.kind)):
            if self.active[a] and self.kind[a] == k and self.holder[a] >= 0:
                out.setdefault(self.holder[a], []).append(a)
        return out

    def tools_multiplier(self, ids: np.ndarray) -> np.ndarray:
        out = np.ones(len(ids))
        held = self._held("tools") if self.kind else {}
        if held:
            out[np.isin(ids, list(held))] = self.cfg.tools_production
        return out

    def medicine_holders(self, ids: np.ndarray) -> np.ndarray:
        held = self._held("medicine") if self.kind else {}
        return np.isin(ids, list(held)) if held else np.zeros(len(ids), dtype=bool)

    def weapons_holders(self, ids: np.ndarray) -> np.ndarray:
        held = self._held("weapons") if self.kind else {}
        return np.isin(ids, list(held)) if held else np.zeros(len(ids), dtype=bool)

    def medicine_artifacts(self, sid: int) -> list[int]:
        return self._held("medicine").get(sid, [])

    def attacked(self, model, attacker: int) -> bool:
        """A band attacks the teachers: the attack fails, and the weapons are witnessed.
        Called by the war system (Round 6); returns whether the attack succeeded."""
        if not self.present:
            return False
        cell = int(model.settlements.cell[attacker])
        model.log_event("fought", [("settlement", attacker), ("teachers", 0)], cell, 1.0)
        model.log_event("displayed", [("teachers", 0), ("settlement", attacker), ("technology", self._tech("weapons"))],
                        cell, 1.0)
        model.log_event("lost", [("settlement", attacker), ("teachers", 0)], cell, 1.0)
        return False
