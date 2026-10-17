"""Phase 5, knowledge: teacher contact, invention, diffusion and loss (spec section 7).

Per settlement per year, for each technique it does not hold:

- invention: small base odds (``invention_scale`` x the technique's ``invention_rate``),
  scaled by population, by division of labor, and by the share of the technique's
  prerequisites the settlement holds (``** invention_prerequisite_exponent``);
- diffusion, only when every prerequisite is held: each adjacent settlement holding it
  gives a chance ``diffusion_rate`` x the receiver's openness.

A held technique is usable where the settlement is (sailing needs a coast; herding and
farming techniques need some capacity for their mode on the cell; everything else,
size-tied techniques included, is usable anywhere), or latent. Each held technique has a
strength:

- usable: it falls every year the technique goes unused (``decay_unused``) and, used or
  not, faster in small populations (``decay_small``); use restores it (``practice_gain``);
- latent: it falls only by the much slower ``decay_latent``, and not at all while an
  adjacent settlement practises the technique.

Techniques that need minerals (``resources`` in the graph) are invented, adopted and used
only with access to each of them: a deposit within ``world.minerals.access_radius_cells``
of the settlement's cell, or one in its polity's territory (spec section 5; trade comes
later). Held without access they are latent. Using one works a deposit in reach
(``work_deposits``). Late techniques (``late`` in the graph) are invented and adopted only
with ``knowledge.late_techniques_enabled``. Diffusion scales with the receiver's polity's
openness.

At zero the technique is gone. A settlement has ``practised`` a technique once it used it
while holding it. Events: ``forgot`` only when a practised technique is lost (latent and
never-used ones expire silently, counted in ``latent_expired``); ``learned`` only when the
technique gained is usable where the settlement is. Teacher contact runs first
(``agents/teachers.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deluge.agents.settlement import MODES
from deluge.config import KnowledgeConfig
from deluge.world.worlddata import MINERALS

USED_WITH = ("forage", "herd", "farm", "size", "coast", "always")


@dataclass(frozen=True)
class KnowledgeGraph:
    """The technique graph from config, compiled to arrays (column order = ``names``)."""

    names: tuple[str, ...]
    labels: tuple[str, ...]  # chronicle names
    prereq: np.ndarray  # (T, T) bool: prereq[t, p] = p is a prerequisite of t
    requires: np.ndarray  # (T, minerals) bool: the technique needs access to the mineral (MINERALS order)
    order: tuple[int, ...]  # topological order
    invention_rate: np.ndarray
    used_with: np.ndarray  # index into USED_WITH
    unlocks: np.ndarray  # mode index unlocked, -1 for none
    effect: dict  # effect name -> (T,) array; multipliers default 1, additive effects 0
    late: np.ndarray  # (T,) bool: part of the late graph
    military: np.ndarray  # (T,) bool: category military
    acquirable: np.ndarray  # (T,) bool: can be invented or diffused (late ones only with late_techniques_enabled)

    MULTIPLIERS = ("farm_yield", "herd_yield", "river_farm_yield", "production", "spoilage", "drought_loss",
                   "diffusion", "famine_mortality", "admin_cap", "military", "defense", "raid_range",
                   "memory_capacity")
    ADDITIVE = ("migration_radius", "production_min_population")
    HIGHEST = ("sea_steps",)  # the highest over the techniques held counts (sea tiers), default 0

    @classmethod
    def from_config(cls, cfg: KnowledgeConfig) -> KnowledgeGraph:
        names = tuple(cfg.techniques)
        index = {n: i for i, n in enumerate(names)}
        T = len(names)
        prereq = np.zeros((T, T), dtype=bool)
        requires = np.zeros((T, len(MINERALS)), dtype=bool)
        for n, tc in cfg.techniques.items():
            for p in tc.prerequisites:
                if p not in index:
                    raise KeyError(f"technique {n}: unknown prerequisite {p}")
                prereq[index[n], index[p]] = True
            for r in tc.resources:
                if r not in MINERALS:
                    raise KeyError(f"technique {n}: unknown resource {r}; one of {MINERALS}")
                requires[index[n], MINERALS.index(r)] = True
            if tc.used_with not in USED_WITH:
                raise ValueError(f"technique {n}: used_with must be one of {USED_WITH}")
            if tc.unlocks and tc.unlocks not in MODES:
                raise ValueError(f"technique {n}: unlocks must be one of {MODES} or empty")
        order, done = [], set()
        while len(order) < T:
            ready = [i for i in range(T) if i not in done and all(p in done for p in np.flatnonzero(prereq[i]))]
            if not ready:
                raise ValueError("knowledge graph has a prerequisite cycle")
            order += ready
            done.update(ready)
        effect = {}
        for name in cls.MULTIPLIERS:
            effect[name] = np.array([cfg.techniques[n].effects.get(name, 1.0) for n in names])
        for name in cls.ADDITIVE + cls.HIGHEST:
            effect[name] = np.array([cfg.techniques[n].effects.get(name, 0.0) for n in names])
        effect["store_cap_years"] = np.array([cfg.techniques[n].effects.get("store_cap_years", np.nan) for n in names])
        late = np.array([cfg.techniques[n].late for n in names], dtype=bool)
        return cls(
            late=late,
            military=np.array([cfg.techniques[n].category == "military" for n in names], dtype=bool),
            acquirable=~late | cfg.late_techniques_enabled,
            names=names,
            labels=tuple(cfg.techniques[n].label for n in names),
            prereq=prereq,
            requires=requires,
            order=tuple(order),
            invention_rate=np.array([cfg.techniques[n].invention_rate for n in names]),
            used_with=np.array([USED_WITH.index(cfg.techniques[n].used_with) for n in names]),
            unlocks=np.array([MODES.index(cfg.techniques[n].unlocks) if cfg.techniques[n].unlocks else -1
                              for n in names]),
            effect=effect,
        )

    def index(self, name: str) -> int:
        return self.names.index(name)

    def multiplier(self, knows: np.ndarray, name: str, mask: np.ndarray | None = None) -> np.ndarray:
        """Product of effect ``name`` over the techniques each row holds (and ``mask``)."""
        k = knows if mask is None else knows & mask
        return np.exp(k @ np.log(self.effect[name]))

    def total(self, knows: np.ndarray, name: str) -> np.ndarray:
        return knows @ self.effect[name]

    def highest(self, knows: np.ndarray, name: str) -> np.ndarray:
        """Highest value of effect ``name`` over the techniques each row holds (0 for none)."""
        return np.where(knows, self.effect[name], 0.0).max(axis=-1)

    def mode_available(self, knows: np.ndarray) -> np.ndarray:
        """(S, modes) bool: foraging always, other modes once a technique unlocks them."""
        out = np.zeros((knows.shape[0], len(MODES)), dtype=bool)
        out[:, 0] = True
        for t in np.flatnonzero(self.unlocks >= 0):
            out[:, self.unlocks[t]] |= knows[:, t]
        return out

    def prerequisites_held(self, knows: np.ndarray) -> np.ndarray:
        """(S, T) float: how many of the technique's prerequisites are held (exact counts)."""
        return knows.astype(float) @ self.prereq.T.astype(float)

    def prerequisites_met(self, knows: np.ndarray, held: np.ndarray | None = None) -> np.ndarray:
        """(S, T) bool: every prerequisite of the technique is held (``held``: the counts of
        ``prerequisites_held``, when already computed)."""
        held = self.prerequisites_held(knows) if held is None else held
        return held == self.prereq.sum(axis=1)[None, :]


def production_multiplier(model, ids: np.ndarray) -> np.ndarray:
    """Production bonus from techniques (each from its minimum population up) and from
    tools artifacts."""
    g = model.knowledge
    t = model.settlements
    pop = t.population[ids]
    eligible = pop[:, None] >= g.effect["production_min_population"][None, :]
    return g.multiplier(t.knows[ids], "production", eligible) * model.teachers_state.tools_multiplier(ids)


def _by_used_with(model, ids: np.ndarray, columns: np.ndarray) -> np.ndarray:
    """Spread a (S, len(USED_WITH)) table to (S, T) by each technique's used_with."""
    return columns[:, model.knowledge.used_with]


def deposit_sources(model, ids: np.ndarray) -> np.ndarray:
    """(S, minerals): the deposit cell each settlement works for each mineral, -1 for none.

    Spec section 5: access by sitting on a deposit cell, by polity control of one, or by
    trade with a settlement that has access. A deposit within
    ``world.minerals.access_radius_cells`` of the settlement's cell (``model.deposit_source``,
    the stand-in for local reach and trade until trade exists) comes first; otherwise a
    deposit in its polity's territory (``polities.territory_radius`` of any member
    settlement; ``model.polities.deposit``, updated each year by the politics phase)."""
    t = model.settlements
    own = model.deposit_source[t.cell[ids]]
    via_polity = model.polities.deposit[t.polity[ids]]
    return np.where(own >= 0, own, via_polity)


def resource_access(model, ids: np.ndarray) -> np.ndarray:
    """(S, minerals) bool: the settlement has access to a deposit of the mineral."""
    return deposit_sources(model, ids) >= 0

def resources_met(model, ids: np.ndarray) -> np.ndarray:
    """(S, T) bool: the settlement has access to every mineral the technique needs."""
    lacking = (~resource_access(model, ids)).astype(np.int64)
    return (lacking @ model.knowledge.requires.T.astype(np.int64)) == 0


def usable_mask(model, ids: np.ndarray) -> np.ndarray:
    """(S, T) bool: the technique can be used where the settlement is. Sailing needs a
    coast, herding and farming techniques need some capacity for their mode on the cell,
    techniques that need minerals need access to them; everything else (size-tied
    techniques included) is usable anywhere."""
    cells = model.settlements.cell[ids]
    ok = np.ones((len(ids), len(USED_WITH)), dtype=bool)
    ok[:, 1] = model.capacity["herd"][cells] > 0
    ok[:, 2] = model.capacity["farm"][cells] > 0
    ok[:, 4] = model.is_coast[cells]
    return _by_used_with(model, ids, ok) & resources_met(model, ids)


def used_mask(model, ids: np.ndarray) -> np.ndarray:
    """(S, T) bool: the technique is in use this year."""
    k, t = model.config.knowledge, model.settlements
    use = np.zeros((len(ids), len(USED_WITH)), dtype=bool)
    use[:, :3] = t.shares[ids] >= k.use_share
    use[:, 3] = t.population[ids] >= k.size_use_population
    use[:, 4] = model.is_coast[t.cell[ids]]
    use[:, 5] = True
    return _by_used_with(model, ids, use)


def neighbour_counts(model, ids: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """(S, T) number of adjacent settlements with each flag set (``flags`` is (S, T)
    over the same settlements)."""
    t = model.settlements
    per_cell = np.zeros((model.geom.n_cells + 1, flags.shape[1]), dtype=np.uint8)
    per_cell[t.cell[ids]] = flags
    # -1 reads the zero padding row; at most 6 neighbours, so uint8 sums are exact.
    return per_cell[model.geom.neighbors[t.cell[ids]]].sum(axis=1, dtype=np.uint8).astype(np.int64)


def work_deposits(model, ids: np.ndarray, in_use: np.ndarray) -> None:
    """Settlements using a technique that needs a mineral work the nearest deposit of it in reach.
    The first year a settlement works a deposit is a `used` event (deposit, settlement).
    With `world.minerals.depletion` on, each working settlement takes
    `depletion_per_settlement_year` of the deposit's abundance a year; an exhausted
    deposit is a `depleted` event and is gone."""
    t, g = model.settlements, model.knowledge
    working = (in_use.astype(np.int64) @ g.requires.astype(np.int64)) > 0  # (S, minerals)
    source = deposit_sources(model, ids)  # (S, minerals): the deposit cell each works
    working &= source >= 0
    if not working.any():
        return
    for i, m in zip(*np.nonzero(working & (t.works[ids] != source))):
        sid, cell = int(ids[i]), int(source[i, m])
        t.works[sid, m] = cell
        model.log_event("used", [("deposit", deposit_id(cell, int(m))), ("settlement", sid)], cell,
                        float(t.population[sid]))
    mc = model.config.world.minerals
    if mc.depletion:
        for i, m in zip(*np.nonzero(working)):
            cell = int(source[i, m])
            if model.deposits[cell, m] == 0.0:
                continue  # used up earlier this year by another settlement
            model.deposits[cell, m] = max(model.deposits[cell, m] - mc.depletion_per_settlement_year, 0.0)
            if model.deposits[cell, m] == 0.0:
                model.log_event("depleted", [("deposit", deposit_id(cell, int(m)))], cell, 0.0)
        model.update_deposits()


def deposit_id(cell: int, mineral: int) -> int:
    """Local id of the deposit of `mineral` (MINERALS index) on `cell`."""
    return cell * len(MINERALS) + mineral


def grant(model, sid: int, technique: int, source: int) -> bool:
    """Settlement ``sid`` adopts ``technique`` from settlement ``source`` outside the yearly
    diffusion (war and conquest, ``systems/war.py``), under the same conditions: every
    prerequisite held, the minerals it needs in reach, and acquirable. Logged as
    ``learned`` from the source when usable where ``sid`` is. Returns whether it was gained."""
    g, t = model.knowledge, model.settlements
    if t.knows[sid, technique] or not g.acquirable[technique]:
        return False
    ids = np.array([sid])
    if not (g.prerequisites_met(t.knows[ids])[0, technique] and resources_met(model, ids)[0, technique]):
        return False
    t.knows[sid, technique] = True
    t.strength[sid, technique] = 1.0
    model.note_technique_gained(technique, model.year, False)
    if usable_mask(model, ids)[0, technique]:
        model.log_event("learned", [("settlement", sid), ("technique", technique), ("settlement", source)],
                        int(t.cell[sid]), float(t.population[sid]))
    return True


def run(model) -> None:
    model.teachers_state.step(model)
    k, g, t = model.config.knowledge, model.knowledge, model.settlements
    rng = model.streams["knowledge"]
    ids = t.alive_ids()
    if len(ids) == 0:
        return
    year = model.year
    pop = t.population[ids]
    knows = t.knows[ids]
    usable = usable_mask(model, ids)
    used = used_mask(model, ids) & usable
    practised = t.practised[ids] | (knows & used)

    # Loss. Usable techniques decay while idle and, used or not, faster in small groups;
    # use restores them. Latent ones (held where they cannot be used) decay slowly, and
    # not at all while an adjacent settlement practises them.
    latent = knows & ~usable
    kept_alive = neighbour_counts(model, ids, knows & used) > 0
    small = np.clip(k.small_reference_population / np.maximum(pop, 1.0) - 1.0, 0.0, k.small_factor_cap)
    s = t.strength[ids]
    active_change = -k.decay_unused * (~used) - k.decay_small * small[:, None] + k.practice_gain * used
    s = s + np.where(latent, -k.decay_latent * ~kept_alive, active_change)
    s = np.where(knows, np.minimum(s, 1.0), 0.0)
    lost = knows & (s <= 0.0)
    t.strength[ids] = np.maximum(s, 0.0)
    if lost.any():
        knows = knows & ~lost
        crashed = pop < k.crash_share * t.peak_population[ids]
        for i, j in zip(*np.nonzero(lost & practised)):
            sid = int(ids[i])
            model.log_event("forgot", [("settlement", sid), ("technique", int(j))], int(t.cell[sid]), float(pop[i]))
            model.note_technique_lost(int(j), bool(crashed[i]))
        model.stats["latent_expired"] += int((lost & ~practised).sum())
        practised = practised & ~lost
        t.knows[ids] = knows
    t.practised[ids] = practised
    work_deposits(model, ids, knows & used)

    # Invention.
    held = g.prerequisites_held(knows)
    prereq_share = np.where(g.prereq.sum(axis=1) > 0, held / np.maximum(g.prereq.sum(axis=1), 1), 1.0)
    dol = knows[:, g.index("division_of_labor")]
    odds = (k.invention_scale * g.invention_rate[None, :]
            * (pop[:, None] / k.invention_reference_population) ** k.invention_population_exponent
            * (1.0 + k.division_of_labor_invention_bonus * dol[:, None])
            * prereq_share ** k.invention_prerequisite_exponent)
    # Techniques that need minerals are only invented or adopted with access to them.
    # Late techniques only while knowledge.late_techniques_enabled.
    access = resources_met(model, ids) & g.acquirable[None, :]
    invented = ~knows & access & (rng.random(knows.shape) < odds)

    # Diffusion from adjacent settlements, scaled by the receiver's polity's openness.
    contacts = neighbour_counts(model, ids, knows)
    openness = model.openness_of(ids)
    rate = k.diffusion_rate * openness * g.multiplier(knows, "diffusion")
    p = 1.0 - (1.0 - np.clip(rate, 0.0, 1.0))[:, None] ** contacts
    adopted = (~knows & ~invented & access & g.prerequisites_met(knows, held) & (contacts > 0)
               & (rng.random(knows.shape) < p))

    gained = invented | adopted
    if gained.any():
        t.knows[ids] = knows | gained
        t.strength[ids] = np.where(gained, 1.0, t.strength[ids])
        nb = model.geom.neighbors[t.cell[ids]]
        holder = np.zeros((model.geom.n_cells + 1, len(g.names)), dtype=bool)
        holder[t.cell[ids]] = knows
        for i, j in zip(*np.nonzero(gained)):
            model.note_technique_gained(int(j), year, bool(invented[i, j]))
            if not usable[i, j]:
                continue  # held latent: no event until it could be used
            sid = int(ids[i])
            actors = [("settlement", sid), ("technique", int(j))]
            if adopted[i, j]:
                src = nb[i][holder[nb[i], j]][0]
                actors.append(("settlement", int(model.occupant[src])))
            model.log_event("learned", actors, int(t.cell[sid]), float(pop[i]))