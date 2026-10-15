"""Phase 7: war and conquest (spec section 9).

Each year, first the conquered settlements assimilate (``assimilate``), then, with
``war.enabled``, the polities choose and fight their battles (``fight``).

Choice (the polity brains, Round 7). The candidate targets of a polity are the foreign
settlements in contact (``polities.contact_radius``) with one of its own. The polity's
per-neighbour head scores each candidate from its target features
(``brains/features.py``: estimated win chance, stores, fertility, culture distance,
perceived appearance and dialect distance, already at war, a capital). Each polity takes
its best-scoring target and attacks it with chance ``attack_rate`` x its aggression output
x score x min(food per head, 1) (its surplus capacity to fight); a campaign rather than a
raid with chance = its campaign output. A polity already at war with ``max_wars`` polities
only attacks those. The teachers are not chosen as a target: they hold no stores or land
(``TeachersState.attacked`` stays for a later round).

Strength S = P x mobilization x M_tech x (0.5 + cohesion) x D. Mobilization rises with
militarism and hierarchy; M_tech is the product of the capital's military multipliers
(bronze, horses, iron, ...) and the teachers' weapons while a member holds them; D =
exp(-km from the capital / (``logistics_km`` x the capital's raid_range multipliers:
horses, sailing, writing)). The attacker fights with its whole polity at D; the defender
with the target settlement's people in full and the rest of its polity at its own D, times
the terrain's ``terrain_defense``, the target's defense multiplier (sedentary building) and
``home_bonus``.

Resolution: the attacker wins with chance Sa^k / (Sa^k + Sd^k), k = ``win_steepness``. Each
side loses ``casualty_rate`` x the other's strength (more for the loser), capped; the
attacker's dead are spread over its settlements by population, the defender's come from the
target. Military techniques may pass between the two settlements that fought. A raid won
carries off part of the target's store. A campaign won takes the target and possibly its
neighbours of the same polity, with settlers from the attacking settlement; if the target
was the loser's capital, the loser loses cohesion and moves its capital, or, with too few
people left, is destroyed and the conqueror takes everything it had.

After conquest a settlement keeps its own culture and techniques, gets ``grievance`` and is
a subject (``subject_of``) of its conqueror. Each year its culture moves toward its ruler's
at ``assimilation_rate`` x the ruler's openness (x (1 + memory weight x shared-memory
overlap; zero until Round 8)), its techniques pass to the ruler's capital at
``subject_diffusion_rate``, and its grievance decays; once grievance is low and its culture
close to its ruler's it is assimilated. Grievance raises secession and lowers cohesion
(``politics``, ``cohesion``); each battle counts as recent war in frontier pressure, and
as a win or a loss in the brain inputs. Members that are not subjects hold their polity's
culture. Every member's grievance also moves by ``polities.marker_grievance_rate`` x its
polity's grievance marker parameters . its perceived appearance and dialect distance from
the capital (zero in the founders; floored at 0).

Events (raw): ``fought`` (attacker polity, defender polity, target settlement; dead) for
battles with at least ``battle_log_min_losses`` dead; ``won`` (attacker, defender, target;
rations taken) for every raid won; ``took`` (conqueror, settlement, former polity; people)
for every settlement conquered; ``lost`` (polity, its capital, conqueror; people) for a
captured capital; ``died`` (polity, conqueror) for a destroyed polity.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from deluge.agents.polity import marker_index
from deluge.brains.features import target_features
from deluge.brains.network import OUT, target_scores
from deluge.systems import identity, knowledge
from deluge.systems.cohesion import culture_distance, settlement_contacts, trait_index
from deluge.world.worlddata import terrain_table

WIN_EPS = 1e-12


def subject_memory_overlap(model, sids: np.ndarray) -> np.ndarray:
    """Shared-memory overlap between subject settlements and their rulers: zero until
    memory exists (Round 8)."""
    return np.zeros(len(sids))


def win_chance(sa, sd, steepness: float):
    sa, sd = np.asarray(sa, float), np.asarray(sd, float)
    return np.where(sa > 0, 1.0 / (1.0 + (sd / np.maximum(sa, WIN_EPS)) ** steepness), 0.0)


# --- assimilation -----------------------------------------------------------------------------


def assimilate(model) -> None:
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    ids = t.alive_ids()
    if len(ids) == 0:
        return
    pol = t.polity[ids]
    subject = (t.subject_of[ids] >= 0) & (t.subject_of[ids] == pol)
    native = ids[~subject]
    t.culture[native] = P.culture[t.polity[native]]
    t.subject_of[native] = -1
    subjects = ids[subject]
    if len(subjects):
        ruler = t.polity[subjects]
        rate = (w.assimilation_rate * model.openness_of(subjects)
                * (1.0 + w.assimilation_memory_weight * subject_memory_overlap(model, subjects)))
        t.culture[subjects] += rate[:, None] * (P.culture[ruler] - t.culture[subjects])
        # Techniques the conquered hold pass to the ruler's capital.
        capital = P.capital[ruler]
        offer = t.knows[subjects] & ~t.knows[capital] & g.acquirable[None, :]
        if offer.any():
            draw = model.streams["war"].random(offer.shape) < w.subject_diffusion_rate
            for i, j in zip(*np.nonzero(offer & draw)):
                knowledge.grant(model, int(capital[i]), int(j), int(subjects[i]))
    t.grievance[ids] *= 1.0 - w.grievance_decay
    app, dia = identity.core_distances(model, ids)
    e = P.marker[pol][:, marker_index("grievance_appearance", "grievance_dialect")]
    rate = model.config.polities.marker_grievance_rate
    t.grievance[ids] = np.maximum(t.grievance[ids] + rate * (e[:, 0] * app + e[:, 1] * dia), 0.0)
    gap = culture_distance(t.culture[ids], P.culture[pol])
    done = subject & (t.grievance[ids] < w.grievance_floor) & (gap < w.assimilated_distance)
    t.subject_of[ids[done]] = -1


# --- strength ---------------------------------------------------------------------------------


def _mobilization_and_tech(model, ids: np.ndarray, pol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per polity id: mobilization share and military multiplier (capital's techniques,
    teachers' weapons)."""
    w, cfg = model.config.war, model.config
    t, P, g = model.settlements, model.polities, model.knowledge
    n = P.n
    culture = P.culture[:n]
    mob = np.clip(w.mobilization_base + w.mobilization_militarism * culture[:, trait_index(model, "militarism")]
                  + w.mobilization_hierarchy * culture[:, trait_index(model, "hierarchy")], 0.0, w.mobilization_max)
    tech = g.multiplier(t.knows[P.capital[:n]], "military")
    armed = np.zeros(n, dtype=bool)
    armed[pol[model.teachers_state.weapons_holders(ids)]] = True
    return mob, np.where(armed, tech * cfg.teachers.weapons_military, tech)


def polity_strength(model) -> np.ndarray:
    """Per polity id: people x mobilization x M_tech x (0.5 + cohesion), before distance and
    terrain (the brains' strength-ratio inputs)."""
    t, P = model.settlements, model.polities
    ids = t.alive_ids()
    pol = t.polity[ids]
    people = np.bincount(pol, weights=t.population[ids], minlength=P.n)
    mob, tech = _mobilization_and_tech(model, ids, pol)
    return people * mob * tech * (0.5 + P.cohesion[: P.n])


def forces(model) -> SimpleNamespace:
    """Per-polity numbers (indexed by polity id) that strength is built from, this year."""
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    ids = t.alive_ids()
    pol = t.polity[ids]
    n = P.n
    people = np.bincount(pol, weights=t.population[ids], minlength=n)
    fed = np.bincount(pol, weights=t.population[ids] * t.food_ratio[ids], minlength=n) / np.maximum(people, 1.0)
    mob, tech = _mobilization_and_tech(model, ids, pol)
    capital = P.capital[:n]
    reach = w.logistics_km * g.multiplier(t.knows[capital], "raid_range")
    members: dict[int, list[int]] = {}
    for sid, p in zip(ids.tolist(), pol.tolist()):
        members.setdefault(p, []).append(sid)
    return SimpleNamespace(people=people, fed=fed, mob=mob, tech=tech, reach=reach,
                           capital_cell=t.cell[capital].copy(), members=members,
                           terrain=terrain_table(w.terrain_defense)[model.world.terrain.ravel()])


def strengths(model, F, a: np.ndarray, b: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strength of attacker polities ``a`` and defender polities ``b`` at target settlements ``x``."""
    w = model.config.war
    t, P, g = model.settlements, model.polities, model.knowledge
    cell = t.cell[x]
    km_a = model.geom.hex_distance(F.capital_cell[a], cell) * model.unit_km
    km_b = model.geom.hex_distance(F.capital_cell[b], cell) * model.unit_km
    sa = F.people[a] * F.mob[a] * F.tech[a] * (0.5 + P.cohesion[a]) * np.exp(-km_a / F.reach[a])
    local = t.population[x]
    rest = np.maximum(F.people[b] - local, 0.0) * np.exp(-km_b / F.reach[b])
    sd = ((local + rest) * F.mob[b] * F.tech[b] * (0.5 + P.cohesion[b]) * F.terrain[cell]
          * g.multiplier(t.knows[x], "defense") * w.home_bonus)
    return sa, sd


# --- choice -----------------------------------------------------------------------------------


def at_war_counts(model) -> np.ndarray:
    counts = np.zeros(model.polities.n, dtype=np.int64)
    for a, b in model.wars:
        counts[a] += 1
        counts[b] += 1
    return counts


def choose(model, F) -> list[tuple[int, int, int, int, bool]]:
    """This year's attacks: (attacker, defender, attacking settlement, target, campaign),
    in attacker id order."""
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    ids = t.alive_ids()
    if len(ids) == 0:
        return []
    contacts = settlement_contacts(model, ids)
    pol = t.polity[ids]
    other = np.where(contacts >= 0, t.polity[np.maximum(contacts, 0)], -1)
    i, k = np.nonzero((other >= 0) & (other != pol[:, None]))
    if len(i) == 0:
        return []
    src, x, a = ids[i], contacts[i, k], pol[i]
    _, first = np.unique(a * (t.n + 1) + x, return_index=True)  # one attacking settlement per (polity, target)
    src, x, a = src[first], x[first], a[first]
    b = t.polity[x]
    busy = at_war_counts(model) >= w.max_wars
    if busy[a].any():
        allowed = ~busy[a] | np.array([(min(p, q), max(p, q)) in model.wars for p, q in zip(a.tolist(), b.tolist())])
        src, x, a, b = src[allowed], x[allowed], a[allowed], b[allowed]
        if len(a) == 0:
            return []
    sa, sd = strengths(model, F, a, b, x)
    p_win = win_chance(sa, sd, w.win_steepness)
    brains = model.brains
    score = target_scores(target_features(model, a, b, x, p_win), brains.target_w[a])
    order = np.lexsort((-score, a))
    best = order[np.r_[True, a[order][1:] != a[order][:-1]]]  # top score per attacker, attacker id order
    out = brains.output[a[best]]
    p_attack = w.attack_rate * out[:, OUT["aggression"]] * score[best] * np.clip(F.fed[a[best]], 0.0, 1.0)
    attack = rng.random(len(best)) < p_attack
    campaign = rng.random(len(best)) < out[:, OUT["campaign"]]
    return [(int(a[j]), int(b[j]), int(src[j]), int(x[j]), bool(c))
            for j, go, c in zip(best.tolist(), attack.tolist(), campaign.tolist()) if go]


# --- resolution -------------------------------------------------------------------------------


def take(model, F, a: int, b: int, x: int, src: int) -> None:
    """Polity ``a`` takes settlement ``x`` from ``b``; settlers come from ``src`` (-1: none)."""
    w = model.config.war
    t = model.settlements
    people = float(t.population[x])
    t.polity[x] = a
    t.subject_of[x] = a
    t.grievance[x] = max(float(t.grievance[x]), w.grievance_conquest)
    if src >= 0 and t.ended[src] < 0:
        settlers = min(w.settler_share * people, w.settler_source_cap * float(t.population[src]))
        if settlers > 0:
            identity.blend(model, x, src, settlers)
            t.population[src] -= settlers
            t.population[x] += settlers
    F.members[b].remove(x)
    F.members.setdefault(a, []).append(x)
    F.people[b] -= people
    F.people[a] += people
    model.stats["conquests"] += 1
    model.log_event("took", [("polity", a), ("settlement", x), ("polity", b)], int(t.cell[x]), people)


def campaign_won(model, F, a: int, b: int, src: int, x: int) -> None:
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    capital = int(P.capital[b])
    before = max(float(F.people[b]), 0.0)
    take(model, F, a, b, x, src)
    near = np.append(model.occupant, -1)[model.contact_cells[t.cell[x]]]
    for y in near[near >= 0].tolist():
        if t.polity[y] == b and y != capital and rng.random() < w.neighbor_capture_chance:
            take(model, F, a, b, int(y), src)
    if x != capital:
        return
    model.stats["capitals_taken"] += 1
    model.log_event("lost", [("polity", b), ("settlement", x), ("polity", a)], int(t.cell[x]), float(t.population[x]))
    rest = F.members[b]
    left = max(float(F.people[b]), 0.0)
    if rest and left >= w.destroy_ratio * F.people[a]:
        new = max(rest, key=lambda s: (t.population[s], -s))
        P.capital[b] = new
        F.capital_cell[b] = t.cell[new]
        P.cohesion[b] *= 1.0 - w.capital_cohesion_loss
        return
    for y in list(rest):
        take(model, F, a, b, y, -1)
    model.stats["polities_destroyed"] += 1
    model.end_polity(b, "conquered", merged_into=a, people=before)


def exchange_military(model, s1: int, s2: int) -> None:
    """After a battle, each military technique one side's settlement holds may pass to the other's."""
    t, g = model.settlements, model.knowledge
    rng = model.streams["war"]
    for giver, taker in ((s1, s2), (s2, s1)):
        for j in np.flatnonzero(g.military & t.knows[giver] & ~t.knows[taker]).tolist():
            if rng.random() < model.config.war.enemy_military_diffusion:
                knowledge.grant(model, taker, j, giver)


def battle(model, F, a: int, b: int, src: int, x: int, campaign: bool) -> bool:
    """Resolve one raid or campaign; returns whether the attacker won."""
    w = model.config.war
    t, P = model.settlements, model.polities
    rng = model.streams["war"]
    sa, sd = (float(v[0]) for v in strengths(model, F, np.array([a]), np.array([b]), np.array([x])))
    won = bool(rng.random() < win_chance(sa, sd, w.win_steepness))
    loss_a = min(w.casualty_rate * sd * (1.0 if won else w.loser_casualty_factor),
                 w.attacker_casualty_cap * F.people[a] * F.mob[a])
    loss_d = min(w.casualty_rate * sa * (w.loser_casualty_factor if won else 1.0),
                 w.defender_casualty_cap * float(t.population[x]))
    own = np.array(F.members[a], dtype=np.int64)
    pop = t.population[own]
    if pop.sum() > 0 and loss_a > 0:
        t.population[own] = pop - loss_a * pop / pop.sum()
    t.population[x] -= loss_d
    F.people[a] -= loss_a
    F.people[b] -= loss_d
    model.stats["battles"] += 1
    model.stats["campaigns" if campaign else "raids"] += 1
    model.stats["war_deaths"] += loss_a + loss_d
    P.war_recent[a] += 1.0
    P.war_recent[b] += 1.0
    P.wins_recent[a if won else b] += 1.0
    P.losses_recent[b if won else a] += 1.0
    model.wars[(min(a, b), max(a, b))] = model.year
    cell = int(t.cell[x])
    if loss_a + loss_d >= w.battle_log_min_losses:
        model.log_event("fought", [("polity", a), ("polity", b), ("settlement", x)], cell, loss_a + loss_d)
    exchange_military(model, src, x)
    if won:
        if campaign:
            campaign_won(model, F, a, b, src, x)
        else:
            loot = w.raid_loot_share * float(t.store[x])
            t.store[x] -= loot
            t.store[src] += loot
            model.log_event("won", [("polity", a), ("polity", b), ("settlement", x)], cell, loot)
    return won


def fight(model) -> None:
    F = forces(model)
    t, P = model.settlements, model.polities
    for a, b, src, x, campaign in choose(model, F):
        # Skip what this year's earlier battles changed: a polity gone, a target taken.
        if P.ended[a] >= 0 or P.ended[b] >= 0 or t.polity[x] != b or t.polity[src] != a:
            continue
        battle(model, F, a, b, src, x, campaign)


def run(model) -> None:
    w = model.config.war
    assimilate(model)
    P = model.polities
    P.war_recent[: P.n] *= 1.0 - w.war_recent_decay
    keep = 1.0 - model.config.brains.features.war_record_decay
    P.wins_recent[: P.n] *= keep
    P.losses_recent[: P.n] *= keep
    if w.enabled:
        fight(model)
    year = model.year
    model.wars = {pair: y for pair, y in model.wars.items()
                  if year - y <= w.war_memory_years and P.ended[pair[0]] < 0 and P.ended[pair[1]] < 0}
    model.wars_active = len(model.wars)
