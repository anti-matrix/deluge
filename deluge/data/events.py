"""Global event log (spec sections 11 and 12): raw records only.

Every record is (year, action, actors, place, magnitude):

- ``action`` is a generic verb code from ``ACTIONS``, the only vocabulary the log has.
  The spec's list is extended by ``learned`` and ``forgot`` for techniques gained and
  lost, ``depleted`` for a mineral deposit used up (only with depletion switched on), and
  ``took`` for a settlement taken by conquest.
- ``actors`` is a tuple of entity ids. An entity id encodes its kind and its local id
  (``local * len(ENTITY_KINDS) + kind``): the teachers, a lineage, a settlement, a
  technique, a technology the teachers display, an artifact, a mineral deposit, a polity,
  a monument. Polity events reuse the generic verbs: ``founded`` (a founding polity at
  year 0, from its band; a split, from its parent polity), ``joined`` (a merger),
  ``died`` (extinction, fragmentation, or destruction by the second actor), ``built`` (a
  monument). War events (``systems/war.py``): ``fought`` (a battle: attacker, defender,
  target settlement; dead), ``won`` (a raid won; rations taken), ``took`` (conqueror,
  settlement, former polity; people), ``lost`` (a polity losing its capital to the third
  actor).
- ``crossed`` is a sea crossing: a band reaching its new cell over sea (migration, a
  daughter founding, joining kin, migration of the core) or a polity's settlement attacking
  over sea. Its magnitude is the number of sea cells crossed (the fewest any path within
  the reach takes), its place the cell reached, and it is logged right after the event it is
  part of (``link``).
- ``place`` is a cell id (-1 when the event has no place).
- ``magnitude`` is the size of the effect (people, deaths, techniques, ...). An event is
  logged only when its magnitude is at least ``events.min_magnitude[action]``.
- ``link`` is the index of the event this one is part of (-1 for none).
- ``people`` is, per actor, how many people it had when the event was logged (settlements
  and polities; NaN for other entities). The memory system reads it for salience.
- ``polity_people`` is, per actor, the people of the polity it is or belongs to (a polity
  actor, or a settlement actor's polity), last counted when the event was logged; NaN for
  other entities and for a polity not counted yet. Only the chronicle reads it.
- ``origin`` is the cell a sea crossing set out from (-1 for every other event). Only the
  chronicle reads it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = (
    "arrived", "taught", "displayed", "departed", "gave", "used", "broke", "fought", "won",
    "lost", "moved", "founded", "built", "starved", "died", "joined", "learned", "forgot", "depleted", "took",
    "crossed",
)
ACTION_CODE = {name: i for i, name in enumerate(ACTIONS)}
ENTITY_KINDS = ("teachers", "lineage", "settlement", "technique", "technology", "artifact", "deposit", "polity",
                "monument")
KIND_CODE = {name: i for i, name in enumerate(ENTITY_KINDS)}
PEOPLE_KINDS = ("teachers", "lineage", "settlement", "polity")  # entities that are groups of people
TEACHERS = 0  # local id of the one teachers entity


def entity(kind: str, local: int) -> int:
    return int(local) * len(ENTITY_KINDS) + KIND_CODE[kind]


def decode(eid: int) -> tuple[str, int]:
    local, kind = divmod(int(eid), len(ENTITY_KINDS))
    return ENTITY_KINDS[kind], local


class EventLog:
    def __init__(self, min_magnitude) -> None:
        unknown = set(min_magnitude) - set(ACTIONS)
        missing = set(ACTIONS) - set(min_magnitude)
        if unknown or missing:
            raise KeyError(f"events.min_magnitude: unknown {sorted(unknown)}, missing {sorted(missing)}")
        self.min_magnitude = np.array([min_magnitude[a] for a in ACTIONS])
        self.year: list[int] = []
        self.action: list[int] = []
        self.actors: list[tuple[int, ...]] = []
        self.place: list[int] = []
        self.magnitude: list[float] = []
        self.link: list[int] = []
        self.people: list[tuple[float, ...]] = []
        self.polity_people: list[tuple[float, ...]] = []
        self.origin: list[int] = []

    def __len__(self) -> int:
        return len(self.year)

    def log(self, year: int, action: str, actors, place: int, magnitude: float, link: int = -1,
            people=None, polity_people=None, origin: int = -1) -> bool:
        code = ACTION_CODE[action]
        if magnitude < self.min_magnitude[code]:
            return False
        actors = tuple(int(a) for a in actors)
        # The columns the chronicle reads first are appended last (``Chronicle._snapshot``).
        self.link.append(int(link))
        self.people.append(tuple(float(p) for p in people) if people is not None else (np.nan,) * len(actors))
        self.polity_people.append(tuple(float(p) for p in polity_people) if polity_people is not None
                                  else (np.nan,) * len(actors))
        self.origin.append(int(origin))
        self.year.append(int(year))
        self.action.append(code)
        self.actors.append(actors)
        self.place.append(int(place))
        self.magnitude.append(float(magnitude))
        return True

    def record(self, i: int) -> dict:
        return {"year": self.year[i], "action": ACTIONS[self.action[i]], "actors": self.actors[i],
                "place": self.place[i], "magnitude": self.magnitude[i], "link": self.link[i]}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": np.asarray(self.year, dtype=np.int64),
            "action": pd.Categorical.from_codes(np.asarray(self.action, dtype=np.int64), categories=list(ACTIONS)),
            "actors": [list(a) for a in self.actors],
            "place": np.asarray(self.place, dtype=np.int64),
            "magnitude": np.asarray(self.magnitude, dtype=float),
            "link": np.asarray(self.link, dtype=np.int64),
            "people": [list(p) for p in self.people],
            "polity_people": [list(p) for p in self.polity_people],
            "origin": np.asarray(self.origin, dtype=np.int64),
        })

    def write(self, path: Path) -> None:
        df = self.to_frame()
        df["action"] = df["action"].astype(str)
        df.to_parquet(path, index=False)
