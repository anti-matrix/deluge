"""Global event log (spec sections 11 and 12): raw records only.

Every record is (year, action, actors, place, magnitude):

- ``action`` is a generic verb code from ``ACTIONS``, the only vocabulary the log has.
  The spec's list is extended by ``learned`` and ``forgot`` for techniques gained and
  lost, and ``depleted`` for a mineral deposit used up (only with depletion switched on).
- ``actors`` is a tuple of entity ids. An entity id encodes its kind and its local id
  (``local * len(ENTITY_KINDS) + kind``): the teachers, a lineage, a settlement, a
  technique, a technology the teachers display, an artifact, a mineral deposit, a polity,
  a monument. Polity events reuse the generic verbs: ``founded`` (a founding polity at
  year 0, from its band; a split, from its parent polity), ``joined`` (a merger),
  ``died`` (extinction or fragmentation), ``built`` (a monument).
- ``place`` is a cell id (-1 when the event has no place).
- ``magnitude`` is the size of the effect (people, deaths, techniques, ...). An event is
  logged only when its magnitude is at least ``events.min_magnitude[action]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACTIONS = (
    "arrived", "taught", "displayed", "departed", "gave", "used", "broke", "fought", "won",
    "lost", "moved", "founded", "built", "starved", "died", "joined", "learned", "forgot", "depleted",
)
ACTION_CODE = {name: i for i, name in enumerate(ACTIONS)}
ENTITY_KINDS = ("teachers", "lineage", "settlement", "technique", "technology", "artifact", "deposit", "polity",
                "monument")
KIND_CODE = {name: i for i, name in enumerate(ENTITY_KINDS)}
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

    def __len__(self) -> int:
        return len(self.year)

    def log(self, year: int, action: str, actors, place: int, magnitude: float) -> bool:
        code = ACTION_CODE[action]
        if magnitude < self.min_magnitude[code]:
            return False
        self.year.append(int(year))
        self.action.append(code)
        self.actors.append(tuple(int(a) for a in actors))
        self.place.append(int(place))
        self.magnitude.append(float(magnitude))
        return True

    def record(self, i: int) -> dict:
        return {"year": self.year[i], "action": ACTIONS[self.action[i]], "actors": self.actors[i],
                "place": self.place[i], "magnitude": self.magnitude[i]}

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": np.asarray(self.year, dtype=np.int64),
            "action": pd.Categorical.from_codes(np.asarray(self.action, dtype=np.int64), categories=list(ACTIONS)),
            "actors": [list(a) for a in self.actors],
            "place": np.asarray(self.place, dtype=np.int64),
            "magnitude": np.asarray(self.magnitude, dtype=float),
        })

    def write(self, path: Path) -> None:
        df = self.to_frame()
        df["action"] = df["action"].astype(str)
        df.to_parquet(path, index=False)
