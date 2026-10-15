"""Readable lines from the raw event log, for the browser chronicle.

Each line is a template over the raw fields (year, action, actors, place, magnitude).
The templates only say what the sim did, in its own terms; entities are named by kind
and id (``settlement 412 (group C)``, ``polity 37 (group C)``, a technique by its label from the knowledge graph,
``copper deposit at cell (31, 12)``, ``the teachers``). No names or story language are
invented.

``Chronicle`` is a read-only view over the whole log: it formats lines lazily and caches
them, filters by founding group, event type, year range and text, and pages the result.
The log itself is never trimmed; the view only chooses what to show.
"""

from __future__ import annotations

import numpy as np

from deluge.agents.teachers import ARTIFACT_KINDS
from deluge.data.events import ACTION_CODE, ACTIONS, decode
from deluge.world.worlddata import MINERALS

TEMPLATES = {
    "arrived": "{0} arrived at {1}, by {2}",
    "taught": "{0} taught {1}: {rest}",
    "displayed": "{0} displayed {2} at {1}",
    "departed": "{0} departed from {1}, by {2}",
    "gave": "{0} gave {1}: {rest}",
    "used": "{1} used {0}",
    "broke": "{0} held by {1} broke",
    "fought": "{0} attacked {1}",
    "won": "{0} won against {1}",
    "lost": "{0} lost against {1}",
    "moved": "{0} moved",
    "founded": "{0} was founded from {1}",
    "built": "{0} built {1} at {2}",
    "starved": "{0} starved",
    "died": "{0} came to an end",
    "joined": "{0} joined {1}",
    "learned": "{0} learned {1}{from_}",
    "forgot": "{0} forgot {1}",
    "depleted": "{0} was used up",
    "took": "{0} took {1} from {2}",
}
# Templates by action, the kind of the first actor and the number of actors, then by
# action and kind, where they differ from TEMPLATES (war events between polities).
TEMPLATES_BY_KIND = {
    ("fought", "polity"): "{0} attacked {1} at {2}",
    ("won", "polity"): "{0} raided {1} at {2}",
    ("lost", "polity"): "{0} lost its capital {1} to {2}",
    ("died", "polity", 2): "{0} came to an end, taken by {1}",
}
MAGNITUDE = {
    "arrived": "{m:,.0f} people there",
    "taught": "{m:.0f} techniques passed on",
    "gave": "{m:.0f} artifacts",
    "used": "{m:,.0f} famine deaths spared",
    "moved": "{m:,.0f} people",
    "founded": "{m:,.0f} people",
    "built": "{m:,.0f} rations of labour",
    "starved": "{m:,.0f} deaths",
    "died": "{m:,.0f} people left",
    "joined": "{m:,.0f} people",
    "learned": "{m:,.0f} people",
    "forgot": "{m:,.0f} people",
    "took": "{m:,.0f} people",
}
# Magnitude by action and the kind of the first actor, where it differs from MAGNITUDE.
MAGNITUDE_BY_KIND = {
    ("used", "deposit"): "{m:,.0f} people working it",
    ("died", "polity"): "{m:,.0f} people in it",
    ("fought", "polity"): "{m:,.0f} dead",
    ("won", "polity"): "{m:,.0f} rations taken",
    ("lost", "polity"): "{m:,.0f} people",
}
if set(TEMPLATES) != set(ACTIONS):
    raise RuntimeError("chronicle templates must cover every action")


class Namer:
    """Names entities from the model's registries."""

    def __init__(self, model) -> None:
        self.model = model

    def lineage_of(self, eid: int) -> int:
        """Founding group of a settlement entity (-1 for other entities)."""
        kind, i = decode(eid)
        if kind == "settlement":
            return int(self.model.settlements.lineage[i])
        if kind == "lineage":
            return i
        if kind == "polity":
            return int(self.model.polities.lineage[i])
        return -1

    def __call__(self, eid: int) -> str:
        kind, i = decode(eid)
        m = self.model
        if kind == "teachers":
            return "the teachers"
        if kind == "lineage":
            return f"group {m.lineages.names[i]}"
        if kind == "settlement":
            return f"settlement {i} (group {m.lineages.names[int(m.settlements.lineage[i])]})"
        if kind == "polity":
            return f"polity {i} (group {m.lineages.names[int(m.polities.lineage[i])]})"
        if kind == "monument":
            return f"monument {i}"
        if kind == "technique":
            return m.knowledge.labels[i]
        if kind == "technology":
            return m.config.teachers.technologies[i]
        if kind == "artifact":
            return f"artifact {i} ({ARTIFACT_KINDS[m.teachers_state.kind[i]]})"
        if kind == "deposit":
            cell, mineral = divmod(i, len(MINERALS))
            x, y = m.geom.xy(cell)
            return f"{MINERALS[mineral].replace('_', ' ')} deposit at cell ({x}, {y})"
        return f"{kind} {i}"


def line(record: dict, name: Namer) -> str:
    action = record["action"]
    actors = record["actors"]
    names = [name(a) for a in actors]
    padded = names + [""] * 3
    kind = decode(actors[0])[0] if actors else ""
    template = (TEMPLATES_BY_KIND.get((action, kind, len(actors))) or TEMPLATES_BY_KIND.get((action, kind))
                or TEMPLATES[action])
    text = template.format(*padded, rest=", ".join(names[2:]) or "nothing",
                                    from_=f" from {names[2]}" if action == "learned" and len(names) > 2 else "")
    if action == "learned" and len(names) == 2:
        text += " (invented)"
    mag = MAGNITUDE_BY_KIND.get((action, decode(actors[0])[0])) if actors else None
    mag = mag or MAGNITUDE.get(action)
    if mag:
        text += f" ({mag.format(m=record['magnitude'])})"
    return f"Year {record['year']}: {text}."


class Chronicle:
    """Read-only view over a model's whole event log (see the module docstring)."""

    def __init__(self, model) -> None:
        self.model = model
        self.name = Namer(model)
        self._lines: list[str | None] = []
        self._groups: list[int] = []  # per event: bit g set when founding group g takes part

    def __len__(self) -> int:
        return len(self.model.events)

    def _sync(self) -> None:
        log = self.model.events
        for i in range(len(self._groups), len(log)):
            mask = 0
            for a in log.actors[i]:
                g = self.name.lineage_of(a)
                if g >= 0:
                    mask |= 1 << g
            self._groups.append(mask)
            self._lines.append(None)

    def line(self, i: int) -> str:
        self._sync()
        if self._lines[i] is None:
            self._lines[i] = line(self.model.events.record(i), self.name)
        return self._lines[i]

    def select(self, groups: set[int] | None = None, actions: set[str] | None = None,
               year_min: int | None = None, year_max: int | None = None, text: str = "") -> np.ndarray:
        """Indices of the matching events, newest first. ``text`` matches the formatted
        line, case-insensitively."""
        self._sync()
        log = self.model.events
        keep = np.ones(len(log), dtype=bool)
        if actions is not None:
            keep &= np.isin(np.asarray(log.action, dtype=np.int64), [ACTION_CODE[a] for a in actions])
        if groups is not None:
            want = sum(1 << g for g in groups)
            keep &= (np.asarray(self._groups, dtype=np.int64) & want) != 0
        years = np.asarray(log.year, dtype=np.int64)
        if year_min is not None:
            keep &= years >= year_min
        if year_max is not None:
            keep &= years <= year_max
        idx = np.flatnonzero(keep)[::-1]
        needle = text.strip().lower()
        if needle:
            idx = np.array([i for i in idx.tolist() if needle in self.line(i).lower()], dtype=np.int64)
        return idx

    def page(self, indices: np.ndarray, page: int, size: int) -> list[str]:
        """Lines of page ``page`` (0 = newest) of ``indices``, ``size`` per page."""
        return [self.line(int(i)) for i in indices[page * size:(page + 1) * size]]

    @staticmethod
    def pages(indices: np.ndarray, size: int) -> int:
        return max(1, -(-len(indices) // size))

    def text(self, indices: np.ndarray | None = None) -> str:
        """The chronicle as plain text, oldest first: every event, or ``indices``."""
        self._sync()
        order = range(len(self)) if indices is None else sorted(int(i) for i in indices)
        return "\n".join(self.line(i) for i in order) + "\n"


def lines(model, limit: int, groups: set[int] | None = None, actions: set[str] | None = None) -> list[str]:
    """The newest ``limit`` events as lines, newest first, optionally only those
    involving the given founding groups and of the given actions."""
    c = Chronicle(model)
    return c.page(c.select(groups=groups, actions=actions), 0, limit)
