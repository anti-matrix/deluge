"""Readable lines from the raw event log, for the browser chronicle.

Each line is a template over the raw fields (year, action, actors, place, magnitude).
The templates only say what the sim did, in its own terms; entities are named by kind
and id (``settlement 412 (group C)``, ``technique agriculture``, ``the teachers``). No
names or story language are invented.
"""

from __future__ import annotations

from deluge.agents.teachers import ARTIFACT_KINDS
from deluge.data.events import ACTIONS, decode

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
    "built": "{0} built",
    "starved": "{0} starved",
    "died": "{0} came to an end",
    "joined": "{0} joined {1}",
    "learned": "{0} learned {1}{from_}",
    "forgot": "{0} forgot {1}",
}
MAGNITUDE = {
    "arrived": "{m:,.0f} people there",
    "taught": "{m:.0f} techniques passed on",
    "gave": "{m:.0f} artifacts",
    "used": "{m:,.0f} famine deaths spared",
    "moved": "{m:,.0f} people",
    "founded": "{m:,.0f} people",
    "starved": "{m:,.0f} deaths",
    "died": "{m:,.0f} people left",
    "joined": "{m:,.0f} people",
    "learned": "{m:,.0f} people",
    "forgot": "{m:,.0f} people",
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
        if kind == "technique":
            return m.knowledge.names[i].replace("_", " ")
        if kind == "technology":
            return m.config.teachers.technologies[i]
        if kind == "artifact":
            return f"artifact {i} ({ARTIFACT_KINDS[m.teachers_state.kind[i]]})"
        return f"{kind} {i}"


def line(record: dict, name: Namer) -> str:
    action = record["action"]
    actors = record["actors"]
    names = [name(a) for a in actors]
    padded = names + [""] * 3
    text = TEMPLATES[action].format(*padded, rest=", ".join(names[2:]) or "nothing",
                                    from_=f" from {names[2]}" if action == "learned" and len(names) > 2 else "")
    if action == "learned" and len(names) == 2:
        text += " (invented)"
    mag = MAGNITUDE.get(action)
    if mag:
        text += f" ({mag.format(m=record['magnitude'])})"
    return f"Year {record['year']}: {text}."


def lines(model, limit: int, groups: set[int] | None = None, actions: set[str] | None = None) -> list[str]:
    """The newest ``limit`` events as lines, newest first, optionally only those
    involving the given founding groups and of the given actions."""
    log = model.events
    name = Namer(model)
    out = []
    for i in range(len(log) - 1, -1, -1):
        action = ACTIONS[log.action[i]]
        if actions is not None and action not in actions:
            continue
        if groups is not None and not any(name.lineage_of(a) in groups for a in log.actors[i]):
            continue
        out.append(line(log.record(i), name))
        if len(out) >= limit:
            break
    return out
