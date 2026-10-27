"""The role of each actor in an event or a memory record, and sentences built from roles
(Round 13). Nothing here enters the simulation: it only reads model state.

- Roles (``roles`` in ``configs/narrator.yaml``): an actor's role is fixed by the slot it holds
  in its event, (verb, kind of the event's first actor, verb of the event it is part of,
  position), looked up per "verb/first kind/linked verb", "verb/first kind" or "verb" (the
  most specific that exists). Memory records carry each actor's slot (``Memory.roles``): a
  substituted actor holds the slot of the one it replaced, a fused record the slots of the
  actors that won its vote.
- Sentences (``sentences``): per verb, the first shape whose roles are exactly the roles present.
  A role names its actors (several joined by "and"); things (``thing_roles``) stand only in
  their own role, never beside the people.
- A record whose actors no longer hold a set of roles of its verb (an actor's slot from
  another verb, a group of people in a thing's role or the reverse, no shape for that set) is
  told in the loose form (``loose``): the groups of people, then what was done, where and when.
"""

from __future__ import annotations

import re

from deluge.data.events import ACTIONS, ENTITY_KINDS, PEOPLE_KINDS, decode

FIELD = re.compile(r"\{(\w+)\}")
SPECIAL = {"at", "to", "be", "count"}


class RoleTable:
    """The roles and sentence shapes of the phrase table."""

    def __init__(self, data: dict) -> None:
        self.data = data
        self.table = data["roles"]
        self.things = set(data["thing_roles"])
        self.shapes = {verb: [(s, self.fields(s)) for s in shapes] for verb, shapes in data["sentences"].items()}
        self.run_shapes = {verb: [(s, self.fields(s)) for s in shapes]
                           for verb, shapes in data.get("run_sentences", {}).items()}

    @staticmethod
    def fields(shape: str) -> frozenset:
        return frozenset(f for f in FIELD.findall(shape) if f not in SPECIAL)

    def spec(self, action: int, first_kind: int, via: int) -> list:
        verb = ACTIONS[action]
        kind = ENTITY_KINDS[first_kind] if first_kind >= 0 else ""
        for key in (f"{verb}/{kind}/{ACTIONS[via]}" if via >= 0 else None, f"{verb}/{kind}", verb):
            if key and key in self.table:
                return self.table[key]
        return []

    def role(self, slot, actor: int) -> str | None:
        """The role an actor holds by its slot (action, first kind, linked verb, position)."""
        if slot is None:
            return None
        action, first, via, k = slot
        spec = self.spec(action, first, via)
        if not spec:
            return None
        entry = spec[min(k, len(spec) - 1)]
        if k >= len(spec) and not (isinstance(entry, str) and entry.endswith("+")):
            return None
        if isinstance(entry, dict):
            entry = entry.get(decode(actor)[0])
            if entry is None:
                return None
        return entry.rstrip("+")

    def slots(self, action: int, actors, via: int = -1) -> list[tuple]:
        """The slots of an event's actors."""
        first = actors[0] % len(ENTITY_KINDS) if len(actors) else -1
        return [(action, first, via, k) for k in range(len(actors))]

    def fits(self, role: str, actor: int) -> bool:
        """A thing's role holds a thing, every other role a group of people."""
        return (role in self.things) == (decode(actor)[0] not in PEOPLE_KINDS)

    def shape(self, action: int, roles: frozenset, count: int = 1) -> tuple[str | None, bool]:
        """(the sentence shape of ``action`` for exactly these roles, whether it is a run's own
        shape: used when ``count`` > 1 and the verb has one); (None, False) if there is none."""
        verb = ACTIONS[action]
        if count > 1:  # a run's shape counts the things it leaves out ("raised 112 monuments")
            for s, f in self.run_shapes.get(verb, ()):
                if f <= roles and (roles - f) <= self.things:
                    return s, True
        for s, f in self.shapes.get(verb, ()):
            if f == roles:
                return s, False
        return None, False


def run_key(action: int, actors, place: int, count_kinds) -> tuple:
    """What makes events one run: the verb, the actors (those of ``count_kinds``, made anew by
    each event, by kind only: a monument each time) and the place."""
    kinds = {ENTITY_KINDS.index(k) for k in count_kinds}
    return (int(action), tuple(-1 - a % len(ENTITY_KINDS) if a % len(ENTITY_KINDS) in kinds else int(a)
                               for a in actors), int(place))


def table_for(data: dict) -> RoleTable:
    """The role table of a phrase table (kept on it)."""
    t = data.get("_role_table")
    if t is None:
        t = RoleTable(data)
        data["_role_table"] = t
    return t
