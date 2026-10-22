"""Where things are, in words: the names of places, made on demand (spec section 12; Round 13).

Nothing here enters the simulation: it only reads model state.

- A cell has a name only while a settlement stands on it, and then it has that settlement's
  name: a settlement and its cell never have two names. An empty cell has no name.
- Who stood where, and when, comes from the occupancy log (``OccupancyLog``): the model notes
  every arrival on and departure from a cell (founding, moving, ending), stamped with the
  length of the event log at that moment. Event ``i`` of the log therefore sees the cells as
  they stood when it was logged. The log is written by the model and read only here.
- A place with no name (an empty cell, or the site an actor of the event has just arrived on)
  is told relative to the nearest named place: by direction (eight points; north is up the
  map) and distance in plain words ("east of Mebu", "far to the south of Mebu"), up to
  ``reports.place_far_cells`` hex steps; beyond that by its land ("the land of Mebu", the
  nearest named place on the same connected land that year) or as an unpeopled land, and at
  sea as the open sea.
- A region (memory records coarsened to ``memory.region_cells`` squares) is named after the
  most important named place inside it at the time of the report: its most populous
  settlement (then the lowest id); a region with none is told relative to its centre cell.

Settlement names themselves are made by ``naming.proper_name`` only when first needed; no table
of names is built for the map.
"""

from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right

import numpy as np

from deluge.data.events import ACTION_CODE, decode

ARRIVALS = (ACTION_CODE["founded"], ACTION_CODE["moved"])  # events with which a settlement arrives on its cell
DIRECTIONS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


class OccupancyLog:
    """Arrivals on and departures from cells, in order: (stamp, cell, settlement or -1).

    ``stamp`` is the length of the event log when it happened. Kept as compact typed arrays
    (24 bytes per change)."""

    def __init__(self) -> None:
        self.stamp = array("q")
        self.cell = array("q")
        self.sid = array("q")

    def __len__(self) -> int:
        return len(self.stamp)

    def add(self, stamp: int, cell: int, sid: int) -> None:
        self.stamp.append(int(stamp))
        self.cell.append(int(cell))
        self.sid.append(int(sid))

    def nbytes(self) -> int:
        return 3 * self.stamp.itemsize * len(self.stamp)


class Gazetteer:
    """Reads the occupancy log of one model (rebuilt when the log has grown)."""

    def __init__(self, model) -> None:
        self.model = model
        self.n = -1
        self._landmass: dict[float, np.ndarray] = {}

    # --- who stood where -------------------------------------------------------------------------

    def _index(self):
        log = getattr(self.model, "_occupancy", None)
        if log is None:
            return None
        if len(log) != self.n:
            stamp = np.frombuffer(log.stamp, dtype=np.int64)
            cell = np.frombuffer(log.cell, dtype=np.int64)
            order = np.argsort(cell, kind="stable")  # by cell, then in the order they happened
            self.width = int(stamp[-1]) + 2 if len(stamp) else 1
            self.keys = cell[order] * self.width + stamp[order]
            self.sids = np.frombuffer(log.sid, dtype=np.int64)[order].copy()
            # Per cell, for one cell at a time: (stamps, settlements) in the order they happened.
            cells = cell[order]
            starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]]) if len(cells) else np.zeros(0, int)
            ends = np.r_[starts[1:], len(cells)]
            st, sd = stamp[order].tolist(), self.sids.tolist()
            self.by_cell = {int(cells[a]): (st[a:b], sd[a:b]) for a, b in zip(starts.tolist(), ends.tolist())}
            self.n = len(log)
        return self

    def occupants(self, cells, at: int | None = None, before: bool = False) -> np.ndarray:
        """The settlement standing on each of ``cells`` (-1: none) when event ``at`` was logged
        (None: now). ``before``: just before it, leaving out what happened with that event."""
        cells = np.asarray(cells, dtype=np.int64)
        if at is None or self._index() is None:
            return self.model.occupant[cells].copy()
        at = min(int(at), self.width - 1)
        key = cells * self.width + at
        k = np.searchsorted(self.keys, key, side="left" if before else "right") - 1
        ok = (k >= 0) & (self.keys[np.maximum(k, 0)] // self.width == cells)
        return np.where(ok, self.sids[np.maximum(k, 0)], -1)

    def occupant(self, cell: int, at: int | None = None, before: bool = False) -> int:
        if at is None or self._index() is None:
            return int(self.model.occupant[cell])
        stamps, sids = self.by_cell.get(int(cell), ((), ()))
        k = (bisect_left if before else bisect_right)(stamps, int(at)) - 1
        return sids[k] if k >= 0 else -1

    # --- telling a place ---------------------------------------------------------------------------

    def _land(self, year: int) -> np.ndarray:
        m = self.model
        year = min(max(int(year), 0), len(m.sea_level_series) - 1)
        level = float(m.sea_level_series[year])
        labels = self._landmass.get(level)
        if labels is None:
            labels = m.geom.components(m.elevation_flat >= level)
            self._landmass[level] = labels
        return labels

    def nearest(self, cell: int, at: int | None, before: bool = False, exclude=()) -> tuple[int, int]:
        """(settlement, hex steps) of the named place nearest ``cell`` (the lowest cell id among
        equals), leaving out the settlements ``exclude``; (-1, -1) if none stands anywhere."""
        m = self.model
        occ = self.occupants(np.arange(m.geom.n_cells), at, before)
        if len(exclude):
            occ[np.isin(occ, np.asarray(list(exclude), dtype=np.int64))] = -1
        cells = np.flatnonzero(occ >= 0)
        if len(cells) == 0:
            return -1, -1
        d = m.geom.hex_distance(cells, cell)
        k = int(np.argmin(d))  # cells ascend: the lowest id among the nearest
        return int(occ[cells[k]]), int(d[k])

    def direction(self, src: int, dst: int) -> str:
        pos = self.model.geom.positions
        dx, dy = pos[dst] - pos[src]
        return DIRECTIONS[int(np.round(np.arctan2(dy, dx) / (np.pi / 4))) % 8]

    def relative(self, cell: int, at: int | None, sp, before: bool = False, exclude=()) -> str:
        """``cell`` told relative to the nearest named place (see the module docstring)."""
        return self._relative(cell, at, sp, before, exclude)[1]

    def _relative(self, cell: int, at: int | None, sp, before: bool = False, exclude=()) -> tuple[str, str]:
        """(form used, text) of ``relative``."""
        m = self.model
        p = sp.data["places"]
        cfg = m.config.reports
        year = int(m.events.year[at]) if at is not None and at < len(m.events.year) else max(m.year - 1, 0)
        sid, d = self.nearest(cell, at, before, exclude)
        if 0 <= d <= cfg.place_far_cells:
            from deluge.data.events import entity

            src = int(m.settlements.cell[sid]) if at is None else self._cell_of(sid, at, before)
            form = "near" if d <= cfg.place_near_cells else "far"
            return form, p[form].format(direction=self.direction(src, cell), name=sp.entity(entity("settlement", sid)))
        if m.elevation_flat[cell] < float(m.sea_level_series[min(max(year, 0), len(m.sea_level_series) - 1)]):
            return "sea", p["sea"]
        labels = self._land(year)
        occ = self.occupants(np.arange(m.geom.n_cells), at, before)
        if len(exclude):
            occ[np.isin(occ, np.asarray(list(exclude), dtype=np.int64))] = -1
        same = np.flatnonzero((occ >= 0) & (labels == labels[cell]))
        if len(same):
            from deluge.data.events import entity

            k = int(np.argmin(m.geom.hex_distance(same, cell)))
            return "land", p["land"].format(name=sp.entity(entity("settlement", int(occ[same[k]]))))
        return "empty_land", p["empty_land"]

    def _cell_of(self, sid: int, at: int, before: bool) -> int:
        """The cell settlement ``sid`` stood on when event ``at`` was logged."""
        self._index()
        log = self.model._occupancy
        stamp = np.frombuffer(log.stamp, dtype=np.int64)
        sids = np.frombuffer(log.sid, dtype=np.int64)
        hit = np.flatnonzero((sids == sid) & ((stamp < at) if before else (stamp <= at)))
        return int(np.frombuffer(log.cell, dtype=np.int64)[hit[-1]]) if len(hit) else int(self.model.settlements.cell[sid])

    def place(self, cell: int, sp, at: int | None = None) -> str:
        """A cell's name when event ``at`` was logged (None: now), or where it lies."""
        if cell < 0:
            return sp.data["places"]["none"]
        sid = self.occupant(cell, at)
        if sid >= 0:
            from deluge.data.events import entity

            return sp.data["places"]["cell"].format(name=sp.entity(entity("settlement", sid)))
        return self.relative(cell, at, sp)

    def event_place(self, i: int, sp) -> str:
        """Where event ``i`` happened, as the people then could say it: the cell's name, unless
        one of the event's own settlements arrived there with it or with the event it is part
        of (a founding, a move, its sea crossing); then the site as it lay before, relative to
        the other named places."""
        ev = self.model.events
        cell = int(ev.place[i])
        if cell < 0:
            return sp.data["places"]["none"]
        arrival = int(ev.link[i]) if ev.link[i] >= 0 else i
        if ev.action[arrival] not in ARRIVALS:
            return self.place(cell, sp, i)
        now, then = self.occupant(cell, i), self.occupant(cell, arrival, before=True)
        mine = {loc for kind, loc in map(decode, ev.actors[i]) if kind == "settlement"}
        if now >= 0 and now != then and now in mine:
            return self.relative(cell, arrival, sp, before=True, exclude=(now,))
        return self.place(cell, sp, i)

    def region(self, region: int, sp) -> str:
        """A memory region, named after its most important named place now (module docstring)."""
        m = self.model
        r = m.config.memory.region_cells
        cols = m.memory.region_cols
        ry, rx = divmod(int(region), cols)
        x, y = m.geom.coords
        inside = np.flatnonzero((x // r == rx) & (y // r == ry))
        occ = m.occupant[inside]
        occ = occ[occ >= 0]
        p = sp.data["places"]
        if len(occ):
            from deluge.data.events import entity

            pop = m.settlements.population[occ]
            best = int(occ[np.lexsort((occ, -pop))[0]])
            return p["region"].format(name=sp.entity(entity("settlement", best)))
        cx = min(rx * r + r // 2, m.geom.width - 1)
        cy = min(ry * r + r // 2, m.geom.height - 1)
        form, where = self._relative(m.geom.cid(cx, cy), None, sp)
        return p["region_unnamed"].format(where=where) if form in ("near", "far") else where

    def name_of(self, cell: int, sp, at: int | None = None) -> str | None:
        """The bare name of a cell (its settlement's), None while nobody stands on it."""
        sid = self.occupant(cell, at) if cell >= 0 else -1
        if sid < 0:
            return None
        from deluge.data.events import entity

        return sp.entity(entity("settlement", sid))


def gazetteer(model) -> Gazetteer:
    """The model's gazetteer (kept off its pickled state, like the names)."""
    g = model.__dict__.get("_gazetteer")
    if g is None:
        g = model.__dict__["_gazetteer"] = Gazetteer(model)
    return g
