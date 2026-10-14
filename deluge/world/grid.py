"""Hex grid geometry and the cell field registry.

Layout convention (matches Mesa 3.5.1's ``HexGrid`` connectivity and its hex-mesh drawer):

- A cell is ``(x, y)`` with ``0 <= x < width`` (column) and ``0 <= y < height`` (row).
- Fields are NumPy arrays of shape ``(width, height)``, like Mesa ``PropertyLayer.data``.
- The flat cell id is ``x * height + y`` (C order of a ``(width, height)`` array), so
  ``field.ravel()[cid]`` is ``field[x, y]``.
- Pointy-top hexes of size 1: center ``(sqrt(3) * (x + 0.5 * (y even)), 1.5 * y)``.
  Even rows are shifted right. This is the layout Mesa's connectivity implies; Mesa's
  own ``cell.position`` uses the opposite shift and is not used here.
- Row 0 is the southern edge; latitude rises with ``y``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from mesa.discrete_space import HexGrid, PropertyLayer

# Neighbor offsets (dx, dy), copied from mesa.discrete_space.HexGrid._connect_cells_2d.
_ODD_ROW_OFFSETS = ((-1, -1), (0, -1), (-1, 0), (1, 0), (-1, 1), (0, 1))
_EVEN_ROW_OFFSETS = ((0, -1), (1, -1), (-1, 0), (1, 0), (0, 1), (1, 1))

SQRT3 = float(np.sqrt(3.0))


@dataclass(frozen=True)
class HexGeometry:
    width: int
    height: int
    torus: bool = False

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    @property
    def shape(self) -> tuple[int, int]:
        return (self.width, self.height)

    def cid(self, x: int, y: int) -> int:
        return x * self.height + y

    def xy(self, cid: int) -> tuple[int, int]:
        return divmod(cid, self.height)

    @cached_property
    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """(x, y) column and row index of every flat cell id."""
        cids = np.arange(self.n_cells)
        return cids // self.height, cids % self.height

    @cached_property
    def positions(self) -> np.ndarray:
        """(n_cells, 2) hex centers in hex units (size 1)."""
        x, y = self.coords
        px = SQRT3 * (x + 0.5 * (y % 2 == 0))
        py = 1.5 * y
        return np.stack([px, py], axis=1)

    @cached_property
    def neighbors(self) -> np.ndarray:
        """(n_cells, 6) neighbor ids, -1 where the neighbor is off the map."""
        out = np.full((self.n_cells, 6), -1, dtype=np.int64)
        for cid in range(self.n_cells):
            x, y = self.xy(cid)
            offsets = _ODD_ROW_OFFSETS if y % 2 else _EVEN_ROW_OFFSETS
            for k, (dx, dy) in enumerate(offsets):
                nx, ny = x + dx, y + dy
                if self.torus:
                    nx, ny = nx % self.width, ny % self.height
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    out[cid, k] = self.cid(nx, ny)
        out.setflags(write=False)
        return out

    @cached_property
    def neighbor_lists(self) -> tuple[tuple[int, ...], ...]:
        """Neighbor ids as plain tuples, for fast pure-Python loops."""
        return tuple(tuple(int(n) for n in row if n >= 0) for row in self.neighbors)

    @cached_property
    def neighbor_mask(self) -> np.ndarray:
        return self.neighbors >= 0

    def neighbor_any(self, flags: np.ndarray) -> np.ndarray:
        """For a flat bool array, True where any neighbor is True."""
        padded = np.append(flags, False)
        return padded[self.neighbors].any(axis=1)

    def neighbor_mean(self, values: np.ndarray, include_self: bool = True) -> np.ndarray:
        """Mean over existing neighbors (and the cell itself if include_self) of a flat array."""
        padded = np.append(values, 0.0)
        total = padded[self.neighbors].sum(axis=1)
        count = self.neighbor_mask.sum(axis=1).astype(float)
        if include_self:
            total = total + values
            count = count + 1.0
        return total / count

    def latitude(self, lat_south: float, lat_north: float) -> np.ndarray:
        """Flat array of latitudes in degrees, linear in row index."""
        _, y = self.coords
        return lat_south + (lat_north - lat_south) * y / max(self.height - 1, 1)

    def components(self, flags: np.ndarray) -> np.ndarray:
        """Connected-component label per cell over True cells (-1 elsewhere), labels by size desc."""
        labels = np.full(self.n_cells, -1, dtype=np.int64)
        nbrs = self.neighbor_lists
        sizes = []
        for start in np.flatnonzero(flags):
            if labels[start] >= 0:
                continue
            label = len(sizes)
            labels[start] = label
            queue = deque([start])
            size = 0
            while queue:
                c = queue.popleft()
                size += 1
                for n in nbrs[c]:
                    if flags[n] and labels[n] < 0:
                        labels[n] = label
                        queue.append(n)
            sizes.append(size)
        if not sizes:
            return labels
        order = np.argsort(-np.asarray(sizes, dtype=np.int64), kind="stable")
        remap = np.empty(len(sizes), dtype=np.int64)
        remap[order] = np.arange(len(sizes))
        out = labels.copy()
        out[labels >= 0] = remap[labels[labels >= 0]]
        return out

    def distance_from(self, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Multi-source BFS over the hex graph.

        Returns (steps, nearest): hex steps to the nearest source cell and that source's id,
        both -1 where no source is reachable.
        """
        steps = np.full(self.n_cells, -1, dtype=np.int64)
        nearest = np.full(self.n_cells, -1, dtype=np.int64)
        src = np.flatnonzero(sources)
        steps[src] = 0
        nearest[src] = src
        st = steps.tolist()
        ne = nearest.tolist()
        nbrs = self.neighbor_lists
        queue = deque(src.tolist())
        while queue:
            c = queue.popleft()
            d = st[c] + 1
            s = ne[c]
            for n in nbrs[c]:
                if st[n] < 0:
                    st[n] = d
                    ne[n] = s
                    queue.append(n)
        return np.asarray(st, dtype=np.int64), np.asarray(ne, dtype=np.int64)

    def hex_vertices(self) -> np.ndarray:
        """(n_cells, 6, 2) polygon vertices for drawing, pointy-top hexes of size 1."""
        angles = np.deg2rad(np.array([90, 30, -30, -90, -150, 150], dtype=float))
        ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return self.positions[:, None, :] + ring[None, :, :]


def hex_unit_km(height: int, lat_south: float, lat_north: float, km_per_degree: float) -> float:
    """Kilometres per hex unit (hex circumradius), from the latitude span over the rows.
    Rows are 1.5 units apart; neighbouring centers are sqrt(3) units apart."""
    km_per_row = (lat_north - lat_south) * km_per_degree / max(height - 1, 1)
    return km_per_row / 1.5


def hex_area_km2(unit_km: float) -> float:
    return 1.5 * SQRT3 * unit_km * unit_km


def build_mesa_grid(geom: HexGeometry, fields: dict[str, np.ndarray], random=None) -> HexGrid:
    """A Mesa HexGrid with one PropertyLayer per field (static copies of the arrays)."""
    grid = HexGrid((geom.width, geom.height), torus=geom.torus, random=random)
    for name, data in fields.items():
        grid.add_property_layer(PropertyLayer.from_data(name, np.asarray(data).reshape(geom.shape)))
    return grid


def as_field(geom: HexGeometry, flat: np.ndarray) -> np.ndarray:
    """Flat per-cell array -> (width, height) field view."""
    return np.asarray(flat).reshape(geom.shape)
