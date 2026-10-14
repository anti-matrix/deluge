import random

import numpy as np
import pytest
from mesa.discrete_space import HexGrid

from deluge.world.grid import SQRT3, HexGeometry, build_mesa_grid


@pytest.mark.parametrize("w,h,torus", [(100, 60, False), (7, 5, False), (6, 4, True)])
def test_neighbors_match_mesa_hexgrid(w, h, torus):
    geom = HexGeometry(w, h, torus)
    grid = HexGrid((w, h), torus=torus, random=random.Random(0))
    for (x, y), cell in grid._cells.items():
        mesa_nbrs = {geom.cid(*n.coordinate) for n in cell.neighborhood}
        ours = set(geom.neighbor_lists[geom.cid(x, y)])
        assert ours == mesa_nbrs, (x, y)


def test_neighbors_are_adjacent_in_drawn_geometry():
    geom = HexGeometry(30, 20)
    pos = geom.positions
    for c, nbrs in enumerate(geom.neighbor_lists):
        d = np.linalg.norm(pos[list(nbrs)] - pos[c], axis=1)
        np.testing.assert_allclose(d, SQRT3, rtol=1e-12)


def test_flat_ids_follow_field_layout():
    geom = HexGeometry(5, 3)
    field = np.arange(15).reshape(geom.shape)
    for x in range(5):
        for y in range(3):
            assert field.ravel()[geom.cid(x, y)] == field[x, y]
            assert geom.xy(geom.cid(x, y)) == (x, y)


def test_components_sorted_by_size():
    geom = HexGeometry(10, 4)
    flags = np.zeros(geom.n_cells, dtype=bool)
    flags[[geom.cid(0, 0)]] = True
    for x in range(5, 9):
        flags[geom.cid(x, 2)] = True
    labels = geom.components(flags)
    assert labels[geom.cid(6, 2)] == 0
    assert labels[geom.cid(0, 0)] == 1
    assert (labels[~flags] == -1).all()


def test_latitude_span():
    geom = HexGeometry(4, 60)
    lat = geom.latitude(-60.0, 70.0)
    assert lat.min() == pytest.approx(-60.0)
    assert lat.max() == pytest.approx(70.0)


def test_mesa_grid_property_layers_match_fields():
    geom = HexGeometry(8, 6)
    elev = np.arange(48, dtype=float).reshape(geom.shape)
    grid = build_mesa_grid(geom, {"elevation": elev}, random=random.Random(0))
    for (x, y), cell in grid._cells.items():
        assert cell.elevation == elev[x, y]
