"""Simple plate tectonics for relief (spec section 5, step 2).

Plates are warped Voronoi regions around random seeds, each with a random motion vector
and a continental bias. Boundaries are classified by the relative motion of the two
plates along the boundary normal:

- convergent: a mountain range along the boundary (peaks and passes from noise), and a
  plateau behind it on the overriding side (the plate moving less toward the boundary);
- divergent: a narrow rift valley inside a broad lowland basin;
- otherwise (transform): no relief.

All outputs are in relief units, added to the continent relief before elevation is
mapped to metres, so ranges can also raise island arcs out of the sea and rifts can sink
land into inland seas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deluge.config import TectonicsConfig
from deluge.world import noise
from deluge.world.grid import SQRT3, HexGeometry


@dataclass(frozen=True)
class Plates:
    plate_id: np.ndarray  # flat, plate of each cell
    velocity: np.ndarray  # (n_plates, 2) motion vectors in map units
    bias: np.ndarray  # (n_plates,) continental bias in [-1, 1]
    convergence: np.ndarray  # flat, mean approach speed across the boundary (nan off-boundary)
    convergent: np.ndarray  # flat bool
    divergent: np.ndarray  # flat bool
    strength: np.ndarray  # flat, |convergence| / speed_max clipped to [0, 1]
    overriding: np.ndarray  # flat bool, boundary cell lies on the overriding plate


def normalized_positions(geom: HexGeometry) -> np.ndarray:
    """Hex centers scaled so the map spans u in [0, 1]; v keeps the aspect ratio."""
    return geom.positions / (SQRT3 * geom.width)


def map_aspect(geom: HexGeometry) -> float:
    return (1.5 * geom.height) / (SQRT3 * geom.width)


def make_plates(geom: HexGeometry, rng: np.random.Generator, cfg: TectonicsConfig) -> Plates:
    uv = normalized_positions(geom)
    n = int(rng.integers(cfg.plates_min, cfg.plates_max + 1))
    seeds = rng.uniform([0.0, 0.0], [1.0, map_aspect(geom)], size=(n, 2))
    angle = rng.uniform(0.0, 2 * np.pi, size=n)
    speed = rng.uniform(cfg.speed_min, cfg.speed_max, size=n)
    velocity = np.stack([np.cos(angle), np.sin(angle)], axis=1) * speed[:, None]
    bias = rng.uniform(-1.0, 1.0, size=n)

    warped = noise.warp(uv, rng, cfg.boundary_warp_frequency, cfg.boundary_warp_octaves,
                        cfg.noise_persistence, cfg.noise_lacunarity, cfg.boundary_warp_strength)
    d2 = ((warped[:, None, :] - seeds[None, :, :]) ** 2).sum(axis=2)
    plate_id = np.argmin(d2, axis=1)

    nb = geom.neighbors
    valid = nb >= 0
    other = np.where(valid, plate_id[np.where(valid, nb, 0)], -1)
    differ = valid & (other != plate_id[:, None])
    # Unit direction from each cell toward each neighbour.
    dirs = (geom.positions[np.where(valid, nb, 0)] - geom.positions[:, None, :]) / SQRT3
    v_own = velocity[plate_id]  # (N, 2)
    v_other = velocity[np.where(differ, other, 0)]  # (N, 6, 2)
    rel = v_own[:, None, :] - v_other
    approach = (rel * dirs).sum(axis=2)  # >0: closing across that edge
    count = differ.sum(axis=1)
    on_boundary = count > 0
    convergence = np.full(geom.n_cells, np.nan)
    convergence[on_boundary] = (np.where(differ, approach, 0.0).sum(axis=1) / np.maximum(count, 1))[on_boundary]

    # Overriding side: the plate whose own velocity carries it less toward the boundary.
    normal = np.where(differ[:, :, None], dirs, 0.0).sum(axis=1)
    norm = np.linalg.norm(normal, axis=1)
    normal = normal / np.where(norm > 0, norm, 1.0)[:, None]
    first_other = other[np.arange(geom.n_cells), np.argmax(differ, axis=1)]
    own_push = (v_own * normal).sum(axis=1)
    other_push = -(velocity[np.maximum(first_other, 0)] * normal).sum(axis=1)
    overriding = on_boundary & (own_push < other_push)

    conv = np.nan_to_num(convergence, nan=0.0)
    return Plates(
        plate_id=plate_id,
        velocity=velocity,
        bias=bias,
        convergence=convergence,
        convergent=on_boundary & (conv > cfg.convergence_threshold),
        divergent=on_boundary & (conv < -cfg.convergence_threshold),
        strength=np.clip(np.abs(conv) / cfg.speed_max, 0.0, 1.0),
        overriding=overriding,
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tectonic_relief(geom: HexGeometry, plates: Plates, rng: np.random.Generator,
                    cfg: TectonicsConfig, unit_km: float) -> np.ndarray:
    """Relief from boundaries: ranges and plateaus at convergent, rifts and basins at
    divergent boundaries."""
    step_km = SQRT3 * unit_km
    uv = normalized_positions(geom)
    relief = np.zeros(geom.n_cells)

    if plates.convergent.any():
        steps, src = geom.distance_from(plates.convergent)
        d = steps * step_km
        s = plates.strength[src]
        peaks = (1.0 - cfg.peak_noise_weight) + cfg.peak_noise_weight * noise.fbm(
            uv, rng, cfg.peak_noise_frequency, cfg.peak_noise_octaves, cfg.noise_persistence,
            cfg.noise_lacunarity, ridged=True, sharpness=cfg.peak_noise_sharpness)
        relief += cfg.mountain_height * s * np.exp(-((d / cfg.mountain_width_km) ** 2)) * peaks
        behind = (plates.plate_id == plates.plate_id[src]) & plates.overriding[src]
        plateau = 1.0 - _smoothstep((d - cfg.plateau_width_km) / cfg.plateau_falloff_km)
        relief += np.where(behind, cfg.plateau_height * s * plateau, 0.0)

    if plates.divergent.any():
        steps, src = geom.distance_from(plates.divergent)
        d = steps * step_km
        s = plates.strength[src]
        # Same noise profile as the peaks, so rifts break into separate basins.
        segments = (1.0 - cfg.peak_noise_weight) + cfg.peak_noise_weight * noise.fbm(
            uv, rng, cfg.peak_noise_frequency, cfg.peak_noise_octaves, cfg.noise_persistence,
            cfg.noise_lacunarity, ridged=True, sharpness=cfg.peak_noise_sharpness)
        relief -= cfg.rift_depth * s * np.exp(-((d / cfg.rift_width_km) ** 2)) * segments
        relief -= cfg.basin_depth * s * np.exp(-((d / cfg.basin_width_km) ** 2))
    return relief
