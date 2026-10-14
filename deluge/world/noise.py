"""Vectorized OpenSimplex 2D noise.

A NumPy port of ``opensimplex``'s ``_noise2`` that evaluates many arbitrary points at
once. Without numba (an optional extra, round 11) the library runs one point at a time
in pure Python, about 25 us each; hex-cell centers are not a regular grid, so
``noise2array`` would also evaluate twice the points needed. The permutation table,
gradients and constants come from the installed library, and
``tests/test_world_noise.py`` checks that values match ``OpenSimplex.noise2``.
"""

from __future__ import annotations

import numpy as np
import opensimplex
from opensimplex.constants import GRADIENTS2, NORM_CONSTANT2, SQUISH_CONSTANT2, STRETCH_CONSTANT2

_SQ = SQUISH_CONSTANT2
_GRAD = np.asarray(GRADIENTS2, dtype=float)


def permutation(seed: int) -> np.ndarray:
    return np.asarray(opensimplex.OpenSimplex(seed)._perm, dtype=np.int64)


def _contribution(perm: np.ndarray, xsb: np.ndarray, ysb: np.ndarray, dx: np.ndarray,
                  dy: np.ndarray) -> np.ndarray:
    attn = 2.0 - dx * dx - dy * dy
    index = perm[(perm[xsb & 0xFF] + ysb) & 0xFF] & 0x0E
    ext = _GRAD[index] * dx + _GRAD[index + 1] * dy
    attn = np.where(attn > 0.0, attn, 0.0)
    attn = attn * attn
    return attn * attn * ext


def noise2(perm: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OpenSimplex noise at points (x[i], y[i]); same values as ``OpenSimplex.noise2``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stretch = (x + y) * STRETCH_CONSTANT2
    xs = x + stretch
    ys = y + stretch
    xsb = np.floor(xs).astype(np.int64)
    ysb = np.floor(ys).astype(np.int64)
    squish = (xsb + ysb) * _SQ
    xb = xsb + squish
    yb = ysb + squish
    xins = xs - xsb
    yins = ys - ysb
    in_sum = xins + yins
    dx0 = x - xb
    dy0 = y - yb

    value = _contribution(perm, xsb + 1, ysb, dx0 - 1 - _SQ, dy0 - _SQ)
    value += _contribution(perm, xsb, ysb + 1, dx0 - _SQ, dy0 - 1 - _SQ)

    lower = in_sum <= 1
    x_gt_y = xins > yins

    # Lower triangle at (0,0).
    zl = 1 - in_sum
    l_near = (zl > xins) | (zl > yins)
    # Upper triangle at (1,1).
    zu = 2 - in_sum
    u_near = (zu < xins) | (zu < yins)

    xsv = np.where(
        lower,
        np.where(l_near, np.where(x_gt_y, xsb + 1, xsb - 1), xsb + 1),
        np.where(u_near, np.where(x_gt_y, xsb + 2, xsb), xsb),
    )
    ysv = np.where(
        lower,
        np.where(l_near, np.where(x_gt_y, ysb - 1, ysb + 1), ysb + 1),
        np.where(u_near, np.where(x_gt_y, ysb, ysb + 2), ysb),
    )
    dxe = np.where(
        lower,
        np.where(l_near, np.where(x_gt_y, dx0 - 1, dx0 + 1), dx0 - 1 - 2 * _SQ),
        np.where(u_near, np.where(x_gt_y, dx0 - 2 - 2 * _SQ, dx0 - 2 * _SQ), dx0),
    )
    dye = np.where(
        lower,
        np.where(l_near, np.where(x_gt_y, dy0 + 1, dy0 - 1), dy0 - 1 - 2 * _SQ),
        np.where(u_near, np.where(x_gt_y, dy0 - 2 * _SQ, dy0 - 2 - 2 * _SQ), dy0),
    )

    # The (0,0) or (1,1) vertex.
    xsb0 = np.where(lower, xsb, xsb + 1)
    ysb0 = np.where(lower, ysb, ysb + 1)
    dx0 = np.where(lower, dx0, dx0 - 1 - 2 * _SQ)
    dy0 = np.where(lower, dy0, dy0 - 1 - 2 * _SQ)
    value += _contribution(perm, xsb0, ysb0, dx0, dy0)
    value += _contribution(perm, xsv, ysv, dxe, dye)
    return value / NORM_CONSTANT2


# --- point-cloud helpers (uv: (n, 2) coordinates) -------------------------------------------


def simplex(uv: np.ndarray, rng: np.random.Generator, frequency: float) -> np.ndarray:
    """One octave at a fresh seed and offset drawn from ``rng``."""
    seed = int(rng.integers(0, 2**31 - 1))
    offset = rng.uniform(0.0, 1000.0, size=2)
    return noise2(permutation(seed), uv[:, 0] * frequency + offset[0], uv[:, 1] * frequency + offset[1])


def fbm(uv: np.ndarray, rng: np.random.Generator, frequency: float, octaves: int,
        persistence: float, lacunarity: float, ridged: bool = False,
        sharpness: float = 2.0) -> np.ndarray:
    """Fractal noise normalized by total amplitude: roughly [-1, 1], or [0, 1] when
    ``ridged`` ((1 - |n|) ** sharpness per octave, crests read as ridges)."""
    perm = permutation(int(rng.integers(0, 2**31 - 1)))
    offsets = rng.uniform(0.0, 1000.0, size=(octaves, 2))
    total = np.zeros(len(uv))
    amp, freq, norm = 1.0, frequency, 0.0
    for o in range(octaves):
        n = noise2(perm, uv[:, 0] * freq + offsets[o, 0], uv[:, 1] * freq + offsets[o, 1])
        total += amp * ((1.0 - np.abs(n)) ** sharpness if ridged else n)
        norm += amp
        amp *= persistence
        freq *= lacunarity
    return total / norm


def warp(uv: np.ndarray, rng: np.random.Generator, frequency: float, octaves: int,
         persistence: float, lacunarity: float, strength: float) -> np.ndarray:
    """Domain-warped coordinates: uv displaced by a fractal noise vector field."""
    dx = fbm(uv, rng, frequency, octaves, persistence, lacunarity)
    dy = fbm(uv, rng, frequency, octaves, persistence, lacunarity)
    return uv + strength * np.stack([dx, dy], axis=1)
