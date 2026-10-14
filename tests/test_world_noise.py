import numpy as np
import opensimplex

from deluge.world import noise


def test_vectorized_noise_matches_opensimplex():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50.0, 50.0, size=(2000, 2))
    for seed in (0, 1, 12345):
        ref = opensimplex.OpenSimplex(seed)
        expected = np.array([ref.noise2(float(x), float(y)) for x, y in pts])
        got = noise.noise2(noise.permutation(seed), pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


def test_noise_on_integer_lattice_and_negatives():
    xs, ys = np.meshgrid(np.arange(-5, 6, dtype=float), np.arange(-5, 6, dtype=float))
    ref = opensimplex.OpenSimplex(7)
    expected = np.array([ref.noise2(float(x), float(y)) for x, y in zip(xs.ravel(), ys.ravel())])
    got = noise.noise2(noise.permutation(7), xs.ravel(), ys.ravel())
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
