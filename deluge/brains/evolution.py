"""Brain inheritance and mutation (spec section 10).

- Founding polities: independent random genomes from the ``brains`` stream (history
  seed): every weight and bias ~ N(0, ``init_sigma`` / sqrt(fan in)), except the weights
  that read appearance or dialect (``marker_mask``), drawn with ``marker_init_sigma``
  instead (0 by default: they start at zero and only mutation moves them, either way).
- Splits and fragmentation: the child's genome is the parent's plus N(0,
  ``mutation_sigma``) per weight and, with ``jump_chance`` per weight, a larger jump of sd
  ``jump_sigma``. With ``frozen`` the child carries an exact copy (control condition):
  every brain is then its founder's.
- Mergers: the larger partner keeps its brain; with
  ``merge_crossover_chance`` it becomes a uniform crossover with the smaller one's.
- Conquest: the conquered polity ends and its brain with it (nothing to do here).
"""

from __future__ import annotations

import numpy as np

from deluge.config import BrainsConfig


def founder_genome(fan_in: np.ndarray, marker_mask: np.ndarray, cfg: BrainsConfig,
                   rng: np.random.Generator) -> np.ndarray:
    sigma = np.where(marker_mask, cfg.marker_init_sigma, cfg.init_sigma) / np.sqrt(fan_in)
    return (rng.standard_normal(len(fan_in)) * sigma).astype(np.float32)


def mutate(genome: np.ndarray, cfg: BrainsConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.frozen:
        return genome.copy()
    n = len(genome)
    noise = rng.normal(0.0, cfg.mutation_sigma, n)
    jump = rng.random(n) < cfg.jump_chance
    noise[jump] += rng.normal(0.0, cfg.jump_sigma, int(jump.sum()))
    return (genome + noise).astype(np.float32)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform crossover: each weight from ``a`` or ``b`` with equal chance."""
    return np.where(rng.random(len(a)) < 0.5, b, a).astype(np.float32)
