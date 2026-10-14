"""Single seeded generator with named child streams.

All randomness in a run flows from one root ``numpy.random.SeedSequence``. Each system
draws from its own child stream, spawned from the root with a spawn key derived from
the stream's name (the mechanism ``SeedSequence.spawn`` uses internally). Because the
key comes from the name and not from spawn order, adding a stream or drawing more from
one system never shifts the numbers another system sees.
"""

from __future__ import annotations

import zlib

import numpy as np


def _name_key(name: str) -> int:
    # crc32 is stable across processes and platforms, unlike hash().
    return zlib.crc32(name.encode("utf-8"))


class RngStreams:
    def __init__(self, seed: int, names: tuple[str, ...] | list[str]):
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stream names: {names}")
        keys = [_name_key(n) for n in names]
        if len(set(keys)) != len(keys):
            raise ValueError("stream name key collision; rename a stream")
        self.seed = seed
        self.root = np.random.SeedSequence(seed)
        self._streams: dict[str, np.random.Generator] = {
            name: np.random.Generator(np.random.PCG64(self._child(name))) for name in names
        }

    def _child(self, name: str) -> np.random.SeedSequence:
        return np.random.SeedSequence(
            self.root.entropy, spawn_key=self.root.spawn_key + (_name_key(name),)
        )

    def __getitem__(self, name: str) -> np.random.Generator:
        return self._streams[name]

    def __contains__(self, name: str) -> bool:
        return name in self._streams

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._streams)

    def state(self) -> dict[str, dict]:
        """Bit-generator state of every stream, for checkpoints and determinism checks."""
        return {name: g.bit_generator.state for name, g in self._streams.items()}
