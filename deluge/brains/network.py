"""Fixed-topology MLP in NumPy, batched across polities (spec section 10).

Each polity's brain is one flat genome (float32): inputs -> ``hidden`` tanh units ->
the scalar outputs (``OUTPUTS``, sigmoid, 0..1), plus two per-neighbour heads: the target
head (war) and, since Round 14, the alliance head. Each turns the hidden state into one
weight per feature of its own (``Q h + q0``; ``QA h + qa0``); a candidate's score is then
sigmoid(features . weights), so what a polity looks for in a neighbour depends on its own
state, and any feature (appearance and dialect distance included) can count for or
against, or not at all.

``forward`` evaluates every brain in one batched pass (einsum over stacked weights).

``BrainTable`` holds, per polity id (ids are never reused): the genome, set once at birth
(``brains/evolution.py``) and kept after the polity ends, so the table is also the archive
of every brain born (spec section 12; a merger's crossover files
the replaced genome in ``replaced``); this year's outputs and target weights; the
recurrent state (with ``brains.recurrent_units`` > 0); and the settlement counts of the
last years (the territory-growth input).

Outputs that drive a system: expand (founding daughter settlements), aggression, campaign
(campaign rather than raid), tribute, monuments, farming and herding (subsistence shift),
newcomers (openness to newcomers), core_migration, alliance_seeking (proposing alliances,
``systems/alliances.py``). Trade openness is computed and logged; no system reads it yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

OUTPUTS = ("expand", "aggression", "campaign", "trade_openness", "alliance_seeking", "tribute", "monuments",
           "farming", "herding", "newcomers", "core_migration")
OUT = {name: i for i, name in enumerate(OUTPUTS)}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free logistic


@dataclass(frozen=True)
class Layout:
    """Where each weight sits in the flat genome."""

    inputs: tuple[str, ...]
    hidden: int
    outputs: tuple[str, ...]
    target: tuple[str, ...]
    alliance: tuple[str, ...] = ()

    @property
    def n_in(self) -> int:
        return len(self.inputs)

    def _blocks(self) -> list[tuple[str, tuple[int, ...], int]]:
        """(name, shape, fan in) of each block, in genome order."""
        i, h, o, f, fa = self.n_in, self.hidden, len(self.outputs), len(self.target), len(self.alliance)
        return [("w1", (i, h), i), ("b1", (h,), i), ("w2", (h, o), h), ("b2", (o,), h),
                ("q", (h, f), h), ("q0", (f,), h), ("qa", (h, fa), h), ("qa0", (fa,), h)]

    @property
    def size(self) -> int:
        return sum(int(np.prod(shape)) for _, shape, _ in self._blocks())

    def slices(self) -> dict[str, tuple[slice, tuple[int, ...]]]:
        out, at = {}, 0
        for name, shape, _ in self._blocks():
            n = int(np.prod(shape))
            out[name] = (slice(at, at + n), shape)
            at += n
        return out

    def fan_in(self) -> np.ndarray:
        """Fan in of the layer each genome entry belongs to."""
        return np.concatenate([np.full(int(np.prod(shape)), fan, dtype=float) for _, shape, fan in self._blocks()])

    def mask(self, inputs: tuple[str, ...] = (), target: tuple[str, ...] = (),
             alliance: tuple[str, ...] = ()) -> np.ndarray:
        """Genome entries that read the named inputs (their rows of w1) or weigh the named
        target or alliance features (their columns of q / qa and entries of q0 / qa0)."""
        m = np.zeros(self.size, dtype=bool)
        s = self.slices()
        idx = np.arange(self.size)
        w1 = idx[s["w1"][0]].reshape(s["w1"][1])
        q, q0 = idx[s["q"][0]].reshape(s["q"][1]), idx[s["q0"][0]]
        for name in inputs:
            m[w1[self.inputs.index(name)]] = True
        for name in target:
            j = self.target.index(name)
            m[q[:, j]] = True
            m[q0[j]] = True
        qa, qa0 = idx[s["qa"][0]].reshape(s["qa"][1]), idx[s["qa0"][0]]
        for name in alliance:
            j = self.alliance.index(name)
            m[qa[:, j]] = True
            m[qa0[j]] = True
        return m

    def unpack(self, genomes: np.ndarray) -> dict[str, np.ndarray]:
        """Views of a (n, size) genome stack as (n, *shape) weight arrays."""
        n = genomes.shape[0]
        return {name: genomes[:, sl].reshape((n,) + shape) for name, (sl, shape) in self.slices().items()}

    def to_dict(self) -> dict:
        return {"inputs": list(self.inputs), "hidden": self.hidden, "outputs": list(self.outputs),
                "target": list(self.target), "alliance": list(self.alliance),
                "blocks": {name: {"start": sl.start, "stop": sl.stop, "shape": list(shape)}
                           for name, (sl, shape) in self.slices().items()}}


def forward(layout: Layout, genomes: np.ndarray, x: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every brain at once: (outputs (n, O) in 0..1, target weights (n, F), alliance weights
    (n, FA), hidden (n, H))."""
    w = layout.unpack(genomes.astype(np.float64))
    h = np.tanh(np.einsum("ni,nih->nh", x, w["w1"]) + w["b1"])
    out = sigmoid(np.einsum("nh,nho->no", h, w["w2"]) + w["b2"])
    target = np.einsum("nh,nhf->nf", h, w["q"]) + w["q0"]
    alliance = np.einsum("nh,nhf->nf", h, w["qa"]) + w["qa0"]
    return out, target, alliance, h


def target_scores(features: np.ndarray, weights: np.ndarray, shift=0.0) -> np.ndarray:
    """Score of each candidate, 0..1: sigmoid(features (n, F) . its polity's weights (n, F) +
    ``shift``) (the appearance prior, Round 14, shifts the sum: it multiplies the odds)."""
    return sigmoid((features * weights).sum(axis=1) + shift)


class BrainTable:
    def __init__(self, layout: Layout, recurrent_units: int, growth_window: int, capacity: int = 64):
        self.layout = layout
        self.genome = np.zeros((capacity, layout.size), dtype=np.float32)
        self.output = np.zeros((capacity, len(layout.outputs)))
        self.target_w = np.zeros((capacity, len(layout.target)))
        self.alliance_w = np.zeros((capacity, len(layout.alliance)))
        self.state = np.zeros((capacity, recurrent_units))
        self.settlement_history = np.full((capacity, growth_window), -1, dtype=np.int64)
        self.replaced: dict[int, list[tuple[int, np.ndarray]]] = {}  # pid -> [(year, genome replaced)]

    COLUMNS = ("genome", "output", "target_w", "alliance_w", "state", "settlement_history")

    def ensure(self, pid: int) -> None:
        while pid >= len(self.genome):
            for name in self.COLUMNS:
                old = getattr(self, name)
                fill = -1 if old.dtype.kind == "i" else 0
                new = np.full((2 * old.shape[0],) + old.shape[1:], fill, dtype=old.dtype)
                new[: old.shape[0]] = old
                setattr(self, name, new)

    def born(self, pid: int, genome: np.ndarray, output: np.ndarray, target_w: np.ndarray,
             state: np.ndarray, alliance_w: np.ndarray | None = None) -> None:
        self.ensure(pid)
        self.genome[pid] = genome
        self.output[pid] = output
        self.target_w[pid] = target_w
        self.alliance_w[pid] = 0.0 if alliance_w is None else alliance_w
        self.state[pid] = state
        self.settlement_history[pid] = -1

    def replace(self, pid: int, genome: np.ndarray, year: int) -> None:
        """A living polity's brain changes (a merger's crossover); the old one is filed."""
        self.replaced.setdefault(pid, []).append((year, self.genome[pid].copy()))
        self.genome[pid] = genome

    def birth_genomes(self, n: int) -> np.ndarray:
        """The genome each of polities 0..n-1 was born with."""
        g = self.genome[:n].copy()
        for pid, history in self.replaced.items():
            if pid < n:
                g[pid] = history[0][1]
        return g


def archive_frame(model) -> pd.DataFrame:
    """Every brain born (spec section 12): polity, parent, founding group, year of birth
    and the genome it was born with (layout: ``model.brain_layout.to_dict()``)."""
    P = model.polities
    n = P.n
    return pd.DataFrame({
        "polity": np.arange(n),
        "parent": P.parent[:n].copy(),
        "lineage": P.lineage[:n].copy(),
        "founded": P.founded[:n].copy(),
        "weights": list(model.brains.birth_genomes(n)),
    })
