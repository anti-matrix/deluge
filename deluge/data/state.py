"""A digest of the full model state, to check that a change leaves results identical.

``state_digest(model)`` hashes every attribute of the model (settlements, polities, brains,
memory stores and records, the event log, the teachers, the RNG streams, the collected
data, ...) and everything reachable from it, exactly: floats by their bits, dicts in their
insertion order, sets sorted. Left out: private attributes (leading underscore: Mesa's
internals and the sim's caches, which hold nothing a later year reads that is not
recomputed), dataclass fields marked ``reports_only`` (kept for the reports, never read by
the simulation: the roles of a memory record's actors), the Mesa grid and agent objects (agents are digested by id and cell; their state
lives in the tables), and bound methods. The config enters by the hash of the sections the
simulation reads (``simulation_config_hash``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections.abc import Mapping

import numpy as np

SKIP_MODEL_KEYS = frozenset({"subscribers", "data_registry", "step", "grid"})
# Config sections the simulation never reads (outputs, browser UI, maps, reports).
NON_SIMULATION_SECTIONS = ("output", "viz", "render", "reports", "chronicle")


def simulation_config_hash(cfg) -> str:
    """Hash of the config without ``NON_SIMULATION_SECTIONS``."""
    import json

    d = {k: v for k, v in cfg.to_dict().items() if k not in NON_SIMULATION_SECTIONS}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def _agent_key(agent) -> tuple:
    cell = getattr(agent, "cell", None)
    return ("agent", type(agent).__name__, getattr(agent, "unique_id", None),
            getattr(cell, "coordinate", None) if cell is not None else None)


def _feed(h, x, seen: set) -> None:
    import mesa
    from mesa.discrete_space import PropertyLayer

    from deluge.config import Config
    from deluge.rng import RngStreams

    if x is None or isinstance(x, (bool, np.bool_)):
        h.update(f"<{x!r}>".encode())
    elif isinstance(x, (int, np.integer)):
        h.update(b"i" + str(int(x)).encode())
    elif isinstance(x, (float, np.floating)):
        h.update(b"f" + float(x).hex().encode())
    elif isinstance(x, (str, bytes)):
        h.update(b"s" + (x.encode() if isinstance(x, str) else x))
    elif isinstance(x, np.ndarray):
        h.update(f"a{x.dtype.str}{x.shape}".encode())
        if x.dtype == object:
            for v in x.ravel().tolist():
                _feed(h, v, seen)
        else:
            h.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, Mapping):
        h.update(f"d{len(x)}".encode())
        for k, v in x.items():
            _feed(h, k, seen)
            _feed(h, v, seen)
    elif isinstance(x, (list, tuple)):
        h.update(f"l{len(x)}".encode())
        for v in x:
            _feed(h, v, seen)
    elif isinstance(x, (set, frozenset)):
        sub = []
        for v in x:
            hv = hashlib.sha256()
            _feed(hv, v, seen)
            sub.append(hv.digest())
        h.update(f"S{len(x)}".encode() + b"".join(sorted(sub)))
    elif isinstance(x, np.random.Generator):
        _feed(h, x.bit_generator.state, seen)
    elif isinstance(x, random.Random):
        _feed(h, x.getstate(), seen)
    elif isinstance(x, RngStreams):
        _feed(h, x.state(), seen)
    elif isinstance(x, Config):
        h.update(b"c" + simulation_config_hash(x).encode())
    elif isinstance(x, PropertyLayer):
        _feed(h, x.data, seen)
    elif isinstance(x, mesa.Agent):
        _feed(h, _agent_key(x), seen)
    elif isinstance(x, mesa.DataCollector):
        df = x.get_model_vars_dataframe()
        _feed(h, list(df.columns), seen)
        _feed(h, df.to_numpy(dtype=float), seen)
    elif callable(x) and not dataclasses.is_dataclass(x) and not hasattr(x, "__dict__"):
        h.update(b"<callable>")
    else:
        if id(x) in seen:
            h.update(b"<seen>")
            return
        seen.add(id(x))
        h.update(type(x).__name__.encode())
        if dataclasses.is_dataclass(x):
            items = [(f.name, getattr(x, f.name)) for f in dataclasses.fields(x) if not f.metadata.get("reports_only")]
        elif hasattr(x, "__dict__"):
            items = sorted((k, v) for k, v in vars(x).items() if not k.startswith("_"))
        else:
            items = [(k, getattr(x, k)) for k in getattr(type(x), "__slots__", ())]
        for k, v in items:
            if callable(v) and hasattr(v, "__self__"):
                continue  # bound methods
            h.update(k.encode())
            _feed(h, v, seen)


def state_digest(model) -> str:
    """SHA-256 over the model's full state (see the module docstring)."""
    h = hashlib.sha256()
    seen = {id(model)}
    for k in sorted(vars(model)):
        if k.startswith("_") or k in SKIP_MODEL_KEYS:
            continue
        v = vars(model)[k]
        if k in ("settlement_agents", "polity_agents"):
            v = {i: _agent_key(a) for i, a in v.items()}
        h.update(k.encode())
        _feed(h, v, seen)
    return h.hexdigest()


def part_digests(model) -> dict[str, str]:
    """One digest per top-level attribute, to find where two states differ."""
    out = {}
    for k in sorted(vars(model)):
        if k.startswith("_") or k in SKIP_MODEL_KEYS:
            continue
        v = vars(model)[k]
        if k in ("settlement_agents", "polity_agents"):
            v = {i: _agent_key(a) for i, a in v.items()}
        h = hashlib.sha256()
        _feed(h, v, {id(model)})
        out[k] = h.hexdigest()
    return out
