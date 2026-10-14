"""Config loader: YAML files into frozen dataclasses.

Unknown keys are rejected so a typo in a config never silently falls back to a default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


@dataclass(frozen=True)
class RunConfig:
    seed: int
    years: int


@dataclass(frozen=True)
class RngConfig:
    streams: tuple[str, ...]


@dataclass(frozen=True)
class OutputConfig:
    runs_dir: str


@dataclass(frozen=True)
class Config:
    run: RunConfig
    rng: RngConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Stable hash of the full config, for run metadata."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """Return a new Config with nested overrides applied, e.g. {"run": {"seed": 7}}."""
        return _build(Config, _deep_merge(self.to_dict(), overrides), "config")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH, overrides: dict[str, Any] | None = None) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return _build(Config, raw, "config")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _build(cls: type, raw: Any, where: str) -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")
    missing = names - set(raw)
    if missing:
        raise KeyError(f"{where}: missing keys {sorted(missing)}")
    kwargs = {}
    for name in names:
        hint = hints[name]
        value = raw[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{where}.{name}")
        elif getattr(hint, "__origin__", None) is tuple:
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = _coerce(hint, value, f"{where}.{name}")
    return cls(**kwargs)


def _coerce(hint: Any, value: Any, where: str) -> Any:
    if hint is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected bool, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected int, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected float, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected str, got {value!r}")
        return value
    return value
