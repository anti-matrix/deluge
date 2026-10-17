"""Reading headless run folders (``scripts/run.py``, ``scripts/batch.py``) for analysis.

``Run(path)`` loads a complete run's tables lazily; ``batch_runs(dir)`` lists the complete
runs in a batch folder (each subfolder with a complete ``meta.json``).
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd


class Run:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        self.name = self.path.name

    def table(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path / name)

    @cached_property
    def model(self) -> pd.DataFrame:
        return self.table("model.parquet")

    @cached_property
    def lineages(self) -> pd.DataFrame:
        return self.table("lineages.parquet")

    @cached_property
    def polities(self) -> pd.DataFrame:
        return self.table("polities.parquet")

    @cached_property
    def polity_tree(self) -> pd.DataFrame:
        return self.table("polity_tree.parquet")

    @cached_property
    def founders(self) -> pd.DataFrame:
        return self.table("founders.parquet")

    @cached_property
    def settlements(self) -> pd.DataFrame:
        return self.table("settlements.parquet")

    @cached_property
    def events(self) -> pd.DataFrame:
        return self.table("events.parquet")

    @cached_property
    def techniques(self) -> pd.DataFrame:
        return self.table("techniques.parquet")

    @cached_property
    def brains(self) -> pd.DataFrame:
        return self.table("brains/brains.parquet")

    @cached_property
    def memory_ops(self) -> pd.DataFrame:
        return self.table("memory_ops.parquet")

    @cached_property
    def memory_final(self) -> pd.DataFrame:
        """The memory snapshot at the end of the run."""
        snaps = sorted((self.path / "memory_snapshots").glob("year_*.parquet"))
        return pd.read_parquet(snaps[-1]) if snaps else pd.DataFrame()

    @property
    def final_year(self) -> int:
        return int(self.meta["final_year"])

    @property
    def groups(self) -> list[str]:
        return list(self.founders.sort_values("lineage").group)


def batch_runs(batch_dir: str | Path) -> list[Run]:
    out = []
    for meta in sorted(Path(batch_dir).glob("*/meta.json")):
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("complete"):
                out.append(Run(meta.parent))
        except (OSError, json.JSONDecodeError):
            continue
    return out
