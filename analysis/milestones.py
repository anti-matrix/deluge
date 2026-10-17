"""Milestone report: the first year each technique was held anywhere, first invented, and
first held per founding lineage (``techniques.parquet``, ``deluge/data/collectors.py``).

    python analysis/milestones.py RUN_OR_BATCH_DIR

For a run folder, writes ``milestones.md`` there. For a batch folder, writes one per run
and ``milestones.md`` in the batch folder: per technique, in how many runs it appeared and
the median and range of its first year. "Held" counts a technique held latent too (held
where it cannot be used); a year of -1 is shown as a dash (never).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.runs import Run, batch_runs  # noqa: E402


def _year(v) -> str:
    return "-" if v is None or int(v) < 0 else str(int(v))


def run_table(run: Run) -> pd.DataFrame:
    """Techniques by first year held anywhere (never held last)."""
    df = run.techniques.copy()
    df["_order"] = np.where(df.first_held < 0, np.iinfo(np.int64).max, df.first_held)
    return df.sort_values(["_order", "technique"]).drop(columns="_order")


def run_report(run: Run) -> str:
    df = run_table(run)
    groups = run.groups
    lines = [f"# Milestones: {run.name}", "",
             f"World seed {run.meta['world_seed']}, history seed {run.meta['history_seed']}, "
             f"{run.final_year} years. First year a settlement held each technique (anywhere, and per "
             "founding lineage), and its first independent invention; - = never.", "",
             "| technique | first held | first invented | " + " | ".join(groups) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in groups) + " |"]
    for _, r in df.iterrows():
        per = [_year(r[f"first_held_{g}"]) for g in groups]
        lines.append(f"| {r.label} | {_year(r.first_held)} | {_year(r.first_invented)} | " + " | ".join(per) + " |")
    held = int((df.first_held >= 0).sum())
    lines += ["", f"{held} of {len(df)} techniques held somewhere by year {run.final_year}."]
    return "\n".join(lines) + "\n"


def batch_report(runs: list[Run]) -> str:
    rows = {}
    for run in runs:
        for _, r in run.techniques.iterrows():
            rows.setdefault((r.technique, r.label), []).append(int(r.first_held))
    lines = [f"# Milestones across {len(runs)} runs", "",
             "Runs: " + ", ".join(f"{r.name} ({r.final_year} years)" for r in runs) + ".", "",
             "Per technique: in how many runs a settlement held it, and the first year it was held "
             "(median, earliest, latest over those runs).", "",
             "| technique | runs | median | earliest | latest |", "| --- | --- | --- | --- | --- |"]
    table = []
    for (name, label), years in rows.items():
        got = [y for y in years if y >= 0]
        table.append((np.median(got) if got else np.inf, label, len(got), got))
    for med, label, n, got in sorted(table, key=lambda x: (x[0], x[1])):
        if got:
            lines.append(f"| {label} | {n} of {len(runs)} | {med:.0f} | {min(got)} | {max(got)} |")
        else:
            lines.append(f"| {label} | 0 of {len(runs)} | - | - | - |")
    lines += ["", "Per-run tables: " + ", ".join(f"`{r.name}/milestones.md`" for r in runs) + "."]
    return "\n".join(lines) + "\n"


def write(path: str | Path) -> list[Path]:
    path = Path(path)
    if (path / "meta.json").exists():
        runs, batch = [Run(path)], None
    else:
        runs, batch = batch_runs(path), path
    written = []
    for run in runs:
        out = run.path / "milestones.md"
        out.write_text(run_report(run), encoding="utf-8")
        written.append(out)
    if batch is not None and runs:
        out = batch / "milestones.md"
        out.write_text(batch_report(runs), encoding="utf-8")
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    for p in write(argv[0]):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
