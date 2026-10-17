"""Round 12 calibration: run a batch and check it against the physical bounds in
``configs/bounds.yaml``; write the bounds table (value, bound, ok/broken) per run and for the
batch, and the report-only numbers (never tuned).

    python scripts/acceptance/bounds.py --world-seed 1 --history-seeds 1 2 --batch-dir runs/round12_it1 \
        --overrides configs/calibration.yaml --workers 2
    python scripts/acceptance/bounds.py --batch-dir runs/round12_it1 --no-run   # check finished runs only

Runs go through ``scripts/batch.py`` (complete runs are skipped, interrupted ones resume).
Writes ``bounds.md`` and ``bounds.json`` in the batch directory.

Bounds:
- density: the largest density over the ceiling of its most intensive mode, of any settlement
  in any year (``ceilings.parquet``, ``deluge/data/bounds.py``), is at most 1;
- world: world population / the sum of cell ceilings, its largest over the years, at most 1;
- polity floor: the median lifespan of polities that ever held ``min_settlements`` or more
  settlements (the living counted to the run's end) is at least ``min_median_lifespan``;
- monuments: monuments built <= surplus person-years / ``person_years_per_monument``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analysis.runs import Run, batch_runs  # noqa: E402
from deluge.config import DEFAULT_CONFIG_PATH, load_config, read_overrides  # noqa: E402
from deluge.data.bounds import load_bounds  # noqa: E402

BOUNDS = REPO_ROOT / "configs" / "bounds.yaml"


def real_year(year: int, offset: int) -> str:
    """Sim year as a calendar year (no year 0: offset - 1 is 1 BCE, offset is 1 CE)."""
    r = int(year) - offset
    return f"{-r:,} BCE" if r < 0 else f"{r + 1:,} CE"


def run_config(run: Run):
    meta = run.meta
    path = meta.get("config_path") or str(DEFAULT_CONFIG_PATH)
    ov = meta.get("overrides_path")
    return load_config(path, read_overrides(ov) if ov else None)


# --- bounds ----------------------------------------------------------------------------------------


def check_bounds(run: Run, bounds: dict) -> list[dict]:
    rows = []
    ceil = run.table("ceilings.parquet") if (run.path / "ceilings.parquet").exists() else None
    if ceil is not None and len(ceil):
        i = int(ceil.max_ratio.idxmax())
        r = ceil.loc[i]
        rows.append({"bound": "cell density / ceiling of its mode", "value": float(r.max_ratio),
                     "limit": "<= 1", "ok": bool(r.max_ratio <= 1.0),
                     "detail": f"year {int(r.year):,}, cell {int(r.max_cell)}, {r.max_density:,.1f}/km2"})
        ratio = ceil.population / ceil.world_ceiling.where(ceil.world_ceiling > 0)
        j = int(ratio.idxmax())
        rows.append({"bound": "world population / sum of cell ceilings", "value": float(ratio[j]),
                     "limit": "<= 1", "ok": bool(ratio[j] <= 1.0),
                     "detail": f"year {int(ceil.year[j]):,}: {ceil.population[j]:,.0f} of {ceil.world_ceiling[j]:,.0f}"})
    else:
        rows.append({"bound": "cell density / ceiling of its mode", "value": float("nan"), "limit": "<= 1",
                     "ok": None, "detail": "no ceilings.parquet"})
    pf = bounds["polity_floor"]
    tree = run.polity_tree
    end = run.final_year
    ended = np.where(tree.ended >= 0, tree.ended, end)
    big = tree.peak_settlements >= pf["min_settlements"]
    life = (ended - tree.founded)[big]
    med = float(np.median(life)) if len(life) else float("nan")
    rows.append({"bound": f"median lifespan, polities with {pf['min_settlements']}+ settlements", "value": med,
                 "limit": f">= {pf['min_median_lifespan']}",
                 "ok": None if np.isnan(med) else bool(med >= pf["min_median_lifespan"]),
                 "detail": f"{int(big.sum()):,} polities"})
    model = run.model
    built = int(model.monuments_built.sum())
    if "surplus" in model:
        n = bounds["monument_ceiling"]["person_years_per_monument"]
        surplus = float(model.surplus.sum())
        cap = surplus / n
        rows.append({"bound": "monuments built", "value": built, "limit": f"<= {cap:,.0f}", "ok": bool(built <= cap),
                     "detail": f"surplus {surplus:,.3g} person-years / {n:,}"})
    else:
        rows.append({"bound": "monuments built", "value": built, "limit": "n/a", "ok": None,
                     "detail": "no surplus column"})
    return rows


# --- report only -----------------------------------------------------------------------------------


def population_at(run: Run, years) -> dict:
    m = run.model
    out = {}
    for y in years:
        rows = m[m.year <= y - 1]  # model.parquet row "year" is the year just simulated
        out[y] = float(rows.population.iloc[-1]) if len(rows) and y <= run.final_year else None
    return out


def origin_records(run: Run, years, until: int) -> dict:
    """Per snapshot year and lineage: distinct surviving records whose provenance includes an
    event of years 0 to ``until`` (copies alike in every field in sibling stores counted once)."""
    ev_year = run.events.year.to_numpy()
    lineage = dict(zip(run.polity_tree.polity, run.polity_tree.group))
    out = {}
    for y in years:
        path = run.path / "memory_snapshots" / f"year_{y:05d}.parquet"
        if not path.exists():
            out[y] = None
            continue
        snap = pd.read_parquet(path)
        seen: dict[str, set] = {g: set() for g in run.groups}
        for r in snap.itertuples():
            prov = tuple(int(i) for i in r.provenance)
            if not any(ev_year[i] <= until for i in prov):
                continue
            key = (prov, tuple(int(a) for a in r.actors), r.action, r.year, r.year_span, r.place, r.place_level)
            seen.setdefault(lineage.get(r.polity, "?"), set()).add(key)
        out[y] = {g: len(s) for g, s in seen.items()}
    return out


def report_numbers(run: Run, bounds: dict) -> dict:
    rep = bounds["report"]
    cfg = run_config(run)
    tech = run.techniques
    first = {}
    for r in tech.itertuples():
        first[r.technique] = {"all": int(r.first_held),
                              **{g: int(getattr(r, f"first_held_{g}")) for g in run.groups}}
    tree = run.polity_tree
    return {
        "first_held": first,
        "population": population_at(run, rep["population_years"]),
        "polity_births": int((tree.founded > 0).sum()),
        "monuments": int(run.model.monuments_built.sum()),
        "chronicle_entries": run.meta.get("chronicle_entries"),
        "origin_records": origin_records(run, rep["origin_years"], cfg.reports.origins_until_year),
    }


# --- markdown --------------------------------------------------------------------------------------


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if np.isnan(v):
            return "-"
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def _ok(v) -> str:
    return "n/a" if v is None else ("ok" if v else "**broken**")


def markdown(results: dict, bounds: dict) -> str:
    off = bounds["report"]["calendar_offset"]
    names = list(results)
    lines = ["# Bounds", "", f"Runs: {', '.join(names)}.", "", "## Bounds per run", "",
             "| run | bound | value | bound | status | detail |", "|---|---|---:|---|---|---|"]
    for n, res in results.items():
        for b in res["bounds"]:
            lines.append(f"| {n} | {b['bound']} | {_fmt(b['value'])} | {b['limit']} | {_ok(b['ok'])} | {b['detail']} |")
    lines += ["", "## Bounds for the batch", "", "| bound | worst value | status |", "|---|---:|---|"]
    by_bound: dict[str, list] = {}
    for res in results.values():
        for b in res["bounds"]:
            by_bound.setdefault(b["bound"], []).append(b)
    for bound, bs in by_bound.items():
        vals = [b["value"] for b in bs if b["value"] is not None and not (isinstance(b["value"], float)
                                                                          and np.isnan(b["value"]))]
        worst = (min(vals) if bs[0]["limit"].startswith(">=") else max(vals)) if vals else None
        oks = [b["ok"] for b in bs if b["ok"] is not None]
        status = None if not oks else all(oks)
        lines.append(f"| {bound} | {_fmt(worst)} | {_ok(status)} |")
    lines += ["", "## Report only (never tuned)", "", "### Polity births, monuments, chronicle", "",
              "| run | polity births | monuments | chronicle entries |", "|---|---:|---:|---:|"]
    for n, res in results.items():
        r = res["report"]
        lines.append(f"| {n} | {_fmt(r['polity_births'])} | {_fmt(r['monuments'])} | {_fmt(r['chronicle_entries'])} |")
    years = bounds["report"]["population_years"]
    lines += ["", "### World population", "", "| run | " + " | ".join(f"year {y:,} ({real_year(y, off)})" for y in years)
              + " |", "|---|" + "---:|" * len(years)]
    for n, res in results.items():
        lines.append(f"| {n} | " + " | ".join(_fmt(res["report"]["population"][y]) for y in years) + " |")
    lines += ["", "### First appearance of each technique (sim year / real year)", ""]
    for n, res in results.items():
        first = res["report"]["first_held"]
        groups = [g for g in next(iter(first.values())) if g != "all"]
        lines += [f"**{n}**", "", "| technique | first anywhere | " + " | ".join(groups) + " |",
                  "|---|---|" + "---|" * len(groups)]
        for t, d in first.items():
            def cell(y):
                return "-" if y < 0 else f"{y:,} ({real_year(y, off)})"
            lines.append(f"| {t} | {cell(d['all'])} | " + " | ".join(cell(d[g]) for g in groups) + " |")
        lines.append("")
    lines += ["### Origin records (events of years 0-100) held, per lineage", ""]
    for n, res in results.items():
        orig = res["report"]["origin_records"]
        oyears = list(orig)
        groups = sorted({g for v in orig.values() if v for g in v})
        lines += [f"**{n}**", "", "| lineage | " + " | ".join(f"year {y:,}" for y in oyears) + " |",
                  "|---|" + "---:|" * len(oyears)]
        for g in groups:
            lines.append(f"| {g} | " + " | ".join("-" if orig[y] is None else str(orig[y].get(g, 0)) for y in oyears)
                         + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def evaluate(batch_dir: Path) -> dict:
    bounds = load_bounds(BOUNDS)
    results = {}
    for run in batch_runs(batch_dir):
        results[run.name] = {"bounds": check_bounds(run, bounds), "report": report_numbers(run, bounds)}
    (batch_dir / "bounds.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    (batch_dir / "bounds.md").write_text(markdown(results, bounds), encoding="utf-8")
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch-dir", required=True)
    p.add_argument("--no-run", action="store_true", help="only check the complete runs already in --batch-dir")
    p.add_argument("--world-seed", type=int, default=1)
    p.add_argument("--history-seeds", type=int, nargs="+", default=[1])
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--overrides", default=None)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args(argv)
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / batch_dir
    code = 0
    if not args.no_run:
        import batch

        cmd = ["--world-seed", str(args.world_seed), "--history-seeds", *map(str, args.history_seeds),
               "--batch-dir", str(batch_dir)]
        for flag, v in (("--years", args.years), ("--overrides", args.overrides), ("--workers", args.workers)):
            if v is not None:
                cmd += [flag, str(v)]
        code = batch.main(cmd)
    evaluate(batch_dir)
    print(f"wrote {batch_dir / 'bounds.md'}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
