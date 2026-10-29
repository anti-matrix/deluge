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
- monuments: monuments built <= surplus person-years / ``person_years_per_monument``;
- war deaths (Round 14): each battle's dead are at most the fighting-age share of the target's
  people plus the attacker's mobilized force within logistics reach and its starting settlement
  (``campaigns.parquet``);
- report only (Round 14): near-capacity growth (settlements at ``near_share`` of their
  capacity grow at most ``max_rate`` over ``window_years``; ``settlement_samples.parquet``) and
  spread speed (each lineage's frontier over land advances at most ``max_km_per_year`` over
  ``window_years``; ``frontier.parquet``); the largest polity every 500 years
  (``polity_scale.parquet``). Their status is shown, never counted as a broken bound.
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
from deluge.data.bounds import (battle_death_limit, frontier_speeds, largest_window_rate, load_bounds,  # noqa: E402
                                 near_capacity_rates)

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
    rows += growth_rows(run, bounds)
    rows.append(war_deaths_row(run, bounds))
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


def yearly_population(run: Run) -> dict[str, np.ndarray]:
    """The world's and each founding group's people, indexed by year (model.parquet)."""
    m = run.model.sort_values("year")
    n = int(m.year.max()) + 1
    out = {}
    for name, col in [("world", "population")] + [(g, f"pop_{g}") for g in run.groups]:
        s = np.zeros(n)
        s[m.year.to_numpy(dtype=np.int64)] = m[col].to_numpy(dtype=float)
        out[name] = s
    return out


def industry_year(run: Run, technique: str) -> int | None:
    t = run.techniques
    hit = t[t.technique == technique]
    y = int(hit.first_held.iloc[0]) if len(hit) else -1
    return y if y >= 0 else None


def growth_rows(run: Run, bounds: dict) -> list[dict]:
    """Report-only rows: near-capacity growth (all settlements) and spread speed per lineage."""
    rows = []
    g = bounds["near_capacity_growth"]
    path = run.path / "settlement_samples.parquet"
    if path.exists():
        w = near_capacity_rates(pd.read_parquet(path), g["near_share"], g["window_years"],
                                industry_year(run, g["industry_technique"]))
        if len(w):
            i = w.rate.idxmax()
            over = int((w.rate > g["max_rate"]).sum())
            rows.append({"bound": f"near-capacity growth, {g['window_years']} years (report only)",
                         "value": float(w.rate[i]), "limit": f"<= {g['max_rate']}", "ok": None,
                         "report_ok": bool(w.rate[i] <= g["max_rate"]),
                         "detail": f"{over:,} of {len(w):,} windows over; median {100 * w.rate.median():.3f}%/yr; "
                                   f"worst: settlement {int(w.settlement[i])}, years {int(w.year[i]):,}-"
                                   f"{int(w.year[i]) + g['window_years']:,}, {w.start[i]:,.0f} -> {w.end[i]:,.0f}"})
        else:
            rows.append({"bound": f"near-capacity growth, {g['window_years']} years (report only)",
                         "value": float("nan"), "limit": f"<= {g['max_rate']}", "ok": None,
                         "detail": "no settlement at the share of its capacity at a window's start"})
    s = bounds["spread_speed"]
    path = run.path / "frontier.parquet"
    if path.exists():
        sp = frontier_speeds(pd.read_parquet(path), s["window_years"])
        for lin, name in enumerate(run.groups):
            mine = sp[sp.lineage == lin]
            if not len(mine):
                rows.append({"bound": f"spread speed, {name} (report only)", "value": float("nan"),
                             "limit": f"<= {s['max_km_per_year']} km/yr", "ok": None, "detail": "no window"})
                continue
            i = mine.km_per_year.idxmax()
            rows.append({"bound": f"spread speed, {name} (report only)", "value": float(mine.km_per_year[i]),
                         "limit": f"<= {s['max_km_per_year']} km/yr", "ok": None,
                         "report_ok": bool(mine.km_per_year[i] <= s["max_km_per_year"]),
                         "detail": f"years {int(mine.year[i]):,}-{int(mine.year[i]) + s['window_years']:,}: "
                                   f"{mine.from_km[i]:,.0f} -> {mine.to_km[i]:,.0f} km"})
    return rows


def polity_scale(run: Run) -> list[dict]:
    path = run.path / "polity_scale.parquet"
    return pd.read_parquet(path).to_dict("records") if path.exists() else []


def war_deaths_row(run: Run, bounds: dict) -> dict:
    share = bounds["war_deaths"]["fighting_age_share"]
    path = run.path / "campaigns.parquet"
    c = pd.read_parquet(path) if path.exists() else None
    if c is None or "present_people" not in c or not len(c):
        return {"bound": "battle dead / people present", "value": float("nan"), "limit": "<= 1", "ok": None,
                "detail": "no campaigns.parquet"}
    allies = c.dead_allies.fillna(0.0) if "dead_allies" in c else 0.0
    dead = c.dead_attacker + c.dead_defender + allies
    limit = battle_death_limit(c.target_people, c.present_people, c.mobilization, share,
                               c.allied_mobilized if "allied_mobilized" in c else 0.0)
    ratio = dead / np.maximum(limit, 1e-9)
    i = int(ratio.idxmax())
    return {"bound": "battle dead / people present", "value": float(ratio[i]), "limit": "<= 1",
            "ok": bool(ratio[i] <= 1.0),
            "detail": f"{int((ratio > 1).sum()):,} of {len(c):,} battles over; worst: year {int(c.year[i]):,}, "
                      f"{dead[i]:,.0f} dead against {limit[i]:,.0f}"}


# --- report only -----------------------------------------------------------------------------------


def largest_battle(run: Run) -> dict | None:
    path = run.path / "campaigns.parquet"
    if not path.exists():
        return None
    c = pd.read_parquet(path)
    if not len(c):
        return None
    dead = c.dead_attacker + c.dead_defender
    i = int(dead.idxmax())
    return {"year": int(c.year[i]), "dead": float(dead[i]), "attacker": int(c.attacker[i]),
            "defender": int(c.defender[i]), "target": int(c.target[i]), "target_people": float(c.target_people[i]),
            "dead_attacker": float(c.dead_attacker[i]), "dead_defender": float(c.dead_defender[i])}


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
    g = bounds["near_capacity_growth"]
    pops = yearly_population(run)
    until = industry_year(run, g["industry_technique"])
    end = run.final_year
    ended = np.where(tree.ended >= 0, tree.ended, end)
    big = tree.peak_settlements >= bounds["polity_floor"]["min_settlements"]
    return {
        "first_held": first,
        "population": population_at(run, rep["population_years"]),
        "lineage_population": {y: {n: (float(p[y - 1]) if y - 1 < len(p) else None) for n, p in pops.items()}
                               for y in rep["population_years"] if y <= end},
        "growth": {n: largest_window_rate(p, g["window_years"], 0.0, until) for n, p in pops.items()},
        "largest_battle": largest_battle(run),
        "mergers": int(run.model.mergers.sum()),
        "polity_scale": polity_scale(run),
        "median_lifespan": float(np.median((ended - tree.founded)[big])) if big.any() else None,
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
            status = _ok(b["ok"]) if "report_ok" not in b else ("within (report)" if b["report_ok"] else "over (report)")
            lines.append(f"| {n} | {b['bound']} | {_fmt(b['value'])} | {b['limit']} | {status} | {b['detail']} |")
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
    lines += ["", "### People per founding group (Round 14)", ""]
    for n, res in results.items():
        lp = res["report"].get("lineage_population") or {}
        if not lp:
            continue
        names = list(next(iter(lp.values())))
        lines += [f"**{n}**", "", "| year | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
        for y, row in lp.items():
            lines.append(f"| {int(y):,} | " + " | ".join(_fmt(row[k]) for k in names) + " |")
        lines.append("")
    lines += ["### Largest 500-year growth rate (every window before industry, no minimum size)", "",
              "| run | " + " | ".join(next(iter(results.values()))["report"]["growth"]) + " |",
              "|---|" + "---:|" * len(next(iter(results.values()))["report"]["growth"])]
    for n, res in results.items():
        lines.append(f"| {n} | " + " | ".join("-" if np.isnan(r) else f"{100 * r:.3f}% (from {s:,})"
                                              for r, s in res["report"]["growth"].values()) + " |")
    lines += ["", "### Largest battle, mergers, median lifespan (polities with 3+ settlements)", "",
              "| run | largest battle | mergers | median lifespan |", "|---|---|---:|---:|"]
    for n, res in results.items():
        r = res["report"]
        b = r["largest_battle"]
        battle = "-" if not b else (f"year {b['year']:,}: {b['dead']:,.0f} dead ({b['dead_attacker']:,.0f} attacker, "
                                    f"{b['dead_defender']:,.0f} defender; target {b['target_people']:,.0f} people), "
                                    f"polity {b['attacker']} on {b['defender']}")
        lines.append(f"| {n} | {battle} | {_fmt(r['mergers'])} | {_fmt(r['median_lifespan'])} |")
    lines += ["", "### Largest polity every 500 years (report only)", ""]
    for n, res in results.items():
        rows = res["report"].get("polity_scale") or []
        if not rows:
            continue
        lines += [f"**{n}**", "", "| year | polity | lineage | people | settlements | writing | hierarchy | techniques |",
                  "|---:|---:|---:|---:|---:|---|---:|---:|"]
        for r in rows:
            lines.append(f"| {int(r['year']):,} | {int(r['polity'])} | {int(r['lineage'])} | {r['population']:,.0f} | "
                         f"{int(r['settlements']):,} | {'yes' if r['writing'] else 'no'} | {r['hierarchy']:.2f} | "
                         f"{int(r['techniques'])} |")
        lines.append("")
    lines += ["### First appearance of each technique (sim year / real year)", ""]
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
