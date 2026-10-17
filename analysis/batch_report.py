"""Analysis report over the runs of a batch (spec section 12, "Multi-run analysis").

    python analysis/batch_report.py BATCH_DIR

Writes ``analysis_report.md`` in the batch folder, one section per question:

1. Which founding lineage ends up largest (people at the last ``lineages.parquet`` row), and
   what goes with it: start capacity, contact with the teachers (visit order, first year,
   techniques taught, artifacts) and climate luck (share of its people-years on cells in
   drought over the first ``EARLY_YEARS`` years). Rank correlations over every (run, lineage) pair.
2. When farming spreads, how fast and from where: each lineage's first farming year, the
   years the farming share of all people first reaches each of ``FARMING_SHARES``, and the
   lineage that farmed first.
3. How long polities last and how large they grow (``polity_tree.parquet``).
4. Which memories survive longest (age of the records alive at the end, by the verb of
   their oldest original event), which entities are substituted in (``memory_ops``), and
   whether the same verbs lead the oldest records in every run.
5. What surviving brains converge on: weight distance from their founding brain and the
   mean brain outputs of the founders against the survivors.

Every number is read from the run tables; nothing here names a kind of story.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.runs import Run, batch_runs  # noqa: E402
from deluge.brains.network import OUTPUTS  # noqa: E402
from deluge.data.events import ENTITY_KINDS, decode  # noqa: E402

EARLY_YEARS = 500
FARMING_SHARES = (0.1, 0.5)
OLDEST_RECORDS = 20
PREDICTORS = ("start_capacity", "teachers_visit_order", "teachers_first_year", "techniques_taught", "artifacts",
              "early_drought_share")


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3g}") -> list[str]:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


# --- 1. the largest lineage ----------------------------------------------------------------------


def lineage_outcomes(run: Run) -> pd.DataFrame:
    """Per lineage: final people and share, and the predictors."""
    lin = run.lineages
    last = lin[lin.year == lin.year.max()].set_index("lineage")
    f = run.founders.set_index("lineage").copy()
    f["final_population"] = last.population
    f["final_share"] = last.population / max(float(last.population.sum()), 1e-12)
    f["final_polities"] = last.polities
    f["extinct"] = last.population <= 0
    early = lin[lin.year < EARLY_YEARS]
    early = early[early.year == early.year.max()].set_index("lineage")
    f["early_drought_share"] = early.drought_people_years / early.people_years.clip(lower=1e-12)
    f["run"] = run.name
    return f.reset_index()


def section_largest(runs: list[Run]) -> list[str]:
    frames = [lineage_outcomes(r) for r in runs]
    lines = ["## 1. Which founding lineage ends up largest", ""]
    winners = Counter()
    rows = []
    for run, f in zip(runs, frames):
        top = f.loc[f.final_population.idxmax()]
        winners[top.group] += 1
        rows.append({"run": run.name, "years": run.final_year, "largest": top.group,
                     "its share": float(top.final_share),
                     "extinct": ", ".join(f[f.extinct].group) or "none"})
    lines += _md_table(pd.DataFrame(rows))
    lines += ["", "Times largest: " + ", ".join(f"{g} {winners.get(g, 0)}" for g in runs[0].groups) + ".", ""]
    allf = pd.concat(frames, ignore_index=True)
    lines += ["Per lineage and run (share of all people at the end, and the predictors):", ""]
    show = allf[["run", "group", "final_share", *PREDICTORS]]
    lines += _md_table(show)
    corr = []
    for p in PREDICTORS:
        x = allf[p].astype(float)
        rho = allf.final_share.corr(x, method="spearman") if x.nunique() > 1 else np.nan
        corr.append({"predictor": p, "rank correlation with final share": rho})
    lines += ["", f"Rank correlation over the {len(allf)} (run, lineage) pairs (Spearman; few runs: read as a hint):",
              ""]
    lines += _md_table(pd.DataFrame(corr), "{:+.2f}")
    return lines + [""]


# --- 2. farming ------------------------------------------------------------------------------


def section_farming(runs: list[Run]) -> list[str]:
    lines = ["## 2. When farming spreads, how fast, and from where", ""]
    rows = []
    for run in runs:
        f = run.founders.set_index("group")
        lin = run.lineages.groupby("year")[["farming_population", "population"]].sum()
        share = lin.farming_population / lin.population.clip(lower=1e-12)
        row = {"run": run.name}
        for g in run.groups:
            row[f"first farming {g}"] = int(f.loc[g, "first_farming_year"])
        for s in FARMING_SHARES:
            hit = share.index[share >= s]
            row[f"{s:.0%} of people farm"] = int(hit[0]) if len(hit) else -1
        farmed = f[f.first_farming_year >= 0]
        row["first to farm"] = farmed.first_farming_year.idxmin() if len(farmed) else "none"
        visited = f[f.teachers_visit_order >= 0].sort_values("teachers_visit_order")
        row["teachers visited first"] = visited.index[0] if len(visited) else "none"
        row[f"farming share at end"] = float(share.iloc[-1])
        rows.append(row)
    lines += _md_table(pd.DataFrame(rows))
    return lines + ["", "-1: not reached. First farming year: the first year a settlement of the lineage farmed at "
                        "least the knowledge.use_share of its people.", ""]


# --- 3. polities -------------------------------------------------------------------------------


def section_polities(runs: list[Run]) -> list[str]:
    lines = ["## 3. How long polities last, and how large they grow", ""]
    rows = []
    for run in runs:
        t = run.polity_tree
        ended = t.ended >= 0
        life = np.where(ended, t.ended, run.final_year) - t.founded
        el = life[ended]
        rows.append({
            "run": run.name, "polities": len(t), "ended": int(ended.sum()),
            "lifespan median": float(np.median(el)) if len(el) else np.nan,
            "p90": float(np.percentile(el, 90)) if len(el) else np.nan,
            "max (any)": int(life.max()), "share >= 100 y": float((life >= 100).mean()),
            "share >= 1000 y": float((life >= 1000).mean()),
            "peak settlements median": float(t.peak_settlements.median()),
            "p99": float(t.peak_settlements.quantile(0.99)), "max": int(t.peak_settlements.max()),
        })
    lines += _md_table(pd.DataFrame(rows))
    ends = Counter()
    for run in runs:
        ends.update(c for c in run.polity_tree.end_cause if c)
    lines += ["", "How polities ended (all runs): " + ", ".join(f"{k} {v}" for k, v in ends.most_common()) + ".",
              "Lifespans of ended polities; shares and maxima over all polities, the living counted to the end.", ""]
    return lines


# --- 4. memory -------------------------------------------------------------------------------


def memory_ages(run: Run) -> pd.DataFrame:
    """Records alive at the end: age (end year - year of the oldest original event) and the
    verb of that oldest event."""
    mem = run.memory_final
    if mem.empty:
        return pd.DataFrame(columns=["age", "origin_action"])
    ev = run.events
    years = ev.year.to_numpy()
    actions = ev.action.astype(str).to_numpy()
    oldest = [min(p, key=lambda i: (years[i], i)) for p in mem.provenance]
    return pd.DataFrame({"age": run.final_year - years[oldest], "origin_action": actions[oldest],
                         "provenance": [len(p) for p in mem.provenance], "store": mem.polity.to_numpy()})


def section_memory(runs: list[Run]) -> list[str]:
    lines = ["## 4. Which memories survive longest, and which actors are substituted in", ""]
    ages = {r.name: memory_ages(r) for r in runs}
    allm = pd.concat([a.assign(run=n) for n, a in ages.items()], ignore_index=True)
    if allm.empty:
        return lines + ["No memory records (memory disabled).", ""]
    by = allm.groupby("origin_action").age.agg(["count", "median", "max"]).sort_values("median", ascending=False)
    by = by.reset_index().rename(columns={"origin_action": "verb of oldest original event", "count": "records"})
    lines += ["Records alive at the end of each run, by the verb of their oldest original event (age in years):", ""]
    lines += _md_table(by)
    lines += ["", f"Verbs of the {OLDEST_RECORDS} oldest records in each run (do the same ones recur?):", ""]
    rows = []
    for name, a in ages.items():
        top = a.nlargest(OLDEST_RECORDS, "age")
        rows.append({"run": name, "oldest age": int(top.age.max()) if len(top) else -1,
                     "verbs": ", ".join(f"{k} {v}" for k, v in Counter(top.origin_action).most_common())})
    lines += _md_table(pd.DataFrame(rows))
    counts = []
    for run in runs:
        ops = run.memory_ops
        sub = ops[ops.op == "substituted"]
        counts.append((run.name, len(sub), Counter(decode(int(b))[0] for b in sub.b)))
    kinds = [k for k in ENTITY_KINDS if any(c[k] for _, _, c in counts)]
    rows = [{"run": name, "substitutions": n, **{f"in: {k}": c.get(k, 0) for k in kinds}} for name, n, c in counts]
    lines += ["", "Actor substitutions over each run, by the kind of entity substituted in:", ""]
    lines += _md_table(pd.DataFrame(rows))
    rows = []
    for run in runs:
        mem, ev = run.memory_final, run.events
        teachers = 0  # entity id of the teachers (local 0, kind 0)
        holding = [a for a in mem.actors if teachers in list(a)]
        drifted = sum(1 for a, p in zip(mem.actors, mem.provenance)
                      if teachers in list(a) and not any(teachers in list(ev.actors.iloc[i]) for i in p))
        rows.append({"run": run.name, "records naming the teachers": len(holding),
                     "of which none of the original events involved them": drifted})
    lines += ["", "The teachers as an actor in records alive at the end:", ""]
    lines += _md_table(pd.DataFrame(rows))
    return lines + [""]


# --- 5. brains -------------------------------------------------------------------------------


def section_brains(runs: list[Run]) -> list[str]:
    lines = ["## 5. What surviving brains converge on, compared with the founders'", ""]
    rows, outs = [], []
    for run in runs:
        b = run.brains.set_index("polity")
        tree = run.polity_tree.set_index("polity")
        W = {p: np.asarray(w, dtype=float) for p, w in b.weights.items()}
        founders = {int(r.lineage): p for p, r in b[b.parent < 0].iterrows()}
        alive = tree.index[tree.ended < 0]
        dist = [np.linalg.norm(W[p] - W[founders[int(tree.loc[p, "lineage"])]]) for p in alive]
        scale = np.mean([np.linalg.norm(W[p]) for p in founders.values()])
        rows.append({"run": run.name, "survivors": len(alive), "mean distance from own founder": float(np.mean(dist))
                     if dist else np.nan, "founder weight norm": float(scale)})
        pol = run.polities
        first = pol[(pol.year == pol.year.min()) & (pol.parent < 0)]
        last = pol[pol.year == pol.year.max()]
        for o in OUTPUTS:
            outs.append({"run": run.name, "output": o, "founders": first[f"brain_{o}"].mean(),
                         "survivors": last[f"brain_{o}"].mean()})
    lines += _md_table(pd.DataFrame(rows))
    o = pd.DataFrame(outs).groupby("output", sort=False)[["founders", "survivors"]].mean().reset_index()
    o["change"] = o.survivors - o.founders
    lines += ["", "Mean brain outputs (0..1), founders at year 0 against the polities alive at the last "
                  "polities.parquet row, averaged over runs:", ""]
    lines += _md_table(o, "{:.3f}")
    return lines + [""]


def report(runs: list[Run]) -> str:
    head = ["# Batch analysis report", "",
            f"{len(runs)} runs: " + ", ".join(f"{r.name} (world {r.meta['world_seed']}, history "
                                              f"{r.meta['history_seed']}, {r.final_year} years)" for r in runs) + ".",
            f"Git commit {runs[0].meta.get('git_commit')}, config hash {runs[0].meta['config_hash'][:12]}.", ""]
    body = section_largest(runs) + section_farming(runs) + section_polities(runs) + section_memory(runs)
    body += section_brains(runs)
    return "\n".join(head + body)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    runs = batch_runs(argv[0])
    if not runs:
        print(f"no complete runs in {argv[0]}")
        return 1
    out = Path(argv[0]) / "analysis_report.md"
    out.write_text(report(runs), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
