"""Round 14 step 7: alliances and kin markers, the appearance prior on against off.

    python scripts/acceptance/alliances.py --runs runs/r14_accept/w1_h1_on runs/r14_accept/w1_h1_off ... \
        --out runs/r14_accept/alliances.md

Each run folder is a complete ``scripts/run.py`` run (names ending in ``_on`` / ``_off`` give the
condition; the rest of the name is the seeds). Per run:

- alliances formed, mean duration (of those that ended; and of all, the living counted to the
  run's end), betrayals;
- the share of alliances within one founding lineage against the share of polity pairs in contact
  within one lineage (pair-years, ``diplomacy.parquet``), so geography is controlled; the same for
  wars (attacks, ``campaigns.parquet``);
- the evolved per-neighbour head weights at the end (``brains_final.parquet``), mean and sd over
  the living polities, on appearance difference, dialect similarity and shared descent, for
  alliance scoring and for target scoring;
- polity births, median lifespan of polities with 3 or more settlements, mergers;
- alliances per polity: over each polity's life and at the end (max, median, share with 4 or
  more), and whether networks snowballed: the largest network and its share of the living
  polities over time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from analysis.runs import Run  # noqa: E402

MARKERS = ("appearance_distance", "dialect_similarity", "shared_descent")


def per_polity(al: pd.DataFrame, n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=np.int64)
    if len(al):
        np.add.at(counts, al.a.to_numpy(), 1)
        np.add.at(counts, al.b.to_numpy(), 1)
    return counts


def summary(run: Run) -> dict:
    tree = run.polity_tree
    lineage = tree.set_index("polity").group if "group" in tree else tree.set_index("polity").lineage
    end = run.final_year
    al = run.table("alliances.parquet")
    dip = run.table("diplomacy.parquet")
    camp = run.table("campaigns.parquet")
    heads = run.table("brains_final.parquet")
    lin = tree.set_index("polity").lineage
    out: dict = {}
    out["alliances"] = len(al)
    ended = al[al.ended >= 0]
    out["mean_years_ended"] = float((ended.ended - ended.began).mean()) if len(ended) else float("nan")
    out["mean_years_all"] = float((np.where(al.ended >= 0, al.ended, end) - al.began).mean()) if len(al) else float("nan")
    out["betrayals"] = int((al.how == "betrayed").sum())
    out["ended_by_rescoring"] = int((al.how == "rescored").sum())
    same = (lin[al.a].to_numpy() == lin[al.b].to_numpy()) if len(al) else np.zeros(0, bool)
    out["alliance_same_lineage"] = float(same.mean()) if len(al) else float("nan")
    out["contact_same_lineage"] = float(dip.contact_pairs_same_lineage.sum() / max(dip.contact_pairs.sum(), 1))
    w_same = lin[camp.attacker].to_numpy() == lin[camp.defender].to_numpy()
    out["wars"] = len(camp)
    out["war_same_lineage"] = float(w_same.mean()) if len(camp) else float("nan")
    for head in ("alliance", "target"):
        for f in MARKERS:
            col = heads[f"{head}_{f}"] if f"{head}_{f}" in heads else pd.Series(dtype=float)
            out[f"{head}_{f}"] = (float(col.mean()), float(col.std(ddof=0))) if len(col) else (float("nan"),) * 2
    ended_t = np.where(tree.ended >= 0, tree.ended, end)
    big = tree.peak_settlements >= 3
    out["polity_births"] = int((tree.founded > 0).sum())
    out["median_lifespan_3plus"] = float(np.median((ended_t - tree.founded)[big])) if big.any() else float("nan")
    out["mergers"] = int(run.model.mergers.sum())
    life = per_polity(al, len(tree))
    living_al = al[al.ended < 0]
    now = per_polity(living_al, len(tree))
    alive = (tree.ended < 0).to_numpy()
    out["allies_life_max"] = int(life.max()) if len(life) else 0
    out["allies_life_median_allied"] = float(np.median(life[life > 0])) if (life > 0).any() else 0.0
    out["allies_life_share_4plus"] = float((life >= 4).mean()) if len(life) else 0.0
    out["allies_end_max"] = int(now[alive].max()) if alive.any() else 0
    out["allies_end_median"] = float(np.median(now[alive])) if alive.any() else 0.0
    out["allies_end_share_4plus"] = float((now[alive] >= 4).mean()) if alive.any() else 0.0
    ps = run.path / "prior_shift.parquet"
    pr = pd.read_parquet(ps) if ps.exists() else pd.DataFrame()
    for k in ("shift", "would_shift"):
        out[f"prior_{k}"] = float(pr[k].mean()) if len(pr) else float("nan")
    for k in ("alliance_term", "target_term"):
        out[f"brain_{k}"] = (float(pr[k].mean()), float(pr[k].abs().mean())) if len(pr) else (float("nan"),) * 2
    peak = dip.loc[dip.largest_network.idxmax()] if len(dip) else None
    out["largest_network_peak"] = (int(peak.largest_network), int(peak.year), int(peak.polities)) if peak is not None else None
    last = dip.iloc[-1] if len(dip) else None
    out["largest_network_end"] = (int(last.largest_network), int(last.polities), int(last.networks)) if last is not None else None
    out["max_allies_peak"] = int(dip.max_allies.max()) if len(dip) else 0
    out["network_curve"] = [(int(r.year), int(r.largest_network), int(r.polities)) for r in dip.itertuples()
                            if r.year % 500 == 0]
    return out


def fmt(v) -> str:
    if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return "-" if np.isnan(v[0]) else f"{v[0]:+.3f} ± {v[1]:.3f}"
    if isinstance(v, float):
        return "-" if np.isnan(v) else (f"{v:.3f}" if abs(v) < 10 else f"{v:,.0f}")
    return f"{v:,}" if isinstance(v, int) else str(v)


ROWS = [
    ("alliances formed", "alliances"), ("mean years (ended)", "mean_years_ended"),
    ("mean years (all, living to the end)", "mean_years_all"), ("betrayals", "betrayals"),
    ("ended by rescoring", "ended_by_rescoring"),
    ("alliances within one lineage", "alliance_same_lineage"), ("contact pairs within one lineage", "contact_same_lineage"),
    ("attacks", "wars"), ("attacks within one lineage", "war_same_lineage"),
    ("alliance head: appearance difference", "alliance_appearance_distance"),
    ("alliance head: dialect similarity", "alliance_dialect_similarity"),
    ("alliance head: shared descent", "alliance_shared_descent"),
    ("target head: appearance difference", "target_appearance_distance"),
    ("target head: dialect similarity", "target_dialect_similarity"),
    ("target head: shared descent", "target_shared_descent"),
    ("prior's shift of a score's sum, mean over pairs in contact at the end (+ alliance, - target)", "prior_shift"),
    ("... the same were the prior on", "prior_would_shift"),
    ("brain's own appearance term in the alliance sum (mean, mean of |.|)", "brain_alliance_term"),
    ("brain's own appearance term in the target sum (mean, mean of |.|)", "brain_target_term"),
    ("polity births", "polity_births"), ("median lifespan, 3+ settlements", "median_lifespan_3plus"),
    ("mergers", "mergers"),
    ("alliances per polity over its life: max", "allies_life_max"),
    ("... median (polities with any)", "allies_life_median_allied"), ("... share with 4+", "allies_life_share_4plus"),
    ("allies per living polity at the end: max", "allies_end_max"), ("... median", "allies_end_median"),
    ("... share with 4+", "allies_end_share_4plus"),
    ("largest network at its peak (size, year, living polities)", "largest_network_peak"),
    ("largest network at the end (size, living polities, networks)", "largest_network_end"),
    ("most allies of one polity at any sample", "max_allies_peak"),
]


def markdown(results: dict[str, dict]) -> str:
    names = list(results)
    lines = ["# Alliances and kin markers: prior on vs off", "", "| | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for label, key in ROWS:
        lines.append(f"| {label} | " + " | ".join(fmt(results[n][key]) for n in names) + " |")
    lines += ["", "## Largest alliance network every 500 years (size / living polities)", ""]
    for n in names:
        lines.append(f"- {n}: " + ", ".join(f"{y:,}: {s}/{p}" for y, s, p in results[n]["network_curve"]))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    results = {Path(r).name: summary(Run(r)) for r in args.runs}
    Path(args.out).write_text(markdown(results), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
