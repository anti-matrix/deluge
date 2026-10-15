"""Round 5 acceptance: polities and cohesion, appearance and dialect (spec sections 8, 13).

    python scripts/acceptance/round5_acceptance.py --world-seed 1 --jobs 1:12900 2:3000 3:3000

Each job is history_seed:years, run in parallel (at most cores - 1 processes). Every job
takes snapshots every ``--snapshot`` years: polity count and sizes, cohesion, the polity
tree so far, and appearance and dialect measures. Reported per job: run time, final
population, polity count over time (and whether it rises and falls), births by cause,
the largest polity's size and lifespan, whether every polity traces to a founding group,
and how appearance and dialect drifted and blended. Writes
runs/round5_acceptance_w<world>.json and runs/round5_polity_tree_w<world>_h<history>.parquet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from deluge.data import chronicle
from deluge.data.events import ACTIONS, decode
from deluge.model import DelugeModel
from deluge.systems import identity

REPO_ROOT = Path(__file__).resolve().parents[2]


def snapshot(m: DelugeModel) -> dict:
    P, t = m.polities, m.settlements
    pids = P.alive_ids()
    ids = t.alive_ids()
    sizes = np.bincount(t.polity[ids], minlength=P.n)[pids]
    app, dia = identity.lineage_means(m)
    sid0 = [f.settlement for f in m.lineages.founders]
    return {
        "year": m.year,
        "population": round(float(m.population)),
        "settlements": int(len(ids)),
        "polities": int(len(pids)),
        "polities_ever": int(P.n),
        "settlements_per_polity": {"median": float(np.median(sizes)), "p90": float(np.quantile(sizes, 0.9)),
                                   "max": int(sizes.max())},
        "cohesion": {"median": round(float(np.median(P.cohesion[pids])), 3),
                     "p90": round(float(np.quantile(P.cohesion[pids], 0.9)), 3)},
        "monuments_maintained": int(m.monuments.maintained.sum()) if len(m.monuments) else 0,
        # How far each surviving people's mean appearance / dialect moved from its founding band's.
        "lineage_appearance_shift": [None if np.isnan(a).any() else round(float(np.linalg.norm(a - m.founder_markers[0][g])), 3)
                                     for g, a in enumerate(app)],
        "lineage_dialect_shift": [None if np.isnan(d).any() else round(float(np.linalg.norm(d - m.founder_markers[1][g])), 3)
                                  for g, d in enumerate(dia)],
        "spread": {k: round(v, 4) for k, v in identity.spread(m).items()},
        "contacts": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in identity.contact_distances(m).items()},
        "founder_settlements_alive": int(sum(t.ended[s] < 0 for s in sid0)),
    }


def run_job(world_seed: int, history_seed: int, years: int, every: int) -> dict:
    t0 = time.perf_counter()
    m = DelugeModel(world_seed=world_seed, history_seed=history_seed, years=years)
    setup = time.perf_counter() - t0
    t = m.settlements
    sid0 = [f.settlement for f in m.lineages.founders]
    # Founding markers, read through the identity system's own measure at year 0.
    m.founder_markers = identity.lineage_means(m)
    founder_contact = {
        "appearance": float(np.mean([np.linalg.norm(m.founder_markers[0][a] - m.founder_markers[0][b])
                                     for a in range(5) for b in range(a + 1, 5)])),
        "dialect": float(np.mean([np.linalg.norm(m.founder_markers[1][a] - m.founder_markers[1][b])
                                  for a in range(5) for b in range(a + 1, 5)])),
    }
    snaps = [snapshot(m)]
    while m.running:
        m.step()
        if m.year % every == 0 or not m.running:
            snaps.append(snapshot(m))
    seconds = time.perf_counter() - t0
    P = m.polities
    df = m.datacollector.get_model_vars_dataframe()
    n_pol = df["n_polities"].to_numpy()
    tree = m.lineages.polity_tree(m)
    tree.to_parquet(REPO_ROOT / "runs" / f"round5_polity_tree_w{world_seed}_h{history_seed}.parquet", index=False)
    founders = {int(P.lineage[p]): p for p in range(5)}
    traced = all(m.lineages.polity_ancestry(P, pid)[-1] == founders[int(P.lineage[pid])] for pid in range(P.n))
    life = np.where(P.ended[: P.n] >= 0, P.ended[: P.n], m.year) - P.founded[: P.n]
    big = int(np.argmax(P.peak_settlements[: P.n]))
    big_pop = int(np.argmax(P.peak_population[: P.n]))
    ended = P.ended[: P.n] >= 0
    # Rise and fall: the polity count's largest drop from a running maximum, and how many
    # century steps it fell.
    centuries = n_pol[::100]
    running_max = np.maximum.accumulate(n_pol)
    log = m.events
    polity_lines = sum(1 for i in range(len(log)) if log.actors[i] and decode(log.actors[i][0])[0] == "polity")
    text = chronicle.Chronicle(m).text()
    return {
        "history_seed": history_seed,
        "years": m.year,
        "seconds": round(seconds, 1),
        "setup_seconds": round(setup, 1),
        "population_end": round(float(m.population)),
        "settlements_end": int(len(t.alive_ids())),
        "polities_end": int(len(P.alive_ids())),
        "polity_count_every_500y": [int(x) for x in n_pol[::500]] + [int(n_pol[-1])],
        "polity_count_peak": int(n_pol.max()),
        "polity_count_peak_year": int(np.argmax(n_pol)),
        "polity_count_largest_drop": int((running_max - n_pol).max()),
        "polity_count_centuries_falling": int((np.diff(centuries) < 0).sum()),
        "polity_count_centuries": int(len(centuries) - 1),
        "polity_births": int(P.n - 5),
        "secessions": int(df["secessions"].sum()),
        "fragmentations": int(df["fragmentations"].sum()),
        "extinctions": int(df["polity_extinctions"].sum()),
        "mergers": int(df["mergers"].sum()),
        "monuments_built": int(df["monuments_built"].sum()),
        "lifespan_years": {"median": float(np.median(life[ended])) if ended.any() else None,
                           "p90": float(np.quantile(life[ended], 0.9)) if ended.any() else None,
                           "max": int(life.max())},
        "largest_by_settlements": {"polity": big, "group": m.lineages.names[int(P.lineage[big])],
                                   "peak_settlements": int(P.peak_settlements[big]),
                                   "peak_population": round(float(P.peak_population[big])),
                                   "founded": int(P.founded[big]), "ended": int(P.ended[big]),
                                   "lifespan": int(life[big])},
        "largest_by_population": {"polity": big_pop, "group": m.lineages.names[int(P.lineage[big_pop])],
                                  "peak_settlements": int(P.peak_settlements[big_pop]),
                                  "peak_population": round(float(P.peak_population[big_pop])),
                                  "founded": int(P.founded[big_pop]), "ended": int(P.ended[big_pop]),
                                  "lifespan": int(life[big_pop])},
        "every_polity_traces_to_a_founder": bool(traced),
        "polities_per_group_end": {m.lineages.names[g]: int((P.lineage[P.alive_ids()] == g).sum()) for g in range(5)},
        "founder_mean_pair_distance": {k: round(v, 3) for k, v in founder_contact.items()},
        "polity_chronicle_lines": polity_lines,
        "chronicle_lines": text.count("\n"),
        "events_by_action": {ACTIONS[a]: int(n) for a, n in enumerate(np.bincount(log.action, minlength=len(ACTIONS)))
                             if n},
        "snapshots": snaps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--jobs", nargs="+", default=["1:12900", "2:3000", "3:3000"], help="history_seed:years")
    parser.add_argument("--snapshot", type=int, default=500, help="years between snapshots")
    args = parser.parse_args(argv)
    jobs = [tuple(int(x) for x in j.split(":")) for j in args.jobs]
    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, args.world_seed, h, y, args.snapshot) for h, y in jobs]
        results = [f.result() for f in futures]
    out = REPO_ROOT / "runs" / f"round5_acceptance_w{args.world_seed}.json"
    out.write_text(json.dumps({"world_seed": args.world_seed, "runs": results}, indent=1), encoding="utf-8")
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k != "snapshots"}, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
