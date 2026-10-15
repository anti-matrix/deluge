"""Round 6 acceptance: war and conquest (spec sections 9, 13), run manually.

    python scripts/acceptance/round6_acceptance.py --world-seed 1 --jobs 1:5000 2:5000 3:5000

Each job is history_seed:years, run in parallel (at most cores - 1 processes). Reported per
job: run time, polity births, polity count over time (and whether it rises and falls),
the largest polity's size and lifespan, wars, battles and conquests per millennium,
median cohesion over time, large states and whether they later broke up, and how
appearance and dialect blended across founding peoples after conquest.

Large states: polities whose peak reached ``--large`` settlements. A large state broke up
when it ended fragmented, or when settlements seceding from it (and from what it
fragmented into) reached half its peak size.

Blending after conquest: every settlement taken by a polity whose capital belongs to
another founding people is tracked from the year it is taken: appearance and dialect
distance to its ruler's capital then, and at each later snapshot while it is still ruled
by the same polity. Writes runs/round6_acceptance_w<world>.json.
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

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_job(world_seed: int, history_seed: int, years: int, every: int, large: int) -> dict:
    from deluge.agents.polity import END_CAUSES
    from deluge.data import chronicle
    from deluge.data.events import ACTION_CODE, ACTIONS, decode
    from deluge.model import DelugeModel
    from deluge.systems import identity

    t0 = time.perf_counter()
    m = DelugeModel(world_seed=world_seed, history_seed=history_seed, years=years)
    t, P = m.settlements, m.polities
    took = ACTION_CODE["took"]
    tracked: dict[int, dict] = {}  # conquered settlement -> its record
    seen = 0
    snaps = []

    def markers(sid: int, other: int) -> tuple[float, float]:
        # Distances between two settlements' markers (analysis only, no effect on the run).
        app = np.linalg.norm(t.appearance_mean[sid] - t.appearance_mean[other])
        dia = np.linalg.norm(t.dialect[sid] - t.dialect[other])
        return float(app), float(dia)

    while m.running:
        m.step()
        log = m.events
        for i in range(seen, len(log)):
            if log.action[i] != took:
                continue
            ruler = decode(log.actors[i][0])[1]
            sid = decode(log.actors[i][1])[1]
            cap = int(P.capital[ruler])
            if t.lineage[sid] != t.lineage[cap] and sid not in tracked:
                app, dia = markers(sid, cap)
                tracked[sid] = {"settlement": sid, "year": int(log.year[i]), "ruler": ruler,
                                "people": m.lineages.names[int(t.lineage[sid])],
                                "ruler_people": m.lineages.names[int(t.lineage[cap])],
                                "cell": [int(v) for v in m.geom.xy(int(t.cell[sid]))],
                                "appearance_0": round(app, 4), "dialect_0": round(dia, 4), "later": []}
        seen = len(log)
        if m.year % every == 0 or not m.running:
            pids = P.alive_ids()
            ids = t.alive_ids()
            for rec in tracked.values():
                sid, ruler = rec["settlement"], rec["ruler"]
                if t.ended[sid] < 0 and t.polity[sid] == ruler and P.ended[ruler] < 0:
                    app, dia = markers(sid, int(P.capital[ruler]))
                    rec["later"].append([m.year, round(app, 4), round(dia, 4)])
            sizes = np.bincount(t.polity[ids], minlength=P.n)[pids]
            snaps.append({"year": m.year, "population": round(float(m.population)), "polities": int(len(pids)),
                          "largest_settlements": int(sizes.max()) if len(sizes) else 0,
                          "cohesion_median": round(float(np.median(P.cohesion[pids])), 3) if len(pids) else None,
                          "contacts": {k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in identity.contact_distances(m).items()}})
    seconds = time.perf_counter() - t0
    df = m.datacollector.get_model_vars_dataframe()
    n_pol = df["n_polities"].to_numpy()
    n = P.n
    life = np.where(P.ended[:n] >= 0, P.ended[:n], m.year) - P.founded[:n]
    ended = P.ended[:n] >= 0
    big = int(np.argmax(P.peak_settlements[:n]))
    centuries = n_pol[::100]
    running_max = np.maximum.accumulate(n_pol)

    # Large states and their breakup.
    parent = P.parent[:n]
    children: dict[int, list[int]] = {}
    for c, p in enumerate(parent.tolist()):
        if p >= 0:
            children.setdefault(p, []).append(c)
    large_states = []
    for pid in np.flatnonzero(P.peak_settlements[:n] >= large).tolist():
        cause = END_CAUSES[int(P.end_cause[pid])]
        kids = children.get(pid, [])
        lost_to_secession = int(sum(P.peak_settlements[k] for k in kids if P.founded[k] > P.founded[pid]))
        large_states.append({
            "polity": pid, "group": m.lineages.names[int(P.lineage[pid])],
            "peak_settlements": int(P.peak_settlements[pid]), "peak_population": round(float(P.peak_population[pid])),
            "founded": int(P.founded[pid]), "ended": int(P.ended[pid]), "end_cause": cause, "lifespan": int(life[pid]),
            "child_polities": len(kids), "settlements_in_children_at_peak": lost_to_secession,
            "broke_up": bool(cause == "fragmented" or lost_to_secession >= P.peak_settlements[pid] / 2),
        })

    per_millennium = {}
    for col in ("battles", "raids", "campaigns", "conquests", "capitals_taken", "polities_destroyed",
                "secessions", "fragmentations"):
        x = df[col].to_numpy()
        per_millennium[col] = [int(x[k:k + 1000].sum()) for k in range(0, len(x), 1000)]
    wars = df["wars_active"].to_numpy()
    per_millennium["wars_active_mean"] = [round(float(wars[k:k + 1000].mean()), 1) for k in range(0, len(wars), 1000)]
    pid_mil = {}
    for k in range(0, m.year, 1000):
        pid_mil[k] = int(((P.founded[:n] >= k) & (P.founded[:n] < k + 1000) & (parent >= 0)).sum())
    per_millennium["polity_births"] = list(pid_mil.values())

    blends = []
    for rec in tracked.values():
        if rec["later"]:
            y, app, dia = rec["later"][-1]
            rec["appearance_end"], rec["dialect_end"], rec["end_year"] = app, dia, y
            rec["appearance_change"] = round(app - rec["appearance_0"], 4)
            rec["dialect_change"] = round(dia - rec["dialect_0"], 4)
            blends.append(rec)
    blends.sort(key=lambda r: r["appearance_change"])
    text = chronicle.Chronicle(m)
    war_lines = text.select(actions={"fought", "won", "took", "lost"})
    destroyed = [i for i in text.select(actions={"died"}) if len(m.events.actors[int(i)]) == 2]
    log = m.events
    return {
        "history_seed": history_seed,
        "years": m.year,
        "seconds": round(seconds, 1),
        "population_end": round(float(m.population)),
        "polities_end": int(len(P.alive_ids())),
        "polity_births": int(n - 5),
        "polity_count_every_250y": [int(x) for x in n_pol[::250]] + [int(n_pol[-1])],
        "polity_count_peak": int(n_pol.max()),
        "polity_count_largest_drop": int((running_max - n_pol).max()),
        "polity_count_centuries_falling": int((np.diff(centuries) < 0).sum()),
        "polity_count_centuries_rising": int((np.diff(centuries) > 0).sum()),
        "polity_count_centuries": int(len(centuries) - 1),
        "per_millennium": per_millennium,
        "cohesion_median_every_500y": [s["cohesion_median"] for s in snaps],
        "largest_settlements_every_500y": [s["largest_settlements"] for s in snaps],
        "lifespan_years": {"median": float(np.median(life[ended])) if ended.any() else None,
                           "p90": float(np.quantile(life[ended], 0.9)) if ended.any() else None},
        "largest_polity": {"polity": big, "group": m.lineages.names[int(P.lineage[big])],
                           "peak_settlements": int(P.peak_settlements[big]),
                           "peak_population": round(float(P.peak_population[big])),
                           "founded": int(P.founded[big]), "ended": int(P.ended[big]),
                           "end_cause": END_CAUSES[int(P.end_cause[big])], "lifespan": int(life[big])},
        "large_states": large_states,
        "end_causes": {c: int((P.end_cause[:n] == k).sum()) for k, c in enumerate(END_CAUSES) if c},
        "conquests_across_peoples_tracked": len(tracked),
        "blends_strongest": blends[:5],
        "blend_summary": {
            "tracked_with_later_snapshot": len(blends),
            "appearance_change_median": round(float(np.median([b["appearance_change"] for b in blends])), 4)
            if blends else None,
            "dialect_change_median": round(float(np.median([b["dialect_change"] for b in blends])), 4)
            if blends else None,
        },
        "war_chronicle_lines": int(len(war_lines)),
        "destroyed_polity_lines": int(len(destroyed)),
        "chronicle_sample": [text.line(int(i)) for i in war_lines[:6]] + [text.line(int(i)) for i in destroyed[:2]],
        "events_by_action": {ACTIONS[a]: int(k) for a, k in enumerate(np.bincount(log.action, minlength=len(ACTIONS)))
                             if k},
        "snapshots": snaps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--jobs", nargs="+", default=["1:5000", "2:5000", "3:5000"], help="history_seed:years")
    parser.add_argument("--snapshot", type=int, default=500, help="years between snapshots")
    parser.add_argument("--large", type=int, default=30, help="settlements at peak for a large state")
    parser.add_argument("--out", default=None, help="output file (default runs/round6_acceptance_w<world>.json)")
    args = parser.parse_args(argv)
    jobs = [tuple(int(x) for x in j.split(":")) for j in args.jobs]
    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, args.world_seed, h, y, args.snapshot, args.large) for h, y in jobs]
        results = [f.result() for f in futures]
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round6_acceptance_w{args.world_seed}.json"
    out.write_text(json.dumps({"world_seed": args.world_seed, "runs": results}, indent=1), encoding="utf-8")
    for r in results:
        print(json.dumps({k: v for k, v in r.items() if k not in ("snapshots", "large_states", "blends_strongest")},
                         indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
