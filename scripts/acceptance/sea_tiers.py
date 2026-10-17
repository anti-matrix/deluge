"""Sea tiers (Round 11 step 1), run manually: one history measured for comparison.

    python scripts/acceptance/sea_tiers.py --world-seed 1 --history-seed 1 --years 3000 [--root DIR]

``--root`` imports the sim from another checkout (a worktree of an earlier commit), so the
same measurement runs on old and new code. Measured (analysis side only; the wrappers record
what the sim does and change nothing):

- polities at the end, polity births, mean cohesion (mean over years), mergers, battles,
  median lifespan of ended polities;
- sea crossings by type (founding, move, join, core migration, war) and by tier, both the
  tier the crossing needed (sea cells: up to sailing's, up to open-sea voyaging's, beyond)
  and the highest tier the crossing settlement held;
- the first crossing to another landmass (land connected components of that year) per tier;
- first year each sea technique was held and first invented;
- whether any living store holds origin records (years 0 to ``reports.origins_until_year``)
  at the checkpoint years, and how many of them still hold early fields;
- run time of the steps (measurement at checkpoints excluded).

Writes the origins report (``analysis/narrator.py``) at the end with ``--report STEM``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SEA_TECHNIQUES = ("sailing", "open_sea_voyaging", "ocean_navigation")
TIER_NAMES = ("sailing", "open-sea voyaging", "ocean navigation")
NEEDED_LIMITS = (3, 8)  # sea cells: <= 3 sailing, <= 8 open-sea voyaging, more: ocean navigation


def needed_tier(sea_cells: int) -> str:
    for limit, name in zip(NEEDED_LIMITS, TIER_NAMES):
        if sea_cells <= limit:
            return name
    return TIER_NAMES[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-seed", type=int, default=1)
    p.add_argument("--history-seed", type=int, default=1)
    p.add_argument("--years", type=int, default=3000)
    p.add_argument("--root", default=str(REPO_ROOT))
    p.add_argument("--checkpoints", type=int, nargs="+", default=[1000, 2000, 3000])
    p.add_argument("--report", default=None, help="stem of the origins report written at the end")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    from analysis.provenance import History, origin_records
    from deluge.data.events import ACTIONS, decode
    from deluge.model import DelugeModel
    import deluge
    if root not in Path(deluge.__file__).resolve().parents:
        raise RuntimeError(f"deluge imported from {deluge.__file__}, not from {root}")

    t0 = time.perf_counter()
    m = DelugeModel(world_seed=args.world_seed, history_seed=args.history_seed, years=args.years)
    g, geom = m.knowledge, m.geom
    steps = g.effect["sea_steps"]
    tier_of_steps = {}
    for name, label in zip(SEA_TECHNIQUES, TIER_NAMES):
        if name in g.names:
            tier_of_steps[int(steps[g.index(name)])] = label

    def held_tier(sid: int) -> str:
        held = m.settlements.knows[sid] & (steps > 0)
        if not held.any():
            return "none"
        top = int(steps[held].max())
        return tier_of_steps.get(top, f"{top} sea cells")

    labels_cache = {"sea": None, "labels": None}

    def landmass(cell: int) -> int:
        if labels_cache["sea"] is None or not np.array_equal(labels_cache["sea"], m.is_sea):
            land = ~m.is_sea
            c, k = np.nonzero(geom.neighbors >= 0)
            t = geom.neighbors[c, k]
            ok = land[c] & land[t]
            n = geom.n_cells
            graph = csr_matrix((np.ones(int(ok.sum())), (c[ok], t[ok])), shape=(n, n))
            _, lab = connected_components(graph, directed=False)
            labels_cache["sea"], labels_cache["labels"] = m.is_sea.copy(), np.where(land, lab, -1)
        return int(labels_cache["labels"][cell])

    previous_cell: dict[int, int] = {}
    orig_move = m.move_settlement

    def move_settlement(sid, cell):
        previous_cell[int(sid)] = int(m.settlements.cell[sid])
        orig_move(sid, cell)

    m.move_settlement = move_settlement
    crossings = []
    orig_log_crossing = m.log_crossing

    def log_crossing(sea_cells, actors, place, link, origin_cell=-1):
        if sea_cells > 0:
            kinds = [k for k, _ in actors]
            if kinds[0] == "polity":
                kind, src = "war", actors[1][1]
                origin = int(m.settlements.cell[src])
            else:
                act = ACTIONS[m.events.action[link]] if link >= 0 else ""
                if act == "founded":
                    kind, src = "founding", actors[1][1]
                    origin = int(m.settlements.cell[src])
                elif act == "joined":
                    kind, src = "join", actors[0][1]
                    origin = int(m.settlements.cell[src])
                else:
                    kind = "core migration" if "polity" in kinds else "move"
                    src = actors[0][1]
                    origin = previous_cell.get(int(src), -1)
            a = landmass(origin) if origin >= 0 else -1
            b = landmass(int(place))
            crossings.append({"year": m.year, "type": kind, "sea_cells": int(sea_cells),
                              "needed": needed_tier(int(sea_cells)), "held": held_tier(int(src)),
                              "other_landmass": bool(a >= 0 and b >= 0 and a != b)})
        orig_log_crossing(sea_cells, actors, place, link, origin_cell)

    m.log_crossing = log_crossing

    stepping = 0.0
    origins = {}
    for target in sorted(set(args.checkpoints) | {args.years}):
        s0 = time.perf_counter()
        while m.running and m.year < target:
            m.step()
        stepping += time.perf_counter() - s0
        if target in args.checkpoints:
            hist = History(m)
            per_group, current = {}, 0
            for gi, gname in enumerate(m.lineages.names):
                recs = origin_records(m, gi, hist)
                per_group[gname] = len(recs)
                current += sum(r["current"] for r in recs)
            origins[str(target)] = {"entries": sum(per_group.values()), "holding_early_fields": current,
                                    "per_group": per_group}

    df = m.datacollector.get_model_vars_dataframe()
    P = m.polities
    n = P.n
    life = np.where(P.ended[:n] >= 0, P.ended[:n], m.year) - P.founded[:n]
    ended = P.ended[:n] >= 0
    by_type = Counter(c["type"] for c in crossings)
    by_needed = Counter(c["needed"] for c in crossings)
    by_held = Counter(c["held"] for c in crossings)
    by_type_needed = Counter(f"{c['type']} / {c['needed']}" for c in crossings)
    first_other = {}
    for c in crossings:
        if c["other_landmass"]:
            for key in (f"needed {c['needed']}", f"held {c['held']}"):
                first_other.setdefault(key, {"year": c["year"], "type": c["type"], "sea_cells": c["sea_cells"]})
    first_known = {name: (int(m.first_known_year[g.index(name)]) if name in g.names else None)
                   for name in SEA_TECHNIQUES}
    first_invented = {name: (int(m.first_invented_year[g.index(name)]) if name in g.names else None)
                      for name in SEA_TECHNIQUES}
    out = {
        "root": str(root), "world_seed": args.world_seed, "history_seed": args.history_seed, "years": m.year,
        "seconds_steps": round(stepping, 1), "seconds_total": round(time.perf_counter() - t0, 1),
        "polities_end": int(len(P.alive_ids())), "polity_births": int(n - len(m.lineages.founders)),
        "cohesion_mean": round(float(df["mean_cohesion"].mean()), 4),
        "mergers": int(df["mergers"].sum()), "battles": int(df["battles"].sum()),
        "lifespan_median": float(np.median(life[ended])) if ended.any() else None,
        "population_end": round(float(m.population)),
        "sea_crossings": len(crossings), "sea_crossings_counted": int(df["sea_crossings"].sum()),
        "by_type": dict(by_type), "by_needed_tier": dict(by_needed), "by_held_tier": dict(by_held),
        "by_type_and_needed_tier": dict(sorted(by_type_needed.items())),
        "crossings_to_other_landmass": sum(c["other_landmass"] for c in crossings),
        "first_to_other_landmass": first_other,
        "first_known_year": first_known, "first_invented_year": first_invented,
        "origin_records": origins,
    }
    if args.report:
        from analysis import narrator
        md, js = narrator.write(m, args.report)
        out["report"] = [str(md), str(js)]
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
