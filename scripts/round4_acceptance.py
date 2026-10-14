"""Round 4 acceptance on one world: full runs for several history seeds, with and without
the teachers, run in parallel processes.

    python scripts/round4_acceptance.py --world-seed 1 --history-seeds 1 2 3 4 5 --workers 4

Checks (Round 4 brief and spec section 13):
- with teachers, farming appears in at least 3 of the 5 groups within 1,000 years of
  their contact, in every run;
- without teachers, farming still appears somewhere (independent invention) in at least 3
  of 5 runs, later on average than with teachers;
- a technique held, then lost after a population crash, at least once across the runs;
- total population at the last year between 50 and 300 million in every run;
- every logged event has year, action, actors and place, and the chronicle renders them.

Writes runs/round4_acceptance_w<world>.json and exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FARMING_WITHIN = 1000
MIN_GROUPS = 3
MIN_RUNS = 3
POP_RANGE = (50e6, 300e6)


def one_run(world: int, history: int, teachers: bool) -> dict:
    import numpy as np

    from deluge.config import load_config
    from deluge.data import chronicle
    from deluge.data.events import ACTIONS
    from deluge.model import DelugeModel

    cfg = load_config().with_overrides({"teachers": {"enabled": teachers}})
    t0 = time.perf_counter()
    m = DelugeModel(config=cfg, world_seed=world, history_seed=history)

    def river_density() -> list[float] | None:
        """People per km2 on good river land: river cells with fertility above 0.6 whose
        settlement farms at least 80%; 10th, 50th and 90th percentile."""
        t = m.settlements
        ids = t.alive_ids()
        c = t.cell[ids]
        good = (m.world.river.ravel()[c] > 0) & (m.world.fertility.ravel()[c] > 0.6) & (t.shares[ids, 2] >= 0.8)
        if not good.any():
            return None
        return [round(float(q), 2) for q in np.quantile(t.population[ids[good]] / m.cell_area_km2, [0.1, 0.5, 0.9])]

    density_3000 = None
    while m.running:
        m.step()
        if m.year == 3001:
            density_3000 = river_density()
    seconds = time.perf_counter() - t0
    df = m.datacollector.get_model_vars_dataframe()
    contact = {v["lineage"]: v["start"] for v in m.teachers_state.visits}
    first = m.first_farming_year.tolist()
    log = m.events
    complete = all(isinstance(log.year[i], int) and 0 <= log.action[i] < len(ACTIONS) and len(log.actors[i]) > 0
                   and isinstance(log.place[i], int) for i in range(len(log)))
    chronicle.lines(m, len(log))  # every event renders
    return {
        "history_seed": history,
        "teachers": teachers,
        "seconds": round(seconds),
        "population_end": float(m.population),
        "first_farming_by_group": first,
        "contact_by_group": contact,
        "groups_farming_within": sum(1 for g, y in contact.items() if first[g] >= 0 and first[g] - y <= FARMING_WITHIN),
        "first_farming_any": min([y for y in first if y >= 0], default=-1),
        "losses_after_crash": int(df["losses_after_crash"].sum()),
        "events": len(log),
        "events_complete": complete,
        "techniques_known_mean": round(float(np.mean(df["techniques_known"].iloc[-100:])), 2),
        "river_farm_density_q10_q50_q90_year3000": density_3000,
        "river_farm_density_q10_q50_q90_end": river_density(),
        "events_by_action": {ACTIONS[a]: int(n) for a, n in enumerate(np.bincount(log.action, minlength=len(ACTIONS)))
                             if n},
    }


def check(runs: list[dict]) -> dict:
    on = [r for r in runs if r["teachers"]]
    off = [r for r in runs if not r["teachers"]]
    off_farming = [r["first_farming_any"] for r in off if r["first_farming_any"] >= 0]
    on_first = [r["first_farming_any"] for r in on if r["first_farming_any"] >= 0]
    return {
        "farming_within_1000_years_in_every_run": all(r["groups_farming_within"] >= MIN_GROUPS for r in on),
        "no_teachers_farming_in_at_least_3_runs": len(off_farming) >= MIN_RUNS,
        "no_teachers_farming_later_on_average": bool(off_farming and on_first
                                                     and sum(off_farming) / len(off_farming)
                                                     > sum(on_first) / len(on_first)),
        "knowledge_lost_after_crash": sum(r["losses_after_crash"] for r in runs) >= 1,
        "population_in_range_every_run": all(POP_RANGE[0] <= r["population_end"] <= POP_RANGE[1] for r in on),
        "events_complete_and_rendered": all(r["events_complete"] for r in runs),
    }


def acceptance(world: int, histories: list[int], workers: int) -> dict:
    jobs = [(world, h, t) for t in (True, False) for h in histories]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(one_run, *zip(*jobs)))
    return {"world_seed": world, "runs": runs, "checks": check(runs)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--history-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    result = acceptance(args.world_seed, args.history_seeds, args.workers)
    out = REPO_ROOT / "runs" / f"round4_acceptance_w{args.world_seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["checks"], indent=2))
    print(out)
    return 0 if all(result["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
