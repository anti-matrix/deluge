"""Round 3 acceptance (foragers) on one world, several history seeds, run manually:

    python scripts/acceptance/round3_acceptance.py --world-seed 1 --history-seeds 1 2 3 4 5

Teachers and invention are switched off, as in Round 3. Checks: total population at
year 3,000 between 0.5 and 10 million in every run, and every founding group survives in
at least one run. Exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

YEARS = 3000
POP_RANGE = (0.5e6, 10e6)
FORAGERS = {"teachers": {"enabled": False}, "knowledge": {"invention_scale": 0.0}}


def one_run(world: int, history: int) -> dict:
    from deluge.config import load_config
    from deluge.model import DelugeModel

    m = DelugeModel(config=load_config(overrides=FORAGERS), world_seed=world, history_seed=history, years=YEARS)
    m.run()
    return {"history_seed": history, "population": float(m.population),
            "lineage_population": [float(x) for x in m.lineage_population]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--history-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args(argv)
    workers = max(1, min(len(args.history_seeds), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = list(pool.map(one_run, [args.world_seed] * len(args.history_seeds), args.history_seeds))
    checks = {
        "population_in_range": all(POP_RANGE[0] <= r["population"] <= POP_RANGE[1] for r in runs),
        "every_group_survives_somewhere": all(any(r["lineage_population"][g] > 0 for r in runs)
                                              for g in range(len(runs[0]["lineage_population"]))),
    }
    print(json.dumps({"runs": runs, "checks": checks}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
