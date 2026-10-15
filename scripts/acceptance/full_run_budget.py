"""One full-length run (run.years, 12,900) and the spec's time budget, run manually:

    python scripts/acceptance/full_run_budget.py --world-seed 3 --history-seed 3

Checks that the run steps through every year, stops, collects one row per year, and
finishes within the budget (spec section 14: 10 minutes; Round 11 profiles and speeds
up). Exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

FULL_RUN_BUDGET_S = 600.0


def main(argv: list[str] | None = None) -> int:
    from deluge.model import DelugeModel

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=3)
    parser.add_argument("--history-seed", type=int, default=3)
    parser.add_argument("--budget", type=float, default=FULL_RUN_BUDGET_S, help="seconds")
    args = parser.parse_args(argv)
    m = DelugeModel(world_seed=args.world_seed, history_seed=args.history_seed)
    years = m.config.run.years
    t0 = time.perf_counter()
    m.run()
    elapsed = time.perf_counter() - t0
    df = m.datacollector.get_model_vars_dataframe()
    checks = {
        "all_years_stepped": m.year == years and m.steps == years and not m.running,
        "one_row_per_year": list(df["year"]) == list(range(years)),
        "within_budget": elapsed < args.budget,
    }
    print(json.dumps({"years": years, "seconds": round(elapsed, 1), "checks": checks}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
