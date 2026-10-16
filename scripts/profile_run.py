"""Time or profile one headless run (Round 11), run manually:

    python scripts/profile_run.py --world-seed 1 --history-seed 1 --years 12900 [--profile out.prof]

Prints the seconds spent stepping up to each checkpoint year and in all; with ``--profile``,
runs under cProfile and prints the top ``--top`` functions by cumulative time. With
``--digest`` it prints the full-state digest at the end (``deluge/data/state.py``).
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time


def main(argv: list[str] | None = None) -> int:
    from deluge.data.state import state_digest
    from deluge.model import DelugeModel

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-seed", type=int, default=1)
    p.add_argument("--history-seed", type=int, default=1)
    p.add_argument("--years", type=int, default=3000)
    p.add_argument("--checkpoints", type=int, nargs="*", default=[1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000])
    p.add_argument("--profile", default=None, help="write cProfile stats here")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--digest", action="store_true")
    p.add_argument("--out", default=None, help="write the timings as JSON here")
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    m = DelugeModel(world_seed=args.world_seed, history_seed=args.history_seed, years=args.years)
    setup = time.perf_counter() - t0
    prof = cProfile.Profile() if args.profile else None
    marks = {}
    t1 = time.perf_counter()
    if prof:
        prof.enable()
    for target in sorted({c for c in args.checkpoints if c < args.years} | {args.years}):
        while m.running and m.year < target:
            m.step()
        marks[target] = round(time.perf_counter() - t1, 1)
        print(f"year {m.year}: {marks[target]} s", flush=True)
    if prof:
        prof.disable()
        prof.dump_stats(args.profile)
        pstats.Stats(prof).sort_stats("cumulative").print_stats(args.top)
    out = {"years": m.year, "setup_seconds": round(setup, 1), "step_seconds": marks[args.years], "at_year": marks,
           "polities": int(m.n_polities), "population": round(float(m.population))}
    if args.digest:
        out["digest"] = state_digest(m)
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
