"""Round 4b acceptance (functional): one full-length run, reported, not judged.

    python scripts/acceptance/round4b_acceptance.py --world-seed 1 --history-seed 1

Reports the run time, the final population, when each technique after iron first
appeared (and whether by invention), how many settlements worked each mineral, and the
event counts. Writes runs/round4b_acceptance_w<world>_h<history>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from deluge.config import load_config
from deluge.data import chronicle
from deluge.data.events import ACTIONS, decode
from deluge.model import DelugeModel
from deluge.world.worlddata import MINERALS

REPO_ROOT = Path(__file__).resolve().parents[2]
LATE_AFTER = "iron"  # techniques listed after this one in the graph are the late graph


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--history-seed", type=int, default=1)
    parser.add_argument("--years", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    t0 = time.perf_counter()
    m = DelugeModel(config=cfg, world_seed=args.world_seed, history_seed=args.history_seed, years=args.years)
    setup = time.perf_counter() - t0
    m.run()
    seconds = time.perf_counter() - t0
    g = m.knowledge
    late = g.names[g.names.index(LATE_AFTER) + 1:]
    t = m.settlements
    ids = t.alive_ids()
    log = m.events
    worked = {}
    for i in range(len(log)):
        if ACTIONS[log.action[i]] == "used" and decode(log.actors[i][0])[0] == "deposit":
            mineral = MINERALS[decode(log.actors[i][0])[1] % len(MINERALS)]
            worked[mineral] = worked.get(mineral, 0) + 1
    text = chronicle.Chronicle(m).text()  # every event renders
    result = {
        "world_seed": args.world_seed,
        "history_seed": args.history_seed,
        "years": m.year,
        "seconds": round(seconds, 1),
        "setup_seconds": round(setup, 1),
        "population_end": round(float(m.population)),
        "settlements_end": int(len(ids)),
        "deposit_cells": {mn: int((m.deposits[:, i] > 0).sum()) for i, mn in enumerate(MINERALS)},
        "late_first_known_year": {n: int(m.first_known_year[g.index(n)]) for n in late},
        "late_first_invented_year": {n: int(m.first_invented_year[g.index(n)]) for n in late},
        "late_holders_end": {n: int(t.knows[ids, g.index(n)].sum()) for n in late},
        "all_first_known_year": {n: int(y) for n, y in zip(g.names, m.first_known_year)},
        "holders_end": {n: int(t.knows[ids, g.index(n)].sum()) for n in g.names},
        "deposit_work_events": worked,
        "events": len(log),
        "chronicle_lines": text.count("\n"),
        "events_by_action": {ACTIONS[a]: int(n) for a, n in enumerate(np.bincount(log.action, minlength=len(ACTIONS)))
                             if n},
    }
    out = REPO_ROOT / "runs" / f"round4b_acceptance_w{args.world_seed}_h{args.history_seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
