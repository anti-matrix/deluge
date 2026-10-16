"""Full-state digests of the two reference runs (``tests/test_state_identity.py``), run manually:

    python scripts/state_digests.py            # print them
    python scripts/state_digests.py --write    # store them as the new reference

The reference runs: the small test world for 500 years, and world seed 1, history seed 1
for 1,000 years. A change meant to leave results identical (Round 11) must leave both
digests as stored; a change meant to alter results rewrites them, in its own commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "tests" / "data" / "state_digests.json"
RUNS = {
    "small_500": {"small": True, "years": 500},
    "w1_h1_1000": {"small": False, "world_seed": 1, "history_seed": 1, "years": 1000},
}


def digest_of(name: str) -> str:
    from deluge.config import load_config, read_overrides
    from deluge.data.state import state_digest
    from deluge.model import DelugeModel

    spec = dict(RUNS[name])
    overrides = read_overrides(REPO_ROOT / "configs" / "worlds" / "small_test.yaml") if spec.pop("small") else None
    m = DelugeModel(config=load_config(overrides=overrides), **spec)
    m.run()
    return state_digest(m)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--write", action="store_true")
    args = p.parse_args(argv)
    out = {name: digest_of(name) for name in args.runs}
    print(json.dumps(out, indent=1))
    if args.write:
        ref = json.loads(REFERENCE.read_text(encoding="utf-8")) if REFERENCE.exists() else {}
        ref.update(out)
        REFERENCE.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE.write_text(json.dumps(ref, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
