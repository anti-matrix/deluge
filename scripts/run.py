"""Headless single run.

    python scripts/run.py --config configs/default.yaml --world-seed 1 --history-seed 42

Writes runs/<run_id>/model.parquet, events.parquet (the raw event log), entities.parquet
(kind and label of every entity id in the log) and meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mesa
import pandas as pd

from deluge.config import DEFAULT_CONFIG_PATH, load_config, read_overrides
from deluge.data.chronicle import Namer
from deluge.data.events import decode
from deluge.model import DelugeModel

REPO_ROOT = Path(__file__).resolve().parent.parent


def git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return out.stdout.strip() or None


def rng_fingerprint(model: DelugeModel) -> str:
    blob = json.dumps(model.streams.state(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def entity_table(model: DelugeModel) -> pd.DataFrame:
    """Kind and label of every entity id that appears in the event log."""
    ids = sorted({a for actors in model.events.actors for a in actors})
    name = Namer(model)
    rows = [(eid, *decode(eid), name(eid)) for eid in ids]
    return pd.DataFrame(rows, columns=["entity", "kind", "local_id", "label"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overrides", default=None,
                        help="YAML file of nested overrides applied on top of --config "
                             "(e.g. configs/worlds/small_test.yaml)")
    parser.add_argument("--world-seed", type=int, default=None, help="overrides run.world_seed (the map)")
    parser.add_argument("--history-seed", type=int, default=None, help="overrides run.history_seed")
    parser.add_argument("--years", type=int, default=None, help="overrides run.years")
    parser.add_argument("--out", default=None, help="run directory (default: <runs_dir>/<run_id>)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.world_seed is not None:
        overrides["world_seed"] = args.world_seed
    if args.history_seed is not None:
        overrides["history_seed"] = args.history_seed
    if args.years is not None:
        overrides["years"] = args.years
    cfg = load_config(args.config, read_overrides(args.overrides) if args.overrides else None)
    if overrides:
        cfg = cfg.with_overrides({"run": overrides})

    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%S}_w{cfg.run.world_seed}_h{cfg.run.history_seed}"
    out_dir = Path(args.out) if args.out else REPO_ROOT / cfg.output.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = DelugeModel(config=cfg)
    setup = time.perf_counter() - t0
    fingerprint = rng_fingerprint(model)
    t1 = time.perf_counter()
    model.run()
    stepping = time.perf_counter() - t1
    runtime = time.perf_counter() - t0

    model.datacollector.get_model_vars_dataframe().to_parquet(out_dir / "model.parquet", index=True)
    model.events.write(out_dir / "events.parquet")
    entity_table(model).to_parquet(out_dir / "entities.parquet", index=False)
    meta = {
        "run_id": run_id,
        "world_seed": cfg.run.world_seed,
        "history_seed": cfg.run.history_seed,
        "years": cfg.run.years,
        "final_year": model.year,
        "config_path": str(Path(args.config).resolve()),
        "overrides_path": str(Path(args.overrides).resolve()) if args.overrides else None,
        "config_hash": cfg.hash(),
        "rng_fingerprint": fingerprint,
        "git_commit": git_commit(),
        "started_utc": started.isoformat(),
        "runtime_seconds": runtime,
        "setup_seconds": setup,  # world generation and model construction
        "step_seconds": stepping,
        "python": platform.python_version(),
        "mesa": mesa.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{model.year} years in {stepping:.3f}s (setup {setup:.3f}s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
