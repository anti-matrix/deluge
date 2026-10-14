"""Headless single run.

    python scripts/run.py --config configs/default.yaml --seed 42

Writes runs/<run_id>/model.parquet and runs/<run_id>/meta.json.
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

from deluge.config import DEFAULT_CONFIG_PATH, load_config
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None, help="overrides run.seed")
    parser.add_argument("--years", type=int, default=None, help="overrides run.years")
    parser.add_argument("--out", default=None, help="run directory (default: <runs_dir>/<run_id>)")
    args = parser.parse_args(argv)

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.years is not None:
        overrides["years"] = args.years
    cfg = load_config(args.config, {"run": overrides} if overrides else None)

    started = datetime.now(timezone.utc)
    run_id = f"{started:%Y%m%dT%H%M%S}_seed{cfg.run.seed}"
    out_dir = Path(args.out) if args.out else REPO_ROOT / cfg.output.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = DelugeModel(config=cfg)
    fingerprint = rng_fingerprint(model)
    model.run()
    runtime = time.perf_counter() - t0

    model.datacollector.get_model_vars_dataframe().to_parquet(out_dir / "model.parquet", index=True)
    meta = {
        "run_id": run_id,
        "seed": cfg.run.seed,
        "years": cfg.run.years,
        "final_year": model.year,
        "config_path": str(Path(args.config).resolve()),
        "config_hash": cfg.hash(),
        "rng_fingerprint": fingerprint,
        "git_commit": git_commit(),
        "started_utc": started.isoformat(),
        "runtime_seconds": runtime,
        "python": platform.python_version(),
        "mesa": mesa.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{model.year} years in {runtime:.3f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
