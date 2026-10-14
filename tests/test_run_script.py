"""Round 0 acceptance: run.py completes 11,000 empty steps in under 5 seconds, and two
runs with the same seed produce identical output."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "scripts" / "run.py"


def _run(out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict, float]:
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, str(RUN), "--seed", str(seed), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(out_dir / "model.parquet"), meta, wall


def test_run_script_acceptance(tmp_path):
    df_a, meta_a, wall_a = _run(tmp_path / "a", seed=42)
    df_b, meta_b, _ = _run(tmp_path / "b", seed=42)
    df_c, meta_c, wall_c = _run(tmp_path / "c", seed=43)

    assert meta_a["final_year"] == 11000
    assert len(df_a) == 11000
    # The 11,000 steps themselves.
    assert meta_a["runtime_seconds"] < 5.0, f"steps took {meta_a['runtime_seconds']:.2f}s"
    # Whole process including interpreter start-up and `import mesa` (~2.3 s on its own),
    # measured on a warm launch: the first launch after an install can exceed this on
    # Windows while .pyc files are written and scanned.
    assert wall_c < 5.0, f"run.py took {wall_c:.2f}s (warm)"

    pd.testing.assert_frame_equal(df_a, df_b)
    assert meta_a["rng_fingerprint"] == meta_b["rng_fingerprint"]
    assert meta_a["config_hash"] == meta_b["config_hash"]

    assert meta_a["rng_fingerprint"] != meta_c["rng_fingerprint"]
