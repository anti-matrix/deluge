"""Headless run.py: completes 11,000 years within budget, and two runs with the same seed
produce identical output."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from test_model import FULL_RUN_BUDGET_S

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "scripts" / "run.py"


def _run(out_dir: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    subprocess.run(
        [sys.executable, str(RUN), "--seed", str(seed), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(out_dir / "model.parquet"), meta


def test_run_script_acceptance(tmp_path):
    df_a, meta_a = _run(tmp_path / "a", seed=42)
    df_b, meta_b = _run(tmp_path / "b", seed=42)
    df_c, meta_c = _run(tmp_path / "c", seed=43)

    assert meta_a["final_year"] == 11000
    assert len(df_a) == 11000
    assert meta_a["runtime_seconds"] < FULL_RUN_BUDGET_S, f"run took {meta_a['runtime_seconds']:.2f}s"

    pd.testing.assert_frame_equal(df_a, df_b)
    assert meta_a["rng_fingerprint"] == meta_b["rng_fingerprint"]
    assert meta_a["config_hash"] == meta_b["config_hash"]

    assert meta_a["rng_fingerprint"] != meta_c["rng_fingerprint"]
    assert not df_a.equals(df_c)
