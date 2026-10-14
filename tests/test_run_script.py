"""Headless run.py: two runs with the same seeds produce identical output (model table,
event log, metadata), different seeds differ, and the event log is written.

Since Round 4 a full 11,000-year run takes minutes, so these runs are 2,000 years; the
full-length run and its time budget are checked in test_model.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "scripts" / "run.py"
YEARS = 2000


def _run(out_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    subprocess.run(
        [sys.executable, str(RUN), "--world-seed", str(seed), "--history-seed", str(seed),
         "--years", str(YEARS), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(out_dir / "model.parquet"), pd.read_parquet(out_dir / "events.parquet"), meta


def test_run_script_acceptance(tmp_path):
    df_a, ev_a, meta_a = _run(tmp_path / "a", seed=3)
    df_b, ev_b, meta_b = _run(tmp_path / "b", seed=3)
    df_c, ev_c, meta_c = _run(tmp_path / "c", seed=4)

    assert meta_a["final_year"] == YEARS
    assert (meta_a["world_seed"], meta_a["history_seed"]) == (3, 3)
    assert len(df_a) == YEARS

    pd.testing.assert_frame_equal(df_a, df_b)
    pd.testing.assert_frame_equal(ev_a, ev_b)
    assert meta_a["rng_fingerprint"] == meta_b["rng_fingerprint"]
    assert meta_a["config_hash"] == meta_b["config_hash"]

    assert meta_a["rng_fingerprint"] != meta_c["rng_fingerprint"]
    assert not df_a.equals(df_c)

    assert len(ev_a) > 0 and set(ev_a.columns) == {"year", "action", "actors", "place", "magnitude"}
    entities = pd.read_parquet(tmp_path / "a" / "entities.parquet")
    assert set(entities.entity) == {a for actors in ev_a.actors for a in actors}
