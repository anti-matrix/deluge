"""Headless run.py: two runs with the same seeds produce identical output (model table,
event log, metadata), different seeds differ, and the event log is written.

On the small test world (``--overrides configs/worlds/small_test.yaml``) for 300 years;
the full-length run and its time budget are checked in test_model.py (slow).
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import SMALL_OVERRIDES_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "scripts" / "run.py"
YEARS = 300


def _run(out_dir: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    subprocess.run(
        [sys.executable, str(RUN), "--overrides", str(SMALL_OVERRIDES_PATH), "--history-seed", str(seed),
         "--years", str(YEARS), "--out", str(out_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    return pd.read_parquet(out_dir / "model.parquet"), pd.read_parquet(out_dir / "events.parquet"), meta


def test_run_script_acceptance(tmp_path, small_cfg):
    df_a, ev_a, meta_a = _run(tmp_path / "a", seed=3)
    df_b, ev_b, meta_b = _run(tmp_path / "b", seed=3)

    assert meta_a["final_year"] == YEARS
    assert (meta_a["world_seed"], meta_a["history_seed"]) == (small_cfg.run.world_seed, 3)
    assert len(df_a) == YEARS

    pd.testing.assert_frame_equal(df_a, df_b)
    pd.testing.assert_frame_equal(ev_a, ev_b)
    assert meta_a["rng_fingerprint"] == meta_b["rng_fingerprint"]
    assert meta_a["config_hash"] == meta_b["config_hash"]

    assert len(ev_a) > 0 and set(ev_a.columns) == {"year", "action", "actors", "place", "magnitude"}
    entities = pd.read_parquet(tmp_path / "a" / "entities.parquet")
    assert set(entities.entity) == {a for actors in ev_a.actors for a in actors}
    brains = pd.read_parquet(tmp_path / "a" / "brains" / "brains.parquet")
    tree = pd.read_parquet(tmp_path / "a" / "polity_tree.parquet")
    layout = json.loads((tmp_path / "a" / "brains" / "layout.json").read_text(encoding="utf-8"))
    assert list(brains.polity) == list(tree.polity) and list(brains.parent) == list(tree.parent)
    assert {len(w) for w in brains.weights} == {layout["blocks"]["q0"]["stop"]}


@pytest.mark.slow
def test_run_script_seeds_differ(tmp_path):
    df_a, _, meta_a = _run(tmp_path / "a", seed=3)
    df_c, _, meta_c = _run(tmp_path / "c", seed=4)
    assert meta_a["rng_fingerprint"] != meta_c["rng_fingerprint"]
    assert not df_a.equals(df_c)
