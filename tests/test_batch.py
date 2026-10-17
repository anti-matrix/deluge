"""Round 10: batch runs and the analysis reports.

``scripts/batch.py`` runs histories in parallel processes into one folder each, skips
complete runs when rerun, and limits its workers by free memory; ``analysis/batch_report.py``
and ``analysis/milestones.py`` read the outputs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from conftest import SMALL_OVERRIDES_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from batch import worker_limit  # noqa: E402


def test_worker_limit_follows_free_memory():
    assert worker_limit(2, 1000, 500, available_mb=5000) == 2
    assert worker_limit(2, 1000, 500, available_mb=2000) == 1
    assert worker_limit(2, 1000, 500, available_mb=100) == 1  # at least one
    assert worker_limit(8, 1000, 500, available_mb=None) <= 8


def test_batch_runs_and_reports(tmp_path):
    batch = tmp_path / "batch"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "batch.py"), "--overrides", str(SMALL_OVERRIDES_PATH),
           "--world-seed", "3", "--history-seeds", "1", "2", "--years", "60", "--batch-dir", str(batch)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    summary = json.loads((batch / "batch.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in summary["runs"]] == ["complete", "complete"]
    for h in (1, 2):
        meta = json.loads((batch / f"w3_h{h}" / "meta.json").read_text(encoding="utf-8"))
        assert meta["complete"] and meta["history_seed"] == h and meta["final_year"] == 60
    again = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    assert again.count("complete, skipped") == 2

    from analysis import batch_report, milestones
    from analysis.runs import batch_runs

    runs = batch_runs(batch)
    text = batch_report.report(runs)
    for n in range(1, 6):
        assert f"## {n}." in text
    written = milestones.write(batch)
    assert batch / "milestones.md" in written and (batch / "w3_h1" / "milestones.md").exists()
    tech = pd.read_parquet(batch / "w3_h1" / "techniques.parquet")
    assert {"first_held", "first_invented"} <= set(tech.columns)
    assert (tech.filter(like="first_held_") >= -1).all().all()
