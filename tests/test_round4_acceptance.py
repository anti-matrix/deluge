"""Round 4 acceptance on world seed 1, history seeds 1-5, with and without the teachers:
ten full-length runs (run.years, 12,900 since Round 4b) in parallel processes (about 20+
minutes on 4 cores).

Marked slow and skipped by a plain ``pytest``; run with ``pytest -m slow``. The same
check is scripts/round4_acceptance.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from round4_acceptance import acceptance  # noqa: E402


@pytest.mark.slow
def test_round4_acceptance():
    result = acceptance(world=1, histories=[1, 2, 3, 4, 5], workers=4)
    failed = {k: v for k, v in result["checks"].items() if not v}
    assert not failed, (failed, result["runs"])
