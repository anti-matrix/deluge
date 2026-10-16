"""Round 11: speed-ups leave results identical.

The full model state (``deluge/data/state.py``) after 500 years on the small test world and
after 1,000 years on world seed 1, history seed 1 must match the digests stored in
``tests/data/state_digests.json`` (``scripts/state_digests.py`` writes them).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from state_digests import REFERENCE, digest_of  # noqa: E402


def reference(name: str) -> str:
    return json.loads(REFERENCE.read_text(encoding="utf-8"))[name]


def test_state_digest_is_deterministic():
    from conftest import small_model
    from deluge.data.state import state_digest

    a, b = small_model(years=20), small_model(years=20)
    a.run()
    b.run()
    assert state_digest(a) == state_digest(b)
    b.settlements.population[0] += 1e-9
    assert state_digest(a) != state_digest(b)


def test_small_world_500_years_unchanged():
    assert digest_of("small_500") == reference("small_500")


@pytest.mark.slow
def test_world_1_1000_years_unchanged():
    assert digest_of("w1_h1_1000") == reference("w1_h1_1000")
