"""Every attack, as the war display draws it (Round 13): the browser map's campaign arrows,
supply lines and war banners (``viz/portrayal.py``).

The war phase notes each raid or campaign it resolves (``war.battle``): the year, attacker
and defender polities, the settlement it set out from and the one attacked, the cells of
those and of the attacker's capital then, the battle's event in the log (-1 when none was
logged), its sea crossing's event (-1 when it crossed no sea; that event's ``link`` is the
battle's event) and the logistics decay it fought at: exp(-km from the capital to the
target / the capital's reach), the distance term of the attacker's strength. Written by the
model, never read by the simulation (private: left out of the state digest, kept in
checkpoints).
"""

from __future__ import annotations

from array import array

import numpy as np

INT_COLUMNS = ("year", "attacker", "defender", "src", "target", "src_cell", "target_cell", "capital", "capital_cell",
               "event", "crossed")


class CampaignLog:
    def __init__(self) -> None:
        for c in INT_COLUMNS:
            setattr(self, c, array("q"))
        self.decay = array("d")

    def __len__(self) -> int:
        return len(self.year)

    def add(self, decay: float, **values: int) -> None:
        for c in INT_COLUMNS:
            getattr(self, c).append(int(values[c]))
        self.decay.append(float(decay))

    def columns(self, since_year: int | None = None) -> dict[str, np.ndarray]:
        """The log as arrays (the attacks of ``since_year`` on, all if None)."""
        out = {c: np.frombuffer(getattr(self, c), dtype=np.int64) for c in INT_COLUMNS}
        out["decay"] = np.frombuffer(self.decay, dtype=np.float64)
        if since_year is not None and len(self):
            k = int(np.searchsorted(out["year"], since_year, side="left"))
            out = {c: v[k:] for c, v in out.items()}
        return out
