"""Every attack, as the war display draws it (Round 13): the browser map's campaign arrows,
supply lines and war banners (``viz/portrayal.py``).

The war phase notes each raid or campaign it resolves (``war.battle``): the year, attacker
and defender polities, the settlement it set out from and the one attacked, the cells of
those and of the attacker's capital then, the battle's event in the log (-1 when none was
logged), its sea crossing's event (-1 when it crossed no sea; that event's ``link`` is the
battle's event) and the logistics decay it fought at: exp(-km from the capital to the
target / the capital's reach), the distance term of the attacker's strength. For the war
deaths bound (Round 14, ``configs/bounds.yaml``): the attacker's and defender's dead, the
target's people before the battle, the attacker's people within logistics reach of the
target (settlements no farther from it than the reach) and its mobilization share; allies that
joined either side (Round 14 alliances): their dead and their mobilized people within reach. Written by the
model, never read by the simulation (private: left out of the state digest, kept in
checkpoints).
"""

from __future__ import annotations

from array import array

import numpy as np

INT_COLUMNS = ("year", "attacker", "defender", "src", "target", "src_cell", "target_cell", "capital", "capital_cell",
               "event", "crossed")
FLOAT_COLUMNS = ("dead_attacker", "dead_defender", "target_people", "present_people", "mobilization",
                 "dead_allies", "allied_mobilized")


class CampaignLog:
    def __init__(self) -> None:
        for c in INT_COLUMNS:
            setattr(self, c, array("q"))
        self.decay = array("d")
        for c in FLOAT_COLUMNS:
            setattr(self, c, array("d"))

    def __len__(self) -> int:
        return len(self.year)

    def add(self, decay: float, **values) -> None:
        for c in INT_COLUMNS:
            getattr(self, c).append(int(values[c]))
        self.decay.append(float(decay))
        for c in FLOAT_COLUMNS:
            getattr(self, c).append(float(values.get(c, np.nan)))

    def columns(self, since_year: int | None = None) -> dict[str, np.ndarray]:
        """The log as arrays (the attacks of ``since_year`` on, all if None)."""
        out = {c: np.frombuffer(getattr(self, c), dtype=np.int64) for c in INT_COLUMNS}
        out["decay"] = np.frombuffer(self.decay, dtype=np.float64)
        for c in FLOAT_COLUMNS:
            col = getattr(self, c, None)  # logs pickled before Round 14 lack these
            out[c] = np.frombuffer(col, dtype=np.float64) if col is not None and len(col) == len(self) \
                else np.full(len(self), np.nan)
        if since_year is not None and len(self):
            k = int(np.searchsorted(out["year"], since_year, side="left"))
            out = {c: v[k:] for c, v in out.items()}
        return out

    def frame(self):
        import pandas as pd

        return pd.DataFrame(self.columns())
