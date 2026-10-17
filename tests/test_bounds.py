"""Round 12: physical bounds (``configs/bounds.yaml``, ``deluge/data/bounds.py``) and what the
run recorder measures against them. Nothing here enters the simulation."""

import numpy as np

from conftest import small_model
from deluge.agents.settlement import MODES
from deluge.data.bounds import Ceilings, bounds_path, load_bounds
from deluge.data.collectors import RunRecorder


def ceilings(model) -> Ceilings:
    return Ceilings(load_bounds(bounds_path(model.config)), model.knowledge)


def test_ceilings_follow_mode_and_techniques():
    m = small_model(years=1)
    g, c = m.knowledge, ceilings(m)
    raw = load_bounds(bounds_path(m.config))["density_ceilings"]
    knows = np.zeros((3, len(g.names)), dtype=bool)
    knows[1, g.index("plough")] = True
    knows[2, [g.index("irrigation"), g.index("synthetic_fertilizers")]] = True
    per = c.per_mode(knows)
    assert per[0].tolist() == [raw[mode]["base"] for mode in MODES]
    assert per[1, 2] == raw["farm"]["raised_by"]["plough"] and per[1, 0] == raw["forage"]["base"]
    assert per[2, 2] == raw["farm"]["raised_by"]["synthetic_fertilizers"]  # the highest counts
    # The ceiling of the most intensive mode practised (at least use_share of the people).
    shares = np.array([[0.95, 0.0, 0.05], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]])
    s = c.settlement(shares, knows, 0.1)
    assert s.tolist() == [raw["forage"]["base"], raw["farm"]["raised_by"]["plough"],
                          raw["farm"]["raised_by"]["synthetic_fertilizers"]]
    area = 100.0
    assert np.allclose(c.ratio(np.array([150.0, 0.0, 0.0]), shares, knows, area, 0.1), [0.5, 0.0, 0.0])
    # World: foraging only until a mode is known somewhere.
    assert c.world(knows[0], 10, area) == 10 * area * raw["forage"]["base"]
    assert c.world(knows[1], 10, area) == 10 * area * raw["forage"]["base"]  # plough without agriculture
    knows[1, g.index("agriculture")] = True
    assert c.world(knows[1], 10, area) == 10 * area * raw["farm"]["raised_by"]["plough"]


def test_recorder_measures_ceilings_and_surplus():
    m = small_model(years=60)
    rec = RunRecorder(m)
    while m.running:
        m.step()
        rec.observe(m)
    df = rec.ceilings_frame()
    model = m.datacollector.get_model_vars_dataframe()
    assert len(df) == 60 and df.year.tolist() == list(range(60))
    assert np.allclose(df.population.to_numpy(), model.population.to_numpy())
    assert (df.max_ratio > 0).all() and (df.world_ceiling >= df.population).all()
    assert (model.surplus >= 0).all() and model.surplus.sum() > 0
