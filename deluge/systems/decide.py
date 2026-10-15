"""Phase 6: brains decide (spec sections 4 and 10).

Every living polity's network reads its features (``brains/features.py``) and sets its
priorities for the year (``model.brains.output``) and the weights its per-neighbour head
gives each target feature (``model.brains.target_w``), all in one batched pass. A polity
born later in the year carries its parent's priorities (a new founding polity: its own
brain's response to zero inputs) until its first decision.
"""

from __future__ import annotations

from deluge.brains import features, network


def run(model) -> None:
    P, B = model.polities, model.brains
    pids = P.alive_ids()
    if len(pids) == 0:
        return
    x = features.assemble(model, pids)
    out, target_w, hidden = network.forward(model.brain_layout, B.genome[pids], x)
    B.output[pids] = out
    B.target_w[pids] = target_w
    k = model.config.brains.recurrent_units
    if k:
        B.state[pids] = hidden[:, :k]
    model.brain_output_mean = out.mean(axis=0)
