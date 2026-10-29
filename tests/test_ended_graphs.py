"""Round 14 step 2: ended polities and extinct peoples in the graphs.

- Lineage tree: an ended polity's branch stops at the year it ended, with an end mark and how it
  ended on hover; only living polities' branches run to the current year; a people with no
  living polity keeps its last polities in the tree.
- Population per founding group: a people that died out drops to zero at that year, its line
  stops there, and its label says it ended.
"""

import re

import numpy as np
import pytest

from conftest import small_model
from deluge.agents.polity import END_CAUSES
from deluge.viz import app, panels


@pytest.fixture(scope="module")
def ended_run():
    m = small_model({"polities": {"secession_base": 0.5}}, years=400)
    for _ in range(250):
        m.step()
    # One founding people dies out: every settlement of the smallest one ends now.
    lp = np.asarray(m.lineage_population)
    g = int(np.flatnonzero(lp > 0)[np.argmin(lp[lp > 0])])
    t = m.settlements
    for sid in t.alive_ids()[t.lineage[t.alive_ids()] == g].tolist():
        m.end_settlement(sid, "dissolved")
    death = m.year  # the first year it holds no people
    for _ in range(60):
        m.step()
    return m, g, death


def _x(model, year, pad=30):
    w = app.VIZ.tree_width_px
    return pad + (w - pad - 10) * year / max(model.year, 1)


def test_tree_branches_stop_where_the_polity_ended(ended_run):
    m, g, death = ended_run
    P = m.polities
    svg = panels.lineage_tree(m, app.VIZ)
    shown = {int(p): int(e) for p, e in re.findall(r'<g data-polity="(\d+)" data-end="(-?\d+)">', svg)}
    ended = [p for p in shown if P.ended[p] >= 0]
    living = [p for p in shown if P.ended[p] < 0]
    assert ended and living
    words = panels.inspector_words(app.VIZ)["ended"]
    for p in shown:
        block = svg.split(f'<g data-polity="{p}" ')[1].split("</g>")[0]
        x2 = float(re.search(r'x2="([\d.]+)"', block).group(1))
        if P.ended[p] >= 0:
            assert shown[p] == P.ended[p] and x2 == pytest.approx(_x(m, P.ended[p]), abs=0.06)
            assert block.count("<line") == 3  # branch, end mark, hover area
            cause = END_CAUSES[int(P.end_cause[p])]
            assert words[cause].split("{")[0].strip() in block  # how it ended
        else:
            assert shown[p] == m.year and x2 == pytest.approx(_x(m, m.year), abs=0.06)
            assert block.count("<line") == 2  # no end mark
    # The people that died out keeps its last polities in the tree, ended where it ended.
    mine = [p for p in shown if P.lineage[p] == g]
    assert mine and all(P.ended[p] >= 0 for p in mine)
    assert max(P.ended[p] for p in mine) == death


def test_population_line_of_a_people_that_died_out_stops(ended_run):
    m, g, death = ended_run
    svg = app.charts(m)[0]
    name = m.config.founders.names[g]
    label = f"{name} (ended {death:,})"
    assert f">{label}<" in svg  # the legend marks it ended
    pts = re.search(rf'points="([^"]*)" data-label="{re.escape(label)}"', svg).group(1).split()
    xs = panels.series(m, ["n_polities"])["year"]
    w, pad = app.VIZ.chart_width_px, 34
    sx = lambda x: pad + (w - pad - 8) * (x - xs[0]) / (xs[-1] - xs[0])  # noqa: E731
    last_x, last_y = (float(v) for v in pts[-1].split(","))
    assert last_x == pytest.approx(sx(death), abs=0.06)  # stops at the year it died out
    assert last_y == pytest.approx(app.VIZ.chart_height_px - 18, abs=0.06)  # at zero
    # The living peoples run to the last year.
    for n in m.config.founders.names:
        if n == name:
            continue
        p = re.search(rf'points="([^"]*)" data-label="{re.escape(n)}"', svg).group(1).split()
        assert float(p[-1].split(",")[0]) == pytest.approx(sx(xs[-1]), abs=0.06)
