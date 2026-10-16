"""Side panels of the browser UI (spec section 12), as plain SVG and HTML strings.

- ``line_chart``: series over the years (population per founding group, polity count, ...),
  thinned to at most ``viz.chart_points`` points per series.
- ``lineage_tree``: a snapshot of the polity descent tree: every living polity and its
  ancestors, one horizontal line per polity from its founding to its end (or now), joined to
  its parent at the year it split off, coloured by founding group.
- ``inspector``: one polity: culture vector, current brain outputs, the heaviest symbols of its
  memory store with the records and original events behind them, and the narrator's "What
  they remember" for its origin records (``analysis/narrator.py``).

Everything here only reads model state.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np

from deluge.brains.network import OUTPUTS
from deluge.config import VizConfig
from deluge.viz.portrayal import lineage_palette

REPO = Path(__file__).resolve().parents[2]


def series(model, names) -> dict[str, list[float]]:
    """Model-level data collected so far, by reporter name (``year`` included)."""
    mv = model.datacollector.model_vars
    n = min(len(mv["year"]), *(len(mv[k]) for k in names))
    return {k: list(mv[k][:n]) for k in ("year", *names)}


def _thin(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) <= k:
        return np.arange(len(x))
    return np.unique(np.linspace(0, len(x) - 1, k).round().astype(int))


def line_chart(vcfg: VizConfig, title: str, xs, lines: list[tuple[str, list[float], str]]) -> str:
    """An SVG line chart: ``lines`` = [(label, values, colour)] over the years ``xs``."""
    w, h, pad = vcfg.chart_width_px, vcfg.chart_height_px, 34
    xs = np.asarray(xs, dtype=float)
    if len(xs) < 2:
        return f"<div><b>{html.escape(title)}</b><br><i>no data yet</i></div>"
    keep = _thin(xs, vcfg.chart_points)
    x0, x1 = float(xs[0]), float(xs[-1])
    top = max([max(v) for _, v, _ in lines if len(v)] + [1e-9])
    sx = lambda x: pad + (w - pad - 8) * (x - x0) / max(x1 - x0, 1e-9)  # noqa: E731
    sy = lambda y: h - 18 - (h - 18 - 16) * y / top  # noqa: E731
    parts = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">',
             f'<text x="{pad}" y="12" font-size="11" font-weight="bold">{html.escape(title)}</text>',
             f'<text x="2" y="{sy(top) + 4:.0f}" font-size="9">{top:,.0f}</text>',
             f'<text x="{pad}" y="{h - 4}" font-size="9">{x0:,.0f}</text>',
             f'<text x="{w - 40}" y="{h - 4}" font-size="9">{x1:,.0f}</text>',
             f'<line x1="{pad}" y1="{h - 18}" x2="{w - 8}" y2="{h - 18}" stroke="#999"/>']
    for k, (label, vals, color) in enumerate(lines):
        v = np.asarray(vals, dtype=float)
        pts = " ".join(f"{sx(xs[i]):.1f},{sy(v[i]):.1f}" for i in keep if i < len(v))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}"/>')
        parts.append(f'<text x="{pad + 4 + 46 * k}" y="26" font-size="9" fill="{color}">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def lineage_tree(model, vcfg: VizConfig) -> str:
    """SVG snapshot of the descent tree of the living polities (and their ancestors)."""
    P = model.polities
    alive = P.alive_ids().tolist()
    nodes: set[int] = set()
    for p in alive:
        nodes.update(model.lineages.polity_ancestry(P, p))
    if not nodes:
        return "<i>No polities.</i>"
    children: dict[int, list[int]] = {}
    roots = []
    for p in sorted(nodes):
        parent = int(P.parent[p])
        (children.setdefault(parent, []) if parent in nodes else roots).append(p)
    # Rows: post-order, a polity after its children (leaves consecutive), at the mean of its
    # children's rows and its own (iterative: descent chains can be long).
    order: dict[int, float] = {}
    for r in sorted(roots, key=lambda r: (int(P.lineage[r]), r)):
        stack = [(r, False)]
        while stack:
            p, done = stack.pop()
            kids = sorted(children.get(p, []), key=lambda c: (int(P.founded[c]), c))
            if not done:
                stack.append((p, True))
                stack.extend((c, False) for c in reversed(kids))
                continue
            own = float(len(order))
            ys = [order[c] for c in kids]
            order[p] = float(np.mean(ys + [own])) if ys else own
    now = model.year
    rows = max(len(order), 1)
    w, h, pad = vcfg.tree_width_px, min(vcfg.tree_max_height_px, max(120, rows * vcfg.tree_row_px + 30)), 30
    sx = lambda y: pad + (w - pad - 10) * y / max(now, 1)  # noqa: E731
    sy = lambda r: 20 + (h - 30) * r / rows  # noqa: E731
    palette = lineage_palette(vcfg)
    parts = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">',
             f'<text x="{pad}" y="12" font-size="11" font-weight="bold">Polity descent: {len(alive)} living, '
             f'{len(nodes)} with ancestors (year {max(now - 1, 0):,})</text>']
    for p, r in order.items():
        end = int(P.ended[p]) if P.ended[p] >= 0 else now
        c = palette[int(P.lineage[p])]
        parts.append(f'<line x1="{sx(P.founded[p]):.1f}" y1="{sy(r):.1f}" x2="{sx(end):.1f}" y2="{sy(r):.1f}" '
                     f'stroke="{c}" stroke-width="{1.6 if P.ended[p] < 0 else 0.8}"/>')
        parent = int(P.parent[p])
        if parent in order:
            parts.append(f'<line x1="{sx(P.founded[p]):.1f}" y1="{sy(order[parent]):.1f}" '
                         f'x2="{sx(P.founded[p]):.1f}" y2="{sy(r):.1f}" stroke="{c}" stroke-width="0.6"/>')
    parts.append(f'<text x="{pad}" y="{h - 2}" font-size="9">year 0</text>')
    parts.append(f'<text x="{w - 60}" y="{h - 2}" font-size="9">year {max(now - 1, 0):,}</text></svg>')
    return "".join(parts)


def _bars(labels, values, color: str) -> str:
    rows = "".join(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px">'
                   f'<span style="width:9.5em">{html.escape(str(k))}</span>'
                   f'<span style="display:inline-block;height:10px;width:{100 * max(min(v, 1.0), 0.0):.0f}px;'
                   f'background:{color}"></span><span>{v:.2f}</span></div>' for k, v in zip(labels, values))
    return f"<div>{rows}</div>"


def inspector(model, pid: int, vcfg: VizConfig, n_symbols: int | None = None) -> str:
    """HTML for polity ``pid``."""
    if REPO not in map(Path, sys.path):
        sys.path.insert(0, str(REPO))
    from analysis import myths, narrator
    from deluge.data.chronicle import Namer

    P = model.polities
    if pid < 0 or pid >= P.n:
        return "<i>Click a claimed cell to inspect its polity.</i>"
    names = model.lineages.names
    g = int(P.lineage[pid])
    color = lineage_palette(vcfg)[g]
    status = "living" if P.ended[pid] < 0 else f"ended in year {int(P.ended[pid])}"
    out = [f'<h3 style="margin:0">Polity {pid} <span style="color:{color}">(group {names[g]})</span></h3>',
           f"<div>Founded year {int(P.founded[pid])}, parent {int(P.parent[pid])}; {status}; "
           f"{int(P.n_settlements[pid])} settlements, {P.population[pid]:,.0f} people; cohesion "
           f"{P.cohesion[pid]:.2f}, pressure {P.pressure[pid]:.2f}, elite ratio {P.elite_ratio[pid]:.3f}</div>",
           "<h4>Culture vector</h4>", _bars(model.config.founders.culture_traits, P.culture[pid], color),
           "<h4>Brain outputs (this year's decision)</h4>", _bars(OUTPUTS, model.brains.output[pid], "#555")]
    st = model.memory
    name = Namer(model)
    hist = myths.History(model)
    k = vcfg.inspector_symbols if n_symbols is None else n_symbols
    out.append("<h4>Heaviest memory symbols and their provenance</h4>")
    syms = myths._store_symbols(model, pid)[:k]
    if not syms:
        out.append("<i>Empty store.</i>")
    mids = st.stores.get(pid, [])
    for s, w in syms:
        out.append(f"<div><b>Symbol {s}</b> (weight {w:.2f}): {html.escape(myths.symbol_label(model, name, s))}</div>")
        recs = sorted((st.records[m] for m in mids if s in st.records[m].symbols),
                      key=lambda r: (-r.vividness, r.mem_id))[:1]
        for rec in recs:
            lines = myths._record(model, name, hist, rec, max_events=vcfg.inspector_events, max_ops=4)
            out.append('<pre style="white-space:pre-wrap;font-size:11px;margin:2px 0 6px 0">'
                       + html.escape("\n".join(lines)) + "</pre>")
    out.append("<h4>What they remember (origin records)</h4>")
    entries = [e for e in narrator.origin_records(model, g, hist) if pid in e["stores"]]
    if not entries:
        out.append(f"<i>No record in this store descends from years 0 to "
                   f"{model.config.reports.origins_until_year}.</i>")
    else:
        sp = narrator.Speaker(model, g, narrator.people_dialect(model, g), narrator.data_for(model))
        items = []
        for e in entries[: vcfg.inspector_origins]:
            tag = "(a)" if e["current"] else "(b)"
            items.append(f"<li>{tag} {html.escape(narrator.remembered(sp, narrator.record_fields(e['record']))['text'])}"
                         f" <span style='color:#777'>(record #{e['record'].mem_id})</span></li>")
        out.append("<ul>" + "".join(items) + "</ul>")
        if len(entries) > vcfg.inspector_origins:
            out.append(f"<i>... {len(entries) - vcfg.inspector_origins} more.</i>")
    return "".join(out)
