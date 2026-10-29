"""Side panels of the browser UI (spec section 12), as plain SVG and HTML strings.

- ``line_chart``: series over the years (population per founding group, polity count, ...),
  thinned to at most ``viz.chart_points`` points per series.
- ``lineage_tree``: a snapshot of the polity descent tree: every living polity and its
  ancestors (and, for a founding group with none left, its last polities and theirs), one
  horizontal line per polity from its founding to its end (with a small end mark) or, while
  it lives, to now, joined to its parent at the year it split off, coloured by founding
  group; hovering a branch shows the polity's name, years and how it ended.
- ``died_out``: a people's population line drops to zero where it died out and stops there.
- ``inspector``: one polity, in the narrator's language (``deluge/naming.py``, words in
  ``viz.inspector_file``): who they are, their culture and this year's brain outputs in words,
  and "What they remember": for each of the heaviest symbols of its store, the narrator's
  sentence for its strongest record, with the raw record (fields, provenance, transformation
  log) under a collapsed "details"; then the same for its origin records.

Everything here only reads model state.
"""

from __future__ import annotations

import html
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from deluge.brains.network import OUTPUTS
from deluge.config import DEFAULT_CONFIG_PATH, VizConfig
from deluge.naming import Speaker
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
    lx = pad + 4
    for label, vals, color in lines:
        v = np.asarray(vals, dtype=float)
        # A shorter series (a line that stops, e.g. a people that died out) ends at its last point.
        idx = [i for i in keep if i < len(v)] + ([len(v) - 1] if len(v) and len(v) - 1 not in keep else [])
        pts = " ".join(f"{sx(xs[i]):.1f},{sy(v[i]):.1f}" for i in idx)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{pts}" '
                     f'data-label="{html.escape(label)}"/>')
        parts.append(f'<text x="{lx}" y="26" font-size="9" fill="{color}">{html.escape(label)}</text>')
        lx += max(46, 6 * len(label) + 8)
    parts.append("</svg>")
    return "".join(parts)


def died_out(xs, values) -> tuple[list[float], int | None]:
    """A people's series cut where it died out: if its last value is 0, the series up to its
    first zero after the last year with people (the line drops to zero and stops there), and
    that year; else the series whole and None."""
    v = list(values)
    if not v or v[-1] > 0:
        return v, None
    alive = [i for i, x in enumerate(v) if x > 0]
    if not alive:
        return v[:1], int(xs[0])
    end = alive[-1] + 1
    return v[: end + 1], int(xs[end])


def lineage_tree(model, vcfg: VizConfig) -> str:
    """SVG snapshot of the descent tree of the living polities (and their ancestors)."""
    P = model.polities
    alive = P.alive_ids().tolist()
    nodes: set[int] = set()
    for p in alive:
        nodes.update(model.lineages.polity_ancestry(P, p))
    # A founding group with no living polity: its last polities (those that ended last) and theirs.
    n = P.n
    living_groups = set(P.lineage[alive].tolist())
    for g in range(len(model.lineages.names)):
        mine = np.flatnonzero(P.lineage[:n] == g)
        if g in living_groups or len(mine) == 0:
            continue
        last = mine[P.ended[mine] == P.ended[mine].max()]
        for p in last.tolist():
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
    from deluge.data.events import entity

    from deluge.agents.polity import END_CAUSES

    palette = lineage_palette(vcfg)
    sp = Speaker(model, tags=True)
    words = inspector_words(vcfg)
    parts = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">',
             f'<text x="{pad}" y="12" font-size="11" font-weight="bold">Polity descent: {len(alive)} living, '
             f'{len(nodes)} with ancestors (year {max(now - 1, 0):,})</text>']
    for p, r in order.items():
        ended = P.ended[p] >= 0
        end = int(P.ended[p]) if ended else now  # an ended polity's branch stops at its end; the living run on
        c = palette[int(P.lineage[p])]
        label = sp.entity(entity("polity", p))
        years = f"{int(P.founded[p]):,} to {int(P.ended[p]):,}" if ended else f"{int(P.founded[p]):,} on"
        how = ""
        if ended:
            other = int(P.merged_into[p])
            how = "; " + words["ended"][END_CAUSES[int(P.end_cause[p])]].format(
                year=int(P.ended[p]), other=sp.entity(entity("polity", other)) if other >= 0 else "")
        friends = [b if a == p else a for a, b in sorted(model.alliances) if p in (a, b)]
        if friends:  # hover shows its allies (Round 14)
            how += words["tree_allies"].format(list=", ".join(sp.entity(entity("polity", q)) for q in friends))
        x1, x2, y = sx(P.founded[p]), sx(end), sy(r)
        mark =(f'<line x1="{x2:.1f}" y1="{y - 3:.1f}" x2="{x2:.1f}" y2="{y + 3:.1f}" stroke="{c}" '
                f'stroke-width="1.2"/>') if ended else ""
        parts.append(f'<g data-polity="{p}" data-end="{end}"><title>{html.escape(label[:1].upper() + label[1:])}, '
                     f'years {years}{html.escape(how)}</title>'
                     f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
                     f'stroke="{c}" stroke-width="{1.6 if not ended else 0.8}"/>{mark}'
                     f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{max(x2, x1 + 3):.1f}" y2="{y:.1f}" '
                     f'stroke="transparent" stroke-width="4"/></g>')
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


@lru_cache(maxsize=4)
def _load_words(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspector_words(vcfg: VizConfig) -> dict:
    """The inspector's plain-language words (``viz.inspector_file``)."""
    return _load_words(str(Path(DEFAULT_CONFIG_PATH).parent / vcfg.inspector_file))


def level_word(words: dict, v: float) -> str:
    for bound, word in words["levels"]:
        if v < bound:
            return word
    return words["levels"][-1][1]


def trait_word(words: dict, trait: str, v: float) -> str:
    low, high = words["culture"][trait]
    if abs(v - 0.5) <= words["culture_middle"]:
        return words["between"].format(low=low, high=high)
    return high if v > 0.5 else low


def _worded_bars(rows, color: str) -> str:
    """Rows of (label, word, value 0..1) as labelled bars."""
    out = "".join(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px">'
                  f'<span style="width:14em">{html.escape(label)}</span>'
                  f'<span style="width:12em;color:#444">{html.escape(word)}</span>'
                  f'<span style="display:inline-block;height:10px;width:{80 * max(min(v, 1.0), 0.0):.0f}px;'
                  f'background:{color}"></span><span style="color:#777">{v:.2f}</span></div>'
                  for label, word, v in rows)
    return f"<div>{out}</div>"


def allies_html(model, pid: int, words: dict, sp) -> str:
    """"Allies and enemies": current allies (since when), current wars, past alliances (how each
    ended, betrayals named) of polity ``pid`` (``model.alliances``, ``model.alliance_history``,
    ``model.wars``)."""
    from deluge.data.events import entity

    def name(q: int) -> str:
        return sp.entity(entity("polity", int(q)))

    now = sorted((b if a == pid else a, y) for (a, b), y in model.alliances.items() if pid in (a, b))
    wars = sorted((b if a == pid else a, y) for (a, b), y in model.wars.items() if pid in (a, b))
    past = [h for h in getattr(model, "alliance_history", []) if pid in (h[0], h[1])]
    lines = [words["allies_now"].format(list=", ".join(words["ally"].format(name=name(q), year=y) for q, y in now))
             if now else words["no_allies"],
             words["wars_now"].format(list=", ".join(words["war"].format(name=name(q), year=y) for q, y in wars))
             if wars else words["no_wars"]]
    if past:
        items = []
        for a, b, begin, end, how, by in past:
            other = b if a == pid else a
            form = words.get(f"past_{how}", words["past_ended"])
            items.append(form.format(name=name(other), begin=begin, end=end, by=name(by) if by >= 0 else ""))
        lines.append(words["past"].format(list="; ".join(items)))
    return "".join(f'<div style="font-size:12px" data-allies="{k}">{html.escape(s)}</div>'
                   for k, s in zip(("now", "wars", "past"), lines))


def looks_html(model, pid: int, vcfg: VizConfig, words: dict, sp, color: str) -> str:
    """"How they look": per appearance trait, bars for the polity's people, all of its founding
    people now and its founders at year 0 (drift and mixing show as the gap between them), then
    one line on how different the polities in contact look to it (``identity.neighbour_looks``:
    the perceived difference its brain reads), the three most alike and the three most unlike."""
    from deluge.data.events import entity
    from deluge.systems import identity

    seen = identity.looks(model, pid)
    cols = list(zip(("polity", "lineage", "founders"), words["looks_columns"], (color, "#888888", "#cccccc")))
    head = "".join(f'<span style="display:inline-flex;align-items:center;gap:3px;margin-right:10px">'
                   f'<span style="display:inline-block;width:10px;height:10px;background:{c}"></span>'
                   f'{html.escape(label)}</span>' for _, label, c in cols)
    rows = []
    for j, trait in enumerate(model.config.ancestry.appearance_traits):
        label, low, high = words["appearance"].get(trait, [trait.replace("_", " "), "low", "high"])
        bars = ""
        for key, col_label, c in cols:
            v = float(seen[key][j])
            bars += ('<div style="display:flex;align-items:center;gap:4px;height:11px">'
                     + ("<span style='font-size:10px;color:#999'>unknown</span>" if np.isnan(v) else
                        f'<span title="{html.escape(col_label)}: {v:.2f}" style="display:inline-block;height:8px;'
                        f'width:{80 * min(max(v, 0.0), 1.0):.0f}px;background:{c}"></span>'
                        f'<span style="font-size:10px;color:#777">{v:.2f}</span>') + "</div>")
        rows.append(f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;margin:2px 0" '
                    f'data-trait="{trait}"><span style="width:14em">{html.escape(label)} '
                    f'<span style="color:#888">({html.escape(low)} to {html.escape(high)})</span></span>'
                    f'<div>{bars}</div></div>')
    if model.config.ancestry.blind_perception:
        line = words["looks_blind"]
    else:
        near = identity.neighbour_looks(model, pid)
        if not near:
            line = words["looks_no_neighbours"]
        else:
            def named(items):
                return ", ".join(f"{sp.entity(entity('polity', q))} {d:.2f}" for q, d in items)
            line = words["looks_neighbours"].format(closest=named(near[:3]), farthest=named(near[::-1][:3]))
    return (f'<div style="font-size:11px;margin-bottom:2px">{head}</div>' + "".join(rows)
            + f'<div style="font-size:12px;margin-top:4px" data-looks="neighbours">{html.escape(line)}</div>')


def _details(lines: list[str]) -> str:
    return ('<details style="margin:0 0 6px 1em"><summary style="font-size:11px;color:#666;cursor:pointer">'
            'details</summary><pre style="white-space:pre-wrap;font-size:11px;margin:2px 0">'
            + html.escape("\n".join(lines)) + "</pre></details>")


def inspector(model, pid: int, vcfg: VizConfig, n_symbols: int | None = None) -> str:
    """HTML for polity ``pid``, in the narrator's language (``deluge/naming.py``)."""
    if REPO not in map(Path, sys.path):
        sys.path.insert(0, str(REPO))
    from analysis import myths, narrator
    from deluge.agents.polity import END_CAUSES
    from deluge.data.chronicle import Namer
    from deluge.data.events import entity

    P = model.polities
    if pid < 0 or pid >= P.n:
        return "<i>Click a claimed cell to inspect its polity.</i>"
    words = inspector_words(vcfg)
    sp = Speaker(model, tags=True)
    plain = Speaker(model)
    g = int(P.lineage[pid])
    color = lineage_palette(vcfg)[g]
    me = plain.entity(entity("polity", pid))
    parent = int(P.parent[pid])
    origin = (f"founded in year {int(P.founded[pid])} from {sp.entity(entity('polity', parent))}" if parent >= 0
              else f"founded in year {int(P.founded[pid])}, the first polity of its people")
    if P.ended[pid] < 0:
        status = "living"
    else:
        other = int(P.merged_into[pid])
        status = words["ended"][END_CAUSES[int(P.end_cause[pid])]].format(
            year=int(P.ended[pid]), other=sp.entity(entity("polity", other)) if other >= 0 else "")
    capital = plain.entity(entity("settlement", int(P.capital[pid])))
    coh, pres = float(P.cohesion[pid]), float(min(P.pressure[pid], 1.0))
    cw = words["cohesion"][int(coh >= 0.5)]
    pw = words["pressure"][int(pres >= 0.5)]
    out = [f'<h3 style="margin:0">{html.escape(me[:1].upper() + me[1:])} '
           f'<span style="color:{color};font-size:13px">({model.lineages.names[g]})</span></h3>',
           f"<div>A people of {html.escape(sp.people(g))}; {html.escape(origin)}; {html.escape(status)}. "
           f"{int(P.n_settlements[pid]):,} settlement{'' if P.n_settlements[pid] == 1 else 's'}, "
           f"{P.population[pid]:,.0f} people; capital {html.escape(capital)}.</div>",
           f"<div>{html.escape(cw[:1].upper() + cw[1:])} (cohesion {coh:.2f}); {html.escape(pw)} "
           f"(pressure {P.pressure[pid]:.2f}); elite ratio {P.elite_ratio[pid]:.3f}.</div>",
           "<h4>Their ways</h4>",
           _worded_bars([(tr.replace("_", " "), trait_word(words, tr, float(v)), float(v))
                         for tr, v in zip(model.config.founders.culture_traits, P.culture[pid])], color),
           "<h4>Allies and enemies</h4>",
           allies_html(model, pid, words, sp),
           "<h4>How they look</h4>",
           looks_html(model, pid, vcfg, words, sp, color),
           "<h4>What they lean toward this year</h4>",
           _worded_bars([(words["outputs"][o], level_word(words, float(v)), float(v))
                         for o, v in zip(OUTPUTS, model.brains.output[pid])], "#555")]
    st = model.memory
    name = Namer(model)
    hist = myths.History(model)
    k = vcfg.inspector_symbols if n_symbols is None else n_symbols
    out.append("<h4>What they remember</h4>")
    syms = myths._store_symbols(model, pid)[:k]
    if not syms:
        out.append("<i>Nothing: the store is empty.</i>")
    mids = st.stores.get(pid, [])
    for s, w in syms:
        recs = sorted((st.records[m] for m in mids if s in st.records[m].symbols),
                      key=lambda r: (-r.vividness, r.mem_id))[:1]
        for rec in recs:
            told = narrator.remembered(sp, narrator.record_fields(rec))["text"]
            lines = [f"Symbol {s} (weight {w:.2f}): {myths.symbol_label(model, name, s)}"]
            lines += myths._record(model, name, hist, rec, max_events=vcfg.inspector_events, max_ops=4)
            out.append(f"<div>{html.escape(told)}</div>" + _details(lines))
    out.append("<h4>What they remember of the first years</h4>")
    entries = [e for e in narrator.origin_records(model, g, hist) if pid in e["stores"]]
    if not entries:
        out.append(f"<i>No record in this store descends from years 0 to "
                   f"{model.config.reports.origins_until_year}.</i>")
    else:
        items = []
        for e in entries[: vcfg.inspector_origins]:
            rec = e["record"]
            told = narrator.remembered(sp, narrator.record_fields(rec))["text"]
            happened = [h["text"] for h in narrator.happened(model, sp, rec.provenance)]
            items.append(f"<li>{html.escape(told)}"
                         + _details([f"Record #{rec.mem_id} ({'still holds' if e['current'] else 'descends from'} "
                                     f"those years). What happened:"] + happened) + "</li>")
        out.append("<ul style='margin:0'>" + "".join(items) + "</ul>")
        if len(entries) > vcfg.inspector_origins:
            out.append(f"<i>... {len(entries) - vcfg.inspector_origins} more.</i>")
    return "".join(out)
