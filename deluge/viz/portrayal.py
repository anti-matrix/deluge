"""What the browser map shows, computed from model state (spec section 12).

The map is drawn in the browser by ``viz/hexmap.vue`` from what this module computes in
NumPy: one RGB colour per cell for the chosen layer (``cell_colors``), the polity borders
(``polity_borders``) and the overlay marks (``marks``), and the hover text of a cell
(``tooltip``). Every layer is drawn over the terrain with the year's sea:

- polity: every claimed cell in its founding group's hue, darker the higher the "shade by"
  variable (``SHADE_BY``: population density, technology level, subsistence output; log
  scales where noted), with thick borders between polities of different founding groups and
  thin ones between polities of one group, each over a light outline. A polity's isolated
  single cells (a claimed cell touching none of its own, while the polity holds others) are
  filled but get no outline of their own, so enclaves do not clutter the map as dots;
- terrain; population density (people per cell, opacity); subsistence mode (each settled
  cell the share-weighted mix of the mode colours); cohesion (each claimed cell its polity's
  cohesion on a colour map); a chosen technique's spread (settlements holding it, usable or
  latent where they are); the founding group of each settlement; mineral deposits;
  techniques usable per cell.

Overlays: capitals (dots in the group's colour), monuments, the teachers' path and position
during the contact phase, and active wars as arrows from attacker to defender capital,
dashed when the last battle crossed the sea (``WarTracker``). One lineage palette
(``lineage_palette``) serves every lineage-coloured element. Everything here only reads
model state.
"""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb

from deluge.agents.settlement import MODES
from deluge.config import VizConfig
from deluge.data.events import ACTION_CODE, decode
from deluge.world.render import SEA_COLOR, TERRAIN_COLORS
from deluge.world.worlddata import MINERALS, Terrain

LAYERS = ("polity", "terrain", "population density", "subsistence mode", "cohesion", "technique", "lineage",
          "resources", "techniques known")


def resource_colors(vcfg: VizConfig) -> list[str]:
    """Resources colours in code order: MINERALS, then several."""
    return [vcfg.resource_colors[m] for m in MINERALS] + [vcfg.resource_colors["several"]]


def lineage_palette(vcfg: VizConfig) -> tuple[str, ...]:
    """The one lineage palette (``viz.lineage_colors``): every lineage-coloured element of the
    UI (lineage and polity layers, capitals, the population-per-group plot, the lineage tree,
    legends) reads it here."""
    return tuple(vcfg.lineage_colors)


def shade(color: str, factor: float) -> tuple[float, float, float, float]:
    """``color`` darkened toward black (factor < 0) or lightened toward white (factor > 0)."""
    rgb = np.array(to_rgb(color))
    rgb = rgb * (1.0 + factor) if factor < 0 else rgb + (1.0 - rgb) * factor
    return (*rgb.tolist(), 1.0)


def _rgb(colors) -> np.ndarray:
    return np.array([to_rgb(c) for c in colors])


def _log_share(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """log10(x) mapped to 0 at ``lo`` (and below) .. 1 at ``hi`` (and above)."""
    with np.errstate(divide="ignore"):
        s = (np.log10(x) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    return np.clip(np.nan_to_num(s, neginf=0.0), 0.0, 1.0)


# --- "shade by" variables (1 drawn darkest) ---------------------------------------------------------


def density_share(model, vcfg: VizConfig) -> np.ndarray:
    """Per cell: people per km2 on a log scale, 0 at ``polity_density_min``, 1 at ``polity_density_max``."""
    return _log_share(model.cell_population() / model.cell_area_km2, vcfg.polity_density_min,
                      vcfg.polity_density_max)


def technology_share(model, vcfg: VizConfig) -> np.ndarray:
    """Per cell: techniques usable there over ``knowledge_vmax`` (capped at 1)."""
    return np.clip(model.knowledge_layer.data.ravel() / vcfg.knowledge_vmax, 0.0, 1.0)


def output_share(model, vcfg: VizConfig) -> np.ndarray:
    """Per cell: food available (people x food per head, in rations) per km2, log scale between
    ``polity_output_min`` and ``polity_output_max``."""
    t = model.settlements
    ids = t.alive_ids()
    food = np.zeros(model.geom.n_cells)
    food[t.cell[ids]] = t.population[ids] * t.food_ratio[ids]
    return _log_share(food / model.cell_area_km2, vcfg.polity_output_min, vcfg.polity_output_max)


SHADE_BY = {"population density": density_share, "technology level": technology_share,
            "subsistence output": output_share}


# --- layers -------------------------------------------------------------------------------------------


def base_colors(model) -> np.ndarray:
    """(cells, 3) terrain colours with the year's sea drawn over them."""
    terrain = model.world.terrain.ravel()
    table = _rgb([TERRAIN_COLORS[t] for t in Terrain])
    out = table[terrain]
    out[model.is_sea] = to_rgb(SEA_COLOR)
    return out


def polity_shades(vcfg: VizConfig) -> list[list[tuple]]:
    """Per founding group, its hue in ``polity_shade_levels`` brightness steps, lightest first."""
    n = vcfg.polity_shade_levels
    factors = np.linspace(vcfg.polity_light, -vcfg.polity_dark, n) if n > 1 else np.zeros(1)
    return [[shade(c, float(f)) for f in factors] for c in lineage_palette(vcfg)]


def claimed(model) -> np.ndarray:
    """Per cell: the polity claiming it, -1 for none."""
    return model.polity_layer.data.ravel().astype(np.int64) - 1


def polity_fill(model, vcfg: VizConfig, shade_by: str = "population density") -> tuple[np.ndarray, np.ndarray]:
    """(cells, RGBA) of the claimed cells: the claiming polity's founding-group hue, darker the
    higher the ``shade_by`` value."""
    owner = claimed(model)
    cells = np.flatnonzero(owner >= 0)
    lineage = model.polities.lineage[owner[cells]]
    value = SHADE_BY[shade_by](model, vcfg)[cells]
    level = np.rint(value * (vcfg.polity_shade_levels - 1)).astype(np.int64)
    table = np.array(polity_shades(vcfg))  # (groups, levels, 4)
    return cells, table[lineage, level]


def _blend(base: np.ndarray, cells: np.ndarray, rgb: np.ndarray, alpha: np.ndarray | float) -> None:
    a = np.broadcast_to(np.asarray(alpha, dtype=float), (len(cells),))[:, None]
    base[cells] = base[cells] * (1.0 - a) + rgb * a


def settled(model) -> np.ndarray:
    t = model.settlements
    return t.alive_ids()


def cell_colors(model, vcfg: VizConfig, layer: str = "polity", shade_by: str = "population density",
                technique: str | None = None) -> np.ndarray:
    """(cells, 3) uint8: the map for ``layer`` (``LAYERS``) over terrain and sea."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}")
    out = base_colors(model)
    t = model.settlements
    ids = settled(model)
    cells = t.cell[ids]
    if layer == "polity":
        c, rgba = polity_fill(model, vcfg, shade_by)
        out[c] = rgba[:, :3]
    elif layer == "population density":
        x = np.clip(t.population[ids] / vcfg.population_vmax, 0.0, 1.0)
        k = vcfg.population_alpha_curve
        alpha = np.log1p(k * x) / np.log1p(k) if k > 0 else x
        _blend(out, cells, np.array(to_rgb(vcfg.population_color)), alpha)
    elif layer == "subsistence mode":
        modes = _rgb([vcfg.mode_colors[m] for m in MODES])
        out[cells] = t.shares[ids] @ modes
    elif layer == "cohesion":
        owner = claimed(model)
        c = np.flatnonzero(owner >= 0)
        out[c] = colormaps[vcfg.cohesion_cmap](np.clip(model.polities.cohesion[owner[c]], 0.0, 1.0))[:, :3]
    elif layer == "technique":
        if technique is not None:
            from deluge.systems.knowledge import usable_mask

            j = model.knowledge.index(technique)
            holds = t.knows[ids, j]
            usable = usable_mask(model, ids)[:, j] if len(ids) and model.capacity else np.zeros(len(ids), bool)
            out[cells[holds & usable]] = to_rgb(vcfg.knowledge_color)
            out[cells[holds & ~usable]] = to_rgb(vcfg.latent_color)
    elif layer == "lineage":
        out[cells] = _rgb(lineage_palette(vcfg))[t.lineage[ids]]
    elif layer == "resources":
        code = model.resources_layer.data.ravel().astype(np.int64)
        c = np.flatnonzero(code > 0)
        out[c] = _rgb(resource_colors(vcfg))[code[c] - 1]
    elif layer == "techniques known":
        known = model.knowledge_layer.data.ravel()
        c = np.flatnonzero(known > 0)
        _blend(out, c, np.array(to_rgb(vcfg.knowledge_color)), np.clip(known[c] / vcfg.knowledge_vmax, 0.0, 1.0))
    return np.rint(np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


# --- borders ------------------------------------------------------------------------------------------


def enclaves(model, owner: np.ndarray | None = None) -> np.ndarray:
    """Per cell: an isolated single cell of its polity (claimed, no neighbour of the same
    polity) while that polity claims other cells too."""
    owner = claimed(model) if owner is None else owner
    nb = model.geom.neighbors
    padded = np.append(owner, -2)
    same = (padded[nb] == owner[:, None]) & (owner[:, None] >= 0)
    alone = (owner >= 0) & ~same.any(axis=1)
    counts = np.bincount(owner[owner >= 0], minlength=model.polities.n)
    return alone & (counts[np.maximum(owner, 0)] > 1)


def polity_borders(model) -> tuple[np.ndarray, np.ndarray]:
    """Shared hex edges between cells claimed by different polities, as (n, 2, 2) segments in
    map units: (between founding groups, between polities of one group). Edges of enclave
    cells (``enclaves``) are left out."""
    owner = claimed(model)
    lone = enclaves(model, owner)
    nb = model.geom.neighbors
    c, k = np.nonzero(nb >= 0)
    n = nb[c, k]
    keep = (c < n) & (owner[c] >= 0) & (owner[n] >= 0) & (owner[c] != owner[n]) & ~lone[c] & ~lone[n]
    c, n = c[keep], n[keep]
    pos = model.geom.positions
    mid = 0.5 * (pos[c] + pos[n])
    d = pos[n] - pos[c]
    perp = np.column_stack([-d[:, 1], d[:, 0]]) / np.linalg.norm(d, axis=1)[:, None]
    seg = np.stack([mid - 0.5 * perp, mid + 0.5 * perp], axis=1)  # hex side length 1
    lin = model.polities.lineage
    apart = lin[owner[c]] != lin[owner[n]]
    return seg[apart], seg[~apart]


def line_sets(model, vcfg: VizConfig, layer: str) -> list[tuple[np.ndarray, float, str]]:
    """(segments (n, 2, 2), width px, colour) in drawing order: the polity borders on the polity
    layer, each over a light outline."""
    if layer != "polity":
        return []
    peoples, polities = polity_borders(model)
    out = []
    for seg, width in ((polities, vcfg.border_polities_width), (peoples, vcfg.border_peoples_width)):
        if len(seg):
            out.append((seg, width + vcfg.border_outline_width, vcfg.border_outline_color))
            out.append((seg, width, vcfg.border_color))
    return out


# --- overlays -----------------------------------------------------------------------------------------


class WarTracker:
    """Reads the event log as it grows: per polity pair at war, who attacked in its last
    battle and whether that battle crossed the sea."""

    def __init__(self) -> None:
        self.cursor = 0
        self.last: dict[tuple[int, int], tuple[int, int, int, bool]] = {}  # pair -> (year, attacker, defender, sea)

    def update(self, model) -> None:
        ev = model.events
        n = min(len(ev.year), len(ev.action), len(ev.actors), len(ev.link))
        fought, crossed = ACTION_CODE["fought"], ACTION_CODE["crossed"]
        for i in range(self.cursor, n):
            code = ev.action[i]
            if code != fought and code != crossed:
                continue
            actors = [decode(a) for a in ev.actors[i]]
            if not actors or actors[0][0] != "polity":
                continue
            a = actors[0][1]
            b = next((x for kind, x in actors[1:] if kind == "polity"), -1)
            if b < 0:
                continue
            pair = (min(a, b), max(a, b))
            if code == fought:
                self.last[pair] = (ev.year[i], a, b, False)
            elif pair in self.last:
                y, a0, b0, _ = self.last[pair]
                self.last[pair] = (y, a0, b0, True)
        self.cursor = n


def marks(model, vcfg: VizConfig, wars: WarTracker | None = None) -> dict:
    """Overlay marks in map units: capitals, monuments, the teachers' path and position, wars."""
    P, t = model.polities, model.settlements
    pos = model.geom.positions
    palette = lineage_palette(vcfg)
    pids = P.alive_ids()
    cap_cell = t.cell[P.capital[pids]]
    dots = [[float(pos[c, 0]), float(pos[c, 1]), palette[int(P.lineage[p])]] for p, c in zip(pids.tolist(),
                                                                                           cap_cell.tolist())]
    mon = model.monuments
    alive = (t.ended[mon.settlement] < 0) & mon.maintained if len(mon) else np.zeros(0, dtype=bool)
    squares = [[float(pos[c, 0]), float(pos[c, 1])] for c in np.unique(mon.cell[alive]).tolist()]
    out = {"dots": dots, "squares": squares, "dot_px": vcfg.capital_px, "square_px": vcfg.monument_px,
           "square_color": vcfg.monument_color, "arrow_color": vcfg.war_color, "arrow_px": vcfg.war_px,
           "path_color": vcfg.teachers_color, "path_px": vcfg.teachers_px, "arrows": [], "path": [], "here": None}
    ts = model.teachers_state
    if ts.present or (ts.visits and not ts.departed):
        cells = [int(t.cell[v["settlement"]]) for v in ts.visits if v.get("settlement", -1) >= 0]
        out["path"] = [[float(pos[c, 0]), float(pos[c, 1])] for c in cells]
        if ts.agent is not None:
            x, y = ts.agent.cell.coordinate
            c = model.geom.cid(int(x), int(y))
            out["here"] = [float(pos[c, 0]), float(pos[c, 1])]
    if wars is not None:
        wars.update(model)
        for pair, year in sorted(model.wars.items()):
            y, a, b, sea = wars.last.get(pair, (year, pair[0], pair[1], False))
            if P.ended[a] >= 0 or P.ended[b] >= 0:
                continue
            ca, cb = int(t.cell[P.capital[a]]), int(t.cell[P.capital[b]])
            out["arrows"].append([float(pos[ca, 0]), float(pos[ca, 1]), float(pos[cb, 0]), float(pos[cb, 1]),
                                  bool(sea and y == year)])
    return out


# --- hover --------------------------------------------------------------------------------------------


def cell_snapshot(model) -> dict[str, np.ndarray]:
    """What a cell's hover text needs, copied at a consistent moment (between steps)."""
    t = model.settlements
    ids = t.alive_ids()
    sid = np.full(model.geom.n_cells, -1, dtype=np.int64)
    sid[t.cell[ids]] = ids
    pop = model.cell_population()
    owner = claimed(model)
    return {"settlement": sid, "population": pop, "polity": owner,
            "settlement_polity": np.where(sid >= 0, t.polity[np.maximum(sid, 0)], -1),
            "lineage": np.where(sid >= 0, t.lineage[np.maximum(sid, 0)], -1),
            "polity_lineage": np.where(owner >= 0, model.polities.lineage[np.maximum(owner, 0)], -1),
            "sea": model.is_sea.copy()}


def tooltip(model, snap: dict[str, np.ndarray], cell: int) -> str:
    """HTML hover text of ``cell``: terrain, elevation, settlement, polity, lineage, population."""
    names = model.lineages.names
    x, y = model.geom.xy(cell)
    terrain = "sea" if snap["sea"][cell] else Terrain(int(model.world.terrain.ravel()[cell])).name.lower()
    lines = [f"<b>cell ({x}, {y})</b>: {terrain}, elevation {model.elevation_flat[cell]:,.0f} m"]
    sid = int(snap["settlement"][cell])
    if sid >= 0:
        sp, g = int(snap["settlement_polity"][cell]), int(snap["lineage"][cell])
        lines.append(f"settlement {sid} of polity {sp}, group {names[g]}")
        lines.append(f"population {snap['population'][cell]:,.0f}")
    else:
        lines.append("no settlement")
    p = int(snap["polity"][cell])
    if p >= 0 and (sid < 0 or p != int(snap["settlement_polity"][cell])):
        lines.append(f"claimed by polity {p}, group {names[int(snap['polity_lineage'][cell])]}")
    elif p < 0 and sid < 0:
        lines.append("unclaimed")
    return "<br>".join(lines)


# --- text ---------------------------------------------------------------------------------------------


def legend_html(vcfg: VizConfig, layer: str, group_names, technique: str | None = None) -> str:
    """Colour key for the layer: one row per entry, its swatch followed by its own label."""
    palette = lineage_palette(vcfg)
    note = ""
    if layer == "lineage":
        items = [(f"group {n}", c) for n, c in zip(group_names, palette)]
    elif layer == "polity":
        items = [(f"polities of group {n}", c) for n, c in zip(group_names, palette)]
        note = ("Darker: more of the shade-by variable. Thick lines: borders between founding groups; thin lines: "
                "between polities of one group. Enclave cells have no outline of their own.")
    elif layer == "resources":
        items = list(zip([m.replace("_", " ") for m in MINERALS] + ["several"], resource_colors(vcfg)))
    elif layer == "subsistence mode":
        items = [(m, vcfg.mode_colors[m]) for m in MODES]
        note = "Mixed modes are drawn as the share-weighted mix of the colours."
    elif layer == "technique":
        items = [(f"{technique or 'technique'}: usable", vcfg.knowledge_color),
                 (f"{technique or 'technique'}: held, not usable there", vcfg.latent_color)]
    elif layer == "cohesion":
        cmap = colormaps[vcfg.cohesion_cmap]
        items = [(f"cohesion {v:.1f}", "#%02x%02x%02x" % tuple(int(255 * c) for c in cmap(v)[:3]))
                 for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    else:
        return ""
    rows = "".join(f'<div style="display:flex;align-items:center;gap:0.4em;white-space:nowrap">'
                   f'<span style="flex:none;display:inline-block;width:0.9em;height:0.9em;background:{c};'
                   f'border:1px solid #555"></span><span>{label}</span></div>' for label, c in items)
    if note:
        note = f'<div style="margin-top:0.3em">{note}</div>'
    return f'<div class="legend">{rows}{note}</div>'


def shown_year(model) -> int:
    """The year the map shows: the last simulated year (year 0 before the first step,
    when the model already holds year 0's sea)."""
    return max(model.year - 1, 0)


def status_text(model) -> str:
    year = shown_year(model)
    modes = ", ".join(f"{m} {p:,.0f}" for m, p in zip(("foraging", "herding", "farming"), model.mode_population))
    ts = model.teachers_state
    teachers = ("present" if ts.present else "departed" if ts.departed and ts.visits
                else "not arrived" if model.config.teachers.enabled else "disabled")
    lines = [
        f"**Year {year:,}** of {model.config.run.years:,}",
        f"Sea level: {model.sea_level_series[year]:+.1f} m",
        f"Land cells: {int(model.is_sea.size - model.is_sea.sum()):,}",
        f"Temperature anomaly: {model.temp_anomaly:+.2f} °C",
        f"Population: {model.population:,.0f} in {model.n_settlements:,} settlements",
        "By group: " + ", ".join(f"{name} {p:,.0f}" for name, p in
                                 zip(model.lineages.names, model.lineage_population)),
        f"By mode: {modes}",
        f"Techniques per settlement: {model.techniques_known:.1f}; teachers: {teachers}",
        f"Polities: {model.n_polities:,}; mean cohesion {model.mean_cohesion:.2f}; largest "
        f"{model.largest_polity_settlements:,} settlements, {model.largest_polity_population:,.0f} people; "
        f"wars {model.wars_active:,}",
        f"World seed: {model.config.run.world_seed}, history seed: {model.config.run.history_seed}",
    ]
    return "  \n".join(lines)
