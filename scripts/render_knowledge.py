"""Knowledge maps, population by subsistence mode, and the teachers' visits.

    python scripts/render_knowledge.py --world-seed 1 --history-seed 1

Writes to runs/knowledge_w<world>_h<history>/ (git-ignored):
- techniques_year_<year>.png for each year in render.knowledge_years: one small map per
  technique, cells whose settlement holds it marked, latent holdings (held where they
  cannot be used) in their own colour;
- population_by_mode.png: people working each subsistence mode over time;
- teachers_timeline.png: which group the teachers stayed with, when, what passed on and
  what they left;
- events.parquet: the run's raw event log.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from deluge.agents.settlement import MODES  # noqa: E402
from deluge.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from deluge.model import DelugeModel  # noqa: E402
from deluge.systems.knowledge import usable_mask  # noqa: E402
from deluge.world import render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
HOLDS = "#1a9850"
LATENT = "#9970ab"
EMPTY_LAND = "#d9d9d9"
SETTLED = "#fdae61"


def technique_panels(m: DelugeModel, path: Path, dpi: int) -> Path:
    names = m.knowledge.names
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.7 * rows), dpi=dpi)
    t = m.settlements
    ids = t.alive_ids()
    verts = m.geom.hex_vertices()
    base = np.where(m.is_sea[:, None], np.array(to_rgba(render.SEA_COLOR)), np.array(to_rgba(EMPTY_LAND)))
    base[t.cell[ids]] = to_rgba(SETTLED)
    usable = usable_mask(m, ids)
    for j, ax in enumerate(axes.ravel()):
        ax.set_axis_off()
        if j >= len(names):
            continue
        colors = base.copy()
        held = t.knows[ids, j]
        holders = ids[held & usable[:, j]]
        latent = ids[held & ~usable[:, j]]
        colors[t.cell[holders]] = to_rgba(HOLDS)
        colors[t.cell[latent]] = to_rgba(LATENT)
        coll = PolyCollection(verts, linewidths=0, antialiased=False)
        coll.set_facecolor(colors)
        ax.add_collection(coll)
        pos = m.geom.positions
        ax.set_xlim(pos[:, 0].min() - 1, pos[:, 0].max() + 1)
        ax.set_ylim(pos[:, 1].min() - 1, pos[:, 1].max() + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{names[j].replace('_', ' ')}: {len(holders)} + {len(latent)} latent of {len(ids)}",
                     fontsize=9)
    year = max(m.year - 1, 0)
    fig.suptitle(f"Techniques held, year {year} (green: holds it and can use it here; purple: holds it latent, "
                 f"unusable here; orange: settlement without it; grey: empty land)", fontsize=11)
    return render._save(fig, path)


def mode_curve(df, path: Path, title: str, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=dpi)
    for mode in MODES:
        ax.plot(df.index, df[f"pop_{mode}"], label=mode)
    ax.plot(df.index, df["population"], color="k", linewidth=1.0, label="total")
    ax.set_yscale("log")
    ax.set_ylim(bottom=10)
    ax.set_xlabel("year")
    ax.set_ylabel("people")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    return render._save(fig, path)


def teachers_timeline(m: DelugeModel, path: Path, dpi: int) -> Path:
    visits = m.teachers_state.visits
    names = m.lineages.names
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=dpi)
    for v in visits:
        y = names.index(names[v["lineage"]])
        end = v["end"] if v["end"] is not None else m.year
        ax.barh(y, end - v["start"] + 1, left=v["start"], color="#4575b4", alpha=0.8)
        taught = ", ".join(n.replace("_", " ") for n in v["taught"]) or "nothing"
        left = f"; left: {', '.join(v['artifacts'])}" if v["artifacts"] else ""
        ax.text(end + 1.5, y, f"taught: {taught}{left}", va="center", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"group {n}" for n in names])
    ax.set_xlabel("year")
    last = max((v["end"] or 0) for v in visits) if visits else 1
    ax.set_xlim(-1, last + 60)
    ax.set_title("The teachers' visits (bars: years with each group)")
    ax.invert_yaxis()
    return render._save(fig, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--world-seed", type=int, default=None)
    parser.add_argument("--history-seed", type=int, default=None)
    parser.add_argument("--years", type=int, default=None, help="run length (default: the configured run)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    world = cfg.run.world_seed if args.world_seed is None else args.world_seed
    history = cfg.run.history_seed if args.history_seed is None else args.history_seed
    years = args.years or cfg.run.years
    out = REPO_ROOT / cfg.output.runs_dir / f"knowledge_w{world}_h{history}"
    out.mkdir(parents=True, exist_ok=True)
    m = DelugeModel(config=cfg, world_seed=world, history_seed=history, years=years)
    marks = sorted(y for y in cfg.render.knowledge_years if y < years)
    paths = []
    while m.running:
        m.step()
        if m.year - 1 in marks:
            paths.append(technique_panels(m, out / f"techniques_year_{m.year - 1:05d}.png", cfg.render.dpi))
    df = m.datacollector.get_model_vars_dataframe().set_index("year")
    paths.append(mode_curve(df, out / "population_by_mode.png",
                            f"Population by subsistence mode, world {world}, history {history}", cfg.render.dpi))
    paths.append(teachers_timeline(m, out / "teachers_timeline.png", cfg.render.dpi))
    m.events.write(out / "events.parquet")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
