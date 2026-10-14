"""Population maps, the per-group curve and the Round 3 acceptance numbers.

    python scripts/render_population.py --world-seed 1 --history-seeds 1 2 3 4 5

For the first history seed, writes to runs/population_w<world>_h<history>/ (git-ignored)
a population-density map at each year in render.population_years and population per
founding group over time. For every history seed, records total population, settlements
and each group's population at the last of those years in summary.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from deluge.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from deluge.model import DelugeModel  # noqa: E402
from deluge.world import render  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_history(cfg, world_seed: int, history_seed: int, out: Path | None) -> dict:
    years = cfg.render.population_years
    m = DelugeModel(config=cfg, world_seed=world_seed, history_seed=history_seed, years=max(years) + 1)
    lats = (cfg.world.lat_south, cfg.world.lat_north)
    rc = cfg.render
    paths = []
    while True:
        shown = max(m.year - 1, 0)  # the year the model state now holds
        if out is not None and shown in years and not any(p.stem.endswith(f"{shown:05d}") for p in paths):
            dens = m.cell_population() / m.cell_area_km2
            title = (f"People per km2, world {world_seed}, history {history_seed}, year {shown} "
                     f"(total {m.population:,.0f} in {m.n_settlements:,} settlements; grey = empty land)")
            paths.append(render.field_map(m.geom, np.where(dens > 0, dens, np.nan), m.is_sea, title,
                                          out / f"population_year_{shown:05d}.png", "magma_r",
                                          LogNorm(rc.population_density_min, rc.population_density_max),
                                          "people / km2", lats, rc.dpi))
        if m.year > max(years):
            break
        m.step()
    df = m.datacollector.get_model_vars_dataframe().set_index("year")
    last = max(years)
    result = {
        "history_seed": history_seed,
        "year": last,
        "population": float(df.loc[last, "population"]),
        "settlements": int(df.loc[last, "n_settlements"]),
        "groups": {n: float(df.loc[last, f"pop_{n}"]) for n in m.lineages.names},
        "founders": [{"name": f.name, "col_row": m.geom.xy(f.cell), "people": f.population}
                     for f in m.lineages.founders],
    }
    if out is not None:
        fig, ax = plt.subplots(figsize=(11, 5.5), dpi=rc.dpi)
        for n in m.lineages.names:
            ax.plot(df.index, df[f"pop_{n}"], label=f"group {n}")
        ax.plot(df.index, df["population"], color="k", linewidth=1.2, label="total")
        ax.set_yscale("log")
        ax.set_xlabel("year")
        ax.set_ylabel("people")
        ax.set_title(f"Population per founding group, world {world_seed}, history {history_seed}")
        ax.grid(alpha=0.3)
        ax.legend()
        paths.append(render._save(fig, out / "population_by_group.png"))
        result["images"] = [str(p) for p in paths]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--world-seed", type=int, default=None)
    parser.add_argument("--history-seeds", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    world = cfg.run.world_seed if args.world_seed is None else args.world_seed
    histories = args.history_seeds or [cfg.run.history_seed]
    out = REPO_ROOT / cfg.output.runs_dir / f"population_w{world}_h{histories[0]}"
    out.mkdir(parents=True, exist_ok=True)
    results = [run_history(cfg, world, h, out if i == 0 else None) for i, h in enumerate(histories)]
    names = list(results[0]["groups"])
    summary = {
        "world_seed": world,
        "runs": results,
        "groups_extinct_in_every_run": [n for n in names if all(r["groups"][n] == 0 for r in results)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in results:
        print(f"history {r['history_seed']}: year {r['year']} population {r['population']:,.0f} "
              f"settlements {r['settlements']} groups " + ", ".join(f"{k} {v:,.0f}" for k, v in r["groups"].items()))
    print("groups extinct in every run:", summary["groups_extinct_in_every_run"])
    for p in results[0].get("images", []):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
