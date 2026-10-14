"""Render the world layers of one seed to PNG files.

    python scripts/render_world.py --seed 42

Writes to runs/maps_seed<seed>/ by default (git-ignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from deluge.config import DEFAULT_CONFIG_PATH, load_config
from deluge.model import DelugeModel
from deluge.world import render
from deluge.world.procedural import landmass_report

REPO_ROOT = Path(__file__).resolve().parent.parent
LANDMASS_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--flood-years", default="0,1000,2000,3000")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = cfg.with_overrides({"run": {"seed": args.seed}})
    out = Path(args.out) if args.out else REPO_ROOT / cfg.output.runs_dir / f"maps_seed{cfg.run.seed}"
    out.mkdir(parents=True, exist_ok=True)

    model = DelugeModel(config=cfg)
    model.step()  # year 0: climate and capacity
    geom, world = model.geom, model.world
    lats = (cfg.world.lat_south, cfg.world.lat_north)
    sea0 = float(model.sea_level_series[0])
    elev = world.elevation.ravel()
    sea = elev < sea0
    paths = []

    paths.append(render.elevation_map(geom, elev, sea0, f"Elevation (seed {cfg.run.seed}), coast at year 0",
                                      out / "elevation.png", lats))
    paths.append(render.terrain_map(geom, world.terrain.ravel(), "Terrain at year 0", out / "terrain.png", lats))
    paths.append(render.river_map(geom, world, sea0, f"Rivers ({int((world.river > 0).sum())} cells)",
                                  out / "rivers.png", lats))
    paths.append(render.field_map(geom, world.base_temp.ravel(), sea, "Base temperature", out / "base_temp.png",
                                  "coolwarm", Normalize(-20, 30), "°C", lats))
    paths.append(render.field_map(geom, world.base_rain.ravel(), sea, "Base rainfall", out / "base_rain.png",
                                  "YlGnBu", Normalize(0, 2500), "mm / year", lats))
    paths.append(render.field_map(geom, world.fertility.ravel(), sea, "Fertility", out / "fertility.png",
                                  "YlGn", Normalize(0, 1), "0..1", lats))
    paths.append(render.field_map(geom, world.wild_food.ravel(), sea, "Wild food", out / "wild_food.png",
                                  "YlOrBr", Normalize(0, 1), "0..1", lats))
    for mode in ("forage", "herd", "farm"):
        k = model.capacity[mode]
        positive = k[k > 0]
        norm = LogNorm(max(positive.min(), 1.0), positive.max()) if len(positive) else Normalize(0, 1)
        paths.append(render.field_map(geom, np.where(k > 0, k, np.nan), sea,
                                      f"Carrying capacity, {mode}, year 0 (total {k.sum():,.0f} people; grey = 0)",
                                      out / f"capacity_{mode}.png", "viridis", norm, "people per cell", lats))

    rep = landmass_report(geom, elev, sea0, cfg.world.landmasses)
    colors = np.full(len(elev), render.LAND_COLOR, dtype=object)
    colors[sea] = render.SEA_COLOR
    legend = [("sea", render.SEA_COLOR), ("islands", render.LAND_COLOR)]
    for i, size in enumerate(rep.landmasses):
        c = LANDMASS_COLORS[i % len(LANDMASS_COLORS)]
        colors[rep.labels == i] = c
        legend.append((f"landmass {i + 1}: {size} cells", c))
    paths.append(render.categorical_map(geom, f"Landmasses at year 0 ({len(rep.landmasses)})",
                                        out / "landmasses.png", lats, legend, list(colors)))

    years = [int(y) for y in args.flood_years.split(",")]
    levels = {y: float(model.sea_level_series[y]) for y in years}
    for y in years:
        paths.append(render.land_sea_map(geom, elev, y, levels[y], levels[years[0]],
                                         out / f"land_sea_year_{y:05d}.png", lats))
    paths.append(render.land_sea_panels(geom, elev, levels, out / "land_sea_panels.png", lats))
    paths.append(render.climate_curves(model.sea_level_series, model.temp_anomaly_series,
                                       out / "climate_curves.png"))

    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
