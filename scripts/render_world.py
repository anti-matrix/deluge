"""Render a world to PNG files and summarize its geography.

    python scripts/render_world.py --seeds 1 2 3

For each seed, writes to runs/maps_seed<seed>/ (git-ignored): physical.png, flooding.png,
the per-layer maps, capacity maps at year 0, climate_curves.png and summary.json. Prints a
summary of each world.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from deluge.config import DEFAULT_CONFIG_PATH, load_config
from deluge.model import DelugeModel
from deluge.rng import RngStreams
from deluge.world import geography, procedural, render
from deluge.world.summary import summarize

REPO_ROOT = Path(__file__).resolve().parent.parent
LANDMASS_COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def render_seed(cfg, out: Path, layers: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    world, report = procedural.generate_world_with_report(cfg, RngStreams(cfg.run.world_seed, cfg.rng.streams)["world"])
    model = DelugeModel(config=cfg)  # same seed: same world; used for curves and capacity
    assert np.array_equal(model.world.elevation, world.elevation)
    model.step()  # year 0: climate and capacity
    geom, rc = model.geom, cfg.render
    lats = (cfg.world.lat_south, cfg.world.lat_north)
    curve = model.sea_level_series
    sea0 = float(curve[0])
    elev = world.elevation.ravel()
    sea = elev < sea0

    summary = summarize(cfg, world, report)
    marks = [(geom.cid(b.col, b.row), f"bridge drowns yr {b.year_drowned} ({b.crest_m:+.0f} m)")
             for b in summary.bridges if b.during_pulse]
    paths = [
        render.physical_map(geom, world, sea0, f"Physical map, world seed {cfg.run.world_seed}, year 0 "
                            f"(sea level {sea0:+.0f} m)", out / "physical.png", model.unit_km, rc, lats, marks),
        render.flooding_panels(geom, world, {y: float(curve[y]) for y in rc.flood_years}, out / "flooding.png",
                               model.unit_km, model.cell_area_km2, rc, lats, marks),
    ]
    if layers:
        paths += [
            render.elevation_map(geom, elev, sea0, "Elevation, coast at year 0", out / "elevation.png", lats, rc.dpi),
            render.terrain_map(geom, world.terrain.ravel(), "Terrain at year 0", out / "terrain.png", lats, rc.dpi),
            render.field_map(geom, world.base_temp.ravel(), sea, "Base temperature", out / "base_temp.png",
                             "coolwarm", Normalize(-25, 32), "°C", lats, rc.dpi),
            render.field_map(geom, world.base_rain.ravel(), sea, "Base rainfall", out / "base_rain.png",
                             "YlGnBu", Normalize(0, 2500), "mm / year", lats, rc.dpi),
            render.field_map(geom, world.fertility.ravel(), sea, "Fertility", out / "fertility.png",
                             "YlGn", Normalize(0, 1), "0..1", lats, rc.dpi),
            render.field_map(geom, world.wild_food.ravel(), sea, "Wild food (without coast bonus)",
                             out / "wild_food.png", "YlOrBr", Normalize(0, 1), "0..1", lats, rc.dpi),
            render.climate_curves(curve, model.temp_anomaly_series, out / "climate_curves.png", rc.dpi),
        ]
        for mode in ("forage", "herd", "farm"):
            k = model.capacity[mode]
            pos = k[k > 0]
            norm = LogNorm(max(pos.min(), 1.0), pos.max()) if len(pos) else Normalize(0, 1)
            paths.append(render.field_map(
                geom, np.where(k > 0, k, np.nan), sea,
                f"Carrying capacity, {mode}, year 0 (total {k.sum():,.0f} people; grey = 0)",
                out / f"capacity_{mode}.png", "viridis", norm, "people per cell", lats, rc.dpi))
        rep = geography.landmass_report(geom, elev, sea0, cfg.world.landmasses)
        colors = np.full(len(elev), render.LAND_COLOR, dtype=object)
        colors[sea] = render.SEA_COLOR
        legend = [("sea", render.SEA_COLOR), (f"islands ({rep.islands})", render.LAND_COLOR)]
        for i, size in enumerate(rep.landmasses):
            c = LANDMASS_COLORS[i % len(LANDMASS_COLORS)]
            colors[rep.labels == i] = c
            legend.append((f"landmass {i + 1}: {size} cells", c))
        paths.append(render.categorical_map(geom, f"Landmasses at year 0 ({len(rep.landmasses)})",
                                            out / "landmasses.png", lats, legend, list(colors), rc.dpi))
    data = summary.to_dict()
    data["images"] = [str(p) for p in paths]
    (out / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="world seeds")
    parser.add_argument("--no-layers", action="store_true", help="only the physical and flooding maps")
    args = parser.parse_args(argv)

    base = load_config(args.config)
    for seed in args.seeds or [base.run.world_seed]:
        cfg = base.with_overrides({"run": {"world_seed": seed}})
        out = REPO_ROOT / cfg.output.runs_dir / f"maps_seed{seed}"
        data = render_seed(cfg, out, layers=not args.no_layers)
        print(json.dumps({k: v for k, v in data.items() if k != "images"}, indent=1))
        for p in data["images"]:
            print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
