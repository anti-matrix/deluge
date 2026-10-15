"""Round 7 acceptance: brains and neuroevolution (spec sections 10, 13), run manually.

    python scripts/acceptance/round7_acceptance.py --world-seed 1 --seeds 1 2 3 4 5 --years 5000

Runs every history seed under three conditions: normal, frozen brains (no mutation, every
polity carries its founder's brain) and blind perception (every perceived appearance and
dialect distance zero), in parallel (at most cores - 1 processes). Reported per run:

- Outcomes: population, polity births, polity count over time, the largest polity's size
  and lifespan, battles and conquests per millennium, median cohesion, polity lifespans.
- Brain outputs: the mean of each output over living polities every 500 years, and its
  spread (sd) across polities then.
- Founders vs survivors: for each living polity at the end, the distance of its genome
  from its founding polity's (L2 and per weight), its generations from the founder, and
  how differently the two genomes behave: mean |output difference| on the polity's actual
  inputs at the end and on a fixed random input set.
- Appearance and dialect (descriptive, no target): in the living polities, the target
  head's weight on perceived appearance and dialect distance and the marker culture
  parameters (sign and size); and whether war targets and seceding settlements differ in
  perceived distance from what was available (the analysis wraps ``war.target_features``,
  ``war.choose`` and ``politics.secession`` in this process only and reads the markers
  only through ``identity`` perception).

Writes runs/round7_acceptance_w<world>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "normal": {},
    "frozen": {"brains": {"frozen": True}},
    "blind": {"ancestry": {"blind_perception": True}},
}


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def run_job(world_seed: int, history_seed: int, years: int, condition: str, every: int) -> dict:
    from deluge.agents.polity import END_CAUSES, MARKER_EFFECTS
    from deluge.brains import features, network
    from deluge.config import load_config
    from deluge.model import DelugeModel
    from deluge.systems import identity, politics, war

    cfg = load_config(overrides=CONDITIONS[condition])
    t0 = time.perf_counter()
    m = DelugeModel(config=cfg, world_seed=world_seed, history_seed=history_seed, years=years)
    t, P, B = m.settlements, m.polities, m.brains
    layout = m.brain_layout
    tcol = {name: i for i, name in enumerate(features.TARGET_FEATURES)}

    # --- analysis hooks (this process only) ---------------------------------------------------
    war_rec = {"chosen_app": [], "chosen_dia": [], "avail_app": [], "avail_dia": [], "own_mean_app": [],
               "own_mean_dia": []}
    last = {}
    orig_tf, orig_choose, orig_secession = war.target_features, war.choose, politics.secession

    def tf_hook(model, a, b, x, p_win):
        f = orig_tf(model, a, b, x, p_win)
        last["a"], last["x"], last["f"] = a.copy(), x.copy(), f
        return f

    def choose_hook(model, F):
        last.clear()
        attacks = orig_choose(model, F)
        if attacks and "f" in last:
            a, x, f = last["a"], last["x"], last["f"]
            war_rec["avail_app"] += f[:, tcol["appearance_distance"]].tolist()
            war_rec["avail_dia"] += f[:, tcol["dialect_distance"]].tolist()
            for att, _, _, tgt, _ in attacks:
                rows = np.flatnonzero(a == att)
                hit = rows[x[rows] == tgt][0]
                war_rec["chosen_app"].append(float(f[hit, tcol["appearance_distance"]]))
                war_rec["chosen_dia"].append(float(f[hit, tcol["dialect_distance"]]))
                war_rec["own_mean_app"].append(float(f[rows, tcol["appearance_distance"]].mean()))
                war_rec["own_mean_dia"].append(float(f[rows, tcol["dialect_distance"]].mean()))
        return attacks

    sec_rec = {"leaver_app": [], "leaver_dia": [], "stayer_app": 0.0, "stayer_dia": 0.0, "stayers": 0}

    def secession_hook(model):
        ids = t.alive_ids()
        before = t.polity[ids].copy()
        app, dia = identity.core_distances(model, ids)
        not_capital = ids != P.capital[before]
        n0 = P.n
        orig_secession(model)
        left = (t.polity[ids] != before) & (t.polity[ids] >= n0)
        sec_rec["leaver_app"] += app[left].tolist()
        sec_rec["leaver_dia"] += dia[left].tolist()
        stay = not_capital & ~left
        sec_rec["stayer_app"] += float(app[stay].sum())
        sec_rec["stayer_dia"] += float(dia[stay].sum())
        sec_rec["stayers"] += int(stay.sum())

    war.target_features, war.choose, politics.secession = tf_hook, choose_hook, secession_hook

    snaps = []
    while m.running:
        m.step()
        if m.year % every == 0 or not m.running:
            pids = P.alive_ids()
            out = B.output[pids]
            snaps.append({"year": m.year, "polities": int(len(pids)),
                          "output_sd": {k: _r(v, 3) for k, v in zip(network.OUTPUTS, out.std(axis=0))},
                          "marker_mean": {k: _r(v, 3) for k, v in zip(MARKER_EFFECTS, P.marker[pids].mean(axis=0))}})
    war.target_features, war.choose, politics.secession = orig_tf, orig_choose, orig_secession
    seconds = time.perf_counter() - t0

    # --- outcomes -------------------------------------------------------------------------------
    df = m.datacollector.get_model_vars_dataframe()
    n_pol = df["n_polities"].to_numpy()
    n = P.n
    life = np.where(P.ended[:n] >= 0, P.ended[:n], m.year) - P.founded[:n]
    ended = P.ended[:n] >= 0
    big = int(np.argmax(P.peak_settlements[:n]))
    per_millennium = {col: [int(df[col].to_numpy()[k:k + 1000].sum()) for k in range(0, len(df), 1000)]
                      for col in ("battles", "conquests", "secessions", "fragmentations", "polities_destroyed")}
    per_millennium["wars_active_mean"] = [round(float(df["wars_active"].to_numpy()[k:k + 1000].mean()), 1)
                                          for k in range(0, len(df), 1000)]
    brain_means = {name: [_r(df[f"brain_{name}"].to_numpy()[k:k + every].mean(), 3)
                          for k in range(0, len(df), every)] for name in network.OUTPUTS}

    # --- founders vs survivors ------------------------------------------------------------------
    pids = P.alive_ids()
    founder = np.array([m.lineages.polity_ancestry(P, int(p))[-1] for p in pids.tolist()], dtype=np.int64)
    depth = np.array([len(m.lineages.polity_ancestry(P, int(p))) - 1 for p in pids.tolist()])
    g_now = B.genome[pids].astype(np.float64)
    g_0 = B.genome[founder].astype(np.float64)
    l2 = np.linalg.norm(g_now - g_0, axis=1)
    x_now = features.assemble(m, pids)
    out_now = network.forward(layout, B.genome[pids], x_now)[0]
    out_0 = network.forward(layout, B.genome[founder], x_now)[0]
    rng = np.random.default_rng(12345)
    x_std = rng.uniform(-1.0, 1.0, (256, layout.n_in))
    x_std[:, [i for i, name in enumerate(layout.inputs) if name.startswith(("memory_", "recurrent_"))]] = 0.0
    std_diff = []
    for p, f in zip(pids.tolist(), founder.tolist()):
        o1 = network.forward(layout, np.repeat(B.genome[[p]], len(x_std), 0), x_std)[0]
        o0 = network.forward(layout, np.repeat(B.genome[[f]], len(x_std), 0), x_std)[0]
        std_diff.append(float(np.abs(o1 - o0).mean()))
    std_diff = np.array(std_diff)
    sigma = cfg.brains
    drift_sd = np.sqrt(sigma.mutation_sigma ** 2 + sigma.jump_chance * sigma.jump_sigma ** 2)
    survivors = {}
    for g in np.unique(P.lineage[pids]).tolist():
        sel = P.lineage[pids] == g
        survivors[m.lineages.names[g]] = {
            "polities": int(sel.sum()),
            "generations_median": _r(np.median(depth[sel]), 1),
            "weight_l2_median": _r(np.median(l2[sel])),
            "weight_l2_neutral_expectation": _r(drift_sd * np.sqrt(layout.size * np.median(depth[sel]))),
            "weight_mean_abs_change": _r(np.median(np.abs(g_now[sel] - g_0[sel]).mean(axis=1))),
            "output_diff_actual_inputs": _r(np.median(np.abs(out_now[sel] - out_0[sel]).mean(axis=1))),
            "output_diff_random_inputs": _r(np.median(std_diff[sel])),
        }

    # --- appearance and dialect (descriptive) --------------------------------------------------
    w = B.target_w[pids]
    mask_app = layout.mask(target=("appearance_distance",))
    mask_dia = layout.mask(target=("dialect_distance",))
    mask_in = layout.mask(inputs=features.MARKER_FEATURES)
    pop = P.population[pids]

    def describe(v):
        return {"mean": _r(np.mean(v)), "median": _r(np.median(v)), "sd": _r(np.std(v)),
                "pop_weighted_mean": _r(np.average(v, weights=np.maximum(pop, 1.0))),
                "positive": int((v > 0).sum()), "negative": int((v < 0).sum()), "zero": int((v == 0).sum())}

    markers = {
        "target_weight_appearance": describe(w[:, tcol["appearance_distance"]]),
        "target_weight_dialect": describe(w[:, tcol["dialect_distance"]]),
        "target_weight_win_chance_for_scale": describe(w[:, tcol["win_chance"]]),
        "genome_rms_target_appearance": _r(np.sqrt((g_now[:, mask_app] ** 2).mean())),
        "genome_rms_target_dialect": _r(np.sqrt((g_now[:, mask_dia] ** 2).mean())),
        "genome_rms_marker_inputs": _r(np.sqrt((g_now[:, mask_in] ** 2).mean())),
        "genome_rms_other_inputs_for_scale": _r(np.sqrt((g_now[:, ~(mask_in | mask_app | mask_dia)] ** 2).mean())),
        "culture_parameters": {name: describe(P.marker[pids, j]) for j, name in enumerate(MARKER_EFFECTS)},
    }
    wars = {
        "attacks_recorded": len(war_rec["chosen_app"]),
        "chosen_target_appearance_mean": _r(np.mean(war_rec["chosen_app"])) if war_rec["chosen_app"] else None,
        "attackers_candidates_appearance_mean": _r(np.mean(war_rec["own_mean_app"])) if war_rec["own_mean_app"] else None,
        "all_candidates_appearance_mean": _r(np.mean(war_rec["avail_app"])) if war_rec["avail_app"] else None,
        "chosen_target_dialect_mean": _r(np.mean(war_rec["chosen_dia"])) if war_rec["chosen_dia"] else None,
        "attackers_candidates_dialect_mean": _r(np.mean(war_rec["own_mean_dia"])) if war_rec["own_mean_dia"] else None,
        "all_candidates_dialect_mean": _r(np.mean(war_rec["avail_dia"])) if war_rec["avail_dia"] else None,
    }
    st = max(sec_rec["stayers"], 1)
    secessions = {
        "leavers": len(sec_rec["leaver_app"]),
        "leaver_appearance_mean": _r(np.mean(sec_rec["leaver_app"])) if sec_rec["leaver_app"] else None,
        "stayer_appearance_mean": _r(sec_rec["stayer_app"] / st),
        "leaver_dialect_mean": _r(np.mean(sec_rec["leaver_dia"])) if sec_rec["leaver_dia"] else None,
        "stayer_dialect_mean": _r(sec_rec["stayer_dia"] / st),
    }

    return {
        "condition": condition, "history_seed": history_seed, "years": m.year, "seconds": round(seconds, 1),
        "outcomes": {
            "population_end": round(float(m.population)),
            "population_peak": round(float(df["population"].max())),
            "polities_end": int(len(pids)),
            "polity_births": int(n - 5),
            "polity_count_every_500y": [int(v) for v in n_pol[::500]] + [int(n_pol[-1])],
            "polity_count_mean": round(float(n_pol.mean()), 1),
            "largest_polity": {"polity": big, "group": m.lineages.names[int(P.lineage[big])],
                               "peak_settlements": int(P.peak_settlements[big]),
                               "peak_population": round(float(P.peak_population[big])),
                               "lifespan": int(life[big]), "end_cause": END_CAUSES[int(P.end_cause[big])]},
            "largest_settlements_mean": round(float(df["largest_polity_settlements"].mean()), 1),
            "battles": int(df["battles"].sum()), "conquests": int(df["conquests"].sum()),
            "wars_active_mean": round(float(df["wars_active"].mean()), 2),
            "cohesion_mean": round(float(df["mean_cohesion"].mean()), 3),
            "lifespan_median": float(np.median(life[ended])) if ended.any() else None,
            "lineages_surviving": int(len(np.unique(P.lineage[pids]))),
            "per_millennium": per_millennium,
        },
        "brain_output_means_every_500y": brain_means,
        "snapshots": snaps,
        "survivors_vs_founders": survivors,
        "markers": markers,
        "war_targets": wars,
        "secessions_by_distance": secessions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--years", type=int, default=5000)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--snapshot", type=int, default=500, help="years between snapshots")
    parser.add_argument("--out", default=None, help="output file (default runs/round7_acceptance_w<world>.json)")
    args = parser.parse_args(argv)
    jobs = [(c, s) for s in args.seeds for c in args.conditions]
    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    workers = max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_job, args.world_seed, s, args.years, c, args.snapshot) for c, s in jobs]
        results = [f.result() for f in futures]
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round7_acceptance_w{args.world_seed}.json"
    out.write_text(json.dumps({"world_seed": args.world_seed, "years": args.years, "runs": results}, indent=1),
                   encoding="utf-8")
    for r in results:
        o = r["outcomes"]
        print(r["condition"], r["history_seed"], r["seconds"], "s | pop", o["population_end"], "births",
              o["polity_births"], "polities", o["polities_end"], "largest", o["largest_polity"]["peak_settlements"],
              "battles", o["battles"])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
