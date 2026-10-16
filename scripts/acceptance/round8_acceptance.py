"""Round 8 acceptance: memory and myth (spec sections 11, 13), run manually.

    python scripts/acceptance/round8_acceptance.py --world-seed 1 --seeds 1 2 3 --years 3000

Runs every history seed under three conditions, at most ``--workers`` processes at once:
normal, control (memory recorded and transmitted, every consequence weight zero) and absent
(memory off). Reported per run:

- Checks: every store within its capacity at the end of every year; every record's
  provenance resolves to logged events, and every copied record to a logged copy.
- Control vs absent: whether the model table and the event log are identical.
- Outcomes (normal vs control): polity lifespans, cohesion, mergers, battles, wars,
  polity count, population.
- Mergers: year, absorbed and absorbing polity, their founding groups.
- Descriptive, normal runs (no target): for each founding group at the end, which original
  events its living stores still descend from, by kind of event (an analysis-side reading
  of the raw log: the teachers' arrival, teaching, displays, gifts and departure; moves,
  joins and ends of bands whose cell the sea took, near a drowned land bridge or not;
  conquests; battles; famines; the year-0 foundings; splits; ...), how old the oldest is and
  how far the records were transformed; and whether the teachers stand as actors in records
  of events they took no part in.

The normal run of the first seed also writes its memory report (``analysis/myths.py``) to
runs/round8_myths_w<world>_h<seed>.md. Results go to runs/round8_acceptance_w<world>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = {
    "normal": {},
    "control": {"memory": {"control": True}},
    "absent": {"memory": {"enabled": False}},
}
TEACHER_VERBS = ("arrived", "taught", "displayed", "gave", "departed")


def _r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def fingerprint(model) -> str:
    df = model.datacollector.get_model_vars_dataframe()
    h = hashlib.sha256(np.ascontiguousarray(df.to_numpy(dtype=float)).tobytes())
    ev = model.events
    for col in (ev.year, ev.action, ev.place, ev.magnitude):
        h.update(np.asarray(col).tobytes())
    h.update(repr(ev.actors).encode())
    return h.hexdigest()


def classify_events(model) -> list[str]:
    """A kind per logged event, read from the raw log (analysis only)."""
    from deluge.data.events import ACTIONS, decode
    from deluge.world import geography

    ev = model.events
    sea = model.sea_level_series
    elev = model.elevation_flat
    cfg = model.config.world.land_bridges
    pulses = geography.pulse_windows(sea, cfg.pulse_rate_factor)
    land0 = elev >= sea[0]
    bridges = geography.land_bridges(model.geom, elev, sea, cfg.min_share_of_land * land0.sum(), pulses)
    crests = np.array([b.crest_cell for b in bridges], dtype=np.int64)
    loc: dict[int, int] = {}
    out = []

    def drowned(cell, year):
        return cell >= 0 and elev[cell] < sea[year]

    def near_bridge(cell):
        return len(crests) and int(model.geom.hex_distance(crests, cell).min()) <= 2

    for i in range(len(ev)):
        act, actors, year, place = ACTIONS[ev.action[i]], ev.actors[i], ev.year[i], ev.place[i]
        kinds = [decode(a) for a in actors]
        k = "other"
        if any(kd == "teachers" for kd, _ in kinds):
            k = f"teachers {act}"
        elif act in ("moved", "joined", "died") and kinds and kinds[0][0] == "settlement":
            sid = kinds[0][1]
            prev = loc.get(sid, -1) if act != "died" else place
            if drowned(prev, year):
                k = "band displaced by the sea" + (" at a land bridge" if near_bridge(prev) else "")
            else:
                k = {"moved": "band moved", "joined": "band joined another", "died": "settlement ended"}[act]
        elif act == "founded" and kinds and kinds[0][0] == "polity":
            k = "year-0 founding" if year == 0 else "polity split off"
        elif act == "founded":
            k = "settlement founded"
        elif act in ("took", "lost") or (act == "died" and len(actors) == 2):
            k = "conquest"
        elif act == "died":
            k = "polity ended"
        elif act in ("fought", "won"):
            k = "battle"
        elif act == "starved":
            k = "famine"
        elif act == "built":
            k = "monument"
        elif act in ("learned", "forgot"):
            k = f"technique {act}"
        elif act == "joined":
            k = "union"
        else:
            k = act
        out.append(k)
        if act in ("founded", "moved") and kinds and kinds[0][0] == "settlement":
            loc[kinds[0][1]] = place
        elif act == "founded" and len(kinds) > 1 and kinds[1][0] == "settlement":
            loc[kinds[1][1]] = place
    return out


def describe_memory(model) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from analysis.myths import History, substituted_actors
    from deluge.data.events import entity
    from deluge.systems.memory import OPS

    st, P = model.memory, model.polities
    kinds = classify_events(model)
    hist = History(model)
    teachers = entity("teachers", 0)
    year = model.year
    groups = {}
    longest = defaultdict(lambda: {"records": 0, "oldest_year": None})
    for g, gname in enumerate(model.lineages.names):
        pids = [int(p) for p in P.alive_ids() if P.lineage[p] == g]
        recs = [st.records[m] for p in pids for m in st.stores.get(p, ())]
        by_kind = defaultdict(lambda: {"records": 0, "events": set(), "oldest_year": None})
        ops = defaultdict(int)
        n_subst_teachers = 0
        teacher_events = set()
        for r in recs:
            for i in r.provenance:
                k = kinds[i]
                d = by_kind[k]
                d["events"].add(i)
                y = model.events.year[i]
                d["oldest_year"] = y if d["oldest_year"] is None else min(d["oldest_year"], y)
                if k.startswith("teachers"):
                    teacher_events.add(i)
            for k in {kinds[i] for i in r.provenance}:
                by_kind[k]["records"] += 1
                L = longest[k]
                L["records"] += 1
                y0 = min(model.events.year[i] for i in r.provenance if kinds[i] == k)
                L["oldest_year"] = y0 if L["oldest_year"] is None else min(L["oldest_year"], y0)
            for row in hist.transformations(r.mem_id):
                ops[OPS[row[1]]] += 1
            if teachers in r.actors and teachers in substituted_actors(model, r):
                n_subst_teachers += 1
        groups[gname] = {
            "living_polities": len(pids), "records": len(recs),
            "surviving_by_kind": {k: {"records": v["records"], "events": len(v["events"]),
                                      "oldest_year": v["oldest_year"], "oldest_age": year - v["oldest_year"]}
                                  for k, v in sorted(by_kind.items(), key=lambda kv: kv[1]["oldest_year"])},
            "teacher_events_surviving": sorted({kinds[i] for i in teacher_events}),
            "records_with_teachers_substituted_in": n_subst_teachers,
            "records_with_teachers_as_actor": sum(teachers in r.actors for r in recs),
            "transformations_per_record": {k: _r(v / max(len(recs), 1), 2) for k, v in sorted(ops.items())},
            "detail": {
                "year_exact_share": _r(np.mean([r.year_span == 1 for r in recs])) if recs else None,
                "place_exact_share": _r(np.mean([r.place_level == 0 for r in recs])) if recs else None,
                "fused_share": _r(np.mean([len(r.provenance) > 1 for r in recs])) if recs else None,
                "provenance_events_mean": _r(np.mean([len(r.provenance) for r in recs]), 2) if recs else None,
            },
        }
    # The teachers' events in the whole log, and whether any record anywhere still descends from them.
    t_events = [i for i, k in enumerate(kinds) if k.startswith("teachers")]
    alive_prov = {i for r in st.records.values() for i in r.provenance}
    in_living = {i for p in P.alive_ids().tolist() for m in st.stores.get(p, ()) for i in st.records[m].provenance}
    return {
        "by_group": groups,
        "longest_lasting_kinds": {k: {**v, "oldest_age": year - v["oldest_year"]}
                                  for k, v in sorted(longest.items(), key=lambda kv: kv[1]["oldest_year"])},
        "teacher_events": {"logged": len(t_events),
                           "in_any_live_record": sum(i in alive_prov for i in t_events),
                           "in_living_stores": sum(i in in_living for i in t_events),
                           "kinds_logged": sorted({kinds[i] for i in t_events})},
        "teachers_substituted_in_total": sum(1 for r in st.records.values()
                                             if teachers in r.actors and teachers in substituted_actors(model, r)),
    }


def run_job(world_seed: int, history_seed: int, years: int, condition: str, report: bool) -> dict:
    sys.path.insert(0, str(REPO_ROOT))
    from deluge.config import load_config
    from deluge.data.events import ACTIONS, decode
    from deluge.model import DelugeModel
    from deluge.systems import memory

    cfg = load_config(overrides=CONDITIONS[condition])
    t0 = time.perf_counter()
    m = DelugeModel(config=cfg, world_seed=world_seed, history_seed=history_seed, years=years)
    over_capacity = 0
    worst = -10 ** 9
    orig_run = memory.run

    def checked_run(model):
        nonlocal over_capacity, worst
        orig_run(model)
        st = model.memory
        if not st.enabled:
            return
        pids = model.polities.alive_ids()
        caps = dict(zip(pids.tolist(), memory.capacity(model, pids).tolist()))
        for pid, store in st.stores.items():
            d = len(store) - caps.get(pid, -1)
            worst = max(worst, d)
            over_capacity += d > 0

    import deluge.model as model_module
    original = model_module.PHASES  # worker processes run several jobs: restore after each
    phases = list(original)
    phases[phases.index(orig_run)] = checked_run
    model_module.PHASES = tuple(phases)
    try:
        m.run()
    finally:
        model_module.PHASES = original
    seconds = time.perf_counter() - t0

    df = m.datacollector.get_model_vars_dataframe()
    P = m.polities
    n = P.n
    life = np.where(P.ended[:n] >= 0, P.ended[:n], m.year) - P.founded[:n]
    ended = P.ended[:n] >= 0
    pids = P.alive_ids()
    ev = m.events
    mergers = []
    for i in range(len(ev)):
        if ACTIONS[ev.action[i]] == "joined" and decode(ev.actors[i][0])[0] == "polity":
            a, b = decode(ev.actors[i][0])[1], decode(ev.actors[i][1])[1]
            mergers.append({"year": ev.year[i], "absorbed": a, "into": b,
                            "groups": [m.lineages.names[int(P.lineage[a])], m.lineages.names[int(P.lineage[b])]],
                            "absorbed_parent": int(P.parent[a]), "into_parent": int(P.parent[b])})
    out = {
        "condition": condition, "history_seed": history_seed, "years": m.year, "seconds": round(seconds, 1),
        "fingerprint": fingerprint(m),
        "outcomes": {
            "population_end": round(float(m.population)),
            "polity_births": int(n - 5),
            "polities_end": int(len(pids)),
            "polity_count_mean": round(float(df["n_polities"].mean()), 1),
            "polity_count_every_500y": [int(v) for v in df["n_polities"].to_numpy()[::500]],
            "lifespan_median": float(np.median(life[ended])) if ended.any() else None,
            "lifespan_mean": _r(life[ended].mean(), 1) if ended.any() else None,
            "lifespan_p90": float(np.percentile(life[ended], 90)) if ended.any() else None,
            "cohesion_mean": _r(df["mean_cohesion"].mean()),
            "cohesion_mean_last_1000y": _r(df["mean_cohesion"].to_numpy()[-1000:].mean()),
            "mergers": int(df["mergers"].sum()),
            "battles": int(df["battles"].sum()),
            "conquests": int(df["conquests"].sum()),
            "secessions": int(df["secessions"].sum()),
            "fragmentations": int(df["fragmentations"].sum()),
            "wars_active_mean": _r(df["wars_active"].mean(), 2),
            "lineages_surviving": int(len(np.unique(P.lineage[pids]))),
        },
        "mergers": mergers,
    }
    st = m.memory
    if st.enabled:
        bad_prov = sum(1 for r in st.records.values() if not r.provenance
                       or any(not (0 <= i < len(ev)) for i in r.provenance))
        copied = {row[3] for row in st.ops if memory.OPS[row[1]] in ("copied", "borrowed")}
        unlogged_copies = sum(1 for r in st.records.values() if r.parent >= 0 and r.mem_id not in copied)
        ops = defaultdict(int)
        for row in st.ops:
            ops[memory.OPS[row[1]]] += 1
        out["memory"] = {
            "years_with_a_store_over_capacity": over_capacity, "max_store_minus_capacity": worst,
            "records_live": len(st.records), "records_written": st.created, "symbols": len(st.symbol_key),
            "records_with_unresolved_provenance": bad_prov, "copied_records_without_logged_copy": unlogged_copies,
            "store_size_mean": _r(np.mean([len(s) for p, s in st.stores.items() if P.ended[p] < 0]), 1),
            "ops": dict(ops),
        }
    if condition == "normal":
        out["descriptive"] = describe_memory(m)
        if report:
            from analysis.myths import report as render
            path = REPO_ROOT / "runs" / f"round8_myths_w{world_seed}_h{history_seed}.md"
            path.write_text(render(m, title=f"Memory report, world seed {world_seed}, history seed {history_seed}, "
                                           f"year {m.year}"), encoding="utf-8")
            out["report"] = str(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world-seed", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--years", type=int, default=3000)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    (REPO_ROOT / "runs").mkdir(exist_ok=True)
    jobs = [(c, s) for s in args.seeds for c in args.conditions]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_job, args.world_seed, s, args.years, c, s == args.seeds[0]) for c, s in jobs]
        results = [f.result() for f in futures]
    for s in args.seeds:
        fp = {r["condition"]: r["fingerprint"] for r in results if r["history_seed"] == s}
        if "control" in fp and "absent" in fp:
            print(f"seed {s}: control == absent: {fp['control'] == fp['absent']}")
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / f"round8_acceptance_w{args.world_seed}.json"
    out.write_text(json.dumps({"world_seed": args.world_seed, "years": args.years, "runs": results}, indent=1,
                              default=int), encoding="utf-8")
    for r in results:
        o = r["outcomes"]
        print(r["condition"], r["history_seed"], r["seconds"], "s | polities", o["polities_end"], "births",
              o["polity_births"], "mergers", o["mergers"], "battles", o["battles"], "cohesion", o["cohesion_mean"],
              "lifespan", o["lifespan_median"], "| memory", r.get("memory", {}).get("years_with_a_store_over_capacity"),
              r.get("memory", {}).get("records_with_unresolved_provenance"))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
