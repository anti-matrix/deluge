"""Headless single run, resumable from its checkpoints.

    python scripts/run.py --config configs/default.yaml --world-seed 1 --history-seed 42 [--out DIR]

Writes into the run directory (spec section 12):

- ``model.parquet`` every year; ``lineages.parquet`` and ``polities.parquet`` at the
  intervals in ``output`` (``deluge/data/collectors.py``); ``events.parquet`` (the raw event
  log) with ``entities.parquet`` (kind and label of every entity id in it);
  ``polity_tree.parquet``; ``brains/brains.parquet`` (every brain born: polity, parent, group,
  year, weights) with ``brains/layout.json``; ``techniques.parquet`` (first year each
  technique was held anywhere, invented, and held per lineage: the milestone report);
  ``founders.parquet`` (each founding lineage's start and contact with the teachers) and
  ``settlements.parquet`` (every settlement's lineage, parent, polity, cell and years);
  ``ceilings.parquet`` (every year: land use and world population at the density ceilings
  of ``configs/bounds.yaml``, Round 12); Round 14: ``settlement_samples.parquet`` (each
  settlement's people and capacity), ``frontier.parquet`` (each lineage's farthest settlement
  over land from its year-0 cell), ``polity_scale.parquet`` (the largest polity) and
  ``campaigns.parquet`` (every attack, with its dead and the people present);
- ``memory_snapshots/year_NNNNN.parquet`` every ``output.memory_snapshot_every_years`` and at
  the end: every live memory record with its provenance (``memory.parquet``: the same at the
  end); ``memory_ops.parquet`` (the transformation log) at the end;
- ``checkpoints/year_NNNNN.pkl`` every ``output.checkpoint_every_years`` (the latest
  ``output.checkpoints_kept``): the pickled model and recorder;
- ``meta.json`` at the end: seeds, config hash, git commit, run time, library versions,
  checkpoints written and resumed from, the chronicle's length; ``"complete": true``.

If the run directory holds checkpoints and no complete ``meta.json``, the run resumes from
the latest checkpoint (``--fresh`` starts over). The history is the same either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from deluge.config import DEFAULT_CONFIG_PATH, load_config, read_overrides  # noqa: E402

VERSIONED = ("mesa", "numpy", "numba", "scipy", "pandas", "pyarrow", "opensimplex", "pyyaml")
CHECKPOINT_STEM = "year_"


def git_commit() -> str | None:
    """HEAD's commit: from git if it is on the PATH, else read from .git directly."""
    git = shutil.which("git")
    if git is not None:
        try:
            out = subprocess.run([git, "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                 check=True)
            return out.stdout.strip() or None
        except (subprocess.CalledProcessError, OSError):
            pass
    head = REPO_ROOT / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if not ref.startswith("ref: "):
            return ref
        name = ref[5:]
        loose = REPO_ROOT / ".git" / name
        if loose.exists():
            return loose.read_text(encoding="utf-8").strip()
        packed = REPO_ROOT / ".git" / "packed-refs"
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line.endswith(" " + name):
                return line.split()[0]
    except OSError:
        pass
    return None


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in VERSIONED:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def rng_fingerprint(model) -> str:
    blob = json.dumps(model.streams.state(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def entity_table(model) -> pd.DataFrame:
    """Kind and label of every entity id that appears in the event log."""
    from deluge.data.chronicle import Namer
    from deluge.data.events import decode

    ids = sorted({a for actors in model.events.actors for a in actors})
    name = Namer(model)
    rows = [(eid, *decode(eid), name(eid)) for eid in ids]
    return pd.DataFrame(rows, columns=["entity", "kind", "local_id", "label"])


def checkpoints(out_dir: Path) -> list[Path]:
    """The run's checkpoints, oldest first."""
    return sorted((out_dir / "checkpoints").glob(f"{CHECKPOINT_STEM}*.pkl"))


def is_complete(out_dir: Path) -> bool:
    meta = out_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        return bool(json.loads(meta.read_text(encoding="utf-8")).get("complete"))
    except (OSError, json.JSONDecodeError):
        return False


def save_checkpoint(out_dir: Path, state: dict, kept: int) -> Path:
    """Pickle the run state atomically (write, then rename); keep the latest ``kept`` (0: all)."""
    folder = out_dir / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{CHECKPOINT_STEM}{state['model'].year:05d}.pkl"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    if kept > 0:
        for old in checkpoints(out_dir)[:-kept]:
            old.unlink()
    return path


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_memory_snapshot(model, out_dir: Path) -> None:
    from deluge.systems import memory

    if not model.memory.enabled:
        return
    folder = out_dir / "memory_snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    df = memory.records_frame(model)
    df.insert(0, "snapshot_year", model.year)  # ``year`` is the record's own
    df.to_parquet(folder / f"year_{model.year:05d}.parquet", index=False)


def founders_frame(model) -> pd.DataFrame:
    """Per founding lineage: where it started (cell, row, year-0 forage capacity there), its
    first farming year, and its contact with the teachers (visit order, years, techniques
    taught, artifacts received)."""
    rows = []
    visits = model.teachers_state.visits
    for f in model.lineages.founders:
        mine = [v for v in visits if v["lineage"] == f.lineage]
        x, y = model.geom.xy(f.cell)
        rows.append({
            "lineage": f.lineage, "group": f.name, "cell": f.cell, "x": int(x), "y": int(y),
            "start_population": f.population, "start_capacity": float(model.founder_capacity[f.cell]),
            "first_farming_year": int(model.first_farming_year[f.lineage]),
            "teachers_visit_order": (model.teachers_state.visited.index(f.lineage)
                                     if f.lineage in model.teachers_state.visited else -1),
            "teachers_first_year": min((v["start"] for v in mine), default=-1),
            "teachers_years": sum((v["end"] if v["end"] is not None else model.year) - v["start"] for v in mine),
            "techniques_taught": sum(len(v["taught"]) for v in mine),
            "artifacts": sum(len(v["artifacts"]) for v in mine),
            **{f"culture_{k}": v for k, v in f.culture.items()},
        })
    return pd.DataFrame(rows)


def settlements_frame(model) -> pd.DataFrame:
    """Every settlement ever founded: lineage, parent, polity (its last), cell, years."""
    t = model.settlements
    n = t.n
    return pd.DataFrame({"settlement": range(n), "lineage": t.lineage[:n], "parent": t.parent[:n],
                         "polity": t.polity[:n], "cell": t.cell[:n], "founded": t.founded[:n],
                         "ended": t.ended[:n], "population": t.population[:n]})


def alliances_frame(model) -> pd.DataFrame:
    """Every alliance (Round 14): the two polities (low id first), the years it began and ended (-1:
    living at the end), how it ended (rescored, betrayed, ended: a partner ended) and by whom."""
    rows = [(a, b, began, end, how, by) for a, b, began, end, how, by in model.alliance_history]
    rows += [(a, b, began, -1, "", -1) for (a, b), began in sorted(model.alliances.items())]
    return pd.DataFrame(rows, columns=["a", "b", "began", "ended", "how", "by"])


def heads_frame(model) -> pd.DataFrame:
    """Each living polity's per-neighbour head weights this year (Round 14): target_<feature> and
    alliance_<feature> (the weights the brain gives each feature of a neighbour)."""
    from deluge.brains.features import ALLIANCE_FEATURES, TARGET_FEATURES

    P, B = model.polities, model.brains
    pids = P.alive_ids()
    df = pd.DataFrame({"polity": pids, "lineage": P.lineage[pids]})
    for j, f in enumerate(TARGET_FEATURES):
        df[f"target_{f}"] = B.target_w[pids, j]
    for j, f in enumerate(ALLIANCE_FEATURES):
        df[f"alliance_{f}"] = B.alliance_w[pids, j]
    return df


def prior_frame(model) -> pd.DataFrame:
    """Per polity pair in contact at the end (Round 14), both ways: the appearance similarity of
    their capitals, the shift the appearance prior adds to the alliance score's sum and takes from
    the target score's sum (0 when it is off; ``would_shift``: its size were it on), and the brain's
    own appearance terms (head weight x perceived appearance difference) for alliance and target."""
    from deluge.brains.features import ALLIANCE_FEATURES, TARGET_FEATURES
    from deluge.systems import identity, politics

    P, B = model.polities, model.brains
    pairs = politics.contact_pairs(model)
    if len(pairs) == 0:
        return pd.DataFrame(columns=["a", "b", "similarity", "shift", "would_shift", "alliance_term", "target_term"])
    a = np.r_[pairs[:, 0], pairs[:, 1]]
    b = np.r_[pairs[:, 1], pairs[:, 0]]
    ca, cb = P.capital[a], P.capital[b]
    app, _ = identity.pair_distances(model, ca, cb)
    sim, _ = identity.pair_similarities(model, ca, cb)
    kp = model.config.memory.kin_prior
    return pd.DataFrame({"a": a, "b": b, "similarity": sim, "shift": identity.kin_prior(model, ca, cb),
                         "would_shift": kp.weight * sim,
                         "alliance_term": B.alliance_w[a, ALLIANCE_FEATURES.index("appearance_distance")] * app,
                         "target_term": B.target_w[a, TARGET_FEATURES.index("appearance_distance")] * app})


def chronicle_entries(model) -> int:
    """Length of the browser chronicle of the whole run (entries told one by one plus the
    generation summaries)."""
    from deluge.data.chronicle import Chronicle

    view = Chronicle(model)
    view.sync()
    return len(view)


def write_tables(model, recorder, out_dir: Path) -> None:
    from deluge.brains.network import archive_frame
    from deluge.systems import memory

    model.datacollector.get_model_vars_dataframe().to_parquet(out_dir / "model.parquet", index=True)
    founders_frame(model).to_parquet(out_dir / "founders.parquet", index=False)
    settlements_frame(model).to_parquet(out_dir / "settlements.parquet", index=False)
    recorder.lineages_frame().to_parquet(out_dir / "lineages.parquet", index=False)
    recorder.polities_frame().to_parquet(out_dir / "polities.parquet", index=False)
    recorder.first_held_frame(model).to_parquet(out_dir / "techniques.parquet", index=False)
    recorder.ceilings_frame().to_parquet(out_dir / "ceilings.parquet", index=False)
    recorder.samples_frame().to_parquet(out_dir / "settlement_samples.parquet", index=False)
    recorder.frontier_frame().to_parquet(out_dir / "frontier.parquet", index=False)
    recorder.scale_frame().to_parquet(out_dir / "polity_scale.parquet", index=False)
    recorder.diplomacy_frame().to_parquet(out_dir / "diplomacy.parquet", index=False)
    alliances_frame(model).to_parquet(out_dir / "alliances.parquet", index=False)
    heads_frame(model).to_parquet(out_dir / "brains_final.parquet", index=False)
    prior_frame(model).to_parquet(out_dir / "prior_shift.parquet", index=False)
    if getattr(model, "_campaigns", None) is not None:
        model._campaigns.frame().to_parquet(out_dir / "campaigns.parquet", index=False)
    model.events.write(out_dir / "events.parquet")
    entity_table(model).to_parquet(out_dir / "entities.parquet", index=False)
    model.lineages.polity_tree(model).to_parquet(out_dir / "polity_tree.parquet", index=False)
    (out_dir / "brains").mkdir(exist_ok=True)
    archive_frame(model).to_parquet(out_dir / "brains" / "brains.parquet", index=False)
    (out_dir / "brains" / "layout.json").write_text(json.dumps(model.brain_layout.to_dict(), indent=1),
                                                    encoding="utf-8")
    if model.memory.enabled:
        memory.records_frame(model).to_parquet(out_dir / "memory.parquet", index=False)
        memory.ops_frame(model).astype({"op": str}).to_parquet(out_dir / "memory_ops.parquet", index=False)


def run(cfg, out_dir: Path, config_path: str, overrides_path: str | None, fresh: bool = False,
        log=print) -> dict:
    """Run (or resume) the history ``cfg`` describes into ``out_dir``; returns the metadata."""
    from deluge.data.collectors import RunRecorder
    from deluge.model import DelugeModel

    out = cfg.output
    out_dir.mkdir(parents=True, exist_ok=True)
    found = [] if fresh else checkpoints(out_dir)
    if found:
        state = load_checkpoint(found[-1])
        model = state["model"]
        if state["config_hash"] != cfg.hash():
            # Only sections the simulation never reads (outputs, UI, reports) may differ, e.g. how often
            # checkpoints are written; the history is the same.
            from deluge.data.state import simulation_config_hash

            if simulation_config_hash(model.config) != simulation_config_hash(cfg):
                raise ValueError(f"{found[-1]} was written with another config")
            log(f"resuming with other output settings (the simulation's config is the same)")
        state["resumed_from"].append(model.year)
        log(f"resumed from {found[-1].name} (year {model.year})")
    else:
        started = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        model = DelugeModel(config=cfg)
        state = {"model": model, "recorder": RunRecorder(model), "config_hash": cfg.hash(),
                 "started_utc": started.isoformat(), "setup_seconds": time.perf_counter() - t0,
                 "step_seconds": 0.0, "rng_fingerprint": rng_fingerprint(model), "resumed_from": [],
                 "checkpoints_written": []}
    recorder = state["recorder"]
    t1 = time.perf_counter()
    while model.running:
        model.step()
        recorder.observe(model)
        if model.year % out.memory_snapshot_every_years == 0 or not model.running:
            write_memory_snapshot(model, out_dir)
        if model.running and model.year % out.checkpoint_every_years == 0:
            state["step_seconds"] += time.perf_counter() - t1
            state["checkpoints_written"].append(model.year)
            path = save_checkpoint(out_dir, state, out.checkpoints_kept)
            log(f"year {model.year}: checkpoint {path.name}, {state['step_seconds']:.0f} s stepping so far")
            t1 = time.perf_counter()
    state["step_seconds"] += time.perf_counter() - t1
    write_tables(model, recorder, out_dir)
    meta = {
        "run_id": out_dir.name,
        "complete": True,
        "world_seed": cfg.run.world_seed,
        "history_seed": cfg.run.history_seed,
        "years": cfg.run.years,
        "final_year": model.year,
        "config_path": str(Path(config_path).resolve()),
        "overrides_path": str(Path(overrides_path).resolve()) if overrides_path else None,
        "config_hash": state["config_hash"],
        "rng_fingerprint": state["rng_fingerprint"],
        "git_commit": git_commit(),
        "started_utc": state["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "setup_seconds": state["setup_seconds"],  # world generation and model construction
        "step_seconds": state["step_seconds"],  # stepping, summed over resumed sessions; checkpoint writes excluded
        "runtime_seconds": state["setup_seconds"] + state["step_seconds"],
        "resumed_from_years": state["resumed_from"],
        "checkpoints_written": state["checkpoints_written"],
        "chronicle_entries": chronicle_entries(model),
        "platform": platform.platform(),
        "versions": versions(),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--overrides", default=None,
                        help="YAML file of nested overrides applied on top of --config "
                             "(e.g. configs/worlds/small_test.yaml)")
    parser.add_argument("--world-seed", type=int, default=None, help="overrides run.world_seed (the map)")
    parser.add_argument("--history-seed", type=int, default=None, help="overrides run.history_seed")
    parser.add_argument("--years", type=int, default=None, help="overrides run.years")
    parser.add_argument("--out", default=None, help="run directory (default: <runs_dir>/<run_id>)")
    parser.add_argument("--fresh", action="store_true", help="ignore checkpoints in --out and start over")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in (("world_seed", args.world_seed), ("history_seed", args.history_seed),
                                   ("years", args.years)) if v is not None}
    cfg = load_config(args.config, read_overrides(args.overrides) if args.overrides else None)
    if overrides:
        cfg = cfg.with_overrides({"run": overrides})
    if args.out:
        out_dir = Path(args.out)
    else:
        run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_w{cfg.run.world_seed}_h{cfg.run.history_seed}"
        out_dir = REPO_ROOT / cfg.output.runs_dir / run_id
    if is_complete(out_dir) and not args.fresh:
        print(f"{out_dir} is complete")
        return 0
    meta = run(cfg, out_dir, args.config, args.overrides, fresh=args.fresh, log=lambda s: print(s, flush=True))
    print(f"{meta['final_year']} years in {meta['step_seconds']:.3f}s (setup {meta['setup_seconds']:.3f}s) -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
