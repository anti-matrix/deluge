"""Multi-seed runs in parallel processes (Round 10, spec section 12).

    python scripts/batch.py --world-seed 1 --history-seeds 1 2 3 4 --years 1500 --batch-dir runs/batch_w1

Each history runs as its own process (``scripts/run.py``) into ``<batch-dir>/w<W>_h<H>/``,
its log in ``run.log`` there. At most ``--workers`` (default ``output.batch_workers``) run at
once, and fewer when memory is short: a run starts only while the available memory is at
least ``output.batch_run_memory_mb`` + ``output.batch_memory_reserve_mb`` (or nothing else is
running). Rerunning the same command skips complete runs and resumes interrupted ones from
their latest checkpoint. ``batch.json`` in the batch directory records each run's outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN = REPO_ROOT / "scripts" / "run.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run import is_complete  # noqa: E402

from deluge.config import DEFAULT_CONFIG_PATH, load_config, read_overrides  # noqa: E402

MEMORY_RECHECK_SECONDS = 30.0


def available_memory_mb() -> float | None:
    """Memory available to new processes, in MB (None if it cannot be read)."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**20
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / 2**20
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def worker_limit(requested: int, run_mb: float, reserve_mb: float, available_mb: float | None) -> int:
    """Parallel runs: at most ``requested`` and the cores, and only as many as fit in memory
    (at least one)."""
    limit = min(requested, os.cpu_count() or 1)
    if available_mb is not None:
        limit = min(limit, int((available_mb - reserve_mb) // run_mb))
    return max(1, limit)


class Scheduler:
    """Starts runs while memory allows; a run waits for another to finish when it does not."""

    def __init__(self, run_mb: float, reserve_mb: float, log) -> None:
        self.need = run_mb + reserve_mb
        self.running = 0
        self.cond = threading.Condition()
        self.log = log

    def start(self, name: str) -> None:
        with self.cond:
            while self.running > 0:
                free = available_memory_mb()
                if free is None or free >= self.need:
                    break
                self.log(f"{name}: waiting for memory ({free:.0f} MB available, {self.need:.0f} MB needed)")
                self.cond.wait(MEMORY_RECHECK_SECONDS)
            self.running += 1

    def done(self) -> None:
        with self.cond:
            self.running -= 1
            self.cond.notify_all()


def run_one(scheduler: Scheduler, out_dir: Path, args, history_seed: int, log) -> dict:
    name = out_dir.name
    if is_complete(out_dir):
        log(f"{name}: complete, skipped")
        return {"run": name, "history_seed": history_seed, "status": "complete (skipped)"}
    scheduler.start(name)
    try:
        resuming = (out_dir / "checkpoints").exists() and any((out_dir / "checkpoints").glob("*.pkl"))
        log(f"{name}: {'resuming' if resuming else 'starting'}")
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(RUN), "--config", args.config, "--world-seed", str(args.world_seed),
               "--history-seed", str(history_seed), "--out", str(out_dir)]
        if args.overrides:
            cmd += ["--overrides", args.overrides]
        if args.years is not None:
            cmd += ["--years", str(args.years)]
        t0 = time.perf_counter()
        with open(out_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(f"# {' '.join(cmd)}\n")
            f.flush()
            code = subprocess.run(cmd, cwd=REPO_ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        seconds = time.perf_counter() - t0
    finally:
        scheduler.done()
    status = "complete" if code == 0 and is_complete(out_dir) else f"failed (exit {code})"
    log(f"{name}: {status} after {seconds:.0f} s")
    return {"run": name, "history_seed": history_seed, "status": status, "seconds": round(seconds, 1),
            "resumed": resuming}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--overrides", default=None)
    p.add_argument("--world-seed", type=int, default=1)
    seeds = p.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--history-seeds", type=int, nargs="+")
    seeds.add_argument("--n-seeds", type=int, help="history seeds 1..N")
    p.add_argument("--years", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="default: output.batch_workers")
    p.add_argument("--batch-dir", required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config, read_overrides(args.overrides) if args.overrides else None)
    out = cfg.output
    history_seeds = args.history_seeds or list(range(1, args.n_seeds + 1))
    batch_dir = Path(args.batch_dir)
    if not batch_dir.is_absolute():
        batch_dir = REPO_ROOT / batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    free = available_memory_mb()
    workers = worker_limit(args.workers or out.batch_workers, out.batch_run_memory_mb,
                           out.batch_memory_reserve_mb, free)
    log(f"{len(history_seeds)} runs, {workers} workers ({free:.0f} MB available)" if free is not None
        else f"{len(history_seeds)} runs, {workers} workers")
    scheduler = Scheduler(out.batch_run_memory_mb, out.batch_memory_reserve_mb, log)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, scheduler, batch_dir / f"w{args.world_seed}_h{h}", args, h, log)
                   for h in history_seeds]
        results = [f.result() for f in futures]
    summary = {"world_seed": args.world_seed, "history_seeds": history_seeds, "years": args.years,
               "workers": workers, "seconds": round(time.perf_counter() - t0, 1), "runs": results}
    (batch_dir / "batch.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    failed = [r for r in results if not r["status"].startswith("complete")]
    log(f"done: {len(results) - len(failed)} complete, {len(failed)} failed -> {batch_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
