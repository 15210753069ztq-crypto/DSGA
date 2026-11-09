"""DSGA-DV under the Proceed-aligned protocol (20/5/75, L=60, H in {1, 24}, block-wise release).

Stages:
    search      validation-stream grid search of the DSGA-DV hyperparameters (--warmup-validation)
    dsga-first  search, then each final run as soon as its search is complete
    select      per (dataset, H) configuration with the lowest validation prequential MSE
    final       DSGA-DV (view switching, S) on the full stream with the selected configuration

GPU rule: a GPU receives a new job only if every compute process on it was started by
this scheduler. A GPU on which another process has been seen is excluded for the rest of the run.

    python run_pa_protocol.py search --gpus 0 1 2 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "dataset"
OUT = HERE / "outputs" / "pa2"
DATASETS = {
    "ETTh2": "ETT-small/ETTh2.csv",
    "Weather": "weather/weather.csv",
    "Electricity": "electricity/electricity.csv",
    "Exchange": "exchange_rate/exchange_rate.csv",
    "PEMS08": "PEMS/PEMS08.npz",
    "Traffic": "traffic/traffic.csv",
    "SCA": "sca_stream_v2_csv/sca_stream_v2_seed00.csv",
    "Air": "air_stream/air_stream.csv",
    "Energy": "energy_stream/energy_stream.csv",
}
HORIZONS = (1, 24)
# Larger streams use three seeds for selection only; formal runs always use five.
SEARCH_SEEDS = {"ETTh2": 5, "Exchange": 5, "SCA": 5}
GRID = {
    "lr_scale": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
    "adapt_epochs": (5, 10, 20, 40),
    "mod_rank": (2, 4),
    "minimum_effect_z": (2.0, 2.5, 3.0),
}
PROTOCOL = ["--split", "proceed", "--lag", "60", "--online-blocks", "8", "--base-epochs", "20",
            "--device", "cuda:0"]
MAIN = [sys.executable, str(HERE / "run_pems08_prequential_multihorizon.py")]
SIZE = {"Traffic": 0, "Electricity": 1, "Weather": 3, "PEMS08": 4, "Air": 5,
        "Energy": 6, "ETTh2": 7, "SCA": 8, "Exchange": 9}


def cost(name: str) -> tuple[int, int]:
    """(GPU slots, required free MiB) of a job."""
    dataset = name.split("_")[1] if name.startswith(("hv_", "final_")) else name.split("_")[0]
    big, mid = dataset in ("Traffic", "Air"), dataset in ("Electricity", "PEMS08", "Energy")
    if name.startswith(("hv_", "final_")):
        return (2, 6000) if big else ((1, 4000) if mid else (1, 2500))
    return (4, 12000) if big else ((2, 6000) if mid else (1, 3000))


def tag(c: dict) -> str:
    return f"lr{c['lr_scale']:g}_ep{c['adapt_epochs']}_r{c['mod_rank']}_z{c['minimum_effect_z']:g}"


def configs() -> list[dict]:
    return [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]


def data_args(dataset: str, horizon: int) -> list[str]:
    return ["--data", str(DATA / DATASETS[dataset]), "--dataset-name", dataset,
            "--horizon", str(horizon)]


# Gradient accumulation for streams whose full training batch does not fit in 16 GB.
MICRO = {"Traffic": 32, "Air": 32, "Electricity": 64}


def micro_args(dataset: str) -> list[str]:
    return ["--micro-batch", str(MICRO[dataset])] if dataset in MICRO else []


def dv_args(c: dict) -> list[str]:
    return ["--dual-view", "ftl", "--weight-mod", "--null-windows", "128", "--alpha", "0.025",
            "--graph-top-k", "8", "--lr-scale", str(c["lr_scale"]),
            "--adapt-epochs", str(int(c["adapt_epochs"])), "--mod-rank", str(int(c["mod_rank"])),
            "--minimum-effect-z", str(c["minimum_effect_z"])]


def available(datasets) -> list[str]:
    keep = [d for d in datasets if (DATA / DATASETS[d]).exists()]
    for d in sorted(set(datasets) - set(keep)):
        print(f"skip {d}: {DATA / DATASETS[d]} not found", flush=True)
    return keep


def search_jobs(datasets):
    """(name, command, output, priority); lower priority runs first."""
    jobs = []
    for d, h in itertools.product(datasets, HORIZONS):
        for c in configs():
            out = OUT / "search" / f"{d}_H{h}" / tag(c)
            jobs.append((f"hv_{d}_H{h}_{tag(c)}", MAIN + data_args(d, h) + PROTOCOL + [
                "--seeds", str(SEARCH_SEEDS.get(d, 3)), "--methods", "DSGA",
                "--warmup-validation", *dv_args(c), "--output", str(out)], out, 40 + SIZE[d]))
    return jobs


def gpu_processes() -> dict[int, set[int]]:
    index = {}
    for line in subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout.splitlines():
        i, uuid = (x.strip() for x in line.split(","))
        index[uuid] = int(i)
    procs: dict[int, set[int]] = {i: set() for i in index.values()}
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "," in line:
            uuid, pid = (x.strip() for x in line.split(","))
            if uuid in index and pid.isdigit():
                procs[index[uuid]].add(int(pid))
    return procs


def free_mib(gpu: int) -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used",
                          "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True, check=True).stdout
    total, used = (int(v) for v in out.strip().split(","))
    return total - used


def is_ours(pid: int) -> bool:
    """Processes started by any run of this scheduler (they write under OUT)."""
    try:
        return str(OUT) in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False


def orphan_outputs() -> set[str]:
    """Output directories of jobs still running from an earlier scheduler process."""
    outputs = set()
    for proc in Path("/proc").iterdir():
        if proc.name.isdigit() and is_ours(int(proc.name)):
            args = (proc / "cmdline").read_bytes().split(b"\0")
            if b"--output" in args:
                outputs.add(args[args.index(b"--output") + 1].decode())
    return outputs


def schedule(jobs, gpus, per_gpu, threads, retries=3, refresh=None) -> None:
    """Large jobs reserve a GPU (no new small jobs there) until enough memory is free.
    A GPU with a foreign process is skipped; it becomes eligible again only after it
    has been free of foreign processes for ten minutes."""
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    busy = orphan_outputs()
    queue = sorted((j for j in jobs if not (j[2] / "method_summary.csv").exists()
                    and str(j[2]) not in busy), key=lambda j: j[3])
    attempts = {j[0]: 0 for j in queue}
    running: dict[str, tuple] = {}
    foreign_seen: dict[int, float] = {}
    started_at: dict[int, float] = {}
    settle: dict[int, float] = {}
    reserved: dict[int, str] = {}
    print(f"{len(queue)} jobs to run, {len(busy)} still running from an earlier scheduler, "
          f"{len(jobs) - len(queue) - len(busy)} finished", flush=True)
    while queue or running or refresh is not None:
        if refresh is not None:
            known = {j[0] for j in queue} | set(running) | set(attempts)
            new = [j for j in refresh() if j[0] not in known]
            for job in new:
                attempts[job[0]] = 0
                print(f"queued {job[0]}", flush=True)
            if new:
                queue = sorted(queue + new, key=lambda j: j[3])
            elif not queue and not running:
                break
        for name, (proc, gpu, job) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            if proc.returncode == 0:
                print(f"done {name}", flush=True)
            elif attempts[name] < retries:
                print(f"retry {name} (exit {proc.returncode})", flush=True)
                queue.insert(0, job)
            else:
                print(f"FAILED {name}", flush=True)
        procs = gpu_processes()
        now = time.time()
        eligible = []
        for gpu in gpus:
            mine = {proc.pid for proc, _, _ in running.values()}
            if {p for p in procs.get(gpu, set()) if p not in mine and not is_ours(p)}:
                if now - foreign_seen.get(gpu, 0.0) > 600:
                    print(f"GPU{gpu} has foreign processes; skipped", flush=True)
                foreign_seen[gpu] = now
                reserved.pop(gpu, None)
            elif now - foreign_seen.get(gpu, -1e9) > 600:
                eligible.append(gpu)
        queue = [j for j in queue if not (j[2] / "method_summary.csv").exists()]
        names = {j[0] for j in queue}
        reserved = {g: n for g, n in reserved.items() if n in names and g in eligible}
        # The highest-priority job that cannot be placed anywhere reserves the least-loaded GPU.
        if queue and queue[0][0] not in reserved.values():
            head = queue[0]
            fits = [g for g in eligible if g not in reserved
                    and sum(cost(n)[0] for n, (_, gg, _) in running.items() if gg == g)
                    + cost(head[0])[0] <= per_gpu and free_mib(g) >= cost(head[0])[1]]
            if not fits and eligible:
                free_gpus = [g for g in eligible if g not in reserved] or eligible
                target = max(free_gpus, key=free_mib)
                reserved[target] = head[0]
                print(f"GPU{target} reserved for {head[0]}", flush=True)
        for gpu in eligible:
            if not queue or now - started_at.get(gpu, 0.0) < settle.get(gpu, 0.0):
                continue
            load = sum(cost(n)[0] for n, (_, g, _) in running.items() if g == gpu)
            free = free_mib(gpu)
            if gpu in reserved:
                candidates = [k for k, j in enumerate(queue) if j[0] == reserved[gpu]]
            else:
                candidates = [k for k, j in enumerate(queue) if j[0] not in reserved.values()]
            pick = next((k for k in candidates if load + cost(queue[k][0])[0] <= per_gpu
                         and free >= cost(queue[k][0])[1]), None)
            if pick is None:
                continue
            job = queue.pop(pick)
            reserved.pop(gpu, None)
            started_at[gpu] = now
            settle[gpu] = 45.0 if cost(job[0])[0] >= 2 else 5.0
            name, cmd, out, _ = job
            attempts[name] = attempts.get(name, 0) + 1
            out.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=str(threads),
                       MKL_NUM_THREADS=str(threads))
            log = open(OUT / "logs" / f"{name}.try{attempts[name]}.log", "w")
            running[name] = (subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log,
                                              stderr=subprocess.STDOUT), gpu, job)
        time.sleep(5)
    print("all jobs finished", flush=True)


def select(datasets) -> pd.DataFrame:
    rows = []
    for d, h in itertools.product(datasets, HORIZONS):
        for c in configs():
            path = OUT / "search" / f"{d}_H{h}" / tag(c) / "method_summary.csv"
            if path.exists():
                row = pd.read_csv(path).iloc[0]
                rows.append({"dataset": d, "horizon": h, **c,
                             "val_mse": row.prequential_mse_mean, "val_std": row.prequential_mse_std})
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "search_summary.csv", index=False)
    # Ties are broken toward fewer epochs and lower rank (cheaper configurations).
    best = (grid.sort_values(["dataset", "horizon", "val_mse", "adapt_epochs", "mod_rank"])
            .groupby(["dataset", "horizon"]).head(1))
    best.to_csv(OUT / "selection.csv", index=False)
    print(best.to_string(index=False))
    return best


def final_jobs(best: pd.DataFrame):
    jobs = []
    for row in best.itertuples():
        c = {k: getattr(row, k) for k in GRID}
        out = OUT / "final" / f"{row.dataset}_H{row.horizon}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "selected_config.json").write_text(json.dumps(
            {k: (int(v) if k in ("adapt_epochs", "mod_rank") else float(v)) for k, v in c.items()},
            indent=2), encoding="utf-8")
        jobs.append((f"final_{row.dataset}_H{row.horizon}",
                     MAIN + data_args(row.dataset, int(row.horizon)) + PROTOCOL
                     + ["--seeds", "5", "--methods", "Frozen", "DSGA", *dv_args(c),
                        "--output", str(out)], out, SIZE[row.dataset]))
    return jobs


def search_complete(dataset: str, horizon: int) -> bool:
    return all((OUT / "search" / f"{dataset}_H{horizon}" / tag(c) / "method_summary.csv").exists()
               for c in configs())


def dsga_first_jobs(datasets):
    """Remaining search jobs; small streams first so that finals arrive early."""
    order = {"Exchange": 0, "SCA": 1, "ETTh2": 2, "PEMS08": 3, "Energy": 4,
             "Weather": 6, "Electricity": 7, "Traffic": 8, "Air": 9}
    return [(name, cmd, out, 100 + order[name.split("_")[1]])
            for name, cmd, out, _ in search_jobs(datasets)]


def ready_finals(datasets):
    pairs = [(d, h) for d, h in itertools.product(datasets, HORIZONS) if search_complete(d, h)]
    if not pairs:
        return []
    best = select([d for d in datasets])
    best = best[[ (r.dataset, int(r.horizon)) in pairs for r in best.itertuples()]]
    return [(n, c, o, -10 + p) for n, c, o, p in final_jobs(best)
            if not (o / "method_summary.csv").exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("search", "dsga-first", "select", "final"))
    # GPUs with processes from other users (or other projects) are skipped automatically.
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--per-gpu", type=int, default=4)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    datasets = available(args.datasets)
    if args.stage == "search":
        schedule(search_jobs(datasets), args.gpus, args.per_gpu, args.threads)
    elif args.stage == "dsga-first":
        state = {"t": 0.0, "jobs": []}

        def refresh():
            # Re-check finished searches at most once a minute.
            if time.time() - state["t"] > 60:
                state["t"], state["jobs"] = time.time(), ready_finals(datasets)
            return state["jobs"]

        schedule(dsga_first_jobs(datasets), args.gpus, args.per_gpu, args.threads,
                 refresh=refresh)
    elif args.stage == "select":
        select(datasets)
    else:
        schedule(final_jobs(select(datasets)), args.gpus, args.per_gpu, args.threads)


if __name__ == "__main__":
    main()
