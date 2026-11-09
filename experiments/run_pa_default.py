"""DSGA-DV with its default configuration under protocol A (no per-dataset selection).

Runs on CPU while no GPU is free; uses at most --max-gpus GPUs that have no process of
anyone else (including other projects of the same account).
    python run_pa_default.py --max-gpus 2 --cpu-jobs 6 --threads 4
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import time

import run_pa_protocol as P

DEFAULT = {"lr_scale": 1.0, "adapt_epochs": 5, "mod_rank": 2, "minimum_effect_z": 2.5}
OUT = P.OUT / "final_default"


def jobs(datasets):
    out = []
    for d, h in itertools.product(datasets, P.HORIZONS):
        target = OUT / f"{d}_H{h}"
        cmd = P.MAIN + P.data_args(d, h) + P.PROTOCOL[:-2] + [
            "--seeds", "5", "--methods", "Frozen", "DSGA", *P.dv_args(DEFAULT), "--output", str(target)]
        out.append((f"default_{d}_H{h}", cmd, target, -P.SIZE[d]))
    return sorted(out, key=lambda j: j[3])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--per-gpu", type=int, default=3)
    parser.add_argument("--cpu-jobs", type=int, default=6)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    queue = [j for j in jobs(P.available(list(P.DATASETS))) if not (j[2] / "method_summary.csv").exists()]
    attempts, running, claimed = {j[0]: 0 for j in queue}, {}, []
    print(f"{len(queue)} jobs to run", flush=True)
    while queue or running:
        for name, (proc, where, job) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[name]
            if proc.returncode == 0:
                print(f"done {name} ({where})", flush=True)
            elif attempts[name] < 3:
                print(f"retry {name} (exit {proc.returncode})", flush=True)
                queue.insert(0, job)
            else:
                print(f"FAILED {name}", flush=True)
        procs = P.gpu_processes()
        ours = {proc.pid for proc, _, _ in running.values()}
        for g in range(8):
            foreign = {p for p in procs.get(g, set()) if p not in ours and not P.is_ours(p)}
            if g in claimed and foreign:
                claimed.remove(g)
                print(f"GPU{g} released (foreign process appeared)", flush=True)
            elif g not in claimed and not foreign and len(claimed) < args.max_gpus:
                claimed.append(g)
                print(f"GPU{g} claimed", flush=True)
        slots = [f"gpu{g}" for g in claimed
                 if sum(1 for _, w, _ in running.values() if w == f"gpu{g}") < args.per_gpu]
        if sum(1 for _, w, _ in running.values() if w == "cpu") < args.cpu_jobs:
            slots.append("cpu")
        for where in slots:
            if not queue:
                break
            job = queue.pop(0)
            name, cmd, _, _ = job
            attempts[name] += 1
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            env["CUDA_VISIBLE_DEVICES"] = "" if where == "cpu" else where[3:]
            device = "cpu" if where == "cpu" else "cuda:0"
            log = open(OUT / "logs" / f"{name}.try{attempts[name]}.log", "w")
            running[name] = (subprocess.Popen(cmd + ["--device", device], cwd=P.HERE, env=env,
                                              stdout=log, stderr=subprocess.STDOUT), where, job)
        time.sleep(15)
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
