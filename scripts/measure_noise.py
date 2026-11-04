"""Measure label noise for every sample in the PDEBench Darcy file (read-only).

noise = ||label - ours|| / ||ours||, where `ours` is the converged steady-state solution
from darcy.solver (ghost boundary, PDEBench's discretisation). The dataset is not modified;
this only produces per-sample measurements used to select samples.

Writes results/label_noise.csv with one row per sample:
    idx, noise_rel_l2, frac_low, label_max, steady_max

Usage: python -m scripts.measure_noise [--workers 12]
"""

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

from darcy.solver import solve

PATH = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"
CHUNK = 250


def work(start):
    with h5py.File(PATH, "r") as f:
        nu = f["nu"][start:start + CHUNK].astype(np.float64)
        lab = f["tensor"][start:start + CHUNK, 0].astype(np.float64)
    rows = []
    for k in range(len(nu)):
        u = solve(nu[k], boundary="ghost")
        rows.append((start + k,
                     float(np.linalg.norm(lab[k] - u) / np.linalg.norm(u)),
                     float((nu[k] < 0.55).mean()),   # binary field: 0.1 or 1.0
                     float(lab[k].max()),
                     float(u.max())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="results/label_noise.csv")
    args = ap.parse_args()
    with h5py.File(PATH, "r") as f:
        n = f["nu"].shape[0]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(work, range(0, n, CHUNK)):
            rows.extend(r)
            print(f"{len(rows)}/{n} samples, {time.time() - t0:.0f}s", flush=True)
    rows.sort()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "noise_rel_l2", "frac_low", "label_max", "steady_max"])
        for r in rows:
            w.writerow([r[0]] + [f"{v:.6e}" for v in r[1:]])
    print(f"wrote {args.out} ({len(rows)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
