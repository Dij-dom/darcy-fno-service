"""Compare our steady-state solver against the PDEBench Darcy labels.

For each selected sample:
  * rel_l2_ghost / rel_l2_face: ||label - ours|| / ||ours|| for both boundary conventions
  * for the ghost convention, split d = label - ours into its component along phi_1, the
    slowest-decaying eigenmode of that sample's operator, and the remainder.
    If the labels are unconverged time-stepping results, d should be almost entirely phi_1.

Writes per-sample metrics to results/solver_validation.npz and prints a summary.

Usage: python scripts/validate_solver.py [--n 500] [--seed 0] [--all]
"""

import argparse
import os
import time

import h5py
import numpy as np
import scipy.sparse.linalg as spla

from darcy.solver import assemble, solve

PATH = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"


def slowest_mode(a):
    A = assemble(a, "ghost")
    vals, vecs = spla.eigsh(A, k=1, sigma=0, which="LM")
    v = vecs[:, 0]
    return float(vals[0]) * a.shape[0] ** 2, v * np.sign(v.sum())  # eigenvalue of -div(a grad)


def q(x, ps=(0, 50, 90, 99, 100)):
    return ", ".join(f"p{p} {np.percentile(x, p):.2e}" for p in ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="results/solver_validation.npz")
    args = ap.parse_args()

    with h5py.File(PATH, "r") as f:
        nu = f["nu"][()]
        lab = f["tensor"][:, 0].astype(np.float64)
    N = len(nu)
    idx = np.arange(N) if args.all else np.sort(np.random.default_rng(args.seed).choice(N, args.n, replace=False))
    const = np.where(nu.reshape(N, -1).min(1) == nu.reshape(N, -1).max(1))[0]
    idx = np.union1d(idx, const)  # always include the constant-input samples

    rec = {k: [] for k in ["idx", "rel_ghost", "rel_face", "rel_after_mode1", "mode1_frac",
                           "mode1_coef", "lambda1", "frac_low", "max_ours", "max_label"]}
    t0 = time.time()
    for c, i in enumerate(idx):
        a = nu[i].astype(np.float64)
        ug, uf = solve(a, boundary="ghost"), solve(a, boundary="face")
        d = lab[i] - ug
        lam, phi = slowest_mode(a)
        coef = float(d.ravel() @ phi)
        rest = d.ravel() - coef * phi
        nd = np.linalg.norm(d)
        rec["idx"].append(i)
        rec["rel_ghost"].append(nd / np.linalg.norm(ug))
        rec["rel_face"].append(np.linalg.norm(lab[i] - uf) / np.linalg.norm(uf))
        rec["rel_after_mode1"].append(np.linalg.norm(rest) / np.linalg.norm(ug))
        rec["mode1_frac"].append(1 - (np.linalg.norm(rest) / nd) ** 2 if nd > 0 else 1.0)
        rec["mode1_coef"].append(coef)
        rec["lambda1"].append(lam)
        rec["frac_low"].append(float((a < 0.55).mean()))  # binary field: 0.1 or 1.0
        rec["max_ours"].append(ug.max())
        rec["max_label"].append(lab[i].max())
        if (c + 1) % 100 == 0:
            print(f"{c + 1}/{len(idx)} samples, {time.time() - t0:.0f}s", flush=True)
    rec = {k: np.array(v) for k, v in rec.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **rec)

    isc = np.isin(rec["idx"], const)
    print(f"\nsamples: {len(idx)} ({isc.sum()} constant-input), time {time.time() - t0:.0f}s")
    print(f"constant a=0.1: our max {rec['max_ours'][isc][0]:.4f}; label max range "
          f"[{rec['max_label'][isc].min():.4f}, {rec['max_label'][isc].max():.4f}]")
    print(f"rel L2 label vs ours, ghost boundary: mean {rec['rel_ghost'].mean():.3e}; {q(rec['rel_ghost'])}")
    print(f"rel L2 label vs ours, face  boundary: mean {rec['rel_face'].mean():.3e}; {q(rec['rel_face'])}")
    print(f"rel L2 after removing slowest mode:   mean {rec['rel_after_mode1'].mean():.3e}; {q(rec['rel_after_mode1'])}")
    print(f"share of ||d||^2 in slowest mode:      {q(rec['mode1_frac'], (0, 1, 10, 50))}")
    print(f"slowest-mode coefficient sign: {(rec['mode1_coef'] > 0).sum()} positive, "
          f"{(rec['mode1_coef'] <= 0).sum()} non-positive")
    print(f"corr(frac of a=0.1, rel_ghost) = {np.corrcoef(rec['frac_low'], rec['rel_ghost'])[0, 1]:.3f}; "
          f"corr(lambda1, log rel_ghost) = {np.corrcoef(rec['lambda1'], np.log(rec['rel_ghost']))[0, 1]:.3f}")
    for thr in (1e-3, 1e-2, 5e-2):
        print(f"samples with rel L2 (ghost) > {thr:g}: {(rec['rel_ghost'] > thr).sum()} / {len(idx)}")


if __name__ == "__main__":
    main()
