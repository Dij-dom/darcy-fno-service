"""Create train / val / test index splits (the dataset itself is never modified).

Design ("middle ground", decided 2026-11-04):
  1. Test set: drawn from ALL samples before any filtering, stratified by the fraction
     of low-permeability cells, so it covers the full input distribution. It is later
     scored against both the original labels and the converged solver solution.
  2. Train / val: from the remaining samples, keep only labels with noise
     (rel L2 vs converged solution, results/label_noise.csv) <= --threshold,
     then split stratified the same way.
  3. Samples with byte-identical inputs are grouped and never cross splits.

Writes splits/splits.json with indices and metadata.

Usage: python -m scripts.make_splits [--threshold 0.01] [--test-frac 0.1] [--val-frac 0.1] [--seed 0]
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict

import h5py
import numpy as np

PATH = "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"
NOISE_CSV = "results/label_noise.csv"
BINS = [0.0, 0.25, 0.5, 0.75, 1.01]  # fraction of a = 0.1 cells


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stratified_pick(groups, strata, frac, rng):
    """Pick about `frac` of the groups in each stratum. groups: list of index arrays."""
    picked = []
    for s in np.unique(strata):
        g = [i for i in range(len(groups)) if strata[i] == s]
        rng.shuffle(g)
        n_s = sum(len(groups[i]) for i in g)
        target, got = round(frac * n_s), 0
        for i in g:
            if got >= target:
                break
            picked.append(i)
            got += len(groups[i])
    return set(picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="splits/splits.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    with h5py.File(PATH, "r") as f:
        nu = f["nu"][()]
    n = len(nu)
    noise = np.genfromtxt(NOISE_CSV, delimiter=",", names=True)
    assert np.array_equal(noise["idx"].astype(int), np.arange(n))
    frac_low = (nu < 0.55).reshape(n, -1).mean(1)
    stratum = np.digitize(frac_low, BINS) - 1

    # group byte-identical inputs
    by_hash = defaultdict(list)
    for i in range(n):
        by_hash[hashlib.sha1(nu[i].tobytes()).hexdigest()].append(i)
    groups = [np.array(v) for v in by_hash.values()]
    dup_groups = [g.tolist() for g in groups if len(g) > 1]
    gstrat = np.array([stratum[g[0]] for g in groups])

    # 1. test from everything
    test_g = stratified_pick(groups, gstrat, args.test_frac, rng)
    test = np.sort(np.concatenate([groups[i] for i in test_g]))

    # 2. filter the rest; a duplicate group is kept only if all its members pass
    rest = [i for i in range(len(groups)) if i not in test_g
            and np.all(noise["noise_rel_l2"][groups[i]] <= args.threshold)]
    rest_groups = [groups[i] for i in rest]
    val_g = stratified_pick(rest_groups, gstrat[rest], args.val_frac, rng)
    val = np.sort(np.concatenate([rest_groups[i] for i in val_g]))
    train = np.sort(np.concatenate([rest_groups[i] for i in range(len(rest_groups)) if i not in val_g]))

    assert not (set(train) & set(val)) and not (set(train) & set(test)) and not (set(val) & set(test))
    excluded = np.setdiff1d(np.arange(n), np.concatenate([train, val, test]))

    def summary(ix):
        z = noise["noise_rel_l2"][ix]
        return {"n": int(len(ix)),
                "per_stratum": [int((stratum[ix] == s).sum()) for s in range(len(BINS) - 1)],
                "noise_median": float(np.median(z)), "noise_max": float(z.max())}

    out = {
        "dataset": os.path.basename(PATH), "dataset_md5": md5(PATH),
        "noise_csv_md5": md5(NOISE_CSV),
        "threshold": args.threshold, "test_frac": args.test_frac, "val_frac": args.val_frac,
        "seed": args.seed, "strata_frac_low_bins": BINS[:-1] + [1.0],
        "duplicate_input_groups": dup_groups,
        "summary": {"train": summary(train), "val": summary(val), "test": summary(test),
                    "excluded_noisy": summary(excluded)},
        "train": train.tolist(), "val": val.tolist(), "test": test.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"duplicate input groups: {len(dup_groups)} (sizes {sorted(len(g) for g in dup_groups)[-5:]} largest)")
    for k, v in out["summary"].items():
        print(f"{k:15s} {v}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
