"""Consistency check on samples whose permeability is constant (all 0.1).

Identical inputs should give identical solutions. This script measures how much
their pressure fields differ, and whether the differences have the shape of the
slowest diffusion mode sin(pi x) sin(pi y) (the signature of a time-stepped solve
stopped before reaching steady state).

Usage: python scripts/check_constant_samples.py [path.hdf5]
"""

import sys

import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/2D_DarcyFlow_beta1.0_Train.hdf5"
with h5py.File(path, "r") as f:
    nu = f["nu"][()]
    u = f["tensor"][:, 0]
    x = f["x-coordinate"][()]

flat = nu.reshape(len(nu), -1)
const = np.where(flat.max(1) == flat.min(1))[0]
uc = u[const]
umax = uc.max((1, 2))
print(f"constant-permeability samples: {const.size}, values {np.unique(nu[const])}")
print(f"u max across them:  min {umax.min():.4f}, max {umax.max():.4f}")
print(f"u mean across them: min {uc.mean((1, 2)).min():.4f}, max {uc.mean((1, 2)).max():.4f}")
print(f"max |u_i - u_j| between them: {np.abs(uc - uc[0]).max():.4f}")

best = uc[umax.argmin()]
X, Y = np.meshgrid(x, x, indexing="ij")
mode = np.sin(np.pi * X) * np.sin(np.pi * Y)
cos = [(d * mode).sum() / np.linalg.norm(d) / np.linalg.norm(mode)
       for d in uc - best if np.abs(d).max() > 1e-3]
print(f"(u_i - u_lowest) vs sin(pi x)sin(pi y), cosine similarity: n={len(cos)}, "
      f"min {min(cos):.4f}, median {np.median(cos):.4f}")

m = u.max((1, 2))
frac_low = (nu == 0.1).mean((1, 2))
print(f"all samples: corr(fraction of a=0.1, u max) = {np.corrcoef(frac_low, m)[0, 1]:.3f}")
print(f"all samples: u max > {umax.min():.3f} (the lowest constant-sample max): {(m > umax.min()).sum()}")
