"""Steady-state finite-volume solver for 2D Darcy flow.

Solves  -div(a grad u) = beta  on the unit square with u = 0 outside the domain,
on an N x N grid of cell centres x_i = (i + 0.5) / N.

The discretisation mirrors PDEBench's generator
(pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_2D_multi_solution_Hydra.py,
commit 9bbdb62):
  * 5-point flux form, face coefficient = arithmetic mean of the two neighbouring cells;
  * boundary faces use the edge cell's coefficient (their "trans" ghost copy for nu);
  * the zero value is imposed at the ghost-cell centre, one full cell (h) outside the
    edge cell (their "Neumann" mode, which actually zeroes the ghost cells).

PDEBench reaches (approximately) this steady state by explicit time stepping to t = 2;
here we solve the linear system directly, so the result is the converged answer.

`boundary="face"` instead imposes u = 0 on the wall itself (h/2 from the edge cell),
the textbook cell-centred Dirichlet condition. It exists to test which convention the
dataset uses, not for production.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

BOUNDARY_FACTOR = {"ghost": 1.0, "face": 2.0}


def assemble(a: np.ndarray, boundary: str = "ghost") -> sp.csr_matrix:
    """Matrix A (scaled by h^2 removed, i.e. A u = beta h^2) for coefficient field a[i, j],
    where i indexes x and j indexes y. Unknowns are ordered u.ravel() (C order)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"expected square 2D coefficient field, got shape {a.shape}")
    if np.any(~np.isfinite(a)) or np.any(a <= 0):
        raise ValueError("coefficient must be finite and strictly positive")
    bf = BOUNDARY_FACTOR[boundary]
    n = a.shape[0]
    idx = np.arange(n * n).reshape(n, n)

    # interior face coefficients (arithmetic mean)
    kx = 0.5 * (a[1:, :] + a[:-1, :])   # faces between i and i+1, shape (n-1, n)
    ky = 0.5 * (a[:, 1:] + a[:, :-1])   # faces between j and j+1, shape (n, n-1)

    diag = np.zeros((n, n))
    diag[1:, :] += kx
    diag[:-1, :] += kx
    diag[:, 1:] += ky
    diag[:, :-1] += ky
    # boundary faces: coefficient of the edge cell, distance to the zero value set by bf
    diag[0, :] += bf * a[0, :]
    diag[-1, :] += bf * a[-1, :]
    diag[:, 0] += bf * a[:, 0]
    diag[:, -1] += bf * a[:, -1]

    rows = [idx.ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(), idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    cols = [idx.ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(), idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    vals = [diag.ravel(), -kx.ravel(), -kx.ravel(), -ky.ravel(), -ky.ravel()]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n * n, n * n))


def solve(a: np.ndarray, beta: float = 1.0, boundary: str = "ghost") -> np.ndarray:
    """Pressure field u[i, j] (float64) for coefficient field a[i, j]."""
    n = np.asarray(a).shape[0]
    h = 1.0 / n
    A = assemble(a, boundary).tocsc()
    rhs = np.full(n * n, beta * h * h)
    return spla.spsolve(A, rhs).reshape(n, n)


def residual(a: np.ndarray, u: np.ndarray, beta: float = 1.0, boundary: str = "ghost") -> float:
    """Relative residual ||A u - b|| / ||b|| of a candidate field u under this discretisation."""
    n = np.asarray(a).shape[0]
    h = 1.0 / n
    b = np.full(n * n, beta * h * h)
    return float(np.linalg.norm(assemble(a, boundary) @ np.asarray(u, np.float64).ravel() - b)
                 / np.linalg.norm(b))


def torsion_series(x: np.ndarray, y: np.ndarray, length: float = 1.0, offset: float = 0.0,
                   terms: int = 199) -> np.ndarray:
    """Analytic solution of -lap u = 1 on the square [-offset, length - offset]^2 with u = 0
    on its boundary (double sine series over odd modes). Used only to test the solver."""
    X, Y = np.meshgrid(np.asarray(x, np.float64) + offset, np.asarray(y, np.float64) + offset,
                       indexing="ij")
    u = np.zeros_like(X)
    for m in range(1, terms + 1, 2):
        sx = np.sin(m * np.pi * X / length)
        for k in range(1, terms + 1, 2):
            coef = 16.0 / (np.pi ** 2 * m * k) / (np.pi ** 2 * (m * m + k * k) / length ** 2)
            u += coef * sx * np.sin(k * np.pi * Y / length)
    return u
