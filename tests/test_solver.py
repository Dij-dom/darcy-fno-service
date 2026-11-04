"""Solver tests that do not depend on the dataset."""

import numpy as np
import pytest

from darcy.solver import assemble, residual, solve, torsion_series


def centres(n):
    return (np.arange(n) + 0.5) / n


def err_vs_analytic(n, boundary):
    x = centres(n)
    h = 1.0 / n
    u = solve(np.ones((n, n)), boundary=boundary)
    if boundary == "face":   # zero on the walls x = 0, 1
        ref = torsion_series(x, x, length=1.0, offset=0.0)
    else:                    # zero at ghost centres x = -h/2, 1 + h/2
        ref = torsion_series(x, x, length=1.0 + h, offset=h / 2)
    return np.abs(u - ref).max() / np.abs(ref).max()


@pytest.mark.parametrize("boundary", ["face", "ghost"])
def test_constant_coefficient_matches_analytic_and_converges(boundary):
    errs = [err_vs_analytic(n, boundary) for n in (32, 64, 128)]
    assert errs[-1] < 1e-3, errs
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(orders > 1.8), (errs, orders)   # second-order accurate


def test_scaling_with_coefficient_and_beta():
    rng = np.random.default_rng(0)
    a = np.where(rng.random((32, 32)) > 0.5, 1.0, 0.1)
    u = solve(a)
    assert np.allclose(solve(10 * a), u / 10, rtol=1e-10)
    assert np.allclose(solve(a, beta=3.0), 3 * u, rtol=1e-10)


def test_symmetric_matrix_positive_solution_zero_residual():
    rng = np.random.default_rng(1)
    a = np.where(rng.random((32, 32)) > 0.5, 1.0, 0.1)
    A = assemble(a)
    assert abs(A - A.T).max() < 1e-14
    u = solve(a)
    assert u.min() > 0
    assert residual(a, u) < 1e-10


def test_transpose_equivariance():
    rng = np.random.default_rng(2)
    a = np.where(rng.random((32, 32)) > 0.5, 1.0, 0.1)
    assert np.allclose(solve(a.T), solve(a).T, atol=1e-12)


def test_rejects_bad_coefficients():
    with pytest.raises(ValueError):
        solve(np.zeros((8, 8)))
    with pytest.raises(ValueError):
        solve(np.ones((8, 4)))
