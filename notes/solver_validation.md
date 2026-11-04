# Solver validation and label noise (CP1)

## How PDEBench generated the labels (read from source)

Source: `pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_2D_multi_solution_Hydra.py`
and `pdebench/data_gen/data_gen_NLE/utils.py`, PDEBench commit `9bbdb62` (2025-05-15);
config `config/multi/config_2D.yaml`; launched by `run_DarcyFlow2D.sh`.

- Labels come from **explicit time stepping** of du/dt = div(a grad u) + beta up to
  `fin_time = 2.0`, not from a steady-state solve.
- Initial condition (`init_multi_2DRand`): constant offset u0 ~ U(0.1, 10) plus a random
  Fourier field, optionally windowed.
- Boundary: `bc_2D(mode="Neumann")` **sets ghost cells to zero**, i.e. a Dirichlet
  condition imposed at the ghost-cell centre, h outside the edge cell (not on the wall).
- Face coefficient: arithmetic mean of neighbouring cells; coefficient ghosts copy the edge.
- Permeability: sum of 5 random Gaussians thresholded at the mean -> {1.0, 0.1}.
- The dataset file's `tensor` holds a single time frame; that it is the final frame
  (t = 2) is inferred from the residual analysis below, not stated in the code we read.

## Our solver (`darcy/solver.py`)

Direct sparse solve of the same spatial discretisation (so the converged limit of
PDEBench's time stepping). Tests in `tests/test_solver.py` (no dataset needed):
constant-coefficient case vs analytic double-sine series, both boundary conventions.

| N | max rel error, wall boundary | max rel error, ghost boundary |
|---|---|---|
| 32 | 1.61e-03 | 7.23e-04 |
| 64 | 4.08e-04 | 1.86e-04 |
| 128 | 1.04e-04 | 4.75e-05 |

Second-order convergence for both. Plus scaling, symmetry, positivity, residual tests.

## Comparison with dataset labels

`python -m scripts.validate_solver --n 500 --seed 0`: 500 random samples + the 46
constant-input samples = 546. d = label - ours.

- Ghost-boundary convention fits far better than the wall convention:
  median rel L2 2.77e-03 vs 4.45e-02. **The dataset uses the ghost convention.**
- Removing the component of d along phi_1 (slowest eigenmode of that sample's operator)
  leaves a rel L2 of mean 3.6e-05, max 1.95e-04. phi_1 holds >= 97.8% of ||d||^2 in every
  sample (median 100%).
- Constant a = 0.1: our max 0.7482; labels range 0.7434 to 1.2273.
- Coefficient on phi_1 is positive in 541 / 546 samples (labels mostly above steady state,
  consistent with the positive offset u0 in the initial condition).

**Conclusion:** label = converged solution + (unconverged slowest mode) to within ~1e-4.
Our solver reproduces PDEBench's discretisation; the remaining disagreement is label noise.

### Size of the noise (rel L2 of label vs converged solution)

- Over the 546 samples: median 2.77e-03, p90 2.30e-01, p99 5.28e-01, max 5.72e-01.
- > 1e-3: 338 / 546; > 1e-2: 199 / 546; > 5e-2: 113 / 546.
- Strongly tied to the amount of low permeability (slow diffusion):
  corr(fraction of a = 0.1, log noise) = 0.914; corr(lambda_1, log noise) = -0.837.

| fraction of a = 0.1 | n | median noise | share with noise > 5% |
|---|---|---|---|
| [0, 0.25) | 48 | 3.21e-04 | 0.00 |
| [0.25, 0.5) | 218 | 6.69e-04 | 0.00 |
| [0.5, 0.75) | 166 | 1.05e-02 | 0.12 |
| [0.75, 1] | 114 | 2.10e-01 | 0.81 |

For scale (unverified, from memory): published FNO errors on Darcy are around 1e-2 rel L2,
so if that holds, this noise is larger
than the error we are trying to measure for a substantial share of samples.
