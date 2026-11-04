# Research: how is this label noise handled elsewhere? (2026-11-04)

## What is known about this dataset

- **PDEBench issue #102** (open, filed 2026-10-19, no maintainer reply as of today):
  identical constant-0.1 inputs with very different targets in the sibling file
  `2D_DarcyFlow_beta0.01_Train.hdf5`. Reporter suspects unconverged time stepping and
  points to `Data_Merge.py` taking `test[:batch, -2]`, but did not confirm the cause.
  Our CP1 analysis confirms that mechanism quantitatively for the beta = 1.0 file.
  https://github.com/pdebench/PDEBench/issues/102
- **PDEBench issue #41** (2023): their metric code includes the input channel when
  computing Darcy errors, which a maintainer confirmed overestimates model performance.
  Published PDEBench Darcy scores are therefore not a clean reference.
  https://github.com/pdebench/PDEBench/issues/41
- No official corrected dataset or recommended treatment found.

## Which time the labels are taken at (measured)

`Data_Merge.py` line 175 stores frame `-2` for 2D data. Frames are saved every 0.25,
with the last slot overwritten by the state at t = 2.0, so `-2` is t = 1.75.
Check on the 46 constant-0.1 samples (initial state ~ constant u0 ~ U(0.1, 10),
lambda_1 = 1.9433): the noise coefficient c = (u0 * 104.55 - 53.80) * exp(-lambda_1 t).

| assumed t | implied u0 range over the 46 samples | possible? |
|---|---|---|
| 1.75 | [0.43, 9.38] | yes |
| 2.00 | [0.37, 14.93] | no (u0 <= 10) |

Largest coefficient with u0 in [9, 10] gives t = 1.73 to 1.79. **Labels are the t = 1.75
snapshot.** (A regression of log c on lambda_1 over all samples gave t ~ 0.89, but that
regression is confounded because the starting distance depends on lambda_1; not used.)

## What others do (from web search; secondary sources marked)

- **Original FNO Darcy data (Li et al.)**: generated with a second-order finite-difference
  solver at 421 x 421 and downsampled, i.e. converged steady-state labels from a direct
  solve. (Source: PDEInvBench appendix, https://arxiv.org/abs/2605.25353; not checked
  against the FNO paper itself.)
- **Selection bias from filtering**: PDEformer-2 notes that discarding samples a solver
  handles badly makes the training set under-represent numerically hard PDE instances.
  (https://arxiv.org/abs/2507.15409; quoted via search summary, full text not read.)
- **Noise robustness**: some papers inject noise into training data to test FNO
  robustness on Darcy, Burgers and Navier-Stokes
  (https://fastmachinelearning.org/iccad2023/file/fastml-iccad-23-final5.pdf).
- **Label-free training** (physics-informed / variational losses) avoids solver labels
  entirely (e.g. https://arxiv.org/html/2609.36216). Out of scope for 72 hours.
- At least one public project re-solves PDEBench Darcy inputs with PDEBench's own stencil
  (conjugate gradients) to get training targets (search result, not inspected).

Not found: any peer-reviewed treatment of this specific PDEBench Darcy defect, or a
community-standard noise threshold for filtering.
