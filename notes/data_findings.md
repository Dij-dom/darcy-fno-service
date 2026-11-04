# Data findings (CP0)

Measured by `scripts/inspect_data.py` (raw output in `notes/data_inspection.md`) and
`scripts/check_constant_samples.py`. Interpretations are marked as such.

## Measured facts

- File: 10,000 samples. `nu` (permeability) `(10000, 128, 128)` float32;
  `tensor` (pressure) `(10000, 1, 128, 128)` float32; file attribute `beta = 1.0`.
- No NaN or Inf anywhere.
- Permeability is binary: exactly two values, 0.1 and 1.0. 9,954 samples contain both;
  46 samples are constant 0.1. No sample is constant 1.0. Fraction of cells at 1.0 per
  sample: min 0.000, median 0.511, max 0.964.
- Pressure: min 1.78e-4, max 1.235, mean 0.162. Always positive.
- Grid: x and y are `(i + 0.5) / 128`, i = 0..127, i.e. **cell centres** of a uniform
  128 x 128 grid on [0, 1]^2 with h = 1/128. The boundary itself is not stored, which
  is why the smallest pressure is small but non-zero.
- **Identical inputs, different outputs.** The 46 constant-0.1 samples should all have
  the same solution, but their max pressure ranges from 0.743 to 1.227
  (max pointwise difference 0.332). Each solution is still symmetric.
- The difference between each of those solutions and the lowest one has cosine
  similarity 0.9999 with sin(pi x) sin(pi y) (45 of 45 samples).

## Interpretation (not yet verified)

- For −∇·(a∇u) = 1 with constant a = 0.1 and u = 0 on the boundary, I expect
  max u ≈ 0.0737 / a ≈ 0.737 (torsion function of the unit square; constant quoted
  from memory). The lowest constant sample (0.743) is close to that; the others are not.
- sin(pi x) sin(pi y) is the slowest-decaying mode of the diffusion operator. Differences
  of exactly that shape are what you get if the solutions were produced by time-stepping
  ∂u/∂t − ∇·(a∇u) = β towards steady state and stopped before it converged. I believe
  PDEBench generated Darcy this way, but I have not checked their code or paper yet.
- If that is right, part of the target fields is **label noise** that the input cannot
  predict. That puts a floor under any model's test error and affects comparisons with
  our finite-difference solver (CP1) and with published numbers.

**Correction (CP1):** the comparison above with 0.737 was misleading. Under PDEBench's
own discretisation the converged constant-0.1 maximum is 0.7482 (see
`notes/solver_validation.md`). The 0.743 sample lies *below* it (its random initial
condition started below steady state); it is not the "cleanest" sample.

## To verify in CP1 (done, see `notes/solver_validation.md`)

1. Run our FD solver on the constant-0.1 input; check its max against 0.743 and 0.737.
2. Measure FD vs dataset error over all samples, and project the residual onto
   sin(pi x) sin(pi y) to see how much of the mismatch is this one mode.
3. Read PDEBench's Darcy generation code to confirm or reject the time-stepping explanation.
