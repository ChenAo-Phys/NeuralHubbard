# Fig. S5: linear solvers

The SR system A x = b, with A of shape N_s x N_p, frozen for two trained 8x8 states at t' = -0.2:
`pfaffian` (Pfaffian backflow, N_p = 1,191,936) and `nnfpeps` (NN-fPEPS with a per-site embedding,
N_p = 4,599,808). The samples are split into disjoint groups of N_s (4, 4, 4, 4, 2, 1 groups at
N_s = 2^11 ... 2^16); errors are standard errors over groups. Each A100 GPU holds 512 samples.

- `krylov_iterations.csv`: one row per solve with CG (on A^T A x = A^T b) or LSMR (on A x = b), in
  FP32 without shift from x = 0, for a fixed `n_iter`: time `time_s` (s), residual
  `eta` = |A x - b| / |b| and step norm `xnorm`.
- `krylov_summary.csv`: the plotted Krylov points, interpolated (log-log) at the iteration count
  where CG reaches eta = 0.1; `*_sem` are standard errors over groups.
- `cholesky.csv`: FP64 solves x = A^T (A A^T + lambda)^-1 b at the `lambda` that gives eta = 0.01,
  with mean time `t_solve_s` and `xnorm`. `extrapolated` = 1 marks N_s = 65536, where the kernel
  exceeds the GPU memory and the time is a power-law extrapolation. `lambda_source` is `spectrum`
  (per-group eigendecomposition) or `lambda_scan` (interpolated on a lambda grid, no error bar).
