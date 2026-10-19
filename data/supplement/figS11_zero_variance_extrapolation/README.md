# Fig. S11: zero-variance extrapolation on 16x4

The training curve of an NN-fPEPS with N_p = 37,791,744 (weights not released) on the 16x4 lattice
at t' = 0 (28 + 28 electrons, N = 64), trained under the symmetry projection, and the fits that give
E0/N = -0.76722(6). `code/plots/zero_variance_extrapolation.py` redoes all fits from the curve.

- `training_curve.csv`: per `step` (0-10000), the total energy `E` (t) and variance `VarE` (t^2) of
  the 8192 samples of the step, in float32 as logged. `step_norm` is not used by the fit.
- `grid.csv`: one orthogonal-distance fit E = k VarE + E0 per window `start` <= step < `end` and
  bin size `bin` (steps): slope `k`, total `E0` and `E0_err` (t), residual variance `res_var`,
  lever arm `R`, extrapolation distance `D` = k x mean binned VarE, and `kept`.

A fit is kept if it converged, k > 0, E0 lies below every binned energy of its window, and E0_err is
at most 5 times the median (44 of 45 fits). E0 is the mean of the kept intercepts. Its error adds in
quadrature the mean fit error (0.0030) and the spread of the intercepts (0.0020), divided by N.
