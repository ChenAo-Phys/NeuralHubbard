# Fig. S4: optimizer benchmark

BCS backflow on the 8x8 lattice at t' = -0.2 (28 + 28 electrons), trained from a cold start for
5000 iterations with 12288 samples each. Each curve is the best run of its optimizer.

- `training_curves.csv`: per `iteration`, `<opt>_E_per_site` (E/N, units of t) and
  `<opt>_var_per_site` (sigma^2/N, units of t^2) of the samples drawn before the update, for
  `<opt>` = `SR`, `SPRING`, `MARCH`, `AdamSR`, stored in float32.
- `binned_curves.csv`: the plotted curves, E/N averaged over 50-iteration bins centred at
  `iteration_bin_center`. The one outlier with E/N < -50/64 (SR, iteration 43) is left out.

The best runs use C_0 = 1 (SR, AdamSR) or 0.5 (SPRING, MARCH); SR samples with electron hops and
neighbor exchanges, the others with electron hops only. The network is the BCS backflow of Table S1
(`benchmark_figS4_bcs`), with residual block i scaled by 1/sqrt(i+1) instead of RMSNorm.
