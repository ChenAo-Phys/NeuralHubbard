# Fig. S10: extrapolation of the pure fPEPS energy in the gauge weight

The bootstrap extrapolation E(w) = E0 + c w (SM Eq. (S27)) of the per-D energies of
[Fig. S9](../figS09_fpeps_L16_convergence/) in the gauge weight w (SM Eq. (S26)). Each bootstrap
sample draws p and the cluster noise for every D and keeps or drops D = 8 with equal probability.

- `w_extrapolation.csv`: per `D`, the gauge weight `w` and the bootstrap estimate `E_per_site` with
  its error `E_err` (orange markers; the same as in Fig. S9).
- `bootstrap_lines.csv`: slope `c` and intercept `E0` (per site) of the 100 fitted lines shown.
- `extrapolated_energy.csv`: mean `E0_per_site`, standard deviation `E0_err` and the 16th and 84th
  percentiles of E0 over all `n_samples` = 10,000 bootstrap samples (pink marker).

The result, E/N = -0.7602(14), is the "fPEPS (extrapolated)" row of Table S4 and the estimated
exact energy of Fig. 3(a).
