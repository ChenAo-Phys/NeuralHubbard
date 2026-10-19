# Fig. S9: convergence of the pure fPEPS energy

E/N of the simple-update fPEPS on the 16x16 lattice at t' = 0 (period-8 stripe initial state,
bond dimension D = 8-16), evaluated by cluster-boundary contraction (SM Sec. VI A). The states are
not released.

- `cluster_boundary.csv`: `E_per_site` per `D`, cluster radius `r` and boundary bond dimension
  `chi`. D = 10 and 14 have only r >= 4 and chi >= 128.
- `full_boundary.csv`: `E_per_site` per `D` and `chi` from the full periodic boundary
  contraction, shown for comparison only.
- `chi_fits.csv`: the fits E(chi) = `E_inf` + `a` chi^-`p` (SM Eq. (S24)), p = 1 and 2, to the
  r = 7 energies at the three `chi_fitted`.
- `per_D_estimates.csv`: the bootstrap estimate `E_per_site` and its error `E_err` per `D`
  (orange markers), and the cluster noise `sigma_r` = |E(r = 7) - E(r = 6)| at chi = 384 (SM
  Eq. (S25)).

Errors are standard deviations over bootstrap samples (SM Sec. VI B). The D = 16 estimate,
E/N = -0.7545(15), is the "fPEPS (D = 16)" row of Table S4.
