# Table S4: energy benchmarks

E/N and sigma^2/N = (<H^2> - E^2)/N, in units of t and t^2, without and with symmetry projection.

- `tableS4.csv`: all 32 rows as printed, with errors in parentheses. Literature rows carry a
  `citation_key`; `training_average` lists the dagger cells (averages over optimization steps);
  `data_unprojected` and `data_projected` name the file behind each of our values. Keys not
  explained in the Fig. 3 and 4 READMEs: `Chen_PNAS26_HFPS` (Chen et al., PNAS 123, e2535288123
  (2026)), `Gu_NC26_NQS-Hubbard` (Gu et al., Nat. Commun. 17, 7838 (2026)), `Loehr_arxiv25_NNBF`
  (Loehr and Clark, arXiv:2510.26906), `Sorella_PRB23_VAFQMC` (Sorella, PRB 107, 115133 (2023)),
  `Wu_Science24_Vscore` (Wu et al., Science 386, 296 (2024)).
- `energy_series/<state>.npz`: per-sweep means `E_mean` and `E_sqr` of the local energy and its
  square (total, not per site). E/N = mean(E_mean) / N and sigma^2/N = (mean(E_sqr) -
  mean(E_mean)^2) / N, with block-jackknife errors as in `code/measure/assemble.py`.
  `*_unprojected` are the unreleased parent runs of the projected states.
- `training_traces/<state>.csv`: total `E` and `VarE` of the training batch per `step` over the
  averaged window; the dagger value is the mean over the file divided by N.

The extrapolated 16x4 row comes from [Fig. S11](../figS11_zero_variance_extrapolation/), and the
pure fPEPS rows from [Fig. S9](../figS09_fpeps_L16_convergence/) and
[Fig. S10](../figS10_fpeps_L16_w_extrap/). The unprojected 16x16 NN-fPEPS energies at t' = 0 and
-0.2 were rounded twice, so they differ from the energy series by one in the last digit.
