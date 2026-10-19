# Fig. 3: energies and order parameters at t' = 0

| file | panel | content |
| --- | --- | --- |
| `panel_a_energy.csv` | (a) | E/N and sigma^2/N per method and L; `plotted` and `err_drawn` say what the figure draws |
| `panel_b_pair_corr_vectors.csv` | (b) | C_p for every displacement (`dx`, `dy`) of the 16x16 lattice, with leave-one-sweep-out jackknife errors |
| `panel_b_pair_corr_shells.csv` | (b) | the drawn points: mean C_p over the `n_vectors` displacements of equal \|r\|, error sqrt(sum err^2) / n_vectors |
| `panel_b_plateaus.csv` | (b) | dashed lines: `Delta_SC2`, the mean C_p over the `n_vectors` displacements with \|r\| >= `r_max` = L sqrt(2) / 4 |
| `panel_c_orders.csv` | (c) | `Delta_SC`, `Delta_SC2` (the figure shows 10 Delta_SC^2) and `M2_str` at wavevector `q` |
| `panel_c_fits.csv` | (c) | weighted fits y = a + b / L^p over `L_used`: `intercept` = a, `intercept_err_drawn` scaled by sqrt(chi2/dof) when that exceeds 1, `Delta_SC_TDL` = sqrt(intercept) |

Literature: ACE, SCALE (`Gu_PRX-Intelligence_SCALE`) and HFPS (`Roth_arxiv25_HubbardSC`).
`PEPS (extrapolated)` is the pure fPEPS estimate of Fig. S10. The BCS backflow M2_str is fitted in
1/L^2, all others in 1/L. The 16x16 BCS backflow energy is measured on the released weights (step
4999), its C_p, Delta_SC and M2_str on step 3999 (`tp0_L16_bcs_step3999`, Delta_SC error from block
size 16). The 32x32 NN-fPEPS C_p was not measured.
