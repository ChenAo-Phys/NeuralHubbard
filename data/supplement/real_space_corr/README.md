# Figs. S12-S21: real-space correlations

The folders `figS12_*` to `figS21_*` share one format. Each holds the translation-averaged
correlations of one projected state, named by the folder and by `state` in `maps.npz`: the spin
correlation C_s(r) = <S_0 . S_r>, the charge correlation C_c(r) = <(n_0 - n_bar)(n_r - n_bar)> with
n_bar = 7/8, and the pair correlation C_p(r) of main-text Eq. (2).

- `cuts.csv`: the plotted cuts for r = 0 ... L/2, columns `Cs_x`, `Cs_x_err`, ..., `Cp_y_err`, with
  `*_x` at r = (r, 0) (panel a) and `*_y` at r = (0, r) (panel b). The figures show C_s, 50 C_c and
  100 C_p; the stored values are unscaled.
- `maps.npz`: the full maps `Cs`, `Cc`, `Cp` and their `*_err`, shape (L, L) indexed by r = (rx, ry)
  and symmetrized as [C(r) + C(-r)] / 2; `Delta_SC2_plateau`, the mean of `Cp` over |r| >= `r_max`;
  and the measurement settings (`n_sweeps_*`, `cc_source`, `swap_xy`, ...).
- `series.npz`: per-sweep `Cs`, and `Cc` where it was measured per sweep, shape (n_sweeps, L, L).

C_s and C_c errors are standard errors over blocks of 4 sweeps, or over configurations when C_c comes
from a snapshot of configurations (`cc_source` = `snapshot`); C_p errors are leave-one-sweep-out
jackknife errors. NaN marks a quantity that was not measured.

## Folders

| figure | folder | state | notes |
| --- | --- | --- | --- |
| S12 | `figS12_real_space_corr_tp0_16x16_bcs` | BCS backflow, 16x16, t' = 0 | measured at step 3999 of the run released as `tp0_L16_bcs` (step 4999); C_c from a snapshot of 8192 configurations |
| S13 | `figS13_real_space_corr_tp0_16x16_pfaffian` | Pfaffian backflow, 16x16, t' = 0 | three independent runs combined: 128 sweeps for C_s and C_p, 256 for C_c |
| S14 | `figS14_real_space_corr_tp0_16x16_nnfpeps` | NN-fPEPS, 16x16, t' = 0 | |
| S15 | `figS15_real_space_corr_tp0_32x32_bcs` | BCS backflow, 32x32, t' = 0 | |
| S16 | `figS16_real_space_corr_tp0_32x32_nnfpeps` | NN-fPEPS, 32x32, t' = 0 | C_p not measured; C_c from snapshots of 2 x 8192 configurations |
| S17 | `figS17_real_space_corr_tpm02_16x16_pfaffian` | Pfaffian backflow, 16x16, t' = -0.2 | two independent runs combined: 128 sweeps for C_s and C_p, 96 for C_c |
| S18 | `figS18_real_space_corr_tpm02_16x16_nnfpeps` | NN-fPEPS, 16x16, t' = -0.2 | x and y swapped in the figure (see below); C_c from a snapshot of 8192 configurations |
| S19 | `figS19_real_space_corr_tpm02_24x24_pfaffian` | Pfaffian backflow, 24x24, t' = -0.2 | |
| S20 | `figS20_real_space_corr_tpm02_32x32_pfaffian` | Pfaffian backflow, 32x32, t' = -0.2 | |
| S21 | `figS21_real_space_corr_tpm02_32x32_nnfpeps` | NN-fPEPS, 32x32, t' = -0.2 | x and y swapped in the figure (see below) |

For Figs. S18 and S21 (`swap_xy` = True), the figure and `cuts.csv` exchange x and y so that the
stripes are modulated along x; `maps.npz` and `series.npz` keep the lattice frame.
