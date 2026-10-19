# Data

The data behind the figures and tables of the paper and its Supplementary Materials (SM), one
folder per item. Each folder README describes its columns; Figs. S12-S21 share
[`supplement/real_space_corr/README.md`](supplement/real_space_corr/README.md). The figures
themselves are in [`images/`](../images/).

| item | folder | content |
| --- | --- | --- |
| Fig. 3 | [`fig3_order_params_tp0/`](fig3_order_params_tp0/) | t' = 0: E/N against L, C_p(r) on 16x16, Delta_SC^2 and M^2_str against 1/L |
| Fig. 4 | [`fig4_order_params_tpm02/`](fig4_order_params_tpm02/) | the same at t' = -0.2 |
| Table S1 | [`supplement/tableS1_network/`](supplement/tableS1_network/) | CNN settings and exact parameter counts |
| Fig. S4 | [`supplement/figS04_optimizer_benchmark/`](supplement/figS04_optimizer_benchmark/) | training curves of SR, SPRING, MARCH and AdamSR |
| Fig. S5 | [`supplement/figS05_solver_combined/`](supplement/figS05_solver_combined/) | Cholesky, CG and LSMR on frozen SR systems |
| Fig. S6 | [`supplement/figS06_amplitude_scaling/`](supplement/figS06_amplitude_scaling/) | time per amplitude against N |
| Table S3 | [`supplement/tableS3_amplitude_cost/`](supplement/tableS3_amplitude_cost/) | time, operation count and MFU of each amplitude component |
| Fig. S7 | [`supplement/figS07_vmc_scaling/`](supplement/figS07_vmc_scaling/) | weak and strong scaling of one VMC step |
| Fig. S9 | [`supplement/figS09_fpeps_L16_convergence/`](supplement/figS09_fpeps_L16_convergence/) | pure fPEPS on 16x16 at t' = 0: E/N against chi, r and D |
| Fig. S10 | [`supplement/figS10_fpeps_L16_w_extrap/`](supplement/figS10_fpeps_L16_w_extrap/) | extrapolation of the pure fPEPS energy in the gauge weight |
| Fig. S11 | [`supplement/figS11_zero_variance_extrapolation/`](supplement/figS11_zero_variance_extrapolation/) | zero-variance extrapolation on 16x4 |
| Table S4 | [`supplement/tableS4_energy_benchmarks/`](supplement/tableS4_energy_benchmarks/) | all rows, per-sweep energy series and training traces |
| Figs. S12-S21 | [`supplement/real_space_corr/`](supplement/real_space_corr/) | real-space C_s, C_c and C_p of each projected state, one subfolder per figure |

Figs. 1, 2, S1-S3 and S8 are schematics and Table S2 is text, so they have no data folder.

## Conventions

- Model, units and projection are as in the [top-level README](../README.md#conventions).
  Lattices are L x L periodic with N = L^2 and 1/8 hole doping unless a folder README says
  otherwise (16x4 in Fig. S11 and Table S4, half filling in Table S3 and Fig. S6).
- `E_per_site` is E/N and `VarE_per_site` is (<H^2> - <H>^2)/N; files with total energies say
  so. A `state` column names the released weights in `weights/`, except the
  unreleased checkpoints `tp0_L16_bcs_step3999` (Fig. 3) and `benchmark_*` (Table S1).
- Displacements r = (rx, ry) are taken mod L, and |r| is the minimum-image length. C_p,
  Delta_SC^2 and M^2_str follow main-text Eqs. (2)-(4), C_s and C_c SM Eq. (S28).
- Each Monte Carlo sweep averages over 8192 independent chains. Errors are one standard error.
  Unless a folder README says otherwise, they come from a block jackknife over sweeps: the largest
  over block sizes 1, 2, 4 and 8 that leave at least 8 blocks. "As printed" columns are rounded as
  in the paper.

## File formats

CSV files have one header row, sometimes after `#` comment lines, and quote text fields that
contain commas. Read them with the standard library:
`rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))`.
NPZ files load with `numpy.load(path)` (no pickle needed). The figures are regenerated from these
files by the scripts in [`code/plots/`](../code/plots/) (see [`code/README.md`](../code/README.md)).
