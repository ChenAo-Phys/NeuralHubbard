# Fig. S7: scaling of one VMC step

One optimization step of the 16x16 NN-fPEPS at t' = -0.2 (N_p = 4,470,784, 8 symmetry images per
amplitude, LSMR fixed to 100 iterations), on nodes with 4 A100 GPUs each.

- `weak_scaling.csv`: 256 samples per GPU, 1-128 nodes.
- `strong_scaling.csv`: N_s = 8192 samples, 8-128 nodes.

Times are in seconds, averaged over `n_timed_steps` steps after one warm-up step: sampling
(`t_sampling_s`), local energies (`t_Eloc_s`), Jacobian (`t_jacobian_s`), the two LSMR solves of
AdamSR (`t_solver_s`) and the whole step (`t_total_s`, which includes the unplotted `t_update_s`);
`t_total_std_s` is the spread over steps. Throughput `F_over_t_PFLOPs` = N_s x 3.4975e13 FLOP /
t_total, and `MFU_percent` = (F/t) / (gpus x 125.8 TFLOP/s), with 125.8 TFLOP/s the
time-weighted peak of the 16x16 `fPEPS (trans)` and `Bottleneck CNN` rows of Table S3 (cf. SM
Eq. (S23)).
