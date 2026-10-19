# Table S1: networks

- `network_settings.json`: the content of Table S1 for the `bottleneck` CNN (NN-fPEPS) and the
  `normal` CNN (BCS and Pfaffian backflows) under `networks`, and the settings common to both under
  `shared`.
- `parameter_counts.csv`: the number of float32 parameters `N_p` in the released L x L checkpoints
  `weights/<state>.eqx` and in the benchmark states of Figs. S4 and S5 (`benchmark_*`, not
  released). It is split into the CNN (`N_p_cnn` = `embedding` + `conv_weights` + `conv_biases`),
  the BCS backflow `head`, the pairing weights `W`, `scalars`, and zero `padding` that rounds N_p up
  to a multiple of 4096.

All production NN-fPEPS, the 16x4 one included, have N_p = 4,470,784; the backflows have
1.19-1.28 M, growing with N through W. The Fig. S5 NN-fPEPS has a per-site embedding
(N_p = 4,599,808). The weights per block in Table S1 count convolution kernels only: 2Cc + 9c^2
(bottleneck) and 2 x 9C^2 (normal).
