# Table S3: cost of one amplitude

`amplitude_cost.csv`: one row per component and lattice size L = 8, 16, 32, 64 (N = L^2, half
filling), on one A100 GPU.

- `component`: `Determinant`, `Pfaffian`, `fPEPS`, `fPEPS (trans)`, `Bottleneck CNN` or
  `Normal CNN`. `subcomponent` (`HOTRG steps`, `Final exact contraction`) marks the gray fPEPS rows.
- As printed: time per configuration `t_us` (microseconds), operation count `F_MFLOP`, throughput
  `F_over_t_TFLOPs`, `arithmetic` and `MFU_percent`. `t_us_unrounded` and `F_MFLOP_unrounded` are
  before rounding.
- `peak_TFLOPs`: the MFU denominator, 156 (TF32), 19.5 (FP32) or 52 (3xTF32, three TF32 products
  per FP32 product); for `mixed` rows, the time-weighted peak of the two sub-rows.
- `batch`: configurations per call. `source`: `this work`, or `fermix library benchmark` for the
  determinant and Pfaffian ([fermix](https://github.com/ChenAo-Phys/fermix)).
