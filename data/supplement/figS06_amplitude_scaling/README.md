# Fig. S6: time per amplitude

`amplitude_time.csv`: time per amplitude in microseconds on one A100 GPU, for L x L lattices
(N = L^2) at half filling, as printed in Table S3. `determinant_us` and `pfaffian_us` are
N x N matrices from the fermix benchmarks, `fpeps_us` is the fPEPS with D = chi = 4, and
`fpeps_trans_us` covers all L^2 translations in one call. These are the `Determinant`, `Pfaffian`,
`fPEPS` and `fPEPS (trans)` rows of [Table S3](../tableS3_amplitude_cost/). The gray guide lines
of the figure are 2.8e-7 N^3, 0.019 N and 0.075 N ln N microseconds.
