# Code

Run the scripts from the repository root after installing `requirements.txt` (see the
[top-level README](../README.md)). The module docstrings document the functions and formulas.

| file or folder | content |
| --- | --- |
| `states.py` | registry and loaders of the released states: `load_state`, `load_spins`, `make_sampler` |
| `quick_start.py` | load a state, print amplitudes and a small-sample energy |
| `measure_orderparams.py` | Monte Carlo measurement of the energy, correlations and order parameters |
| `train_pfaffian_16x16_tpm02.py` | demo trainer of the 16x16, t' = -0.2 Pfaffian backflow |
| `distributed.py` | multi-process JAX setup behind `--distributed` |
| `backflow/` | BCS and Pfaffian backflow networks with Pallas kernels for the pairing matrix |
| `nnfpeps/` | NN-fPEPS network, HOTRG contraction and Pallas kernels |
| `measure/` | packed estimators of the correlation functions and CPU assembly of the results |
| `plots/` | figure scripts; they read only `data/` |

## Demo scripts

```bash
python code/quick_start.py [STATE] [--nconfigs 16]
```

Loads a state (default `tpm02_L16_pfaffian`), prints the projected log|psi| and sign of the
first stored configurations and compares a small-sample E/N with the paper value.

```bash
python code/measure_orderparams.py --state STATE --out DIR [--nsamples 8192] [--nsweeps 64] \
    [--thermal X] [--sweep-steps X] [--channels energy,spin,charge,dwave] [--distributed]
```

Samples the state from its released chains and measures E/N, sigma^2/N, the spin, charge and
pair correlations, Delta_SC and M^2_str. `--thermal` and `--sweep-steps` (units of N Metropolis
steps) default to the paper's values for the state. `DIR/series.npz` holds the per-sweep data,
saved periodically so that an interrupted run is usable, and `DIR/results.json` and
`DIR/results.npz` the results; `cd code && python -m measure.assemble ../DIR/series.npz`
re-assembles them. For `tp0_16x4_nnfpeps` use `--channels energy`.

```bash
python code/train_pfaffian_16x16_tpm02.py [--outdir runs/pfaffian_16x16_tpm02] [--nsamples 1024] \
    [--steps N] [--symm] [--init PARAMS.eqx] [--init-spins SPINS] [--c0 1.0] [--distributed]
```

The recipe of `tpm02_L16_pfaffian`: 5000 unprojected steps from a cold start, then
`--symm --init <stage-1 outdir>/params_4999.eqx` for 1000 projected steps. The paper used 8192
samples; with 1024, a step takes about 1 minute (6 minutes with `--symm`) and 23 GiB on one
A100, but training is slower: after 300 steps from a cold start E/N is about 0.01 higher than
with 8192 samples, which follow the paper's training curve. Rerunning the same command resumes
from the newest checkpoint in `--outdir`. To continue from the released state, use `--symm
--init weights/tpm02_L16_pfaffian.eqx --init-spins weights/tpm02_L16_pfaffian_spins.npz` and
a lower step-norm cap, e.g. `--c0 0.1`.

One process uses all GPUs it sees. For several nodes, launch one process per node (e.g. with
`srun`) and add `--distributed`; the measurement initializes this automatically when
`SLURM_NTASKS > 1`. The default forward chunks (`max_parallel` in `states.py`, also an argument
of `load_state`) are sized for one 80 GB GPU; lower them on smaller GPUs.

## Figures

Each script in `plots/` writes PDF files to `code/plots/output/`, or to `--outdir DIR`.
`order_params.py` makes Figs. 3 and 4 and `real_space_corr.py` Figs. S12-S21 (select some with
e.g. `S12 S21`). The other scripts are named after the data folder of their figure (Figs. S4-S7
and S11); `zero_variance_extrapolation.py` also prints the extrapolated E0/N.

## NN-fPEPS module

- The modules in `nnfpeps/` import each other by top-level name; `states.py` adds the folder to
  `sys.path`.
- The first import runs GPU self-tests for 4-15 minutes without output and stores pass markers
  in `FPEPS_CACHE_DIR` (default `~/.cache`); run it once in a single process before a multi-GPU
  job. A failed test warns and falls back to the slower reference contraction. Warnings of
  cotengra about missing `kahypar` or `optuna` are harmless.
- The import calls `jax.devices()`, so multi-process scripts must call
  `distributed.initialize()` before loading an NN-fPEPS state.
- `FPEPS_SKIP_SELFTEST=1` skips the tests and `FPEPS_REFERENCE=1` disables the fast kernels. On
  a CPU, jax 0.11 needs `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=`.

## Model details not spelled out in the paper

- BCS backflow: an extra trainable pairing term delta_d F^d (F^d = -1 on x bonds, +1 on y
  bonds), a learned linear head to the 64 features per spin, and no bias in the last convolution.
- The backflow checkpoints contain an unused zero vector that rounds the parameter count up to a
  multiple of 4096 (included in Table S1).
- NN-fPEPS: the HOTRG path (`yy` at 16x16, `yyy` at 32x32, none at 16x4, contracted exactly) is
  part of the state but not stored in the `.eqx` file; `build_nnfpeps` sets it.
- `tp0_L16_bcs` was trained with the einsum pairing matrix, `load_state(name, use_kernel=False)`;
  the default kernel path changes its log|psi| by about 1e-3.
