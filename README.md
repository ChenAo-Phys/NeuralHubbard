# Extreme competition between stripes and superconductivity in the hole-doped Hubbard model

Ao Chen, Johnnie Gray, Si-Jing Du, Garnet Kin-Lic Chan (California Institute of Technology)

**Paper:** arXiv: TBD

We study the two-dimensional Hubbard model at U = 8 and 1/8 hole doping, with t' = 0 and
t' = -0.2, using three neural-network variational wavefunctions: a BCS backflow, a Pfaffian
backflow and a neural-network fermionic PEPS (NN-fPEPS), on periodic lattices up to 32 x 32.
Uniform d-wave superconductivity, superconducting stripes and non-superconducting stripes turn
out to be nearly degenerate on the milli-t energy scale. This repository provides the trained
states, the code to evaluate and measure them, a demo trainer, and the data and scripts behind
the figures and tables of the paper and its Supplementary Materials (SM).

## Layout

```
NeuralHubbard/
├── requirements.txt   pinned Python environment
├── weights/           the 11 released states: parameters and equilibrated Monte Carlo chains
├── code/              models, loaders, measurement and training scripts, figure scripts
├── data/              data behind the figures and tables, one folder each
└── images/            figures in the paper
```

Details are in [weights/README.md](weights/README.md), [code/README.md](code/README.md) and
[data/README.md](data/README.md).

## Installation

Linux, Python 3.12 and an NVIDIA A100 or newer GPU. The default settings assume 80 GB of GPU memory, see
[code/README.md](code/README.md#demo-scripts). The figure scripts need a LaTeX installation.

```bash
python3.12 -m venv neuralhubbard && source neuralhubbard/bin/activate
pip install -r requirements.txt
```

- **NN-fPEPS:** the first import runs GPU self-tests for 4-15 minutes, see
  [code/README.md](code/README.md#nn-fpeps-module).

## Usage

Load a state and evaluate its amplitudes, from the `code/` folder:

```python
import states

state, H, info = states.load_state("tpm02_L16_pfaffian")  # quantax Variational, Hamiltonian, metadata
spins = states.load_spins(info.name)  # equilibrated chains, int8 array of shape (8192, 2N)
psi = state(spins[:16])               # projected amplitudes
print(psi.logabs, psi.sign)
```

The scripts run from the repository root:

```bash
# amplitudes and a small-sample energy of a released state
python code/quick_start.py tpm02_L16_pfaffian

# energy, correlations and order parameters (by default 64 sweeps of 8192 samples)
python code/measure_orderparams.py --state tpm02_L16_pfaffian --out runs/tpm02_L16_pfaffian

# training: 5000 unprojected steps from a cold start, then 1000 symmetry-projected steps
python code/train_pfaffian_16x16_tpm02.py --outdir runs/stage1
python code/train_pfaffian_16x16_tpm02.py --symm --init runs/stage1/params_4999.eqx --outdir runs/stage2

# figures (written to code/plots/output/)
python code/plots/order_params.py
```

[code/README.md](code/README.md) documents all scripts and options, including multi-GPU runs.

## Released states

`weights/` holds 11 symmetry-projected states, named `<tp0|tpm02>_L<L>_<bcs|pfaffian|nnfpeps>`
for t' = 0 or -0.2 on an L x L lattice, plus `tp0_16x4_nnfpeps`. Their training steps,
parameter counts and energies are listed in [weights/README.md](weights/README.md). The
t' = -0.2 BCS backflow of the paper is from Rende et al. (arXiv:2608.12465) and is not included.

## Conventions

- Hubbard model with t = 1 and U = 8 on periodic lattices, N_up = N_dn = 7N/16. Energies are in
  units of t.
- A configuration is an int8 array of length 2N with +1 (occupied) or -1 (empty): the N
  spin-up occupations, then the N spin-down occupations. Site (x, y) has index `x * Ly + y`.
- All states are projected onto the trivial representation of
  `Flip(axis=0) @ Flip(axis=1) @ SpinInverse()` (8 elements); translation invariance is built
  into the networks. `states.load_state(name, symm=False)` gives the unprojected network.
- Amplitudes are `quantax.utils.LogArray`s with log|psi| far outside the float32 range: read
  `psi.logabs` and `psi.sign`.
- quantax holds one lattice per process, so load one state per process.
- The code runs in float32 with TF32 matrix products (the JAX default), so log|psi| agrees
  only to about 1e-2 between devices and compiled programs.

## Citation

```bibtex
@article{Chen2026ExtremeCompetition,
  title   = {Extreme competition between stripes and superconductivity in the hole-doped Hubbard model},
  author  = {Chen, Ao and Gray, Johnnie and Du, Si-Jing and Chan, Garnet Kin-Lic},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).

## Acknowledgements

The code builds on [quantax](https://github.com/ChenAo-Phys/quantax),
[JAX](https://github.com/jax-ml/jax), [Equinox](https://github.com/patrick-kidger/equinox),
[quimb](https://github.com/jcmgray/quimb), [symmray](https://github.com/jcmgray/symmray) and
[fermix](https://github.com/ChenAo-Phys/fermix).

This work was supported by the U.S. Department of Energy, Office of Science, through Award
No. DE-SC0018140. Most computations were performed at the National Energy Research Scientific
Computing Center (NERSC), a U.S. Department of Energy Office of Science User Facility at
Lawrence Berkeley National Laboratory, under NERSC Award No. ERCAP0023924. Some computations
were performed at the Caltech Resnick High Performance Computing Center. We thank Riccardo
Rende and Antoine Georges for useful discussions about their earlier results. We acknowledge
technical support from NVIDIA Corporation, and we thank Linsey Rodenbach, Nicola Pancotti and
other members of the NVIDIA Quantum Applied Research team, as well as Katie Klymko (NERSC), for
supporting this project. Claude Code (Anthropic) was used to write parts of the simulation
code, to submit and monitor part of the computations, and to draft parts of the manuscript.
The authors reviewed and verified all code, computations and text.
