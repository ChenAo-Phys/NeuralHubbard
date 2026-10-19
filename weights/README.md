# Released weights

The eleven symmetry-projected states of the paper, for the Hubbard model with U = 8 at 1/8 hole
doping on periodic lattices. Each state has two files:

- `<name>.eqx`: the parameters, saved with `equinox.tree_serialise_leaves`. The network is
  built by `code/states.py`, which also checks the md5 checksum of the file.
- `<name>_spins.npz`: 8192 Monte Carlo chains equilibrated under the projected state, stored as
  `bits = np.packbits(spins > 0, axis=-1)` and `shape = (8192, 2N)`. Start sampling from them,
  as `states.make_sampler` does: the striped states thermalize slowly.

Load them from the `code/` folder with `states.load_state(name)` and `states.load_spins(name)`
(see the [top-level README](../README.md#usage)).

| name | ansatz | lattice | t' | training steps | parameters | E/N | sigma^2/N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tp0_16x4_nnfpeps` | NN-fPEPS | 16x4 | 0 | 8500 + 2000 | 4,470,784 | -0.766455(14) | 0.00555(9) |
| `tp0_L16_bcs` | BCS backflow | 16x16 | 0 | 10000 + 5000 | 1,208,320 | -0.75878(2) | 0.0255(20) |
| `tp0_L16_pfaffian` | Pfaffian backflow | 16x16 | 0 | 5000 + 1000 | 1,208,320 | -0.75813(1) | 0.0259(2) |
| `tp0_L16_nnfpeps` | NN-fPEPS | 16x16 | 0 | 5000 + 10000 | 4,470,784 | -0.75851(2) | 0.0256(6) |
| `tp0_L32_bcs` | BCS backflow | 32x32 | 0 | 1000 + 766 | 1,257,472 | -0.75617(1) | 0.046(10) |
| `tp0_L32_nnfpeps` | NN-fPEPS | 32x32 | 0 | 1300 + 800 | 4,470,784 | -0.75677(1) | 0.0346(4) |
| `tpm02_L16_pfaffian` | Pfaffian backflow | 16x16 | -0.2 | 5000 + 1000 | 1,208,320 | -0.74533(1) | 0.0272(5) |
| `tpm02_L16_nnfpeps` | NN-fPEPS | 16x16 | -0.2 | 4700 + 4450 | 4,470,784 | -0.74467(2) | 0.0367(31) |
| `tpm02_L24_pfaffian` | Pfaffian backflow | 24x24 | -0.2 | 5000 + 896 | 1,241,088 | -0.74585(1) | 0.0257(3) |
| `tpm02_L32_pfaffian` | Pfaffian backflow | 32x32 | -0.2 | 903 + 378 | 1,282,048 | -0.74569(1) | 0.0294(5) |
| `tpm02_L32_nnfpeps` | NN-fPEPS | 32x32 | -0.2 | 1200 + 250 | 4,470,784 | -0.74257(2) | 0.0502(7) |

- **Training steps:** unprojected + projected optimization steps on the final lattice. Except
  for the 16x16 backflows and the 16x4 NN-fPEPS, each state was warm-started from a smaller
  lattice, as described in the SM.
- **E/N and sigma^2/N**: the projected values of
  SM Table S4, measured with 8192 samples per sweep.
