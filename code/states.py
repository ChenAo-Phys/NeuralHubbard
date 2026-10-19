r"""
Registry and loaders of the released variational states.

All states are symmetry-projected states of the Hubbard model with :math:`U = 8`, :math:`t = 1`
and :math:`t' = 0` (``tp0``) or :math:`-0.2` (``tpm02``) on periodic lattices at 1/8 hole
doping. Names are ``<tp0|tpm02>_L<L>_<bcs|pfaffian|nnfpeps>``, plus ``tp0_16x4_nnfpeps``.

.. code-block:: python

    import states  # from the code/ folder

    state, H, info = states.load_state("tpm02_L16_pfaffian")
    spins = states.load_spins(info.name)  # equilibrated chains, int8 (8192, 2N)
    psi = state(spins[:16])  # projected amplitudes: psi.logabs, psi.sign

quantax allows one lattice per process, so load one state per process. A configuration is a
``+-1`` array of length 2N, spin up then spin down, site index ``x * Ly + y``. The NN-fPEPS
states need an NVIDIA GPU, and their first import takes several minutes (see ``README.md``).
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "weights")
NNFPEPS_DIR = os.path.join(CODE_DIR, "nnfpeps")

U = 8.0


@dataclass(frozen=True)
class StateInfo:
    """
    Metadata of a released state.

    :param name: Registry key, also the stem of ``weights/<name>.eqx``.
    :param ansatz: ``"bcs"``, ``"pfaffian"`` or ``"nnfpeps"``.
    :param L: Extent along x.
    :param tprime: Next-nearest-neighbour hopping, 0 or -0.2.
    :param n_up: Number of spin-up electrons.
    :param n_dn: Number of spin-down electrons.
    :param weights: File name in the weights folder.
    :param md5: md5 checksum of the weight file.
    :param nparams: Number of float32 parameters, zero padding included.
    :param contract_path: HOTRG contraction path of NN-fPEPS, ``""`` for an exact
        contraction, None for the backflows.
    :param sampler: Monte Carlo settings of the paper: ``ph_neighbor`` and ``se_neighbor``
        (``n_neighbor`` of ``ParticleHop`` and ``SiteExchange``), and ``sweep``, ``thermal``
        and ``train_sweep`` (Metropolis steps per chain between samples, for thermalization
        and between optimization steps, in units of N).
    :param paper: Projected E/N and sigma^2/N of Table S4, keys ``E_per_site``, ``E_err``,
        ``var_per_site``, ``var_err``.
    :param notes: Remarks on training and measurement.
    :param max_parallel: Default forward chunk per device, in configurations before the
        symmetry expansion.
    :param Ly: Extent along y, None for an ``L x L`` lattice.
    """

    name: str
    ansatz: str
    L: int
    tprime: float
    n_up: int
    n_dn: int
    weights: str
    md5: str
    nparams: int
    contract_path: Optional[str]
    sampler: dict = field(default_factory=dict)
    paper: dict = field(default_factory=dict)
    notes: str = ""
    max_parallel: int = 64
    Ly: Optional[int] = None

    @property
    def shape(self) -> tuple[int, int]:
        """Lattice extents ``(Lx, Ly)``."""
        return (self.L, self.L if self.Ly is None else self.Ly)

    @property
    def N(self) -> int:
        """Number of lattice sites."""
        Lx, Ly = self.shape
        return Lx * Ly


# n_neighbor of the samplers: nearest neighbours, or nearest and next-nearest
_NN = 1
_NNN = (1, 2)

# fmt: off
# name, ansatz, L or (Lx, Ly), t', md5, nparams, contract path, max_parallel,
# (ParticleHop, SiteExchange) n_neighbor, (measurement sweep, thermalization,
# optimization sweep) in units of N, paper (E/N, error, sigma^2/N, error)
_TABLE = [
    ("tp0_L16_bcs", "bcs", 16, 0.0, "cdfdb8e6065b904232bf10dec6ca2227", 1_208_320, None, 512,
     (_NN, _NN), (4, 50, 2), (-0.75878, 2e-5, 0.0255, 0.0020)),
    ("tp0_L16_pfaffian", "pfaffian", 16, 0.0, "17b770fe57367b09b27ad37538842ba0", 1_208_320, None, 512,
     (_NN, _NN), (4, 50, 2), (-0.75813, 1e-5, 0.0259, 0.0002)),
    ("tp0_L16_nnfpeps", "nnfpeps", 16, 0.0, "fd97581ed9cbdc39f2f9b40b0082d600", 4_470_784, "yy", 128,
     (_NN, _NN), (4, 50, 2), (-0.75851, 2e-5, 0.0256, 0.0006)),
    ("tp0_L32_bcs", "bcs", 32, 0.0, "e45a997c3202fae9df7c4f38bd7e6acc", 1_257_472, None, 128,
     (_NN, _NN), (5, 40, 4), (-0.75617, 1e-5, 0.046, 0.010)),
    ("tp0_L32_nnfpeps", "nnfpeps", 32, 0.0, "219b2bc18b11c6ad6fca7fc03e44f054", 4_470_784, "yyy", 64,
     (_NN, _NN), (4, 4, 4), (-0.75677, 1e-5, 0.0346, 0.0004)),
    ("tpm02_L16_pfaffian", "pfaffian", 16, -0.2, "aef48e8b7c10cfa183f8c0e8dcf84d06", 1_208_320, None, 512,
     (_NNN, _NNN), (4, 50, 3), (-0.74533, 1e-5, 0.0272, 0.0005)),
    ("tpm02_L16_nnfpeps", "nnfpeps", 16, -0.2, "d8f729c1470b5ad6e207c87e1bef2d32", 4_470_784, "yy", 128,
     (_NN, _NNN), (4, 20, 2.5), (-0.74467, 2e-5, 0.0367, 0.0031)),
    ("tpm02_L24_pfaffian", "pfaffian", 24, -0.2, "17653481d380416e6df7033ba7902aee", 1_241_088, None, 256,
     (_NNN, _NNN), (4, 50, 3), (-0.74585, 1e-5, 0.0257, 0.0003)),
    ("tpm02_L32_pfaffian", "pfaffian", 32, -0.2, "83a4eb0c591b893e29b451c7b95cb0b1", 1_282_048, None, 128,
     (_NNN, _NNN), (4, 40, 4), (-0.74569, 1e-5, 0.0294, 0.0005)),
    ("tpm02_L32_nnfpeps", "nnfpeps", 32, -0.2, "383d3ff4d431d4aea0f52c72ee35216f", 4_470_784, "yyy", 64,
     (_NNN, _NNN), (4, 40, 4), (-0.74257, 2e-5, 0.0502, 0.0007)),
    ("tp0_16x4_nnfpeps", "nnfpeps", (16, 4), 0.0, "cd7a2e22ace25a3adeeb00f0dcfaf011", 4_470_784, "", 256,
     (_NN, _NN), (2, 50, 2), (-0.766455, 1.4e-5, 0.00555, 9e-5)),
]
# fmt: on

_NOTES = {
    "tp0_L16_bcs": "Step 4999 of the projected optimization, trained on the einsum path "
    "(use_kernel=False). The paper's order parameters and correlations were measured on "
    "step 3999 of the same run.",
    "tp0_L16_nnfpeps": "Warm start from a 16x8 NN-fPEPS.",
    "tp0_L32_bcs": "Warm start from the unprojected 16x16 BCS backflow, W tiled to 32x32.",
    "tp0_L32_nnfpeps": "Warm start from the unprojected 16x16 NN-fPEPS. The paper's "
    "measurement started from the optimization chains, hence the short thermalization.",
    "tpm02_L16_nnfpeps": "Warm start from a 16x8 NN-fPEPS.",
    "tpm02_L24_pfaffian": "Warm start from the unprojected 16x16 Pfaffian backflow, W "
    "extended to 24x24.",
    "tpm02_L32_pfaffian": "Warm start from the unprojected 24x24 Pfaffian backflow, W "
    "extended to 32x32.",
    "tpm02_L32_nnfpeps": "Warm start from the unprojected 16x16 NN-fPEPS. The paper's "
    "values use the first 32 of 36 measurement sweeps.",
    "tp0_16x4_nnfpeps": "Contracted exactly (contract_path ''). Trained unprojected for "
    "8000 steps with a site-dependent embedding and 500 steps with the embedding averaged "
    "over the sites, then 2000 steps under the projection. E/N and sigma^2/N measured "
    "with measure_orderparams.py (64 sweeps of 8192 samples).",
}


def _make_info(row: tuple) -> StateInfo:
    name, ansatz, L, tprime, md5, nparams, path, mp, nb, sweeps, paper = row
    Lx, Ly = (L, L) if isinstance(L, int) else L
    n = 7 * Lx * Ly // 16
    return StateInfo(
        name=name,
        ansatz=ansatz,
        L=Lx,
        Ly=None if Ly == Lx else Ly,
        tprime=tprime,
        n_up=n,
        n_dn=n,
        weights=f"{name}.eqx",
        md5=md5,
        nparams=nparams,
        contract_path=path,
        sampler=dict(
            ph_neighbor=nb[0],
            se_neighbor=nb[1],
            sweep=sweeps[0],
            thermal=sweeps[1],
            train_sweep=sweeps[2],
        ),
        paper=dict(zip(("E_per_site", "E_err", "var_per_site", "var_err"), paper)),
        notes=_NOTES.get(name, ""),
        max_parallel=mp,
    )


STATES: dict[str, StateInfo] = {row[0]: _make_info(row) for row in _TABLE}


def get_info(name: str) -> StateInfo:
    """`StateInfo` of a released state."""
    try:
        return STATES[name]
    except KeyError:
        raise KeyError(
            f"Unknown state {name!r}; available: {', '.join(STATES)}"
        ) from None


def weights_path(name: str, weights_dir: Optional[str] = None) -> str:
    """Path of ``<weights_dir>/<name>.eqx``, default folder ``weights/`` of the repository."""
    info = get_info(name)
    path = os.path.join(weights_dir or WEIGHTS_DIR, info.weights)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def file_md5(path: str) -> str:
    """md5 checksum of a file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def setup_system(L: int, tprime: float, Ly: Optional[int] = None):
    """
    Define the global periodic L x Ly lattice with ``7 * L * Ly // 16`` electrons per spin
    and return ``(lattice, H)``, H the Hubbard model with U = 8 and t' = tprime.

    quantax allows one lattice per process: calling this again with the same system reuses
    the lattice, a different system raises ``RuntimeError``.
    """
    import quantax as qtx

    if Ly is None:
        Ly = L
    if (L * Ly) % 16:
        raise ValueError("1/8 hole doping needs L * Ly divisible by 16.")
    n = 7 * L * Ly // 16
    try:
        lattice = qtx.get_sites()
    except RuntimeError:  # no system defined yet
        lattice = None
    if lattice is None:
        lattice = qtx.sites.Grid(
            (L, Ly), particle_type=qtx.PARTICLE_TYPE.spinful_fermion, Nparticles=(n, n)
        )
    else:
        same = (
            isinstance(lattice, qtx.sites.Lattice)
            and tuple(lattice.shape) == (1, L, Ly)
            and lattice.particle_type == qtx.PARTICLE_TYPE.spinful_fermion
            and lattice.Nparticles == (n, n)
            and np.all(lattice.boundary == 1)
        )
        if not same:
            raise RuntimeError(
                "A different quantax lattice is already defined in this process; "
                "quantax allows one lattice per process."
            )
    if tprime == 0:
        H = qtx.operator.Hubbard(U=U)
    else:
        H = qtx.operator.Hubbard(U=U, t=[1.0, tprime], n_neighbor=[1, 2])
    return lattice, H


def build_model(info: StateInfo, use_kernel: bool = True):
    """
    Network of a state with random parameters, on the lattice defined by `setup_system`.

    :param use_kernel: Backflows only: Pallas pairing kernels (default) or the einsum path.
        The NN-fPEPS fast paths are controlled by the ``FPEPS_*`` environment variables.
    """
    if CODE_DIR not in sys.path:
        sys.path.insert(0, CODE_DIR)
    if info.ansatz in ("bcs", "pfaffian"):
        from backflow import build_backflow

        return build_backflow(info.ansatz, info.L, use_kernel=use_kernel)
    if info.ansatz == "nnfpeps":
        # the fPEPS modules import one another by top-level name
        if NNFPEPS_DIR not in sys.path:
            sys.path.insert(0, NNFPEPS_DIR)
        from nnfpeps_model import build_nnfpeps

        return build_nnfpeps(info.L, contract_path=info.contract_path, Ly=info.Ly)
    raise ValueError(f"Unknown ansatz {info.ansatz!r}")


def paper_symmetry():
    """Projection of the paper: ``Flip(axis=0) @ Flip(axis=1) @ SpinInverse()``, trivial
    characters, 8 elements. Translations are built into all networks."""
    from quantax.symmetry import Flip, SpinInverse

    return Flip(axis=0) @ Flip(axis=1) @ SpinInverse()


def load_state(
    name: str,
    weights_dir: Optional[str] = None,
    symm=True,
    max_parallel=None,
    use_kernel: bool = True,
):
    """
    Load a released state.

    :param name: Key of `STATES`.
    :param weights_dir: Folder of the weight files, default ``weights/``.
    :param symm: True for the projection of the paper (`paper_symmetry`), False for the
        unprojected network, or a quantax ``Symmetry``.
    :param max_parallel: Forward chunk per device in configurations before the symmetry
        expansion, default ``info.max_parallel``. See ``quantax.state.Variational``.
    :param use_kernel: Backflows only, see `build_model`.

    :return: ``(state, H, info)``: the ``quantax.state.Variational`` state, the
        Hamiltonian, and the `StateInfo`.
    """
    import quantax as qtx
    from quantax.symmetry import Identity, Symmetry

    info = get_info(name)
    path = weights_path(name, weights_dir)
    if file_md5(path) != info.md5:
        warnings.warn(f"{path} differs from the released file (md5 {info.md5}).")
    _, H = setup_system(info.L, info.tprime, info.Ly)
    model = build_model(info, use_kernel=use_kernel)
    if isinstance(symm, Symmetry):
        symmetry = symm
    else:
        symmetry = paper_symmetry() if symm else Identity()
    if max_parallel is None:
        max_parallel = info.max_parallel
    state = qtx.state.Variational(
        model, param_file=path, symm=symmetry, max_parallel=max_parallel
    )
    if state.nparams != info.nparams:
        raise RuntimeError(
            f"{name}: {state.nparams} parameters, expected {info.nparams}."
        )
    return state, H, info


def unpack_spins(bits: np.ndarray, shape) -> np.ndarray:
    """Inverse of ``np.packbits(spins > 0, axis=-1)``: int8 ``+-1`` array of ``shape``."""
    shape = tuple(int(x) for x in shape)
    s = np.unpackbits(bits, axis=-1, count=shape[-1]).astype(np.int8)
    return (2 * s - 1).reshape(shape)


def load_spins(name: str, weights_dir: Optional[str] = None) -> np.ndarray:
    """
    Equilibrated Monte Carlo chains of a state, ``weights/<name>_spins.npz``.

    :return: int8 ``+-1`` array of shape (nchains, 2N), spin-up occupations first.
    """
    info = get_info(name)
    path = os.path.join(weights_dir or WEIGHTS_DIR, f"{name}_spins.npz")
    with np.load(path) as f:
        spins = unpack_spins(f["bits"], f["shape"])
    N = info.N
    if spins.ndim != 2 or spins.shape[1] != 2 * N:
        raise ValueError(f"{path}: shape {spins.shape}, expected (nchains, {2 * N}).")
    if np.any((spins[:, :N] > 0).sum(1) != info.n_up) or np.any(
        (spins[:, N:] > 0).sum(1) != info.n_dn
    ):
        raise ValueError(f"{path}: wrong particle numbers.")
    return spins


def make_sampler(
    state,
    info: StateInfo,
    nsamples: int,
    initial_spins=None,
    sweep_steps: Optional[int] = None,
    thermal_steps: Optional[int] = None,
    seed: Optional[int] = None,
    weights_dir: Optional[str] = None,
):
    """
    Metropolis sampler of the paper: ``MixSampler`` of ``ParticleHop(nsamples // 2)`` and
    ``SiteExchange(nsamples // 2)`` with the neighbour ranges of ``info.sampler``.

    :param nsamples: Total number of chains; ``nsamples // 2`` must be divisible by the
        number of devices.
    :param initial_spins: Initial chains, default the released chains, repeated or
        truncated to ``nsamples``.
    :param sweep_steps: Metropolis steps per chain between samples, default
        ``info.sampler["sweep"] * N``.
    :param thermal_steps: Thermalization steps per chain, run on construction, default
        ``info.sampler["thermal"] * N``.
    :param seed: Reset the quantax random key first.
    :param weights_dir: Folder of the released chains, default ``weights/``.
    """
    import quantax as qtx

    if nsamples % 2:
        raise ValueError("nsamples must be even.")
    if seed is not None:
        qtx.set_random_seed(seed)
    N = info.N
    if initial_spins is None:
        initial_spins = load_spins(info.name, weights_dir)
    spins = np.asarray(initial_spins, dtype=np.int8)
    if spins.ndim == 2 and spins.shape[0] != nsamples:
        spins = np.resize(spins, (nsamples, 2 * N))
    if sweep_steps is None:
        sweep_steps = int(round(info.sampler["sweep"] * N))
    if thermal_steps is None:
        thermal_steps = int(round(info.sampler["thermal"] * N))

    ph = qtx.sampler.ParticleHop(
        state, nsamples // 2, thermal_steps=0, n_neighbor=info.sampler["ph_neighbor"]
    )
    se = qtx.sampler.SiteExchange(
        state, nsamples // 2, thermal_steps=0, n_neighbor=info.sampler["se_neighbor"]
    )
    return qtx.sampler.MixSampler(
        [ph, se],
        thermal_steps=thermal_steps,
        sweep_steps=sweep_steps,
        initial_spins=spins,
    )
