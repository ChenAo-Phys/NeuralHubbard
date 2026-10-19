r"""Neural backflow pairing wavefunctions for the spinful Hubbard model.

A translation-equivariant residual CNN (``quantax.model.ResConv``: 16 blocks, 64 channels,
3x3 circular convolutions, GELU, RMSNorm) maps a configuration :math:`n` to per-site
features :math:`y^\sigma_i(n)`.

**Pfaffian backflow** (`PfaffianBackflow`, main-text Eq. (5))

.. math::

    \psi(n) = \mathrm{pf}\,[n \star f(n) \star n], \qquad
    f^{\sigma\sigma'}_{ij}(n) = p_{ij} \sum_{\alpha=1}^{d} W^{\sigma\sigma'}_{i-j,\alpha}\,
        y^\sigma_{i\alpha}(n)\, y^{\sigma'}_{j\alpha}(n),

with :math:`d = 32` features per spin (the two halves of the 64 output channels),
:math:`p_{ij} \propto e^{-\lambda d(i,j)}` normalised over all pairs, :math:`\lambda =
\mathrm{softplus}(\theta)`, the minimum-image distance :math:`d(i,j)`, and
:math:`W^{\sigma'\sigma}_{-r} = -W^{\sigma\sigma'}_r`.

**BCS backflow** (`BCSBackflow`, main-text Eq. (6) plus a nearest-neighbour d-wave term)

.. math::

    \psi(n) = (-1)^{N_\uparrow (N_\uparrow - 1)/2}
        \det\,[n_\uparrow \star \big(f^{\uparrow\downarrow}(n) + \delta_d F^d\big)
        \star n_\downarrow],

with :math:`d = 64` features from a linear head on the CNN output (scaled by
:math:`1/\sqrt{17}`), and :math:`F^d_{ij} = -1` on x bonds and :math:`+1` on y bonds.

Both amplitudes are multiplied by ``fermion_inverse_sign(s)``; modes are spin major and
sites row major, ``i = x * L + y``. ``use_kernel=True`` computes the occupied block with the
Pallas kernels `pf_kernel` and `bcs_kernel` (their pure-JAX reference off GPU) and
``fermix``; ``use_kernel=False`` builds the full pairing matrix with ``jnp.einsum``. Both
paths share the parameters. Define the quantax lattice before building a model.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import quantax as qtx
from quantax.model import ResConv
from quantax.nn import fermion_idx, fermion_inverse_sign, lecun_normal
from quantax.symmetry import Identity
from quantax.utils import LogArray
import fermix

from . import pf_kernel, bcs_kernel

NBLOCKS = 16
CHANNELS = 64
KERNEL_SIZE = (3, 3)


class _EinsumTables(NamedTuple):
    disp_idx: np.ndarray  # (N, N) displacement index i - j
    neg_idx: np.ndarray  # (N,) r -> -r
    self_conj: np.ndarray  # displacements with 2r = 0
    dist: np.ndarray  # (N, N) minimum-image distance, float32
    fd: np.ndarray  # (N, N) nearest-neighbour d-wave form factor, float32


@functools.lru_cache(maxsize=None)
def _einsum_tables(LX: int, LY: int) -> _EinsumTables:
    xy = np.stack(np.meshgrid(np.arange(LX), np.arange(LY), indexing="ij"), -1)
    xy = xy.reshape(-1, 2)
    shape = np.array([LX, LY])
    dxy = (xy[:, None, :] - xy[None, :, :]) % shape
    disp_idx = dxy[..., 0] * LY + dxy[..., 1]
    neg_idx = ((-xy[:, 0]) % LX) * LY + (-xy[:, 1]) % LY
    self_conj = np.flatnonzero(neg_idx == np.arange(LX * LY))
    dmin = np.minimum(dxy, shape - dxy)
    dist = np.sqrt((dmin**2).sum(-1)).astype(np.float32)
    fd = np.zeros((LX * LY, LX * LY), np.float32)
    fd[(dmin[..., 0] == 1) & (dmin[..., 1] == 0)] = -1.0  # x bonds
    fd[(dmin[..., 0] == 0) & (dmin[..., 1] == 1)] = 1.0  # y bonds
    return _EinsumTables(disp_idx, neg_idx, self_conj, dist, fd)


def _lattice_geometry() -> tuple[int, int, int, int]:
    """(LX, LY, n_up, n_dn) of the global lattice; checks the row-major site order."""
    lattice = qtx.get_lattice()
    if not lattice.is_fermion or not lattice.is_spinful:
        raise ValueError("The backflow ansatzes need a spinful-fermion lattice.")
    if len(lattice.shape) != 3 or lattice.shape[0] != 1:
        raise ValueError(f"Expected a 2D lattice with one site per cell, got {lattice.shape}.")
    LX, LY = (int(n) for n in lattice.shape[1:])
    N = lattice.Nsites
    index = np.asarray(lattice.index_from_xyz).reshape(LX, LY)
    if not np.array_equal(index, np.arange(N).reshape(LX, LY)):
        raise ValueError("Lattice site order is not row major x * LY + y.")
    if not isinstance(lattice.Nparticles, tuple):
        raise ValueError("Define the lattice with Nparticles=(n_up, n_dn).")
    n_up, n_dn = (int(n) for n in lattice.Nparticles)
    return LX, LY, n_up, n_dn


def _identity(x):
    return x


def _with_kernel(module: eqx.Module, use_kernel: bool) -> eqx.Module:
    """Copy of ``module`` with the static field ``use_kernel`` replaced (same leaves)."""
    new = jax.tree_util.tree_map(lambda x: x, module)
    object.__setattr__(new, "use_kernel", use_kernel)
    return new


def _pad_size(*trees) -> int:
    """Zeros needed to round the parameter count up to a multiple of 4096."""
    leaves = jax.tree.leaves(eqx.filter(trees, eqx.is_inexact_array))
    return (-sum(x.size for x in leaves)) % 4096


class PfaffianBackflow(eqx.Module):
    """Pfaffian backflow (see the module docstring).

    Parameters: ``net`` (ResConv, output channels split into :math:`y^\\uparrow |
    y^\\downarrow`), ``W`` (3, N, d) holding :math:`[W^{\\uparrow\\downarrow},
    V^{\\uparrow\\uparrow}, V^{\\downarrow\\downarrow}]` with :math:`W^{\\sigma\\sigma}_r =
    (V^{\\sigma\\sigma}_r - V^{\\sigma\\sigma}_{-r}) / 2`, ``theta``, and a zero ``pad`` that
    rounds the parameter count to a multiple of 4096.
    """

    net: eqx.Module
    W: jax.Array
    theta: jax.Array
    pad: jax.Array
    d: int = eqx.field(static=True)
    holomorphic: bool = eqx.field(static=True)
    geom: pf_kernel.PairGeometry = eqx.field(static=True)
    use_kernel: bool = eqx.field(static=True)

    def __init__(self, net: eqx.Module, channels: int, use_kernel: bool = True):
        assert channels % 2 == 0, "channels are split into y^up | y^dn"
        LX, LY, n_up, n_dn = _lattice_geometry()
        N = LX * LY
        self.net = net
        self.d = channels // 2
        self.W = jax.random.normal(
            qtx.get_subkeys(), (3, N, self.d), jnp.float32
        ) / np.sqrt(self.d)
        self.theta = jnp.array(0.55, dtype=jnp.float32)
        self.pad = jnp.zeros((_pad_size(self.net, self.W, self.theta),), jnp.float32)
        self.holomorphic = False
        self.geom = pf_kernel.PairGeometry(LX, LY, self.d, n_up, n_dn)
        self.use_kernel = use_kernel

    @property
    def tables(self) -> _EinsumTables:
        return _einsum_tables(self.geom.LX, self.geom.LY)

    def with_kernel(self, use_kernel: bool) -> "PfaffianBackflow":
        """The same state on the kernel (True) or einsum (False) path."""
        return _with_kernel(self, use_kernel)

    def features(self, s: jax.Array) -> jax.Array:
        """(2, N, d) backflow features :math:`y^\\sigma_{i\\alpha}`."""
        x = self.net(s)  # (channels, LX, LY)
        y = x.reshape(2, self.d, self.geom.N)  # channels [0:d] up, [d:2d] down
        return jnp.swapaxes(y, 1, 2)

    def W_blocks(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """:math:`(W^{\\uparrow\\downarrow}, W^{\\uparrow\\uparrow},
        W^{\\downarrow\\downarrow})`, the same-spin ones odd in r."""
        neg_idx = self.tables.neg_idx
        W_ud = self.W[0]
        W_uu = (self.W[1] - self.W[1][neg_idx]) / 2
        W_dd = (self.W[2] - self.W[2][neg_idx]) / 2
        return W_ud, W_uu, W_dd

    def raw_pairing_matrix(self, s: jax.Array) -> jax.Array:
        """(2N, 2N) f from the four spin blocks by einsum, before (f - f^T) / 2."""
        tables = self.tables
        disp_idx = tables.disp_idx
        y = self.features(s)
        W_ud, W_uu, W_dd = self.W_blocks()
        f_ud = jnp.einsum("ia,ija,ja->ij", y[0], W_ud[disp_idx], y[1])
        f_uu = jnp.einsum("ia,ija,ja->ij", y[0], W_uu[disp_idx], y[0])
        f_dd = jnp.einsum("ia,ija,ja->ij", y[1], W_dd[disp_idx], y[1])
        lam = jax.nn.softplus(self.theta)
        pref = jnp.exp(-lam * jnp.asarray(tables.dist))
        pref = pref / jnp.sum(pref)
        f = jnp.block([[f_uu, f_ud], [-f_ud.T, f_dd]])
        return f * jnp.tile(pref, (2, 2))

    def pairing_matrix(self, s: jax.Array) -> jax.Array:
        """Exactly antisymmetric (2N, 2N) pairing matrix by einsum."""
        f = self.raw_pairing_matrix(s)
        return (f - f.T) / 2

    def occupied_block(self, s: jax.Array) -> jax.Array:
        """(Ne, Ne) block :math:`n \\star f \\star n` on the selected path."""
        idx = fermion_idx(s)  # occupied up modes, then down modes
        if self.use_kernel:
            return pf_kernel.pairing_matrix_occ(
                self.features(s), idx, self.W, self.theta, self.geom
            )
        f = self.pairing_matrix(s)
        return f[idx, :][:, idx]

    def __call__(self, s: jax.Array) -> LogArray:
        F = self.occupied_block(s)
        sign, logabs = fermix.slogpf(F)
        return LogArray(sign, logabs) * fermion_inverse_sign(s)


class BCSBackflow(eqx.Module):
    """BCS backflow (see the module docstring).

    Parameters: ``trunk`` (ResConv layers without the final activation and
    symmetrisation), ``head`` (2, d, channels) giving :math:`y^\\uparrow, y^\\downarrow`,
    ``W`` (N, d), ``theta``, the d-wave amplitude ``delta_d``, and a zero ``pad``.
    """

    trunk: tuple
    head: jax.Array
    W: jax.Array
    theta: jax.Array
    delta_d: jax.Array
    pad: jax.Array
    nblocks: int = eqx.field(static=True)
    holomorphic: bool = eqx.field(static=True)
    geom: bcs_kernel.UDGeometry = eqx.field(static=True)
    use_kernel: bool = eqx.field(static=True)

    def __init__(
        self, nblocks: int, channels: int, kernel_size, d: int, use_kernel: bool = True
    ):
        LX, LY, n_up, n_dn = _lattice_geometry()
        N = LX * LY
        resconv = ResConv(
            nblocks, channels, kernel_size, use_rmsnorm=True, dtype=jnp.float32
        )
        self.trunk = resconv.layers[:-2]
        self.head = lecun_normal(qtx.get_subkeys(), (2, d, channels), jnp.float32)
        self.W = lecun_normal(qtx.get_subkeys(), (N, d), jnp.float32)
        self.theta = jnp.array(0.55, dtype=jnp.float32)
        self.delta_d = jnp.array(0.1, dtype=jnp.float32)
        size = _pad_size(self.trunk, self.head, self.W, self.theta, self.delta_d)
        self.pad = jnp.zeros((size,), jnp.float32)
        self.nblocks = nblocks
        self.holomorphic = False
        self.geom = bcs_kernel.UDGeometry(LX, LY, d, n_up, n_dn)
        self.use_kernel = use_kernel

    @property
    def tables(self) -> _EinsumTables:
        return _einsum_tables(self.geom.LX, self.geom.LY)

    def with_kernel(self, use_kernel: bool) -> "BCSBackflow":
        """The same state on the kernel (True) or einsum (False) path."""
        return _with_kernel(self, use_kernel)

    def trunk_out(self, s: jax.Array) -> jax.Array:
        x = s
        for layer in self.trunk:
            x = layer(x)
        return x

    def features(self, s: jax.Array) -> jax.Array:
        """(2, N, d) backflow features :math:`y^\\sigma_{i\\alpha}`."""
        x = self.trunk_out(s) / jnp.sqrt(self.nblocks + 1)
        return jnp.einsum("kdc,cn->knd", self.head, x.reshape(x.shape[0], -1))

    def pairing_matrix(self, s: jax.Array) -> jax.Array:
        """(N, N) :math:`f^{\\uparrow\\downarrow} + \\delta_d F^d` by einsum."""
        tables = self.tables
        y = self.features(s)
        f = jnp.einsum("ia,ija,ja->ij", y[0], self.W[tables.disp_idx], y[1])
        lam = jax.nn.softplus(self.theta)
        pref = jnp.exp(-lam * jnp.asarray(tables.dist))
        f = f * pref / jnp.sum(pref)
        return f + self.delta_d * jnp.asarray(tables.fd)

    def occupied_block(self, s: jax.Array) -> jax.Array:
        """(N_up, N_dn) block :math:`n_\\uparrow \\star (f + \\delta_d F^d) \\star
        n_\\downarrow` on the selected path."""
        iu, idn = fermion_idx(s, separate_spins=True)
        if self.use_kernel:
            F = bcs_kernel.pairing_block_ud(
                self.features(s), iu, idn, self.W, self.theta, self.geom
            )
            return F + self.delta_d * jnp.asarray(self.tables.fd)[iu, :][:, idn]
        f = self.pairing_matrix(s)
        return f[iu, :][:, idn]

    def __call__(self, s: jax.Array) -> LogArray:
        F = self.occupied_block(s)
        if self.use_kernel:
            sign, logabs = fermix.slogdet(F)
        else:
            sign, logabs = jnp.linalg.slogdet(F)
        n = F.shape[0]
        sign = sign * (-1) ** (n * (n - 1) // 2)
        return LogArray(sign, logabs) * fermion_inverse_sign(s)


def build_backflow(kind: str, L: int, use_kernel: bool = True) -> eqx.Module:
    """
    Backflow ansatz of the paper on the global L x L lattice, with random parameters.

    :param kind: ``"bcs"`` (`BCSBackflow`) or ``"pfaffian"`` (`PfaffianBackflow`).
    :param L: Linear size of the lattice defined beforehand.
    :param use_kernel: Pallas kernels and fermix (default) or the einsum path.

    Load trained weights with ``quantax.state.Variational(model, param_file)``.
    """
    LX, LY, _, _ = _lattice_geometry()
    if (LX, LY) != (L, L):
        raise ValueError(f"The lattice is {LX}x{LY}, not {L}x{L}.")
    if kind == "pfaffian":
        net = ResConv(
            nblocks=NBLOCKS,
            channels=CHANNELS,
            kernel_size=KERNEL_SIZE,
            use_rmsnorm=True,
            use_final_bias=True,
            final_activation=_identity,
            trans_symm=Identity(),
            dtype=jnp.float32,
        )
        return PfaffianBackflow(net, CHANNELS, use_kernel=use_kernel)
    if kind == "bcs":
        return BCSBackflow(NBLOCKS, CHANNELS, KERNEL_SIZE, d=64, use_kernel=use_kernel)
    raise ValueError(f"Unknown backflow kind {kind!r}; use 'bcs' or 'pfaffian'.")
