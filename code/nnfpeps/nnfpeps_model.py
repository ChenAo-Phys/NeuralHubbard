"""NN-fPEPS ansatz of the paper.

A bottleneck residual CNN maps the configuration to the site tensors of a Z2
fermionic PEPS (bond dimension D = 4), which is contracted by HOTRG
(truncation chi = 4) along ``contract_path`` followed by an exact contraction
of the remaining grid. `fPEPS_hotrg_symm.SymmNeuralPEPS` returns the
zero-momentum translation average of the amplitude; the point-group and spin
projections of the paper are applied by the surrounding
``quantax.state.Variational`` (``symm=Flip(axis=0) @ Flip(axis=1) @ SpinInverse()``).
With ``contract_path=""`` (the 16 x 4 state) the whole lattice is contracted
exactly and the plain `fPEPS_hotrg_symm.NeuralPEPS` is already translation
invariant, so no average is taken.

The field layout of the network classes reproduces the released checkpoints
leaf by leaf; do not reorder or rename fields.
"""

import os
import sys
from typing import Callable, Optional, Sequence, Tuple, Union

# the fPEPS module family imports itself by top-level module name
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import equinox as eqx
from equinox.nn import RMSNorm
from quantax import get_lattice, get_subkeys
from quantax.nn import Conv, Embedding

from fPEPS_hotrg_symm import NeuralPEPS, SymmNeuralPEPS

D = 4
CHI = 4
# 16 Z2 sectors x (D/2)^4 virtual blocks x 2 physical states per site
CHANNELS = 2 * D**4
NBLOCKS = 16
COMPRESSED_CHANNELS = 128
KERNEL_SIZE = (3, 3)
NPARAMS = 4_470_784

# HOTRG paths of the released checkpoints, keyed by L of an L x L lattice:
# L -> L/4 rows ("yy") or L -> L/8 ("yyy"), then the exact contraction of the
# L x 4 grid
CONTRACT_PATHS = {16: "yy", 32: "yyy"}
# rectangular lattices, keyed by (Lx, Ly): the 16 x 4 state is contracted exactly
CONTRACT_PATHS_RECT = {(16, 4): ""}


class _ConvBlock(eqx.Module):
    """Residual bottleneck block: RMSNorm -> gelu -> 1x1 conv (C -> 128) -> gelu
    -> 3x3 circular conv (128 -> 128) -> gelu -> 1x1 conv (128 -> C), plus the
    residual."""

    norm: Callable
    pw_conv1: Conv
    conv1: Conv
    pw_conv2: Conv

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        dtype: DTypeLike = jnp.float32,
    ):
        Lx, Ly = get_lattice().shape[1:]
        self.norm = RMSNorm(
            (channels, Lx, Ly), use_weight=False, use_bias=False, dtype=dtype
        )
        keys = get_subkeys(3)
        self.pw_conv1 = Conv(
            in_channels=channels,
            out_channels=COMPRESSED_CHANNELS,
            kernel_size=1,
            dtype=dtype,
            key=keys[0],
        )
        self.conv1 = Conv(
            in_channels=COMPRESSED_CHANNELS,
            out_channels=COMPRESSED_CHANNELS,
            kernel_size=kernel_size,
            dtype=dtype,
            key=keys[1],
        )
        self.pw_conv2 = Conv(
            in_channels=COMPRESSED_CHANNELS,
            out_channels=channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[2],
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.norm(x)
        x = jax.nn.gelu(x)
        x = self.pw_conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.pw_conv2(x)
        return x + residual


class BottleneckResConv(eqx.Module):
    """Deep residual CNN with bottleneck blocks.

    Maps a configuration (``+-1`` array of length 2N, spin-up block first) to
    features of shape ``(channels, Lx, Ly)``, one column per site. The
    embedding is shared by all sites (``sublattice=None``) and the convolutions
    are circular on the periodic lattice, so the network is translation
    covariant, as `SymmNeuralPEPS` requires.
    """

    nblocks: int
    channels: int
    kernel_size: Union[int, Sequence[int]]
    dtype: DTypeLike
    embedding: Embedding
    blocks: Tuple[Callable, ...]

    def __init__(
        self,
        nblocks: int = NBLOCKS,
        channels: int = CHANNELS,
        kernel_size: Union[int, Sequence[int]] = KERNEL_SIZE,
        dtype: DTypeLike = jnp.float32,
    ):
        self.nblocks = nblocks
        self.channels = channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.embedding = Embedding(channels, None, dtype)
        self.blocks = tuple(
            _ConvBlock(channels, kernel_size, dtype) for _ in range(nblocks)
        )

    def __call__(self, s: jax.Array) -> jax.Array:
        x = self.embedding(s)
        for block in self.blocks:
            x = block(x)
        x /= jnp.sqrt(self.nblocks + 1)
        return x


def build_nnfpeps(
    L: int,
    contract_path: Optional[str] = None,
    dtype: DTypeLike = jnp.float32,
    Ly: Optional[int] = None,
) -> Union[SymmNeuralPEPS, NeuralPEPS]:
    """
    Build the NN-fPEPS model with random weights, ready to be loaded with
    ``quantax.state.Variational(model, param_file=...)``.

    The global quantax lattice must already exist and be an ``L x Ly`` grid.

    :param L:
        Linear size of the lattice, the extent along x.

    :param contract_path:
        HOTRG contraction path. Default to ``"yy"`` for ``L=16`` and ``"yyy"``
        for ``L=32``, and ``""`` for the 16 x 4 lattice, the paths of the
        released checkpoints; required for other sizes. The path is not stored
        in the weight file but is part of the state: a checkpoint is only valid
        with the path it was trained with.

    :param dtype:
        Data type of the tensor-network contraction. The network weights are
        always float32. Do not sample in float16 (overflow in the exponent
        stripping produces absorbing infinite-amplitude chains).

    :param Ly:
        Extent along y. Default to L.

    :return:
        `SymmNeuralPEPS`, the translation average, or the plain `NeuralPEPS`
        for ``contract_path=""``, whose exact contraction of the translation
        covariant network is translation invariant by itself.
    """
    if Ly is None:
        Ly = L
    lattice = get_lattice()
    if tuple(lattice.shape[1:]) != (L, Ly):
        raise ValueError(
            f"The global lattice has shape {tuple(lattice.shape[1:])}, expected "
            f"({L}, {Ly})."
        )
    if contract_path is None:
        default = CONTRACT_PATHS.get(L) if L == Ly else CONTRACT_PATHS_RECT.get((L, Ly))
        if default is None:
            raise ValueError(
                f"No default contract_path for {L}x{Ly}; pass one explicitly."
            )
        contract_path = default

    net = BottleneckResConv(NBLOCKS, CHANNELS, KERNEL_SIZE, dtype=jnp.float32)
    if contract_path == "":
        return NeuralPEPS(net, D=D, chi=CHI, contract_path="", dtype=dtype)
    return SymmNeuralPEPS(net, D=D, chi=CHI, contract_path=contract_path, dtype=dtype)
