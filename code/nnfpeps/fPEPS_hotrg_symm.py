"""fPEPS HOTRG contraction with shared-representative translation averaging.

HOTRG building blocks: Z2 fermionic site tensors, oblique-projector
compression layers and the exact contraction of the remaining grid.

No dummy modes: the site tensors carry no symmray dummy_modes — the
odd-tensor ordering signs they would generate are computed by hand from the
charge grid instead (`_merge_sign` per HOTRG pair merge, `_final_order_sign`
for the exact grid contraction), with the contraction paths fixed to make
the operand order deterministic. Verified element-wise against a dummy-mode
implementation in f64.

Translation averaging: for a translation-covariant NeuralPEPS (net with
Embedding sublattice=None and circular convolutions), psi(T_t s) of all
Lx*Ly translations share most of the HOTRG work — after the accumulated
block sizes reach (bx, by), the level-input grid of translation t = (tx, ty)
is a whole-grid roll of the representative residue (tx mod bx, ty mod by).
Each HOTRG layer therefore needs only bx*by compress calls (2 at the first
layer, 4 at the second, ...) instead of Lx*Ly, and the final grid is
contracted once per residue class, giving O(N log N) total cost instead of
O(N^2). The neural network runs once.

Fast paths (ON by default on GPU):
- linalg: the projector computation routes batched 8x8 eigh through the
  fused Pallas Jacobi kernel and small cholesky through an unrolled jnp
  implementation (fpeps_pallas_linalg.py), replacing cuSOLVER syevj/potrf
  (99% of the D=4 forward GPU time). Auto-disabled (with a warning) when no
  GPU / Pallas is unavailable; the eigh gradient keeps the stabilized
  custom_vjp below.
- final contraction: `_raw_final_contract` builds the canon'd line blobs
  with a tensor-core block-dot kernel (fpeps_layer_kernels.blob_build,
  factored EXT tables, 2.2x and bitwise vs the XLA einsum path, which
  stays as the f64/reference route and the backward replay) and runs the
  fp32/fp16 ring as a fused Pallas chain kernel
  (fpeps_layer_kernels.ring_chain, accumulator in registers, 2.9-3.1x);
  f64 keeps the pure-jnp loop. f64-verified element-wise against the
  reference quimb line-ring contraction (kept as
  `_reference_raw_final_contract`); an import-time self-test falls back
  to the reference if the symmray fuse conventions ever change.
- layer kernels: `compress_y` at D = chi = 4 routes through the fused
  Pallas layer kernels (fpeps_fast_layer.fast_compress_y_k: merged
  gram+cholesky, projector assembly, patch — one memory pass per unit;
  differentiable, nested-vmap-safe). `compress_x` routes through the SAME
  kernels on the leg-transposed lattice (fast_compress_x_k; an x-merge is
  the y-merge of the transposed network) when cells have
  min(subx, suby) == 1, i.e. pure-x/pure-y contract paths — mixed-history
  x-merges stay stock. The final contraction of vertical-line grids
  ("x..."-family paths, e.g. 16x16 "xx" -> 4x16) likewise evaluates the
  transposed network with the horizontal fast machinery, times the
  closed-form mode-reorder sign `_transpose_reorder_sign`. Each guarded
  by a psi-level import self-test vs the stock path.
- pure-x paths ("x", "xx", ...) at D = chi = 4 skip the per-layer
  transposes altogether: the site tensors are leg-transposed ONCE up front
  (`_transpose_site_frame`), every x-merge then IS the y-merge of the
  transposed lattice (transpose_out of layer k and transpose_in of layer
  k+1 cancel exactly: `transpose_blocks` is an involution), the final grid
  is contracted horizontally in that frame and `_transpose_reorder_sign`
  of the original-frame final charge grid is applied once. Same kernels,
  same operands bit-for-bit, 3 (path "x") / 5 (path "xx") fewer
  transpose_blocks gathers per translation representative than the
  per-layer dispatch (which remains the fallback, FPEPS_REFERENCE_XFRAME=1).
- translation signs: SymmNeuralPEPS uses closed-form translation
  permutation signs (`_trans_perm_signs_all`, prefix sums) instead of
  quantax's generic per-element argsort sign in symmetrize (which was
  59-75% of the Symm forward); verified element-wise vs quantax.
Environment switches: FPEPS_REFERENCE=1 disables all fast paths;
FPEPS_REFERENCE_LINALG=1 / FPEPS_REFERENCE_FINAL=1 / FPEPS_REFERENCE_LAYER=1
/ FPEPS_REFERENCE_SYMM=1 / FPEPS_REFERENCE_RING=1 / FPEPS_REFERENCE_BUILD=1
/ FPEPS_REFERENCE_XFRAME=1 disable them separately (RING = the fused
ring-chain kernel only; BUILD = the tensor-core blob-build kernel only,
falling back to the XLA blobs; XFRAME = the single up-front transpose of
pure-x paths, falling back to the per-layer transposed-frame dispatch);
FPEPS_EIGH_NSWEEPS / FPEPS_EIGH_NSWEEPS_F64 set the Jacobi sweep counts
(defaults 6 / 10); FPEPS_SKIP_SELFTEST=1 skips the import-time self-tests,
whose PASS markers are cached in FPEPS_CACHE_DIR (default ~/.cache).
Without a GPU the Pallas paths switch off and the XLA fallbacks run (slow,
for small checks only).
"""

from typing import Tuple
from functools import lru_cache, partial
import math
import os
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from einops import rearrange
import autoray as ar
import quimb.tensor as qtn
import equinox as eqx
import quantax as qtx
from quimb.tensor import TensorNetworkGen
from quimb.tensor.tensor_core import bonds, tensor_contract
from quimb.tensor.decomp import compute_oblique_projectors
from symmray import Z2FermionicArrayFlat
from symmray.flat.flat_index import FlatIndex
from quantax.utils import LogArray


def default_contract_path(max_unfinished: int = 10) -> str:
    contract_path = ""
    lx, ly = qtx.get_lattice().shape[1:]
    while True:
        if lx % 2 == 1 and ly % 2 == 1:
            raise ValueError("Cannot find valid contraction path")
        if lx % 2 == 0:
            contract_path += "x"
            lx //= 2
            if lx * ly <= max_unfinished:
                return contract_path
        if ly % 2 == 0:
            contract_path += "y"
            ly //= 2
            if lx * ly <= max_unfinished:
                return contract_path


def convert_fermion_input(s: jax.Array) -> jax.Array:
    s = s.reshape(2, -1)
    s = jnp.where(s > 0, 1, 0).astype(jnp.int8)
    s = 2 * (s[0] != s[1]) + s[1]
    return s


def get_charge_inds(charge, axis):
    sectors = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.int8,
    )
    sectors = sectors[:, axis]
    xp = ar.get_namespace(charge)
    if isinstance(charge, jax.Array):
        inds = xp.flatnonzero(sectors == charge, size=8)
    else:
        inds = xp.flatnonzero(sectors == charge)
    return inds


def get_params_charge(params: jax.Array, s: jax.Array) -> Tuple[jax.Array, jax.Array]:
    lattice_shape = qtx.get_lattice().shape[1:]
    sublattice_shape = params.shape[:2]
    reps = tuple(l // subl for l, subl in zip(lattice_shape, sublattice_shape))
    reps = reps + (1,) * (params.ndim - 2)
    params = jnp.tile(params, reps)
    params = params.reshape(-1, *params.shape[2:])
    s = convert_fermion_input(s)

    @jax.vmap
    def slice_params(p, s):
        charge = (s > 1).astype(jnp.int8)
        inds = get_charge_inds(charge, axis=-1)
        return p[inds, :, :, :, :, s % 2], charge

    params, charge = slice_params(params, s)
    params = params.reshape(*lattice_shape, *params.shape[1:])
    charge = charge.reshape(*lattice_shape, 1, 1)
    return params, charge


def unpack(params, charge, pbcx=True, pbcy=True) -> qtn.TensorNetwork2D:
    lx, ly, subx, suby = charge.shape
    bond_dims = params.shape[-4:]
    site_tag_id = "I{},{}"
    x_tag_id = "X{}"
    y_tag_id = "Y{}"
    sectors = jnp.array(
        [
            [
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0],
            ],
        ],
        dtype=jnp.int8,
    )
    duals = [1, 0, 1, 0]  # -x, +x, -y, +y
    circular_i = lambda i: (i % lx if pbcx else i)
    circular_j = lambda j: (j % ly if pbcy else j)

    sites = [(i, j) for i in range(lx) for j in range(ly)]
    tn = TensorNetworkGen.new(sites=sites, site_tag_id=site_tag_id)

    for i in range(lx):
        for j in range(ly):
            indices = (
                FlatIndex(num_charges=2, charge_size=D, dual=dual)
                for D, dual in zip(bond_dims, duals)
            )
            parity = jnp.sum(charge[i, j].flatten()) % 2

            # no dummy modes: the odd-tensor ordering signs they would
            # generate are applied by hand instead (see _merge_sign and
            # _final_order_sign), which requires the deterministic
            # contraction paths used below
            data = Z2FermionicArrayFlat(
                sectors[parity], params[i, j], indices, dummy_modes=()
            )
            site = (i, j)
            inds = [
                f"b({circular_i(i-1)}, {j})-{site}",
                f"b{site}-({circular_i(i+1)}, {j})",
                f"b({i}, {circular_j(j-1)})-{site}",
                f"b{site}-({i}, {circular_j(j+1)})",
            ]
            tags = [site_tag_id.format(i, j), x_tag_id.format(i), y_tag_id.format(j)]
            tn |= qtn.Tensor(data, inds, tags)

    tn.view_as_(qtn.TensorNetwork2D, Lx=lx, Ly=ly, x_tag_id=x_tag_id, y_tag_id=y_tag_id)
    return tn


def unpack_P1(blocks):
    chi_size, D2_size = blocks.shape[1:]
    sub_size = math.isqrt(D2_size // 2)
    tree = {
        "sectors": np.array([[0, 0], [1, 1]], dtype=np.int8),
        "blocks": blocks,
        "indices": (
            {
                "num_charges": 2,
                "charge_size": chi_size,
                "dual": True,
                "subinfo": None,
                "linearmap": None,
            },
            {
                "num_charges": 2,
                "charge_size": D2_size,
                "dual": False,
                "subinfo": {
                    "indices": (
                        {
                            "num_charges": 2,
                            "charge_size": sub_size,
                            "dual": False,
                            "subinfo": None,
                            "linearmap": None,
                        },
                        {
                            "num_charges": 2,
                            "charge_size": sub_size,
                            "dual": False,
                            "subinfo": None,
                            "linearmap": None,
                        },
                    ),
                    "subkeys": np.array(
                        [[[0, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=np.int8
                    ),
                },
                "linearmap": None,
            },
        ),
        "symmetry": "Z2",
        "label": None,
        "phases": None,
        "dummy_modes": (),
    }
    data = Z2FermionicArrayFlat.from_pytree(tree)
    data = data.reshape((chi_size * 2, sub_size * 2, sub_size * 2))
    inds = ["up", "up0", "up1"]
    tags = "P1"
    return qtn.Tensor(data, inds, tags)


def unpack_P2(blocks):
    D2_size, chi_size = blocks.shape[1:]
    sub_size = math.isqrt(D2_size // 2)
    tree = {
        "sectors": np.array([[0, 0], [1, 1]], dtype=np.int8),
        "blocks": blocks,
        "indices": (
            {
                "num_charges": 2,
                "charge_size": D2_size,
                "dual": True,
                "subinfo": {
                    "indices": (
                        {
                            "num_charges": 2,
                            "charge_size": sub_size,
                            "dual": 1,
                            "subinfo": None,
                            "linearmap": None,
                        },
                        {
                            "num_charges": 2,
                            "charge_size": sub_size,
                            "dual": 1,
                            "subinfo": None,
                            "linearmap": None,
                        },
                    ),
                    "subkeys": np.array(
                        [[[0, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=np.int8
                    ),
                },
                "linearmap": None,
            },
            {
                "num_charges": 2,
                "charge_size": chi_size,
                "dual": False,
                "subinfo": None,
                "linearmap": None,
            },
        ),
        "symmetry": "Z2",
        "label": None,
        "phases": None,
        "dummy_modes": (),
    }
    data = Z2FermionicArrayFlat.from_pytree(tree)
    data = data.reshape((sub_size * 2, sub_size * 2, chi_size * 2))
    inds = ["down0", "down1", "down"]
    tags = "P2"
    return qtn.Tensor(data, inds, tags)


@jax.custom_vjp
def eigh_single(x):
    m = jnp.sum(jnp.diag(x)) * 1e-6
    m = jax.lax.stop_gradient(m)
    I = jnp.eye(x.shape[-1], dtype=x.dtype)
    # Jacobi on GPU; the CPU backend does not implement it
    if jax.default_backend() == "gpu":
        impl = jax.lax.linalg.EighImplementation.JACOBI
    else:
        impl = None
    w, v = jax.lax.linalg.eigh(x + m * I, implementation=impl)
    return v, w


def _eigh_fwd(x):
    v, w = eigh_single(x)
    return (v, w), (v, w)


def _eigh_bwd(res, g):
    v, w = res
    gv, gw = g
    if gv is None:
        gv = jnp.zeros_like(v)
    if gw is None:
        gw = jnp.zeros_like(w)

    wH = w.conj().T
    S = wH @ gw
    Aanti = S - S.conj().T

    dv = v[:, None] - v[None, :]
    eps = jnp.max(jnp.abs(v)) * 1e-6
    eps = jnp.where(eps < 1e-12, 1e-12, eps)
    eps = jax.lax.stop_gradient(eps)
    F = dv / (dv * dv + eps * eps)
    F = jnp.where(jnp.eye(v.shape[0], dtype=bool), 0.0, F)

    G = jnp.diag(gv) - 0.5 * (F * Aanti)

    gA = w @ G @ wH
    gA = 0.5 * (gA + gA.conj().T)
    return (gA,)


eigh_single.defvjp(_eigh_fwd, _eigh_bwd)


def eigh_stable(x):
    batch_shape = x.shape[:-2]
    x = x.reshape(-1, *x.shape[-2:])
    v, w = jax.vmap(eigh_single)(x)
    v = v.reshape(batch_shape + v.shape[1:])
    w = w.reshape(batch_shape + w.shape[1:])
    return v, w


# --------------------------------------------------------------------------
# fast-path switches (see the module docstring)
_REF_ALL = bool(os.environ.get("FPEPS_REFERENCE"))
_REF_LINALG = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_LINALG"))
_REF_FINAL = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_FINAL"))
_REF_SYMM = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_SYMM"))
_REF_RING = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_RING"))
_REF_BUILD = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_BUILD"))
_REF_XFRAME = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_XFRAME"))
_EIGH_NSWEEPS = int(os.environ.get("FPEPS_EIGH_NSWEEPS", 6))
_EIGH_NSWEEPS_F64 = int(os.environ.get("FPEPS_EIGH_NSWEEPS_F64", 10))
# final-grid line count from which the fused blob kernel replaces the XLA
# pre+gather path (measured crossover between 32 and 64 lines)
_BLOB_FUSED_NLINE = int(os.environ.get("FPEPS_BLOB_FUSED_NLINE", 48))

_fpl = None
if not _REF_LINALG:
    try:
        import fpeps_pallas_linalg as _fpl_mod

        if jax.devices()[0].platform != "gpu":
            raise RuntimeError("Pallas eigh8 needs a GPU backend")
        # lowering-only probe: catches missing/renamed Pallas-Triton APIs and
        # unsupported primitives without paying the Triton LLVM compile
        jax.jit(partial(_fpl_mod.eigh8, nsweeps=2, tile=8)).lower(
            jax.ShapeDtypeStruct((8, 8, 8), jnp.float32)
        )
        _fpl = _fpl_mod
    except Exception as e:  # noqa: BLE001 - any failure means "use reference"
        warnings.warn(
            f"fPEPS fast linalg disabled, falling back to cuSOLVER: {e!r}"
        )


def _eigh_dispatch(x):
    """autoray 'linalg.eigh' entry: fused Pallas Jacobi for the batched 8x8
    blocks of the D=4 projector computation, stock eigh_stable otherwise."""
    if _fpl is not None and x.shape[-1] == 8:
        if x.dtype == jnp.float64:
            return _fpl.eigh8(x, nsweeps=_EIGH_NSWEEPS_F64, tile=128)
        if x.dtype == jnp.float32:
            return _fpl.eigh8(x, nsweeps=_EIGH_NSWEEPS, tile=128)
        if x.dtype in (jnp.float16, jnp.bfloat16):
            v, w = _fpl.eigh8(
                x.astype(jnp.float32), nsweeps=_EIGH_NSWEEPS, tile=128
            )
            return v.astype(x.dtype), w.astype(x.dtype)
    return eigh_stable(x)


def _cholesky_dispatch(x, upper=False):
    """autoray 'linalg.cholesky' entry: unrolled small-matrix cholesky
    (no cuSOLVER potrf, ~2x faster and CUDA-graph capturable)."""
    if _fpl is not None and x.shape[-1] <= 16:
        half = x.dtype in (jnp.float16, jnp.bfloat16)
        L = _fpl.chol8(x.astype(jnp.float32) if half else x)
        if half:
            L = L.astype(x.dtype)
        return jnp.swapaxes(L.conj(), -1, -2) if upper else L
    return jnp.linalg.cholesky(x, upper=upper)


ar.register_function("jax", "linalg.eigh", _eigh_dispatch)
if _fpl is not None:
    ar.register_function("jax", "linalg.cholesky", _cholesky_dispatch)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, None, None))
def _isomats2(paramsl, paramsr, chargel, charger, chi, axis):
    other_axis = 0 if axis == 1 else 1
    p = jnp.stack([paramsl, paramsr], axis=other_axis)
    c = jnp.stack([chargel, charger], axis=other_axis)
    tn = unpack(p, c, pbcx=False, pbcy=False)
    other_direction = "X" if other_axis == 0 else "Y"
    ltn = tn.select(f"{other_direction}0")
    rtn = tn.select(f"{other_direction}1")
    bix = bonds(ltn, rtn)
    reduce_opts = {"method": "cholesky", "shift": 1e-4}
    Rl = ltn.compute_reduced_factor("right", None, bix, reduce_opts=reduce_opts)
    Rr = rtn.compute_reduced_factor("left", bix, None, reduce_opts=reduce_opts)
    Pl, Pr = compute_oblique_projectors(
        Rl,
        Rr,
        max_bond=chi,
        cutoff=0.0,
        method="svd:eig",
    )
    return Pl.blocks, Pr.blocks


# deterministic patch contraction path (the operand order of every pairwise
# contraction fixes the odd-tensor ordering, which the dummy modes used to
# handle): first the two site tensors (cell A as left operand), then the
# projectors
_PATCH_PATH = [(0, 1), (0, 2), (0, 1)]


def _merge_sign(cA, cB, direction, dtype):
    """Hand sign replacing the dummy-mode phases of one pair merge.

    Contracting cell A (left) with cell B under the site-major canonical
    convention: the combine phase is (-1)^{P_A P_B}. For a y-merge the
    sub-modes of the two cells additionally interleave in canonical order
    ([A_0 B_0 A_1 B_1 ...] per sx-row instead of [A | B]), adding the
    crossing sign (-1)^{sum_{k<l} beta_k alpha_l} with alpha, beta the
    per-sx-row parities of A and B."""
    cA = cA.astype(jnp.int32)
    cB = cB.astype(jnp.int32)
    exponent = jnp.sum(cA) * jnp.sum(cB)
    if direction == "y":
        a = jnp.sum(cA, axis=1) % 2  # (subx,)
        b = jnp.sum(cB, axis=1) % 2
        suffix = jnp.cumsum(a[::-1])[::-1] - a  # sum_{l>k} alpha_l
        exponent += jnp.sum(b * suffix)
    return (1 - 2 * (exponent % 2)).astype(dtype)


@jax.vmap
def _contract2x(params, charge, P1, P2):
    params = params.reshape(2, 1, *params.shape[1:])
    charge = charge.reshape(2, 1, *charge.shape[1:])
    tn = unpack(params, charge, pbcx=False, pbcy=False)
    tn.reindex_(
        {
            "b(-1, 0)-(0, 0)": "left",
            "b(0, 0)-(1, 0)": "bond",
            "b(1, 0)-(2, 0)": "right",
            "b(0, -1)-(0, 0)": "up0",
            "b(0, 0)-(0, 1)": "down0",
            "b(1, -1)-(1, 0)": "up1",
            "b(1, 0)-(1, 1)": "down1",
        }
    )
    P1 = unpack_P1(P1)
    P2 = unpack_P2(P2)
    tn |= P1
    tn |= P2
    tn = tn.contract(optimize=_PATCH_PATH)
    tn.transpose_("left", "right", "up", "down")
    tn.data.sort_stack(inplace=True)
    tn.data.phase_sync(inplace=True)
    blocks = tn.data.blocks
    return blocks * _merge_sign(charge[0, 0], charge[1, 0], "x", blocks.dtype)


@jax.vmap
def _contract2y(params, charge, P1, P2):
    params = params.reshape(1, 2, *params.shape[1:])
    charge = charge.reshape(1, 2, *charge.shape[1:])
    tn = unpack(params, charge, pbcx=False, pbcy=False)
    tn.reindex_(
        {
            "b(0, -1)-(0, 0)": "left",
            "b(0, 0)-(0, 1)": "bond",
            "b(0, 1)-(0, 2)": "right",
            "b(-1, 0)-(0, 0)": "up0",
            "b(0, 0)-(1, 0)": "down0",
            "b(-1, 1)-(0, 1)": "up1",
            "b(0, 1)-(1, 1)": "down1",
        }
    )
    P1 = unpack_P1(P1)
    P2 = unpack_P2(P2)
    tn |= P1
    tn |= P2
    tn = tn.contract(optimize=_PATCH_PATH)
    tn.transpose_("up", "down", "left", "right")
    tn.data.sort_stack(inplace=True)
    tn.data.phase_sync(inplace=True)
    blocks = tn.data.blocks
    return blocks * _merge_sign(charge[0, 0], charge[0, 1], "y", blocks.dtype)


@partial(jax.vmap, in_axes=(0, 0, None, None), out_axes=0)
def compress_x(params, charge, chi, return_internal):
    params = jnp.swapaxes(params, 0, 1)
    params_shift = jnp.roll(params, shift=-1, axis=0)
    charge = jnp.swapaxes(charge, 0, 1)
    charge_shift = jnp.roll(charge, shift=-1, axis=0)
    Pl, Pr = _isomats2(params, params_shift, charge, charge_shift, chi, 0)
    Pr = jnp.roll(Pr, shift=1, axis=0)
    out_params = _contract2x(params, charge, Pr, Pl)
    if return_internal:
        return out_params, (Pr, Pl)
    else:
        return out_params


@partial(jax.vmap, in_axes=(1, 1, None, None), out_axes=1)
def compress_y(params, charge, chi, return_internal):
    params_shift = jnp.roll(params, shift=-1, axis=0)
    charge_shift = jnp.roll(charge, shift=-1, axis=0)
    Pl, Pr = _isomats2(params, params_shift, charge, charge_shift, chi, 1)
    Pr = jnp.roll(Pr, shift=1, axis=0)
    out_params = _contract2y(params, charge, Pr, Pl)
    if return_internal:
        return out_params, (Pr, Pl)
    else:
        return out_params


def strip_exponent(params, positional=False):
    block_axes = tuple(range(2, params.ndim))
    norm = jnp.sqrt(jnp.sum(jnp.abs(params) ** 2, axis=block_axes, keepdims=True))
    norm = jax.lax.stop_gradient(norm)
    params /= norm
    exponent = jnp.log10(norm)
    if not positional:
        exponent = jnp.sum(exponent)
    return params, exponent


def _ssa_to_linear(ssa, n):
    """Convert an SSA-form contraction path (each step referencing the input
    index or the id of a previous step's output) to opt_einsum's linear form
    (indices into the shrinking list of live tensors)."""
    ids = list(range(n))
    linear = []
    for a, b in ssa:
        ia, ib = ids.index(a), ids.index(b)
        if ia > ib:
            ia, ib = ib, ia
        linear.append((ia, ib))
        del ids[ib], ids[ia]
        ids.append(n + len(linear) - 1)
    return linear


def _line_ring_path(lx, ly):
    """Deterministic near-optimal contraction path for the row-major lx x ly
    PBC grid built by unpack(): contract each line along the shorter axis
    into a blob, then multiply the resulting ring of blobs sequentially along
    the longer axis. Replaces cotengra's randomized "auto-hq" search, which
    otherwise changes the traced HLO (and jax's persistent compilation cache
    key) on every process."""
    if ly <= lx:
        lines = [[i * ly + j for j in range(ly)] for i in range(lx)]
    else:
        lines = [[i * ly + j for i in range(lx)] for j in range(ly)]
    ssa, nid, blobs = [], lx * ly, []
    for line in lines:
        blob = line[0]
        for t in line[1:]:
            ssa.append((blob, t))
            blob = nid
            nid += 1
        blobs.append(blob)
    blob = blobs[0]
    for b in blobs[1:]:
        ssa.append((blob, b))
        blob = nid
        nid += 1
    return _ssa_to_linear(ssa, lx * ly)


def _final_order_sign(charge, dtype):
    """Hand sign replacing the dummy-mode phases of the exact grid
    contraction: (-1)^{C(P,2)} from the pairwise combine phases of the P
    odd-parity cells, times the crossing sign of restoring the canonical
    site-major order from the _line_ring_path concatenation order — within
    a grid row, the sx-rows of different cells interleave:
    Inv = sum over i, j<j', k>k' of gamma[i,j,k] * gamma[i,j',k'] with
    gamma the per-sx-row parities; when ly > lx the path concatenates cells
    column-major, adding the cross-row inversions
    sum over j<j', i>i' of pcell[i,j] * pcell[i',j']."""
    gamma = jnp.sum(charge.astype(jnp.int32), axis=3) % 2  # (lx, ly, subx)
    pcell = jnp.sum(gamma, axis=2) % 2
    P = jnp.sum(pcell)
    exponent = P * (P - 1) // 2
    lx, ly, subx = gamma.shape
    U = jnp.triu(jnp.ones((ly, ly), jnp.int32), 1)  # j < j'
    L = jnp.tril(jnp.ones((subx, subx), jnp.int32), -1)  # k > k'
    exponent += jnp.einsum("ijk,iJK,jJ,kK->", gamma, gamma, U, L)
    if ly > lx:
        Lx_ = jnp.tril(jnp.ones((lx, lx), jnp.int32), -1)  # i > i'
        exponent += jnp.einsum("ij,IJ,iI,jJ->", pcell, pcell, Lx_, U)
    return (1 - 2 * (exponent % 2)).astype(dtype)


def _reference_raw_final_contract(params, charge):
    """final_contract without the configuration-dependent reorder sign:
    the exact contraction of the remaining grid, canonically site-ordered.
    Reference implementation (quimb line-ring path); the default
    `_raw_final_contract` below is the fast parity-block version, verified
    element-wise against this one in f64."""
    tn = unpack(params, charge)
    path = _line_ring_path(*charge.shape[:2])
    sign, exponent = tn.contract(optimize=path, strip_exponent=True)
    sign = jnp.where(jnp.isnan(sign), 0.0, sign)
    sign *= _final_order_sign(charge, sign.dtype)
    return sign, exponent


def _line_blob_blocks(line_params, line_charge, horizontal):
    """Fused parity-block transfer matrix of one line of the final grid.

    Builds the line as its own small TN (PBC along the line, open ring legs),
    contracts it sequentially (same operand order as _line_ring_path), and
    fuses the ring legs in transverse order. The fuse output is already
    canonical for this construction — index order ('U', 'D'), sectors
    row-ascending (block k = row charge k), trivial phases; checked for all
    parities and revalidated by the import-time self-test — so the blocks
    are returned raw, with no transpose / sort_stack / phase_sync passes."""
    llen = line_charge.shape[0]
    if horizontal:  # line along y, ring along x
        p = line_params[None]
        c = line_charge[None]
        tn = unpack(p, c, pbcx=False, pbcy=True)
        up = [f"b(-1, {j})-(0, {j})" for j in range(llen)]
        down = [f"b(0, {j})-(1, {j})" for j in range(llen)]
        sites = [(0, j) for j in range(llen)]
    else:  # line along x, ring along y
        p = line_params[:, None]
        c = line_charge[:, None]
        tn = unpack(p, c, pbcx=True, pbcy=False)
        up = [f"b({j}, -1)-({j}, 0)" for j in range(llen)]
        down = [f"b({j}, 0)-({j}, 1)" for j in range(llen)]
        sites = [(j, 0) for j in range(llen)]
    blob = tn[tn.site_tag(*sites[0])]
    for site in sites[1:]:
        blob = tensor_contract(blob, tn[tn.site_tag(*site)], preserve_tensor=True)
    blob = blob.fuse({"U": up, "D": down})
    assert blob.inds == ("U", "D")
    return blob.data.blocks


def _strip_blocks(blocks, exponent):
    axes = tuple(range(blocks.ndim - 3, blocks.ndim))
    nrm = jnp.sqrt(jnp.sum(jnp.abs(blocks) ** 2, axis=axes, keepdims=True))
    nrm = jax.lax.stop_gradient(nrm)
    nrm = jnp.where((nrm == 0.0) | ~jnp.isfinite(nrm), 1.0, nrm)
    exponent = exponent + jnp.sum(jnp.log10(nrm))
    return blocks / nrm, exponent


def _transpose_reorder_sign(charge, dtype):
    """Sign relating the value of the exact grid contraction to that of its
    leg-transposed (x <-> y) network: (-1)^{#inversions between the X-major
    and Y-major canonical mode orders over odd sites} = (-1)^{sum over site
    pairs with X < X', Y > Y' of q[X,Y] q[X',Y']}, with q the site-charge
    grid (cell (i, j), sub-site (k, l) at X = i sx + k, Y = j sy + l).
    Prefix sums, O(N). Calibrated against the reference contraction (exact
    on all min(subx, suby) == 1 grids, 0/256 flips per shape; (2,2)-sub
    cells — a mixed x/y merge history — additionally need a within-cell
    term and are NOT covered: callers must guard)."""
    lx, ly, sx, sy = charge.shape
    q = rearrange(charge, "i j k l -> (i k) (j l)").astype(jnp.int32)
    # C[X, Y] = sum_{X' > X, Y' < Y} q[X', Y']
    colpre = jnp.concatenate(
        [jnp.zeros_like(q[:, :1]), jnp.cumsum(q, axis=1)[:, :-1]], axis=1
    )
    suf = jnp.flip(jnp.cumsum(jnp.flip(colpre, 0), axis=0), 0)
    C = jnp.concatenate([suf[1:], jnp.zeros_like(suf[:1])], axis=0)
    exponent = jnp.sum(q * C)
    return (1 - 2 * (exponent % 2)).astype(dtype)


def _fast_raw_final_contract(params, charge):
    """Fast `_raw_final_contract`: vmapped fused line blobs + a pure-jnp
    parity-block ring product (batched (2, s, s) GEMM chain).

    The +-1 sign rule was calibrated step-by-step against symmray
    tensor_contract (unique match on 8 random charge configurations):
    contracting acc (X=k, R_i=(k+P)%2) with blob (R_i, R_{i+1}) gives
    new[k] = (-1)^c acc[k] @ b[c] with c = (k+P)%2 and P the accumulated
    line parity; the closing trace follows the same rule with no extra wrap
    sign. Verified element-wise in f64 against the reference on 8x4, 8x8
    ("y" and "xyx") and 16x16 ("yy") pipelines."""
    lx, ly = charge.shape[:2]
    if (
        ly > lx
        and lx == 4
        and params.shape[-4:] == (2, 2, 2, 2)
        and min(charge.shape[-2:]) == 1
    ):
        # vertical-line grid (an "x..."-family contract path, e.g. 16x16
        # "xx" -> 4x16): evaluate the leg-transposed network instead, which
        # is horizontal and picks up the fast blob build + ring kernel; the
        # two canonical mode orders differ by the closed-form sign above.
        # (2,2)-sub cells keep the generic vertical path below.
        try:
            import fpeps_fast_layer as _FLY

            pcell = jnp.sum(
                charge.astype(jnp.int32).reshape(lx, ly, -1), axis=-1
            ) % 2
            pT = _FLY.transpose_blocks(params, pcell)
            pT = jnp.swapaxes(pT, 0, 1)
            cT = jnp.swapaxes(jnp.swapaxes(charge, 0, 1), -2, -1)
            sign, exponent = _fast_raw_final_contract(pT, cT)
            return sign * _transpose_reorder_sign(charge, sign.dtype), exponent
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"transposed final contraction unavailable ({e!r}); "
                "using the generic vertical-line path"
            )
    horizontal = ly <= lx
    if horizontal:
        line_params, line_charge = params, charge
    else:
        line_params = jnp.swapaxes(params, 0, 1)
        line_charge = jnp.swapaxes(charge, 0, 1)
    nline = line_charge.shape[0]
    p_line = jnp.sum(
        line_charge.astype(jnp.int32).reshape(nline, -1), axis=1
    ) % 2  # (nline,)

    llen = line_charge.shape[1]
    # the per-step block select (prefix parity P_i) and contraction phase
    # (-1)^c are known from the charges alone; computed up front so the
    # fused blob kernel can fold them into its output directly
    P = ((jnp.cumsum(p_line) - p_line) % 2).astype(jnp.int32)  # prefix parity
    k = jnp.arange(2, dtype=jnp.int32)[None, :]
    c = (k + P[:, None]) % 2  # (nline, 2)
    line0 = (jnp.arange(nline) == 0)[:, None]
    sgn = jnp.where(line0, 1.0, (1.0 - 2.0 * c)).astype(params.dtype)

    blobs = None
    canoned = False
    if horizontal and llen == 4 and line_params.shape[-4:] == (2, 2, 2, 2):
        try:
            import fpeps_fast_layer as _FLY

            # _fpl gate: the kernel branches need working Pallas-Triton; a
            # broken backend traces fine here and would only fail at
            # compile time, outside this try
            if (
                _fpl is not None
                and not _REF_BUILD
                and line_params.dtype in (jnp.float16, jnp.bfloat16,
                                          jnp.float32)
            ):
                # tensor-core block-dot build (factored EXT tables): 2.2x
                # over the XLA path and BITWISE equal to it (same K-order
                # accumulation), any line count — supersedes the old
                # >=48-line blob_canon dispatch. f64 keeps the XLA path
                # below (the verification domain).
                import fpeps_layer_kernels as _FK

                blobs = _FK.blob_build(line_params, line_charge, c, sgn)
                canoned = True
            else:
                # canon-composed factored einsum: same planned-pair XLA
                # path, blob assembled per charge block, no 65536-float
                # pre, no separate canon pass (also the kernel's backward
                # replay and the f64/reference route)
                blobs = _FLY.fast_line_blobs_canon_xla(
                    line_params, line_charge, c, sgn
                )
                canoned = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"fast line blobs unavailable ({e!r}); using "
                          "the generic line-blob path")
    if blobs is None:
        blobs = jax.vmap(lambda p, cc: _line_blob_blocks(p, cc, horizontal))(
            line_params, line_charge
        )  # (nline, 2, s, s)
    if not canoned:
        blobs = jnp.take_along_axis(
            blobs, c[:, :, None, None].astype(jnp.int32), axis=1
        ) * sgn.astype(blobs.dtype)[:, :, None, None]

    # Blobs are NOT individually normalized (that costs a full reduce pass
    # over all blob data); instead the (tiny) ring accumulator is stripped
    # every 2 steps, which keeps fp32 in range since site tensors enter
    # pre-stripped (blob magnitudes ~<= 1e3). NOTE the cadence is
    # load-bearing: every 4 steps already overflows the fp32 strip norm.
    exponent = jnp.zeros((), params.dtype)

    sign = None
    if (
        _fpl is not None
        and not _REF_RING
        and nline >= 2
        and blobs.shape == (nline, 2, 128, 128)
        and blobs.dtype in (jnp.float16, jnp.bfloat16, jnp.float32)
    ):
        try:
            import fpeps_layer_kernels as _FK

            # fused ring kernel: the accumulator stays in registers for the
            # whole chain and each blob streams from HBM once, vs nline-1
            # separate batched GEMMs with acc/strip round-trips (2.9-3.1x
            # at 16/32 lines). f64 keeps the jnp loop below (the kernel is
            # fp32-compute) and remains the verification path.
            sign, exponent = _FK.ring_chain(blobs)
        except Exception as e:  # noqa: BLE001
            sign = None
            warnings.warn(f"fused ring kernel unavailable ({e!r}); "
                          "using the jnp ring")
    if sign is None:
        # fp16's narrow range needs a strip every step; wider dtypes every 2
        cadence = 1 if blobs.dtype in (jnp.float16, jnp.bfloat16) else 2
        acc = blobs[0]
        for i in range(1, nline - 1):
            acc = jnp.einsum("kij,kjl->kil", acc, blobs[i])
            if i % cadence == 0:
                acc, exponent = _strip_blocks(acc, exponent)
        # closing step: same rule, trace instead of full product
        sign = jnp.sum(acc * jnp.swapaxes(blobs[nline - 1], -1, -2))

    sign = jnp.where(jnp.isnan(sign), 0.0, sign)
    sign *= _final_order_sign(charge, sign.dtype)
    return sign, exponent


def _final_selftest():
    """Compare fast vs reference final contraction on tiny random grids
    (both line orientations, random EVEN-total-parity charges — physical
    configurations always have even total fermion parity; odd networks
    contract to 0). Guards against symmray versions changing the fuse
    conventions `_line_blob_blocks` relies on. The (8, 4) grid exercises
    the llen=4 production path: planned line blobs with canon-composed
    extraction tables + the fused ring-chain kernel; the (4, 8) grid the
    transposed evaluation of vertical-line grids ("x..."-family paths)."""
    rng = np.random.default_rng(0)
    for lx, ly in ((4, 2), (2, 4), (8, 4), (4, 8)):
        params = jnp.asarray(
            rng.normal(size=(8, lx, ly, 8, 2, 2, 2, 2)), jnp.float32
        )
        charge = rng.integers(0, 2, size=(8, lx, ly, 1, 1))
        charge[:, 0, 0, 0, 0] ^= charge.reshape(8, -1).sum(1) % 2  # even total
        charge = jnp.asarray(charge, jnp.int8)
        s_ref, e_ref = jax.vmap(_reference_raw_final_contract)(params, charge)
        s_new, e_new = jax.vmap(_fast_raw_final_contract)(params, charge)
        dl = jnp.abs(
            jnp.log10(jnp.abs(s_ref)) + e_ref - jnp.log10(jnp.abs(s_new)) - e_new
        )
        # random unstructured tensors cancel heavily, so fp32 noise on
        # near-cancellation samples reaches ~0.1 in log10 (~0.4 on the
        # bigger (8, 4) grid); a convention break gives O(1) errors on most
        # samples (median catches it) and ~half the signs flipped
        if (
            bool(jnp.any(jnp.sign(s_ref) != jnp.sign(s_new)))
            or float(jnp.median(dl)) > 0.01
            or float(jnp.max(dl)) > (3.0 if lx * ly > 8 else 1.0)
        ):
            return False
    return True


def _selftest_code_tag():
    """md5 of the fast-path sources + calibrated tables, mixed into the
    self-test cache keys: a version-triple-only key lets a PASS marker
    written by OLD code suppress the self-test for NEW code (and a marker
    written by the healthy ranks of a half-broken env poisons the rest),
    so code changes must invalidate the marker."""
    import hashlib

    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.md5()
    for name in (
        "fPEPS_hotrg_symm.py",
        "fpeps_fast_layer.py",
        "fpeps_layer_kernels.py",
        "fpeps_pallas_linalg.py",
        "fpeps_blob_tables.npz",
    ):
        try:
            with open(os.path.join(here, name), "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(name.encode())
    return h.hexdigest()[:10]


def _selftest_lib_tag():
    import symmray
    import quimb

    return (
        f"{getattr(symmray, '__version__', '?')}_{jax.__version__}"
        f"_{getattr(quimb, '__version__', '?')}_{_selftest_code_tag()}"
    )


def _selftest_cache_dir():
    return os.path.expanduser(os.environ.get("FPEPS_CACHE_DIR") or "~/.cache")


def _selftest_cache_path():
    return os.path.join(
        _selftest_cache_dir(),
        f"fpeps_final_selftest_{_selftest_lib_tag()}.ok",
    )


if _REF_FINAL:
    _raw_final_contract = _reference_raw_final_contract
else:
    _raw_final_contract = _fast_raw_final_contract
    if not os.environ.get("FPEPS_SKIP_SELFTEST"):
        # the self-test costs ~15 s, so cache the PASS per library-version set
        _cache = _selftest_cache_path()
        if not os.path.exists(_cache):
            try:
                ok = _final_selftest()
            except Exception as e:  # noqa: BLE001
                ok = False
                warnings.warn(
                    f"fPEPS fast final contraction self-test errored: {e!r}"
                )
            if ok:
                try:
                    os.makedirs(os.path.dirname(_cache), exist_ok=True)
                    with open(_cache, "w") as f:
                        f.write("ok\n")
                except OSError:
                    pass
            else:
                warnings.warn(
                    "fPEPS fast final contraction self-test FAILED "
                    "(symmray fuse conventions changed?); using the reference "
                    "implementation"
                )
                _raw_final_contract = _reference_raw_final_contract


# --------------------------------------------------------------------------
# default-on fused Pallas layer kernels: compress_y at D = chi = 4 routes
# through fpeps_fast_layer.fast_compress_y_k (merged gram+cholesky,
# projector assembly and patch as register kernels; differentiable via
# jnp-reference custom_vjp, nested-vmap-safe via custom_vmap), and
# compress_x through its transposed-frame wrapper fast_compress_x_k (the
# same kernels on leg-transposed tensors) when the cells have
# min(subx, suby) == 1, i.e. pure-x / pure-y contract paths; x-merges of
# mixed-history cells stay on the stock path.
_REF_LAYER = _REF_ALL or bool(os.environ.get("FPEPS_REFERENCE_LAYER"))

_compress_y_stock = compress_y
_compress_x_stock = compress_x


def _layer_selftest(FLY):
    """Compare the fused-kernel compress_y against the stock path at PSI
    level: one y-merge of random 4x4 grids, then the exact reference
    contraction of the 4x2 result. The comparison must close the network —
    per-tensor output blocks differ by the projector sign gauge, which only
    cancels pairwise across bonds. Guards the calibrated sign tables and the
    mirrored svd:eig recipe against symmray/quimb convention changes."""
    rng = np.random.default_rng(0)
    B, lx, nk = 8, 4, 2
    params = jnp.asarray(
        rng.normal(size=(B, lx, nk, 2, 8, 2, 2, 2, 2)), jnp.float32
    )
    charge = rng.integers(0, 2, size=(B, lx, nk, 2, 1, 1))
    charge[:, 0, 0, 0, 0, 0] ^= charge.reshape(B, -1).sum(1) % 2  # even total
    charge = jnp.asarray(charge, jnp.int8)
    mcharge = rearrange(charge, "b x y y1 x0 y0 -> b x y x0 (y1 y0)")

    def psi(cy, p, c, mc):
        return _reference_raw_final_contract(cy(p, c, 4, False), mc)

    s_ref, e_ref = jax.jit(jax.vmap(partial(psi, _compress_y_stock)))(
        params, charge, mcharge
    )
    s_new, e_new = jax.jit(jax.vmap(partial(psi, FLY.fast_compress_y_k)))(
        params, charge, mcharge
    )
    dl = jnp.abs(
        jnp.log10(jnp.abs(s_ref)) + e_ref - jnp.log10(jnp.abs(s_new)) - e_new
    )
    # same criteria as _final_selftest: a convention break gives O(1) errors
    # and flipped signs; fp32 noise on cancellation-heavy samples stays small
    return not (
        bool(jnp.any(jnp.sign(s_ref) != jnp.sign(s_new)))
        or float(jnp.median(dl)) > 0.01
        or float(jnp.max(dl)) > 1.0
    )


def _layerx_selftest(FLY):
    """Same psi-level guard for the transposed-frame compress_x: layer-1
    ((1,1) cells) and layer-2 ((2,1) cells, inputs from a REAL stock
    layer-1 merge) x-merges of random 8x4 grids, closed by the reference
    contraction and compared against the stock compress_x."""
    rng = np.random.default_rng(0)
    B = 8
    params = jnp.asarray(
        rng.normal(size=(B, 8, 4, 8, 2, 2, 2, 2)), jnp.float32
    )
    charge = rng.integers(0, 2, size=(B, 8, 4, 1, 1))
    charge[:, 0, 0, 0, 0] ^= charge.reshape(B, -1).sum(1) % 2  # even total
    charge = jnp.asarray(charge, jnp.int8)

    def close(cx, p, c):
        lx, ly, sx, sy = c.shape
        p4 = p.reshape(lx // 2, 2, ly, *p.shape[2:])
        c4 = c.reshape(lx // 2, 2, ly, sx, sy)
        out = cx(p4, c4, 4, False)
        mc = rearrange(c4, "x q y sx sy -> x y (q sx) sy")
        return out, mc

    # separate jits so the (expensive) reference-final trace compiles once
    # per grid shape and is shared between the stock and fast psi
    psi = jax.jit(jax.vmap(_reference_raw_final_contract))
    for _ in range(2):  # layer 1 ((1,1) cells), then layer 2 ((2,1) cells)
        out_ref, mch = jax.jit(
            jax.vmap(partial(close, _compress_x_stock))
        )(params, charge)
        out_new, _ = jax.jit(
            jax.vmap(partial(close, FLY.fast_compress_x_k))
        )(params, charge)
        s_ref, e_ref = psi(out_ref, mch)
        s_new, e_new = psi(out_new, mch)
        dl = jnp.abs(
            jnp.log10(jnp.abs(s_ref)) + e_ref
            - jnp.log10(jnp.abs(s_new)) - e_new
        )
        if (
            bool(jnp.any(jnp.sign(s_ref) != jnp.sign(s_new)))
            or float(jnp.median(dl)) > 0.01
            or float(jnp.max(dl)) > 1.0
        ):
            return False
        params, _ = jax.vmap(strip_exponent)(out_ref)
        charge = mch
    return True


def _layer_selftest_cache_path(tag="layer"):
    return os.path.join(
        _selftest_cache_dir(),
        f"fpeps_{tag}_selftest_{_selftest_lib_tag()}.ok",
    )


def _run_cached_selftest(tag, fn, fail_msg):
    """Run `fn` once per (library, code-hash) tuple, caching a PASS marker;
    returns False (with a warning) on failure or error."""
    if os.environ.get("FPEPS_SKIP_SELFTEST"):
        return True
    cache = _layer_selftest_cache_path(tag)
    if os.path.exists(cache):
        return True
    try:
        ok = fn()
    except Exception as e:  # noqa: BLE001
        ok = False
        warnings.warn(f"fPEPS {tag} self-test errored: {e!r}")
    if ok:
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w") as f:
                f.write("ok\n")
        except OSError:
            pass
    else:
        warnings.warn(fail_msg)
    return ok


_FLY_LAYER = None
_FLY_LAYER_X = None
if not _REF_LAYER and _fpl is not None:  # layer kernels need working Triton
    try:
        import fpeps_fast_layer as _FLY_mod

        if not _run_cached_selftest(
            "layer",
            lambda: _layer_selftest(_FLY_mod),
            "fPEPS fused layer kernel self-test FAILED (symmray conventions "
            "changed?); compress_y stays on the stock path",
        ):
            _FLY_mod = None
        _FLY_LAYER = _FLY_mod
        if _FLY_LAYER is not None and _run_cached_selftest(
            "layerx",
            lambda: _layerx_selftest(_FLY_LAYER),
            "fPEPS transposed-frame compress_x self-test FAILED; compress_x "
            "stays on the stock path",
        ):
            _FLY_LAYER_X = _FLY_LAYER
    except Exception as e:  # noqa: BLE001 - any failure means "use stock"
        warnings.warn(
            f"fPEPS fused layer kernels disabled, using stock compress_y: {e!r}"
        )


def _y_crossing_sign(charge):
    """The sub-row interleave part of _merge_sign(direction="y") —
    (-1)^{sum_k b_k sum_{l>k} a_l}, a/b the per-subx-row parities of the
    two cells. The pi-indexed kernel sign tables carry only the parity
    product (-1)^{P_A P_B}, which is the whole y merge sign at subx == 1
    (any pure-y path); y-merges of cells with an x-merge history (mixed
    contract paths) need this remainder. charge (..., nk, 2, sx, sy) ->
    +-1 (..., nk)."""
    c = charge.astype(jnp.int32)
    a = jnp.sum(c[..., 0, :, :], axis=-1) % 2  # (..., nk, sx)
    b = jnp.sum(c[..., 1, :, :], axis=-1) % 2
    suffix = jnp.flip(jnp.cumsum(jnp.flip(a, -1), axis=-1), -1) - a
    return 1 - 2 * (jnp.sum(b * suffix, axis=-1) % 2)


def compress_y(params, charge, chi, return_internal=False):
    """Stock compress_y, routed through the fused Pallas layer kernels when
    applicable (D = chi = 4 y-merge: params (..lx, nk, 2, 8, 2,2,2,2)).
    fp16/bf16 inputs run the kernels in fp32 (like the eigh dispatch).
    Disable with FPEPS_REFERENCE_LAYER=1 (or FPEPS_REFERENCE=1)."""
    if (
        _FLY_LAYER is not None
        and chi == 4
        and not return_internal
        and params.ndim == 8
        and params.shape[2] == 2
        and params.shape[3:] == (8, 2, 2, 2, 2)
        and params.dtype in (jnp.float16, jnp.bfloat16, jnp.float32,
                             jnp.float64)
    ):
        half = params.dtype in (jnp.float16, jnp.bfloat16)
        out = _FLY_LAYER.fast_compress_y_k(
            params.astype(jnp.float32) if half else params, charge, chi
        )
        if charge.shape[-2] > 1:  # subx > 1: cells carry an x-merge history
            out = out * _y_crossing_sign(charge).astype(out.dtype)[
                ..., None, None, None, None, None
            ]
        return out.astype(params.dtype) if half else out
    return _compress_y_stock(params, charge, chi, return_internal)


def compress_x(params, charge, chi, return_internal=False):
    """Stock compress_x, routed through the transposed-frame fused y-merge
    kernels when applicable (D = chi = 4 x-merge of min(subx, suby) == 1
    cells, i.e. any pure-x / pure-y merge history: params
    (lx2, 2, ly, 8, 2,2,2,2)). fp16/bf16 inputs run the kernels in fp32.
    Disable with FPEPS_REFERENCE_LAYER=1 (or FPEPS_REFERENCE=1)."""
    if (
        _FLY_LAYER_X is not None
        and chi == 4
        and not return_internal
        and params.ndim == 8
        and params.shape[1] == 2
        and params.shape[3:] == (8, 2, 2, 2, 2)
        and min(charge.shape[-2:]) == 1
        and params.dtype in (jnp.float16, jnp.bfloat16, jnp.float32,
                             jnp.float64)
    ):
        if params.dtype in (jnp.float16, jnp.bfloat16):
            out = _FLY_LAYER_X.fast_compress_x_k(
                params.astype(jnp.float32), charge, chi
            )
            return out.astype(params.dtype)
        return _FLY_LAYER_X.fast_compress_x_k(params, charge, chi)
    return _compress_x_stock(params, charge, chi, return_internal)


def _pure_x_frame_ok(contract_path, chi, params, charge):
    """Whether a contract path can run as a pure-y path of the leg-transposed
    lattice with ONE up-front transpose (see the module docstring): pure-x
    path, the transposed-frame layer kernels available (D = chi = 4, float
    blocks (8, 2,2,2,2), (1,1) site cells), and a final grid that is
    horizontal in the transposed frame (Lx / 2^k <= Ly), so
    `_fast_raw_final_contract` does not transpose it back."""
    if _REF_XFRAME or _FLY_LAYER_X is None:
        return False
    if not contract_path or set(contract_path) != {"x"}:
        return False
    if isinstance(chi, int):
        chi = (chi,) * len(contract_path)
    if any(c != 4 for c in chi):
        return False
    if params.shape[-5:] != (8, 2, 2, 2, 2) or charge.shape[-2:] != (1, 1):
        return False
    if params.dtype not in (jnp.float16, jnp.bfloat16, jnp.float32,
                            jnp.float64):
        return False
    lx, ly = charge.shape[:2]
    return lx % (2 ** len(contract_path)) == 0 and lx // 2 ** len(contract_path) <= ly


def _transpose_site_frame(params, charge):
    """Leg-transpose every site tensor ((l, r, u, d) -> (u, d, l, r), static
    per-parity sector map + Koszul sign, `fpeps_fast_layer.transpose_blocks`)
    and swap the lattice axes: params (lx, ly, 8, ...) -> (ly, lx, 8, ...),
    charge (lx, ly, sx, sy) -> (ly, lx, sy, sx). An x-merge of the original
    lattice is exactly a y-merge of this frame (delta == 1, calibrated in
    fpeps_fast_layer), and `transpose_blocks` is an involution, so a pure-x
    pipeline needs this transpose once instead of in/out at every layer."""
    import fpeps_fast_layer as _FLY

    p = jnp.sum(charge.astype(jnp.int32), axis=(-2, -1)) % 2
    pT = _FLY.transpose_blocks(params, p)
    cT = jnp.swapaxes(jnp.swapaxes(charge, 0, 1), -2, -1)
    return jnp.swapaxes(pT, 0, 1), cT


def _xframe_final_sign(charge_T, dtype):
    """`_transpose_reorder_sign` of the ORIGINAL-frame final charge grid,
    given the transposed-frame one (ly, lx, sy, sx)."""
    c_orig = jnp.swapaxes(jnp.swapaxes(charge_T, 0, 1), -2, -1)
    return _transpose_reorder_sign(c_orig, dtype)


def final_contract(s, params, charge, exponent, xframe=False):
    sign, final_exponent = _raw_final_contract(params, charge)
    if xframe:
        sign *= _xframe_final_sign(charge, sign.dtype)
    sign *= qtx.nn.fermion_reorder_sign(s)
    exponent += final_exponent
    exponent *= jnp.log(10).astype(exponent.dtype)
    return LogArray(sign, exponent)


@lru_cache
def _trans_symm(sector: int = 0):
    """Cached TransND, kept out of the eqx.Module so its internal arrays
    don't become model parameters."""
    return qtx.symmetry.TransND(sector=sector)


def _trans_perm_signs_all(s):
    """sign(s, T_t) of EVERY translation in lexicographic (kx, ky) order —
    closed-form replacement of quantax's generic ``_permutation_sign``
    (per-element argsort + cycle parity), which dominates the
    SymmNeuralPEPS forward (59% at 32x32, 75% at 64x64).

    Derivation: the mode permutation of a translation acts identically on
    the up and down blocks, so occupied-mode destinations never mix blocks
    and the sign factorizes over species. quantax's convention (perm[i] =
    i + k rows: T_g s = s[perm]) sends occupied mode (x, y) to
    ((x - kx) % Lx, (y - ky) % Ly): the FIRST kx rows / ky columns wrap to
    the end. Per species with occupation grid O in x-major order the
    inversion parity is a cross-row block swap NB (n - NB) (NB = occupancy
    of the first kx rows) plus within-row rolls sum_x b_x (n_x - b_x)
    (b_x = occupancy of the first ky columns of row x) — all prefix sums,
    O(L^2) total. NOTE the direction matters: the occupied SET is not
    transported, so sign(g) != sign(g^-1) in general. Verified element-wise
    against quantax's _permutation_sign."""
    lattice = qtx.get_lattice()
    Lx, Ly = lattice.shape[1:]
    occ = (s > 0).reshape(2, Lx, Ly).astype(jnp.int32)
    parity = jnp.zeros((Lx, Ly), jnp.int32)
    for spin in range(2):
        O = occ[spin]
        n_x = O.sum(1)  # (Lx,)
        n = n_x.sum()
        # NB(kx) = occupancy of the first kx rows (those that wrap to the end)
        NB = jnp.concatenate(
            [jnp.zeros(1, jnp.int32), jnp.cumsum(n_x)]
        )[:Lx]
        cross = NB * (n - NB)  # (Lx,) indexed by kx
        # b_x(ky) = occupancy of the first ky columns of row x
        bx = jnp.concatenate(
            [jnp.zeros((Lx, 1), jnp.int32), jnp.cumsum(O, axis=1)],
            axis=1,
        )[:, :Ly]
        within = jnp.sum(bx * (n_x[:, None] - bx), axis=0)  # (Ly,) by ky
        parity = parity + cross[:, None] + within[None, :]
    return 1 - 2 * (parity.reshape(-1) % 2)


def _trans_symmetrize(psi, s):
    """TransND(sector=0).symmetrize with the translation signs from the
    closed form above; otherwise identical (same character normalization)."""
    symm = _trans_symm(0)
    sign = _trans_perm_signs_all(s)
    character = sign * symm._character
    character = (character * character[0] / character.size).astype(psi.dtype)
    return (psi * character).sum()


def _reorder_signs_all(s):
    """fermion_reorder_sign(T_t s) for every translation t at once, without
    materialising the (Lx*Ly, 2*N) array of translated configurations.

    The sign counts the (up_i, dn_j) pairs with j < i in the site-major
    order of the translated grid. The count splits into a cross-row term
    (up row sums x prefix down row sums, depends only on kx) and a
    within-row term (depends only on ky); both are evaluated for all shifts
    with static cyclic-order masks. Returns a (Lx*Ly,) +-1 array in
    lexicographic (kx, ky) order."""
    lattice = qtx.get_lattice()
    Lx, Ly = lattice.shape[1:]
    occ = (s > 0).reshape(2, Lx, Ly).astype(jnp.int32)
    U, Dn = occ[0], occ[1]

    posx = (np.arange(Lx)[None, :] - np.arange(Lx)[:, None]) % Lx
    Wx = jnp.asarray(posx[:, :, None] > posx[:, None, :])  # [k, i, j]: j before i
    posy = (np.arange(Ly)[None, :] - np.arange(Ly)[:, None]) % Ly
    Wy = jnp.asarray(posy[:, :, None] > posy[:, None, :])

    cross_row = jnp.einsum("i,j,kij->k", U.sum(1), Dn.sum(1), Wx)  # (Lx,)
    within_row = jnp.einsum("xy,xz,kyz->k", U, Dn, Wy)  # (Ly,)
    parity = (cross_row[:, None] + within_row[None, :]).reshape(-1) % 2
    return 1 - 2 * parity


def shared_transavg_psis(s, params, charge, contract_path, chi):
    """psi(T_t s) for every translation t, sharing HOTRG layers between
    translations that only differ by whole-block rolls.

    :param s: the (unshifted) configuration itself.
    :param params: the (Lx, Ly, ...) site-tensor grid of s, as returned by
        get_params_charge and cast to the TN dtype, before strip_exponent.
        Translation covariance of the net guarantees the grid of T_t s is
        roll(params, (-kx, -ky)).
    :param charge: the (Lx, Ly, 1, 1) charge grid of s.
    :return: a length-Lx*Ly LogArray in quantax's lexicographic translation
        order, i.e. entry kx*Ly + ky is psi(T_{(kx,ky)} s).
    """
    if isinstance(chi, int):
        chi = (chi,) * len(contract_path)

    params, exponent = strip_exponent(params)
    # pure-x paths: ONE leg transpose up front, then the whole pipeline is
    # the pure-y pipeline of the transposed lattice (original x = frame axis
    # 1); the rep bookkeeping (bx, by, parent, shift, reps) stays in the
    # original frame, only the data layout is transposed
    xframe = _pure_x_frame_ok(contract_path, chi, params, charge)
    if xframe:
        params, charge = _transpose_site_frame(params, charge)
    V = params[None]  # (R, lx, ly, *block)   [xframe: (R, ly, lx, *block)]
    C = charge[None]  # (R, lx, ly, subx, suby)   [xframe: (R, ly, lx, sy, sx)]
    EX = exponent[None]
    bx = by = 1

    for direction, chi_k in zip(contract_path, chi):
        lx, ly = V.shape[1:3]
        pbx, pby = bx, by
        if direction == "x":
            bx *= 2
        else:
            by *= 2
        # new representatives r = (rx, ry), rx < bx, ry < by: the level input
        # of r is the parent rep (rx mod pbx, ry mod pby) rolled by one grid
        # row/column when the residue crosses the old block
        rx, ry = np.indices((bx, by))
        parent = ((rx % pbx) * pby + ry % pby).reshape(-1)
        shift = jnp.asarray((rx // pbx if direction == "x" else ry // pby).reshape(-1))
        ax = 1 if xframe else (0 if direction == "x" else 1)
        merge = "y" if xframe else direction

        Vp = jax.vmap(lambda v, m: jnp.roll(v, -m, axis=ax))(V[parent], shift)
        Cp = jax.vmap(lambda c, m: jnp.roll(c, -m, axis=ax))(C[parent], shift)
        EX = EX[parent]

        R = len(parent)
        subx, suby = Cp.shape[-2:]
        if merge == "x":
            Vp = Vp.reshape(R, lx // 2, 2, ly, *Vp.shape[3:])
            Cp = Cp.reshape(R, lx // 2, 2, ly, subx, suby)
            V = jax.vmap(lambda p, c: compress_x(p, c, chi_k, False))(Vp, Cp)
            C = rearrange(Cp, "r x x1 y x0 y0 -> r x y (x1 x0) y0")
        else:
            Vp = Vp.reshape(R, lx, ly // 2, 2, *Vp.shape[3:])
            Cp = Cp.reshape(R, lx, ly // 2, 2, subx, suby)
            V = jax.vmap(lambda p, c: compress_y(p, c, chi_k, False))(Vp, Cp)
            C = rearrange(Cp, "r x y y1 x0 y0 -> r x y x0 (y1 y0)")
        V, en = jax.vmap(strip_exponent)(V)
        EX = EX + en

    # final level: one exact contraction per representative. Translations of
    # the same residue class contract the SAME network, only with the grid
    # (and hence the canonical site order of the fermionic dummy modes)
    # rolled by whole blocks — their amplitude differs from the
    # representative's by the sign of cyclically relabeling the odd modes,
    # computable from the charge grid alone.
    raw_sign, raw_exp = jax.vmap(_raw_final_contract)(V, C)
    if xframe:
        # back to the original-frame canonical mode order (one closed-form
        # sign per representative, from the final charge grid alone)
        raw_sign = raw_sign * jax.vmap(
            lambda c: _xframe_final_sign(c, raw_sign.dtype)
        )(C)
        charge = jnp.swapaxes(jnp.swapaxes(charge, 0, 1), -2, -1)
    EX = EX + raw_exp

    Lx, Ly = charge.shape[:2]
    c = charge.reshape(Lx, Ly).astype(jnp.int32)  # charge grid of s itself
    n_row = c.sum(axis=1)
    Ntot = c.sum()
    reorder = _reorder_signs_all(s)

    # translation (kx, ky) rolls the rep grid of (kx mod bx, ky mod by) by
    # (kx - rx, ky - ry) sites in x-major site order: the x-roll moves rows
    # rx..kx-1 past the rest and its parity NB*(Ntot-NB) depends only on kx;
    # the y-roll moves columns ry..ky-1 past the rest within every row and
    # its parity sum_x b_x*(n_x-b_x) depends only on ky. Both follow from
    # prefix sums of the charge grid, vectorized over all translations.
    kx = np.arange(Lx)
    ky = np.arange(Ly)
    zero = jnp.zeros(1, jnp.int32)
    row_cum = jnp.concatenate([zero, jnp.cumsum(n_row)])
    NB = row_cum[kx] - row_cum[kx % bx]  # (Lx,)
    parity_x = NB * (Ntot - NB)
    col_cum = jnp.concatenate(
        [jnp.zeros((Lx, 1), jnp.int32), jnp.cumsum(c, axis=1)], axis=1
    )
    b = col_cum[:, ky] - col_cum[:, ky % by]  # (Lx, Ly)
    parity_y = jnp.sum(b * (n_row[:, None] - b), axis=0)  # (Ly,)
    parity = (parity_x[:, None] + parity_y[None, :]).reshape(-1) % 2
    sigma = (1 - 2 * parity).astype(raw_sign.dtype)

    reps = ((kx[:, None] % bx) * by + ky[None, :] % by).reshape(-1)
    sign = raw_sign[reps] * sigma * reorder
    exponent = EX[reps] * jnp.log(10).astype(sign.dtype)
    return LogArray(sign, exponent)


class NeuralPEPS(eqx.Module):
    """Plain NeuralPEPS: CNN-generated site tensors, HOTRG layers along
    ``contract_path``, exact contraction of the remaining grid."""

    net: eqx.Module
    D: int
    chi: int
    contract_path: str
    dtype: DTypeLike

    def __init__(
        self,
        net: eqx.Module,
        D: int,
        chi: int = None,
        contract_path: str = "",
        dtype: DTypeLike = jnp.float32,
    ):
        self.net = net
        self.D = D
        self.chi = D if chi is None else chi
        self.contract_path = contract_path
        self.dtype = dtype

    def __call__(self, s: jax.Array) -> LogArray:
        x = self.net(s)
        x /= jnp.sqrt(self.D)
        halfD = self.D // 2
        axes = {"sec": 16, "u": halfD, "d": halfD, "l": halfD, "r": halfD, "phys": 2}
        x = rearrange(x, "(sec u d l r phys) Lx Ly -> Lx Ly sec u d l r phys", **axes)
        params, charge = get_params_charge(x, s)
        params = params.astype(self.dtype)
        params, exponent = strip_exponent(params)
        # pure-x paths run as the pure-y pipeline of the leg-transposed
        # lattice after ONE transpose (see shared_transavg_psis)
        xframe = _pure_x_frame_ok(self.contract_path, self.chi, params, charge)
        if xframe:
            params, charge = _transpose_site_frame(params, charge)

        for direction in self.contract_path:
            lx, ly, subx, suby = charge.shape
            if direction == "x" and not xframe:
                params = params.reshape(lx // 2, 2, ly, *params.shape[2:])
                charge = charge.reshape(lx // 2, 2, ly, subx, suby)
                params = compress_x(params, charge, self.chi, False)
                charge = rearrange(charge, "x x1 y x0 y0 -> x y (x1 x0) y0")
            else:  # direction == "y"
                params = params.reshape(lx, ly // 2, 2, *params.shape[2:])
                charge = charge.reshape(lx, ly // 2, 2, subx, suby)
                params = compress_y(params, charge, self.chi, False)
                charge = rearrange(charge, "x y y1 x0 y0 -> x y x0 (y1 y0)")
            params, new_exponent = strip_exponent(params)
            exponent += new_exponent

        return final_contract(s, params, charge, exponent, xframe=xframe)


class SymmNeuralPEPS(eqx.Module):
    """NeuralPEPS returning the zero-momentum translation average of psi,
    with the HOTRG layers shared between translations.

    Requires a translation-covariant ``net`` (Embedding sublattice=None,
    circular convolutions). Equals Variational(NeuralPEPS(...),
    symm=TransND(0)) at a fraction of the cost. Further projections that
    preserve zero momentum (e.g. Flip(axis=0) @ Flip(axis=1) @
    SpinInverse()) go into ``symm`` of the surrounding Variational, which
    then averages this translation-invariant amplitude over that group;
    ``jax.vmap(state.model)`` returns the amplitude without them.

    The fields mirror NeuralPEPS exactly (the symmetry itself is not part of
    the pytree), so weights saved from one class load into the other. The
    ``contract_path`` is not stored with the weights but is part of the
    state: a checkpoint is only valid with the path it was trained with.
    """

    net: eqx.Module
    D: int
    chi: int
    contract_path: str
    dtype: DTypeLike

    def __init__(
        self,
        net: eqx.Module,
        D: int,
        chi: int = None,
        contract_path: str = None,
        dtype: DTypeLike = jnp.float32,
    ):
        self.net = net
        self.D = D
        self.chi = D if chi is None else chi
        if contract_path is None:
            contract_path = default_contract_path()
        self.contract_path = contract_path
        self.dtype = dtype
        _trans_symm(0)  # prime the cache outside jit

    def __call__(self, s: jax.Array) -> LogArray:
        """The translation-averaged psi(s), from the shared psi(T_t s)."""
        x = self.net(s)
        x /= jnp.sqrt(self.D)
        halfD = self.D // 2
        axes = {"sec": 16, "u": halfD, "d": halfD, "l": halfD, "r": halfD, "phys": 2}
        x = rearrange(x, "(sec u d l r phys) Lx Ly -> Lx Ly sec u d l r phys", **axes)
        params, charge = get_params_charge(x, s)
        params = params.astype(self.dtype)

        psi = shared_transavg_psis(s, params, charge, self.contract_path, self.chi)
        if _REF_SYMM:
            return _trans_symm(0).symmetrize(psi, s)
        return _trans_symmetrize(psi, s)
