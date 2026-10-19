"""Pallas (Triton) kernels for the occupied block of the Pfaffian-backflow pairing matrix.

Instead of building the full (2N, 2N) pairing matrix and slicing it, the kernels compute

    F[p, q] = sum_a Y[p, a] * Wt[2 * spin_p + spin_q, site_p - site_q, a] * Y[q, a]

over the 2 Ne occupied modes ``idx = fermion_idx(s)`` only, with ``Y = y[idx]`` and ``Wt``
the four spin blocks of W times the distance envelope. ``F`` is exactly antisymmetric. The
``custom_vjp`` backward is two more Pallas kernels (no atomics), and everything batches
under ``jax.vmap``, including the per-sample Jacobian of quantax. All products are fp32 on
CUDA cores.

Public API: ``PairGeometry``, ``build_wtilde`` (parameters -> Wt), ``pairing_occ`` (GPU
kernel), ``pairing_occ_reference`` (pure JAX) and ``pairing_matrix_occ`` (features -> F,
using the reference off GPU).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

try:
    from jax.experimental import pallas as pl
    import jax.experimental.pallas.triton as plgpu
except ImportError:  # jax builds without Pallas/Triton
    pl = plgpu = None

# Register tiles: each kernel gathers a (TP, TQ, TA) or (TP, TR, TA) fp32 tile of W rows,
# TA being the channel chunk.
PAD = 32  # 2 Ne is padded to a multiple of this; every TP below must divide it
FWD_TILES = dict(TP=16, TQ=32, TA=16, warps=4)
DY_TILES = dict(TP=32, TQ=32, TA=8, warps=8)
DW_TILES = dict(TP=32, TR=16, TA=16, warps=4)


def kernels_available() -> bool:
    """Whether the Pallas kernels can run, i.e. Pallas/Triton is importable and the
    default backend is a GPU."""
    return pl is not None and jax.default_backend() == "gpu"


def _tiles(table, d, *keys):
    """Tile sizes from ``table``; the channel chunk TA is clamped to d."""
    vals = [table[k] for k in keys]
    if "TA" in keys:
        i = keys.index("TA")
        vals[i] = min(vals[i], d)
        assert d % vals[i] == 0, f"d={d} must be a multiple of TA={vals[i]}"
    return vals


class PairGeometry(NamedTuple):
    """Static description of the pairing problem (hashable, usable as nondiff arg)."""

    LX: int
    LY: int
    d: int
    n_up: int
    n_dn: int

    @property
    def N(self) -> int:
        return self.LX * self.LY

    @property
    def n_occ(self) -> int:
        return self.n_up + self.n_dn

    @property
    def n_pad(self) -> int:
        return -(-self.n_occ // PAD) * PAD


# ------------------------------------------------------------------ static tables


def displacement_tables(geom):
    """(DISP_IDX (N, N), NEG_IDX (N,), DIST_R (N,)) in the row-major site order
    ``i = x * LY + y``; ``DIST_R[r]`` is the minimum-image length of displacement r."""
    LX, LY = geom.LX, geom.LY
    xy = np.stack(np.meshgrid(np.arange(LX), np.arange(LY), indexing="ij"), -1)
    xy = xy.reshape(-1, 2)
    shape = np.array([LX, LY])
    dxy = (xy[:, None, :] - xy[None, :, :]) % shape
    disp_idx = dxy[..., 0] * LY + dxy[..., 1]
    neg_idx = ((-xy[:, 0]) % LX) * LY + (-xy[:, 1]) % LY
    rx, ry = np.divmod(np.arange(LX * LY), LY)
    dmin = np.minimum(np.stack([rx, ry], -1), shape - np.stack([rx, ry], -1))
    dist_r = np.sqrt((dmin**2).sum(-1)).astype(np.float32)
    return disp_idx, neg_idx, dist_r


@functools.lru_cache(maxsize=None)
def _tables(geom: PairGeometry):
    """NumPy tables (cached); converted with jnp.asarray at the point of use so that no
    traced constant is stored across jit traces."""
    disp_idx, neg_idx, dist_r = displacement_tables(geom)
    return disp_idx.astype(np.int32), neg_idx.astype(np.int32), dist_r


def build_wtilde(W: jax.Array, theta: jax.Array, geom: PairGeometry) -> jax.Array:
    """(3, N, d) raw parameters -> (4, N, d) spin-block table with the distance envelope.

    Blocks are ordered ``[uu, ud, du, dd]`` = ``blk = 2 * spin_p + spin_q``; ``uu``/``dd``
    are antisymmetrised in ``r`` and ``du(r) = -ud(-r)``, so the whole table satisfies
    ``Wt[2 sq + sp, -r] = -Wt[2 sp + sq, r]`` exactly.  ``pref_r = exp(-lam D_r) /
    sum_{ij} exp(-lam D_ij)`` equals the einsum path's ``pref / pref.sum()``.
    """
    _, neg_idx, dist_r = _tables(geom)
    dist_r = jnp.asarray(dist_r)
    W_ud = W[0]
    W_uu = (W[1] - W[1][neg_idx]) / 2
    W_dd = (W[2] - W[2][neg_idx]) / 2
    W_du = -W_ud[neg_idx]
    lam = jax.nn.softplus(theta)
    pref = jnp.exp(-lam * dist_r)
    pref = pref / (geom.N * jnp.sum(pref))
    return jnp.stack([W_uu, W_ud, W_du, W_dd]) * pref[None, :, None]


# ------------------------------------------------------------------ kernels


def _decode(i, N, LY):
    """Mode index in [0, 2N) -> (spin, x, y) as int32 arrays (integer division only;
    ``%`` lowers poorly inside the Triton kernels)."""
    s = i // N
    site = i - s * N
    x = site // LY
    y = site - x * LY
    return s, x, y


def _wrap(v, L):
    return lax.select(v < 0, v + L, v)


def _pair_offsets(ip, iq, *, N, LX, LY, d):
    """Flat offsets into Wt.reshape(-1) of the rows Wt[blk(p,q), r(p,q)] for a (TP,) x
    (TQ,) tile of modes; r = site_p - site_q periodically."""
    sp, xp, yp = _decode(ip, N, LY)
    sq, xq, yq = _decode(iq, N, LY)
    dx = _wrap(xp[:, None] - xq[None, :], LX)
    dy = _wrap(yp[:, None] - yq[None, :], LY)
    r = dx * LY + dy
    blk = sp[:, None] * 2 + sq[None, :]
    return (blk * N + r) * d


def _fwd_kernel(ip_ref, iq_ref, yp_ref, yq_ref, wt_ref, out_ref, *, N, LX, LY, d, TP, TQ, TA):
    base = _pair_offsets(ip_ref[...], iq_ref[...], N=N, LX=LX, LY=LY, d=d)
    acc = jnp.zeros((TP, TQ), jnp.float32)
    for a0 in range(0, d, TA):
        ar = a0 + lax.broadcasted_iota(jnp.int32, (TA,), 0)
        w = wt_ref[base[:, :, None] + ar[None, None, :]]  # (TP, TQ, TA) gather
        ypa = yp_ref[:, pl.ds(a0, TA)]
        yqa = yq_ref[:, pl.ds(a0, TA)]
        # (Y_p * Y_q) first: commutative, so the (q, p) tile is the exact negative.
        acc = acc + jnp.sum((ypa[:, None, :] * yqa[None, :, :]) * w, axis=-1)
    out_ref[...] = acc


def _fwd_call(Yp: jax.Array, idxp: jax.Array, wt_flat: jax.Array, geom: PairGeometry):
    P, d = Yp.shape
    TP, TQ, TA, warps = _tiles(FWD_TILES, d, "TP", "TQ", "TA", "warps")
    assert P % TP == 0 and P % TQ == 0
    kern = functools.partial(
        _fwd_kernel, N=geom.N, LX=geom.LX, LY=geom.LY, d=d, TP=TP, TQ=TQ, TA=TA
    )
    return pl.pallas_call(
        kern,
        grid=(P // TP, P // TQ),
        in_specs=[
            pl.BlockSpec((TP,), lambda i, j: (i,)),
            pl.BlockSpec((TQ,), lambda i, j: (j,)),
            pl.BlockSpec((TP, d), lambda i, j: (i, 0)),
            pl.BlockSpec((TQ, d), lambda i, j: (j, 0)),
            pl.BlockSpec(wt_flat.shape, lambda i, j: (0,)),
        ],
        out_specs=pl.BlockSpec((TP, TQ), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((P, P), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="pfsbp_pairing_fwd",
    )(idxp, idxp, Yp, Yp, wt_flat)


def _dy_kernel(
    ip_ref, idx_ref, g_ref, y_ref, wt_ref, out_ref, *, N, LX, LY, d, nQ, TP, TQ, TA
):
    """dY[p, a] = sum_q Ghat[p, q] * Wt[blk, r, a] * Y[q, a] for one p-tile."""
    ip = ip_ref[...]
    nA = d // TA

    def body(qi, accs):
        q0 = pl.multiple_of(qi * TQ, TQ)
        iq = idx_ref[pl.ds(q0, TQ)]
        base = _pair_offsets(ip, iq, N=N, LX=LX, LY=LY, d=d)
        g = g_ref[:, pl.ds(q0, TQ)]  # (TP, TQ)
        new = []
        for k in range(nA):
            a0 = k * TA
            ar = a0 + lax.broadcasted_iota(jnp.int32, (TA,), 0)
            w = wt_ref[base[:, :, None] + ar[None, None, :]]
            yqa = y_ref[pl.ds(q0, TQ), pl.ds(a0, TA)]  # (TQ, TA)
            new.append(accs[k] + jnp.sum(g[:, :, None] * w * yqa[None, :, :], axis=1))
        return tuple(new)

    init = tuple(jnp.zeros((TP, TA), jnp.float32) for _ in range(nA))
    accs = lax.fori_loop(0, nQ, body, init)
    for k in range(nA):
        out_ref[:, pl.ds(k * TA, TA)] = accs[k]


def _dy_call(Ghat: jax.Array, Yp: jax.Array, idxp: jax.Array, wt_flat, geom):
    P, d = Yp.shape
    TP, TQ, TA, warps = _tiles(DY_TILES, d, "TP", "TQ", "TA", "warps")
    assert P % TP == 0 and P % TQ == 0
    kern = functools.partial(
        _dy_kernel, N=geom.N, LX=geom.LX, LY=geom.LY, d=d, nQ=P // TQ, TP=TP, TQ=TQ, TA=TA
    )
    return pl.pallas_call(
        kern,
        grid=(P // TP,),
        in_specs=[
            pl.BlockSpec((TP,), lambda i: (i,)),
            pl.BlockSpec((P,), lambda i: (0,)),
            pl.BlockSpec((TP, P), lambda i: (i, 0)),
            pl.BlockSpec((P, d), lambda i: (0, 0)),
            pl.BlockSpec(wt_flat.shape, lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((TP, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((P, d), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="pfsbp_pairing_dy",
    )(idxp, idxp, Ghat, Yp, wt_flat)


def _dw_kernel(
    inv_ref, idx_ref, g_ref, y2_ref, yflat_ref, out_ref, *, N, LX, LY, d, P, nP, TP, TR, TA
):
    """dWt[blk, r, a] = sum_{p, q : blk, r} G[p, q] Y[p, a] Y[q, a] for one r-tile.

    Loops over occupied p; the partner q of (p, r) is the occupied mode at site
    ``site_p - r`` with spin ``sq`` (``inv`` map, -1 if empty), so each (p, r, sq) has at
    most one term and the reduction over p is a plain in-register sum.
    """
    ri = pl.program_id(0)
    r = ri * TR + lax.broadcasted_iota(jnp.int32, (TR,), 0)
    dxr = r // LY
    dyr = r - dxr * LY
    nA = d // TA
    one = jnp.ones((TP,), jnp.float32)
    zero = jnp.zeros((TP,), jnp.float32)
    zero_tile = jnp.zeros((TP, TR), jnp.float32)

    def body(pi, accs):
        p0 = pl.multiple_of(pi * TP, TP)
        parr = p0 + lax.broadcasted_iota(jnp.int32, (TP,), 0)
        ip = idx_ref[pl.ds(p0, TP)]
        sp, xp, yp = _decode(ip, N, LY)
        xq = _wrap(xp[:, None] - dxr[None, :], LX)
        yq = _wrap(yp[:, None] - dyr[None, :], LY)
        site_q = xq * LY + yq  # (TP, TR)
        m_up = lax.select(sp == 0, one, zero)  # p in the up sector
        m_dn = one - m_up
        accs = list(accs)
        for sq in range(2):
            q = inv_ref[site_q + sq * N]  # (TP, TR), -1 where the partner is empty
            valid = q >= 0
            qc = lax.select(valid, q, jnp.zeros_like(q))
            g = g_ref[parr[:, None] * P + qc]  # (TP, TR) gather of G[p, q]
            g = lax.select(valid, g, zero_tile)
            for k in range(nA):
                a0 = k * TA
                ar = a0 + lax.broadcasted_iota(jnp.int32, (TA,), 0)
                yqa = yflat_ref[qc[:, :, None] * d + ar[None, None, :]]  # (TP, TR, TA)
                ypa = y2_ref[pl.ds(p0, TP), pl.ds(a0, TA)]  # (TP, TA)
                c = (g[:, :, None] * ypa[:, None, :]) * yqa
                accs[(0 * 2 + sq) * nA + k] += jnp.sum(m_up[:, None, None] * c, axis=0)
                accs[(1 * 2 + sq) * nA + k] += jnp.sum(m_dn[:, None, None] * c, axis=0)
        return tuple(accs)

    init = tuple(jnp.zeros((TR, TA), jnp.float32) for _ in range(4 * nA))
    accs = lax.fori_loop(0, nP, body, init)
    for blk in range(4):
        for k in range(nA):
            out_ref[blk, :, pl.ds(k * TA, TA)] = accs[blk * nA + k]


def _dw_call(G: jax.Array, Yp: jax.Array, idxp: jax.Array, inv: jax.Array, geom):
    P, d = Yp.shape
    N = geom.N
    TP, TR, TA, warps = _tiles(DW_TILES, d, "TP", "TR", "TA", "warps")
    while N % TR:  # e.g. 4x4 lattices
        TR //= 2
    assert TR >= 4 and P % TP == 0, (N, P, d)
    kern = functools.partial(
        _dw_kernel, N=N, LX=geom.LX, LY=geom.LY, d=d, P=P, nP=P // TP, TP=TP, TR=TR, TA=TA
    )
    return pl.pallas_call(
        kern,
        grid=(N // TR,),
        in_specs=[
            pl.BlockSpec((2 * N,), lambda i: (0,)),
            pl.BlockSpec((P,), lambda i: (0,)),
            pl.BlockSpec((P * P,), lambda i: (0,)),
            pl.BlockSpec((P, d), lambda i: (0, 0)),
            pl.BlockSpec((P * d,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((4, TR, d), lambda i: (0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((4, N, d), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="pfsbp_pairing_dw",
    )(inv, idxp, G.reshape(-1), Yp, Yp.reshape(-1))


# ------------------------------------------------------------------ custom_vjp wrapper


def _pad(Y, idx, geom):
    n, P = geom.n_occ, geom.n_pad
    if P == n:
        return Y, idx.astype(jnp.int32)
    Yp = jnp.pad(Y, ((0, P - n), (0, 0)))
    idxp = jnp.pad(idx.astype(jnp.int32), (0, P - n))  # mode 0: any in-range value
    return Yp, idxp


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def pairing_occ(Y: jax.Array, idx: jax.Array, Wt: jax.Array, geom: PairGeometry):
    """F[p, q] = sum_a Y[p, a] Wt[2 s_p + s_q, site_p - site_q, a] Y[q, a], (2Ne, 2Ne).

    ``Y``: (2Ne, d) fp32 features of the occupied modes; ``idx``: (2Ne,) mode indices
    in [0, 2N) (``fermion_idx(s)``); ``Wt``: (4, N, d) from ``build_wtilde``.
    GPU only; see ``pairing_occ_reference`` for the pure-JAX equivalent.
    """
    F, _ = _pairing_occ_fwd(Y, idx, Wt, geom)
    return F


def _pairing_occ_fwd(Y, idx, Wt, geom):
    n = geom.n_occ
    Yp, idxp = _pad(Y.astype(jnp.float32), idx, geom)
    F = _fwd_call(Yp, idxp, Wt.astype(jnp.float32).reshape(-1), geom)
    return F[:n, :n], (Yp, idxp, Wt)


def _pairing_occ_bwd(geom, res, G):
    Yp, idxp, Wt = res
    n, P = geom.n_occ, geom.n_pad
    Gp = jnp.pad(G.astype(jnp.float32), ((0, P - n), (0, P - n)))
    Ghat = Gp - Gp.T
    dY = _dy_call(Ghat, Yp, idxp, Wt.astype(jnp.float32).reshape(-1), geom)[:n]
    inv = jnp.full((2 * geom.N,), -1, jnp.int32).at[idxp[:n]].set(jnp.arange(n, dtype=jnp.int32))
    dWt = _dw_call(Gp, Yp, idxp, inv, geom)
    return dY.astype(Yp.dtype), None, dWt.astype(Wt.dtype)


pairing_occ.defvjp(_pairing_occ_fwd, _pairing_occ_bwd)


# ------------------------------------------------------------------ reference + helpers


def pairing_occ_reference(Y, idx, Wt, geom: PairGeometry):
    """Pure-JAX version of ``pairing_occ`` (fp32 HIGHEST einsum)."""
    disp_idx = jnp.asarray(_tables(geom)[0])
    N = geom.N
    s = idx // N
    site = idx - s * N
    r = disp_idx[site[:, None], site[None, :]]
    blk = s[:, None] * 2 + s[None, :]
    Wg = Wt[blk, r]  # (n, n, d)
    return jnp.einsum(
        "pa,pqa,qa->pq", Y, Wg, Y, precision=lax.Precision.HIGHEST
    )


def pairing_matrix_occ(y, idx, W, theta, geom: PairGeometry, *, reference=None):
    """Features (2, N, d) + occupied modes -> antisymmetric (2Ne, 2Ne) pairing block.

    Equals ``pairing_matrix(s)[idx][:, idx]`` of the einsum path up to fp32 rounding. The
    kernel output is exactly antisymmetric; the reference output is antisymmetrised.
    ``reference=None`` picks the Pallas kernel on GPU and the pure-JAX reference elsewhere.
    """
    if reference is None:
        reference = not kernels_available()
    N, d = geom.N, geom.d
    Y = y.reshape(2 * N, d)[idx]
    Wt = build_wtilde(W, theta, geom)
    fn = pairing_occ_reference if reference else pairing_occ
    F = fn(Y, idx, Wt, geom)
    if reference:
        F = (F - F.T) / 2
    return F
