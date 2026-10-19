"""Pallas (Triton) kernels for the occupied up x down block of the BCS-backflow pairing matrix.

The kernels compute the (n_up, n_dn) block

    F[p, q] = sum_a Yu[p, a] * Wt[site_p - site_q, a] * Yd[q, a]

directly, with ``Yu = y_up[iu]``, ``Yd = y_dn[idn]`` and ``Wt`` = W times the distance
envelope. They are the single-spin-block version of ``pf_kernel`` with the same forward and
backward structure. The ``delta_d`` term is added outside the kernel.

Public API: ``UDGeometry``, ``build_wt_ud`` (parameters -> Wt), ``pairing_ud`` (GPU
kernel), ``pairing_ud_reference`` (pure JAX) and ``pairing_block_ud`` (features -> F, using
the reference off GPU).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .pf_kernel import PAD, _tiles, _wrap, displacement_tables, kernels_available, pl, plgpu

# Same register tiles as the Pfaffian kernels
FWD_TILES = dict(TP=16, TQ=32, TA=16, warps=4)
DY_TILES = dict(TP=32, TQ=32, TA=8, warps=8)
DW_TILES = dict(TP=32, TR=16, TA=16, warps=4)


class UDGeometry(NamedTuple):
    """Static description of the up x down pairing block (hashable, nondiff arg)."""

    LX: int
    LY: int
    d: int
    n_up: int
    n_dn: int

    @property
    def N(self) -> int:
        return self.LX * self.LY

    @property
    def p_up(self) -> int:
        return -(-self.n_up // PAD) * PAD

    @property
    def p_dn(self) -> int:
        return -(-self.n_dn // PAD) * PAD


@functools.lru_cache(maxsize=None)
def _tables(geom: UDGeometry):
    """NumPy tables (cached); converted with jnp.asarray at the point of use."""
    disp_idx, neg_idx, dist_r = displacement_tables(geom)
    return disp_idx.astype(np.int32), neg_idx.astype(np.int32), dist_r


def build_wt_ud(W: jax.Array, theta: jax.Array, geom: UDGeometry) -> jax.Array:
    """(N, d) raw W -> (N, d) Wt = W * pref, ``pref_r = exp(-lam D_r) / sum_{ij}
    exp(-lam D_ij)``, the einsum path's ``f * pref / jnp.sum(pref)``."""
    _, _, dist_r = _tables(geom)
    lam = jax.nn.softplus(theta)
    pref = jnp.exp(-lam * jnp.asarray(dist_r))
    pref = pref / (geom.N * jnp.sum(pref))
    return W * pref[:, None]


# ------------------------------------------------------------------ kernels


def _offsets(ip, iq, *, LX, LY, d):
    """Flat offsets into Wt.reshape(-1) of the rows Wt[site_p - site_q] for a (TP,) x
    (TQ,) tile of sites (integer division only, as in pf_kernel)."""
    xp = ip // LY
    yp = ip - xp * LY
    xq = iq // LY
    yq = iq - xq * LY
    dx = _wrap(xp[:, None] - xq[None, :], LX)
    dy = _wrap(yp[:, None] - yq[None, :], LY)
    return (dx * LY + dy) * d


def _fwd_kernel(ip_ref, iq_ref, yp_ref, yq_ref, wt_ref, out_ref, *, LX, LY, d, TP, TQ, TA):
    base = _offsets(ip_ref[...], iq_ref[...], LX=LX, LY=LY, d=d)
    acc = jnp.zeros((TP, TQ), jnp.float32)
    for a0 in range(0, d, TA):
        ar = a0 + lax.broadcasted_iota(jnp.int32, (TA,), 0)
        w = wt_ref[base[:, :, None] + ar[None, None, :]]  # (TP, TQ, TA) gather
        ypa = yp_ref[:, pl.ds(a0, TA)]
        yqa = yq_ref[:, pl.ds(a0, TA)]
        acc = acc + jnp.sum((ypa[:, None, :] * yqa[None, :, :]) * w, axis=-1)
    out_ref[...] = acc


def _fwd_call(Yu, iu, Yd, idn, wt_flat, geom: UDGeometry):
    Pu, d = Yu.shape
    Pd = Yd.shape[0]
    TP, TQ, TA, warps = _tiles(FWD_TILES, d, "TP", "TQ", "TA", "warps")
    assert Pu % TP == 0 and Pd % TQ == 0
    kern = functools.partial(_fwd_kernel, LX=geom.LX, LY=geom.LY, d=d, TP=TP, TQ=TQ, TA=TA)
    return pl.pallas_call(
        kern,
        grid=(Pu // TP, Pd // TQ),
        in_specs=[
            pl.BlockSpec((TP,), lambda i, j: (i,)),
            pl.BlockSpec((TQ,), lambda i, j: (j,)),
            pl.BlockSpec((TP, d), lambda i, j: (i, 0)),
            pl.BlockSpec((TQ, d), lambda i, j: (j, 0)),
            pl.BlockSpec(wt_flat.shape, lambda i, j: (0,)),
        ],
        out_specs=pl.BlockSpec((TP, TQ), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((Pu, Pd), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="sbp_ud_pairing_fwd",
    )(iu, idn, Yu, Yd, wt_flat)


def _dy_kernel(ip_ref, iq_ref, g_ref, y_ref, wt_ref, out_ref, *, LX, LY, d, nQ, TP, TQ, TA):
    """dYp[p, a] = sum_q G[p, q] * Wt[site_p - site_q, a] * Yq[q, a] for one p-tile."""
    ip = ip_ref[...]
    nA = d // TA

    def body(qi, accs):
        q0 = pl.multiple_of(qi * TQ, TQ)
        iq = iq_ref[pl.ds(q0, TQ)]
        base = _offsets(ip, iq, LX=LX, LY=LY, d=d)
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


def _dy_call(G, ip, iq, Yq, wt_flat, geom: UDGeometry):
    """Rows ``ip`` (P,), columns ``iq`` (Q,), G (P, Q), Yq (Q, d) -> (P, d)."""
    P, Q = G.shape
    d = Yq.shape[1]
    TP, TQ, TA, warps = _tiles(DY_TILES, d, "TP", "TQ", "TA", "warps")
    assert P % TP == 0 and Q % TQ == 0
    kern = functools.partial(
        _dy_kernel, LX=geom.LX, LY=geom.LY, d=d, nQ=Q // TQ, TP=TP, TQ=TQ, TA=TA
    )
    return pl.pallas_call(
        kern,
        grid=(P // TP,),
        in_specs=[
            pl.BlockSpec((TP,), lambda i: (i,)),
            pl.BlockSpec((Q,), lambda i: (0,)),
            pl.BlockSpec((TP, Q), lambda i: (i, 0)),
            pl.BlockSpec((Q, d), lambda i: (0, 0)),
            pl.BlockSpec(wt_flat.shape, lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((TP, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((P, d), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="sbp_ud_pairing_dy",
    )(ip, iq, G, Yq, wt_flat)


def _dw_kernel(
    inv_ref, iu_ref, g_ref, yu_ref, ydflat_ref, out_ref, *, LX, LY, d, Pd, nP, TP, TR, TA
):
    """dWt[r, a] = sum_{p : site_p - site_q = r} G[p, q] Yu[p, a] Yd[q, a] for one r-tile.

    Loops over occupied up sites p; the partner q of (p, r) is the occupied down site
    ``site_p - r`` (``inv`` map, -1 if empty), so each (p, r) has at most one term.
    """
    ri = pl.program_id(0)
    r = ri * TR + lax.broadcasted_iota(jnp.int32, (TR,), 0)
    dxr = r // LY
    dyr = r - dxr * LY
    nA = d // TA
    zero_tile = jnp.zeros((TP, TR), jnp.float32)

    def body(pi, accs):
        p0 = pl.multiple_of(pi * TP, TP)
        parr = p0 + lax.broadcasted_iota(jnp.int32, (TP,), 0)
        ip = iu_ref[pl.ds(p0, TP)]
        xp = ip // LY
        yp = ip - xp * LY
        xq = _wrap(xp[:, None] - dxr[None, :], LX)
        yq = _wrap(yp[:, None] - dyr[None, :], LY)
        q = inv_ref[xq * LY + yq]  # (TP, TR), -1 where the down site is empty
        valid = q >= 0
        qc = lax.select(valid, q, jnp.zeros_like(q))
        g = g_ref[parr[:, None] * Pd + qc]  # (TP, TR) gather of G[p, q]
        g = lax.select(valid, g, zero_tile)
        new = []
        for k in range(nA):
            a0 = k * TA
            ar = a0 + lax.broadcasted_iota(jnp.int32, (TA,), 0)
            yqa = ydflat_ref[qc[:, :, None] * d + ar[None, None, :]]  # (TP, TR, TA)
            ypa = yu_ref[pl.ds(p0, TP), pl.ds(a0, TA)]  # (TP, TA)
            new.append(accs[k] + jnp.sum((g[:, :, None] * ypa[:, None, :]) * yqa, axis=0))
        return tuple(new)

    init = tuple(jnp.zeros((TR, TA), jnp.float32) for _ in range(nA))
    accs = lax.fori_loop(0, nP, body, init)
    for k in range(nA):
        out_ref[:, pl.ds(k * TA, TA)] = accs[k]


def _dw_call(G, Yu, iu, inv_dn, Yd, geom: UDGeometry):
    Pu, d = Yu.shape
    Pd = Yd.shape[0]
    N = geom.N
    TP, TR, TA, warps = _tiles(DW_TILES, d, "TP", "TR", "TA", "warps")
    while N % TR:  # e.g. 4x4 lattices
        TR //= 2
    assert TR >= 4 and Pu % TP == 0, (N, Pu, d)
    kern = functools.partial(
        _dw_kernel, LX=geom.LX, LY=geom.LY, d=d, Pd=Pd, nP=Pu // TP, TP=TP, TR=TR, TA=TA
    )
    return pl.pallas_call(
        kern,
        grid=(N // TR,),
        in_specs=[
            pl.BlockSpec((N,), lambda i: (0,)),
            pl.BlockSpec((Pu,), lambda i: (0,)),
            pl.BlockSpec((Pu * Pd,), lambda i: (0,)),
            pl.BlockSpec((Pu, d), lambda i: (0, 0)),
            pl.BlockSpec((Pd * d,), lambda i: (0,)),
        ],
        out_specs=pl.BlockSpec((TR, d), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((N, d), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=1),
        name="sbp_ud_pairing_dw",
    )(inv_dn, iu, G.reshape(-1), Yu, Yd.reshape(-1))


# ------------------------------------------------------------------ custom_vjp wrapper


def _pad(Y, idx, P):
    n = Y.shape[0]
    idx = idx.astype(jnp.int32)
    if P == n:
        return Y, idx
    # padded rows: zero features at site 0 (any in-range value)
    return jnp.pad(Y, ((0, P - n), (0, 0))), jnp.pad(idx, (0, P - n))


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def pairing_ud(Yu, Yd, iu, idn, Wt, geom: UDGeometry):
    """F[p, q] = sum_a Yu[p, a] Wt[iu[p] - idn[q], a] Yd[q, a], (n_up, n_dn).

    ``Yu`` (n_up, d) / ``Yd`` (n_dn, d): fp32 features of the occupied up / down sites;
    ``iu`` / ``idn``: their site indices in [0, N) (``fermion_idx(s, separate_spins=True)``);
    ``Wt``: (N, d) from ``build_wt_ud``.  GPU only; see ``pairing_ud_reference``.
    """
    F, _ = _pairing_ud_fwd(Yu, Yd, iu, idn, Wt, geom)
    return F


def _pairing_ud_fwd(Yu, Yd, iu, idn, Wt, geom):
    Yup, iup = _pad(Yu.astype(jnp.float32), iu, geom.p_up)
    Ydp, idnp = _pad(Yd.astype(jnp.float32), idn, geom.p_dn)
    F = _fwd_call(Yup, iup, Ydp, idnp, Wt.astype(jnp.float32).reshape(-1), geom)
    return F[: geom.n_up, : geom.n_dn], (Yup, Ydp, iup, idnp, Wt)


def _pairing_ud_bwd(geom, res, G):
    Yup, Ydp, iup, idnp, Wt = res
    nu, nd, Pu, Pd = geom.n_up, geom.n_dn, geom.p_up, geom.p_dn
    Gp = jnp.pad(G.astype(jnp.float32), ((0, Pu - nu), (0, Pd - nd)))
    wt32 = Wt.astype(jnp.float32)
    neg_idx = jnp.asarray(_tables(geom)[1])
    dYu = _dy_call(Gp, iup, idnp, Ydp, wt32.reshape(-1), geom)[:nu]
    # dYd[q] = sum_p G[p, q] Wt[site_p - site_q] Yu[p] = sum_p G^T[q, p] Wt[-(site_q - site_p)] Yu[p]
    dYd = _dy_call(Gp.T, idnp, iup, Yup, wt32[neg_idx].reshape(-1), geom)[:nd]
    inv_dn = jnp.full((geom.N,), -1, jnp.int32).at[idnp[:nd]].set(jnp.arange(nd, dtype=jnp.int32))
    dWt = _dw_call(Gp, Yup, iup, inv_dn, Ydp, geom)
    return dYu.astype(Yup.dtype), dYd.astype(Ydp.dtype), None, None, dWt.astype(Wt.dtype)


pairing_ud.defvjp(_pairing_ud_fwd, _pairing_ud_bwd)


# ------------------------------------------------------------------ reference + helpers


def pairing_ud_reference(Yu, Yd, iu, idn, Wt, geom: UDGeometry):
    """Pure-JAX version of ``pairing_ud`` (fp32 HIGHEST einsum)."""
    disp_idx = jnp.asarray(_tables(geom)[0])
    r = disp_idx[iu[:, None], idn[None, :]]
    return jnp.einsum("pa,pqa,qa->pq", Yu, Wt[r], Yd, precision=lax.Precision.HIGHEST)


def pairing_block_ud(y, iu, idn, W, theta, geom: UDGeometry, *, reference=None):
    """Features (2, N, d) + occupied up / down sites -> (n_up, n_dn) pairing block.

    Equals ``(f * pref / pref.sum())[iu][:, idn]`` of the einsum path up to fp32
    rounding, without the ``delta_d * FD`` term.  ``reference=None`` picks the Pallas
    kernel on GPU and the pure-JAX reference elsewhere.
    """
    if reference is None:
        reference = not kernels_available()
    Yu = y[0][iu]
    Yd = y[1][idn]
    Wt = build_wt_ud(W, theta, geom)
    fn = pairing_ud_reference if reference else pairing_ud
    return fn(Yu, Yd, iu, idn, Wt, geom)
