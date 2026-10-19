"""Pallas(Triton) kernels for the fPEPS D=4 HOTRG layer hot loops.

The per-(sample, pair) algebra of one compression layer works on tiny fixed
Z2 structures (8-sector x (2,2,2,2) blocks); XLA lowers the equivalent
einsum chains 3-6x off bandwidth (tiny-axis GEMMs, materialized layout
transposes). These kernels keep each unit's algebra in registers, one memory
pass, with the slot bookkeeping folded into static tables:

- `gram_chol(t0, t1, pi)`: BOTH cholesky reduced factors (Rl, Rr) in one
  fused pass (merged r/l grams + in-register shifted chol; one t0/t1 read,
  no XX round-trip). The gram replaces the C-intermediate entirely via the
  factorization sum_rest C.C = [t0 x t0]_(l,u) * [t1 x t1]_(L,d) contracted
  over the bond charges.
- `proj_pre(Rl, Rr_next)` / `proj_post(M, U2, w2, Rl, Rr_next)`: the
  projector assembly around eigh8 (M = Rl.Rr, G = M.M^T; then the quimb
  svd:eig truncation/safe-inverse tail to Pl, Pr).
- `patch_slots(t0, t1, P1, P2, pi)`: the projected pair tensor
  P1 . (t0 . t1) . P2 in slot order (cl, cr, us) (canonical permutation +
  sign tables applied by the caller as in fpeps_fast_layer).
- legacy single-step ops kept: `gram_xx`, `chol_factor`.

The public ops are backward compatible with jax transformations:
custom_vjp (backward = jax.vjp of the verified jnp reference path in
fpeps_fast_layer, so training jacobians work) around custom_vmap (nested
jax.vmap levels -- e.g. samples x translation reps in shared_transavg_psis
-- collapse into the kernels' flat batch instead of hitting the pallas
batching rule).

Tables (row gathers G, slot signs S) come from fpeps_fast_layer._lut and are
selected per-lane from the traced parity index pi via vector-indexed ref
loads. Element access uses the scalar-list technique (Pallas Triton cannot
slice register arrays); loops over (c, f) are fori with traced table reads
to keep the Triton program small.
"""

import os
from functools import lru_cache, partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plt

import fpeps_fast_layer as FL

_TritonParams = getattr(plt, "CompilerParams", None) or plt.TritonCompilerParams

# The layer kernels (gram_chol / proj_pre / proj_post / patch_slots) compute
# their 8x8-block products as scalar FFMA chains, i.e. in full fp32 whatever
# the global matmul precision. Their backwards replay the jnp reference path,
# whose einsums would otherwise drop to TF32 on Ampere/Hopper under the
# default precision, making the gradient that of a ~1e-3-perturbed forward
# (and, before the chol8 pivot floor, flipping near-singular Schur pivots
# negative -> NaN jacobian rows). Pin the replays to full fp32 so forward and
# backward see the same arithmetic; these are tiny latency-bound einsums, so
# the pin costs nothing measurable. blob_build / ring_chain are NOT pinned:
# their forward is a genuine tensor-core dot at the global precision and the
# replay must match it (it is bitwise equal).
def _fp32_replay():
    return jax.default_matmul_precision("highest")


def _warps(tile):
    return int(os.environ.get("FK_WARPS", max(1, tile // 32)))


def _row(ref, lanes, rowvec, scale=None):
    """Load one block row as a list of 16 (TILE,) vectors."""
    out = []
    for j in range(16):
        v = ref[lanes, rowvec, j]
        if scale is not None:
            v = v * scale
        out.append(v)
    return out


def _sel4(pi, tab):
    """Per-lane select of a static 4-entry table (python floats/ints)."""
    t0, t1, t2, t3 = (float(x) for x in tab)
    r = jnp.where(pi == 0, t0, jnp.where(pi == 1, t1,
                                         jnp.where(pi == 2, t2, t3)))
    return r


def _gram_accum(t0_ref, t1_ref, pi, lanes, GA_ref, GB_ref, SA_ref, keep):
    """Accumulate both sector-block 8x8 grams of one y-pair over the kept
    legs, fully in registers. Returns (xx0, xx1): flat row-major tuples of
    64 (TILE,) vectors. Loop over (c, f) is fori with traced table reads so
    the Triton program stays small; c gates the accumulator via where."""
    TILE = pi.shape[0]
    # dim index helpers (t0 dims (l, r, u, M); t1 dims (L, R, M, d))
    if keep == "r":
        iA = lambda l, p, u, M: l * 8 + p * 4 + u * 2 + M  # kept p = r
        iB = lambda L, P, M, d: L * 8 + P * 4 + M * 2 + d  # kept P = R
    else:
        iA = lambda p, r, u, M: p * 8 + r * 4 + u * 2 + M  # kept p = l
        iB = lambda P, R, M, d: P * 8 + R * 4 + M * 2 + d  # kept P = L

    zero = tuple(jnp.zeros((TILE,), t0_ref.dtype) for _ in range(64))

    def fbody(cf, carry):
        xx = [list(carry[0]), list(carry[1])]
        # jnp % / // lower badly in triton; operands non-negative
        c = jax.lax.div(cf, 8)
        f = jax.lax.rem(cf, 8)
        a = []
        b = []
        for m in range(2):
            k = c * 16 + f * 2 + m  # gram tables are (c, f, m)-ordered
            ga = GA_ref[pi * 32 + k]
            gb = GB_ref[pi * 32 + k]
            sa = SA_ref[pi * 32 + k]
            a.append(_row(t0_ref, lanes, ga, sa))
            b.append(_row(t1_ref, lanes, gb))
        for m in range(2):
            for n in range(2):
                # ga[p, q, M, N] = sum_{l, u} a_m[l, p, u, M] a_n[l, q, u, N]
                gapq = {}
                gbpq = {}
                for p in range(2):
                    for q in range(2):
                        for M in range(2):
                            for N in range(2):
                                sA = 0.0
                                sB = 0.0
                                for x1 in range(2):
                                    for x2 in range(2):
                                        if keep == "r":
                                            sA += (a[m][iA(x1, p, x2, M)]
                                                   * a[n][iA(x1, q, x2, N)])
                                            sB += (b[m][iB(x1, p, M, x2)]
                                                   * b[n][iB(x1, q, N, x2)])
                                        else:
                                            sA += (a[m][iA(p, x1, x2, M)]
                                                   * a[n][iA(q, x1, x2, N)])
                                            sB += (b[m][iB(p, x1, M, x2)]
                                                   * b[n][iB(q, x1, N, x2)])
                                gapq[(p, q, M, N)] = sA
                                gbpq[(p, q, M, N)] = sB
                # xx[c][(m, p, P), (n, q, Q)] += sum_{M,N} ga[p,q,M,N] gb[P,Q,M,N]
                for p in range(2):
                    for P in range(2):
                        for q in range(2):
                            for Q in range(2):
                                acc = 0.0
                                for M in range(2):
                                    for N in range(2):
                                        acc += (gapq[(p, q, M, N)]
                                                * gbpq[(P, Q, M, N)])
                                r_ = m * 4 + p * 2 + P
                                q_ = n * 4 + q * 2 + Q
                                # accumulate via where over c (c is traced)
                                cur0 = xx[0][r_ * 8 + q_]
                                cur1 = xx[1][r_ * 8 + q_]
                                is0 = (c == 0)
                                xx[0][r_ * 8 + q_] = jnp.where(is0, cur0 + acc, cur0)
                                xx[1][r_ * 8 + q_] = jnp.where(is0, cur1, cur1 + acc)
        return (tuple(xx[0]), tuple(xx[1]))

    return jax.lax.fori_loop(0, 16, fbody, (zero, zero))


def _gram_kernel(t0_ref, t1_ref, pi_ref, GA_ref, GB_ref, SA_ref, xx_ref, *,
                 keep):
    pi = pi_ref[...]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (pi.shape[0],), 0)
    xx = _gram_accum(t0_ref, t1_ref, pi, lanes, GA_ref, GB_ref, SA_ref, keep)
    for cq in range(2):
        for e in range(64):
            xx_ref[:, cq, e // 8, e % 8] = xx[cq][e]


@lru_cache
def _gram_call(keep, tile):
    Gname = "Ar_gram" if keep == "r" else "Al_gram"
    Hname = "Br_gram" if keep == "r" else "Bl_gram"
    GA, SA, _ = FL._lut(Gname)
    GB, _, _ = FL._lut(Hname)
    # cache NUMPY tables only: jnp constants created inside a trace leak
    GAn = GA.reshape(-1).astype(np.int32)
    GBn = GB.reshape(-1).astype(np.int32)
    SAn = SA.reshape(-1).astype(np.float32)

    def call(t0, t1, pi):
        GAf, GBf, SAf = jnp.asarray(GAn), jnp.asarray(GBn), jnp.asarray(SAn)
        nb = pi.shape
        n = int(np.prod(nb))
        t0f = t0.reshape(n, 8, 16)
        t1f = t1.reshape(n, 8, 16)
        pif = pi.reshape(n).astype(jnp.int32)
        pad = (-n) % tile
        if pad:
            t0f = jnp.concatenate([t0f, jnp.zeros((pad, 8, 16), t0.dtype)])
            t1f = jnp.concatenate([t1f, jnp.zeros((pad, 8, 16), t1.dtype)])
            pif = jnp.concatenate([pif, jnp.zeros((pad,), jnp.int32)])
        npd = t0f.shape[0]
        xx = pl.pallas_call(
            partial(_gram_kernel, keep=keep),
            grid=(npd // tile,),
            in_specs=[
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile,), lambda i: (i,)),
                pl.BlockSpec((128,), lambda i: (0,)),
                pl.BlockSpec((128,), lambda i: (0,)),
                pl.BlockSpec((128,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((tile, 2, 8, 8), lambda i: (i, 0, 0, 0)),
            out_shape=jax.ShapeDtypeStruct((npd, 2, 8, 8), t0.dtype),
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(t0f, t1f, pif, GAf, GBf, SAf)
        if pad:
            xx = xx[:n]
        return xx.reshape(nb + (2, 8, 8))

    return call


def gram_xx(t0, t1, pi, keep, tile=128):
    """XX gram of the y-pair over `keep` legs. t0/t1 (..., 8, 2,2,2,2)."""
    t0f = t0.reshape(t0.shape[:-4] + (16,))
    t1f = t1.reshape(t1.shape[:-4] + (16,))
    return _gram_call(keep, tile)(t0f, t1f, pi)


def _patch_kernel(t0_ref, t1_ref, p1_ref, p2_ref, pi_ref, GA_ref, GB_ref,
                  SA_ref, o_ref):
    TILE = pi_ref.shape[0]
    pi = pi_ref[...]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (TILE,), 0)
    iA = lambda l, r, u, M: l * 8 + r * 4 + u * 2 + M
    iB = lambda L, R, M, d: L * 8 + R * 4 + M * 2 + d

    def body(s3, _):
        c = jax.lax.div(s3, 4)
        C = jax.lax.rem(jax.lax.div(s3, 2), 2)
        s = jax.lax.rem(s3, 2)
        out = [jnp.zeros((TILE,), o_ref.dtype) for _ in range(16)]
        for m in range(2):
            # P1 row for (c, t, m, l, L): fused col = m*4 + l*2 + L
            p1 = {}
            for t in range(2):
                for l in range(2):
                    for L in range(2):
                        p1[(t, l, L)] = p1_ref[lanes, c, t, m * 4 + l * 2 + L]
            for K in range(2):
                k = c * 16 + m * 8 + C * 4 + K * 2 + s
                ga = GA_ref[pi * 32 + k]
                sa = SA_ref[pi * 32 + k]
                gb = GB_ref[pi * 32 + k]
                a = _row(t0_ref, lanes, ga, sa)
                b = _row(t1_ref, lanes, gb)
                p2 = {}
                for T in range(2):
                    for r in range(2):
                        for R in range(2):
                            p2[(T, r, R)] = p2_ref[lanes, C,
                                                   K * 4 + r * 2 + R, T]
                # tmp[t, L, r, u, M] = sum_l p1[t, l, L] a[l, r, u, M]
                tmp = {}
                for t in range(2):
                    for L in range(2):
                        for r in range(2):
                            for u in range(2):
                                for M in range(2):
                                    tmp[(t, L, r, u, M)] = (
                                        p1[(t, 0, L)] * a[iA(0, r, u, M)]
                                        + p1[(t, 1, L)] * a[iA(1, r, u, M)])
                # e[t, r, u, R, d] = sum_{L, M} tmp . b
                # out[t, T, u, d] += sum_{r, R} e . p2
                for t in range(2):
                    for u in range(2):
                        for d in range(2):
                            for T in range(2):
                                acc = 0.0
                                for r in range(2):
                                    for R in range(2):
                                        e = 0.0
                                        for L in range(2):
                                            for M in range(2):
                                                e += (tmp[(t, L, r, u, M)]
                                                      * b[iB(L, R, M, d)])
                                        acc += e * p2[(T, r, R)]
                                j = t * 8 + T * 4 + u * 2 + d
                                out[j] = out[j] + acc
        for j in range(16):
            o_ref[:, s3, j] = out[j]
        return 0

    jax.lax.fori_loop(0, 8, body, 0)


@lru_cache
def _patch_call(tile):
    # patch tables use the plain (c, m, f) slot order and t0/t1 blocks in
    # ORIGINAL dim order, so build gathers from the plan directly
    plans = FL._plan_C("l")
    G_A = np.zeros((4, 32), np.int32)
    G_B = np.zeros((4, 32), np.int32)
    S_A = np.ones((4, 32))
    for (p0, p1), (ia, ib, _) in plans.items():
        piv = p0 * 2 + p1
        G_A[piv], G_B[piv] = ia, ib
        sc = np.asarray(FL.S_CL)[piv].copy()
        sp = np.asarray(FL.S_PATCH)[piv].reshape(2, 2, 2, 2, 2)
        for k in range(32):
            c, m = k >> 4, (k >> 3) & 1
            cr, mr, us = (k >> 2) & 1, (k >> 1) & 1, k & 1
            sc[k] *= sp[c, cr, us, m, mr]
        S_A[piv] = sc
    # cache NUMPY tables only: jnp constants created inside a trace leak
    GAn = G_A.reshape(-1).astype(np.int32)
    GBn = G_B.reshape(-1).astype(np.int32)
    SAn = S_A.reshape(-1).astype(np.float32)

    def call(t0, t1, P1, P2, pi):
        GAf, GBf, SAf = jnp.asarray(GAn), jnp.asarray(GBn), jnp.asarray(SAn)
        nb = pi.shape
        n = int(np.prod(nb))
        t0f = t0.reshape(n, 8, 16)
        t1f = t1.reshape(n, 8, 16)
        P1f = P1.reshape(n, 2, 2, 8)
        P2f = P2.reshape(n, 2, 8, 2)
        pif = pi.reshape(n).astype(jnp.int32)
        pad = (-n) % tile
        if pad:
            t0f = jnp.concatenate([t0f, jnp.zeros((pad, 8, 16), t0.dtype)])
            t1f = jnp.concatenate([t1f, jnp.zeros((pad, 8, 16), t1.dtype)])
            P1f = jnp.concatenate([P1f, jnp.zeros((pad, 2, 2, 8), P1.dtype)])
            P2f = jnp.concatenate([P2f, jnp.zeros((pad, 2, 8, 2), P2.dtype)])
            pif = jnp.concatenate([pif, jnp.zeros((pad,), jnp.int32)])
        npd = t0f.shape[0]
        out = pl.pallas_call(
            _patch_kernel,
            grid=(npd // tile,),
            in_specs=[
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 2, 2, 8), lambda i: (i, 0, 0, 0)),
                pl.BlockSpec((tile, 2, 8, 2), lambda i: (i, 0, 0, 0)),
                pl.BlockSpec((tile,), lambda i: (i,)),
                pl.BlockSpec((128,), lambda i: (0,)),
                pl.BlockSpec((128,), lambda i: (0,)),
                pl.BlockSpec((128,), lambda i: (0,)),
            ],
            out_specs=pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
            out_shape=jax.ShapeDtypeStruct((npd, 8, 16), t0.dtype),
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(t0f, t1f, P1f, P2f, pif, GAf, GBf, SAf)
        if pad:
            out = out[:n]
        return out.reshape(nb + (8, 16))

    return call


def _chol8_regs(a, shift):
    """In-register shifted Cholesky of one 8x8 SPD block given as a flat
    row-major list/tuple of 64 (TILE,) vectors. Includes the symmray
    per-block relative trace shift. Returns L as nested lists (lower
    triangle only, None above the diagonal)."""
    a = list(a)
    tr = a[0]
    for i in range(1, 8):
        tr = tr + a[i * 8 + i]
    sh = tr * shift
    for i in range(8):
        a[i * 8 + i] = a[i * 8 + i] + sh
    L = [[None] * 8 for _ in range(8)]
    for j in range(8):
        s = a[j * 8 + j]
        for k in range(j):
            s = s - L[j][k] * L[j][k]
        d = jnp.sqrt(s)
        inv = 1.0 / d
        L[j][j] = d
        for i in range(j + 1, 8):
            s = a[i * 8 + j]
            for k in range(j):
                s = s - L[i][k] * L[j][k]
            L[i][j] = s * inv
    return L


def _chol_kernel(xx_ref, o_ref, *, upper, shift):
    """Per-unit shifted Cholesky of an 8x8 SPD block, fully in registers.
    xx_ref/o_ref: (TILE, 8, 8). upper=True writes L^H (the "right" reduced
    factor)."""
    a = [xx_ref[:, i, j] for i in range(8) for j in range(8)]
    L = _chol8_regs(a, shift)
    z = jnp.zeros_like(a[0])
    for i in range(8):
        for j in range(8):
            v = L[i][j] if j <= i else z
            if upper:
                o_ref[:, j, i] = v
            else:
                o_ref[:, i, j] = v


@lru_cache
def _chol_call(upper, tile, shift):
    def call(xx):
        nb = xx.shape[:-2]
        n = int(np.prod(nb))
        x = xx.reshape(n, 8, 8)
        pad = (-n) % tile
        if pad:
            eye = jnp.broadcast_to(jnp.eye(8, dtype=xx.dtype), (pad, 8, 8))
            x = jnp.concatenate([x, eye])
        npd = x.shape[0]
        out = pl.pallas_call(
            partial(_chol_kernel, upper=upper, shift=shift),
            grid=(npd // tile,),
            in_specs=[pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0))],
            out_specs=pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0)),
            out_shape=jax.ShapeDtypeStruct((npd, 8, 8), xx.dtype),
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(x)
        if pad:
            out = out[:n]
        return out.reshape(nb + (8, 8))

    return call


def chol_factor(xx, side, shift=1e-4, tile=128):
    """Shifted per-block cholesky reduced factor: side "right" -> upper."""
    return _chol_call(side == "right", tile, shift)(xx)


# --------------------------------------------------------------------------
# fused kernels: merged gram+cholesky (one t0/t1 read for BOTH reduced
# factors, no XX round-trip) and the projector assembly around eigh8.


def _gram_chol_kernel(t0_ref, t1_ref, pi_ref, GAr_ref, GBr_ref, SAr_ref,
                      GAl_ref, GBl_ref, SAl_ref, rl_ref, rr_ref, *, shift):
    """Both reduced cholesky factors of one y-pair in a single pass:
    r-gram -> shifted chol -> Rl (upper), then l-gram -> chol -> Rr (lower).
    Sequential so the register peak stays at one gram accumulation."""
    pi = pi_ref[...]
    TILE = pi.shape[0]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (TILE,), 0)
    z = jnp.zeros((TILE,), t0_ref.dtype)
    for keep, GA, GB, SA, o_ref, upper in (
        ("r", GAr_ref, GBr_ref, SAr_ref, rl_ref, True),
        ("l", GAl_ref, GBl_ref, SAl_ref, rr_ref, False),
    ):
        xx = _gram_accum(t0_ref, t1_ref, pi, lanes, GA, GB, SA, keep)
        for cq in range(2):
            L = _chol8_regs(xx[cq], shift)
            for i in range(8):
                for j in range(8):
                    v = L[i][j] if j <= i else z
                    if upper:
                        o_ref[:, cq, j, i] = v
                    else:
                        o_ref[:, cq, i, j] = v


@lru_cache
def _gram_chol_call(tile, shift):
    # cache NUMPY tables only: jnp constants created inside a trace leak
    ntabs = []
    for keep in ("r", "l"):
        GA, SA, _ = FL._lut("Ar_gram" if keep == "r" else "Al_gram")
        GB, _, _ = FL._lut("Br_gram" if keep == "r" else "Bl_gram")
        ntabs += [GA.reshape(-1).astype(np.int32),
                  GB.reshape(-1).astype(np.int32),
                  SA.reshape(-1).astype(np.float32)]

    def call(t0, t1, pi):
        tabs = [jnp.asarray(t) for t in ntabs]
        nb = pi.shape
        n = int(np.prod(nb))
        t0f = t0.reshape(n, 8, 16)
        t1f = t1.reshape(n, 8, 16)
        pif = pi.reshape(n).astype(jnp.int32)
        pad = (-n) % tile
        if pad:  # padded lanes chol NaN garbage; sliced off below
            t0f = jnp.concatenate([t0f, jnp.zeros((pad, 8, 16), t0.dtype)])
            t1f = jnp.concatenate([t1f, jnp.zeros((pad, 8, 16), t1.dtype)])
            pif = jnp.concatenate([pif, jnp.zeros((pad,), jnp.int32)])
        npd = t0f.shape[0]
        Rl, Rr = pl.pallas_call(
            partial(_gram_chol_kernel, shift=shift),
            grid=(npd // tile,),
            in_specs=[
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 8, 16), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile,), lambda i: (i,)),
            ] + [pl.BlockSpec((128,), lambda i: (0,))] * 6,
            out_specs=[
                pl.BlockSpec((tile, 2, 8, 8), lambda i: (i, 0, 0, 0)),
                pl.BlockSpec((tile, 2, 8, 8), lambda i: (i, 0, 0, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((npd, 2, 8, 8), t0.dtype),
                jax.ShapeDtypeStruct((npd, 2, 8, 8), t0.dtype),
            ],
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(t0f, t1f, pif, *tabs)
        if pad:
            Rl, Rr = Rl[:n], Rr[:n]
        return Rl.reshape(nb + (2, 8, 8)), Rr.reshape(nb + (2, 8, 8))

    return call


def _proj_pre_kernel(rl_ref, rr_ref, m_ref, g_ref):
    """Bond matrix M = Rl @ Rr and its gram G = M @ M^T (the eigh8 input)
    per (TILE, 8, 8) unit (sector axis folded into the batch)."""
    Rl = [[rl_ref[:, i, j] for j in range(8)] for i in range(8)]
    Rr = [[rr_ref[:, i, j] for j in range(8)] for i in range(8)]
    M = [[None] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            s = Rl[i][0] * Rr[0][j]
            for k in range(1, 8):
                s = s + Rl[i][k] * Rr[k][j]
            M[i][j] = s
            m_ref[:, i, j] = s
    for i in range(8):
        for j in range(8):
            s = M[i][0] * M[j][0]
            for k in range(1, 8):
                s = s + M[i][k] * M[j][k]
            g_ref[:, i, j] = s


def _proj_post_kernel(m_ref, u2_ref, w2_ref, rl_ref, rr_ref, pl_ref, pr_ref,
                      *, eps):
    """Projector assembly after eigh8, mirroring fpeps_fast_layer.projectors
    exactly (quimb svd:eig right=False conventions): clip, sqrt, per-block
    smax cutoff, safe_inverse, and the same division order.
    m/u2/w2: (TILE, 8, 8)/(TILE, 8, 2)/(TILE, 2) -> Pl (TILE, 8, 2),
    Pr (TILE, 2, 8)."""
    M = [[m_ref[:, i, j] for j in range(8)] for i in range(8)]
    U2 = [[u2_ref[:, i, t] for t in range(2)] for i in range(8)]
    s2 = [jnp.maximum(w2_ref[:, t], 0.0) for t in range(2)]
    s = [jnp.sqrt(x) for x in s2]
    cutoff = s[1] * (eps * 8.0)  # smax per block (eigenvalues ascending)
    c2 = cutoff * cutoff
    sinv = [s[t] / (s[t] * s[t] + c2) for t in range(2)]
    st = [jnp.sqrt(s[t]) for t in range(2)]
    # Vd[j][t] = (sinv[t] * sum_i U2[i][t] M[i][j]) / st[t]
    Vd = [[None] * 2 for _ in range(8)]
    for t in range(2):
        for j in range(8):
            acc = U2[0][t] * M[0][j]
            for i in range(1, 8):
                acc = acc + U2[i][t] * M[i][j]
            Vd[j][t] = (sinv[t] * acc) / st[t]
    for i in range(8):  # Pl = Rr @ Vd
        row = [rr_ref[:, i, j] for j in range(8)]
        for t in range(2):
            acc = row[0] * Vd[0][t]
            for j in range(1, 8):
                acc = acc + row[j] * Vd[j][t]
            pl_ref[:, i, t] = acc
    Ud = [[U2[i][t] / st[t] for i in range(8)] for t in range(2)]
    for j in range(8):  # Pr = Ud @ Rl
        col = [rl_ref[:, i, j] for i in range(8)]
        for t in range(2):
            acc = Ud[t][0] * col[0]
            for i in range(1, 8):
                acc = acc + Ud[t][i] * col[i]
            pr_ref[:, t, j] = acc


@lru_cache
def _proj_pre_call(tile):
    def call(Rl, Rr):
        nb = Rl.shape[:-2]
        n = int(np.prod(nb))
        a = Rl.reshape(n, 8, 8)
        b = Rr.reshape(n, 8, 8)
        pad = (-n) % tile
        if pad:
            z = jnp.zeros((pad, 8, 8), Rl.dtype)
            a = jnp.concatenate([a, z])
            b = jnp.concatenate([b, z])
        npd = a.shape[0]
        M, G = pl.pallas_call(
            _proj_pre_kernel,
            grid=(npd // tile,),
            in_specs=[pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0))] * 2,
            out_specs=[pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0))] * 2,
            out_shape=[jax.ShapeDtypeStruct((npd, 8, 8), Rl.dtype)] * 2,
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(a, b)
        if pad:
            M, G = M[:n], G[:n]
        return M.reshape(nb + (8, 8)), G.reshape(nb + (8, 8))

    return call


@lru_cache
def _proj_post_call(tile):
    def call(M, U2, w2, Rl, Rr):
        nb = w2.shape[:-1]
        n = int(np.prod(nb))
        eps = float(jnp.finfo(M.dtype).eps)
        Mf = M.reshape(n, 8, 8)
        U2f = U2.reshape(n, 8, 2)
        w2f = w2.reshape(n, 2)
        Rlf = Rl.reshape(n, 8, 8)
        Rrf = Rr.reshape(n, 8, 8)
        pad = (-n) % tile
        if pad:
            z88 = jnp.zeros((pad, 8, 8), M.dtype)
            Mf = jnp.concatenate([Mf, z88])
            U2f = jnp.concatenate([U2f, jnp.zeros((pad, 8, 2), M.dtype)])
            w2f = jnp.concatenate([w2f, jnp.ones((pad, 2), M.dtype)])
            Rlf = jnp.concatenate([Rlf, z88])
            Rrf = jnp.concatenate([Rrf, z88])
        npd = Mf.shape[0]
        Pl, Pr = pl.pallas_call(
            partial(_proj_post_kernel, eps=eps),
            grid=(npd // tile,),
            in_specs=[
                pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 8, 2), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 2), lambda i: (i, 0)),
                pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((tile, 8, 2), lambda i: (i, 0, 0)),
                pl.BlockSpec((tile, 2, 8), lambda i: (i, 0, 0)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((npd, 8, 2), M.dtype),
                jax.ShapeDtypeStruct((npd, 2, 8), M.dtype),
            ],
            compiler_params=_TritonParams(num_warps=_warps(tile)),
        )(Mf, U2f, w2f, Rlf, Rrf)
        if pad:
            Pl, Pr = Pl[:n], Pr[:n]
        return Pl.reshape(nb + (8, 2)), Pr.reshape(nb + (2, 8))

    return call


# --------------------------------------------------------------------------
# public ops: custom_vjp (jnp-reference backward, so training jacobians work)
# around custom_vmap (nested jax.vmap levels collapse into the kernels' flat
# batch instead of hitting the pallas batching rule).

_TILE = 128
_SHIFT = 1e-4


def _collapse_vmap(fn, n_out):
    vf = jax.custom_batching.custom_vmap(fn)

    @vf.def_vmap
    def _rule(axis_size, in_batched, *args):
        args = tuple(
            a if b else jnp.broadcast_to(a[None], (axis_size,) + jnp.shape(a))
            for a, b in zip(args, in_batched))
        out = vf(*args)
        return out, ((True,) * n_out if n_out > 1 else True)

    return vf


def _float0(x):
    return np.zeros(np.shape(x), dtype=jax.dtypes.float0)


@lru_cache
def _gram_chol_op():
    return _collapse_vmap(_gram_chol_call(_TILE, _SHIFT), 2)


def _gram_chol_impl(t0, t1, pi):
    t0f = t0.reshape(t0.shape[:-4] + (16,))
    t1f = t1.reshape(t1.shape[:-4] + (16,))
    return _gram_chol_op()(t0f, t1f, pi)


def _gram_chol_ref(t0, t1, pi):
    """jnp reference (verified v3 path); its vjp defines the backward."""
    Rl = FL._gram_R_lut(FL._lut_gather(t0, pi, "Ar_gram"),
                        FL._lut_gather(t1, pi, "Br_gram"), "right", _SHIFT)
    Rr = FL._gram_R_lut(FL._lut_gather(t0, pi, "Al_gram"),
                        FL._lut_gather(t1, pi, "Bl_gram"), "left", _SHIFT)
    return Rl, Rr


@jax.custom_vjp
def gram_chol(t0, t1, pi):
    """Both cholesky reduced factors of a y-pair in ONE fused kernel pass:
    t0/t1 (..., 8, 2,2,2,2), pi (...,) -> (Rl upper "right", Rr lower
    "left"), each (..., 2, 8, 8). Differentiable and vmap-collapsing."""
    return _gram_chol_impl(t0, t1, pi)


def _gram_chol_fwd(t0, t1, pi):
    return _gram_chol_impl(t0, t1, pi), (t0, t1, pi)


def _gram_chol_bwd(res, cts):
    t0, t1, pi = res
    with _fp32_replay():
        _, vjp = jax.vjp(lambda a, b: _gram_chol_ref(a, b, pi), t0, t1)
        dt0, dt1 = vjp(cts)
    return dt0, dt1, _float0(pi)


gram_chol.defvjp(_gram_chol_fwd, _gram_chol_bwd)


@lru_cache
def _proj_pre_op():
    return _collapse_vmap(_proj_pre_call(_TILE), 2)


def _proj_pre_ref(Rl, Rr):
    M = jnp.matmul(Rl, Rr)
    return M, jnp.matmul(M, jnp.swapaxes(M, -1, -2))


@jax.custom_vjp
def proj_pre(Rl, Rr):
    """Fused bond matrix M = Rl @ Rr_next and gram G = M @ M^T (the eigh8
    input); (..., 8, 8) blocks, any leading dims (incl. the sector axis)."""
    return _proj_pre_op()(Rl, Rr)


def _proj_pre_fwd(Rl, Rr):
    return _proj_pre_op()(Rl, Rr), (Rl, Rr)


def _proj_pre_bwd(res, cts):
    with _fp32_replay():
        _, vjp = jax.vjp(_proj_pre_ref, *res)
        return vjp(cts)


proj_pre.defvjp(_proj_pre_fwd, _proj_pre_bwd)


@lru_cache
def _proj_post_op():
    return _collapse_vmap(_proj_post_call(_TILE), 2)


def _proj_post_ref(M, U2, w2, Rl, Rr):
    """The tail of fpeps_fast_layer.projectors after eigh, verbatim."""
    s2 = jnp.clip(w2, 0.0, None)
    sVH = jnp.matmul(jnp.swapaxes(U2, -1, -2), M)
    s = jnp.sqrt(s2)
    eps = jnp.finfo(s.dtype).eps
    smax = s[..., -1:]
    cutoff = smax * eps * 8
    sinv = s / (s**2 + cutoff**2)
    VH = sinv[..., :, None] * sVH
    st = jnp.sqrt(s)
    Vd = jnp.swapaxes(VH, -1, -2) / st[..., None, :]
    Pl = jnp.matmul(Rr, Vd)
    Ud = jnp.swapaxes(U2, -1, -2) / st[..., :, None]
    Pr = jnp.matmul(Ud, Rl)
    return Pl, Pr


@jax.custom_vjp
def proj_post(M, U2, w2, Rl, Rr):
    """Projector assembly after eigh8: (M, top-2 eigvecs U2, top-2 eigvals
    w2 ascending, Rl, Rr_next) -> (Pl (..., 8, 2), Pr (..., 2, 8))."""
    return _proj_post_op()(M, U2, w2, Rl, Rr)


def _proj_post_fwd(M, U2, w2, Rl, Rr):
    return _proj_post_op()(M, U2, w2, Rl, Rr), (M, U2, w2, Rl, Rr)


def _proj_post_bwd(res, cts):
    with _fp32_replay():
        _, vjp = jax.vjp(_proj_post_ref, *res)
        return vjp(cts)


proj_post.defvjp(_proj_post_fwd, _proj_post_bwd)


@lru_cache
def _patch_op():
    return _collapse_vmap(_patch_call(_TILE), 1)


def _patch_impl(t0, t1, P1, P2, pi):
    t0f = t0.reshape(t0.shape[:-4] + (16,))
    t1f = t1.reshape(t1.shape[:-4] + (16,))
    return _patch_op()(t0f, t1f, P1, P2, pi)


def _patch_ref(t0, t1, P1, P2, pi):
    """jnp reference: the patch segment of fpeps_fast_layer.fast_compress_y
    up to the slot-order output (..., 8, 16), pre canonical permutation."""
    nb = pi.ndim
    Ap = FL._lut_gather(t0, pi, "Al_patch")  # (c,m,C,K,s, r,u,M,l)
    Bp = FL._lut_gather(t1, pi, "Bl_patch")  # (c,m,C,K,s, R,d,L,M)
    Ap = Ap.reshape(Ap.shape[:-2] + (2,) * 9)
    Bp = Bp.reshape(Bp.shape[:-2] + (2,) * 9)
    P1v = P1.reshape(P1.shape[:-1] + (2, 2, 2))  # (c, t, m, l, L)
    tmp = jnp.einsum("...cmCKsruMl,...ctmlL->...cmCKstruLM", Ap, P1v)
    D1 = jnp.einsum("...cmCKstruLM,...cmCKsRdLM->...CcmtsudKrR", tmp, Bp)
    D1 = D1.reshape(D1.shape[:nb] + (2, 64, 8))
    D2 = jnp.einsum("...Cxk,...CkT->...CxT", D1, P2)
    D2 = D2.reshape(D2.shape[:nb] + (2,) * 8)
    D2 = D2.sum(nb + 2)
    out = jnp.transpose(
        D2, tuple(range(nb)) + tuple(nb + np.array([1, 0, 3, 2, 6, 4, 5]))
    )  # (..., c, cr, us, t, T, u, d)
    return out.reshape(out.shape[:nb] + (8, 16))


@jax.custom_vjp
def _patch_slots_op(t0, t1, P1, P2, pi):
    return _patch_impl(t0, t1, P1, P2, pi)


def _patch_fwd(t0, t1, P1, P2, pi):
    return _patch_impl(t0, t1, P1, P2, pi), (t0, t1, P1, P2, pi)


def _patch_bwd(res, ct):
    t0, t1, P1, P2, pi = res
    with _fp32_replay():
        _, vjp = jax.vjp(lambda a, b, c, d: _patch_ref(a, b, c, d, pi),
                         t0, t1, P1, P2)
        dt0, dt1, dP1, dP2 = vjp(ct)
    return dt0, dt1, dP1, dP2, _float0(pi)


_patch_slots_op.defvjp(_patch_fwd, _patch_bwd)


def patch_slots(t0, t1, P1, P2, pi, tile=128):
    """Slot-ordered projected pair tensor (..., 8 slots (c, C, s), 16 dims
    (t, T, u, d)); the caller applies the canonical row permutation.
    Differentiable and vmap-collapsing at the default tile."""
    if tile == _TILE:
        return _patch_slots_op(t0, t1, P1, P2, pi)
    t0f = t0.reshape(t0.shape[:-4] + (16,))
    t1f = t1.reshape(t1.shape[:-4] + (16,))
    return _patch_call(tile)(t0f, t1f, P1, P2, pi)


# --------------------------------------------------------------------------
# fused line-blob kernel for the final exact contraction: the blob pre-GEMM
# (65536 floats per line of which only half is extracted), the calibrated
# extraction gather and the ring canon select/sign were three full memory
# passes; this kernel computes each canon'd blob element directly from the
# two (256, 8) pair operands (L1-resident per unit; the EXT tables have only
# 16 P4 variants and stay in L2). Dominant cost of the Symm forward final
# stage before this (66% of the total at 64x64).


def _blob_kernel(d01_ref, d23_ref, mi_ref, mf_ref, eidx_ref, esgn_ref,
                 o_ref):
    """out[u, k, r, c] = csgn[u, k] * EXT_SGN[p4, csel, r, c]
    * sum_K D01[u, row(p4, csel, r, c), K] * D23[u, col(...), K].
    Block: one (unit, ring slot k, row tile) -> (1, 1, TR, 128)."""
    kk = pl.program_id(1)
    rt = pl.program_id(2)
    TR = o_ref.shape[2]
    p4 = mi_ref[0, 0]
    cs = mi_ref[0, 1 + kk]
    sk = mf_ref[0, kk]
    base = (p4 * 2 + cs) * 16384 + rt * (TR * 128)
    ir = jax.lax.broadcasted_iota(jnp.int32, (TR, 128), 0)
    ic = jax.lax.broadcasted_iota(jnp.int32, (TR, 128), 1)
    off = base + ir * 128 + ic
    idxv = eidx_ref[off]
    sg = esgn_ref[off]
    row = jax.lax.div(idxv, jnp.int32(256)) * 8
    col = jax.lax.rem(idxv, jnp.int32(256)) * 8
    acc = d01_ref[0, row] * d23_ref[0, col]
    for K in range(1, 8):
        acc = acc + d01_ref[0, row + K] * d23_ref[0, col + K]
    o_ref[0, 0, :, :] = acc * (sg * sk)


@lru_cache
def _blob_call(tr):
    _, _, EIDX, ESGN = FL._blob_tables()
    EIDXn = EIDX.reshape(-1).astype(np.int32)
    ESGNn = ESGN.reshape(-1).astype(np.float32)

    def call(D01k, D23k, p4, csel, csgn):
        eidx = jnp.asarray(EIDXn)
        esgn = jnp.asarray(ESGNn)
        nb = p4.shape
        n = int(np.prod(nb))
        half = D01k.dtype in (jnp.float16, jnp.bfloat16)
        ctype = jnp.float32 if half else D01k.dtype
        if D01k.dtype == jnp.float64:
            esgn = esgn.astype(jnp.float64)
        a = D01k.reshape(n, 2048).astype(ctype)
        b = D23k.reshape(n, 2048).astype(ctype)
        mi = jnp.concatenate(
            [p4.reshape(n, 1).astype(jnp.int32),
             csel.reshape(n, 2).astype(jnp.int32)], axis=1)
        mf = csgn.reshape(n, 2).astype(ctype)
        out = pl.pallas_call(
            _blob_kernel,
            grid=(n, 2, 128 // tr),
            in_specs=[
                pl.BlockSpec((1, 2048), lambda u, k, r: (u, 0)),
                pl.BlockSpec((1, 2048), lambda u, k, r: (u, 0)),
                pl.BlockSpec((1, 3), lambda u, k, r: (u, 0)),
                pl.BlockSpec((1, 2), lambda u, k, r: (u, 0)),
                pl.BlockSpec((16 * 32768,), lambda u, k, r: (0,)),
                pl.BlockSpec((16 * 32768,), lambda u, k, r: (0,)),
            ],
            out_specs=pl.BlockSpec((1, 1, tr, 128),
                                   lambda u, k, r: (u, k, r, 0)),
            out_shape=jax.ShapeDtypeStruct((n, 2, 128, 128), ctype),
            compiler_params=_TritonParams(
                num_warps=int(os.environ.get("FK_BLOB_WARPS",
                                             max(1, tr * 128 // 1024)))),
        )(a, b, mi, mf, eidx, esgn)
        out = out.reshape(nb + (2, 128, 128))
        return out.astype(D01k.dtype) if half else out

    return call


_BLOB_TR = int(os.environ.get("FK_BLOB_TR", 32))


@lru_cache
def _blob_op():
    return _collapse_vmap(_blob_call(_BLOB_TR), 1)


def _blob_ref(D01k, D23k, p4, csel, csgn):
    """jnp reference (= fast_line_blobs' pre einsum + extraction + the ring
    canon select/sign of _fast_raw_final_contract); defines the backward."""
    _, _, EIDX, ESGN = FL._blob_tables()
    pre = jnp.einsum("...rK,...cK->...rc", D01k, D23k)
    pre = pre.reshape(pre.shape[:-2] + (65536,))
    idx = jnp.asarray(EIDX.astype(np.int32))[p4]
    sgn = jnp.asarray(ESGN, D01k.dtype)[p4]
    blobs = jnp.take_along_axis(pre, idx, axis=-1) * sgn
    blobs = blobs.reshape(blobs.shape[:-1] + (2, 128, 128))
    blobs = jnp.take_along_axis(
        blobs, csel[..., None, None].astype(jnp.int32), axis=-3)
    return blobs * csgn[..., None, None].astype(blobs.dtype)


@jax.custom_vjp
def blob_canon(D01k, D23k, p4, csel, csgn):
    """Canon'd line blobs (..., 2, 128, 128) from the pair operands
    D01k/D23k (..., 256, 8) (K = (cb, ud, dd) for BOTH — D23 pre-swapped),
    P4 line parity index p4 (...,), ring block select csel (..., 2) and ring
    sign csgn (..., 2). Differentiable and vmap-collapsing."""
    return _blob_op()(D01k, D23k, p4, csel, csgn)


def _blob_fwd(D01k, D23k, p4, csel, csgn):
    return _blob_op()(D01k, D23k, p4, csel, csgn), (D01k, D23k, p4, csel,
                                                    csgn)


def _blob_bwd(res, ct):
    D01k, D23k, p4, csel, csgn = res
    _, vjp = jax.vjp(lambda a, b, sg: _blob_ref(a, b, p4, csel, sg),
                     D01k, D23k, csgn)
    dA, dB, dsg = vjp(ct)
    return dA, dB, _float0(p4), _float0(csel), dsg


blob_canon.defvjp(_blob_fwd, _blob_bwd)


# --------------------------------------------------------------------------
# tensor-core blob-build kernel: the factored EXT tables make each canon'd
# blob 128 charge blocks of 16x16, block = D01[q] (16,8) @ D23t[:,Q] (8,16)
# times a charge-only sign, followed by the partial transpose
# [(a,b),(A,B)] -> [(a,A),(b,B)]. The XLA path pays ~655 KB/line of HBM for
# the gather-expansion + einsum; this kernel reads the 32 KB/line pair
# operands once, runs the block dots on tensor cores (K=8 tf32) with the
# partial transpose as a register permute, and block-stores the blob —
# measured 2.2x (0.47 -> 0.215 us/line), and BITWISE equal to the XLA path
# at both tf32 and HIGHEST (same K-order accumulation).


def _blob_build_kernel(d01_ref, d23t_ref, p4_ref, csel_ref, csgn_ref,
                       qt_ref, Qt_ref, st_ref, o_ref, *, nuc):
    """Grid (lines, 2 slots, 8//nuc): one program builds nuc*8 charge
    blocks of one (line, ring-slot) blob. Blocks: (1,256,8)/(1,8,256) pair
    operands, (1,1,nuc*16,128) output tile."""
    k = pl.program_id(1)
    ug = pl.program_id(2)
    p4 = p4_ref[0]
    cs = csel_ref[0, k]
    sg = csgn_ref[0, k]
    base = p4 * 128 + cs * 64

    def body(i, _):
        # jnp % / // lower badly in triton; operands non-negative
        iu = jax.lax.div(i, 8)
        u = ug * nuc + iu
        d = jax.lax.rem(i, 8)
        idx = base + u * 8 + d
        q = qt_ref[idx]
        Q = Qt_ref[idx]
        s = st_ref[idx] * sg
        G1 = d01_ref[0, pl.ds(q * 16, 16), :]      # (16=(a,b), K8)
        G2 = d23t_ref[0, :, pl.ds(Q * 16, 16)]     # (K8, 16=(A,B))
        T = jnp.dot(G1, G2, preferred_element_type=jnp.float32) * s
        # partial transpose [(a,b),(A,B)] -> [(a,A),(b,B)]
        B = T.reshape(4, 4, 4, 4).transpose(0, 2, 1, 3).reshape(16, 16)
        o_ref[0, 0, pl.ds(iu * 16, 16), pl.ds(d * 16, 16)] = B
        return 0

    jax.lax.fori_loop(0, nuc * 8, body, 0)


_BLOBK_NUC = int(os.environ.get("FK_BLOBK_NUC", 2))
_BLOBK_WARPS = int(os.environ.get("FK_BLOBK_WARPS", 2))


@lru_cache
def _blob_build_tabs():
    qt, Qt, st = FL._blob_tables_factored()  # (16, 2, 8, 8) each
    return (qt.reshape(-1).astype(np.int32),
            Qt.reshape(-1).astype(np.int32),
            st.reshape(-1).astype(np.float32))


@lru_cache
def _blob_build_call(nuc, warps):
    def call(D01r, D23t, P4, csel, csgn):
        nb = P4.shape
        n = int(np.prod(nb))
        d01 = D01r.reshape(n, 256, 8)
        d23 = D23t.reshape(n, 8, 256)
        p4 = P4.reshape(n).astype(jnp.int32)
        cs = csel.reshape(n, 2).astype(jnp.int32)
        sg = csgn.reshape(n, 2).astype(jnp.float32)
        qtn, Qtn, stn = _blob_build_tabs()
        out = pl.pallas_call(
            partial(_blob_build_kernel, nuc=nuc),
            grid=(n, 2, 8 // nuc),
            in_specs=[
                pl.BlockSpec((1, 256, 8), lambda l, k, u: (l, 0, 0)),
                pl.BlockSpec((1, 8, 256), lambda l, k, u: (l, 0, 0)),
                pl.BlockSpec((1,), lambda l, k, u: (l,)),
                pl.BlockSpec((1, 2), lambda l, k, u: (l, 0)),
                pl.BlockSpec((1, 2), lambda l, k, u: (l, 0)),
                pl.BlockSpec((2048,), lambda l, k, u: (0,)),
                pl.BlockSpec((2048,), lambda l, k, u: (0,)),
                pl.BlockSpec((2048,), lambda l, k, u: (0,)),
            ],
            out_specs=pl.BlockSpec((1, 1, nuc * 16, 128),
                                   lambda l, k, u: (l, k, u, 0)),
            out_shape=jax.ShapeDtypeStruct((n, 2, 128, 128), jnp.float32),
            compiler_params=_TritonParams(num_warps=warps),
        )(d01, d23, p4, cs, sg, jnp.asarray(qtn), jnp.asarray(Qtn),
          jnp.asarray(stn))
        return out.reshape(nb + (2, 128, 128))

    return call


@lru_cache
def _blob_build_op():
    return _collapse_vmap(_blob_build_call(_BLOBK_NUC, _BLOBK_WARPS), 1)


def _blob_build_impl(line_params, line_charge, csel, csgn):
    half = line_params.dtype in (jnp.float16, jnp.bfloat16)
    lp = line_params.astype(jnp.float32) if half else line_params
    D01r, D23s, P4 = FL._blob_pair_operands(lp, line_charge)
    D23t = jnp.swapaxes(D23s, -1, -2)  # (..., 8, 256), kernel layout
    out = _blob_build_op()(D01r, D23t, P4, csel,
                           csgn.astype(jnp.float32))
    return out.astype(line_params.dtype) if half else out


@jax.custom_vjp
def blob_build(line_params, line_charge, csel, csgn):
    """Canon'd ring-ready blobs (..., nline, 2, 128, 128) via the
    tensor-core block-dot kernel; BITWISE equal to
    fpeps_fast_layer.fast_line_blobs_canon_xla (which is the backward
    replay, so forward and backward are exactly consistent). fp16/bf16 run
    in fp32; f64 callers use the XLA path. Differentiable and
    vmap-collapsing."""
    return _blob_build_impl(line_params, line_charge, csel, csgn)


def _blob_build_fwd(line_params, line_charge, csel, csgn):
    return (_blob_build_impl(line_params, line_charge, csel, csgn),
            (line_params, line_charge, csel, csgn))


def _blob_build_bwd(res, ct):
    line_params, line_charge, csel, csgn = res
    _, vjp = jax.vjp(
        lambda p, sg: FL.fast_line_blobs_canon_xla(p, line_charge, csel, sg),
        line_params, csgn,
    )
    dp, dsg = vjp(ct)
    return dp, _float0(line_charge), _float0(csel), dsg


blob_build.defvjp(_blob_build_fwd, _blob_build_bwd)


# --------------------------------------------------------------------------
# fused ring-chain kernel for the final contraction: the XLA ring is
# nline-1 separate batched (2,128,128) GEMMs with the accumulator (and the
# strip passes) round-tripping HBM between every step. Here one program per
# (unit, ring slot) keeps the (128,128) accumulator in registers across the
# whole chain, streams each blob from HBM exactly once, strips in-register,
# and closes with the elementwise trace. Measured 2.9-3.1x over the XLA
# ring at the production 16/32-line grids (1 A100, fp32/tf32).


def _ring_kernel(blobs_ref, sign_ref, ex_ref, *, nline, cadence):
    """blobs_ref block (1, nline, 1, 128, 128): the canon'd blobs of one
    (unit, ring slot k). Writes the stripped chain trace and its log10
    exponent. Strip semantics mirror _strip_blocks (zero/nonfinite-guarded),
    but per ring slot instead of jointly over both — the caller recombines
    at the common max exponent."""
    dt = blobs_ref.dtype
    ln10 = jnp.array(np.log(10.0), dt)
    acc0 = blobs_ref[0, 0, 0]
    ex0 = jnp.zeros((), dt)

    def body(i, carry):
        acc, ex = carry
        b = blobs_ref[0, i, 0]
        acc = jnp.dot(acc, b, preferred_element_type=dt)
        do = jax.lax.rem(i, cadence) == 0
        nrm = jnp.sqrt(jnp.sum(acc * acc))
        good = (nrm > 0.0) & (nrm == nrm) & (nrm < jnp.array(np.inf, dt))
        nrm_eff = jnp.where(do & good, nrm, jnp.array(1.0, dt))
        acc = acc * (1.0 / nrm_eff)
        ex = ex + jnp.log(nrm_eff) / ln10
        return acc, ex

    acc, ex = jax.lax.fori_loop(1, nline - 1, body, (acc0, ex0))
    b_last = blobs_ref[0, nline - 1, 0]
    sign_ref[0, 0] = jnp.sum(acc * b_last.T)
    ex_ref[0, 0] = ex


_RING_WARPS = int(os.environ.get("FK_RING_WARPS", 8))


@lru_cache
def _ring_call(nline, cadence, warps):
    def call(blobs):
        nb = blobs.shape[:-4]
        n = int(np.prod(nb))
        b = blobs.reshape((n,) + blobs.shape[-4:])
        sign, ex = pl.pallas_call(
            partial(_ring_kernel, nline=nline, cadence=cadence),
            grid=(n, 2),
            in_specs=[
                pl.BlockSpec((1, nline, 1, 128, 128),
                             lambda u, k: (u, 0, k, 0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((1, 1), lambda u, k: (u, k)),
                pl.BlockSpec((1, 1), lambda u, k: (u, k)),
            ],
            out_shape=[
                jax.ShapeDtypeStruct((n, 2), blobs.dtype),
                jax.ShapeDtypeStruct((n, 2), blobs.dtype),
            ],
            # the chained dot stages BOTH operands in SMEM (dot-output ->
            # dot-lhs layout conversion), so 128x128 fp32 only fits with
            # num_stages=1; warps < 8 spill registers (24x slower)
            compiler_params=_TritonParams(num_warps=warps, num_stages=1),
        )(b)
        return sign.reshape(nb + (2,)), ex.reshape(nb + (2,))

    return call


@lru_cache
def _ring_op(nline, cadence):
    return _collapse_vmap(_ring_call(nline, cadence, _RING_WARPS), 2)


def _ring_impl(blobs):
    nline = blobs.shape[-4]
    half = blobs.dtype in (jnp.float16, jnp.bfloat16)
    b = blobs.astype(jnp.float32) if half else blobs
    s, e = _ring_op(nline, 2)(b)
    # recombine the two ring slots at the common max exponent
    E = jnp.max(e, axis=-1)
    sign = jnp.sum(s * 10.0 ** (e - E[..., None]), axis=-1)
    if half:
        return sign.astype(blobs.dtype), E.astype(blobs.dtype)
    return sign, E


def _ring_ref(blobs):
    """jnp replay of ring_chain's OWN output decomposition — per-slot
    strips (fp32 compute, cadence 2, same zero/nonfinite guard) and the
    max-exponent recombination — leading dims allowed. Its vjp defines the
    backward. The decomposition MUST match `_ring_impl` exactly: the
    (sign, exponent) split differs between per-slot and shared-strip
    conventions by a per-sample power of 10, so replaying a different
    split hands the custom_vjp mismatched `1/sign` cotangents and scales
    gradients wrongly per sample (caught by the model-level fp32 backward
    A/B: 19% grad error at 2e-5 forward error). Strip norms and exponents
    are stop-gradded (pure rescalings), so log-derivatives stay exact."""
    nline = blobs.shape[-4]
    half = blobs.dtype in (jnp.float16, jnp.bfloat16)
    b = blobs.astype(jnp.float32) if half else blobs
    exponent = jnp.zeros(b.shape[:-4] + (2,), b.dtype)
    acc = b[..., 0, :, :, :]
    for i in range(1, nline - 1):
        acc = jnp.einsum("...kij,...kjl->...kil", acc, b[..., i, :, :, :])
        if i % 2 == 0:
            nrm = jnp.sqrt(jnp.sum(acc * acc, axis=(-2, -1), keepdims=True))
            nrm = jax.lax.stop_gradient(nrm)
            nrm = jnp.where((nrm == 0.0) | ~jnp.isfinite(nrm), 1.0, nrm)
            exponent = exponent + jnp.log10(nrm[..., 0, 0])
            acc = acc / nrm
    last = jnp.swapaxes(b[..., nline - 1, :, :, :], -1, -2)
    s = jnp.sum(acc * last, axis=(-2, -1))  # (..., 2)
    E = jnp.max(exponent, axis=-1)
    sign = jnp.sum(s * 10.0 ** (exponent - E[..., None]), axis=-1)
    if half:
        return sign.astype(blobs.dtype), E.astype(blobs.dtype)
    return sign, E


@jax.custom_vjp
def ring_chain(blobs):
    """Stripped ring-chain trace of the canon'd blobs (..., nline, 2, 128,
    128) -> (sign, exponent-log10), one Pallas program per (unit, slot) with
    the accumulator in registers. fp32 compute (fp16/bf16 upcast); f64
    callers must use the jnp loop (the 128x128 f64 dot exceeds Triton SMEM).
    Differentiable (jnp-reference backward) and vmap-collapsing."""
    return _ring_impl(blobs)


def _ring_fwd(blobs):
    return _ring_impl(blobs), (blobs,)


def _ring_bwd(res, cts):
    (blobs,) = res
    _, vjp = jax.vjp(_ring_ref, blobs)
    return vjp(cts)


ring_chain.defvjp(_ring_fwd, _ring_bwd)
