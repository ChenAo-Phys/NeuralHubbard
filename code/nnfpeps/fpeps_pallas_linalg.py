"""Fused batched linalg kernels for the fPEPS D=4 (8x8-block) hot path.

The HOTRG projector computation at D=4, chi=4 spends ~99% of its GPU time in
cuSOLVER batched syevj (8x8 eigh) and potrf (8x8 cholesky). Both are heavily
latency-bound for huge batches of tiny matrices. Replacements:

- ``eigh8``: one fused Pallas(Triton) kernel per batch running a fully
  unrolled cyclic Jacobi (round-robin parallel ordering, 4 disjoint rotations
  per round, 7 rounds per sweep) with the 8x8 matrices held in registers.
  Single pass over memory. Returns (eigenvalues ascending, eigenvectors as
  columns), the jnp.linalg.eigh convention. Gradient via the same stabilized
  custom_vjp formula as fPEPS_hotrg_symm.eigh_single (batched).

- ``chol8``: unrolled Cholesky-Banachiewicz in plain jnp ops (differentiable,
  XLA-fusable, no cuSOLVER call): ~2x faster than batched potrf and CUDA-graph
  capturable.

Verified element-wise against jnp.linalg.eigh / jnp.linalg.cholesky.
"""

from functools import lru_cache, partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.triton as plt

# jax renamed TritonCompilerParams -> CompilerParams around 0.11
_TritonParams = getattr(plt, "CompilerParams", None) or plt.TritonCompilerParams


def _roundrobin_rounds(n):
    """Round-robin tournament: n-1 rounds of n/2 disjoint (p, q) pairs."""
    players = list(range(n))
    rounds = []
    for _ in range(n - 1):
        pairs = [(players[i], players[n - 1 - i]) for i in range(n // 2)]
        rounds.append([(min(p, q), max(p, q)) for p, q in pairs])
        players = [players[0]] + [players[-1]] + players[1:-1]
    return rounds


_ROUNDS8 = _roundrobin_rounds(8)


def _eigh8_kernel(a_ref, w_ref, v_ref, *, nsweeps):
    """Cyclic Jacobi with the 8x8 matrix held as 64+64 per-element (TILE,)
    register vectors (python-list static addressing -- Pallas Triton cannot
    slice/stack register arrays). One fori_loop iteration = one full sweep of
    28 statically-unrolled Givens rotations; each rotation runs the textbook
    symmetric update touching only rows/cols p, q (~40 vector ops)."""
    dt = a_ref.dtype
    one = jnp.array(1.0, dt)
    mone = jnp.array(-1.0, dt)  # avoid unary neg on constants (triton bug)

    # per-element strided loads; the block is L1-resident so this reads the
    # tile from GMEM essentially once
    a = [[a_ref[:, i, j] for j in range(8)] for i in range(8)]
    for i in range(8):
        for j in range(i):  # symmetrize once
            aij = (a[i][j] + a[j][i]) * 0.5
            a[i][j] = aij
            a[j][i] = aij
    v = [[jnp.full_like(a[0][0], 1.0 if i == j else 0.0) for j in range(8)]
         for i in range(8)]

    def sweep(_, carry):
        a = [list(row) for row in carry[0]]
        v = [list(row) for row in carry[1]]
        for pairs in _ROUNDS8:
            for p, q in pairs:
                app, aqq, apq = a[p][p], a[q][q], a[p][q]
                safe = jnp.where(apq == 0.0, one, apq)
                tau = (aqq - app) / (2.0 * safe)
                sgn = jnp.where(tau >= 0.0, one, mone)
                t = sgn / (jnp.abs(tau) + jnp.sqrt(1.0 + tau * tau))
                t = jnp.where(apq == 0.0, apq * 0.0, t)
                c = 1.0 / jnp.sqrt(1.0 + t * t)
                s = t * c
                for j in range(8):
                    if j == p or j == q:
                        continue
                    ajp = a[j][p]
                    ajq = a[j][q]
                    njp = c * ajp - s * ajq
                    njq = s * ajp + c * ajq
                    a[j][p] = njp
                    a[p][j] = njp
                    a[j][q] = njq
                    a[q][j] = njq
                napp = app - t * apq
                naqq = aqq + t * apq
                a[p][p] = napp
                a[q][q] = naqq
                a[p][q] = apq * 0.0
                a[q][p] = apq * 0.0
                for i in range(8):
                    vip = v[i][p]
                    viq = v[i][q]
                    v[i][p] = c * vip - s * viq
                    v[i][q] = s * vip + c * viq
        return (tuple(map(tuple, a)), tuple(map(tuple, v)))

    a, v = jax.lax.fori_loop(
        0, nsweeps, sweep,
        (tuple(map(tuple, a)), tuple(map(tuple, v))),
    )
    a = [list(row) for row in a]
    v = [list(row) for row in v]
    # ascending sort by eigenvalue with a Batcher odd-even network (19
    # compare-exchanges), swapping eigenvector columns along: replaces the
    # argsort + take_along_axis passes outside the kernel
    w = [a[i][i] for i in range(8)]
    _NET8 = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (1, 2), (5, 6), (0, 4), (3, 7), (1, 5), (2, 6), (1, 4), (3, 6),
             (2, 4), (3, 5), (3, 4)]
    for i, j in _NET8:
        gt = w[i] > w[j]
        wi, wj = w[i], w[j]
        w[i] = jnp.where(gt, wj, wi)
        w[j] = jnp.where(gt, wi, wj)
        for k in range(8):
            vki, vkj = v[k][i], v[k][j]
            v[k][i] = jnp.where(gt, vkj, vki)
            v[k][j] = jnp.where(gt, vki, vkj)
    for i in range(8):
        w_ref[:, i] = w[i]
        for j in range(8):
            v_ref[:, i, j] = v[i][j]


def _eigh8_raw(x, nsweeps=8, tile=128):
    """Batched 8x8 symmetric eigh, eigenvalues ascending. x: (nb, 8, 8)."""
    nb = x.shape[0]
    pad = (-nb) % tile
    if pad:
        eye = jnp.broadcast_to(jnp.eye(8, dtype=x.dtype), (pad, 8, 8))
        x = jnp.concatenate([x, eye], axis=0)
    npad = x.shape[0]
    w, v = pl.pallas_call(
        partial(_eigh8_kernel, nsweeps=nsweeps),
        grid=(npad // tile,),
        in_specs=[pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0))],
        out_specs=[
            pl.BlockSpec((tile, 8), lambda i: (i, 0)),
            pl.BlockSpec((tile, 8, 8), lambda i: (i, 0, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((npad, 8), x.dtype),
            jax.ShapeDtypeStruct((npad, 8, 8), x.dtype),
        ],
        # Triton backend (mosaic-gpu lacks sqrt); one thread per matrix lane
        compiler_params=_TritonParams(num_warps=max(1, tile // 32)),
    )(x)
    if pad:
        w, v = w[:nb], v[:nb]
    return w, v


@lru_cache
def _eigh8_call(nsweeps, tile):
    """custom_vmap wrapper: the fPEPS forward calls eigh on tiny (2, 8, 8)
    per-site blocks under nested jax.vmap (samples x site pairs). Without this
    rule each vmap element pads 2 -> tile inside the kernel (64x wasted work);
    with it all vmap levels collapse into the kernel's batch dimension."""

    @jax.custom_batching.custom_vmap
    def call(x):  # (nb, 8, 8), unbatched semantics
        return _eigh8_raw(x, nsweeps=nsweeps, tile=tile)

    @call.def_vmap
    def _rule(axis_size, in_batched, x):
        batch = x.shape[:-2]
        w, v = call(x.reshape((-1,) + x.shape[-2:]))
        return (w.reshape(batch + (8,)), v.reshape(batch + (8, 8))), (True, True)

    return call


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def eigh8(x, nsweeps=8, tile=128):
    """Batched (..., 8, 8) symmetric eigh via one fused Pallas Jacobi kernel.

    Returns (w, v): eigenvalues ascending and eigenvectors as columns, the
    jnp.linalg.eigh convention. The trace shift and gradient stabilization
    mirror fPEPS_hotrg_symm.eigh_single.
    """
    batch = x.shape[:-2]
    x = x.reshape(-1, 8, 8)
    tr = jnp.diagonal(x, axis1=-2, axis2=-1).sum(-1)
    m = jax.lax.stop_gradient(tr) * 1e-6
    x = x + m[:, None, None] * jnp.eye(8, dtype=x.dtype)
    w, v = _eigh8_call(nsweeps, tile)(x)
    return w.reshape(batch + (8,)), v.reshape(batch + (8, 8))


def _eigh8_fwd(x, nsweeps, tile):
    w, v = eigh8(x, nsweeps, tile)
    return (w, v), (w, v)


def _eigh8_bwd(nsweeps, tile, res, g):
    # batched version of fPEPS_hotrg_symm._eigh_bwd (w <-> eigenvalues here).
    # The Jacobi forward is scalar fp32 arithmetic; pin the backward's 8x8
    # matmuls to full fp32 as well (they would drop to TF32 on Ampere under
    # the default precision and hand back the gradient of a perturbed forward).
    with jax.default_matmul_precision("highest"):
        return _eigh8_bwd_impl(w_v=res, g=g)


def _eigh8_bwd_impl(w_v, g):
    w, v = w_v
    gw, gv = g
    if gw is None:
        gw = jnp.zeros_like(w)
    if gv is None:
        gv = jnp.zeros_like(v)

    vH = jnp.swapaxes(v.conj(), -1, -2)
    S = vH @ gv
    Aanti = S - jnp.swapaxes(S.conj(), -1, -2)

    dw = w[..., :, None] - w[..., None, :]
    eps = jnp.max(jnp.abs(w), axis=-1) * 1e-6
    eps = jnp.where(eps < 1e-12, 1e-12, eps)
    eps = jax.lax.stop_gradient(eps)[..., None, None]
    F = dw / (dw * dw + eps * eps)
    F = jnp.where(jnp.eye(8, dtype=bool), 0.0, F)

    G = jnp.zeros_like(v) + F * Aanti * (-0.5)
    i = jnp.arange(8)
    G = G.at[..., i, i].add(gw)

    gA = v @ G @ vH
    gA = 0.5 * (gA + jnp.swapaxes(gA.conj(), -1, -2))
    return (gA,)


eigh8.defvjp(_eigh8_fwd, _eigh8_bwd)


@jax.custom_vjp
def chol8(x):
    """Unrolled batched Cholesky for (..., n, n) real x with small static n.

    Reads only the lower triangle (symmetric-input convention). The Schur
    pivot is floored at eps*|x_jj| before the sqrt: healthy pivots pass
    through bitwise, but a near-singular sector block (pivot ~ 1e-4 * trace
    from the gram shift) recomputed by TF32 GEMMs -- e.g. inside the jnp
    replay of gram_chol's vjp -- can round negative, and an unguarded sqrt
    would turn the whole backward into NaN.
    """
    n = x.shape[-1]
    eps = jnp.finfo(x.dtype).eps
    tiny = jnp.finfo(x.dtype).tiny
    L = jnp.zeros_like(x)
    for j in range(n):
        r = x[..., j:, j] - jnp.einsum(
            "...ik,...k->...i", L[..., j:, :j], L[..., j, :j]
        )
        floor = eps * jnp.abs(x[..., j : j + 1, j]) + tiny
        col = r / jnp.sqrt(jnp.maximum(r[..., :1], floor))
        L = L.at[..., j:, j].set(col)
    return L


def _chol8_solve_LT(L, B):
    """X = L^-T B by unrolled back substitution (L lower, small static n).

    precision="highest": these are tiny (n <= 16) contractions, but as real
    dot_generals they would otherwise drop to TF32 on Ampere, reintroducing
    the very precision loss this backward exists to avoid.
    """
    n = L.shape[-1]
    X = jnp.zeros_like(B)
    for i in reversed(range(n)):
        acc = B[..., i, :] - jnp.einsum(
            "...k,...km->...m",
            L[..., i + 1 :, i],
            X[..., i + 1 :, :],
            precision="highest",
        )
        X = X.at[..., i, :].set(acc / L[..., i, i][..., None])
    return X


def _chol8_fwd(x):
    L = chol8(x)
    return L, L


def _chol8_bwd(L, Lbar):
    # Solve-based cholesky gradient (the jnp.linalg.cholesky rule), reusing
    # the forward L: autodiffing the unrolled loop replays the Schur pivots,
    # whose 1/r0^(3/2) sensitivities explode on the near-singular blocks the
    # forward floor tolerates. This form only divides by diag(L) > 0.
    # x is read as its lower triangle, so the cotangent of the (constant
    # zero) upper triangle of L is dropped and the gradient lives on tril.
    n = L.shape[-1]
    eye = jnp.eye(n, dtype=L.dtype)
    P = jnp.matmul(
        jnp.swapaxes(L, -1, -2), jnp.tril(Lbar), precision="highest"
    )
    P = jnp.tril(P, -1) + 0.5 * P * eye
    W = _chol8_solve_LT(L, P)
    W = jnp.swapaxes(_chol8_solve_LT(L, jnp.swapaxes(W, -1, -2)), -1, -2)
    xbar = jnp.tril(W + jnp.swapaxes(W, -1, -2)) - W * eye
    return (xbar,)


chol8.defvjp(_chol8_fwd, _chol8_bwd)
