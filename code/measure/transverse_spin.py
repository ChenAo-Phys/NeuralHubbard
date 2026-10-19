"""Transverse spin correlations, one packed operator call per reference site.

One Operator.Oloc per term wastes most of each forward chunk, because a single term
connects only a fraction of the samples. Instead, the terms of

    O_a = sum_{j != a} (S^+_a S^-_j + S^-_a S^+_j)

are summed into one operator, its connected configurations are generated and evaluated
once, and each local value is binned by the term it comes from, giving

    T[0, r] = <S^+_a S^-_{a+r}>,    T[1, r] = <S^-_a S^+_{a+r}>

for all r at once. quantax (>= 0.4.0) returns the row <s|O|s'> in
Operator.apply_off_diag, so a term is identified by how s' differs from s: S^+_a S^-_j
moves the up electron of a to j, S^-_a S^+_j the up electron of j to a. The same packing
measures the d-wave terms in dwave_packed.py.
"""

import copy
from functools import partial, reduce
from operator import add, iadd

import jax
import jax.numpy as jnp
from quantax.operator import annihilate_d, annihilate_u, create_d, create_u
from quantax.operator.operator import _get_conn, _get_conn_size


def Sp(i):
    """S^+_i = c^dag_{i up} c_{i dn}."""
    return create_u(*i) @ annihilate_d(*i)


def Sm(i):
    """S^-_i = c^dag_{i dn} c_{i up}."""
    return create_d(*i) @ annihilate_u(*i)


def sum_ops(terms):
    """Sum of operators, accumulated in place (reduce(add) has a quadratic cost)."""
    return reduce(iadd, terms[1:], copy.deepcopy(terms[0]))


def binned_sum(s, psi, segment, s_conn, psi_conn, H_conn, bins, nbins):
    """Inside jit: sum over the samples of <s|O|s'> psi(s') / psi(s), binned on every
    device by bins(s, s') into nbins entries (other bins are dropped)."""
    ndev = jax.device_count()
    nloc = s.shape[0] // ndev
    segment = segment.reshape(ndev, -1) - jnp.arange(ndev)[:, None] * nloc

    @jax.vmap
    def per_device(s, psi, segment, s_conn, psi_conn, H_conn):
        valid = segment >= 0  # padding entries have segment -1
        seg = jnp.where(valid, segment, 0)
        ratio = jnp.asarray(psi_conn / psi[seg])
        val = jnp.where(valid, jnp.real(ratio * H_conn), 0.0)
        return jax.ops.segment_sum(val, bins(s[seg], s_conn), num_segments=nbins)

    out = per_device(
        s.reshape(ndev, nloc, -1),
        psi.reshape(ndev, nloc),
        segment,
        s_conn.reshape(ndev, segment.shape[1], -1),
        psi_conn.reshape(ndev, -1),
        H_conn.reshape(ndev, -1),
    )
    return out.sum(axis=0)


def packed_sum(state, samples, op, kernel, *args):
    """Sum of kernel(s, psi, segment, s', psi(s'), <s|op|s'>, *args) over the groups of
    connected configurations of op. samples.psi must hold the amplitudes."""
    s, psi = samples.spins, samples.psi

    def group(s_conn, H_conn):
        size = _get_conn_size(H_conn, state.forward_chunk, False).item()
        segment, s_conn, H_conn = _get_conn(s_conn, H_conn, size, s, False)
        return kernel(s, psi, segment, s_conn, state(s_conn), H_conn, *args)

    return reduce(add, (group(sc, hc) for _, sc, hc in op.apply_off_diag(s)))


def reference_operator(ref, LX, LY):
    """O_a of the reference site a = ref."""
    sites = [(x, y) for x in range(LX) for y in range(LY) if (x, y) != tuple(ref)]
    return sum_ops([t for j in sites for t in (Sp(ref) @ Sm(j), Sm(ref) @ Sp(j))])


@partial(jax.jit, static_argnums=(6, 7, 8))
def _transverse_sum(s, psi, segment, s_conn, psi_conn, H_conn, ref_idx, LX, LY):
    N = LX * LY
    ax, ay = divmod(ref_idx, LY)

    def bins(s0, s1):
        gain = jnp.argmax(s1[:, :N] > s0[:, :N], axis=1)  # site gaining an up electron
        lose = jnp.argmax(s1[:, :N] < s0[:, :N], axis=1)  # site losing it
        chan = jnp.where(lose == ref_idx, 0, 1)  # 0: S^+_a S^-_j, 1: S^-_a S^+_j
        j = jnp.where(chan == 0, gain, lose)
        return chan * N + ((j // LY - ax) % LX) * LY + (j % LY - ay) % LY

    T = binned_sum(s, psi, segment, s_conn, psi_conn, H_conn, bins, 2 * N)
    return T.reshape(2, N)


@partial(jax.jit, static_argnums=(2, 3))
def _finalize(T, s, ref_idx, N):
    """Set r = 0 to n_up (1 - n_dn) and n_dn (1 - n_up) at a, divide by nsamples."""
    nu = s[:, ref_idx] > 0
    nd = s[:, N + ref_idx] > 0
    diag = jnp.stack([jnp.sum(nu & ~nd), jnp.sum(nd & ~nu)]).astype(T.dtype)
    return T.at[:, 0].set(diag) / s.shape[0]


def transverse_terms(state, samples, op, ref, LX, LY):
    """Sweep means T, shape (2, LX * LY), of the reference site ref with
    op = reference_operator(ref, LX, LY). samples.psi must hold the amplitudes."""
    ref_idx = int(ref[0]) * LY + int(ref[1])
    T = packed_sum(state, samples, op, _transverse_sum, ref_idx, LX, LY)
    return _finalize(T, samples.spins, ref_idx, LX * LY)
