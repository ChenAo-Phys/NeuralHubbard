"""Packed measurement of the d-wave terms of main text Eq. (2),

    F[bl, oL, br, oR] = <A^dag(bl, oL) A(br, oR)>,   G[i, j, s] = <c^dag_{i s} c_{j s}>,

for the left bonds and sites at R and all right bonds and sites, packed as in
transverse_spin.py. The terms sharing a left end are summed into one operator. Every
term moves at most one electron per spin, so a connected configuration s' identifies its
term by the sites at which the up and the down electron arrive (empty in s, occupied in
s'). The diagonal term of each sum, n_{a up} n_{b dn} or n_{i s}, is left out of the
operator and taken from the occupations.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .observables import SPINS, dwave_keys
from .transverse_spin import binned_sum, packed_sum, sum_ops


@partial(jax.jit, static_argnums=(8, 9))
def _term_sum(s, psi, segment, s_conn, psi_conn, H_conn, lut, ab, nterms, N):
    """Bin by lut[c * N + d] with c (d) the arrival site of the up (dn) electron, or
    ab[0] (ab[1]) if none arrives."""

    def bins(s0, s1):
        arrived = s1 > s0
        up, dn = arrived[:, :N], arrived[:, N:]
        c = jnp.where(jnp.any(up, axis=1), jnp.argmax(up, axis=1), ab[0])
        d = jnp.where(jnp.any(dn, axis=1), jnp.argmax(dn, axis=1), ab[1])
        return lut[c * N + d]

    return binned_sum(s, psi, segment, s_conn, psi_conn, H_conn, bins, nterms)


class PackedDwave:
    """F and G terms of dwave_keys(geo, R, primary_only), measured with one packed call
    per left bond and orientation and one per left site and spin."""

    def __init__(self, geo, R=(0, 0), primary_only=True):
        left_bonds, left_sites, self.F_keys, self.G_keys = dwave_keys(
            geo, R, primary_only
        )
        N = len(geo.sites)

        def ends(bond, o):
            return tuple(geo.index(p) for p in geo.string_sites(bond, o))

        right = [(br, oR) for br in geo.all_bonds for oR in (0, 1)]
        lut = np.full(N * N, -1, dtype=np.int32)
        for t, (br, oR) in enumerate(right):
            c, d = ends(br, oR)
            lut[c * N + d] = t
        cd = np.arange(N * N, dtype=np.int32)  # a hop term is its arrival site
        self.luts = dict(pair=lut, u=cd // N, d=cd % N)
        self.luts = {k: jnp.asarray(v) for k, v in self.luts.items()}
        self.nright = len(right)
        self.pairs = []  # ((a, b), position of the diagonal term, operator)
        for bl in left_bonds:
            for oL in (0, 1):
                op = sum_ops(
                    [
                        geo.pair_string(bl, oL) @ geo.pair_string(br, oR).H
                        for br, oR in right
                        if (br, oR) != (bl, oL)
                    ]
                )
                self.pairs.append((ends(bl, oL), right.index((bl, oL)), op))
        self.hops = [  # (i, spin, operator)
            (geo.index(i), s, sum_ops([geo.hop(i, j, s) for j in geo.sites if j != i]))
            for i in left_sites
            for s in SPINS
        ]
        self.left_sites = [geo.index(i) for i in left_sites]

    def measure(self, state, samples):
        """Sweep means F, G in the order of F_keys, G_keys."""
        s = samples.spins
        N = s.shape[1] // 2
        occ = s > 0

        def mean(op, lut, ab, nterms):
            ab = jnp.asarray(ab, dtype=jnp.int32)
            T = packed_sum(state, samples, op, _term_sum, lut, ab, nterms, N)
            return T / s.shape[0]

        F = []
        for (a, b), diag, op in self.pairs:
            T = mean(op, self.luts["pair"], (a, b), self.nright)
            nn = jnp.mean((occ[:, a] & occ[:, N + b]).astype(T.dtype))
            F.append(T.at[diag].set(nn))
        G = {}
        for i, sp, op in self.hops:
            T = mean(op, self.luts[sp], (i, i), N)
            n_i = jnp.mean(occ[:, i if sp == "u" else N + i].astype(T.dtype))
            G[i, sp] = T.at[i].set(n_i)
        G = [
            jnp.stack([G[i, "u"], G[i, "d"]], axis=1).reshape(-1)
            for i in self.left_sites
        ]
        return jnp.concatenate(F), jnp.concatenate(G)
