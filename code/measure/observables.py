"""Lattice geometry, operator keys and the diagonal estimators.

Site (x, y) of the periodic LX x LY lattice has index x * LY + y. Configurations are
(batch, 2N) arrays of +-1 occupations, the N spin-up modes first. A nearest-neighbour
bond is keyed (x, y, axis) by its lower endpoint, axis 0 along +x and 1 along +y.
A^dag(bond, o) = c^dag_{a up} c^dag_{b dn} with (a, b) the (lower, upper) endpoints for
o = 0 and swapped for o = 1, so that the bond singlet of main text Eq. (2) is
B^dag = (A^dag(o=0) + A^dag(o=1)) / sqrt(2) and

    Delta^dag_r = 1/4 sum_eta h_eta B^dag(r, r + eta),   h = +1 (x bonds), -1 (y bonds).

Only numpy is imported at module level, so that assemble.py runs without jax.
"""

from fractions import Fraction

import numpy as np

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
FORM_FACTOR = {(1, 0): 1.0, (-1, 0): 1.0, (0, 1): -1.0, (0, -1): -1.0}
SPINS = ("u", "d")
FAMILIES = {"pi4": Fraction(1, 4), "pi6": Fraction(1, 6), "pi8": Fraction(1, 8)}


def paper_family(tprime):
    """Stripe wavevector of the paper: q = pi/8 at t' = 0 and pi/4 at t' = -0.2."""
    return "pi8" if tprime == 0 else "pi4"


def _offset(L, q_over_pi):
    n = q_over_pi * L / 2  # k = 2 pi n / L = q
    if n.denominator != 1:
        raise ValueError(f"q = {q_over_pi} pi is not on the momentum grid of L = {L}")
    return int(n)


def stripe_k(LX, LY, family):
    """Integer momenta n (k = 2 pi n / L) of k_alpha = (pi -+ q, pi), (pi, pi -+ q)."""
    q = FAMILIES[family]
    nx, ny = _offset(LX, q), _offset(LY, q)
    hx, hy = LX // 2, LY // 2
    return ((hx - nx, hy), (hx + nx, hy), (hx, hy - ny), (hx, hy + ny))


def charge_q(LX, LY, family):
    """Integer momenta of the charge wavevectors +-(2q, 0), +-(0, 2q)."""
    q = FAMILIES[family]
    nx, ny = 2 * _offset(LX, q), 2 * _offset(LY, q)
    return ((nx, 0), (LX - nx, 0), (0, ny), (0, LY - ny))


def reference_sites(LX, LY, nref):
    """nref reference sites spread along the diagonal (x, x)."""
    return [(x, x % LY) for x in range(0, LX, max(LX // nref, 1))][:nref]


def displacements(LX, LY):
    """All displacements (N, 2), dr = (x, y) at index x * LY + y."""
    return np.array([(x, y) for x in range(LX) for y in range(LY)])


def min_image(dr, LX, LY):
    """Minimum-image length of integer displacements dr (..., 2)."""
    dr = np.asarray(dr)
    dx = np.minimum(dr[..., 0] % LX, (-dr[..., 0]) % LX)
    dy = np.minimum(dr[..., 1] % LY, (-dr[..., 1]) % LY)
    return np.hypot(dx, dy)


class SquareGeometry:
    """Sites, bonds and the d-wave operators of the periodic LX x LY lattice."""

    def __init__(self, LX, LY):
        self.LX, self.LY = LX, LY
        self.sites = [(x, y) for x in range(LX) for y in range(LY)]
        self.all_bonds = [(x, y, ax) for (x, y) in self.sites for ax in (0, 1)]

    def index(self, site):
        return int(site[0]) * self.LY + int(site[1])

    def wrap(self, site):
        return (site[0] % self.LX, site[1] % self.LY)

    def shift_site(self, site, t):
        return self.wrap((site[0] + t[0], site[1] + t[1]))

    def shift_bond(self, bond, t):
        return (*self.shift_site(bond, t), bond[2])

    def canon_bond(self, site, d):
        """Key of the bond from site along the unit vector d."""
        if d[0] + d[1] < 0:
            site = (site[0] + d[0], site[1] + d[1])
        return (*self.wrap(site), 0 if d[0] != 0 else 1)

    def bond_sites(self, bond):
        """(lower, upper) endpoints of a bond."""
        x, y, axis = bond
        return (x, y), self.wrap((x + 1, y) if axis == 0 else (x, y + 1))

    def string_sites(self, bond, o):
        """(site of c^dag_up, site of c^dag_dn) of A^dag(bond, o)."""
        lo, hi = self.bond_sites(bond)
        return (lo, hi) if o == 0 else (hi, lo)

    def pair_string(self, bond, o):
        """A^dag(bond, o) as a quantax operator."""
        from quantax.operator import create_d, create_u

        a, b = self.string_sites(bond, o)
        return create_u(*a) @ create_d(*b)

    def hop(self, i, j, spin):
        """c^dag_{i s} c_{j s} as a quantax operator (n_{i s} for i = j)."""
        from quantax.operator import annihilate_d, annihilate_u, create_d, create_u
        from quantax.operator import number_d, number_u

        if tuple(i) == tuple(j):
            return number_u(*i) if spin == "u" else number_d(*i)
        if spin == "u":
            return create_u(*i) @ annihilate_u(*j)
        return create_d(*i) @ annihilate_d(*j)


def dwave_keys(geo, R=(0, 0), primary_only=True):
    """(left_bonds, left_sites, F_keys, G_keys) of the d-wave measurement at R.

    The left bonds touch R (only the +x and +y ones with primary_only; assemble.py gets
    the others by translation), the right bonds and sites run over the whole lattice.
    """
    dirs = DIRS[::2] if primary_only else DIRS
    left_bonds = sorted({geo.canon_bond(R, d) for d in dirs})
    left_sites = sorted({s for b in left_bonds for s in geo.bond_sites(b)})
    F_keys = [
        (bl, oL, br, oR)
        for bl in left_bonds
        for oL in (0, 1)
        for br in geo.all_bonds
        for oR in (0, 1)
    ]
    G_keys = [(i, j, s) for i in left_sites for j in geo.sites for s in SPINS]
    return left_bonds, left_sites, F_keys, G_keys


def build_packed_dwave(LX, LY, R=(0, 0), primary_only=True):
    """dwave_packed.PackedDwave of the terms of dwave_keys."""
    from .dwave_packed import PackedDwave

    return PackedDwave(SquareGeometry(LX, LY), R, primary_only)


def F_keys_to_int(F_keys):
    """F keys ((bx, by, bax), oL, (cx, cy, cax), oR) -> int array (n, 8)."""
    return np.array([(*bl, oL, *br, oR) for bl, oL, br, oR in F_keys], dtype=np.int32)


def G_keys_to_int(G_keys):
    """G keys ((ix, iy), (jx, jy), 'u' | 'd') -> int array (n, 5), spin up 0, down 1."""
    return np.array([(*i, *j, SPINS.index(s)) for i, j, s in G_keys], dtype=np.int32)


def F_keys_from_int(keys):
    """Inverse of F_keys_to_int; tuple keys pass through."""
    if isinstance(keys[0], tuple):
        return list(keys)
    return [
        ((a, b, c), o1, (d, e, f), o2)
        for a, b, c, o1, d, e, f, o2 in np.asarray(keys).tolist()
    ]


def G_keys_from_int(keys):
    """Inverse of G_keys_to_int; tuple keys pass through."""
    if isinstance(keys[0], tuple):
        return list(keys)
    return [((a, b), (c, d), SPINS[s]) for a, b, c, d, s in np.asarray(keys).tolist()]


def sz_grid(spins, LX, LY):
    """S^z_i = (n_up - n_dn) / 2 on the (x, y) grid, shape (batch, LX, LY)."""
    import jax.numpy as jnp

    N = LX * LY
    dtype = jnp.result_type(float)
    sz = (spins[:, :N].astype(dtype) - spins[:, N:].astype(dtype)) / 4
    return sz.reshape(-1, LX, LY)


def dn_grid(spins, LX, LY, nbar):
    """Density fluctuation n_i - nbar on the (x, y) grid, shape (batch, LX, LY)."""
    import jax.numpy as jnp

    N = LX * LY
    dtype = jnp.result_type(float)
    n = (spins[:, :N].astype(dtype) + spins[:, N:].astype(dtype)) / 2 + 1
    return n.reshape(-1, LX, LY) - nbar


def autocorr(g):
    """Batch mean of (1/N) sum_i g_i g_{i+r} by FFT, shape (N,), r in site order."""
    import jax.numpy as jnp

    N = g.shape[1] * g.shape[2]
    Fk = jnp.fft.fft2(g, axes=(1, 2))
    A = jnp.real(jnp.fft.ifft2(jnp.abs(Fk) ** 2, axes=(1, 2)))
    return jnp.mean(A, axis=0).reshape(-1) / N
