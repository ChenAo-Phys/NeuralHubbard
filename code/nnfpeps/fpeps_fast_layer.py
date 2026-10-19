"""Hand block-level HOTRG compression layer for the fPEPS D=4 hot path.

The stock `compress_y`/`compress_x` route every pairwise contraction through
symmray's generic flat machinery: because the sector table is traced data, it
emits per-sample dynamic sector-sort gathers, phase gathers and matricization
transposes over the (large) 6-leg pair intermediate C = t0 . t1 — 57% of the
HOTRG GPU time at 32x32.

At D = chi = 4 every tensor in the layer has the fixed Z2 structure
(8 sectors x (2,2,2,2) blocks) and the sector table takes only two static
values (even/odd parity), so the whole layer is planned statically: C is
built by ONE einsum directly in the grouping its consumers need; grams,
cholesky (chol8), truncated SVD (eigh8) and projectors are plain batched
GEMMs; the patch contraction reuses the left-grouped C. Parity dependence is
confined to small per-tensor gather indices and +-1 sign tables calibrated
slot-by-slot against symmray (the established methodology of the fast final
ring). The layer output is element-wise identical to the reference.

Slot conventions for C (y-merge: t0 upper, t1 lower, contract t0.d ~ t1.u):

- keep="r" (right gram):  slots (cg, m, f) with cg = r0+r1, m the (r0, r1)
  combo in symmray fuse order ([(0,0),(1,1)] for cg=0, [(0,1),(1,0)] for 1),
  f = 8 valid rest labels (l0, u0, l1, d1) nested as (co = l0+l1, mo = (l0,
  l1) combo index, us) where us orders the remaining (u0, d1) freedom.
  Block dims (..., cg 2, m 2, f 8, dl0, du0, dl1, dd1, dr0, dr1).
- keep="l" (left gram / patch): mirror with roles of (l, r) swapped:
  slots (cg=l0+l1, m=(l0,l1) combo, f=(cr, mr, us)), dims
  (..., cg, m, f, dr0, du0, dr1, dd1, dl0, dl1).

Legs (l, r, u, d) = positions (0, 1, 2, 3), duals (1, 0, 1, 0).
"""

from functools import lru_cache
import numpy as np
import jax
import jax.numpy as jnp

import fPEPS_hotrg_symm as F
from fpeps_pallas_linalg import chol8, eigh8

SEC = np.array(
    [
        [
            [0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0],
            [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1],
        ],
        [
            [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [0, 1, 1, 1],
            [1, 0, 0, 0], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0],
        ],
    ],
    dtype=np.int8,
)

_COMBOS = {0: [(0, 0), (1, 1)], 1: [(0, 1), (1, 0)]}


def _row_of(p, label):
    for i in range(8):
        if tuple(SEC[p][i]) == tuple(label):
            return i
    raise KeyError((p, label))


@lru_cache
def _plan_C(keep):
    """Static plan for the y-pair C in `keep` grouping (see module doc).

    Returns dict {(p0, p1): (iA (32,), iB (32,), labels (32, 6))} where slot
    order is (cg, m, f) flattened; labels are the C sector labels
    (l0, r0, u0, l1, r1, d1) of each slot (for calibration lookups).
    """
    kpos = 1 if keep == "r" else 0  # kept leg position on both tensors
    opos = 0 if keep == "r" else 1  # the other x-leg
    plans = {}
    for p0 in (0, 1):
        for p1 in (0, 1):
            iA, iB, labs = [], [], []
            for cg in (0, 1):
                for ka, kb in _COMBOS[cg]:
                    slots = []
                    for co in (0, 1):
                        for oi, (oa, ob) in enumerate(_COMBOS[co]):
                            # remaining freedom: u0 fixes t0's bond charge;
                            # d1 must match it
                            for u0 in (0, 1):
                                cb = (p0 - ka - oa - u0) % 2  # t0 bond (d)
                                d1 = (p1 - kb - ob - cb) % 2
                                labA = [0] * 4
                                labA[kpos], labA[opos] = ka, oa
                                labA[2], labA[3] = u0, cb
                                labB = [0] * 4
                                labB[kpos], labB[opos] = kb, ob
                                labB[2], labB[3] = cb, d1
                                slots.append(
                                    (_row_of(p0, labA), _row_of(p1, labB),
                                     labA, labB)
                                )
                    assert len(slots) == 8
                    for a, b, lA, lB in slots:
                        iA.append(a)
                        iB.append(b)
                        labs.append(
                            [lA[0], lA[1], lA[2], lB[0], lB[1], lB[3]]
                        )
            plans[(p0, p1)] = (np.array(iA), np.array(iB), np.array(labs))
    return plans


@lru_cache
def _plan_C_arrays(keep):
    plans = _plan_C(keep)
    iA = np.zeros((4, 32), np.int32)
    iB = np.zeros((4, 32), np.int32)
    for (p0, p1), (a, b, _) in plans.items():
        iA[p0 * 2 + p1], iB[p0 * 2 + p1] = a, b
    return iA, iB  # numpy: safe to cache across jit traces


def _gather_rows(t, idx):
    """t (..., 8, 2,2,2,2), idx (..., 32) -> (..., 32, 2,2,2,2)."""
    ix = idx.reshape(idx.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, idx.shape + t.shape[-4:])
    return jnp.take_along_axis(t, ix, axis=-5)


def build_C(t0, t1, pi, keep, sign_tab):
    """C of the y-pair, grouped per `keep`, in GEMM-friendly layout.

    t0/t1: (..., 8, 2,2,2,2) blocks; pi: (...,) int32 = p0*2 + p1;
    sign_tab: (4, 32) +-1 (parity-combo indexed), or None.
    Returns (..., c 2, m 2, kd0 2, kd1 2, f 8, g 16): slot axes (c, m, f)
    as before, but with the kept-leg dims (kd0, kd1) BEFORE f and the rest
    dims g = 16 flattened (a1, a2, b1, b2) AFTER, so both the gram GEMM
    (contract (f, g)) and the patch P1 GEMM (contract (kd0, kd1)) are
    layout-native with batch (c, m) leading and no materialized transposes.
      keep="r": kd = (dr0, dr1), g = (dl0, du0, dl1, dd1)
      keep="l": kd = (dl0, dl1), g = (dr0, du0, dr1, dd1)
    """
    iA, iB = _plan_C_arrays(keep)
    iA, iB = jnp.asarray(iA), jnp.asarray(iB)
    idxA, idxB = iA[pi], iB[pi]  # (..., 32)
    eA = _gather_rows(t0, idxA)
    eB = _gather_rows(t1, idxB)
    if sign_tab is not None:
        sg = sign_tab[pi]  # (..., 32)
        eA = eA * sg[..., None, None, None, None].astype(eA.dtype)
    # split the slot axis so f can sit between the kept and rest dims
    eA = eA.reshape(eA.shape[:-5] + (2, 2, 8) + eA.shape[-4:])
    eB = eB.reshape(eB.shape[:-5] + (2, 2, 8) + eB.shape[-4:])
    if keep == "r":
        # eA (c,m,f, l,r,u,M), eB (c,m,f, L,R,M,d) -> (c,m, r,R, f, l,u,L,d)
        C = jnp.einsum("...cmflruM,...cmfLRMd->...cmrRfluLd", eA, eB)
    else:
        # -> (c,m, l,L, f, r,u,R,d)
        C = jnp.einsum("...cmflruM,...cmfLRMd->...cmlLfruRd", eA, eB)
    return C.reshape(C.shape[:-9] + (2, 2, 2, 2, 8, 16))


# --------------------------------------------------------------------------
# calibrated +-1 tables (slot-by-slot against symmray). Rows indexed by
# pi = p0*2 + p1; the C
# tables depend only on p0 and are identical for both groupings.
_SC_EVEN = [1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1,
            1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, -1]
_SC_ODD = [-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1,
           1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1]
S_CR = jnp.asarray(np.array([_SC_EVEN, _SC_EVEN, _SC_ODD, _SC_ODD], np.float64))
S_CL = S_CR

# f-dependent Koszul signs of the gram sum, applied to BOTH copies on the
# m=1 combo slots (diagonal blocks unaffected, off-diagonal gets sigma_f);
# solved by lstsq against symmray, parity-independent (verified all 4 combos)
_SG_L = np.ones((2, 2, 8))
_SG_L[0, 1] = [-1, 1, 1, -1, -1, 1, 1, -1]
_SG_L[1, 1] = [1, -1, -1, 1, 1, -1, -1, 1]
_SG_R = np.ones((2, 2, 8))
_SG_R[0, 1] = [1, -1, -1, 1, -1, 1, 1, -1]
_SG_R[1, 1] = [1, -1, -1, 1, -1, 1, 1, -1]
SG = {"l": jnp.asarray(_SG_L), "r": jnp.asarray(_SG_R)}


def gram_R(C, keep, side, shift=1e-4):
    """Cholesky reduced factor of the pair gram over the kept legs.

    C: (..., 2, 2, 8, 2,2,2,2, 2, 2) from build_C (signed);
    side "right" -> upper factor (Rl convention), "left" -> lower (Rr).
    Returns (..., 2, 8, 8) blocks of the flat factor matrix.
    """
    Cm = C.reshape(C.shape[:-6] + (2, 2, 4, 128))  # (c, m, p, f*g)
    Cg = Cm * SG[keep][:, :, None, :].repeat(16, -1).reshape(2, 2, 1, 128).astype(
        Cm.dtype
    )
    XX = jnp.einsum("...cmpk,...cnqk->...cmpnq", Cg, Cg)
    XX = XX.reshape(XX.shape[:-5] + (2, 8, 8))
    # shift is applied PER SECTOR BLOCK (symmray blocklevel convention)
    tr = jnp.trace(XX, axis1=-2, axis2=-1)  # (..., 2)
    tr = jax.lax.stop_gradient(tr)
    XXs = XX + (shift * tr)[..., None, None] * jnp.eye(8, dtype=XX.dtype)
    Lo = chol8(XXs)
    if side == "right":
        return jnp.swapaxes(Lo.conj(), -1, -2)
    return Lo


# patch signs s(slot=(cl,cr,us), m, mr): identical for pi in {00, 01, 10},
# globally negated for {11}; not rank-1 factorizable, so the patch keeps the
# (m, mr) axes until this table contracts them (includes _merge_sign)
_SPATCH_BASE = np.array(
    [[1, 1, -1, 1], [1, -1, 1, 1], [-1, -1, 1, -1], [1, -1, 1, 1],
     [-1, 1, -1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [1, 1, -1, 1]],
    np.float64,
).reshape(8, 2, 2)
S_PATCH = jnp.asarray(
    np.stack([_SPATCH_BASE, _SPATCH_BASE, _SPATCH_BASE, -_SPATCH_BASE])
)


def patch_out(C_l, P1, P2, pi):
    """Patch contraction: out = P1 . (t0 . t1) . P2 in canonical order.

    C_l: left-grouped C (..., cl 2, m 2, f 8, dr0, du0, dr1, dd1, dl0, dl1)
    with f nested as (cr, mr, us);
    P1: (..., 2, chi_half 2, 8) (fused-l -> new l leg, from Pr of bond x-1);
    P2: (..., 2, 8, chi_half 2) (fused-r -> new r leg, from Pl of bond x).
    Returns (..., 8, 2, 2, 2, 2) canonical site blocks (l', r', u, d),
    element-wise equal to _contract2y (incl. its _merge_sign) up to the
    projector sign-gauge that cancels pairwise in psi.
    """
    nb = C_l.ndim - 6  # leading batch axes
    Cv = C_l.reshape(C_l.shape[:-6] + (2, 2, 4, 128))  # (c, m, p, k=(f g))
    P1v = P1.reshape(P1.shape[:-1] + (2, 4))  # (c, t, m, p)
    P2v = P2  # (..., C, 8, T): fused axis already (mr, r, R)
    # GEMM 1 (batch (c, m), contract p): layout-native
    D1 = jnp.einsum("...cmpk,...ctmp->...cmtk", Cv, P1v)
    # unpack k = (cr, mr, us | r, u, R, d), fold signs, and transpose ONCE to
    # put the GEMM-2 batch (cr) in front and its contraction (mr, r, R) last
    D1 = D1.reshape(D1.shape[:-1] + (2, 2, 2, 2, 2, 2, 2))
    # axes: (..., c, m, t, cr, mr, us, r, u, R, d)
    sg = S_PATCH[pi]  # (..., 8, 2, 2) per (slot=(c, cr, us), m, mr)
    sg = sg.reshape(sg.shape[:-3] + (2, 2, 2, 2, 2))  # (c, cr, us, m, mr)
    # reorder to (c, m, cr, mr, us) then broadcast into D1's axes
    sgb = jnp.transpose(
        sg, tuple(range(sg.ndim - 5)) + (sg.ndim - 5 + 0, sg.ndim - 5 + 3,
                                         sg.ndim - 5 + 1, sg.ndim - 5 + 4,
                                         sg.ndim - 5 + 2)
    )  # (..., c, m, cr, mr, us)
    sgb = sgb[..., :, :, None, :, :, :, None, None, None, None]
    perm5 = tuple(range(nb)) + tuple(
        nb + np.array([3, 0, 1, 2, 5, 7, 9, 4, 6, 8])
    )
    Dt = jnp.transpose(D1 * sgb.astype(D1.dtype), perm5)
    # axes now (..., cr, c, m, t, us, u, d, mr, r, R)
    Dt = Dt.reshape(Dt.shape[:nb] + (2, 64, 8))
    D2 = jnp.einsum("...Cxk,...CkT->...CxT", Dt, P2v)
    D2 = D2.reshape(D2.shape[:nb] + (2, 2, 2, 2, 2, 2, 2, 2))
    # axes (..., cr, c, m, t, us, u, d, T): sum m, reorder to slot layout
    D2 = D2.sum(nb + 2)
    out = jnp.transpose(
        D2, tuple(range(nb)) + tuple(nb + np.array([1, 0, 3, 2, 6, 4, 5]))
    )  # (..., c, cr, us, t, T, u, d)
    out = out.reshape(out.shape[:nb] + (8, 2, 2, 2, 2))  # (slot, t, T, u, d)
    perm = jnp.asarray(_out_perm_table())[pi]  # (..., 8) canonical row
    inv = jnp.argsort(perm, axis=-1)  # row -> slot
    ix = inv.reshape(inv.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, inv.shape + out.shape[-4:])
    return jnp.take_along_axis(out, ix, axis=-5)


@lru_cache
def _out_perm_table():
    """(4, 8) canonical SEC row of patch slot (cl, cr, us), per parity combo."""
    tab = np.zeros((4, 8), np.int32)
    for pA in (0, 1):
        for pB in (0, 1):
            pt = (pA + pB) % 2
            for cl in (0, 1):
                for cr in (0, 1):
                    for us in (0, 1):
                        d1 = (pt + cl + cr + us) % 2
                        tab[pA * 2 + pB, cl * 4 + cr * 2 + us] = _row_of(
                            pt, (cl, cr, us, d1)
                        )
    return tab  # numpy: safe to cache across jit traces


@lru_cache
def _lut(which):
    """Signed gather plans: row indices G (4, 32), slot signs S (4, 32) with
    ALL slot-level sign tables folded in, and the static dim permutation, so
    that each consumer einsum sees its contraction dims innermost with no
    operand transposes and no separate sign passes.

    which:
      "Ar_gram": slots (c,m,f), sign S_CR*SG_r, dims (r, M, l, u)   [A=t0]
      "Br_gram": slots (c,m,f), plain,          dims (R, M, L, d)   [B=t1]
      "Al_gram": slots (c,m,f), sign S_CL*SG_l, dims (l, M, r, u)
      "Bl_gram": slots (c,m,f), plain,          dims (L, M, R, d)
      "Al_patch": slots (c,m,f), sign S_CL*S_PATCH(slot), dims (r, u, M, l)
      "Bl_patch": slots (c,m,f), plain,         dims (R, d, L, M)
    Block dim orders: t0 (l, r, u, M); t1 (L, R, M, d).
    """
    keep = "r" if which[1] == "r" else "l"
    isA = which[0] == "A"
    plans = _plan_C(keep)
    # dim permutation: source order for A (l, r, u, M) / B (L, R, M, d)
    dimperm = {
        "Ar_gram": (1, 3, 0, 2),   # (r, M, l, u)
        "Br_gram": (1, 2, 0, 3),   # (R, M, L, d)
        "Al_gram": (0, 3, 1, 2),   # (l, M, r, u)
        "Bl_gram": (0, 2, 1, 3),   # (L, M, R, d)
        "Al_patch": (1, 2, 3, 0),  # (r, u, M, l)
        "Bl_patch": (1, 3, 0, 2),  # (R, d, L, M)
    }[which]
    G = np.zeros((4, 32), np.int32)
    S = np.ones((4, 32))
    for p0 in (0, 1):
        for p1 in (0, 1):
            piv = p0 * 2 + p1
            iA, iB, _ = plans[(p0, p1)]
            G[piv] = iA if isA else iB
            if isA:
                sc = np.asarray(S_CR if keep == "r" else S_CL)[piv].copy()
                if which.endswith("gram"):
                    sc = sc * np.asarray(SG[keep]).reshape(32)
                elif which == "Al_patch":
                    # S_PATCH(slot=(c, cr, us), m, mr) with slot bits
                    # k = (c, m, cr, mr, us)
                    sp = np.asarray(S_PATCH)[piv].reshape(2, 2, 2, 2, 2)
                    for k in range(32):
                        c, m = k >> 4, (k >> 3) & 1
                        cr, mr, us = (k >> 2) & 1, (k >> 1) & 1, k & 1
                        sc[k] *= sp[c, cr, us, m, mr]
                S[piv] = sc
    if which.endswith("gram"):
        # reorder slots (c, m, f) -> (c, f, m) so the gram einsums see the
        # batch axes (c, f) adjacent (clean batched GEMMs, no transposes)
        perm = np.array([(k >> 4) * 16 + (k & 7) * 2 + ((k >> 3) & 1)
                         for k in range(32)])
        inv = np.argsort(perm)
        G = G[:, inv]
        S = S[:, inv]
    return G, S, dimperm


def _lut_gather(t, pi, which):
    """t (..., 8, 2, 2, 2, 2) site blocks; returns (..., 32, 16): the row
    (sector-pair) gather with folded slot signs, on dim-permuted blocks."""
    G, S, dimperm = _lut(which)
    nb = t.ndim - 5
    tp = jnp.transpose(t, tuple(range(nb + 1)) + tuple(nb + 1 + np.array(dimperm)))
    tp = tp.reshape(tp.shape[: nb + 1] + (16,))
    idx = jnp.asarray(G)[pi]  # (..., 32)
    e = jnp.take_along_axis(tp, idx[..., None], axis=-2)
    return e * jnp.asarray(S, t.dtype)[pi][..., None]


def _gather_e(t0, t1, pi, keep, sign_tab):
    """Gathered pair operands eA, eB (..., c 2, m 2, f 8, 2,2,2,2), the sign
    applied to eA. Block dims: eA (l, r, u, M); eB (L, R, M, d)."""
    iA, iB = _plan_C_arrays(keep)
    iA, iB = jnp.asarray(iA), jnp.asarray(iB)
    eA = _gather_rows(t0, iA[pi])
    eB = _gather_rows(t1, iB[pi])
    if sign_tab is not None:
        sg = sign_tab[pi]  # (..., 32)
        eA = eA * sg[..., None, None, None, None].astype(eA.dtype)
    eA = eA.reshape(eA.shape[:-5] + (2, 2, 8) + eA.shape[-4:])
    eB = eB.reshape(eB.shape[:-5] + (2, 2, 8) + eB.shape[-4:])
    return eA, eB


def gram_R_e(eA, eB, keep, side, shift=1e-4):
    """Cholesky reduced factor WITHOUT materializing C: the rest-dim sum of
    C.C factorizes into cross-combo grams of eA and eB.

    eA must carry the C sign table; the gram Koszul signs SG are applied per
    copy here. Returns (..., 2, 8, 8)."""
    sg = SG[keep][:, :, :]  # (c 2, m 2, f 8)
    eAs = eA * sg[:, :, :, None, None, None, None].astype(eA.dtype)
    if keep == "r":
        # kept dims: eA's r, eB's R
        GA = jnp.einsum("...cmflpuM,...cnflquN->...cmnfpqMN", eAs, eAs)
        GB = jnp.einsum("...cmfLPMd,...cnfLQNd->...cmnfPQMN", eB, eB)
    else:
        # kept dims: eA's l, eB's L
        GA = jnp.einsum("...cmfpruM,...cnfqruN->...cmnfpqMN", eAs, eAs)
        GB = jnp.einsum("...cmfPRMd,...cnfQRNd->...cmnfPQMN", eB, eB)
    # XX[c, (m, p, P), (n, q, Q)] = sum_{f, M, N} GA * GB
    XX = jnp.einsum("...cmnfpqMN,...cmnfPQMN->...cmpPnqQ", GA, GB)
    XX = XX.reshape(XX.shape[:-7] + (2, 8, 8))
    # shift is applied PER SECTOR BLOCK (symmray blocklevel convention)
    tr = jnp.trace(XX, axis1=-2, axis2=-1)  # (..., 2)
    tr = jax.lax.stop_gradient(tr)
    XXs = XX + (shift * tr)[..., None, None] * jnp.eye(8, dtype=XX.dtype)
    Lo = chol8(XXs)
    if side == "right":
        return jnp.swapaxes(Lo.conj(), -1, -2)
    return Lo


def patch_out_e(eA, eB, P1, P2, pi):
    """Patch contraction without materializing C: a 3-operand chain that
    writes the (only) large intermediate directly in the layout the second
    projector GEMM consumes. eA/eB from _gather_e(keep="l")."""
    nb = eA.ndim - 7
    # split f = (cr, mr, us)
    eAf = eA.reshape(eA.shape[:-5] + (2, 2, 2) + eA.shape[-4:])
    eBf = eB.reshape(eB.shape[:-5] + (2, 2, 2) + eB.shape[-4:])
    P1v = P1.reshape(P1.shape[:-1] + (2, 2, 2))  # (c, t, m, dl0, dl1)
    # signs (c, cr, us, m, mr) folded into the small first contraction
    sg = S_PATCH[pi].reshape(S_PATCH[pi].shape[:-3] + (2, 2, 2, 2, 2))
    sgb = jnp.transpose(
        sg, tuple(range(sg.ndim - 5)) + (sg.ndim - 5, sg.ndim - 2,
                                         sg.ndim - 4, sg.ndim - 1,
                                         sg.ndim - 3)
    )  # (..., c, m, cr, mr, us)
    # tmp = sum_dl0 P1 . eA : (..., c, m, cr, mr, us, t, dl1, r, u, M)
    tmp = jnp.einsum("...cmCMslruK,...ctmlL->...cmCMstLruK", eAf, P1v)
    tmp = tmp * sgb[..., None, None, None, None, None].astype(tmp.dtype)
    # D1 = sum_{dl1, M} tmp . eB, written directly in GEMM-2 layout
    # eBf dims (c, m, cr, mr, us, L(dl1), R, K(M), d)
    D1 = jnp.einsum("...cmCMstLruK,...cmCMsLRKd->...CcmtsudMrR", tmp, eBf)
    # (..., cr, c, m, t, us, u, d, mr, r, R)
    D1 = D1.reshape(D1.shape[:nb] + (2, 64, 8))
    P2v = P2  # (..., C, 8=(mr, r, R), T)
    D2 = jnp.einsum("...Cxk,...CkT->...CxT", D1, P2v)
    D2 = D2.reshape(D2.shape[:nb] + (2, 2, 2, 2, 2, 2, 2, 2))
    # (..., cr, c, m, t, us, u, d, T): sum m, reorder to slots (c, cr, us)
    D2 = D2.sum(nb + 2)
    out = jnp.transpose(
        D2, tuple(range(nb)) + tuple(nb + np.array([1, 0, 3, 2, 6, 4, 5]))
    )  # (..., c, cr, us, t, T, u, d)
    out = out.reshape(out.shape[:nb] + (8, 2, 2, 2, 2))
    perm = jnp.asarray(_out_perm_table())[pi]
    inv = jnp.argsort(perm, axis=-1)
    ix = inv.reshape(inv.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, inv.shape + out.shape[-4:])
    return jnp.take_along_axis(out, ix, axis=-5)


def _gram_R_lut(A, B, side, shift=1e-4):
    """Cholesky reduced factor from LUT-layout operands.

    A: (..., 32, 16) -> (c, m, f, p, M, y 4); B: -> (c, m, f, P, M, z 4).
    All contraction dims innermost; no operand transposes."""
    # slots are (c, f, m)-ordered for gram gathers: batch (c, f) adjacent
    A = A.reshape(A.shape[:-2] + (2, 8, 2, 2, 2, 4))
    B = B.reshape(B.shape[:-2] + (2, 8, 2, 2, 2, 4))
    GA = jnp.einsum("...cfmpMy,...cfnqNy->...cmpnqfMN", A, A)
    GB = jnp.einsum("...cfmPMz,...cfnQNz->...cmPnQfMN", B, B)
    XX = jnp.einsum("...cmpnqfMN,...cmPnQfMN->...cmpPnqQ", GA, GB)
    XX = XX.reshape(XX.shape[:-7] + (2, 8, 8))
    tr = jnp.trace(XX, axis1=-2, axis2=-1)  # per sector block
    tr = jax.lax.stop_gradient(tr)
    XXs = XX + (shift * tr)[..., None, None] * jnp.eye(8, dtype=XX.dtype)
    Lo = chol8(XXs)
    if side == "right":
        return jnp.swapaxes(Lo.conj(), -1, -2)
    return Lo


# --------------------------------------------------------------------------
# v4: dense parity-embedded pipeline. Blocks are embedded into dense
# (4, 4, 4, 4) legs with (charge, sub) nesting; all fermionic phases become
# STATIC charge-position masks fused into GEMM epilogues; the only
# parity-dependent steps are the (small) embed/un-embed selections.


@lru_cache
def _emb_tables():
    """EMB (2, 16, 8): dense charge-combo row <- sector row, per parity."""
    E = np.zeros((2, 16, 8))
    for p in (0, 1):
        for row in range(8):
            cl, cr, cu, cd = SEC[p][row]
            E[p, cl * 8 + cr * 4 + cu * 2 + cd, row] = 1.0
    return E


def _embed_dense(t, p):
    """t (..., 8, 2,2,2,2), p (...,) parity -> dense (..., 4, 4, 4, 4)."""
    E = jnp.asarray(_emb_tables(), t.dtype)
    Es = E[p]  # (..., 16, 8)
    tf = t.reshape(t.shape[:-4] + (16,))
    d = jnp.einsum("...cr,...rk->...ck", Es, tf)  # (..., 16, 16)
    d = d.reshape(d.shape[:-2] + (2, 2, 2, 2, 2, 2, 2, 2))
    nb = d.ndim - 8
    # (cl, cr, cu, cd, il, ir, iu, id) -> (cl, il, cr, ir, cu, iu, cd, id)
    d = jnp.transpose(d, tuple(range(nb)) + tuple(nb + np.array(
        [0, 4, 1, 5, 2, 6, 3, 7])))
    return d.reshape(d.shape[:nb] + (4, 4, 4, 4))


def _mask(bits, table):
    """Static +-1 mask (4,)*bits from a charge-combo table (2,)*bits, with
    (charge, sub) nesting on every leg (charge-major repeat)."""
    m = np.asarray(table, np.float64).reshape((2,) * bits)
    for ax in range(bits):
        m = np.repeat(m, 2, axis=ax)
    return jnp.asarray(m)


# --------------------------------------------------------------------------
# fast final-grid line blobs: build each 4-site line transfer matrix via two
# planned pair contractions and one batched GEMM, replacing the symmray
# tensordot chain inside _line_blob_blocks (its gather/transpose bookkeeping
# dominates the final stage).


@lru_cache
def _plan_pair_bonds(enum_by):
    """Plan for D = tA .(m) tB of a y-pair for the line-blob GEMM.

    Slots (q 16, cb 2): q = (lA, rA, lB, rB) labels lexicographic; cb = the
    charge of the pairing bond: enum_by "d" -> tB's outer d bond (used by
    D01, pairing axis cd1); "u" -> tA's outer u bond (D23, axis cu2).
    Returns {(pA, pB): (iA (32,), iB (32,), labels (32, 6))}, slot k = q*2+cb.
    """
    plans = {}
    for pA in (0, 1):
        for pB in (0, 1):
            iA, iB, labs = [], [], []
            for lA in (0, 1):
                for rA in (0, 1):
                    for lB in (0, 1):
                        for rB in (0, 1):
                            for cb in (0, 1):
                                if enum_by == "d":
                                    cm = (pB - lB - rB - cb) % 2
                                    cu = (pA - lA - rA - cm) % 2
                                    cd = cb
                                else:
                                    cu = cb
                                    cm = (pA - lA - rA - cu) % 2
                                    cd = (pB - lB - rB - cm) % 2
                                labA = (lA, rA, cu, cm)
                                labB = (lB, rB, cm, cd)
                                iA.append(_row_of(pA, labA))
                                iB.append(_row_of(pB, labB))
                                labs.append([lA, rA, cu, lB, rB, cd])
            plans[(pA, pB)] = (np.array(iA), np.array(iB), np.array(labs))
    return plans


@lru_cache
def _pair_bond_arrays(enum_by):
    plans = _plan_pair_bonds(enum_by)
    iA = np.zeros((4, 32), np.int32)
    iB = np.zeros((4, 32), np.int32)
    for (pA, pB), (a, b, _) in plans.items():
        iA[pA * 2 + pB], iB[pA * 2 + pB] = a, b
    return iA, iB


def build_D(t0, t1, pi, enum_by, sign_vec=None):
    """Planned pair contraction for line blobs, in GEMM-native layout.

    t0/t1 (..., 8, 2,2,2,2); sign_vec optional (..., 32) slot signs applied
    on the small gathered operand (e.g. the P4-indexed blob tables).
    Returns (..., q 16, ld 4, rd 4, cb 2, ud 2, dd 2): free dims (q, ld, rd)
    leading and the blob-GEMM contraction dims (cb, ud, dd) trailing.
    """
    iA, iB = _pair_bond_arrays(enum_by)
    idxA = jnp.asarray(iA)[pi]
    idxB = jnp.asarray(iB)[pi]
    eA = _gather_rows(t0, idxA)
    eB = _gather_rows(t1, idxB)
    if sign_vec is not None:
        eA = eA * sign_vec[..., None, None, None, None].astype(eA.dtype)
    # slot k = (q, cb) pre-split so the einsum writes the target layout
    eA = eA.reshape(eA.shape[:-5] + (16, 2) + eA.shape[-4:])
    eB = eB.reshape(eB.shape[:-5] + (16, 2) + eB.shape[-4:])
    D = jnp.einsum("...qclruM,...qcLRMd->...qlLrRcud", eA, eB)
    return D.reshape(D.shape[:-8] + (16, 4, 4, 2, 2, 2))


@lru_cache
def _blob_tables():
    """Calibrated line-blob tables (fpeps_blob_tables.npz, matched against
    symmray): per-P4 pair sign tables and the empirically matched extraction
    map from the planned line-pair GEMM output to the symmray-fused
    (2, 128, 128) blob convention (signs included). Guarded by the
    final-contraction import self-test in fPEPS_hotrg_symm."""
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fpeps_blob_tables.npz")
    z = np.load(path)
    return (z["S_D_P4_d"].astype(np.float64), z["S_D_P4_u"].astype(np.float64),
            z["EXT_IDX"].astype(np.int32), z["EXT_SGN"].astype(np.float64))


def fast_line_blobs(line_params, line_charge):
    """Line transfer matrices for the final 4-site-line grid via planned
    pair contractions + one batched GEMM + calibrated extraction; equals
    fPEPS_hotrg_symm._line_blob_blocks element-wise (f64 verified).

    line_params: (nline, 4, 8, 2,2,2,2); line_charge: (nline, 4, sx, sy).
    Returns (nline, 2, 128, 128).
    """
    TD, TU, EIDX, ESGN = _blob_tables()
    p = jnp.sum(line_charge.astype(jnp.int32), axis=(-2, -1)) % 2  # (nl, 4)
    P4 = p[:, 0] * 8 + p[:, 1] * 4 + p[:, 2] * 2 + p[:, 3]
    pi01 = p[:, 0] * 2 + p[:, 1]
    pi23 = p[:, 2] * 2 + p[:, 3]
    t = line_params
    sD = jnp.asarray(TD, t.dtype)[P4]  # (..., 32) slot signs
    sU = jnp.asarray(TU, t.dtype)[P4]
    D01 = build_D(t[:, 0], t[:, 1], pi01, "d", sD)
    D23 = build_D(t[:, 2], t[:, 3], pi23, "u", sU)
    # D layout (q, ld 4, rd 4 | cb 2, ud 2, dd 2): free dims leading, blob
    # contraction (cb, u<->d cross-paired) trailing -> clean batched GEMM
    pre = jnp.einsum("...qabcuv,...QABcvu->...qabQAB", D01, D23)
    pre = pre.reshape(pre.shape[:-6] + (16 * 16 * 256,))
    idx = jnp.asarray(EIDX)[P4]  # (nl, 32768)
    sgn = jnp.asarray(ESGN, t.dtype)[P4]
    blobs = jnp.take_along_axis(pre, idx, axis=-1) * sgn
    return blobs.reshape(blobs.shape[:-1] + (2, 128, 128))


@lru_cache
def _blob_tables_factored():
    """Exact factorization of the EXT tables (verified bitwise): the
    extraction row/col indices split into a charge-block map
    (p4, sector, Uc, Dc) -> (q, Q) with the sub-indices passing through
    untouched (row subs = (Us//4, Ds//4), col subs = (Us%4, Ds%4)), and the
    sign is charge-only. So the blob is 128 charge blocks of 16x16, each
    ``sum_K D01[q, a, b, K] * D23[Q, A, B, K]`` — buildable directly in its
    (2, 128, 128) layout with no 65536-float pre, no extraction gather and
    no ring-canon pass."""
    _, _, EIDX, ESGN = _blob_tables()
    idx = EIDX.reshape(16, 2, 8, 16, 8, 16)  # (p4, k, Uc, Us, Dc, Ds)
    r, c = idx // 256, idx % 256
    qt = (r // 16)[:, :, :, 0, :, 0].astype(np.int32)  # (16, 2, 8, 8)
    Qt = (c // 16)[:, :, :, 0, :, 0].astype(np.int32)
    st = ESGN.reshape(16, 2, 8, 16, 8, 16)[:, :, :, 0, :, 0].astype(np.float64)
    return qt, Qt, st


def _blob_pair_operands(line_params, line_charge):
    """Planned pair operands of the line blobs, leading dims arbitrary:
    D01r (..., 256, 8) rows (q*16 + a*4 + b), D23s (..., 256, 8) rows
    (Q*16 + A*4 + B) with the K axes pre-swapped to D01's (cb, ud, dd)
    order, and the line parity index P4 (...,). Shared by the einsum path
    below and the tensor-core kernel (fpeps_layer_kernels.blob_build) —
    both call the same build_D, so their outputs stay bitwise-comparable."""
    TD, TU, _, _ = _blob_tables()
    pch = jnp.sum(line_charge.astype(jnp.int32), axis=(-2, -1)) % 2
    P4 = pch[..., 0] * 8 + pch[..., 1] * 4 + pch[..., 2] * 2 + pch[..., 3]
    pi01 = pch[..., 0] * 2 + pch[..., 1]
    pi23 = pch[..., 2] * 2 + pch[..., 3]
    t = line_params
    sD = jnp.asarray(TD, t.dtype)[P4]
    sU = jnp.asarray(TU, t.dtype)[P4]
    nd = t.ndim
    cells = [jnp.take(t, j, axis=nd - 6) for j in range(4)]
    D01 = build_D(cells[0], cells[1], pi01, "d", sD)
    D23 = build_D(cells[2], cells[3], pi23, "u", sU)
    sh = D01.shape[:-6]
    D01r = D01.reshape(sh + (256, 8))
    # pre paired D01's K = (cb, ud, dd) with D23's (cb, dd, ud): swap
    D23s = jnp.swapaxes(D23, -1, -2).reshape(sh + (256, 8))
    return D01r, D23s, P4


def fast_line_blobs_canon_xla(line_params, line_charge, csel, csgn):
    """XLA twin of `fast_line_blobs_canon` for below the fused-kernel line
    threshold: the same planned pair contractions as `fast_line_blobs`, but
    the blob is assembled per charge block via the factored EXT tables —
    the 65536-float pre is never materialized, and the extraction and the
    ring block-select/sign of `_fast_raw_final_contract` are folded into
    the block gather (they were two more full passes over blob-sized data).
    Bit-exact vs fast_line_blobs + canon (same contractions, exact +-1s).

    csel/csgn: (nline, 2). Returns (nline, 2, 128, 128) ring-ready blobs."""
    TD, TU, _, _ = _blob_tables()
    qt, Qt, st = _blob_tables_factored()
    p = jnp.sum(line_charge.astype(jnp.int32), axis=(-2, -1)) % 2  # (nl, 4)
    P4 = p[:, 0] * 8 + p[:, 1] * 4 + p[:, 2] * 2 + p[:, 3]
    pi01 = p[:, 0] * 2 + p[:, 1]
    pi23 = p[:, 2] * 2 + p[:, 3]
    t = line_params
    sD = jnp.asarray(TD, t.dtype)[P4]
    sU = jnp.asarray(TU, t.dtype)[P4]
    D01 = build_D(t[:, 0], t[:, 1], pi01, "d", sD)
    D23 = build_D(t[:, 2], t[:, 3], pi23, "u", sU)
    nl = line_charge.shape[0]
    D01 = D01.reshape(nl, 16, 4, 4, 8)
    # pre paired D01's K = (cb, ud, dd) with D23's (cb, dd, ud): swap
    D23 = jnp.swapaxes(D23, -1, -2).reshape(nl, 16, 4, 4, 8)
    cs = csel.astype(jnp.int32)
    qsel = jnp.asarray(qt)[P4[:, None], cs]  # (nl, 2, 8, 8)
    Qsel = jnp.asarray(Qt)[P4[:, None], cs]
    ssel = jnp.asarray(st, t.dtype)[P4[:, None], cs]
    li = jnp.arange(nl)[:, None, None, None]
    G01 = D01[li, qsel]  # (l, k, Uc, Dc, a, b, K)
    G23 = D23[li, Qsel]  # (l, k, Uc, Dc, A, B, K)
    blobs = jnp.einsum("lkUDabK,lkUDABK->lkUaADbB", G01, G23)
    sg = ssel[:, :, :, None, None, :, None, None] * csgn[
        :, :, None, None, None, None, None, None
    ].astype(blobs.dtype)
    return (blobs * sg).reshape(nl, 2, 128, 128)


def fast_line_blobs_canon(line_params, line_charge, csel, csgn):
    """Canon-fused line blobs: pre-GEMM + calibrated extraction + the ring
    canon block-select/sign of _fast_raw_final_contract in ONE Pallas pass
    (fpeps_layer_kernels.blob_canon) — no 65536-float pre materialization,
    no separate canon pass. Element-wise equal to
    take_along_axis(fast_line_blobs(...), csel) * csgn.

    line_params: (nline, 4, 8, 2,2,2,2); line_charge: (nline, 4, sx, sy);
    csel/csgn: (nline, 2). Returns (nline, 2, 128, 128) ring-ready blobs."""
    import fpeps_layer_kernels as FK

    TD, TU, _, _ = _blob_tables()
    p = jnp.sum(line_charge.astype(jnp.int32), axis=(-2, -1)) % 2  # (nl, 4)
    P4 = p[:, 0] * 8 + p[:, 1] * 4 + p[:, 2] * 2 + p[:, 3]
    pi01 = p[:, 0] * 2 + p[:, 1]
    pi23 = p[:, 2] * 2 + p[:, 3]
    t = line_params
    sD = jnp.asarray(TD, t.dtype)[P4]
    sU = jnp.asarray(TU, t.dtype)[P4]
    D01 = build_D(t[:, 0], t[:, 1], pi01, "d", sD)
    D23 = build_D(t[:, 2], t[:, 3], pi23, "u", sU)
    # K = (cb, ud, dd) for D01; the pre einsum binds D23's (cb, ud, dd) as
    # (c, v, u), so swap its last two axes to share the K layout
    D01k = D01.reshape(D01.shape[:-6] + (256, 8))
    D23k = D23.reshape(D23.shape[:-6] + (256, 2, 2, 2))
    D23k = jnp.swapaxes(D23k, -1, -2).reshape(D23k.shape[:-3] + (8,))
    return FK.blob_canon(D01k, D23k, P4, csel, csgn)


def fast_compress_y_k(params, charge, chi, return_internal=False):
    """Drop-in for compress_y using the fused Pallas layer kernels
    (fpeps_layer_kernels): merged gram+cholesky, projector assembly and the
    patch run one memory pass per unit; only eigh8 sits between. All kernel
    ops are differentiable (jnp-reference vjp) and vmap-collapsing."""
    import fpeps_layer_kernels as FK

    assert chi == 4 and not return_internal
    p = jnp.sum(charge.astype(jnp.int32), axis=(-2, -1)) % 2
    pi = p[..., 0] * 2 + p[..., 1]
    t0 = params[:, :, 0]
    t1 = params[:, :, 1]

    Rl, Rr = FK.gram_chol(t0, t1, pi)
    Rr_next = jnp.roll(Rr, -1, axis=0)
    M, G = FK.proj_pre(Rl, Rr_next)
    w, U = F._eigh_dispatch(G)
    U2 = U[..., :, -(chi // 2):]  # top block, ascending (svd:eig convention)
    w2 = w[..., -(chi // 2):]
    Pl, Pr = FK.proj_post(M, U2, w2, Rl, Rr_next)
    P1 = jnp.roll(Pr, 1, axis=0)
    P2 = Pl
    out = FK.patch_slots(t0, t1, P1, P2, pi)
    out = out.reshape(out.shape[:-1] + (2, 2, 2, 2))
    perm = jnp.asarray(_out_perm_table())[pi]
    inv = jnp.argsort(perm, axis=-1)
    ix = inv.reshape(inv.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, inv.shape + out.shape[-4:])
    return jnp.take_along_axis(out, ix, axis=-5)


# --------------------------------------------------------------------------
# x-merge layers via the transposed frame: an x-merge is exactly the y-merge
# of the leg-transposed tensors on the transposed lattice, so ALL y-merge
# machinery (slot tables, Pallas kernels, projector recipe) is reused
# unchanged. The per-tensor fermionic leg transpose (l, r, u, d) ->
# (u, d, l, r) is a STATIC per-parity map at D = 4: an 8-row sector
# permutation + the Koszul phase (-1)^{(cl+cr)(cu+cd)} + a block-dim
# permute — probed element-wise against symmray
# transpose((2,3,0,1)) + sort_stack + phase_sync (self-inverse).
# Correctness of the reduction (delta == 1, no residual sign) verified at
# psi level vs the stock compress_x in f64 (and at import by
# fPEPS_hotrg_symm._layerx_selftest).

_TP_PERM = np.array([[0, 6, 2, 4, 3, 5, 1, 7],
                     [2, 4, 0, 6, 1, 7, 3, 5]], np.int32)
_TP_SGN = np.array([[1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])


def transpose_blocks(t, p):
    """Fermionic leg transpose (l, r, u, d) -> (u, d, l, r) of canonical
    site blocks t (..., 8, dl, dr, du, dd), parity p (...,) int."""
    perm = jnp.asarray(_TP_PERM)[p]
    sgn = jnp.asarray(_TP_SGN, t.dtype)[p]
    ix = perm.reshape(perm.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, perm.shape + t.shape[-4:])
    g = jnp.take_along_axis(t, ix, axis=-5)
    g = g * sgn.reshape(sgn.shape + (1,) * 4)
    nb = g.ndim - 4
    return jnp.transpose(g, tuple(range(nb)) + (nb + 2, nb + 3, nb, nb + 1))


def _compress_x_frame(params, charge):
    """Transposed-frame operands of an x-merge: params (lx2, 2, ly, 8,
    2,2,2,2), charge (lx2, 2, ly, sx, sy) -> (tT, cT) laid out for
    compress_y ((ly, lx2, 2, ...): ring axis = the original y)."""
    p = jnp.sum(charge.astype(jnp.int32), axis=(-2, -1)) % 2
    tT = transpose_blocks(params, p)
    tT = jnp.moveaxis(tT, 2, 0)
    cT = jnp.swapaxes(jnp.moveaxis(charge, 2, 0), -1, -2)
    return tT, cT


def fast_compress_x_k(params, charge, chi, return_internal=False):
    """Drop-in for compress_x at D = chi = 4 via the transposed-frame
    fused y-merge kernels: params (lx2, 2, ly, 8, 2,2,2,2), charge
    (lx2, 2, ly, sx, sy) -> (lx2, ly, 8, 2,2,2,2).

    Valid for min(sx, sy) == 1 cells (any pure-x / pure-y history; the
    dispatcher guards on it): there the transposed y-merge needs no
    interleave crossing sign and the mode-order conventions of the two
    frames coincide, so no residual sign is needed (delta == 1, verified
    at psi level in f64)."""
    assert chi == 4 and not return_internal
    tT, cT = _compress_x_frame(params, charge)
    outT = fast_compress_y_k(tT, cT, chi)
    pout = jnp.sum(
        cT.astype(jnp.int32).reshape(cT.shape[:2] + (-1,)), axis=-1
    ) % 2  # (ly, lx2)
    out = transpose_blocks(outT, pout)
    return jnp.swapaxes(out, 0, 1)


def fast_compress_x(params, charge, chi, return_internal=False):
    """jnp twin of fast_compress_x_k (transposed-frame fast_compress_y);
    the kernel-free reference of the same reduction."""
    assert chi == 4 and not return_internal
    tT, cT = _compress_x_frame(params, charge)
    outT = fast_compress_y(tT, cT, chi)
    pout = jnp.sum(
        cT.astype(jnp.int32).reshape(cT.shape[:2] + (-1,)), axis=-1
    ) % 2
    out = transpose_blocks(outT, pout)
    return jnp.swapaxes(out, 0, 1)


def fast_compress_y(params, charge, chi, return_internal=False):
    """Drop-in for fPEPS_hotrg_symm.compress_y at D = chi = 4.

    params: (lx, nk, 2, 8, 2,2,2,2); charge: (lx, nk, 2, subx, suby).
    Returns (lx, nk, 8, 2,2,2,2); psi is element-wise equal to the reference
    (per-tensor blocks differ only by the projector sign gauge, which
    cancels pairwise across bonds).
    """
    assert chi == 4 and not return_internal
    p = jnp.sum(charge.astype(jnp.int32), axis=(-2, -1)) % 2  # (lx, nk, 2)
    pi = p[..., 0] * 2 + p[..., 1]
    t0 = params[:, :, 0]
    t1 = params[:, :, 1]
    Rl = _gram_R_lut(_lut_gather(t0, pi, "Ar_gram"),
                     _lut_gather(t1, pi, "Br_gram"), "right")
    Rr = _gram_R_lut(_lut_gather(t0, pi, "Al_gram"),
                     _lut_gather(t1, pi, "Bl_gram"), "left")
    Rr_next = jnp.roll(Rr, -1, axis=0)
    Pl, Pr = projectors(Rl, Rr_next, chi // 2, None)
    P1 = jnp.roll(Pr, 1, axis=0)  # projector of bond (x-1, x)
    P2 = Pl

    nb = pi.ndim
    Ap = _lut_gather(t0, pi, "Al_patch")  # (c,m,C,K,s, r,u,M,l)
    Bp = _lut_gather(t1, pi, "Bl_patch")  # (c,m,C,K,s, R,d,L,M)
    Ap = Ap.reshape(Ap.shape[:-2] + (2,) * 9)
    Bp = Bp.reshape(Bp.shape[:-2] + (2,) * 9)
    P1v = P1.reshape(P1.shape[:-1] + (2, 2, 2))  # (c, t, m, l, L)
    tmp = jnp.einsum("...cmCKsruMl,...ctmlL->...cmCKstruLM", Ap, P1v)
    D1 = jnp.einsum("...cmCKstruLM,...cmCKsRdLM->...CcmtsudKrR", tmp, Bp)
    D1 = D1.reshape(D1.shape[:nb] + (2, 64, 8))
    D2 = jnp.einsum("...Cxk,...CkT->...CxT", D1, P2)
    D2 = D2.reshape(D2.shape[:nb] + (2,) * 8)
    # (..., cr, c, m, t, us, u, d, T): sum m, reorder to slots (c, cr, us)
    D2 = D2.sum(nb + 2)
    out = jnp.transpose(
        D2, tuple(range(nb)) + tuple(nb + np.array([1, 0, 3, 2, 6, 4, 5]))
    )  # (..., c, cr, us, t, T, u, d)
    out = out.reshape(out.shape[:nb] + (8, 2, 2, 2, 2))
    perm = jnp.asarray(_out_perm_table())[pi]
    inv = jnp.argsort(perm, axis=-1)
    ix = inv.reshape(inv.shape + (1,) * 4)
    ix = jnp.broadcast_to(ix, inv.shape + out.shape[-4:])
    return jnp.take_along_axis(out, ix, axis=-5)


def projectors(Rl, Rr_next, chi_half=2, bond_sign=None):
    """Oblique projectors of one bond from the two cholesky factors.

    Bit-exact mirror (up to block-level sign equivalence, which propagates
    sign-equivariantly through eigh8 and cancels pairwise in psi) of
    quimb compute_oblique_projectors(method="svd:eig", absorb="both") with
    symmray's blocklevel_svd_via_eig: right=False branch (eigh of M M^T),
    ascending truncation, sinv = s / (s^2 + cutoff^2) regularization, and
    the same division order. Uses the same eigh dispatch as the reference.

    Rl, Rr_next: (..., 2, 8, 8).
    Returns Pl (..., 2, 8, chi_half), Pr (..., 2, chi_half, 8).
    """
    M = jnp.matmul(Rl, Rr_next)
    if bond_sign is not None:  # per-charge phase of the contracted bond
        M = M * bond_sign[..., :, None, None].astype(M.dtype)
    G = jnp.matmul(M, jnp.swapaxes(M, -1, -2))  # x @ xdag (right=False)
    w, U = F._eigh_dispatch(G)  # ascending, same kernel as the reference
    s2 = w[..., -chi_half:]  # keep top block, ASCENDING (descending=False)
    U2 = U[..., :, -chi_half:]  # (..., 2, 8, chi_half)
    s2 = jnp.clip(s2, 0.0, None)
    sVH = jnp.matmul(jnp.swapaxes(U2, -1, -2), M)  # (..., 2, chi_half, 8)
    s = jnp.sqrt(s2)
    eps = jnp.finfo(s.dtype).eps
    smax = s[..., -1:]  # per block (ascending); quimb svd_via_eig convention
    cutoff = smax * eps * 8
    sinv = s / (s**2 + cutoff**2)  # quimb safe_inverse
    VH = sinv[..., :, None] * sVH
    st_sqrt = jnp.sqrt(s)
    Vd = jnp.swapaxes(VH, -1, -2) / st_sqrt[..., None, :]
    Pl = jnp.matmul(Rr_next, Vd)
    Ud = jnp.swapaxes(U2, -1, -2) / st_sqrt[..., :, None]
    Pr = jnp.matmul(Ud, Rl)
    return Pl, Pr
