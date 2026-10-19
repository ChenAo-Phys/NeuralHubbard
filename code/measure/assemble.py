"""Assemble the paper's observables from the per-sweep series of measure_orderparams.py.

    cd code && python -m measure.assemble ../DIR/series.npz

prints a summary and writes results.json (scalars) and results.npz (maps) next to
series.npz. Each series entry is the sample mean of one sweep.

* Energy: E/N and sigma^2/N = (<E^2> - <E>^2) / N.
* Pairing, main text Eq. (2): C_p(r) = <Delta^dag_0 Delta_r> - N_{0,r}. With
  B^dag B = 1/2 sum_{oL, oR} A^dag(oL) A(oR) (see observables.py),

      <Delta^dag_0 Delta_r> = 1/16 sum_{eta, eta'} h_eta h_eta' 1/2 sum_{oL, oR} F,

  and N_{0,r} is the same sum with each F replaced by the Wick product of the normal
  contractions G.
* Eq. (3): Delta^2_SC is the mean of C_p(r) over the lattice vectors with minimum-image
  |r| >= d_max / 2, and Delta_SC its square root.
* Eq. (4): C_s(r) = C_zz(r) + (C_+-(r) + C_-+(r)) / 2, S(k) = sum_r e^{ik.r} C_s(r) and
  M^2_str = (1/4N) sum_alpha S(k_alpha).
* Supplement Eq. (S28): C_c(r) = <(n_0 - nbar)(n_r - nbar)> and its structure factor
  N(q).

Two-point functions are translation averages indexed by r = (x, y) at x * LY + y.

Errors: block jackknife over the sweeps with blocks of 1, 2, 4 and 8 sweeps, keeping
the block sizes that leave at least 8 blocks; the largest error is quoted. The per-r
errors of C_p are leave-one-sweep-out errors.
"""

import argparse
import json
import os

import numpy as np

from .observables import (
    DIRS,
    FORM_FACTOR,
    F_keys_from_int,
    G_keys_from_int,
    SquareGeometry,
    charge_q,
    displacements,
    min_image,
    stripe_k,
)

BLOCK_SIZES = (1, 2, 4, 8)
MIN_BLOCKS = 8


def blocked(x, b):
    """Means over blocks of b consecutive sweeps, dropping an incomplete last block."""
    nb = x.shape[0] // b
    return x[: nb * b].reshape(nb, b, *x.shape[1:]).mean(axis=1)


def jackknife(f, *series):
    """Block jackknife of f(*sweep means) over the leading axis of every series.

    Returns (value, error, errors): errors[b] is the error with blocks of b sweeps for
    every b in BLOCK_SIZES leaving at least MIN_BLOCKS blocks, error their elementwise
    maximum (NaN if none qualifies). f may return arrays.
    """
    x = [np.asarray(s, dtype=np.float64) for s in series]
    value = f(*[s.mean(axis=0) for s in x])
    errors = {}
    for b in BLOCK_SIZES:
        nb = x[0].shape[0] // b
        if nb < MIN_BLOCKS:
            continue
        xb = [blocked(s, b) for s in x]
        tot = [s.sum(axis=0) for s in xb]
        reps = np.array(
            [f(*[(t - s[i]) / (nb - 1) for t, s in zip(tot, xb)]) for i in range(nb)]
        )
        errors[b] = np.sqrt(nb - 1) * reps.std(axis=0)
    if errors:
        error = np.max(np.stack(list(errors.values())), axis=0)
    else:
        error = np.full(np.shape(value), np.nan)
    return value, error, errors


def _blocks(errors):
    return {int(b): float(x) for b, x in errors.items()}


def energy(E_mean, E_sqr, N):
    """E/N and sigma^2/N from the per-sweep means of E_loc and E_loc^2."""
    e, e_err, e_errs = jackknife(lambda m: m / N, E_mean)
    v, v_err, v_errs = jackknife(lambda m, q: (q - m**2) / N, E_mean, E_sqr)
    return dict(
        E_per_site=float(e),
        E_err=float(e_err),
        var_per_site=float(v),
        var_err=float(v_err),
        E_err_blocks=_blocks(e_errs),
        var_err_blocks=_blocks(v_errs),
        nsweeps=len(E_mean),
    )


class DwaveIndex:
    """Columns of the measured F and G terms entering C_p(r) for every displacement r.

    A term whose left bond (left site) was not measured is taken from the measured one
    of the same axis (from R) by translation,

        F[bl + t, oL, br + t, oR] = F[bl, oL, br, oR],  G[i + t, j + t, s] = G[i, j, s],

    which is exact for translation-invariant states.
    """

    def __init__(self, F_keys, G_keys, LX, LY, R=(0, 0)):
        geo = SquareGeometry(LX, LY)
        R = (int(R[0]), int(R[1]))
        F_keys, G_keys = F_keys_from_int(F_keys), G_keys_from_int(G_keys)
        fmap = {k: i for i, k in enumerate(F_keys)}
        gmap = {k: i for i, k in enumerate(G_keys)}
        left_bonds = sorted({k[0] for k in F_keys})
        self.translated = False  # any term taken by translation

        def fcol(bl, oL, br, oR):
            if (bl, oL, br, oR) in fmap:
                return fmap[bl, oL, br, oR]
            self.translated = True
            bp = next(b for b in left_bonds if b[2] == bl[2])
            t = (bp[0] - bl[0], bp[1] - bl[1])
            return fmap[bp, oL, geo.shift_bond(br, t), oR]

        def gcol(i, j, s):
            if (i, j, s) in gmap:
                return gmap[i, j, s]
            self.translated = True
            return gmap[R, geo.shift_site(j, (R[0] - i[0], R[1] - i[1])), s]

        N = LX * LY
        self.fidx = np.empty((N, 16, 4), dtype=np.int64)
        self.gidx = np.empty((N, 16, 4, 2), dtype=np.int64)
        self.ff = np.empty(16)
        for n, dr in enumerate(geo.sites):
            r2 = geo.wrap((R[0] + dr[0], R[1] + dr[1]))
            m = 0
            for dl in DIRS:
                bl = geo.canon_bond(R, dl)
                a, b = geo.bond_sites(bl)
                for d2 in DIRS:
                    br = geo.canon_bond(r2, d2)
                    c, d = geo.bond_sites(br)
                    self.ff[m] = FORM_FACTOR[dl] * FORM_FACTOR[d2]
                    self.fidx[n, m] = [
                        fcol(bl, oL, br, oR) for oL in (0, 1) for oR in (0, 1)
                    ]
                    # (oL, oR) = (0, 0), (0, 1), (1, 0), (1, 1)
                    self.gidx[n, m] = [
                        [gcol(a, c, "u"), gcol(b, d, "d")],
                        [gcol(a, d, "u"), gcol(b, c, "d")],
                        [gcol(b, c, "u"), gcol(a, d, "d")],
                        [gcol(b, d, "u"), gcol(a, c, "d")],
                    ]
                    m += 1
        self.dr = displacements(LX, LY)

    def pair_correlation(self, F, G=None, connected=True):
        """C_p(r), shape (..., N), from term means F (..., nF) and G (..., nG);
        <Delta^dag_0 Delta_r> without the Wick subtraction for connected=False."""
        F = np.asarray(F, dtype=np.float64)
        out = 0.5 * F[..., self.fidx].sum(axis=-1)
        if connected:
            G = np.asarray(G, dtype=np.float64)
            wick = G[..., self.gidx[..., 0]] * G[..., self.gidx[..., 1]]
            out = out - 0.5 * wick.sum(axis=-1)
        return (out * self.ff).sum(axis=-1) / 16


def far_mask(LX, LY):
    """Mask of the displacements with |r| >= r_max = d_max / 2 (Eq. (3)), and r_max."""
    dist = min_image(displacements(LX, LY), LX, LY)
    r_max = dist.max() / 2
    return dist >= r_max - 1e-9, float(r_max)


def dwave(F, F_keys, G, G_keys, LX, LY, R=(0, 0)):
    """C_p(r), Delta^2_SC and Delta_SC from per-sweep F (S, nF) and G (S, nG)."""
    F = np.asarray(F, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    idx = DwaveIndex(F_keys, G_keys, LX, LY, R)
    far, r_max = far_mask(LX, LY)
    n = F.shape[0]

    Cp = idx.pair_correlation(F.mean(axis=0), G.mean(axis=0))
    if n > 1:
        loo = idx.pair_correlation((F.sum(0) - F) / (n - 1), (G.sum(0) - G) / (n - 1))
        Cp_err = np.sqrt(n - 1) * loo.std(axis=0)
    else:
        Cp_err = np.full_like(Cp, np.nan)

    d2 = float(Cp[far].mean())
    d2_errs, d_errs = {}, {}
    for b in BLOCK_SIZES:
        nb = n // b
        if nb < MIN_BLOCKS:
            continue
        Fb, Gb = blocked(F, b), blocked(G, b)
        reps = idx.pair_correlation(
            (Fb.sum(0) - Fb) / (nb - 1), (Gb.sum(0) - Gb) / (nb - 1)
        )[:, far].mean(axis=-1)
        d2_errs[b] = float(np.sqrt(nb - 1) * reps.std())
        d_errs[b] = (
            float(np.sqrt(nb - 1) * np.sqrt(reps).std()) if reps.min() > 0 else np.nan
        )
    return dict(
        dr=idx.dr,
        C_p=Cp,
        C_p_err=Cp_err,
        Delta2_SC=d2,
        Delta2_SC_err=max(d2_errs.values(), default=np.nan),
        Delta_SC=float(np.sqrt(d2)) if d2 > 0 else np.nan,
        Delta_SC_err=max(d_errs.values(), default=np.nan),
        Delta2_SC_err_blocks=d2_errs,
        Delta_SC_err_blocks=d_errs,
        r_max=r_max,
        n_far=int(far.sum()),
        nsweeps=n,
        translation_fill=idx.translated,
    )


def spin_corr_series(Czz, Tpm, Tmp):
    """Per-sweep C_s(r) = C_zz(r) + (C_+-(r) + C_-+(r)) / 2, shape (S, N); Tpm and Tmp
    are (S, N) or (S, NREF, N), the reference sites are averaged."""
    Czz = np.asarray(Czz, dtype=np.float64)
    Tpm = np.asarray(Tpm, dtype=np.float64)
    Tmp = np.asarray(Tmp, dtype=np.float64)
    if Tpm.ndim == 3:
        Tpm, Tmp = Tpm.mean(axis=1), Tmp.mean(axis=1)
    return Czz + 0.5 * (Tpm + Tmp)


def structure_factor(C, LX, LY):
    """S(k) = sum_r e^{ik.r} C(r) on the momentum grid, shape (..., LX, LY)."""
    C = np.asarray(C, dtype=np.float64)
    S = np.fft.ifft2(C.reshape(*C.shape[:-1], LX, LY), axes=(-2, -1)) * (LX * LY)
    return S.real


def family_mean(Sk, ks, N):
    """(1/4N) sum_alpha S(k_alpha) over the four momenta ks of (..., LX, LY) maps."""
    LX, LY = Sk.shape[-2:]
    return np.mean([Sk[..., nx % LX, ny % LY] for nx, ny in ks], axis=0) / N


def spin(Czz, Tpm, Tmp, LX, LY, family):
    """C_s(r), S(k) and M^2_str; M2_zz is the zz part alone (M2_str / 3 for SU(2))."""
    Cs = spin_corr_series(Czz, Tpm, Tmp)
    N = LX * LY
    ks = stripe_k(LX, LY, family)
    m2 = family_mean(structure_factor(Cs, LX, LY), ks, N)
    m2zz = family_mean(structure_factor(Czz, LX, LY), ks, N)
    v, e, errs = jackknife(lambda x: x, m2)
    vz, ez, _ = jackknife(lambda x: x, m2zz)
    C, C_err, _ = jackknife(lambda x: x, Cs)
    return dict(
        C_s=C,
        C_s_err=C_err,
        S_k=structure_factor(C, LX, LY),
        k_alpha=np.array(ks),
        family=family,
        M2_str=float(v),
        M2_str_err=float(e),
        M2_str_err_blocks=_blocks(errs),
        M2_zz=float(vz),
        M2_zz_err=float(ez),
        nsweeps=len(Cs),
    )


def charge(Cnn, LX, LY, family):
    """C_c(r), N(q) and N2 = (1/4N) sum_alpha N(q_alpha) at the charge wavevectors."""
    qs = charge_q(LX, LY, family)
    C, C_err, _ = jackknife(lambda x: x, Cnn)
    n2 = family_mean(structure_factor(Cnn, LX, LY), qs, LX * LY)
    v, e, _ = jackknife(lambda x: x, n2)
    return dict(
        C_c=C,
        C_c_err=C_err,
        N_q=structure_factor(C, LX, LY),
        q_c=np.array(qs),
        N2=float(v),
        N2_err=float(e),
        nsweeps=len(Cnn),
    )


def load_series(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def assemble(series):
    """All observables available in a series dict (the layout of series.npz)."""
    LX, LY = int(series["LX"]), int(series["LY"])
    family = str(series["family"])
    res = dict(LX=LX, LY=LY, family=family)
    for k in ("state", "tprime", "nsamples"):
        if k in series:
            res[k] = np.asarray(series[k]).item()
    if "E_mean" in series:
        res["energy"] = energy(series["E_mean"], series["E_sqr"], LX * LY)
    if "Czz" in series and "Tpm" in series:
        res["spin"] = spin(series["Czz"], series["Tpm"], series["Tmp"], LX, LY, family)
    if "Cnn" in series:
        res["charge"] = charge(series["Cnn"], LX, LY, family)
    if "F" in series:
        R = series.get("R", (0, 0))
        F, G = series["F"], series["G"]
        res["dwave"] = dwave(F, series["F_keys"], G, series["G_keys"], LX, LY, R)
    if "sz_site" in series:
        sz = np.asarray(series["sz_site"], dtype=np.float64).mean(axis=0)
        n = np.asarray(series["n_site"], dtype=np.float64).mean(axis=0)
        res["local"] = dict(
            max_abs_sz=float(np.abs(sz).max()),
            n_min=float(n.min()),
            n_max=float(n.max()),
        )
    return res


def summary(res):
    """Human-readable summary of assemble results."""
    lines = [
        f"{res.get('state', '')} {res['LX']}x{res['LY']}, stripe family {res['family']}"
    ]
    if "energy" in res:
        e = res["energy"]
        lines.append(
            f"E/N = {e['E_per_site']:.6f} +- {e['E_err']:.1e}, "
            f"sigma^2/N = {e['var_per_site']:.5f} +- {e['var_err']:.1e} "
            f"({e['nsweeps']} sweeps)"
        )
    if "dwave" in res:
        d = res["dwave"]
        lines.append(
            f"Delta^2_SC = {d['Delta2_SC']:.6f} +- {d['Delta2_SC_err']:.6f}, "
            f"Delta_SC = {d['Delta_SC']:.6f} +- {d['Delta_SC_err']:.6f} "
            f"(M = {d['n_far']} vectors with |r| >= {d['r_max']:.4f}, "
            f"{d['nsweeps']} sweeps)"
        )
    if "spin" in res:
        s = res["spin"]
        lines.append(
            f"M^2_str = {s['M2_str']:.6f} +- {s['M2_str_err']:.6f} at k_alpha = "
            f"{s['k_alpha'].tolist()}; zz channel alone {s['M2_zz']:.6f}, "
            f"C_s(0) = {s['C_s'][0]:.5f}"
        )
    if "charge" in res:
        c = res["charge"]
        lines.append(
            f"N2(q_c) = {c['N2']:.6f} +- {c['N2_err']:.6f} at "
            f"q_c = {c['q_c'].tolist()}, sum_r C_c(r) = {c['C_c'].sum():.1e}"
        )
    if "local" in res:
        loc = res["local"]
        lines.append(
            f"max |<S^z_i>| = {loc['max_abs_sz']:.1e}, <n_i> in "
            f"[{loc['n_min']:.4f}, {loc['n_max']:.4f}]"
        )
    return "\n".join(lines)


def _json_value(v):
    """Plain JSON value; non-finite floats become null."""
    if isinstance(v, dict):
        return {str(k): _json_value(x) for k, x in v.items()}
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def save_results(res, out_dir):
    """results.json (scalars, NaN as null) and results.npz (arrays) in out_dir."""
    scalars, arrays = {}, {}
    for key, val in res.items():
        if isinstance(val, dict):
            scalars[key] = {}
            for k, v in val.items():
                if isinstance(v, np.ndarray) and v.ndim > 0:
                    arrays[k] = v
                else:
                    scalars[key][k] = _json_value(v)
        else:
            scalars[key] = _json_value(val)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(scalars, fh, indent=1, allow_nan=False)
    np.savez(os.path.join(out_dir, "results.npz"), **arrays)


def main(path):
    """Assemble series.npz at path, print the summary and save the results beside it."""
    res = assemble(load_series(path))
    print(summary(res), flush=True)
    save_results(res, os.path.dirname(os.path.abspath(path)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("series", help="series.npz written by measure_orderparams.py")
    main(p.parse_args().series)
