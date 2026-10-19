"""Recompute the values of our rows of Table S4 from the files in this folder.

Measured rows (energy_series/): mean over sweeps, block-jackknife errors (largest over block sizes
1, 2, 4, 8 leaving >= 8 blocks). Training averages (training_traces/): mean over the stored steps.
Zero-variance extrapolation (../figS11_zero_variance_extrapolation/grid.csv): mean of the kept
window fits, error sqrt(stat^2 + window^2). Exits with 1 if a value differs by more than one unit
in its last printed digit (double rounding).
"""

import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def jackknife(cols, f, blocks=(1, 2, 4, 8)):
    """Largest block-jackknife error of f(column means) over the block sizes."""
    errs = []
    for b in blocks:
        n = len(cols[0]) // b
        if n >= 8:
            means = [a[: n * b].reshape(n, b).mean(1) for a in cols]
            loo = f([(m.sum() - m) / (n - 1) for m in means])  # leave-one-block-out estimates
            errs.append(np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2)))
    return max(errs)


def measured(name):
    """E/N, sigma^2/N and their errors from energy_series/<name>.npz."""
    d = np.load(os.path.join(HERE, "energy_series", f"{name}.npz"))
    em, es = d["E_mean"], d["E_sqr"]
    N = int(d["Lx"]) * int(d["Ly"])
    return dict(
        e=em.mean() / N,
        e_err=jackknife([em], lambda m: m[0] / N),
        v=(es.mean() - em.mean() ** 2) / N,
        v_err=jackknife([em, es], lambda m: (m[1] - m[0] ** 2) / N),
        n_sweeps=len(em),
    )


def training_average(path, N):
    _, E, V = np.loadtxt(path, delimiter=",", skiprows=1).T
    return dict(e=E.mean() / N, v=V.mean() / N, n_steps=len(E))


def zero_variance(path, N):
    """Grid mean of the kept window fits (totals, not per site) and E0/N with its error."""
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    kept = [r for r in rows if r["kept"] == "1"]
    b = np.array([float(r["E0"]) for r in kept])
    stat, window = np.mean([float(r["E0_err"]) for r in kept]), b.std(ddof=1)
    return dict(E0=b.mean(), stat=stat, window=window, e=b.mean() / N,
                e_err=np.hypot(stat, window) / N, n_kept=len(kept), n_windows=len(rows))


def parse(s):
    """'-0.75772(1)' -> (value, decimals, error digits or None)."""
    m = re.fullmatch(r"(-?\d+\.(\d+))(?:\((\d+)\))?", s)
    return float(m.group(1)), len(m.group(2)), (int(m.group(3)) if m.group(3) else None)


def compare(printed, value, err):
    """(recomputed value in the printed format, status)."""
    v, nd, e = parse(printed)
    new = round(value, nd)
    new_e = None if err is None else round(err * 10**nd)
    text = f"{value:.{nd}f}" + ("" if new_e is None else f"({new_e})")
    if new == v and (e is None or new_e == e):
        return text, "ok"
    close_e = e is None or new_e is None or abs(new_e - e) <= 1
    if abs(new - v) <= 10.0**-nd * (1 + 1e-9) and close_e:
        return text, "last digit (double rounding)"
    return text, "MISMATCH"


def main():
    with open(os.path.join(HERE, "tableS4.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["origin"] == "this work"]
    status = []
    for r in rows:
        N = int(r["Lx"]) * int(r["Ly"])
        label = f"t'={r['tprime']:>4} {r['lattice']:>5} {r['ansatz']:<24}"
        if not (r["data_unprojected"] or r["data_projected"]):
            print(f"{label} (no data in this folder)")
        for side in ("unprojected", "projected"):
            path = r[f"data_{side}"]
            if path.startswith("energy_series/"):
                m = measured(os.path.splitext(os.path.basename(path))[0])
                cells = [(r[f"E_{side}"], m["e"], m["e_err"]),
                         (r[f"var_{side}"], m["v"], m["v_err"])]
                src = f"{m['n_sweeps']} sweeps"
            elif path.startswith("training_traces/"):
                m = training_average(os.path.join(HERE, path), N)
                cells = [(r[f"E_{side}"], m["e"], None), (r[f"var_{side}"], m["v"], None)]
                src = f"mean of {m['n_steps']} steps"
            elif path:
                m = zero_variance(os.path.join(HERE, path), N)
                cells = [(r[f"E_{side}"], m["e"], m["e_err"])]
                src = (f"E0 = {m['E0']:.4f} +- {m['stat']:.4f} (stat) +- {m['window']:.4f} "
                       f"(window), {m['n_kept']}/{m['n_windows']} windows, N = {N}")
            else:
                continue
            for printed, value, err in cells:
                text, s = compare(printed, value, err)
                status.append(s)
                print(f"{label} {side:<11} printed {printed:<12} recomputed {text:<12} {s}  "
                      f"[{src}]")
    n_ok, n_bad = status.count("ok"), status.count("MISMATCH")
    print(f"{n_ok}/{len(status)} printed values reproduced, {len(status) - n_ok - n_bad} differ "
          f"in the last digit by double rounding, {n_bad} mismatches")
    return n_bad == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
