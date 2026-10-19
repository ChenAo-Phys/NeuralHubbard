"""Regenerate Fig. S11 (zero-variance extrapolation on 16x4) from data/ and print E0/N.

ODR fits of E = k sigma^2 + E0 to the binned training curve over a grid of windows (odrpack).
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from odrpack import odr_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "figS11_zero_variance_extrapolation"
N = 64
STARTS, ENDS, BINS = [4000, 4500, 5000, 5500, 6000], [8000, 9000, 10001], [25, 50, 100]
REP = (4000, 10001, 50)  # window of panel (a)
SD_REL_MAX = 5.0  # drop fits whose intercept error exceeds 5x the median


def read_csv(path):
    """Columns of a CSV file with '#' comment lines, as float arrays."""
    with open(path) as f:
        names, *rows = csv.reader(line for line in f if not line.startswith("#"))
    return dict(zip(names, np.array(rows, dtype=float).T))


def window(curve, start, end):
    sel = (curve["step"] >= start) & (curve["step"] < end)
    return {key: v[sel] for key, v in curve.items()}


def bin_series(series, size):
    """Means of consecutive bins with errors std / sqrt(size); an incomplete last bin is dropped."""
    binned = series[: series.size // size * size].reshape(-1, size)
    return binned.mean(axis=1), binned.std(axis=1) / np.sqrt(size)


def lever_arm(v):
    """median(v) / (max - min of the medians of 4 consecutive chunks of v)."""
    return np.median(v) / np.ptp([np.median(c) for c in np.array_split(v, 4)])


def extrap(curve, start, end, size):
    """ODR fit of E = k VarE + E0 to the binned window -> (result, (x, y, x_err, y_err, step))."""
    d = window(curve, start, end)
    y, y_err = bin_series(d["E"], size)
    x, x_err = bin_series(d["VarE"], size)
    step, _ = bin_series(d["step"], size)
    res = odr_fit(lambda x, beta: beta[0] * x + beta[1], x, y, beta0=np.polyfit(x, y, 1),
                  weight_x=1 / x_err**2, weight_y=1 / y_err**2, maxit=300)
    return res, (x, y, x_err, y_err, step)


def grid_extrap(curve):
    """Fits over the window grid -> (mean E0, mean E0 error, std of E0, table).

    Table columns: start, end, bin, k, E0, E0_err, res_var, R, D, kept. Kept fits: converged,
    k > 0, E0 below the lowest binned E, E0 error <= SD_REL_MAX x median.
    """
    rows = []
    for start, end, size in itertools.product(STARTS, ENDS, BINS):
        res, (x, y, *_) = extrap(curve, start, end, size)
        k, b = res.beta
        R = lever_arm(window(curve, start, end)["VarE"])
        ok = res.success and k > 0 and b <= y.min()
        rows.append([start, end, size, k, b, res.sd_beta[1], res.res_var, R, abs(k) * x.mean(), ok])
    table = np.array(rows, dtype=float)
    kept = table[:, 9].astype(bool)
    kept &= table[:, 5] <= SD_REL_MAX * np.median(table[kept, 5])
    table[:, 9] = kept
    return table[kept, 4].mean(), table[kept, 5].mean(), table[kept, 4].std(ddof=1), table


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    curve = read_csv(DATA / "training_curve.csv")
    b, stat, syst, table = grid_extrap(curve)
    kept = table[:, 9].astype(bool)
    err = np.hypot(stat, syst)  # error bar of panel (a)
    print(f"E0/N = {b / N:.6f} +- {stat / N:.6f} (stat) +- {syst / N:.6f} (window), "
          f"combined +- {err / N:.6f}; {kept.sum()}/{len(table)} windows kept")
    res, (x, y, x_err, y_err, step) = extrap(curve, *REP)
    k, b_rep = res.beta

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.errorbar(x / N, y / N, xerr=x_err / N, yerr=y_err / N, fmt="none", ecolor="gray", alpha=0.5,
                capsize=2, zorder=1)
    sc = ax.scatter(x / N, y / N, s=14, c=step, cmap="viridis", zorder=2)
    fig.colorbar(sc, ax=ax, label="step")
    x_max = 1.25 * np.quantile(x, 0.9) / N  # a few variance-spike bins lie off scale
    xp = np.linspace(0, x_max, 200)
    ax.plot(xp, (k * xp * N + b_rep) / N, "k--", lw=1.2)
    ax.errorbar([0], [b / N], yerr=[err / N], fmt="o", color="C3", capsize=4, zorder=3)
    ax.set(xlabel=r"$\sigma^2/N$", ylabel="$E/N$", xlim=(-0.03 * x_max, x_max))
    ax.text(0.04, 0.95, "(a)", transform=ax.transAxes, va="top")

    ax = axes[1]
    for j, size in enumerate(BINS):
        t = table[table[:, 2] == size]
        ok = t[:, 9].astype(bool)  # dropped windows are not shown
        pos = np.arange(len(t)) + j * 0.25
        ax.errorbar(pos[ok], t[ok, 4] / N, yerr=t[ok, 5] / N, fmt="o", ms=4, capsize=2,
                    label=f"bin {size}")
    labels = [f"{s:.0f}--{min(e, 10000):.0f}" for s, e in table[table[:, 2] == BINS[0], :2]]
    ax.set_xticks(np.arange(len(labels)) + 0.25, labels, rotation=90, fontsize=9)
    ax.axhspan((b - syst) / N, (b + syst) / N, color="C3", alpha=0.2,
               label=r"grid mean $\pm$ window")
    ax.axhline(b / N, color="C3", lw=1)
    lo = (table[kept, 4] - table[kept, 5]).min() / N
    hi = (table[kept, 4] + table[kept, 5]).max() / N
    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - 4 * pad, hi + pad)  # room for the legend below the points
    ax.set(xlabel="fit window (steps)", ylabel="$E_0/N$")
    ax.legend(loc="lower left")
    ax.text(0.04, 0.95, "(b)", transform=ax.transAxes, va="top")
    fig.tight_layout()
    out = args.outdir / "figS11_zero_variance_extrapolation.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
