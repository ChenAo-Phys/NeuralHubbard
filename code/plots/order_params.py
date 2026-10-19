"""Regenerate main-text Figs. 3 (t' = 0) and 4 (t' = -0.2) from data/.

Panels: (a) E/N against L; (b) C_p(|r|) of the 16x16 states with the Delta_SC^2 plateaus;
(c) 10 Delta_SC^2 and M2_str against 1/L with weighted fits a + b/L^p.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "legend.fontsize": 15,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
COLOR = {
    "NN-fPEPS": "#eda100",
    "BCS backflow": "#2a78d6",
    "Pfaffian backflow": "#1baf7a",
    "CP-AFQMC": "#2f2f2f",
    "ACE": "black",
    "SCALE": "0.6",
    "HFPS": "#8dd7bc",
    "PEPS (extrapolated)": "#f6d080",
}
ANSATZES = ("NN-fPEPS", "BCS backflow", "Pfaffian backflow")
R_MAX = 16 * np.sqrt(2) / 4  # start of the plateau region on the 16x16 lattice
XEND = 0.1  # right end of the 1/L axis
ERR_HIDE = 3e-4  # shorter error bars are not drawn
BAR = dict(capsize=4, capthick=1.3, elinewidth=1.3)
SCALE = {"Delta": 10.0, "M2": 1.0}  # panel (c) shows 10 Delta_SC^2
MARKER = {"Delta": "^", "M2": "s"}
LINE = {"Delta": ":", "M2": "--"}


def read(folder, name):
    with open(DATA / folder / name, newline="") as f:
        return list(csv.DictReader(f))


def column(rows, key):
    return np.array([float(r[key]) for r in rows])


def draw_energy(ax, rows, sizes, xlim, ylim, label_left):
    """Panel (a): labelled reference energies, then our states."""
    for name in dict.fromkeys(r["method"] for r in rows):
        pts = sorted((r for r in rows if r["method"] == name), key=lambda r: int(r["L"]))
        L, E, c = column(pts, "L"), column(pts, "E_per_site"), COLOR[name]
        ax.plot(L, E, "-", lw=1.3, color=c, zorder=2)
        if name in ANSATZES:
            ax.plot(L, E, "o", ms=9, color=c, mec="none", zorder=3)
            continue
        err = column(pts, "E_per_site_err") if any(r["err_drawn"] == "1" for r in pts) else None
        ax.errorbar(L, E, err, fmt="o", ms=9, color=c, mec="none", capsize=3, zorder=3)
        left = name in label_left
        ax.annotate(name, (L[0], E[0]), textcoords="offset points", xytext=(-9 if left else 9, 0),
                    ha="right" if left else "left", va="center", fontsize=14, color=c)
    ax.set_xlim(*xlim)
    ax.set_xticks(sizes, [rf"${n} \times {n}$" for n in sizes], fontsize=13)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.004))
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.set_xlabel("System size")
    ax.set_ylabel(r"Energy per site $E/N$")
    ax.grid(alpha=0.3)


def draw_cp(ax, folder, ylim, bcs_plateau_lw):
    """Panel (b): shell means of C_p and the Delta_SC^2 plateaus (dashed)."""
    ax.set_xlabel(r"$|r|$")
    ax.set_ylabel(r"$C_p(|r|)$")
    ax.set_xlim(-0.5, 12.0)
    ax.set_ylim(*ylim)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(alpha=0.3, which="major")
    ax.axvspan(R_MAX, 12.0, color="0.88", zorder=0)
    ax.text(11.8, 9e-2, r"$|r| \geq r_{\max}$", color="0.35", ha="right", va="top")
    shells = read(folder, "panel_b_pair_corr_shells.csv")
    for p in read(folder, "panel_b_plateaus.csv"):
        name = p["ansatz"]
        if p["plateau_drawn"] == "1":
            lw = bcs_plateau_lw if name == "BCS backflow" else 1.3
            ax.axhline(float(p["Delta_SC2"]), ls="--", lw=lw, color=COLOR[name], zorder=1)
        pts = [r for r in shells if r["ansatz"] == name]
        r, cp = column(pts, "r_abs"), column(pts, "C_p_mean")
        err = column(pts, "C_p_mean_err") if pts[0]["C_p_mean_err"] else None  # none if digitized
        pos = cp > 0  # negative values drop off the log axis
        ax.errorbar(r[pos], cp[pos], None if err is None else err[pos], fmt="o", ms=6, lw=1.2,
                    color=COLOR[name], mec="none", zorder=3)


def line_fit(L, y, err, power=1):
    """Weighted least-squares fit y = y0 + slope / L^power -> (y0, y0 error, slope, chi^2)."""
    x, w = 1 / L**power, 1 / err**2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (w[:, None] * A))
    p = cov @ A.T @ (w * y)
    return p[0], np.sqrt(cov[0, 0]), p[1], float((w * (A @ p - y) ** 2).sum())


def series(rows, name, key):
    """(L, y, err, error drawn) of the plotted Delta_SC^2 ("Delta") or M2_str ("M2") points."""
    pts = sorted((r for r in rows if r["ansatz"] == name and r[f"{key}_plotted"] == "1"),
                 key=lambda r: int(r["L"]))
    L = column(pts, "L")
    if key == "M2":
        return L, column(pts, "M2_str"), column(pts, "M2_str_err"), [False] * len(pts)
    D, De = column(pts, "Delta_SC"), column(pts, "Delta_SC_err")
    return L, D**2, 2 * D * De, [r["Delta_err_drawn"] == "1" for r in pts]


def draw_fit(ax, rows, name, key, power):
    """Fit line and its 1/L -> 0 value, the error enlarged by sqrt(chi^2/dof) if that exceeds 1."""
    L, y, err, _ = series(rows, name, key)
    y0, y0_err, slope, chi2 = line_fit(L, y, err, power)
    if len(L) > 2:
        y0_err *= max(1.0, np.sqrt(chi2 / (len(L) - 2)))
    s, c = SCALE[key], COLOR[name]
    x = np.linspace(0.0, XEND, 2 if power == 1 else 101)
    ax.plot(x, s * (y0 + slope * x**power), LINE[key], lw=1.5, color=c, zorder=1)
    face = dict(mfc="none", mec=c, mew=1.8) if key == "M2" else dict(mec="black", mew=1.0)
    ax.errorbar([0.0], [s * y0], [s * y0_err] if s * y0_err >= ERR_HIDE else None,
                fmt=MARKER[key], ms=9, color=c, zorder=5, **BAR, **face)


def draw_points(ax, rows, name, key, zorder=3):
    """Filled triangles (Delta_SC^2) or hollow squares (M2_str) against 1/L."""
    s, c = SCALE[key], COLOR[name]
    face = dict(mfc="none", mew=1.5) if key == "M2" else dict(mec="none")
    for L, y, err, drawn in zip(*series(rows, name, key)):
        ax.errorbar(1 / L, s * y, s * err if drawn else None, fmt=MARKER[key], ms=8, color=c,
                    zorder=zorder, **BAR, **face)


def orders_tp0(ax, rows, fits):
    for name in ANSATZES:
        for key in ("Delta", "M2"):
            if (name, key) in fits:
                draw_fit(ax, rows, name, key, fits[name, key])
            draw_points(ax, rows, name, key)


def orders_tpm02(ax, rows, fits):
    for name in ("BCS backflow", "Pfaffian backflow", "NN-fPEPS"):
        draw_fit(ax, rows, name, "Delta", fits[name, "Delta"])
    draw_points(ax, rows, "BCS backflow", "Delta")
    afqmc = next(r for r in rows if r["ansatz"] == "CP-AFQMC")
    D, De = float(afqmc["Delta_SC"]), float(afqmc["Delta_SC_err"])
    x, y, e, c = -0.008, 10 * D**2, 10 * 2 * D * De, COLOR["CP-AFQMC"]  # left of 1/L = 0
    ax.errorbar([x], [y], [e], fmt="^", ms=8, color=c, mec="none", zorder=4, **BAR)
    ax.annotate("AFQMC", (x, y + e), textcoords="offset points", xytext=(0, 5), ha="center",
                va="bottom", rotation=90, fontsize=14, color=c)
    draw_points(ax, rows, "NN-fPEPS", "Delta", zorder=4)
    draw_points(ax, rows, "Pfaffian backflow", "Delta", zorder=4)
    for name in ("Pfaffian backflow", "NN-fPEPS", "BCS backflow"):
        draw_fit(ax, rows, name, "M2", fits[name, "M2"])
        draw_points(ax, rows, name, "M2")


def figure(folder, outdir, sizes, xlim, e_ylim, cp_ylim, bcs_plateau_lw, label_left, orders,
           legend_kw):
    fig = plt.figure(figsize=(11.5, 4.4))
    left, right, bottom, top, gap = 0.095, 0.993, 0.145, 0.885, 0.077
    w = (right - left - 2 * gap) / 3
    axes = [fig.add_axes((left + k * (w + gap), bottom, w, top - bottom)) for k in range(3)]
    draw_energy(axes[0], read(folder, "panel_a_energy.csv"), sizes, xlim, e_ylim, label_left)
    draw_cp(axes[1], folder, cp_ylim, bcs_plateau_lw)

    ax = axes[2]
    fits = {(f["ansatz"], f["observable"].split("_")[0]): int(f["fit_form"][-1])
            for f in read(folder, "panel_c_fits.csv") if f["plotted"] == "1"}
    orders(ax, read(folder, "panel_c_orders.csv"), fits)
    ax.axhline(0.0, lw=0.8, color="0.6", zorder=0)
    ax.set_xlim(-0.015, XEND)
    ax.set_ylim(-0.0015, 0.0185)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel(r"$1/L$")
    ax.set_ylabel("Order parameter")
    ax.grid(alpha=0.3)

    for a, letter in zip(axes, "abc"):
        a.text(0.022, 0.975, rf"\textbf{{({letter})}}", transform=a.transAxes, ha="left", va="top",
               zorder=5)
    markers = [
        Line2D([], [], color="black", ls="none", marker="^", ms=8,
               label=r"$\Delta^2_{\mathrm{SC}} \times 10$"),
        Line2D([], [], color="black", ls="none", marker="s", ms=8, mfc="none", mew=1.5,
               label=r"$M^2_{\mathrm{str}}$"),
    ]
    ax.legend(handles=markers, framealpha=0.95, fontsize=14, handletextpad=0.5, labelspacing=0.4,
              borderpad=0.5, **legend_kw)
    lines = [Line2D([], [], color=COLOR[a], lw=2.2, label=a) for a in ANSATZES]
    fig.legend(handles=lines, loc="upper center", ncol=3, frameon=False, columnspacing=1.8,
               handletextpad=0.5, bbox_to_anchor=(0.5, 1.0))
    out = outdir / f"{folder}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure("fig3_order_params_tp0", args.outdir, sizes=(16, 32), xlim=(8, 40),
           e_ylim=(-0.7620, -0.7492), cp_ylim=(1e-5, 1e-1), bcs_plateau_lw=2.8,
           label_left=("HFPS",), orders=orders_tp0, legend_kw=dict(loc="lower right"))
    figure("fig4_order_params_tpm02", args.outdir, sizes=(16, 24, 32), xlim=(12.8, 35.2),
           e_ylim=(-0.7470, -0.7340), cp_ylim=(1e-4, 1e-1), bcs_plateau_lw=1.3, label_left=(),
           orders=orders_tpm02, legend_kw=dict(loc="upper left", bbox_to_anchor=(0.10, 1.0)))


if __name__ == "__main__":
    main()
