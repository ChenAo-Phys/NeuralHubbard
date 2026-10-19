"""Regenerate Fig. S5 (Cholesky, CG and LSMR on frozen 8x8 SR systems) from data/.

CG is placed where its in-sample residual reaches eta = 0.1 and LSMR is run for the same number of
iterations; both are interpolated from the iteration curves in krylov_iterations.csv.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14,
    "legend.fontsize": 11.5,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "figS05_solver_combined"
ETA_KRYLOV = 0.1
SIZES = [2048, 4096, 8192, 16384, 32768, 65536]
NP = {"pfaffian": "1.2", "nnfpeps": "4.6"}  # rows: ansatz and N_p / 10^6
QUANTITIES = {"t": "time_s", "eta": "eta", "x": "xnorm"}
STYLE = {
    "m": dict(marker="o", color="#2a78d6", label="Cholesky"),
    "cg": dict(marker="^", color="#1baf7a", label="CG"),
    "ls": dict(marker="v", color="#d1352b", label="LSMR"),
}


def read_rows(path):
    """Rows of a CSV file with '#' comment lines, as dicts."""
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def loglog_interp(x, xp, fp):
    o = np.argsort(xp)
    return np.exp(np.interp(np.log(x), np.log(xp[o]), np.log(fp[o])))


def curves(rows, ansatz, ns, solver):
    """Iteration counts and {quantity: (n_iter, groups) array} of one solver on one system."""
    rows = [r for r in rows if (r["ansatz"], int(r["Ns"]), r["solver"]) == (ansatz, ns, solver)]
    n = list(dict.fromkeys(int(r["n_iter"]) for r in rows))
    shape = (len(n), max(int(r["group"]) for r in rows) + 1)
    out = {q: np.empty(shape) for q in QUANTITIES}
    for r in rows:
        for q, col in QUANTITIES.items():
            out[q][n.index(int(r["n_iter"])), int(r["group"])] = float(r[col])
    return np.array(n, dtype=float), out


def collect(ansatz):
    """Arrays to plot for one ansatz: Krylov means and errors over groups, Cholesky points."""
    rows = read_rows(DATA / "krylov_iterations.csv")
    d = defaultdict(list)
    for ns in SIZES:
        n_cg, cg = curves(rows, ansatz, ns, "CG")
        groups = range(cg["eta"].shape[1])
        n_star = [loglog_interp(ETA_KRYLOV, cg["eta"][:, g], n_cg) for g in groups]
        for solver, s in (("CG", "cg"), ("LSMR", "ls")):
            n, a = curves(rows, ansatz, ns, solver)
            for q in QUANTITIES:
                v = [loglog_interp(n_star[g], n, a[q][:, g]) for g in groups]
                d[f"{q}_{s}"].append(np.mean(v))
                d[f"{q}_{s}_e"].append(np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
    d = {k: np.array(v) for k, v in d.items()}

    chol = [r for r in read_rows(DATA / "cholesky.csv") if r["ansatz"] == ansatz]
    meas = [r for r in chol if r["extrapolated"] == "0"]
    for k, col in (("NS_m", "Ns"), ("t_m", "t_solve_s"), ("eta_m", "eta"), ("x_m", "xnorm"),
                   ("x_m_e", "xnorm_sem")):
        d[k] = np.array([float(r[col]) for r in meas])
    d["oom"] = [(float(r["Ns"]), float(r["t_solve_s"])) for r in chol if r["extrapolated"] == "1"]
    return d


def draw_row(axs, ansatz, letters, bottom):
    d = collect(ansatz)
    ax, bx, cx = axs
    for a, letter in zip(axs, letters):
        a.set_xscale("log", base=2)
        a.set_xticks(SIZES, [rf"$2^{{{k}}}$" for k in range(11, 17)])
        a.set_xlim(1700, 7.9e4)
        a.grid(True, which="major", color="0.9", lw=0.8)
        if bottom:
            a.set_xlabel(r"samples $N_s$")
        a.text(0.035, 0.955, f"({letter})", transform=a.transAxes, fontsize=13, ha="left",
               va="top")

    def plot(axis, s, x, y, yerr=None):
        axis.errorbar(x, y, yerr, ms=4.5, lw=1.4, capsize=2.5, **STYLE[s])

    plot(ax, "m", d["NS_m"], d["t_m"])
    plot(ax, "cg", SIZES, d["t_cg"], d["t_cg_e"])
    plot(ax, "ls", SIZES, d["t_ls"], d["t_ls_e"])
    for ns, t in d["oom"]:  # the FP64 solve no longer fits in memory: extrapolated time
        ax.plot([d["NS_m"][-1], ns], [d["t_m"][-1], t], ls="--", lw=1.4, color=STYLE["m"]["color"])
        ax.plot(ns, t, marker="x", ms=7, mew=1.8, color=STYLE["m"]["color"])
    ax.set_yscale("log")
    ax.set_ylabel("solve time (s)")

    plot(bx, "m", d["NS_m"], d["eta_m"])
    plot(bx, "cg", SIZES, np.full(len(SIZES), ETA_KRYLOV))
    plot(bx, "ls", SIZES, d["eta_ls"], d["eta_ls_e"])
    bx.set_yscale("log")
    bx.set_ylim(5e-3, 0.4)
    bx.set_yticks([0.01, 0.03, 0.1, 0.3], [r"$0.01$", r"$0.03$", r"$0.1$", r"$0.3$"])
    bx.minorticks_off()
    bx.set_ylabel(r"in-sample residual $\eta$")

    plot(cx, "m", d["NS_m"], d["x_m"], d["x_m_e"])
    plot(cx, "cg", SIZES, d["x_cg"], d["x_cg_e"])
    plot(cx, "ls", SIZES, d["x_ls"], d["x_ls_e"])
    cx.set_yscale("log")
    cx.set_ylabel(r"step norm $\|x\|$")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.2, 6.0), layout="constrained")
    for sf, ansatz, letters in zip(fig.subfigures(2, 1, hspace=0.0), NP, ("abc", "def")):
        axs = sf.subplots(1, 3)
        sf.suptitle(rf"$N_p \approx {NP[ansatz]}$\,M", fontsize=14)
        draw_row(axs, ansatz, letters, bottom=ansatz == "nnfpeps")
        if ansatz == "pfaffian":
            axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.89),
                          handlelength=1.3, borderpad=0.15, labelspacing=0.25)
    out = args.outdir / "figS05_solver_combined.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
