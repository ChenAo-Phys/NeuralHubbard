"""Regenerate Fig. S7 (weak and strong scaling of one VMC step of the 16x16 NN-fPEPS) from data/.

Per mode: time of the parts of a step, and aggregate throughput F/t with the MFU.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "figS07_vmc_scaling"
# column, label, color, marker
PARTS = [
    ("t_total_s", "Total", "black", "o"),
    ("t_Eloc_s", r"$E_\mathrm{loc}$", "#2a78d6", "s"),
    ("t_sampling_s", "Sampling", "#1baf7a", "^"),
    ("t_jacobian_s", "Jacobian", "#eda100", "D"),
    ("t_solver_s", "Solver", "#d1352b", "v"),
]
MFU_COLOR = "#8a4fce"
TITLES = {
    "weak": r"Weak scaling, $256$ samples per GPU",
    "strong": r"Strong scaling, $N_s = 8192$",
}


def read_csv(path):
    """Columns of a CSV file with '#' comment lines, as float arrays."""
    with open(path) as f:
        names, *rows = csv.reader(line for line in f if not line.startswith("#"))
    return dict(zip(names, np.array(rows, dtype=float).T))


def draw(data, strong, ax, bx, letters):
    n = data["nodes"].astype(int)
    for key, label, color, marker in PARTS:
        ax.loglog(n, data[key], marker=marker, ms=5, lw=1.5, color=color, label=label)
    if strong:  # ideal speedup from the first point of each part
        for key, *_ in PARTS:
            ax.loglog(n, data[key][0] * n[0] / n, ls="--", lw=0.8, color="gray")
    ax.set_ylabel("Time per step (s)")
    ax.set_ylim(0.08, 1600)

    rate = data["F_over_t_PFLOPs"]
    bx.loglog(n, rate, marker="o", ms=5, lw=1.5, color="black", label=r"$F / t$")
    bx.loglog(n, rate[0] * n / n[0], ls="--", lw=0.8, color="gray", label="ideal")
    bx.set_ylabel(r"$F / t$ (PFLOP/s)")
    bx.set_ylim(0.05, 30)
    cx = bx.twinx()
    cx.plot(n, data["MFU_percent"], marker="s", ms=5, lw=1.5, ls=":", color=MFU_COLOR, label="MFU")
    cx.set_ylim(0, 25)
    cx.set_ylabel(r"MFU (\%)", color=MFU_COLOR)
    cx.tick_params(axis="y", colors=MFU_COLOR)
    cx.spines["right"].set_color(MFU_COLOR)
    handles, labels = bx.get_legend_handles_labels()
    h2, l2 = cx.get_legend_handles_labels()
    bx.legend(handles + h2, labels + l2, loc="lower right", frameon=False, handlelength=1.6,
              labelspacing=0.3)

    for a, letter in zip((ax, bx), letters):
        a.set_xlabel("Nodes")
        a.set_xticks(n, [f"${x}$" for x in n])
        a.set_xlim(n[0] / 1.3, n[-1] * 1.3)
        a.grid(alpha=0.3, which="major")
        a.text(0.03, 0.97, rf"\textbf{{({letter})}}", transform=a.transAxes, ha="left", va="top")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10.4, 9.0), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.08, hspace=0.0)
    for sub, mode, letters in zip(fig.subfigures(2, 1), TITLES, ("ab", "cd")):
        ax, bx = sub.subplots(1, 2)
        draw(read_csv(DATA / f"{mode}_scaling.csv"), mode == "strong", ax, bx, letters)
        sub.suptitle(rf"\textbf{{{TITLES[mode]}}}", fontsize=16)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=len(PARTS),
               frameon=False, handlelength=1.6, columnspacing=1.4)
    out = args.outdir / "figS07_vmc_scaling.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
