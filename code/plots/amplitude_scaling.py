"""Regenerate Fig. S6 (time per amplitude against lattice size) from data/.

Gray lines are guides to the eye for the asymptotic scalings.
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
    "legend.fontsize": 14,
    "pdf.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "supplement" / "figS06_amplitude_scaling"
# column, label, color, marker
SERIES = [
    ("determinant_us", "Determinant", "#2a78d6", "o"),
    ("pfaffian_us", "Pfaffian", "#1baf7a", "^"),
    ("fpeps_us", "fPEPS", "#eda100", "D"),
    ("fpeps_trans_us", "fPEPS (trans)", "#eb6834", "v"),
]


def read_csv(path):
    """Columns of a CSV file with '#' comment lines, as float arrays."""
    with open(path) as f:
        names, *rows = csv.reader(line for line in f if not line.startswith("#"))
    return dict(zip(names, np.array(rows, dtype=float).T))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    data = read_csv(DATA / "amplitude_time.csv")
    N = data["N"]
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for col, label, color, marker in SERIES:
        ax.loglog(N, data[col], marker=marker, ms=5, lw=1.5, color=color, label=label)

    x = np.array([1100, 2200])
    ax.loglog(x, 2.8e-7 * x**3.0, "--", lw=0.8, color="gray")
    ax.text(560, 1500, r"$\mathcal{O}(N^3)$", color="gray")
    x = np.array([1024, 4096])
    ax.loglog(x, 0.019 * x, ":", lw=0.8, color="gray")
    ax.text(1700, 14, r"$\mathcal{O}(N)$", color="gray")
    x = np.array([128, 512])
    ax.loglog(x, 0.075 * x * np.log(x), "-.", lw=0.8, color="gray")
    ax.text(90, 150, r"$\mathcal{O}(N \log N)$", color="gray")

    ax.set_xticks(N, [rf"${L:.0f}^2$" for L in data["L"]])
    ax.set_xlabel(r"Lattice size $N$")
    ax.set_ylabel(r"Time per amplitude ($\mu$s)")
    ax.grid(alpha=0.3, which="both", axis="y")
    ax.grid(alpha=0.3, which="major", axis="x")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = args.outdir / "figS06_amplitude_scaling.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
